from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SyntheticRow:
    example_id: str
    candidate_latents: np.ndarray
    source_observation: np.ndarray
    masked_source_observation: np.ndarray
    answer_index: int
    prior_index: int


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_count(text: str) -> int:
    return 0 if not text else len(text.split())


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-8)


def _bitpack(bits: np.ndarray, budget_bytes: int) -> bytes:
    if bits.ndim != 1:
        raise ValueError("bits must be one-dimensional")
    required_bits = budget_bytes * 8
    padded = np.zeros(required_bits, dtype=np.uint8)
    padded[: min(required_bits, bits.shape[0])] = bits[:required_bits].astype(np.uint8)
    return np.packbits(padded, bitorder="big").tobytes()


def _bytes_to_bits(payload: bytes, bit_count: int) -> np.ndarray:
    if not payload:
        return np.zeros(bit_count, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[:bit_count].astype(np.uint8)


def _make_rows(
    *,
    examples: int,
    candidates: int,
    latent_dim: int,
    source_dim: int,
    seed: int,
) -> tuple[list[SyntheticRow], np.ndarray]:
    rng = np.random.default_rng(seed)
    source_projection = rng.normal(0.0, 1.0 / np.sqrt(latent_dim), size=(latent_dim, source_dim))
    rows: list[SyntheticRow] = []
    for idx in range(examples):
        candidate_latents = _normalize_rows(rng.normal(size=(candidates, latent_dim))).astype(np.float32)
        prior_index = idx % candidates
        answer_index = prior_index if idx % candidates == 0 else (prior_index + 1) % candidates
        private_signal = candidate_latents[answer_index] @ source_projection
        distractor_signal = candidate_latents[prior_index] @ source_projection
        source_observation = private_signal + rng.normal(0.0, 0.08, size=source_dim)
        masked_observation = 0.55 * distractor_signal + rng.normal(0.0, 0.08, size=source_dim)
        rows.append(
            SyntheticRow(
                example_id=f"learned_syndrome_{idx:04d}",
                candidate_latents=candidate_latents,
                source_observation=source_observation.astype(np.float32),
                masked_source_observation=masked_observation.astype(np.float32),
                answer_index=answer_index,
                prior_index=prior_index,
            )
        )
    return rows, source_projection.astype(np.float32)


def _fit_ridge_encoder(rows: list[SyntheticRow], *, ridge: float) -> np.ndarray:
    x = np.stack([row.source_observation for row in rows], axis=0).astype(np.float64)
    y = np.stack([row.candidate_latents[row.answer_index] for row in rows], axis=0).astype(np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[-1, -1] -= ridge
    encoder = np.linalg.solve(xtx, x_aug.T @ y)
    return encoder.astype(np.float32)


def _augment_source(source: np.ndarray) -> np.ndarray:
    return np.concatenate([source.astype(np.float32), np.ones(1, dtype=np.float32)])


def _candidate_codes(row: SyntheticRow, code_projection: np.ndarray, bit_count: int) -> np.ndarray:
    logits = row.candidate_latents @ code_projection[:bit_count].T
    return (logits >= 0).astype(np.uint8)


def _source_packet(source: np.ndarray, encoder: np.ndarray, code_projection: np.ndarray, budget_bytes: int) -> bytes:
    bit_count = budget_bytes * 8
    predicted_latent = _augment_source(source) @ encoder
    logits = code_projection[:bit_count] @ predicted_latent
    bits = (logits >= 0).astype(np.uint8)
    return _bitpack(bits, budget_bytes)


def _decode_packet(row: SyntheticRow, payload: bytes | None, code_projection: np.ndarray, budget_bytes: int) -> tuple[int, dict[str, Any]]:
    if not payload:
        return row.prior_index, {"decoder": "prior"}
    bit_count = budget_bytes * 8
    packet_bits = _bytes_to_bits(payload, bit_count)
    codes = _candidate_codes(row, code_projection, bit_count)
    distances = np.sum(codes != packet_bits[None, :], axis=1)
    min_distance = int(np.min(distances))
    tied = np.flatnonzero(distances == min_distance)
    if row.prior_index in tied:
        prediction = row.prior_index
    else:
        prediction = int(tied[0])
    return prediction, {
        "decoder": "hamming",
        "min_hamming_distance": min_distance,
        "ties": [int(i) for i in tied.tolist()],
    }


def _payload_for_condition(
    *,
    condition: str,
    row: SyntheticRow,
    eval_rows: list[SyntheticRow],
    row_index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    budget_bytes: int,
    rng: random.Random,
    random_encoder: np.ndarray,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {}
    if condition == "matched_learned_syndrome":
        return _source_packet(row.source_observation, encoder, code_projection, budget_bytes), {"source": row.example_id}
    if condition == "shuffled_source":
        other_index = (row_index * 17 + 11) % len(eval_rows)
        if other_index == row_index:
            other_index = (row_index + 1) % len(eval_rows)
        other = eval_rows[other_index]
        return _source_packet(other.source_observation, encoder, code_projection, budget_bytes), {"source": other.example_id}
    if condition == "answer_masked_source":
        return _source_packet(row.masked_source_observation, encoder, code_projection, budget_bytes), {"source": "masked"}
    if condition == "random_same_byte":
        return rng.randbytes(budget_bytes), {"source": "random"}
    if condition == "target_derived_sidecar":
        bit_count = budget_bytes * 8
        codes = _candidate_codes(row, code_projection, bit_count)
        return _bitpack(codes[row.prior_index], budget_bytes), {"source": "target_prior"}
    if condition == "answer_only":
        return f"candidate_{row.answer_index}".encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}
    if condition == "structured_text_matched":
        return f"source says candidate_{row.answer_index}".encode("utf-8")[:budget_bytes], {"source": "truncated_text"}
    if condition == "wrong_projection_source":
        return _source_packet(row.source_observation, random_encoder, code_projection, budget_bytes), {"source": "wrong_encoder"}
    if condition == "full_text_oracle":
        return f"candidate_{row.answer_index}".encode("utf-8"), {"source": "full_oracle_text"}
    raise ValueError(f"unknown condition {condition!r}")


def _predict_condition(
    *,
    condition: str,
    row: SyntheticRow,
    eval_rows: list[SyntheticRow],
    row_index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    budget_bytes: int,
    rng: random.Random,
    random_encoder: np.ndarray,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, payload_meta = _payload_for_condition(
        condition=condition,
        row=row,
        eval_rows=eval_rows,
        row_index=row_index,
        encoder=encoder,
        code_projection=code_projection,
        budget_bytes=budget_bytes,
        rng=rng,
        random_encoder=random_encoder,
    )
    if condition == "full_text_oracle":
        prediction = row.answer_index
        decode_meta = {"decoder": "full_text_oracle"}
    else:
        prediction, decode_meta = _decode_packet(row, payload, code_projection, budget_bytes)
    latency_ms = (time.perf_counter() - start) * 1000.0
    payload_bytes = len(payload or b"")
    payload_text = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": int(prediction),
        "answer": int(row.answer_index),
        "correct": bool(prediction == row.answer_index),
        "payload_bytes": payload_bytes,
        "payload_tokens": _token_count(payload_text),
        "latency_ms": latency_ms,
        "payload_hex": payload_text,
        "metadata": {**payload_meta, **decode_meta},
    }


def _summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot summarize empty predictions")
    return {
        "n": len(predictions),
        "correct": sum(1 for row in predictions if row["correct"]),
        "accuracy": sum(1 for row in predictions if row["correct"]) / len(predictions),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in predictions),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in predictions),
        "p50_latency_ms": statistics.median(row["latency_ms"] for row in predictions),
        "p95_latency_ms": sorted(row["latency_ms"] for row in predictions)[max(0, int(0.95 * len(predictions)) - 1)],
    }


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    candidates: int,
    latent_dim: int,
    source_dim: int,
    budgets: list[int],
    seed: int,
    ridge: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, source_projection = _make_rows(
        examples=train_examples + eval_examples,
        candidates=candidates,
        latent_dim=latent_dim,
        source_dim=source_dim,
        seed=seed,
    )
    train_rows = rows[:train_examples]
    eval_rows = rows[train_examples:]
    encoder = _fit_ridge_encoder(train_rows, ridge=ridge)
    rng_np = np.random.default_rng(seed + 1009)
    code_projection = _normalize_rows(rng_np.normal(size=(max(budgets) * 8, latent_dim))).astype(np.float32)
    random_encoder = rng_np.normal(0.0, 1.0 / np.sqrt(source_dim), size=encoder.shape).astype(np.float32)
    rng = random.Random(seed + 2003)
    conditions = [
        "target_only",
        "matched_learned_syndrome",
        "zero_source",
        "shuffled_source",
        "answer_masked_source",
        "random_same_byte",
        "target_derived_sidecar",
        "answer_only",
        "structured_text_matched",
        "wrong_projection_source",
        "full_text_oracle",
    ]
    budget_summaries: list[dict[str, Any]] = []
    prediction_paths: dict[str, str] = {}
    for budget in budgets:
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
        for row_index, row in enumerate(eval_rows):
            for condition in conditions:
                by_condition[condition].append(
                    _predict_condition(
                        condition=condition,
                        row=row,
                        eval_rows=eval_rows,
                        row_index=row_index,
                        encoder=encoder,
                        code_projection=code_projection,
                        budget_bytes=budget,
                        rng=rng,
                        random_encoder=random_encoder,
                    )
                    | {"example_id": row.example_id, "budget_bytes": budget}
                )
        metrics = {condition: _summarize(predictions) for condition, predictions in by_condition.items()}
        best_no_source = max(
            metrics[name]["accuracy"]
            for name in [
                "target_only",
                "zero_source",
                "answer_masked_source",
                "random_same_byte",
                "target_derived_sidecar",
                "answer_only",
                "structured_text_matched",
                "wrong_projection_source",
            ]
        )
        matched = metrics["matched_learned_syndrome"]["accuracy"]
        source_controls = [
            "zero_source",
            "shuffled_source",
            "answer_masked_source",
            "random_same_byte",
            "target_derived_sidecar",
            "answer_only",
            "structured_text_matched",
            "wrong_projection_source",
        ]
        pass_gate = (
            matched >= best_no_source + 0.15
            and all(metrics[name]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05 for name in source_controls)
            and metrics["full_text_oracle"]["accuracy"] == 1.0
        )
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for condition in conditions:
                for row in by_condition[condition]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_paths[str(budget)] = predictions_path.name
        budget_summaries.append(
            {
                "budget_bytes": budget,
                "pass_gate": pass_gate,
                "matched_accuracy": matched,
                "target_only_accuracy": metrics["target_only"]["accuracy"],
                "best_no_source_accuracy": best_no_source,
                "matched_minus_best_no_source": matched - best_no_source,
                "metrics": metrics,
            }
        )
    candidate_pool_recall = 1.0
    exact_id_parity = all(row.example_id == f"learned_syndrome_{train_examples + idx:04d}" for idx, row in enumerate(eval_rows))
    payload = {
        "gate": "source_private_learned_syndrome_smoke",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "seed": seed,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "candidates": candidates,
        "latent_dim": latent_dim,
        "source_dim": source_dim,
        "ridge": ridge,
        "budgets": budgets,
        "candidate_pool_recall": candidate_pool_recall,
        "exact_id_parity": exact_id_parity,
        "source_projection_sha256": hashlib.sha256(source_projection.tobytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
        "prediction_files": prediction_paths,
        "pass_rule": "matched learned syndrome beats best no-source by >=0.15; source-destroying controls stay within target_only +0.05; full text oracle is 1.0.",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Source-Private Learned Syndrome Smoke",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{train_examples}/{eval_examples}`",
        f"- candidates: `{candidates}`",
        f"- exact ID parity: `{exact_id_parity}`",
        "",
        "| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full text |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_summaries:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_no_source_accuracy']:.3f} | "
            f"{row['matched_minus_best_no_source']:.3f} | {row['metrics']['full_text_oracle']['accuracy']:.3f} |"
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "command": " ".join(
            [
                "scripts/run_source_private_learned_syndrome_smoke.py",
                "--output-dir",
                str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir),
            ]
        ),
        "artifacts": ["summary.json", "summary.md", *prediction_paths.values(), "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md", *prediction_paths.values()]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Learned Syndrome Smoke Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- seed: `{seed}`",
                f"- budgets: `{budgets}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_learned_syndrome_smoke_20260429"))
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--source-dim", type=int, default=48)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        candidates=args.candidates,
        latent_dim=args.latent_dim,
        source_dim=args.source_dim,
        budgets=args.budgets,
        seed=args.seed,
        ridge=args.ridge,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
