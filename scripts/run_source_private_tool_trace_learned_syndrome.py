from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import (
    Example,
    _deterministic_nonself_index,
    _mask_log_components,
    _mask_repair_diag,
    _prior_prediction,
    make_benchmark,
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_count(text: str) -> int:
    return 0 if not text else len(re.findall(r"\S+", text))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-8)


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2s(value.encode("utf-8"), digest_size=8).digest(), "little")


def _hashed_text_features(text: str, dim: int, *, namespace: str) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9_]+|==|!=|<=|>=|[{}()[\\],.:+-]", text.lower())
    for idx, token in enumerate(tokens):
        for feature in (token, f"{token}@{idx % 7}"):
            h = _stable_hash(f"{namespace}:{feature}")
            sign = 1.0 if (h >> 63) == 0 else -1.0
            vec[h % dim] += sign
    return vec / max(1.0, float(len(tokens)) ** 0.5)


def _source_text(example: Example, *, mode: str) -> str:
    if mode == "matched":
        return example.private_test_log
    if mode == "answer_masked":
        masked = _mask_repair_diag(example.private_test_log)
        masked = _mask_log_components(masked, mask_expected_actual=True, mask_test_name=True)
        lines = []
        for line in masked.splitlines():
            if line.startswith("hidden_input="):
                lines.append("hidden_input=<MASKED>")
            elif "repair_family=" in line:
                lines.append(re.sub(r"repair_family=[A-Za-z0-9_]+", "repair_family=<MASKED>", line))
            elif line.startswith("failure_status="):
                lines.append("failure_status=<MASKED>")
            else:
                lines.append(line)
        return "\n".join(lines)
    if mode == "zero":
        return ""
    raise ValueError(f"unknown source mode {mode!r}")


def _candidate_texts(example: Example) -> list[str]:
    return [
        "\n".join(
            [
                f"patch={candidate.patch_name}",
                f"intent={candidate.patch_intent}",
                f"handles_repair_diag={candidate.handles_diagnostic}",
                f"public_issue={example.public_issue}",
            ]
        )
        for candidate in example.candidates
    ]


def _candidate_matrix(example: Example, feature_dim: int) -> np.ndarray:
    return _normalize_rows(
        np.stack([_hashed_text_features(text, feature_dim, namespace="candidate") for text in _candidate_texts(example)])
    ).astype(np.float32)


def _source_vector(example: Example, feature_dim: int, *, mode: str) -> np.ndarray:
    return _hashed_text_features(_source_text(example, mode=mode), feature_dim, namespace="source").astype(np.float32)


def _fit_ridge_encoder(train_examples: list[Example], *, feature_dim: int, ridge: float) -> np.ndarray:
    x = np.stack([_source_vector(example, feature_dim, mode="matched") for example in train_examples], axis=0).astype(np.float64)
    y = []
    for example in train_examples:
        candidates = _candidate_matrix(example, feature_dim).astype(np.float64)
        answer_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)
        y.append(candidates[answer_index])
    y_arr = np.stack(y, axis=0).astype(np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    xtx = x_aug.T @ x_aug
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[-1, -1] -= ridge
    return np.linalg.solve(xtx, x_aug.T @ y_arr).astype(np.float32)


def _augment(vec: np.ndarray) -> np.ndarray:
    return np.concatenate([vec.astype(np.float32), np.ones(1, dtype=np.float32)])


def _bitpack(bits: np.ndarray, budget_bytes: int) -> bytes:
    required_bits = budget_bytes * 8
    padded = np.zeros(required_bits, dtype=np.uint8)
    padded[: min(required_bits, bits.shape[0])] = bits[:required_bits].astype(np.uint8)
    return np.packbits(padded, bitorder="big").tobytes()


def _bytes_to_bits(payload: bytes | None, bit_count: int) -> np.ndarray:
    if not payload:
        return np.zeros(bit_count, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[:bit_count].astype(np.uint8)


def _candidate_codes(example: Example, code_projection: np.ndarray, feature_dim: int, bit_count: int) -> np.ndarray:
    candidates = _candidate_matrix(example, feature_dim)
    logits = candidates @ code_projection[:bit_count].T
    return (logits >= 0).astype(np.uint8)


def _packet_from_vector(vec: np.ndarray, code_projection: np.ndarray, budget_bytes: int) -> bytes:
    bit_count = budget_bytes * 8
    logits = code_projection[:bit_count] @ vec
    return _bitpack((logits >= 0).astype(np.uint8), budget_bytes)


def _source_packet(example: Example, encoder: np.ndarray, code_projection: np.ndarray, feature_dim: int, budget_bytes: int, *, mode: str) -> bytes:
    predicted = _augment(_source_vector(example, feature_dim, mode=mode)) @ encoder
    return _packet_from_vector(predicted, code_projection, budget_bytes)


def _decode_packet(example: Example, payload: bytes | None, code_projection: np.ndarray, feature_dim: int, budget_bytes: int) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    bit_count = budget_bytes * 8
    packet_bits = _bytes_to_bits(payload, bit_count)
    codes = _candidate_codes(example, code_projection, feature_dim, bit_count)
    distances = np.sum(codes != packet_bits[None, :], axis=1)
    min_distance = int(np.min(distances))
    tied = np.flatnonzero(distances == min_distance)
    candidate_labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(candidate_labels[int(idx)] == prior for idx in tied):
        prediction = prior
    else:
        prediction = candidate_labels[int(tied[0])]
    return prediction, {"decoder": "hamming", "min_hamming_distance": min_distance, "ties": [int(i) for i in tied.tolist()]}


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    rng: random.Random,
    random_encoder: np.ndarray,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {}
    if condition == "matched_learned_syndrome":
        return _source_packet(example, encoder, code_projection, feature_dim, budget_bytes, mode="matched"), {"source": example.example_id}
    if condition == "shuffled_source":
        other = eval_examples[_deterministic_nonself_index(index, len(eval_examples))]
        return _source_packet(other, encoder, code_projection, feature_dim, budget_bytes, mode="matched"), {"source": other.example_id}
    if condition == "answer_masked_source":
        return _source_packet(example, encoder, code_projection, feature_dim, budget_bytes, mode="answer_masked"), {"source": "answer_masked"}
    if condition == "random_same_byte":
        return rng.randbytes(budget_bytes), {"source": "random"}
    if condition == "target_derived_sidecar":
        prior = _prior_prediction(example)
        prior_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)
        return _packet_from_vector(_candidate_matrix(example, feature_dim)[prior_index], code_projection, budget_bytes), {"source": "target_prior"}
    if condition == "answer_only":
        return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_hidden_log"}
    if condition == "wrong_projection_source":
        predicted = _augment(_source_vector(example, feature_dim, mode="matched")) @ random_encoder
        return _packet_from_vector(predicted, code_projection, budget_bytes), {"source": "wrong_encoder"}
    if condition == "full_diag_oracle":
        answer_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)
        return _packet_from_vector(_candidate_matrix(example, feature_dim)[answer_index], code_projection, budget_bytes), {"source": "candidate_oracle"}
    raise ValueError(f"unknown condition {condition!r}")


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    rng: random.Random,
    random_encoder: np.ndarray,
) -> dict[str, Any]:
    start = time.perf_counter()
    payload, metadata = _payload_for_condition(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        encoder=encoder,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        rng=rng,
        random_encoder=random_encoder,
    )
    prediction, decode_meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
    payload_hex = (payload or b"").hex()
    return {
        "condition": condition,
        "prediction": prediction,
        "answer": example.answer_label,
        "correct": prediction == example.answer_label,
        "payload_bytes": len(payload or b""),
        "payload_tokens": _token_count(payload_hex),
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "payload_hex": payload_hex,
        "metadata": {**metadata, **decode_meta},
    }


def _summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in predictions]
    correct = sum(1 for row in predictions if row["correct"])
    return {
        "n": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in predictions),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in predictions),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    budgets: list[int],
    train_seed: int,
    eval_seed: int,
    ridge: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(examples=train_examples, candidates=candidates, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=candidates, seed=eval_seed, family_set=eval_family_set)
    encoder = _fit_ridge_encoder(train_rows, feature_dim=feature_dim, ridge=ridge)
    rng_np = np.random.default_rng(train_seed * 1009 + eval_seed)
    code_projection = _normalize_rows(rng_np.normal(size=(max(budgets) * 8, feature_dim))).astype(np.float32)
    random_encoder = rng_np.normal(0.0, 1.0 / np.sqrt(feature_dim), size=encoder.shape).astype(np.float32)
    rng = random.Random(train_seed * 2003 + eval_seed)
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
        "full_diag_oracle",
    ]
    prediction_files: dict[str, str] = {}
    budget_summaries: list[dict[str, Any]] = []
    for budget in budgets:
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
        for row_index, example in enumerate(eval_rows):
            for condition in conditions:
                by_condition[condition].append(
                    _predict_condition(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        encoder=encoder,
                        code_projection=code_projection,
                        feature_dim=feature_dim,
                        budget_bytes=budget,
                        rng=rng,
                        random_encoder=random_encoder,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        metrics = {condition: _summarize(rows) for condition, rows in by_condition.items()}
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
        controls = [
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
            and all(metrics[name]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05 for name in controls)
            and metrics["full_diag_oracle"]["accuracy"] >= 0.95
        )
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for condition in conditions:
                for row in by_condition[condition]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_files[str(budget)] = predictions_path.name
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
    exact_ids = [row.example_id for row in eval_rows]
    payload = {
        "gate": "source_private_tool_trace_learned_syndrome",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "budgets": budgets,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "ridge": ridge,
        "exact_id_parity": len(exact_ids) == len(set(exact_ids)),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
        "prediction_files": prediction_files,
        "pass_rule": "matched learned syndrome beats best no-source by >=0.15; controls stay within target_only +0.05; full diagnostic oracle >=0.95.",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Source-Private Tool-Trace Learned Syndrome",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{train_family_set}:{train_examples}` / `{eval_family_set}:{eval_examples}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget bytes | Pass | Matched | Target | Best no-source | Delta | Full diag oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_summaries:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_no_source_accuracy']:.3f} | "
            f"{row['matched_minus_best_no_source']:.3f} | {row['metrics']['full_diag_oracle']['accuracy']:.3f} |"
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "artifacts": ["summary.json", "summary.md", *prediction_files.values(), "manifest.json", "manifest.md"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["summary.json", "summary.md", *prediction_files.values()]
        },
        "pass_gate": payload["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Tool-Trace Learned Syndrome Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- train family set: `{train_family_set}`",
                f"- eval family set: `{eval_family_set}`",
                f"- budgets: `{budgets}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_tool_trace_learned_syndrome_20260429"))
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        budgets=args.budgets,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        ridge=args.ridge,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
