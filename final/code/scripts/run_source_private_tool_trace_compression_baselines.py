from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _deterministic_nonself_index,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_learned_syndrome import (  # noqa: E402
    _augment,
    _candidate_matrix,
    _decode_packet,
    _fit_ridge_encoder,
    _normalize_rows,
    _packet_from_vector,
    _source_packet,
    _source_vector,
    _token_count,
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_scalar_calibration(
    train_examples: list[Example],
    *,
    encoder: np.ndarray,
    scalar_projection: np.ndarray,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    projected: list[np.ndarray] = []
    for example in train_examples:
        predicted = _augment(_source_vector(example, feature_dim, mode="matched")) @ encoder
        projected.append(scalar_projection @ predicted)
        projected.extend(_candidate_matrix(example, feature_dim) @ scalar_projection.T)
    values = np.stack(projected, axis=0).astype(np.float32)
    lo = np.quantile(values, 0.01, axis=0).astype(np.float32)
    hi = np.quantile(values, 0.99, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 1e-4)
    return lo, hi


def _quantize_scalar(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bytes:
    scaled = np.clip((values - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8).tobytes()


def _dequantize_scalar(payload: bytes, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    ints = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
    return lo[: ints.shape[0]] + (ints / 255.0) * (hi[: ints.shape[0]] - lo[: ints.shape[0]])


def _scalar_packet(
    example: Example,
    *,
    encoder: np.ndarray,
    scalar_projection: np.ndarray,
    feature_dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    mode: str,
) -> bytes:
    predicted = _augment(_source_vector(example, feature_dim, mode=mode)) @ encoder
    return _quantize_scalar(scalar_projection @ predicted, lo, hi)


def _decode_scalar_packet(
    example: Example,
    payload: bytes | None,
    *,
    scalar_projection: np.ndarray,
    feature_dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    if not payload:
        return _prior_prediction(example), {"decoder": "prior"}
    decoded = _dequantize_scalar(payload, lo, hi)
    candidate_values = _candidate_matrix(example, feature_dim) @ scalar_projection[: len(payload)].T
    distances = np.sum((candidate_values - decoded[None, :]) ** 2, axis=1)
    min_distance = float(np.min(distances))
    tied = np.flatnonzero(np.isclose(distances, min_distance, rtol=1e-6, atol=1e-8))
    labels = [candidate.label for candidate in example.candidates]
    prior = _prior_prediction(example)
    if any(labels[int(idx)] == prior for idx in tied):
        prediction = prior
    else:
        prediction = labels[int(tied[0])]
    return prediction, {"decoder": "scalar_quantized_l2", "min_l2": min_distance, "ties": [int(i) for i in tied.tolist()]}


def _raw_source_sign_packet(example: Example, code_projection: np.ndarray, feature_dim: int, budget_bytes: int) -> bytes:
    return _packet_from_vector(_source_vector(example, feature_dim, mode="matched"), code_projection, budget_bytes)


def _payload_and_decode(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    scalar_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: random.Random,
) -> tuple[str, bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        prediction, meta = _decode_packet(example, None, code_projection, feature_dim, budget_bytes)
        return prediction, None, meta
    if condition == "matched_learned_syndrome":
        payload = _source_packet(example, encoder, code_projection, feature_dim, budget_bytes, mode="matched")
        prediction, meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
        return prediction, payload, meta | {"packet_family": "learned_syndrome"}
    if condition == "scalar_quantized_source":
        payload = _scalar_packet(
            example,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="matched",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized"}
    if condition == "scalar_shuffled_source":
        other = eval_examples[_deterministic_nonself_index(index, len(eval_examples))]
        payload = _scalar_packet(
            other,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="matched",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "source": other.example_id}
    if condition == "scalar_answer_masked_source":
        payload = _scalar_packet(
            example,
            encoder=encoder,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
            mode="answer_masked",
        )
        prediction, meta = _decode_scalar_packet(
            example,
            payload,
            scalar_projection=scalar_projection[:budget_bytes],
            feature_dim=feature_dim,
            lo=lo[:budget_bytes],
            hi=hi[:budget_bytes],
        )
        return prediction, payload, meta | {"packet_family": "scalar_quantized", "source": "answer_masked"}
    if condition == "raw_source_sign_sketch":
        payload = _raw_source_sign_packet(example, code_projection, feature_dim, budget_bytes)
        prediction, meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
        return prediction, payload, meta | {"packet_family": "raw_source_sign_sketch"}
    if condition == "random_same_byte":
        payload = rng.randbytes(budget_bytes)
        prediction, meta = _decode_packet(example, payload, code_projection, feature_dim, budget_bytes)
        return prediction, payload, meta | {"packet_family": "random"}
    raise ValueError(f"unknown condition {condition!r}")


def _predict(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    scalar_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: random.Random,
) -> dict[str, Any]:
    start = time.perf_counter()
    prediction, payload, metadata = _payload_and_decode(
        condition=condition,
        example=example,
        eval_examples=eval_examples,
        index=index,
        encoder=encoder,
        code_projection=code_projection,
        scalar_projection=scalar_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        lo=lo,
        hi=hi,
        rng=rng,
    )
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
        "metadata": metadata,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
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
    rng_np = np.random.default_rng(train_seed * 3001 + eval_seed)
    max_budget = max(budgets)
    code_projection = _normalize_rows(rng_np.normal(size=(max_budget * 8, feature_dim))).astype(np.float32)
    scalar_projection = _normalize_rows(rng_np.normal(size=(max_budget, feature_dim))).astype(np.float32)
    lo, hi = _fit_scalar_calibration(train_rows, encoder=encoder, scalar_projection=scalar_projection, feature_dim=feature_dim)
    rng = random.Random(train_seed * 4001 + eval_seed)
    conditions = [
        "target_only",
        "matched_learned_syndrome",
        "scalar_quantized_source",
        "scalar_shuffled_source",
        "scalar_answer_masked_source",
        "raw_source_sign_sketch",
        "zero_source",
        "random_same_byte",
    ]
    budget_summaries: list[dict[str, Any]] = []
    prediction_files: dict[str, str] = {}
    for budget in budgets:
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
        for row_index, example in enumerate(eval_rows):
            for condition in conditions:
                by_condition[condition].append(
                    _predict(
                        condition=condition,
                        example=example,
                        eval_examples=eval_rows,
                        index=row_index,
                        encoder=encoder,
                        code_projection=code_projection,
                        scalar_projection=scalar_projection,
                        feature_dim=feature_dim,
                        budget_bytes=budget,
                        lo=lo,
                        hi=hi,
                        rng=rng,
                    )
                    | {"example_id": example.example_id, "family_name": example.family_name, "budget_bytes": budget}
                )
        metrics = {condition: _summarize(rows) for condition, rows in by_condition.items()}
        no_source = max(metrics[name]["accuracy"] for name in ["target_only", "zero_source", "random_same_byte"])
        compression = max(
            metrics[name]["accuracy"]
            for name in ["scalar_quantized_source", "raw_source_sign_sketch"]
        )
        matched = metrics["matched_learned_syndrome"]["accuracy"]
        learned_vs_compression_pass = (
            matched >= no_source + 0.15
            and matched >= compression + 0.02
            and metrics["scalar_shuffled_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["scalar_answer_masked_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
        )
        scalar = metrics["scalar_quantized_source"]["accuracy"]
        scalar_controls_ok = (
            metrics["scalar_shuffled_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
            and metrics["scalar_answer_masked_source"]["accuracy"] <= metrics["target_only"]["accuracy"] + 0.05
        )
        scalar_source_packet_pass = scalar >= no_source + 0.15 and scalar_controls_ok
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for condition in conditions:
                for row in by_condition[condition]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_files[str(budget)] = predictions_path.name
        budget_summaries.append(
            {
                "budget_bytes": budget,
                "pass_gate": learned_vs_compression_pass,
                "learned_vs_compression_pass": learned_vs_compression_pass,
                "scalar_source_packet_pass": scalar_source_packet_pass,
                "matched_accuracy": matched,
                "scalar_quantized_source_accuracy": scalar,
                "target_only_accuracy": metrics["target_only"]["accuracy"],
                "best_no_source_accuracy": no_source,
                "best_compression_baseline_accuracy": compression,
                "matched_minus_best_no_source": matched - no_source,
                "matched_minus_best_compression": matched - compression,
                "scalar_minus_best_no_source": scalar - no_source,
                "scalar_controls_ok": scalar_controls_ok,
                "metrics": metrics,
            }
        )
    payload = {
        "gate": "source_private_tool_trace_compression_baselines",
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
        "exact_id_parity": len({row.example_id for row in eval_rows}) == len(eval_rows),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "scalar_projection_sha256": hashlib.sha256(scalar_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] or row["scalar_source_packet_pass"] for row in budget_summaries),
        "pass_rule": "learned syndrome pass: beats target/no-source by >=0.15 and beats best matched-byte compression baseline by >=0.02. Scalar packet pass: scalar quantized source packet beats no-source by >=0.15 and scalar source-destroying controls stay within target_only +0.05.",
        "prediction_files": prediction_files,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Source-Private Tool-Trace Compression Baselines",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{train_family_set}:{train_examples}` / `{eval_family_set}:{eval_examples}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget bytes | Learned > compression | Scalar pass | Syndrome | Scalar | Target | Best no-source | Syndrome - scalar |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_summaries:
        lines.append(
            f"| {row['budget_bytes']} | `{row['learned_vs_compression_pass']}` | "
            f"`{row['scalar_source_packet_pass']}` | {row['matched_accuracy']:.3f} | "
            f"{row['scalar_quantized_source_accuracy']:.3f} | {row['target_only_accuracy']:.3f} | "
            f"{row['best_no_source_accuracy']:.3f} | {row['matched_minus_best_compression']:.3f} |"
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
                "# Source-Private Tool-Trace Compression Baselines Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- budgets: `{budgets}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/source_private_tool_trace_compression_baselines_20260429"))
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budgets", type=int, nargs="+", default=[6])
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
