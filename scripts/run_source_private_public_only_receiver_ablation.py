from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
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
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _remap_candidate_slots,
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_features(example: Example, *, feature_dim: int, candidate_view: str, include_vectors: bool) -> np.ndarray:
    prior = _prior_prediction(example)
    base_rows: list[list[float]] = []
    for candidate in example.candidates:
        base_rows.append(
            [
                1.0,
                float(candidate.prior_score),
                float(candidate.label == prior),
            ]
        )
    base = np.asarray(base_rows, dtype=np.float32)
    if not include_vectors:
        return base
    vectors = _candidate_matrix_for_view(example, feature_dim, candidate_view=candidate_view).astype(np.float32)
    return np.concatenate([base, vectors], axis=1).astype(np.float32)


def _fit_public_receiver(
    train_rows: list[Example],
    *,
    feature_dim: int,
    candidate_view: str,
    ridge: float,
    include_vectors: bool,
) -> np.ndarray:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for example in train_rows:
        features = _public_features(
            example,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            include_vectors=include_vectors,
        )
        for candidate, feature in zip(example.candidates, features, strict=True):
            x_rows.append(feature)
            y_rows.append(float(candidate.label == example.answer_label))
    x = np.stack(x_rows, axis=0).astype(np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    xtx = x.T @ x
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= ridge
    return np.linalg.solve(xtx, x.T @ y).astype(np.float32)


def _predict_public(
    example: Example,
    weights: np.ndarray,
    *,
    feature_dim: int,
    candidate_view: str,
    include_vectors: bool,
) -> tuple[str, list[float]]:
    scores = _public_features(
        example,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
        include_vectors=include_vectors,
    ) @ weights
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score, rtol=1e-6, atol=1e-8))
    prior = _prior_prediction(example)
    if any(example.candidates[int(idx)].label == prior for idx in tied):
        return prior, [float(value) for value in scores]
    return example.candidates[int(tied[0])].label, [float(value) for value in scores]


def _paired_bootstrap(target_correct: list[bool], public_correct: list[bool]) -> dict[str, float]:
    diffs = np.asarray(
        [float(public) - float(target) for target, public in zip(target_correct, public_correct, strict=True)],
        dtype=np.float32,
    )
    point = float(np.mean(diffs)) if len(diffs) else 0.0
    if len(diffs) <= 1:
        return {"point": point, "ci95_low": point, "ci95_high": point}
    rng = np.random.default_rng(20260430 + len(diffs))
    samples = np.empty(2000, dtype=np.float32)
    for sample_index in range(len(samples)):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        samples[sample_index] = float(np.mean(diffs[idx]))
    return {
        "point": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_start_index: int,
    eval_start_index: int,
    train_family_set: str,
    eval_family_set: str,
    diagnostic_table_mode: str,
    candidates: int,
    feature_dim: int,
    ridge: float,
    candidate_view: str,
    remap_slot_seed: int | None,
    include_vectors: bool,
    max_allowed_lift: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = make_benchmark(
        examples=train_examples,
        candidates=candidates,
        seed=train_seed,
        family_set=train_family_set,
        start_index=train_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    eval_rows = make_benchmark(
        examples=eval_examples,
        candidates=candidates,
        seed=eval_seed,
        family_set=eval_family_set,
        start_index=eval_start_index,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    train_rows = _remap_candidate_slots(train_rows, remap_seed=remap_slot_seed)
    eval_rows = _remap_candidate_slots(eval_rows, remap_seed=remap_slot_seed)
    weights = _fit_public_receiver(
        train_rows,
        feature_dim=feature_dim,
        candidate_view=candidate_view,
        ridge=ridge,
        include_vectors=include_vectors,
    )
    rows: list[dict[str, Any]] = []
    for example in eval_rows:
        start = time.perf_counter()
        prediction, scores = _predict_public(
            example,
            weights,
            feature_dim=feature_dim,
            candidate_view=candidate_view,
            include_vectors=include_vectors,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        target_prior = _prior_prediction(example)
        rows.append(
            {
                "example_id": example.example_id,
                "family_name": example.family_name,
                "answer_label": example.answer_label,
                "target_prior_label": target_prior,
                "public_prediction": prediction,
                "public_correct": prediction == example.answer_label,
                "target_correct": target_prior == example.answer_label,
                "public_scores": scores,
                "latency_ms": latency_ms,
            }
        )
    public_correct = [bool(row["public_correct"]) for row in rows]
    target_correct = [bool(row["target_correct"]) for row in rows]
    public_accuracy = sum(public_correct) / len(public_correct)
    target_accuracy = sum(target_correct) / len(target_correct)
    lift = public_accuracy - target_accuracy
    exact_ids = [row.example_id for row in eval_rows]
    train_ids = [row.example_id for row in train_rows]
    train_eval_intersection = sorted(set(train_ids).intersection(exact_ids))
    summary = {
        "n": len(rows),
        "candidate_view": candidate_view,
        "remap_slot_seed": remap_slot_seed,
        "include_vectors": include_vectors,
        "exact_id_sha256": hashlib.sha256("\n".join(exact_ids).encode("utf-8")).hexdigest(),
        "train_id_sha256": hashlib.sha256("\n".join(train_ids).encode("utf-8")).hexdigest(),
        "train_eval_id_intersection_count": len(train_eval_intersection),
        "train_eval_id_intersection_sample": train_eval_intersection[:10],
        "target_only_accuracy": target_accuracy,
        "public_only_accuracy": public_accuracy,
        "public_minus_target": lift,
        "public_only_no_leak": lift <= max_allowed_lift,
        "max_allowed_lift": max_allowed_lift,
        "paired_bootstrap_public_vs_target": _paired_bootstrap(target_correct, public_correct),
        "p50_latency_ms": statistics.median(row["latency_ms"] for row in rows),
        "p95_latency_ms": sorted(row["latency_ms"] for row in rows)[max(0, int(0.95 * len(rows)) - 1)],
    }
    payload = {
        "gate": "source_private_public_only_receiver_ablation",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_start_index": train_start_index,
        "eval_start_index": eval_start_index,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "diagnostic_table_mode": diagnostic_table_mode,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "ridge": ridge,
        "candidate_view": candidate_view,
        "remap_slot_seed": remap_slot_seed,
        "include_vectors": include_vectors,
        "receiver_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "summary": summary,
        "pass_gate": summary["public_only_no_leak"],
        "pass_rule": (
            f"Public-only candidate classifier should stay within +{max_allowed_lift:.2f} accuracy of target-only "
            "to rule out public candidate semantics as a sufficient explanation for packet gains."
        ),
        "interpretation": (
            "A strong public-only ablation trained on public candidate vectors but no source packet. If it is high, "
            "the semantic candidate-view packet result is not a clean source-causality claim on this task."
        ),
    }
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "receiver_weights.json").write_text(
        json.dumps([float(value) for value in weights], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Source-Private Public-Only Receiver Ablation",
        "",
        f"- examples: `{summary['n']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- candidate view: `{candidate_view}`",
        f"- train/eval ID overlap: `{summary['train_eval_id_intersection_count']}`",
        f"- public-only accuracy: `{public_accuracy:.3f}`",
        f"- target-only accuracy: `{target_accuracy:.3f}`",
        f"- public minus target: `{lift:.3f}`",
        f"- CI95 vs target: `[{summary['paired_bootstrap_public_vs_target']['ci95_low']:.3f}, "
        f"{summary['paired_bootstrap_public_vs_target']['ci95_high']:.3f}]`",
        f"- p50 latency ms: `{summary['p50_latency_ms']:.4f}`",
        "",
        f"Pass rule: {payload['pass_rule']}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "artifacts": ["run_summary.json", "summary.json", "summary.md", "predictions.jsonl", "receiver_weights.json"],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["run_summary.json", "summary.json", "summary.md", "predictions.jsonl", "receiver_weights.json"]
        },
        "pass_gate": payload["pass_gate"],
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Public-Only Receiver Ablation Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- public-only accuracy: `{public_accuracy:.3f}`",
                f"- target-only accuracy: `{target_accuracy:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--train-start-index", type=int, default=0)
    parser.add_argument("--eval-start-index", type=int, default=10000)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="legacy")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "diag_only", "slot"], default="semantic")
    parser.add_argument("--remap-slot-seed", type=int, default=None)
    parser.add_argument("--include-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-allowed-lift", type=float, default=0.05)
    parser.add_argument("--require-no-leak", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        train_start_index=args.train_start_index,
        eval_start_index=args.eval_start_index,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        diagnostic_table_mode=args.diagnostic_table_mode,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        ridge=args.ridge,
        candidate_view=args.candidate_view,
        remap_slot_seed=args.remap_slot_seed,
        include_vectors=args.include_vectors,
        max_allowed_lift=args.max_allowed_lift,
    )
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "pass_gate": payload["pass_gate"],
                "public_only_accuracy": payload["summary"]["public_only_accuracy"],
                "target_only_accuracy": payload["summary"]["target_only_accuracy"],
                "public_minus_target": payload["summary"]["public_minus_target"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_no_leak and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
