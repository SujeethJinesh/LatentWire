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
from dataclasses import replace
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_hidden_repair_packet_smoke import Example, _prior_prediction  # noqa: E402
from scripts.run_source_private_product_codebook_target_decoder_smoke import (  # noqa: E402
    ProductCodebookReceiverState,
    _candidate_code_signature,
    _candidate_distances,
    _condition_payload,
    _permute_payload_codes,
    _target_derived_payload,
    build_receiver_state,
)
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _constrained_nonself_index,
    _product_codebook_packet,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


CONTROL_CONDITIONS = [
    "zero_source",
    "label_shuffled_ridge",
    "constrained_shuffled_source",
    "answer_masked_source",
    "permuted_codes",
    "wrong_codebook_packet",
    "random_same_byte",
    "structured_json_same_byte",
    "structured_free_text_same_byte",
    "target_derived_sidecar",
]

EVAL_CONDITIONS = ["target_only", "matched_product_codebook", *CONTROL_CONDITIONS]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_for_rows(state: ProductCodebookReceiverState, rows: list[Example]) -> ProductCodebookReceiverState:
    return replace(state, eval_rows=rows)


def _payload_for_rows(
    *,
    condition: str,
    example: Example,
    state: ProductCodebookReceiverState,
    rows: list[Example],
    index: int,
    rng: random.Random,
) -> bytes | None:
    row_state = _state_for_rows(state, rows)
    payload, _ = _condition_payload(condition=condition, example=example, state=row_state, index=index, rng=rng)
    return payload


def _mask_payload(payload: bytes, *, round_index: int) -> bytes:
    if not payload:
        return payload
    values = bytearray(payload)
    values[round_index % len(values)] = 0
    return bytes(values)


def _deterministic_l2_prediction(
    example: Example,
    payload: bytes | None,
    state: ProductCodebookReceiverState,
) -> str:
    if not payload:
        return _prior_prediction(example)
    distances = _candidate_distances(example, payload, state)
    finite = [float(value) for value in distances if value is not None]
    if not finite:
        return _prior_prediction(example)
    min_distance = min(finite)
    tied = [idx for idx, value in enumerate(distances) if value is not None and abs(float(value) - min_distance) <= 1e-8]
    prior = _prior_prediction(example)
    if any(example.candidates[idx].label == prior for idx in tied):
        return prior
    return example.candidates[tied[0]].label


def _candidate_features(
    example: Example,
    payload: bytes | None,
    state: ProductCodebookReceiverState,
) -> np.ndarray:
    prior = _prior_prediction(example)
    signatures = _candidate_code_signature(example, state)
    payload_codes = list(payload or b"")
    distances_raw = _candidate_distances(example, payload, state)
    has_packet = float(payload is not None and len(payload) > 0)
    distances = np.array(
        [1.0e6 if value is None else float(value) for value in distances_raw],
        dtype=np.float32,
    )
    if has_packet:
        min_distance = float(np.min(distances))
        spread = float(np.std(distances) + 1e-6)
        ranks = np.argsort(np.argsort(distances, kind="stable"), kind="stable").astype(np.float32)
        norm_distance = (distances - min_distance) / spread
        hamming = np.array(
            [sum(int(code == sig_code) for code, sig_code in zip(payload_codes, signature)) for signature in signatures],
            dtype=np.float32,
        )
        hamming = hamming / max(1, len(payload_codes))
    else:
        min_distance = 0.0
        spread = 1.0
        ranks = np.zeros(len(example.candidates), dtype=np.float32)
        norm_distance = np.zeros(len(example.candidates), dtype=np.float32)
        hamming = np.zeros(len(example.candidates), dtype=np.float32)
    rows: list[list[float]] = []
    for idx, candidate in enumerate(example.candidates):
        prior_score = float(candidate.prior_score)
        is_prior = float(candidate.label == prior)
        rows.append(
            [
                1.0,
                has_packet,
                prior_score,
                is_prior,
                -float(distances[idx]) if has_packet else 0.0,
                -float(norm_distance[idx]) if has_packet else 0.0,
                float(ranks[idx]) / max(1, len(example.candidates) - 1),
                float(ranks[idx] == 0.0) if has_packet else 0.0,
                float(min_distance) if has_packet else 0.0,
                float(spread) if has_packet else 0.0,
                float(hamming[idx]),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _fit_score_receiver(
    state: ProductCodebookReceiverState,
    *,
    ridge: float,
    seed: int,
    mask_rounds: int,
    random_rounds: int,
    matched_weight: float,
    mask_weight: float,
    control_weight: float,
    target_only_weight: float,
) -> np.ndarray:
    rng = random.Random(seed)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    sample_weights: list[float] = []

    def add_training_view(example: Example, payload: bytes | None, target_label: str, weight: float) -> None:
        feats = _candidate_features(example, payload, state)
        for candidate, feature in zip(example.candidates, feats, strict=True):
            x_rows.append(feature)
            y_rows.append(float(candidate.label == target_label))
            sample_weights.append(float(weight))

    train_state = _state_for_rows(state, state.train_rows)
    for index, example in enumerate(state.train_rows):
        prior = _prior_prediction(example)
        matched = _product_codebook_packet(
            example,
            encoder=state.encoder,
            codebook=state.codebook,
            feature_dim=state.feature_dim,
            mode="matched",
        )
        add_training_view(example, None, prior, target_only_weight)
        add_training_view(example, matched, example.answer_label, matched_weight)
        for round_index in range(mask_rounds):
            add_training_view(example, _mask_payload(matched, round_index=round_index), example.answer_label, mask_weight)
        control_payloads: list[bytes | None] = [
            None,
            _product_codebook_packet(
                example,
                encoder=state.label_shuffle_encoder,
                codebook=state.codebook,
                feature_dim=state.feature_dim,
                mode="matched",
            ),
            _product_codebook_packet(
                train_state.eval_rows[_constrained_nonself_index(index, train_state.eval_rows)],
                encoder=state.encoder,
                codebook=state.codebook,
                feature_dim=state.feature_dim,
                mode="matched",
            ),
            _product_codebook_packet(
                example,
                encoder=state.encoder,
                codebook=state.codebook,
                feature_dim=state.feature_dim,
                mode="answer_masked",
            ),
            _permute_payload_codes(matched),
            _product_codebook_packet(
                example,
                encoder=state.encoder,
                codebook=state.wrong_codebook,
                feature_dim=state.feature_dim,
                mode="matched",
            ),
            _target_derived_payload(example, state),
        ]
        for payload in control_payloads:
            add_training_view(example, payload, prior, control_weight)
        for _ in range(random_rounds):
            add_training_view(example, rng.randbytes(state.codebook.subspaces), prior, control_weight)

    x = np.stack(x_rows, axis=0).astype(np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    weights = np.sqrt(np.asarray(sample_weights, dtype=np.float64))
    xw = x * weights[:, None]
    yw = y * weights
    xtx = xw.T @ xw
    xtx += ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= ridge
    return np.linalg.solve(xtx, xw.T @ yw).astype(np.float32)


def _receiver_prediction(example: Example, payload: bytes | None, state: ProductCodebookReceiverState, weights: np.ndarray) -> str:
    scores = _candidate_features(example, payload, state) @ weights
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score, rtol=1e-6, atol=1e-8))
    prior = _prior_prediction(example)
    if any(example.candidates[int(idx)].label == prior for idx in tied):
        return prior
    return example.candidates[int(tied[0])].label


def _predict_rows(
    state: ProductCodebookReceiverState,
    weights: np.ndarray,
    *,
    seed: int,
    conditions: list[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(state.eval_rows):
        for condition in conditions:
            start = time.perf_counter()
            payload = _payload_for_rows(
                condition=condition,
                example=example,
                state=state,
                rows=state.eval_rows,
                index=index,
                rng=rng,
            )
            learned_prediction = _receiver_prediction(example, payload, state, weights)
            learned_latency_ms = (time.perf_counter() - start) * 1000.0
            baseline_start = time.perf_counter()
            l2_prediction = _deterministic_l2_prediction(example, payload, state)
            l2_latency_ms = (time.perf_counter() - baseline_start) * 1000.0
            payload_hex = (payload or b"").hex()
            rows.append(
                {
                    "example_id": example.example_id,
                    "family_name": example.family_name,
                    "condition": condition,
                    "answer_label": example.answer_label,
                    "target_prior_label": _prior_prediction(example),
                    "payload_hex": payload_hex,
                    "payload_bytes": len(payload or b""),
                    "payload_tokens": _token_count(payload_hex),
                    "learned_prediction": learned_prediction,
                    "learned_correct": learned_prediction == example.answer_label,
                    "l2_prediction": l2_prediction,
                    "l2_correct": l2_prediction == example.answer_label,
                    "learned_latency_ms": learned_latency_ms,
                    "l2_latency_ms": l2_latency_ms,
                }
            )
    return rows


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    correct = [row["example_id"] for row in rows if row[key]]
    return {
        "correct": len(correct),
        "accuracy": len(correct) / len(rows),
        "correct_ids": correct,
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(row["learned_latency_ms" if key == "learned_correct" else "l2_latency_ms"] for row in rows),
    }


def _paired_bootstrap(rows: list[dict[str, Any]], *, condition_a: str, condition_b: str, key: str) -> dict[str, float]:
    ids = sorted({row["example_id"] for row in rows})
    by = {(row["condition"], row["example_id"]): bool(row[key]) for row in rows}
    diffs = np.array(
        [float(by.get((condition_a, example_id), False)) - float(by.get((condition_b, example_id), False)) for example_id in ids],
        dtype=np.float32,
    )
    point = float(np.mean(diffs)) if len(diffs) else 0.0
    if len(diffs) <= 1:
        return {"point": point, "ci95_low": point, "ci95_high": point}
    rng = np.random.default_rng(20260430 + sum(ord(ch) for ch in condition_a + condition_b + key))
    samples = np.empty(2000, dtype=np.float32)
    for sample_index in range(len(samples)):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        samples[sample_index] = float(np.mean(diffs[idx]))
    return {
        "point": point,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _summarize(rows: list[dict[str, Any]], *, conditions: list[str]) -> dict[str, Any]:
    example_ids = sorted({row["example_id"] for row in rows})
    learned_metrics: dict[str, Any] = {}
    l2_metrics: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        learned_metrics[condition] = _metric(condition_rows, "learned_correct")
        l2_metrics[condition] = _metric(condition_rows, "l2_correct")
    target = learned_metrics["target_only"]["accuracy"]
    matched = learned_metrics["matched_product_codebook"]["accuracy"]
    best_control_condition = max(CONTROL_CONDITIONS, key=lambda condition: learned_metrics[condition]["accuracy"])
    best_control = learned_metrics[best_control_condition]["accuracy"]
    l2_matched = l2_metrics["matched_product_codebook"]["accuracy"]
    exact_id_parity = len(rows) == len(example_ids) * len(conditions)
    learned_vs_target = _paired_bootstrap(rows, condition_a="matched_product_codebook", condition_b="target_only", key="learned_correct")
    learned_vs_l2_same = {
        "point": matched - l2_matched,
        "ci95_low": None,
        "ci95_high": None,
    }
    controls_ok = all(learned_metrics[condition]["accuracy"] <= target + 0.05 for condition in CONTROL_CONDITIONS)
    return {
        "n": len(example_ids),
        "conditions": conditions,
        "exact_id_count": len(example_ids),
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "exact_id_parity": exact_id_parity,
        "target_only_accuracy": target,
        "learned_matched_accuracy": matched,
        "l2_matched_accuracy": l2_matched,
        "best_control_condition": best_control_condition,
        "best_control_accuracy": best_control,
        "learned_minus_target": matched - target,
        "learned_minus_best_control": matched - best_control,
        "learned_minus_l2": matched - l2_matched,
        "learned_controls_ok": controls_ok,
        "pass_gate": exact_id_parity and matched >= target + 0.15 and controls_ok and matched >= l2_matched + 0.03,
        "source_packet_pass": exact_id_parity and matched >= target + 0.15 and controls_ok,
        "pass_rule": (
            "source_packet_pass requires learned matched PQ receiver to beat target-only by >=0.15 with all corrupt "
            "controls within target+0.05. pass_gate additionally requires learned matched accuracy to beat deterministic "
            "PQ L2 by >=0.03 at the same byte budget."
        ),
        "paired_bootstrap": {
            "learned_matched_vs_target": learned_vs_target,
            "learned_matched_vs_l2_same_condition": learned_vs_l2_same,
        },
        "learned_metrics": learned_metrics,
        "l2_metrics": l2_metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Masked-PQ Consistency Receiver",
        "",
        f"- examples: `{summary['n']}`",
        f"- source packet pass: `{summary['source_packet_pass']}`",
        f"- pass gate vs deterministic L2: `{summary['pass_gate']}`",
        f"- learned matched accuracy: `{summary['learned_matched_accuracy']:.3f}`",
        f"- deterministic L2 matched accuracy: `{summary['l2_matched_accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best learned control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
        f"- learned minus target: `{summary['learned_minus_target']:.3f}`",
        f"- learned minus deterministic L2: `{summary['learned_minus_l2']:.3f}`",
        "",
        "| Condition | Learned acc | L2 acc | Learned bytes | Learned p50 ms | L2 p50 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        learned = summary["learned_metrics"][condition]
        l2 = summary["l2_metrics"][condition]
        lines.append(
            f"| {condition} | {learned['accuracy']:.3f} | {l2['accuracy']:.3f} | "
            f"{learned['mean_payload_bytes']:.2f} | {learned['p50_latency_ms']:.4f} | {l2['p50_latency_ms']:.4f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest_markdown(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    s = manifest["summary"]
    path.write_text(
        "\n".join(
            [
                "# Source-Private Masked-PQ Consistency Receiver Manifest",
                "",
                "## Command",
                "",
                "```bash",
                manifest["command"],
                "```",
                "",
                "## Outcome",
                "",
                f"- source packet pass: `{s['source_packet_pass']}`",
                f"- pass gate: `{s['pass_gate']}`",
                f"- learned matched accuracy: `{s['learned_matched_accuracy']:.3f}`",
                f"- deterministic L2 matched accuracy: `{s['l2_matched_accuracy']:.3f}`",
                "",
                "## Artifacts",
                "",
                *[f"- `{artifact}`" for artifact in manifest["artifacts"]],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--train-family-set", default="all")
    parser.add_argument("--eval-family-set", default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--budget-bytes", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", default="slot")
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remap-slot-seed", type=int, default=101)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--mask-rounds", type=int, default=2)
    parser.add_argument("--random-rounds", type=int, default=2)
    parser.add_argument("--matched-weight", type=float, default=4.0)
    parser.add_argument("--mask-weight", type=float, default=2.0)
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument("--target-only-weight", type=float, default=1.0)
    parser.add_argument("--conditions", choices=EVAL_CONDITIONS, nargs="*", default=None)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(args.conditions or EVAL_CONDITIONS)
    state = build_receiver_state(
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        train_family_set=args.train_family_set,
        eval_family_set=args.eval_family_set,
        candidates=args.candidates,
        feature_dim=args.feature_dim,
        budget_bytes=args.budget_bytes,
        ridge=args.ridge,
        candidate_view=args.candidate_view,
        fit_intercept=args.fit_intercept,
        remap_slot_seed=args.remap_slot_seed,
    )
    weights = _fit_score_receiver(
        state,
        ridge=args.receiver_ridge,
        seed=args.seed,
        mask_rounds=args.mask_rounds,
        random_rounds=args.random_rounds,
        matched_weight=args.matched_weight,
        mask_weight=args.mask_weight,
        control_weight=args.control_weight,
        target_only_weight=args.target_only_weight,
    )
    rows = _predict_rows(state, weights, seed=args.seed + 101, conditions=conditions)
    summary = _summarize(rows, conditions=conditions)
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    (output_dir / "receiver_weights.json").write_text(json.dumps([float(value) for value in weights], indent=2), encoding="utf-8")
    artifacts = ["predictions.jsonl", "summary.json", "summary.md", "receiver_weights.json", "manifest.json", "manifest.md"]
    manifest = {
        "command": " ".join(sys.argv),
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "artifacts": artifacts,
        "artifact_sha256": {
            artifact: _sha256_file(output_dir / artifact)
            for artifact in artifacts
            if artifact not in {"manifest.json", "manifest.md"}
        },
        "python": sys.version,
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_manifest_markdown(output_dir / "manifest.md", manifest)
    if args.require_pass and not summary["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
