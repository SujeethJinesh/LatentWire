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
from dataclasses import dataclass
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
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
    _fit_ridge_encoder_for_view,
    _project_source,
)
from scripts.run_source_private_tool_trace_learned_syndrome import (  # noqa: E402
    _bytes_to_bits,
    _decode_packet,
    _packet_from_vector,
    _source_packet,
    _token_count,
)


CONTROL_CONDITIONS = [
    "zero_source",
    "shuffled_source",
    "answer_masked_source",
    "random_same_byte",
    "target_derived_sidecar",
    "answer_only",
    "structured_text_matched",
    "wrong_projection_source",
]
EVAL_CONDITIONS = [
    "target_only",
    "matched_consistency_packet",
    "masked_matched_packet",
    *CONTROL_CONDITIONS,
    "full_diag_oracle",
]


@dataclass(frozen=True)
class ConsistencyReceiverState:
    train_rows: list[Example]
    eval_rows: list[Example]
    encoder: np.ndarray
    wrong_encoder: np.ndarray
    code_projection: np.ndarray
    feature_dim: int
    budget_bytes: int
    candidate_view: str
    fit_intercept: bool

    @property
    def bit_count(self) -> int:
        return self.budget_bytes * 8


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(denom, 1e-8)


def _fit_state(
    *,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    budget_bytes: int,
    ridge: float,
    candidate_view: str,
    fit_intercept: bool,
) -> ConsistencyReceiverState:
    train_rows = make_benchmark(examples=train_examples, candidates=candidates, seed=train_seed, family_set=train_family_set)
    eval_rows = make_benchmark(examples=eval_examples, candidates=candidates, seed=eval_seed, family_set=eval_family_set)
    encoder = _fit_ridge_encoder_for_view(
        train_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        candidate_view=candidate_view,
        fit_intercept=fit_intercept,
    )
    wrong_encoder = _fit_ridge_encoder_for_view(
        train_rows,
        feature_dim=feature_dim,
        ridge=ridge,
        candidate_view=candidate_view,
        fit_intercept=fit_intercept,
        label_shuffle_seed=train_seed * 1009 + eval_seed,
    )
    rng_np = np.random.default_rng(train_seed * 2003 + eval_seed)
    code_projection = _normalize_rows(rng_np.normal(size=(budget_bytes * 8, feature_dim))).astype(np.float32)
    return ConsistencyReceiverState(
        train_rows=train_rows,
        eval_rows=eval_rows,
        encoder=encoder,
        wrong_encoder=wrong_encoder,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        candidate_view=candidate_view,
        fit_intercept=fit_intercept,
    )


def _source_packet_for_state(example: Example, state: ConsistencyReceiverState, *, mode: str, wrong: bool = False) -> bytes:
    encoder = state.wrong_encoder if wrong else state.encoder
    if encoder.shape[0] == state.feature_dim + 1:
        return _source_packet(
            example,
            encoder,
            state.code_projection,
            state.feature_dim,
            state.budget_bytes,
            mode=mode,
        )
    predicted = _project_source(example, encoder=encoder, feature_dim=state.feature_dim, mode=mode)
    return _packet_from_vector(predicted, state.code_projection, state.budget_bytes)


def _target_derived_packet(example: Example, state: ConsistencyReceiverState) -> bytes:
    prior = _prior_prediction(example)
    prior_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)
    candidate_vectors = _candidate_matrix_for_view(example, state.feature_dim, candidate_view=state.candidate_view)
    return _packet_from_vector(candidate_vectors[prior_index], state.code_projection, state.budget_bytes)


def _oracle_packet(example: Example, state: ConsistencyReceiverState) -> bytes:
    answer_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)
    candidate_vectors = _candidate_matrix_for_view(example, state.feature_dim, candidate_view=state.candidate_view)
    return _packet_from_vector(candidate_vectors[answer_index], state.code_projection, state.budget_bytes)


def _mask_for_payload(bit_count: int, *, rng: random.Random, keep_prob: float) -> np.ndarray:
    mask = np.asarray([1.0 if rng.random() < keep_prob else 0.0 for _ in range(bit_count)], dtype=np.float32)
    if float(mask.sum()) <= 0.0:
        mask[rng.randrange(bit_count)] = 1.0
    return mask


def _payload_for_condition(
    *,
    condition: str,
    example: Example,
    rows: list[Example],
    index: int,
    state: ConsistencyReceiverState,
    rng: random.Random,
) -> tuple[bytes | None, np.ndarray | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, None, {"source": condition}
    if condition == "matched_consistency_packet":
        return _source_packet_for_state(example, state, mode="matched"), None, {"source": example.example_id}
    if condition == "masked_matched_packet":
        payload = _source_packet_for_state(example, state, mode="matched")
        return payload, _mask_for_payload(state.bit_count, rng=rng, keep_prob=0.65), {"source": example.example_id, "masked": True}
    if condition == "shuffled_source":
        other = rows[_deterministic_nonself_index(index, len(rows))]
        return _source_packet_for_state(other, state, mode="matched"), None, {"source": other.example_id}
    if condition == "answer_masked_source":
        return _source_packet_for_state(example, state, mode="answer_masked"), None, {"source": "answer_masked"}
    if condition == "random_same_byte":
        return rng.randbytes(state.budget_bytes), None, {"source": "random"}
    if condition == "target_derived_sidecar":
        return _target_derived_packet(example, state), None, {"source": "target_prior"}
    if condition == "answer_only":
        return example.answer_label.encode("utf-8")[: state.budget_bytes], None, {"source": "answer_label_text"}
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[: state.budget_bytes], None, {"source": "truncated_hidden_log"}
    if condition == "wrong_projection_source":
        return _source_packet_for_state(example, state, mode="matched", wrong=True), None, {"source": "wrong_projection"}
    if condition == "full_diag_oracle":
        return _oracle_packet(example, state), None, {"source": "candidate_oracle"}
    raise ValueError(f"unknown condition {condition!r}")


def _candidate_sign_codes(example: Example, state: ConsistencyReceiverState) -> np.ndarray:
    candidate_vectors = _candidate_matrix_for_view(example, state.feature_dim, candidate_view=state.candidate_view)
    logits = candidate_vectors @ state.code_projection[: state.bit_count].T
    return np.where(logits >= 0.0, 1.0, -1.0).astype(np.float32)


def _payload_signs(payload: bytes | None, state: ConsistencyReceiverState) -> np.ndarray:
    bits = _bytes_to_bits(payload, state.bit_count)
    return np.where(bits > 0, 1.0, -1.0).astype(np.float32)


def _receiver_features(
    example: Example,
    payload: bytes | None,
    observed_mask: np.ndarray | None,
    state: ConsistencyReceiverState,
) -> np.ndarray:
    prior = _prior_prediction(example)
    candidate_vectors = _candidate_matrix_for_view(example, state.feature_dim, candidate_view=state.candidate_view)
    has_packet = float(payload is not None and len(payload) > 0)
    if has_packet:
        mask = np.ones(state.bit_count, dtype=np.float32) if observed_mask is None else observed_mask.astype(np.float32)
        observed = max(1.0, float(mask.sum()))
        packet_sign = _payload_signs(payload, state)
        candidate_sign = _candidate_sign_codes(example, state)
        agreement = (candidate_sign * packet_sign[None, :] * mask[None, :]).sum(axis=1) / observed
        match_frac = ((candidate_sign == packet_sign[None, :]).astype(np.float32) * mask[None, :]).sum(axis=1) / observed
        packet_proxy = (mask * packet_sign) @ state.code_projection[: state.bit_count] / max(1.0, observed**0.5)
        packet_proxy = packet_proxy.astype(np.float32)
        vector_dot = candidate_vectors @ packet_proxy
        ranks = np.argsort(np.argsort(-agreement, kind="stable"), kind="stable").astype(np.float32)
        best_agreement = float(np.max(agreement))
        spread = float(np.std(agreement) + 1e-6)
        observed_fraction = observed / float(state.bit_count)
    else:
        agreement = np.zeros(len(example.candidates), dtype=np.float32)
        match_frac = np.zeros(len(example.candidates), dtype=np.float32)
        vector_dot = np.zeros(len(example.candidates), dtype=np.float32)
        ranks = np.zeros(len(example.candidates), dtype=np.float32)
        best_agreement = 0.0
        spread = 1.0
        observed_fraction = 0.0

    rows: list[list[float]] = []
    for idx, candidate in enumerate(example.candidates):
        is_prior = float(candidate.label == prior)
        rank_norm = float(ranks[idx]) / max(1.0, float(len(example.candidates) - 1))
        rows.append(
            [
                1.0,
                has_packet,
                observed_fraction,
                float(candidate.prior_score),
                is_prior,
                float(agreement[idx]),
                float(match_frac[idx]),
                float(vector_dot[idx]),
                float(rank_norm),
                float(rank_norm == 0.0 and has_packet),
                float(best_agreement),
                float(spread),
                float(is_prior * agreement[idx]),
                float(is_prior * match_frac[idx]),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _fit_receiver(
    state: ConsistencyReceiverState,
    *,
    seed: int,
    receiver_ridge: float,
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
    weights: list[float] = []

    def add_view(example: Example, payload: bytes | None, mask: np.ndarray | None, target_label: str, weight: float) -> None:
        features = _receiver_features(example, payload, mask, state)
        for candidate, feature in zip(example.candidates, features, strict=True):
            x_rows.append(feature)
            y_rows.append(float(candidate.label == target_label))
            weights.append(weight)

    for index, example in enumerate(state.train_rows):
        prior = _prior_prediction(example)
        matched = _source_packet_for_state(example, state, mode="matched")
        add_view(example, None, None, prior, target_only_weight)
        add_view(example, matched, None, example.answer_label, matched_weight)
        for _ in range(mask_rounds):
            add_view(
                example,
                matched,
                _mask_for_payload(state.bit_count, rng=rng, keep_prob=rng.uniform(0.45, 0.85)),
                example.answer_label,
                mask_weight,
            )
        control_payloads: list[bytes | None] = [
            None,
            _source_packet_for_state(
                state.train_rows[_deterministic_nonself_index(index, len(state.train_rows))],
                state,
                mode="matched",
            ),
            _source_packet_for_state(example, state, mode="answer_masked"),
            _target_derived_packet(example, state),
            example.answer_label.encode("utf-8")[: state.budget_bytes],
            example.private_test_log.encode("utf-8")[: state.budget_bytes],
            _source_packet_for_state(example, state, mode="matched", wrong=True),
        ]
        for payload in control_payloads:
            add_view(example, payload, None, prior, control_weight)
        for _ in range(random_rounds):
            add_view(example, rng.randbytes(state.budget_bytes), None, prior, control_weight)

    x = np.stack(x_rows, axis=0).astype(np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    sample_weights = np.sqrt(np.asarray(weights, dtype=np.float64))
    xw = x * sample_weights[:, None]
    yw = y * sample_weights
    xtx = xw.T @ xw
    xtx += receiver_ridge * np.eye(xtx.shape[0], dtype=np.float64)
    xtx[0, 0] -= receiver_ridge
    return np.linalg.solve(xtx, xw.T @ yw).astype(np.float32)


def _learned_prediction(
    example: Example,
    payload: bytes | None,
    mask: np.ndarray | None,
    state: ConsistencyReceiverState,
    receiver_weights: np.ndarray,
) -> tuple[str, list[float]]:
    scores = _receiver_features(example, payload, mask, state) @ receiver_weights
    max_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, max_score, rtol=1e-6, atol=1e-8))
    prior = _prior_prediction(example)
    if any(example.candidates[int(idx)].label == prior for idx in tied):
        return prior, [float(value) for value in scores]
    return example.candidates[int(tied[0])].label, [float(value) for value in scores]


def _hamming_prediction(example: Example, payload: bytes | None, state: ConsistencyReceiverState) -> str:
    if not payload:
        return _prior_prediction(example)
    prediction, _ = _decode_packet(example, payload, state.code_projection, state.feature_dim, state.budget_bytes)
    return prediction


def _predict_rows(
    state: ConsistencyReceiverState,
    receiver_weights: np.ndarray,
    *,
    seed: int,
    conditions: list[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(state.eval_rows):
        for condition in conditions:
            start = time.perf_counter()
            payload, mask, metadata = _payload_for_condition(
                condition=condition,
                example=example,
                rows=state.eval_rows,
                index=index,
                state=state,
                rng=rng,
            )
            learned, learned_scores = _learned_prediction(example, payload, mask, state, receiver_weights)
            learned_latency_ms = (time.perf_counter() - start) * 1000.0
            hamming_start = time.perf_counter()
            hamming = _hamming_prediction(example, payload, state)
            hamming_latency_ms = (time.perf_counter() - hamming_start) * 1000.0
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
                    "observed_bits": None if mask is None else int(mask.sum()),
                    "learned_prediction": learned,
                    "learned_correct": learned == example.answer_label,
                    "hamming_prediction": hamming,
                    "hamming_correct": hamming == example.answer_label,
                    "learned_scores": learned_scores,
                    "learned_latency_ms": learned_latency_ms,
                    "hamming_latency_ms": hamming_latency_ms,
                    "metadata": metadata,
                }
            )
    return rows


def _metric(rows: list[dict[str, Any]], key: str, latency_key: str) -> dict[str, Any]:
    correct = [row["example_id"] for row in rows if row[key]]
    return {
        "correct": len(correct),
        "accuracy": len(correct) / len(rows),
        "correct_ids": correct,
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(row[latency_key] for row in rows),
        "p95_latency_ms": sorted(row[latency_key] for row in rows)[max(0, int(0.95 * len(rows)) - 1)],
    }


def _paired_bootstrap(rows: list[dict[str, Any]], *, condition_a: str, condition_b: str, key: str) -> dict[str, float]:
    ids = sorted({row["example_id"] for row in rows})
    by = {(row["condition"], row["example_id"]): bool(row[key]) for row in rows}
    diffs = np.asarray(
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


def _summarize(rows: list[dict[str, Any]], *, conditions: list[str], tolerance_vs_hamming: float) -> dict[str, Any]:
    example_ids = sorted({row["example_id"] for row in rows})
    learned_metrics: dict[str, Any] = {}
    hamming_metrics: dict[str, Any] = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        learned_metrics[condition] = _metric(condition_rows, "learned_correct", "learned_latency_ms")
        hamming_metrics[condition] = _metric(condition_rows, "hamming_correct", "hamming_latency_ms")
    target = learned_metrics["target_only"]["accuracy"]
    matched = learned_metrics["matched_consistency_packet"]["accuracy"]
    hamming_matched = hamming_metrics["matched_consistency_packet"]["accuracy"]
    control_names = [condition for condition in CONTROL_CONDITIONS if condition in learned_metrics]
    best_control_condition = max(control_names, key=lambda condition: learned_metrics[condition]["accuracy"])
    best_control = learned_metrics[best_control_condition]["accuracy"]
    controls_ok = all(learned_metrics[condition]["accuracy"] <= target + 0.05 for condition in control_names)
    exact_id_parity = len(rows) == len(example_ids) * len(conditions)
    source_packet_pass = exact_id_parity and matched >= target + 0.15 and matched >= best_control + 0.15 and controls_ok
    replacement_pass = source_packet_pass and matched >= hamming_matched - tolerance_vs_hamming
    return {
        "n": len(example_ids),
        "conditions": conditions,
        "exact_id_count": len(example_ids),
        "exact_id_sha256": hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest(),
        "exact_id_parity": exact_id_parity,
        "target_only_accuracy": target,
        "learned_matched_accuracy": matched,
        "hamming_matched_accuracy": hamming_matched,
        "best_control_condition": best_control_condition,
        "best_control_accuracy": best_control,
        "learned_minus_target": matched - target,
        "learned_minus_best_control": matched - best_control,
        "learned_minus_hamming": matched - hamming_matched,
        "learned_controls_ok": controls_ok,
        "source_packet_pass": source_packet_pass,
        "pass_gate": replacement_pass,
        "pass_rule": (
            "Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= "
            "best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet "
            f"within {tolerance_vs_hamming:.2f} accuracy of deterministic Hamming packet decoding."
        ),
        "paired_bootstrap": {
            "learned_matched_vs_target": _paired_bootstrap(
                rows,
                condition_a="matched_consistency_packet",
                condition_b="target_only",
                key="learned_correct",
            ),
            "learned_matched_vs_best_control": _paired_bootstrap(
                rows,
                condition_a="matched_consistency_packet",
                condition_b=best_control_condition,
                key="learned_correct",
            ),
            "learned_matched_vs_hamming_same_condition": {
                "point": matched - hamming_matched,
                "ci95_low": None,
                "ci95_high": None,
            },
        },
        "learned_metrics": learned_metrics,
        "hamming_metrics": hamming_metrics,
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_markdown(path: pathlib.Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Masked Consistency Receiver Smoke",
        "",
        f"- examples: `{summary['n']}`",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- source packet pass: `{summary['source_packet_pass']}`",
        f"- learned matched accuracy: `{summary['learned_matched_accuracy']:.3f}`",
        f"- deterministic Hamming matched accuracy: `{summary['hamming_matched_accuracy']:.3f}`",
        f"- target-only accuracy: `{summary['target_only_accuracy']:.3f}`",
        f"- best learned control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
        f"- learned minus target: `{summary['learned_minus_target']:.3f}`",
        f"- learned minus best control: `{summary['learned_minus_best_control']:.3f}`",
        f"- learned minus Hamming: `{summary['learned_minus_hamming']:.3f}`",
        "",
        "| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in summary["conditions"]:
        learned = summary["learned_metrics"][condition]
        hamming = summary["hamming_metrics"][condition]
        lines.append(
            f"| {condition} | {learned['accuracy']:.3f} | {hamming['accuracy']:.3f} | "
            f"{learned['mean_payload_bytes']:.2f} | {learned['p50_latency_ms']:.4f} | {hamming['p50_latency_ms']:.4f} |"
        )
    lines.extend(["", f"Pass rule: {summary['pass_rule']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    output_dir: pathlib.Path,
    train_examples: int,
    eval_examples: int,
    train_seed: int,
    eval_seed: int,
    train_family_set: str,
    eval_family_set: str,
    candidates: int,
    feature_dim: int,
    budget_bytes: int,
    ridge: float,
    receiver_ridge: float,
    candidate_view: str,
    fit_intercept: bool,
    seed: int,
    mask_rounds: int,
    random_rounds: int,
    matched_weight: float,
    mask_weight: float,
    control_weight: float,
    target_only_weight: float,
    tolerance_vs_hamming: float,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _fit_state(
        train_examples=train_examples,
        eval_examples=eval_examples,
        train_seed=train_seed,
        eval_seed=eval_seed,
        train_family_set=train_family_set,
        eval_family_set=eval_family_set,
        candidates=candidates,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        ridge=ridge,
        candidate_view=candidate_view,
        fit_intercept=fit_intercept,
    )
    receiver_weights = _fit_receiver(
        state,
        seed=seed,
        receiver_ridge=receiver_ridge,
        mask_rounds=mask_rounds,
        random_rounds=random_rounds,
        matched_weight=matched_weight,
        mask_weight=mask_weight,
        control_weight=control_weight,
        target_only_weight=target_only_weight,
    )
    eval_conditions = list(conditions or EVAL_CONDITIONS)
    rows = _predict_rows(state, receiver_weights, seed=seed + 104729, conditions=eval_conditions)
    summary = _summarize(rows, conditions=eval_conditions, tolerance_vs_hamming=tolerance_vs_hamming)
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    (output_dir / "receiver_weights.json").write_text(
        json.dumps([float(value) for value in receiver_weights], indent=2) + "\n",
        encoding="utf-8",
    )
    exact_ids = [row.example_id for row in state.eval_rows]
    payload = {
        "gate": "source_private_masked_consistency_receiver_smoke",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "budget_bytes": budget_bytes,
        "ridge": ridge,
        "receiver_ridge": receiver_ridge,
        "candidate_view": candidate_view,
        "fit_intercept": fit_intercept,
        "seed": seed,
        "mask_rounds": mask_rounds,
        "random_rounds": random_rounds,
        "matched_weight": matched_weight,
        "mask_weight": mask_weight,
        "control_weight": control_weight,
        "target_only_weight": target_only_weight,
        "tolerance_vs_hamming": tolerance_vs_hamming,
        "exact_id_parity": summary["exact_id_parity"],
        "exact_id_sha256": hashlib.sha256("\n".join(exact_ids).encode("utf-8")).hexdigest(),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(state.encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(state.code_projection.tobytes()).hexdigest(),
        "receiver_weights_sha256": hashlib.sha256(receiver_weights.tobytes()).hexdigest(),
        "summary": summary,
        "pass_gate": summary["pass_gate"],
        "prediction_file": "predictions.jsonl",
        "pass_rule": summary["pass_rule"],
        "interpretation": (
            "One-step masked-consistency receiver over learned syndrome bytes. Training forces clean/masked "
            "matched packets toward the gold candidate and destructive controls toward the target prior. "
            "This tests whether a learned receiver can replace a hand-written nearest-neighbor packet decoder."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "artifacts": [
            "run_summary.json",
            "summary.json",
            "summary.md",
            "predictions.jsonl",
            "receiver_weights.json",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in ["run_summary.json", "summary.json", "summary.md", "predictions.jsonl", "receiver_weights.json"]
        },
        "pass_gate": payload["pass_gate"],
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private Masked Consistency Receiver Smoke Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- source packet pass: `{summary['source_packet_pass']}`",
                f"- learned matched accuracy: `{summary['learned_matched_accuracy']:.3f}`",
                f"- Hamming matched accuracy: `{summary['hamming_matched_accuracy']:.3f}`",
                f"- best control: `{summary['best_control_condition']}` at `{summary['best_control_accuracy']:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=30)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="all")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--budget-bytes", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--receiver-ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "slot"], default="full")
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--mask-rounds", type=int, default=4)
    parser.add_argument("--random-rounds", type=int, default=2)
    parser.add_argument("--matched-weight", type=float, default=4.0)
    parser.add_argument("--mask-weight", type=float, default=2.0)
    parser.add_argument("--control-weight", type=float, default=2.0)
    parser.add_argument("--target-only-weight", type=float, default=2.0)
    parser.add_argument("--tolerance-vs-hamming", type=float, default=0.05)
    parser.add_argument("--conditions", choices=EVAL_CONDITIONS, nargs="*", default=None)
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = run_gate(
        output_dir=out,
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
        receiver_ridge=args.receiver_ridge,
        candidate_view=args.candidate_view,
        fit_intercept=args.fit_intercept,
        seed=args.seed,
        mask_rounds=args.mask_rounds,
        random_rounds=args.random_rounds,
        matched_weight=args.matched_weight,
        mask_weight=args.mask_weight,
        control_weight=args.control_weight,
        target_only_weight=args.target_only_weight,
        tolerance_vs_hamming=args.tolerance_vs_hamming,
        conditions=args.conditions,
    )
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "pass_gate": payload["pass_gate"],
                "source_packet_pass": payload["summary"]["source_packet_pass"],
                "learned_matched_accuracy": payload["summary"]["learned_matched_accuracy"],
                "hamming_matched_accuracy": payload["summary"]["hamming_matched_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
