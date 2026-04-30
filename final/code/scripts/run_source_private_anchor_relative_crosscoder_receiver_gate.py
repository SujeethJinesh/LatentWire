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

from scripts import run_source_private_candidate_embedding_receiver as base  # noqa: E402
from scripts.run_source_private_hidden_repair_packet_smoke import (  # noqa: E402
    Example,
    _deterministic_nonself_index,
    _prior_prediction,
    make_benchmark,
)
from scripts.run_source_private_tool_trace_compression_baselines import (  # noqa: E402
    _candidate_matrix_for_view,
)
from scripts.run_source_private_tool_trace_learned_syndrome import _token_count  # noqa: E402


CONDITIONS = (
    "target_only",
    "matched_anchor_relative_receiver",
    "zero_source",
    "public_only_sidecar",
    "shuffled_source",
    "answer_masked_source",
    "random_same_byte",
    "answer_only",
    "structured_text_matched",
    "target_derived_sidecar",
    "feature_id_permutation",
    "top_feature_knockout",
    "full_diag_oracle",
)

SOURCE_DESTROYING_CONTROLS = (
    "zero_source",
    "public_only_sidecar",
    "shuffled_source",
    "answer_masked_source",
    "random_same_byte",
    "answer_only",
    "target_derived_sidecar",
    "feature_id_permutation",
)

_CANDIDATE_VIEW = "diag_only"


def _candidate_matrix_mode_for_gate(example: Example, feature_dim: int) -> np.ndarray:
    if base._FEATURE_BACKEND != "hashed":
        raise ValueError("anchor-relative crosscoder gate currently supports hashed features only")
    key = (id(example), feature_dim, f"{base._feature_cache_tag()}:{_CANDIDATE_VIEW}")
    cached = base._CANDIDATE_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    matrix = _candidate_matrix_for_view(example, feature_dim, candidate_view=_CANDIDATE_VIEW)
    base._CANDIDATE_MATRIX_CACHE[key] = matrix
    return matrix


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_order_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    baseline: str,
    seed: int,
    samples: int = 1000,
) -> dict[str, float]:
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    deltas = []
    for example_id, conditions in sorted(by_example.items()):
        if condition not in conditions or baseline not in conditions:
            raise ValueError(f"missing {condition!r}/{baseline!r} row for {example_id}")
        deltas.append(float(conditions[condition]["correct"]) - float(conditions[baseline]["correct"]))
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": statistics.fmean(deltas),
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def _bits_for_source(
    example: Example,
    *,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = base._augment(base._source_vector_mode(example, feature_dim, mode=mode)) @ encoder
    bit_count = budget_bytes * 8
    logits = code_projection[:bit_count] @ predicted
    return (logits >= 0).astype(np.uint8), logits.astype(np.float32)


def _bits_to_payload(bits: np.ndarray, budget_bytes: int) -> bytes:
    return base._bitpack(bits.astype(np.uint8), budget_bytes)


def _permute_payload_bits(payload: bytes, *, budget_bytes: int) -> bytes:
    bit_count = budget_bytes * 8
    bits = base._bytes_to_bits(payload, bit_count).astype(np.uint8)
    if len(bits) <= 1:
        return payload
    shift = min(7, len(bits) - 1)
    permuted = np.roll(bits, shift)
    return _bits_to_payload(permuted, budget_bytes)


def _top_feature_knockout_payload(
    example: Example,
    *,
    encoder: np.ndarray,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    bits, logits = _bits_for_source(
        example,
        encoder=encoder,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        mode="matched",
    )
    top_index = int(np.argmax(np.abs(logits))) if len(logits) else 0
    if len(bits):
        bits[top_index] = 1 - bits[top_index]
    return _bits_to_payload(bits, budget_bytes), {
        "source": "matched_top_feature_flipped",
        "top_feature_index": top_index,
        "top_feature_abs_logit": float(abs(logits[top_index])) if len(logits) else 0.0,
    }


def _public_mean_packet(
    example: Example,
    *,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
) -> bytes:
    candidates = base._packet_candidate_matrix(
        example,
        feature_dim=feature_dim,
        packet_feature_mode=packet_feature_mode,
        anchor_matrix=anchor_matrix,
    )
    return base._packet_from_vector(candidates.mean(axis=0), code_projection, budget_bytes)


def _target_prior_packet(
    example: Example,
    *,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
) -> bytes:
    prior = _prior_prediction(example)
    prior_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == prior)
    candidates = base._packet_candidate_matrix(
        example,
        feature_dim=feature_dim,
        packet_feature_mode=packet_feature_mode,
        anchor_matrix=anchor_matrix,
    )
    return base._packet_from_vector(candidates[prior_index], code_projection, budget_bytes)


def _oracle_packet(
    example: Example,
    *,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
) -> bytes:
    answer_index = next(idx for idx, candidate in enumerate(example.candidates) if candidate.label == example.answer_label)
    candidates = base._packet_candidate_matrix(
        example,
        feature_dim=feature_dim,
        packet_feature_mode=packet_feature_mode,
        anchor_matrix=anchor_matrix,
    )
    return base._packet_from_vector(candidates[answer_index], code_projection, budget_bytes)


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
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
    rng: random.Random,
) -> tuple[bytes | None, dict[str, Any]]:
    if condition in {"target_only", "zero_source"}:
        return None, {}
    if condition == "matched_anchor_relative_receiver":
        return (
            base._source_packet(
                example,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                mode="matched",
            ),
            {"source": example.example_id},
        )
    if condition == "public_only_sidecar":
        return (
            _public_mean_packet(
                example,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                packet_feature_mode=packet_feature_mode,
                anchor_matrix=anchor_matrix,
            ),
            {"source": "public_candidate_mean"},
        )
    if condition == "shuffled_source":
        other = eval_examples[_deterministic_nonself_index(index, len(eval_examples))]
        return (
            base._source_packet(
                other,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                mode="matched",
            ),
            {"source": other.example_id},
        )
    if condition == "answer_masked_source":
        return (
            base._source_packet(
                example,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                mode="answer_masked",
            ),
            {"source": "answer_masked"},
        )
    if condition == "random_same_byte":
        return rng.randbytes(budget_bytes), {"source": "random_same_byte"}
    if condition == "answer_only":
        return example.answer_label.encode("utf-8")[:budget_bytes], {"source": "answer_label_text"}
    if condition == "structured_text_matched":
        return example.private_test_log.encode("utf-8")[:budget_bytes], {"source": "truncated_hidden_log"}
    if condition == "target_derived_sidecar":
        return (
            _target_prior_packet(
                example,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                packet_feature_mode=packet_feature_mode,
                anchor_matrix=anchor_matrix,
            ),
            {"source": "target_prior_candidate"},
        )
    if condition == "feature_id_permutation":
        payload = base._source_packet(
            example,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        return _permute_payload_bits(payload, budget_bytes=budget_bytes), {"source": "matched_bit_permutation"}
    if condition == "top_feature_knockout":
        return _top_feature_knockout_payload(
            example,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
        )
    if condition == "full_diag_oracle":
        return (
            _oracle_packet(
                example,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                packet_feature_mode=packet_feature_mode,
                anchor_matrix=anchor_matrix,
            ),
            {"source": "candidate_oracle"},
        )
    raise ValueError(f"unknown condition {condition!r}")


def _predict_condition(
    *,
    condition: str,
    example: Example,
    eval_examples: list[Example],
    index: int,
    encoder: np.ndarray,
    receiver_kind: str,
    receiver: np.ndarray | None,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    candidate_feature_dims: int,
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
    margin_threshold: float,
    rng: random.Random,
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
        packet_feature_mode=packet_feature_mode,
        anchor_matrix=anchor_matrix,
        rng=rng,
    )
    prediction, decode_meta = base._predict_with_receiver(
        example,
        payload,
        receiver_kind=receiver_kind,
        receiver=receiver,
        code_projection=code_projection,
        feature_dim=feature_dim,
        budget_bytes=budget_bytes,
        candidate_feature_dims=candidate_feature_dims,
        packet_feature_mode=packet_feature_mode,
        anchor_matrix=anchor_matrix,
        margin_threshold=0.0 if condition == "full_diag_oracle" else margin_threshold,
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
        "metadata": {**metadata, **decode_meta},
    }


def _evaluate_for_threshold(
    examples: list[Example],
    *,
    conditions: tuple[str, ...],
    encoder: np.ndarray,
    receiver_kind: str,
    receiver: np.ndarray | None,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    candidate_feature_dims: int,
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
    margin_threshold: float,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    by_condition: dict[str, list[bool]] = {condition: [] for condition in conditions}
    for row_index, example in enumerate(examples):
        for condition in conditions:
            row = _predict_condition(
                condition=condition,
                example=example,
                eval_examples=examples,
                index=row_index,
                encoder=encoder,
                receiver_kind=receiver_kind,
                receiver=receiver,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget_bytes,
                candidate_feature_dims=candidate_feature_dims,
                packet_feature_mode=packet_feature_mode,
                anchor_matrix=anchor_matrix,
                margin_threshold=margin_threshold,
                rng=rng,
            )
            by_condition[condition].append(row["correct"])
    return {condition: sum(values) / len(values) for condition, values in by_condition.items()}


def _calibrate_margin_threshold(
    calibration_examples: list[Example],
    *,
    encoder: np.ndarray,
    receiver_kind: str,
    receiver: np.ndarray | None,
    code_projection: np.ndarray,
    feature_dim: int,
    budget_bytes: int,
    candidate_feature_dims: int,
    packet_feature_mode: str,
    anchor_matrix: np.ndarray | None,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    candidate_thresholds = [0.0]
    for example in calibration_examples:
        payload = base._source_packet(
            example,
            encoder=encoder,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            mode="matched",
        )
        scores = base._score_candidates(
            example,
            payload,
            receiver_kind=receiver_kind,
            receiver=receiver,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            candidate_feature_dims=candidate_feature_dims,
            packet_feature_mode=packet_feature_mode,
            anchor_matrix=anchor_matrix,
        )
        labels = [candidate.label for candidate in example.candidates]
        prior_index = labels.index(_prior_prediction(example))
        candidate_thresholds.append(max(0.0, float(np.max(scores) - scores[prior_index])))
    ordered = sorted(candidate_thresholds)
    quantiles = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
    thresholds = sorted(
        {
            round(ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))], 6)
            for q in quantiles
        }
        | {0.0}
    )
    best_threshold = 0.0
    best_record: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = _evaluate_for_threshold(
            calibration_examples,
            conditions=("target_only", "matched_anchor_relative_receiver", *SOURCE_DESTROYING_CONTROLS, "structured_text_matched"),
            encoder=encoder,
            receiver_kind=receiver_kind,
            receiver=receiver,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget_bytes,
            candidate_feature_dims=candidate_feature_dims,
            packet_feature_mode=packet_feature_mode,
            anchor_matrix=anchor_matrix,
            margin_threshold=threshold,
            seed=seed,
        )
        target = metrics["target_only"]
        best_control = max(metrics[name] for name in SOURCE_DESTROYING_CONTROLS)
        text_relay = metrics["structured_text_matched"]
        record = {
            "threshold": threshold,
            "matched_accuracy": metrics["matched_anchor_relative_receiver"],
            "target_accuracy": target,
            "best_control_accuracy": best_control,
            "structured_text_accuracy": text_relay,
            "controls_ok": all(metrics[name] <= target + 0.02 for name in SOURCE_DESTROYING_CONTROLS),
            "text_relay_ok": text_relay <= target + 0.02
            or metrics["matched_anchor_relative_receiver"] >= text_relay + 0.15,
        }
        if best_record is None:
            best_record = record
            best_threshold = threshold
        elif record["controls_ok"] and record["text_relay_ok"] and (
            not (best_record["controls_ok"] and best_record["text_relay_ok"])
            or record["matched_accuracy"] > best_record["matched_accuracy"]
            or (
                record["matched_accuracy"] == best_record["matched_accuracy"]
                and record["best_control_accuracy"] < best_record["best_control_accuracy"]
            )
        ):
            best_record = record
            best_threshold = threshold
        elif not (best_record["controls_ok"] and best_record["text_relay_ok"]) and (
            record["best_control_accuracy"],
            -record["matched_accuracy"],
        ) < (
            best_record["best_control_accuracy"],
            -best_record["matched_accuracy"],
        ):
            best_record = record
            best_threshold = threshold
    assert best_record is not None
    return best_threshold, best_record


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    return {
        "n": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
        "mean_payload_bytes": statistics.fmean(row["payload_bytes"] for row in rows),
        "mean_payload_tokens": statistics.fmean(row["payload_tokens"] for row in rows),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def _condition_id_parity(by_condition: dict[str, list[dict[str, Any]]], expected_ids: list[str]) -> dict[str, Any]:
    order_hashes = {
        condition: _id_order_sha256([row["example_id"] for row in rows]) for condition, rows in by_condition.items()
    }
    expected_hash = _id_order_sha256(expected_ids)
    return {
        "expected_id_order_sha256": expected_hash,
        "condition_id_order_sha256": order_hashes,
        "all_conditions_same_order": all(value == expected_hash for value in order_hashes.values()),
    }


def _budget_summary(
    *,
    rows: list[dict[str, Any]],
    by_condition: dict[str, list[dict[str, Any]]],
    budget_bytes: int,
    receiver_kind: str,
    receiver_digest: str,
    margin_threshold: float,
    calibration: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    metrics = {condition: _summarize(condition_rows) for condition, condition_rows in by_condition.items()}
    target = metrics["target_only"]["accuracy"]
    matched = metrics["matched_anchor_relative_receiver"]["accuracy"]
    best_control_name = max(SOURCE_DESTROYING_CONTROLS, key=lambda name: metrics[name]["accuracy"])
    best_control = metrics[best_control_name]["accuracy"]
    structured_text = metrics["structured_text_matched"]["accuracy"]
    top_knockout = metrics["top_feature_knockout"]["accuracy"]
    oracle = metrics["full_diag_oracle"]["accuracy"]
    bootstrap_target = _paired_bootstrap(
        rows,
        condition="matched_anchor_relative_receiver",
        baseline="target_only",
        seed=seed,
    )
    bootstrap_control = _paired_bootstrap(
        rows,
        condition="matched_anchor_relative_receiver",
        baseline=best_control_name,
        seed=seed + 1,
    )
    lift = matched - target
    knockout_reduction = 0.0 if lift <= 0 else max(0.0, matched - top_knockout) / lift
    controls_ok = all(metrics[name]["accuracy"] <= target + 0.02 for name in SOURCE_DESTROYING_CONTROLS)
    text_relay_ok = structured_text <= target + 0.02 or matched >= structured_text + 0.15
    pass_gate = (
        matched >= target + 0.15
        and matched >= best_control + 0.15
        and controls_ok
        and text_relay_ok
        and oracle >= 0.95
        and bootstrap_target["ci95_low"] >= 0.10
        and bootstrap_control["ci95_low"] >= 0.10
    )
    return {
        "budget_bytes": budget_bytes,
        "pass_gate": pass_gate,
        "receiver_kind": receiver_kind,
        "receiver_sha256": receiver_digest,
        "matched_accuracy": matched,
        "target_only_accuracy": target,
        "best_source_destroying_control": best_control_name,
        "best_source_destroying_control_accuracy": best_control,
        "structured_text_matched_accuracy": structured_text,
        "top_feature_knockout_accuracy": top_knockout,
        "top_feature_knockout_lift_reduction": knockout_reduction,
        "full_diag_oracle_accuracy": oracle,
        "matched_minus_target": matched - target,
        "matched_minus_best_source_destroying_control": matched - best_control,
        "matched_minus_structured_text": matched - structured_text,
        "controls_ok": controls_ok,
        "text_relay_ok": text_relay_ok,
        "paired_bootstrap_vs_target": bootstrap_target,
        "paired_bootstrap_vs_best_control": bootstrap_control,
        "margin_threshold": margin_threshold,
        "margin_calibration": calibration,
        "metrics": metrics,
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
    candidate_feature_dims: int,
    receiver_kind: str,
    packet_feature_mode: str,
    anchor_count: int,
    candidate_view: str,
    diagnostic_table_mode: str,
    budgets: list[int],
    train_seed: int,
    eval_seed: int,
    ridge: float,
) -> dict[str, Any]:
    global _CANDIDATE_VIEW
    _CANDIDATE_VIEW = candidate_view
    previous_candidate_matrix_mode = base._candidate_matrix_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    base._FEATURE_BACKEND = "hashed"
    base._CANDIDATE_MATRIX_CACHE.clear()
    base._PACKET_CANDIDATE_MATRIX_CACHE.clear()
    base._SOURCE_VECTOR_CACHE.clear()
    base._candidate_matrix_mode = _candidate_matrix_mode_for_gate
    train_rows = make_benchmark(
        examples=train_examples,
        candidates=candidates,
        seed=train_seed,
        family_set=train_family_set,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    eval_rows = make_benchmark(
        examples=eval_examples,
        candidates=candidates,
        seed=eval_seed,
        family_set=eval_family_set,
        diagnostic_table_mode=diagnostic_table_mode,
    )
    packet_dim = feature_dim
    anchor_matrix = None
    anchor_build_mode = "none"
    if packet_feature_mode == "anchor_relative":
        packet_dim = anchor_count
        anchor_matrix = base._build_anchor_matrix(train_rows, feature_dim=feature_dim, anchor_count=anchor_count)
        anchor_build_mode = "first_train_candidates"
    elif packet_feature_mode == "learned_anchor_relative":
        packet_dim = anchor_count
        anchor_matrix = base._build_learned_anchor_matrix(
            train_rows,
            feature_dim=feature_dim,
            anchor_count=anchor_count,
            seed=train_seed * 1009 + eval_seed,
        )
        anchor_build_mode = "deterministic_spherical_kmeans"
    elif packet_feature_mode != "hashed":
        raise ValueError(f"unknown packet feature mode {packet_feature_mode!r}")
    encoder = base._fit_cached_ridge_encoder(
        train_rows,
        feature_dim=feature_dim,
        packet_feature_mode=packet_feature_mode,
        anchor_matrix=anchor_matrix,
        ridge=ridge,
    )
    rng_np = np.random.default_rng(train_seed * 1009 + eval_seed)
    code_projection = base._normalize_rows(rng_np.normal(size=(max(budgets) * 8, packet_dim))).astype(np.float32)
    prediction_files: dict[str, str] = {}
    budget_summaries: list[dict[str, Any]] = []
    eval_ids = [example.example_id for example in eval_rows]
    for budget in budgets:
        if receiver_kind == "ridge":
            receiver: np.ndarray | None = base._fit_receiver(
                train_rows,
                encoder=encoder,
                code_projection=code_projection,
                feature_dim=feature_dim,
                budget_bytes=budget,
                candidate_feature_dims=candidate_feature_dims,
                packet_feature_mode=packet_feature_mode,
                anchor_matrix=anchor_matrix,
                ridge=ridge,
            )
            receiver_digest = hashlib.sha256(receiver.tobytes()).hexdigest()
        elif receiver_kind == "code_similarity":
            receiver = None
            receiver_digest = "code_similarity"
        else:
            raise ValueError(f"unknown receiver kind {receiver_kind!r}")
        margin_threshold, calibration = _calibrate_margin_threshold(
            train_rows,
            encoder=encoder,
            receiver_kind=receiver_kind,
            receiver=receiver,
            code_projection=code_projection,
            feature_dim=feature_dim,
            budget_bytes=budget,
            candidate_feature_dims=candidate_feature_dims,
            packet_feature_mode=packet_feature_mode,
            anchor_matrix=anchor_matrix,
            seed=train_seed * 3011 + eval_seed + budget,
        )
        rng = random.Random(train_seed * 2003 + eval_seed + budget)
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
        rows: list[dict[str, Any]] = []
        for row_index, example in enumerate(eval_rows):
            for condition in CONDITIONS:
                row = _predict_condition(
                    condition=condition,
                    example=example,
                    eval_examples=eval_rows,
                    index=row_index,
                    encoder=encoder,
                    receiver_kind=receiver_kind,
                    receiver=receiver,
                    code_projection=code_projection,
                    feature_dim=feature_dim,
                    budget_bytes=budget,
                    candidate_feature_dims=candidate_feature_dims,
                    packet_feature_mode=packet_feature_mode,
                    anchor_matrix=anchor_matrix,
                    margin_threshold=margin_threshold,
                    rng=rng,
                ) | {
                    "example_id": example.example_id,
                    "family_name": example.family_name,
                    "budget_bytes": budget,
                    "candidate_labels": [candidate.label for candidate in example.candidates],
                }
                by_condition[condition].append(row)
                rows.append(row)
        parity = _condition_id_parity(by_condition, eval_ids)
        predictions_path = output_dir / f"predictions_budget{budget}.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        prediction_files[str(budget)] = predictions_path.name
        summary = _budget_summary(
            rows=rows,
            by_condition=by_condition,
            budget_bytes=budget,
            receiver_kind=receiver_kind,
            receiver_digest=receiver_digest,
            margin_threshold=margin_threshold,
            calibration=calibration,
            seed=train_seed + eval_seed + budget,
        )
        summary["condition_id_parity"] = parity
        summary["exact_ordered_id_parity"] = parity["all_conditions_same_order"]
        summary["pass_gate"] = bool(summary["pass_gate"] and parity["all_conditions_same_order"])
        budget_summaries.append(summary)
    payload = {
        "gate": "source_private_anchor_relative_crosscoder_receiver_gate",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_family_set": train_family_set,
        "eval_family_set": eval_family_set,
        "candidates": candidates,
        "feature_dim": feature_dim,
        "candidate_feature_dims": candidate_feature_dims,
        "candidate_view": candidate_view,
        "diagnostic_table_mode": diagnostic_table_mode,
        "feature_backend": "hashed",
        "packet_feature_mode": packet_feature_mode,
        "packet_dim": packet_dim,
        "anchor_count": anchor_count if packet_feature_mode in {"anchor_relative", "learned_anchor_relative"} else 0,
        "anchor_build_mode": anchor_build_mode,
        "receiver_kind": receiver_kind,
        "budgets": budgets,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "ridge": ridge,
        "conditions": list(CONDITIONS),
        "source_destroying_controls": list(SOURCE_DESTROYING_CONTROLS),
        "exact_id_parity": len(eval_ids) == len(set(eval_ids)),
        "exact_id_sha256": _id_order_sha256(eval_ids),
        "candidate_pool_recall": 1.0,
        "encoder_sha256": hashlib.sha256(encoder.tobytes()).hexdigest(),
        "code_projection_sha256": hashlib.sha256(code_projection.tobytes()).hexdigest(),
        "budget_summaries": budget_summaries,
        "pass_gate": any(row["pass_gate"] for row in budget_summaries),
        "prediction_files": prediction_files,
        "pass_rule": (
            "Matched anchor-relative packet receiver must beat target-only and best source-destroying control "
            "by >=0.15, keep every source-destroying control within target+0.02, avoid being explained by "
            "matched-byte structured text, keep full diagnostic oracle >=0.95, have paired CI95 lower bounds "
            ">=0.10 versus target and best control, and preserve exact ordered-ID parity."
        ),
        "lay_summary": (
            "The source sees a private clue and sends a tiny sign-code fingerprint. The target sees only the "
            "public question and candidate pool plus that fingerprint. Controls replace, mask, shuffle, or "
            "publicly derive the fingerprint to check that any gain comes from private evidence."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", payload)
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
                "# Anchor-Relative Crosscoder Receiver Gate Manifest",
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
    base._candidate_matrix_mode = previous_candidate_matrix_mode
    return payload


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private Anchor-Relative Crosscoder Receiver Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/eval: `{payload['train_family_set']}:{payload['train_examples']}` / `{payload['eval_family_set']}:{payload['eval_examples']}`",
        f"- candidate view: `{payload['candidate_view']}`",
        f"- diagnostic table mode: `{payload['diagnostic_table_mode']}`",
        f"- receiver kind: `{payload['receiver_kind']}`",
        f"- packet feature mode: `{payload['packet_feature_mode']}`",
        f"- packet dim: `{payload['packet_dim']}`",
        f"- exact ID parity: `{payload['exact_id_parity']}`",
        "",
        "| Budget bytes | Pass | Matched | Target | Best control | Text relay | Top knockout | Oracle | CI95 low target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["budget_summaries"]:
        lines.append(
            f"| {row['budget_bytes']} | `{row['pass_gate']}` | {row['matched_accuracy']:.3f} | "
            f"{row['target_only_accuracy']:.3f} | {row['best_source_destroying_control_accuracy']:.3f} | "
            f"{row['structured_text_matched_accuracy']:.3f} | {row['top_feature_knockout_accuracy']:.3f} | "
            f"{row['full_diag_oracle_accuracy']:.3f} | {row['paired_bootstrap_vs_target']['ci95_low']:.3f} |"
        )
    lines.extend(["", "## Pass Rule", "", payload["pass_rule"], "", "## Lay Summary", "", payload["lay_summary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_anchor_relative_crosscoder_receiver_20260430"),
    )
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--train-family-set", choices=["core", "holdout", "all"], default="core")
    parser.add_argument("--eval-family-set", choices=["core", "holdout", "all"], default="holdout")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--candidate-feature-dims", type=int, default=0)
    parser.add_argument("--receiver-kind", choices=["ridge", "code_similarity"], default="ridge")
    parser.add_argument(
        "--packet-feature-mode",
        choices=["hashed", "anchor_relative", "learned_anchor_relative"],
        default="anchor_relative",
    )
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--candidate-view", choices=["full", "no_diag", "semantic", "diag_only", "slot"], default="diag_only")
    parser.add_argument("--diagnostic-table-mode", choices=["legacy", "plausible_decoys"], default="plausible_decoys")
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8])
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
        candidate_feature_dims=args.candidate_feature_dims,
        receiver_kind=args.receiver_kind,
        packet_feature_mode=args.packet_feature_mode,
        anchor_count=args.anchor_count,
        candidate_view=args.candidate_view,
        diagnostic_table_mode=args.diagnostic_table_mode,
        budgets=args.budgets,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        ridge=args.ridge,
    )
    print(json.dumps({"pass_gate": payload["pass_gate"], "output_dir": str(out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
