from __future__ import annotations

"""Held-out uncertainty-router gate for Qwen-to-Phi HellaSwag packets.

This gate tests a smoother receiver than the harm-controlled bucket gate while
preserving the packet contract: Qwen source scores may only be used by the
source-side encoder to emit candidate IDs and quantized uncertainty bins. The
receiver may combine that tiny packet with Phi-local scores, then decide
whether to keep the fixed Qwen hybrid answer or accept a bounded override.
"""

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as nonqwen  # noqa: E402
from scripts import build_source_private_hellaswag_official_train_receiver_calibration as official  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate as denoise  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as receiver_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as source_gate  # noqa: E402
from scripts import build_source_private_hellaswag_qwen_to_phi_oracle_switch_decomposition_gate as oracle  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate_20260504_validation1024_2048"
)
DEFAULT_TRAIN_PATH = source_gate.DEFAULT_TRAIN_PATH
DEFAULT_QWEN_TRAIN_CACHE_DIR = source_gate.DEFAULT_QWEN_TRAIN_CACHE_DIR
DEFAULT_SOURCE_SCORE_CACHE = source_gate.DEFAULT_SOURCE_SCORE_CACHE
DEFAULT_SAMPLE_SEEDS = source_gate.DEFAULT_SAMPLE_SEEDS
DEFAULT_SPLIT_SEEDS = source_gate.DEFAULT_SPLIT_SEEDS
DEFAULT_COMPONENT_RIDGES = source_gate.DEFAULT_COMPONENT_RIDGES
DEFAULT_TARGET_MODEL = pathlib.Path(nonqwen.DEFAULT_PHI_MODEL)
DEFAULT_PHI_TRAIN_SCORE_CACHE = receiver_gate.DEFAULT_OUTPUT / "phi_official_train_score_cache.json"
DEFAULT_RIDGES = source_gate.DICTIONARY_RIDGES
BOOTSTRAP_SAMPLES = denoise.BOOTSTRAP_SAMPLES

ACTION_NAMES = ("hybrid", "qwen_top1", "qwen_top2", "phi_top1", "qwen_mean")
SOURCE_CONDITIONS = (
    "source_row_shuffle",
    "source_score_row_shuffle_before_encoding",
    "candidate_roll_source",
    "code_value_permutation",
    "target_derived_source_packet",
    "random_same_byte_source",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: pathlib.Path | str) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _answers(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _field_array(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([int(row[field]) for row in rows], dtype=np.int64)


def _accuracy(predictions: np.ndarray, answers: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == np.asarray(answers, dtype=np.int64)))


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    return denoise._paired_ci(
        selected=np.asarray(selected, dtype=np.int64),
        baseline=np.asarray(baseline, dtype=np.int64),
        answers=np.asarray(answers, dtype=np.int64),
        seed=seed,
        samples=samples,
    )


def _top2(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), axis=1)
    return order[:, 0].astype(np.int64), order[:, 1].astype(np.int64)


def _z_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return centered / np.where(scale > 1e-8, scale, 1.0)


def _entropy(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)


def _quantile_bins(values: np.ndarray, quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> list[float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    return sorted(set(float(value) for value in np.quantile(values, quantiles)))


def _digitize(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=np.float64), np.asarray(list(bins), dtype=np.float64)).astype(np.int64)


def _one_hot(values: np.ndarray, width: int) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    return [(values == option).astype(np.float64) for option in range(int(width))]


def _threshold_features(values: np.ndarray, thresholds: tuple[float, ...]) -> list[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return [(values < threshold).astype(np.float64) for threshold in thresholds]


def _fit_bins(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    qwen_margin: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, list[float]]:
    qwen_scores = np.asarray(qwen_scores, dtype=np.float64)
    phi_scores = np.asarray(phi_scores, dtype=np.float64)
    hybrid = np.asarray(hybrid, dtype=np.int64)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    row_ids = np.arange(qwen_scores.shape[0])
    q_top1, q_top2 = _top2(qwen_scores)
    p_top1, p_top2 = _top2(phi_scores)
    qz = _z_scores(qwen_scores)
    pz = _z_scores(phi_scores)
    q_margin = qwen_scores[row_ids, q_top1] - qwen_scores[row_ids, q_top2]
    q_entropy = _entropy(qwen_scores)
    q_hybrid_gap = qz[row_ids, q_top1] - qz[row_ids, hybrid]
    p_margin = phi_scores[row_ids, p_top1] - phi_scores[row_ids, p_top2]
    p_entropy = _entropy(phi_scores)
    p_hybrid_gap = pz[row_ids, p_top1] - pz[row_ids, hybrid]
    return {
        "q_margin": _quantile_bins(q_margin[fit_indices]),
        "q_entropy": _quantile_bins(q_entropy[fit_indices]),
        "q_hybrid_gap": _quantile_bins(q_hybrid_gap[fit_indices]),
        "qwen_selected_margin": _quantile_bins(np.asarray(qwen_margin, dtype=np.float64)[fit_indices]),
        "p_margin": _quantile_bins(p_margin[fit_indices]),
        "p_entropy": _quantile_bins(p_entropy[fit_indices]),
        "p_hybrid_gap": _quantile_bins(p_hybrid_gap[fit_indices]),
    }


def _stack_router_features(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    qwen_mean: np.ndarray,
    qwen_margin: np.ndarray,
    bins: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    qwen_scores = np.asarray(qwen_scores, dtype=np.float64)
    phi_scores = np.asarray(phi_scores, dtype=np.float64)
    hybrid = np.asarray(hybrid, dtype=np.int64)
    qwen_mean = np.asarray(qwen_mean, dtype=np.int64)
    qwen_margin = np.asarray(qwen_margin, dtype=np.float64)
    row_ids = np.arange(qwen_scores.shape[0])
    q_top1, q_top2 = _top2(qwen_scores)
    p_top1, p_top2 = _top2(phi_scores)
    qz = _z_scores(qwen_scores)
    pz = _z_scores(phi_scores)
    q_margin_value = qwen_scores[row_ids, q_top1] - qwen_scores[row_ids, q_top2]
    q_entropy_value = _entropy(qwen_scores)
    q_hybrid_gap_value = qz[row_ids, q_top1] - qz[row_ids, hybrid]
    p_margin_value = phi_scores[row_ids, p_top1] - phi_scores[row_ids, p_top2]
    p_entropy_value = _entropy(phi_scores)
    p_hybrid_gap_value = pz[row_ids, p_top1] - pz[row_ids, hybrid]
    source_bins = {
        "q_margin_bin": _digitize(q_margin_value, bins["q_margin"]),
        "q_entropy_bin": _digitize(q_entropy_value, bins["q_entropy"]),
        "q_hybrid_gap_bin": _digitize(q_hybrid_gap_value, bins["q_hybrid_gap"]),
        "qwen_selected_margin_bin": _digitize(qwen_margin, bins["qwen_selected_margin"]),
        "p_margin_bin": _digitize(p_margin_value, bins["p_margin"]),
        "p_entropy_bin": _digitize(p_entropy_value, bins["p_entropy"]),
        "p_hybrid_gap_bin": _digitize(p_hybrid_gap_value, bins["p_hybrid_gap"]),
    }
    actions = np.stack([hybrid, q_top1, q_top2, p_top1, qwen_mean], axis=1).astype(np.int64)
    features: list[np.ndarray] = []
    for role in range(actions.shape[1]):
        candidate = actions[:, role]
        p_action_gap = pz[row_ids, candidate] - pz[row_ids, hybrid]
        p_action_top_gap = pz[row_ids, p_top1] - pz[row_ids, candidate]
        parts: list[np.ndarray] = [np.ones(qwen_scores.shape[0], dtype=np.float64)]
        parts.extend(_one_hot(np.full(qwen_scores.shape[0], role, dtype=np.int64), len(ACTION_NAMES)))
        for ids in (candidate, hybrid, q_top1, q_top2, p_top1, qwen_mean):
            parts.extend(_one_hot(ids, 4))
        for values in source_bins.values():
            parts.extend(_one_hot(values, 5))
        parts.extend(
            [
                (candidate == hybrid).astype(np.float64),
                (candidate == q_top1).astype(np.float64),
                (candidate == q_top2).astype(np.float64),
                (candidate == p_top1).astype(np.float64),
                (candidate == p_top2).astype(np.float64),
                (candidate == qwen_mean).astype(np.float64),
                (hybrid == q_top1).astype(np.float64),
                (hybrid == q_top2).astype(np.float64),
                (p_top1 == hybrid).astype(np.float64),
                (p_top1 == q_top1).astype(np.float64),
                (p_top1 == q_top2).astype(np.float64),
                pz[row_ids, candidate],
                p_action_gap,
                p_action_top_gap,
                pz[row_ids, hybrid],
                pz[row_ids, p_top1],
                p_margin_value,
                p_entropy_value,
                p_hybrid_gap_value,
            ]
        )
        for values in (p_action_gap, p_action_top_gap, p_margin_value, p_entropy_value, p_hybrid_gap_value):
            parts.extend(_threshold_features(values, (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)))
        features.append(np.vstack(parts).T.astype(np.float64))
    diagnostics = {
        "q_top1": q_top1,
        "q_top2": q_top2,
        "p_top1": p_top1,
        "p_top2": p_top2,
        **source_bins,
    }
    return np.stack(features, axis=1), actions, diagnostics


def _predict_router(
    action_features: np.ndarray,
    action_candidates: np.ndarray,
    hybrid: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    scores = np.asarray(action_features, dtype=np.float64) @ weights
    best_action = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_action]
    predictions = np.asarray(hybrid, dtype=np.int64).copy()
    mask = best_score > float(model["threshold"])
    predictions[mask] = action_candidates[np.arange(action_candidates.shape[0]), best_action][mask]
    return predictions.astype(np.int64)


def _evaluate_delta(
    *,
    predictions: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    paired = _paired_ci(
        selected=predictions,
        baseline=baseline,
        answers=answers,
        seed=seed,
        samples=samples,
    )
    return {
        "accuracy": _accuracy(predictions, answers),
        "delta": float(paired["delta"]),
        "ci95_low": float(paired["ci95_low"]),
        "ci95_high": float(paired["ci95_high"]),
        "helps": int(paired["helps"]),
        "harms": int(paired["harms"]),
        "override_count": int(np.sum(predictions != baseline)),
    }


def _fit_router(
    *,
    action_features: np.ndarray,
    action_candidates: np.ndarray,
    hybrid: np.ndarray,
    answers: np.ndarray,
    fit_indices: np.ndarray,
    dev_indices: np.ndarray,
    ridges: Sequence[float],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    dev_indices = np.asarray(dev_indices, dtype=np.int64)
    flat_x = action_features[fit_indices].reshape(-1, action_features.shape[-1])
    flat_candidates = action_candidates[fit_indices].reshape(-1)
    repeated_answers = np.repeat(answers[fit_indices], action_candidates.shape[1])
    repeated_hybrid = np.repeat(hybrid[fit_indices], action_candidates.shape[1])
    target = (flat_candidates == repeated_answers).astype(np.float64) - (
        repeated_hybrid == repeated_answers
    ).astype(np.float64)
    config_rows: list[dict[str, Any]] = []
    noop_model = {
        "weights": np.zeros(action_features.shape[-1], dtype=np.float64).tolist(),
        "threshold": 1.0,
        "l2": 0.0,
        "selection": "no_op",
    }
    noop_eval = _evaluate_delta(
        predictions=hybrid[dev_indices],
        baseline=hybrid[dev_indices],
        answers=answers[dev_indices],
        seed=42160504,
        samples=max(200, min(bootstrap_samples, 1000)),
    )
    best_key: tuple[float, float, float, int, int, str] = (
        float(noop_eval["accuracy"]),
        float(noop_eval["delta"]),
        float(noop_eval["ci95_low"]),
        int(noop_eval["helps"]) - int(noop_eval["harms"]),
        -int(noop_eval["override_count"]),
        "no_op",
    )
    best_model: dict[str, Any] = noop_model
    for l2 in ridges:
        penalty = float(l2) * np.eye(flat_x.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        lhs = flat_x.T @ flat_x + penalty
        rhs = flat_x.T @ target
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs
        dev_scores = action_features[dev_indices] @ weights
        max_scores = np.max(dev_scores, axis=1)
        thresholds = sorted(set(float(value) for value in max_scores))
        noop_threshold = float(np.max(max_scores) + max(1e-9, abs(float(np.max(max_scores))) * 1e-6))
        thresholds.append(noop_threshold)
        if len(thresholds) > 100:
            finite = thresholds[:-1]
            thresholds = sorted({finite[int(round(q * (len(finite) - 1)))] for q in np.linspace(0.0, 1.0, 81)})
            thresholds.append(noop_threshold)
        for threshold in thresholds:
            model = {"weights": weights.tolist(), "threshold": float(threshold), "l2": float(l2)}
            predictions = _predict_router(
                action_features=action_features[dev_indices],
                action_candidates=action_candidates[dev_indices],
                hybrid=hybrid[dev_indices],
                model=model,
            )
            metrics = _evaluate_delta(
                predictions=predictions,
                baseline=hybrid[dev_indices],
                answers=answers[dev_indices],
                seed=42260504 + int(float(l2) * 1000) + len(config_rows),
                samples=max(200, min(bootstrap_samples, 1000)),
            )
            row = {
                "l2": float(l2),
                "threshold": float(threshold),
                "threshold_is_noop": bool(threshold == noop_threshold),
                "official_dev_accuracy": metrics["accuracy"],
                "official_dev_delta_vs_hybrid": metrics["delta"],
                "official_dev_ci95_low_vs_hybrid": metrics["ci95_low"],
                "official_dev_ci95_high_vs_hybrid": metrics["ci95_high"],
                "official_dev_helps_vs_hybrid": metrics["helps"],
                "official_dev_harms_vs_hybrid": metrics["harms"],
                "official_dev_override_count": metrics["override_count"],
            }
            config_rows.append(row)
            key = (
                float(row["official_dev_accuracy"]),
                float(row["official_dev_delta_vs_hybrid"]),
                float(row["official_dev_ci95_low_vs_hybrid"]),
                int(row["official_dev_helps_vs_hybrid"]) - int(row["official_dev_harms_vs_hybrid"]),
                -int(row["official_dev_override_count"]),
                json.dumps({"l2": float(l2), "threshold": float(threshold)}, sort_keys=True),
            )
            if key > best_key:
                best_key = key
                best_model = model
    return best_model, sorted(config_rows, key=lambda row: row["official_dev_accuracy"], reverse=True)


def _method_row(
    *,
    name: str,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    candidate_only: np.ndarray,
    target_only: np.ndarray,
    bootstrap_samples: int,
    raw_payload_bytes: int,
    framed_record_bytes: int,
    source_score_or_logit_vector_exposed: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = _answers(rows)
    vs_hybrid = _paired_ci(
        selected=predictions,
        baseline=fixed_hybrid,
        answers=answers,
        seed=52560504 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_candidate = _paired_ci(
        selected=predictions,
        baseline=candidate_only,
        answers=answers,
        seed=52560604 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    vs_target = _paired_ci(
        selected=predictions,
        baseline=target_only,
        answers=answers,
        seed=52560704 + sum(ord(ch) for ch in name),
        samples=bootstrap_samples,
    )
    return {
        "method": name,
        "eval_rows": len(rows),
        "accuracy": _accuracy(predictions, answers),
        "fixed_hybrid_accuracy": _accuracy(fixed_hybrid, answers),
        "candidate_only_accuracy": _accuracy(candidate_only, answers),
        "target_only_accuracy": _accuracy(target_only, answers),
        "delta_vs_fixed_hybrid": float(vs_hybrid["delta"]),
        "ci95_low_vs_fixed_hybrid": float(vs_hybrid["ci95_low"]),
        "ci95_high_vs_fixed_hybrid": float(vs_hybrid["ci95_high"]),
        "helps_vs_fixed_hybrid": int(vs_hybrid["helps"]),
        "harms_vs_fixed_hybrid": int(vs_hybrid["harms"]),
        "delta_vs_candidate_only": float(vs_candidate["delta"]),
        "ci95_low_vs_candidate_only": float(vs_candidate["ci95_low"]),
        "delta_vs_target_only": float(vs_target["delta"]),
        "ci95_low_vs_target_only": float(vs_target["ci95_low"]),
        "override_count_vs_fixed_hybrid": int(np.sum(predictions != fixed_hybrid)),
        "override_rate_vs_fixed_hybrid": float(np.mean(predictions != fixed_hybrid)),
        "raw_payload_bytes": int(raw_payload_bytes),
        "framed_record_bytes": int(framed_record_bytes),
        "source_private": not source_score_or_logit_vector_exposed,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "source_score_or_logit_vector_exposed": bool(source_score_or_logit_vector_exposed),
        "details": json.dumps(details or {}, sort_keys=True),
    }


def _source_top_predictions(rows: Sequence[dict[str, Any]], rank: int) -> np.ndarray:
    q_scores = np.asarray([row["qwen_source_scores"] for row in rows], dtype=np.float64)
    order = np.argsort(-q_scores, axis=1)
    return order[:, int(rank)].astype(np.int64)


def _pair_oracle(source_top1: np.ndarray, source_top2: np.ndarray, answers: np.ndarray) -> np.ndarray:
    return np.where(source_top1 == answers, source_top1, np.where(source_top2 == answers, source_top2, source_top1))


def _slice_rows(
    *,
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    fixed_hybrid: np.ndarray,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    answers = _answers(rows)
    starts = np.asarray([int(row["_slice_start"]) for row in rows], dtype=np.int64)
    out: list[dict[str, Any]] = []
    for start in sorted(set(starts.tolist())):
        mask = starts == start
        paired = _paired_ci(
            selected=predictions[mask],
            baseline=fixed_hybrid[mask],
            answers=answers[mask],
            seed=62660504 + int(start),
            samples=max(200, min(bootstrap_samples, 1000)),
        )
        out.append(
            {
                "slice_start": int(start),
                "eval_rows": int(np.sum(mask)),
                "method_accuracy": _accuracy(predictions[mask], answers[mask]),
                "fixed_hybrid_accuracy": _accuracy(fixed_hybrid[mask], answers[mask]),
                "delta_vs_fixed_hybrid": float(paired["delta"]),
                "ci95_low_vs_fixed_hybrid": float(paired["ci95_low"]),
                "helps_vs_fixed_hybrid": int(paired["helps"]),
                "harms_vs_fixed_hybrid": int(paired["harms"]),
            }
        )
    return out


def _corrupt_source_inputs(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    mean: np.ndarray,
    margin: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if condition == "matched":
        return qwen_scores, hybrid, mean, margin
    if condition == "source_row_shuffle":
        order = rng.permutation(len(hybrid))
        return qwen_scores[order], hybrid[order], mean[order], margin[order]
    if condition == "source_score_row_shuffle_before_encoding":
        order = rng.permutation(len(hybrid))
        return qwen_scores[order], hybrid, mean[order], margin[order]
    if condition == "candidate_roll_source":
        return np.roll(qwen_scores, shift=1, axis=1), (hybrid + 1) % 4, (mean + 1) % 4, margin
    if condition == "code_value_permutation":
        perm = rng.permutation(4)
        inverse = np.empty(4, dtype=np.int64)
        inverse[perm] = np.arange(4)
        return qwen_scores[:, perm], inverse[hybrid], inverse[mean], margin
    if condition == "target_derived_source_packet":
        return phi_scores, hybrid, np.argmax(phi_scores, axis=1).astype(np.int64), margin
    if condition == "random_same_byte_source":
        order = rng.integers(0, len(hybrid), size=len(hybrid))
        return qwen_scores[order], hybrid[order], mean[order], margin[order]
    raise ValueError(f"unknown source condition {condition!r}")


def _fit_logit_fusion(
    *,
    qwen_scores: np.ndarray,
    phi_scores: np.ndarray,
    hybrid: np.ndarray,
    answers: np.ndarray,
    dev_indices: np.ndarray,
) -> dict[str, Any]:
    qz = _z_scores(qwen_scores)
    pz = _z_scores(phi_scores)
    best: tuple[tuple[float, float, str], dict[str, Any]] | None = None
    for alpha in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        scores = pz[dev_indices] + float(alpha) * qz[dev_indices]
        predictions = np.argmax(scores, axis=1).astype(np.int64)
        key = (
            _accuracy(predictions, answers[dev_indices]),
            float(np.mean((predictions == answers[dev_indices]).astype(float) - (hybrid[dev_indices] == answers[dev_indices]).astype(float))),
            str(alpha),
        )
        model = {"alpha": float(alpha)}
        if best is None or key > best[0]:
            best = (key, model)
    if best is None:
        raise ValueError("no fusion model")
    return best[1]


def _predict_logit_fusion(qwen_scores: np.ndarray, phi_scores: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    scores = _z_scores(phi_scores) + float(model["alpha"]) * _z_scores(qwen_scores)
    return np.argmax(scores, axis=1).astype(np.int64)


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# HellaSwag Qwen-To-Phi Held-Out Uncertainty Router Gate",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- calibration rows: `{h['official_train_calibration_rows']}`",
        f"- eval rows: `{h['eval_rows']}`",
        f"- fixed hybrid accuracy: `{h['fixed_hybrid_accuracy']:.6f}`",
        f"- uncertainty-router accuracy: `{h['uncertainty_router_accuracy']:.6f}`",
        f"- uncertainty-router delta: `{h['uncertainty_router_delta_vs_fixed_hybrid']:.6f}`",
        f"- uncertainty-router CI95 low: `{h['uncertainty_router_ci95_low_vs_fixed_hybrid']:.6f}`",
        f"- overrides / helps / harms: `{h['uncertainty_router_override_count']} / {h['uncertainty_router_helps_vs_fixed_hybrid']} / {h['uncertainty_router_harms_vs_fixed_hybrid']}`",
        f"- source top1/top2 oracle accuracy: `{h['source_top1_or_top2_oracle_accuracy']:.6f}`",
        f"- best destructive: `{h['best_destructive_control_name']}` (`{h['best_destructive_control_accuracy']:.6f}`)",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
    ]
    _resolve(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT,
    train_path: pathlib.Path | str = DEFAULT_TRAIN_PATH,
    qwen_train_cache_dir: pathlib.Path | str = DEFAULT_QWEN_TRAIN_CACHE_DIR,
    phi_train_score_cache: pathlib.Path | str = DEFAULT_PHI_TRAIN_SCORE_CACHE,
    source_score_cache: pathlib.Path | str = DEFAULT_SOURCE_SCORE_CACHE,
    slices: tuple[dict[str, Any], ...] = denoise.DEFAULT_SLICES,
    sample_seeds: tuple[int, ...] = DEFAULT_SAMPLE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    component_ridges: tuple[float, ...] = DEFAULT_COMPONENT_RIDGES,
    ridges: Sequence[float] = DEFAULT_RIDGES,
    fit_rows_per_slice: int = denoise.FIT_ROWS_PER_SLICE,
    select_rows_per_slice: int = denoise.SELECT_ROWS_PER_SLICE,
    train_hidden_rows: int = 512,
    dev_fraction: float = 0.25,
    max_calibration_rows: int | None = None,
    target_model: pathlib.Path | str = DEFAULT_TARGET_MODEL,
    target_device: str = "mps",
    target_dtype: str = "float16",
    target_max_length: int = 256,
    target_normalization: str = "mean",
    target_prompt_mode: str = "continuation",
    local_files_only: bool = True,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    run_date: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_date = run_date or dt.date.today().isoformat()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = source_gate._build_qwen_oob_calibration(
        train_path=train_path,
        qwen_train_cache_dir=qwen_train_cache_dir,
        sample_seeds=sample_seeds,
        split_seeds=split_seeds,
        component_ridges=component_ridges,
        train_hidden_rows=train_hidden_rows,
        dev_fraction=dev_fraction,
    )
    calibration = receiver_gate._limit_calibration(calibration, max_calibration_rows)
    fit_indices, dev_indices = official._official_split_indices(
        len(calibration["rows"]),
        dev_fraction=dev_fraction,
        seed=4242,
    )
    calibration_arc_rows, calibration_row_meta = receiver_gate._arc_rows_for_calibration(
        train_path=train_path,
        calibration_rows=calibration["rows"],
    )
    phi_cache_path = _resolve(phi_train_score_cache)
    phi_cache_existed = phi_cache_path.exists()
    phi_scores, phi_predictions, phi_state, phi_sha = receiver_gate._load_or_build_phi_scores(
        rows=calibration_arc_rows,
        score_cache=phi_cache_path,
        target_model=target_model,
        target_device=target_device,
        target_dtype=target_dtype,
        target_max_length=target_max_length,
        target_normalization=target_normalization,
        target_prompt_mode=target_prompt_mode,
        local_files_only=local_files_only,
    )
    bins = _fit_bins(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_margin=calibration["margin"],
        fit_indices=fit_indices,
    )
    train_features, train_actions, _ = _stack_router_features(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
        bins=bins,
    )
    model, config_rows = _fit_router(
        action_features=train_features,
        action_candidates=train_actions,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=ridges,
        bootstrap_samples=bootstrap_samples,
    )
    label_rng = np.random.default_rng(20260504)
    permuted_answers = calibration["answers"][label_rng.permutation(len(calibration["answers"]))]
    label_model, _ = _fit_router(
        action_features=train_features,
        action_candidates=train_actions,
        hybrid=calibration["hybrid"],
        answers=permuted_answers,
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        ridges=ridges,
        bootstrap_samples=max(200, min(bootstrap_samples, 1000)),
    )
    fusion_model = _fit_logit_fusion(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        dev_indices=dev_indices,
    )

    rows, slice_metadata = denoise._load_rows(
        slices=slices,
        fit_rows_per_slice=fit_rows_per_slice,
        select_rows_per_slice=select_rows_per_slice,
    )
    source_score_metadata = oracle._load_source_scores(rows, source_score_cache)
    eval_rows = [row for row in rows if row["_split"] == "eval"]
    eval_scores = np.asarray([row["qwen_source_scores"] for row in eval_rows], dtype=np.float64)
    eval_phi_scores = np.asarray([row["phi_target_scores"] for row in eval_rows], dtype=np.float64)
    eval_hybrid = _field_array(eval_rows, "qwen_hybrid_prediction")
    eval_mean = _field_array(eval_rows, "selected_prediction")
    eval_margin = np.asarray([float(row.get("selected_margin", 0.0)) for row in eval_rows], dtype=np.float64)
    eval_features, eval_actions, _ = _stack_router_features(
        qwen_scores=eval_scores,
        phi_scores=eval_phi_scores,
        hybrid=eval_hybrid,
        qwen_mean=eval_mean,
        qwen_margin=eval_margin,
        bins=bins,
    )
    fixed_hybrid = eval_hybrid
    candidate_only = eval_mean
    target_only = _field_array(eval_rows, "phi_target_prediction")
    answers = _answers(eval_rows)
    source_top1 = _source_top_predictions(eval_rows, 0)
    source_top2 = _source_top_predictions(eval_rows, 1)
    source_pair_oracle = _pair_oracle(source_top1, source_top2, answers)
    selected_predictions = _predict_router(eval_features, eval_actions, fixed_hybrid, model)
    label_predictions = _predict_router(eval_features, eval_actions, fixed_hybrid, label_model)
    fusion_predictions = _predict_logit_fusion(eval_scores, eval_phi_scores, fusion_model)

    controls: dict[str, tuple[np.ndarray, int, int, bool, dict[str, Any]]] = {
        "heldout_uncertainty_router_packet": (
            selected_predictions,
            3,
            6,
            False,
            {"model": {key: value for key, value in model.items() if key != "weights"}},
        ),
        "fixed_hybrid_vote_on_score_agreement": (fixed_hybrid, 1, 4, False, {}),
        "qwen_candidate_only": (candidate_only, 1, 4, False, {}),
        "phi_target_only": (target_only, 0, 0, False, {}),
        "source_top1_label_control": (source_top1, 1, 4, False, {"source_rank": 1}),
        "source_top2_label_control": (source_top2, 1, 4, False, {"source_rank": 2}),
        "source_top1_or_top2_oracle_diagnostic": (
            source_pair_oracle,
            0,
            0,
            False,
            {"oracle": True, "not_promotable": True},
        ),
        "official_train_label_permutation_router_control": (
            label_predictions,
            3,
            6,
            False,
            {"condition": "official_train_label_permutation"},
        ),
        "raw_source_score_logit_fusion_control": (
            fusion_predictions,
            16,
            19,
            True,
            {"alpha": fusion_model["alpha"], "source_score_or_logit_vector_exposed": True},
        ),
    }
    for condition in SOURCE_CONDITIONS:
        c_scores, c_hybrid, c_mean, c_margin = _corrupt_source_inputs(
            qwen_scores=eval_scores,
            phi_scores=eval_phi_scores,
            hybrid=eval_hybrid,
            mean=eval_mean,
            margin=eval_margin,
            condition=condition,
            seed=20260504 + sum(ord(ch) for ch in condition),
        )
        c_features, c_actions, _ = _stack_router_features(
            qwen_scores=c_scores,
            phi_scores=eval_phi_scores,
            hybrid=c_hybrid,
            qwen_mean=c_mean,
            qwen_margin=c_margin,
            bins=bins,
        )
        corrupted = _predict_router(c_features, c_actions, c_hybrid, model)
        controls[f"{condition}_router_control"] = (
            corrupted,
            3,
            6,
            False,
            {"condition": condition},
        )

    method_rows = [
        _method_row(
            name=name,
            rows=eval_rows,
            predictions=predictions,
            fixed_hybrid=fixed_hybrid,
            candidate_only=candidate_only,
            target_only=target_only,
            bootstrap_samples=bootstrap_samples,
            raw_payload_bytes=raw_bytes,
            framed_record_bytes=framed_bytes,
            source_score_or_logit_vector_exposed=score_exposed,
            details=details,
        )
        for name, (predictions, raw_bytes, framed_bytes, score_exposed, details) in controls.items()
    ]

    eval_diag_best: tuple[tuple[float, float, int, str], np.ndarray] | None = None
    eval_scores_for_model = eval_features @ np.asarray(model["weights"], dtype=np.float64)
    eval_max_scores = np.max(eval_scores_for_model, axis=1)
    for threshold in sorted(set(float(value) for value in eval_max_scores)):
        diag_model = dict(model, threshold=float(threshold))
        diag_predictions = _predict_router(eval_features, eval_actions, fixed_hybrid, diag_model)
        overrides = int(np.sum(diag_predictions != fixed_hybrid))
        if overrides == 0:
            continue
        key = (
            _accuracy(diag_predictions, answers),
            float(np.mean((diag_predictions == answers).astype(float) - (fixed_hybrid == answers).astype(float))),
            -overrides,
            str(threshold),
        )
        if eval_diag_best is None or key > eval_diag_best[0]:
            eval_diag_best = (key, diag_predictions)
    if eval_diag_best is not None:
        method_rows.append(
            _method_row(
                name="eval_label_best_threshold_router_diagnostic",
                rows=eval_rows,
                predictions=eval_diag_best[1],
                fixed_hybrid=fixed_hybrid,
                candidate_only=candidate_only,
                target_only=target_only,
                bootstrap_samples=bootstrap_samples,
                raw_payload_bytes=3,
                framed_record_bytes=6,
                details={"not_promotable": True, "eval_label_selected": True},
            )
        )

    method_rows = sorted(method_rows, key=lambda row: row["accuracy"], reverse=True)
    method_row = next(row for row in method_rows if row["method"] == "heldout_uncertainty_router_packet")
    destructive_rows = [
        row
        for row in method_rows
        if row["method"].endswith("_router_control")
        or row["method"] in {"official_train_label_permutation_router_control"}
    ]
    best_destructive = max(destructive_rows, key=lambda row: row["accuracy"])
    shortcut_names = (
        "source_top1_label_control",
        "source_top2_label_control",
        "raw_source_score_logit_fusion_control",
    )
    best_shortcut = max((row for row in method_rows if row["method"] in shortcut_names), key=lambda row: row["accuracy"])
    slice_rows = _slice_rows(
        rows=eval_rows,
        predictions=selected_predictions,
        fixed_hybrid=fixed_hybrid,
        bootstrap_samples=bootstrap_samples,
    )
    train_content_ids = {row.content_id for row in calibration_arc_rows}
    eval_content_ids = {str(row.get("content_id", row["row_id"])) for row in eval_rows}
    pass_gate = bool(
        method_row["delta_vs_fixed_hybrid"] >= 0.005
        and method_row["ci95_low_vs_fixed_hybrid"] > 0.0
        and method_row["ci95_low_vs_candidate_only"] > 0.0
        and method_row["helps_vs_fixed_hybrid"] > method_row["harms_vs_fixed_hybrid"]
        and all(row["delta_vs_fixed_hybrid"] >= 0.0 for row in slice_rows)
        and method_row["accuracy"] > best_destructive["accuracy"]
        and method_row["accuracy"] > best_shortcut["accuracy"]
        and len(train_content_ids & eval_content_ids) == 0
    )
    selected_dev = next(
        (
            row
            for row in config_rows
            if row["l2"] == model["l2"]
            and row["threshold"] == model["threshold"]
        ),
        {},
    )
    prediction_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(eval_rows):
        for name, (predictions, _, _, _, _) in controls.items():
            pred = int(predictions[row_index])
            prediction_rows.append(
                {
                    "row_id": row["row_id"],
                    "content_id": row.get("content_id", ""),
                    "method": name,
                    "answer_index": int(row["answer_index"]),
                    "prediction_index": pred,
                    "correct": bool(pred == int(row["answer_index"])),
                    "fixed_hybrid_prediction": int(fixed_hybrid[row_index]),
                    "override_fixed_hybrid": bool(pred != int(fixed_hybrid[row_index])),
                }
            )
    headline = {
        "official_train_calibration_rows": int(len(calibration["rows"])),
        "official_train_fit_rows": int(len(fit_indices)),
        "official_train_dev_rows": int(len(dev_indices)),
        "official_train_duplicate_rows_dropped": int(calibration["duplicate_row_count"]),
        "official_train_oob_overlap_rows_dropped": int(calibration["oob_overlap_drop_count"]),
        "official_train_content_overlap_with_eval": int(len(train_content_ids & eval_content_ids)),
        "official_train_qwen_hybrid_accuracy": _accuracy(calibration["hybrid"], calibration["answers"]),
        "official_train_phi_target_accuracy": _accuracy(phi_predictions, calibration["answers"]),
        "selected_l2": float(model["l2"]),
        "selected_threshold": float(model["threshold"]),
        "official_dev_selected_accuracy": selected_dev.get(
            "official_dev_accuracy",
            _accuracy(calibration["hybrid"][dev_indices], calibration["answers"][dev_indices]),
        ),
        "official_dev_selected_delta_vs_hybrid": selected_dev.get("official_dev_delta_vs_hybrid", 0.0),
        "official_dev_selected_ci95_low_vs_hybrid": selected_dev.get("official_dev_ci95_low_vs_hybrid", 0.0),
        "eval_rows": int(len(eval_rows)),
        "fixed_hybrid_accuracy": next(row for row in method_rows if row["method"] == "fixed_hybrid_vote_on_score_agreement")[
            "accuracy"
        ],
        "candidate_only_accuracy": next(row for row in method_rows if row["method"] == "qwen_candidate_only")[
            "accuracy"
        ],
        "phi_target_accuracy": next(row for row in method_rows if row["method"] == "phi_target_only")[
            "accuracy"
        ],
        "source_top1_label_accuracy": next(row for row in method_rows if row["method"] == "source_top1_label_control")[
            "accuracy"
        ],
        "source_top2_label_accuracy": next(row for row in method_rows if row["method"] == "source_top2_label_control")[
            "accuracy"
        ],
        "source_top1_or_top2_oracle_accuracy": next(
            row for row in method_rows if row["method"] == "source_top1_or_top2_oracle_diagnostic"
        )["accuracy"],
        "uncertainty_router_accuracy": method_row["accuracy"],
        "uncertainty_router_delta_vs_fixed_hybrid": method_row["delta_vs_fixed_hybrid"],
        "uncertainty_router_ci95_low_vs_fixed_hybrid": method_row["ci95_low_vs_fixed_hybrid"],
        "uncertainty_router_helps_vs_fixed_hybrid": method_row["helps_vs_fixed_hybrid"],
        "uncertainty_router_harms_vs_fixed_hybrid": method_row["harms_vs_fixed_hybrid"],
        "uncertainty_router_override_count": method_row["override_count_vs_fixed_hybrid"],
        "best_destructive_control_name": best_destructive["method"],
        "best_destructive_control_accuracy": best_destructive["accuracy"],
        "best_shortcut_control_name": best_shortcut["method"],
        "best_shortcut_control_accuracy": best_shortcut["accuracy"],
        "raw_source_score_logit_fusion_alpha": float(fusion_model["alpha"]),
        "phi_train_score_cache_hit": bool(phi_cache_existed),
        "raw_payload_bytes": 3,
        "framed_record_bytes": 6,
        "native_systems_claim_allowed": False,
    }
    payload = {
        "gate": "source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Pass only if the held-out router beats fixed Qwen hybrid by at least 0.005 with positive "
            "paired CI, beats candidate-only with positive paired CI, helps more than harms, is nonnegative "
            "on both eval slices, beats destructive controls, beats source-choice/logit-fusion shortcuts, "
            "and has zero official-train/eval content overlap."
        ),
        "headline": headline,
        "packet_contract": {
            "receiver_visible_payload": (
                "Qwen hybrid/top1/top2/mean candidate IDs plus quantized source-side uncertainty bins; "
                "Phi-local scores are decoder side information."
            ),
            "raw_payload_bytes": 3,
            "framed_record_bytes": 6,
            "source_packet_fields": [
                "hybrid_candidate_id",
                "qwen_top1_candidate_id",
                "qwen_top2_candidate_id",
                "qwen_mean_candidate_id",
                "qwen_margin_bin",
                "qwen_entropy_bin",
                "qwen_top1_vs_hybrid_gap_bin",
                "qwen_selected_margin_bin",
            ],
            "receiver_local_fields": [
                "phi_scores",
                "phi_top1_candidate_id",
                "phi_margin_bin",
                "phi_entropy_bin",
                "phi_top1_vs_hybrid_gap_bin",
            ],
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_or_logits_transmitted": False,
            "raw_qwen_scores_used_only_for_source_side_quantized_packet": True,
        },
        "calibration_row_metadata": calibration_row_meta,
        "quantization_bins": bins,
        "source_score_metadata": source_score_metadata,
        "slice_metadata": slice_metadata,
        "sample_cache_rows": calibration["sample_cache_rows"],
        "component_rows": calibration["component_rows"],
        "method_rows": method_rows,
        "config_rows": config_rows,
        "slice_rows": slice_rows,
        "systems_packet_sideband": {
            "raw_payload_bytes_per_request": 3,
            "framed_record_bytes_per_request": 6,
            "fit_and_eval_wall_time_s": float(time.perf_counter() - started),
            "phi_train_score_cache_hit": bool(phi_cache_existed),
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_score_vector_exposed": False,
            "native_gpu_claims_allowed": False,
        },
        "inputs": {
            "train_path": _display_path(train_path),
            "train_sha256": official._sha256_file(train_path),
            "qwen_train_cache_dir": _display_path(qwen_train_cache_dir),
            "phi_train_score_cache": _display_path(phi_cache_path),
            "phi_train_score_cache_sha256": phi_sha,
            "phi_train_score_model": phi_state,
            "source_score_cache": _display_path(source_score_cache),
            "source_score_cache_sha256": denoise._sha256_file(source_score_cache),
        },
        "interpretation": (
            "This gate is the strongest bounded test of the shallow uncertainty-router branch: it uses "
            "official-train calibration and a smooth ridge scorer, but the receiver-visible source side "
            "remains quantized candidate IDs and uncertainty bins rather than raw Qwen scores."
        ),
        "lay_explanation": (
            "Qwen sends a tiny coarse message saying which answers it thinks are most plausible and how "
            "confident it is. Phi uses its own scores plus a rule learned on training examples to decide "
            "whether to keep Qwen's safe answer, use Qwen's backup, or trust Phi's own favorite."
        ),
    }
    _write_json(output_dir / "hellaswag_qwen_to_phi_heldout_uncertainty_router_gate.json", payload)
    _write_csv(output_dir / "method_rows.csv", method_rows)
    _write_csv(output_dir / "config_rows.csv", config_rows)
    _write_csv(output_dir / "slice_rows.csv", slice_rows)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_markdown(output_dir / "hellaswag_qwen_to_phi_heldout_uncertainty_router_gate.md", payload)
    _write_json(
        output_dir / "manifest.json",
        {
            "gate": payload["gate"],
            "date": run_date,
            "outputs": [
                "hellaswag_qwen_to_phi_heldout_uncertainty_router_gate.json",
                "hellaswag_qwen_to_phi_heldout_uncertainty_router_gate.md",
                "method_rows.csv",
                "config_rows.csv",
                "slice_rows.csv",
                "predictions.jsonl",
            ],
            "headline": headline,
            "inputs": payload["inputs"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--qwen-train-cache-dir", type=pathlib.Path, default=DEFAULT_QWEN_TRAIN_CACHE_DIR)
    parser.add_argument("--phi-train-score-cache", type=pathlib.Path, default=DEFAULT_PHI_TRAIN_SCORE_CACHE)
    parser.add_argument("--source-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_SCORE_CACHE)
    parser.add_argument("--sample-seeds", type=official._parse_int_tuple, default=DEFAULT_SAMPLE_SEEDS)
    parser.add_argument("--split-seeds", type=official._parse_int_tuple, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--component-ridges", type=official._parse_float_tuple, default=DEFAULT_COMPONENT_RIDGES)
    parser.add_argument("--ridges", type=official._parse_float_tuple, default=DEFAULT_RIDGES)
    parser.add_argument("--fit-rows-per-slice", type=int, default=denoise.FIT_ROWS_PER_SLICE)
    parser.add_argument("--select-rows-per-slice", type=int, default=denoise.SELECT_ROWS_PER_SLICE)
    parser.add_argument("--train-hidden-rows", type=int, default=512)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--max-calibration-rows", type=int, default=None)
    parser.add_argument("--target-model", type=pathlib.Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-device", type=str, default="mps")
    parser.add_argument("--target-dtype", type=str, default="float16")
    parser.add_argument("--target-max-length", type=int, default=256)
    parser.add_argument("--target-normalization", type=str, default="mean")
    parser.add_argument("--target-prompt-mode", type=str, default="continuation")
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        qwen_train_cache_dir=args.qwen_train_cache_dir,
        phi_train_score_cache=args.phi_train_score_cache,
        source_score_cache=args.source_score_cache,
        sample_seeds=args.sample_seeds,
        split_seeds=args.split_seeds,
        component_ridges=args.component_ridges,
        ridges=args.ridges,
        fit_rows_per_slice=args.fit_rows_per_slice,
        select_rows_per_slice=args.select_rows_per_slice,
        train_hidden_rows=args.train_hidden_rows,
        dev_fraction=args.dev_fraction,
        max_calibration_rows=args.max_calibration_rows,
        target_model=args.target_model,
        target_device=args.target_device,
        target_dtype=args.target_dtype,
        target_max_length=args.target_max_length,
        target_normalization=args.target_normalization,
        target_prompt_mode=args.target_prompt_mode,
        local_files_only=not args.no_local_files_only,
        bootstrap_samples=args.bootstrap_samples,
        run_date=args.run_date,
    )
    h = payload["headline"]
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "fixed_hybrid_accuracy": h["fixed_hybrid_accuracy"],
                "uncertainty_router_accuracy": h["uncertainty_router_accuracy"],
                "uncertainty_router_delta_vs_fixed_hybrid": h[
                    "uncertainty_router_delta_vs_fixed_hybrid"
                ],
                "uncertainty_router_ci95_low_vs_fixed_hybrid": h[
                    "uncertainty_router_ci95_low_vs_fixed_hybrid"
                ],
                "uncertainty_router_helps_vs_fixed_hybrid": h[
                    "uncertainty_router_helps_vs_fixed_hybrid"
                ],
                "uncertainty_router_harms_vs_fixed_hybrid": h[
                    "uncertainty_router_harms_vs_fixed_hybrid"
                ],
                "source_top1_or_top2_oracle_accuracy": h[
                    "source_top1_or_top2_oracle_accuracy"
                ],
                "best_destructive_control_name": h["best_destructive_control_name"],
                "best_destructive_control_accuracy": h["best_destructive_control_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
