from __future__ import annotations

"""Strict ARC-Challenge behavior-atom decoder Sparse Resonance Packet gate.

This branch keeps the source-private hidden-atom transport from the previous
gate, but replaces unsupervised PCA atoms with train-fit behavior-supervised
directions.  The packet is still computed only from answer-key-forbidden source
hidden innovations at evaluation time; labels and target behavior are used only
to fit the packet basis and receiver on the training disagreement slice.
"""

import argparse
import datetime as dt
import gc
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_behavior_residual_packet_gate as behavior_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_confidence_ecoc_packet_gate as ecoc_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_hidden_atom_decoder_gate as hidden_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as soft_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_"
    "tinyllama_to_qwen3_disagreement"
)
DEFAULT_VALIDATION = soft_gate.DEFAULT_VALIDATION
DEFAULT_TEST = soft_gate.DEFAULT_TEST
DEFAULT_SOURCE_FAMILY_GATE_DIR = soft_gate.DEFAULT_SOURCE_FAMILY_GATE_DIR
DEFAULT_TINY_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_TINY_VALIDATION_SCORE_CACHE
DEFAULT_TINY_TEST_SCORE_CACHE = soft_gate.DEFAULT_TINY_TEST_SCORE_CACHE
DEFAULT_QWEN_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_QWEN_VALIDATION_SCORE_CACHE
DEFAULT_QWEN_TEST_SCORE_CACHE = soft_gate.DEFAULT_QWEN_TEST_SCORE_CACHE
DEFAULT_TINY_MODEL = behavior_gate.DEFAULT_TINY_MODEL
DEFAULT_QWEN3_MODEL = behavior_gate.DEFAULT_QWEN3_MODEL

MATCHED_CONDITION = "matched_behavior_atom_decoder_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_decoder_only",
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "same_source_choice_row_shuffle",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll",
    "candidate_derangement",
    "packet_only_source_index",
    "source_rank_control",
    "source_score_control",
    "source_score_quantized_control",
    "same_byte_visible_text",
    "qwen_substituted_packet",
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
STRICT_REQUIRED_CONTROLS = CONTROL_CONDITIONS
EVENT_GATE_CORRUPTION_CONDITIONS = (
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "same_source_choice_row_shuffle",
    "atom_shuffle",
    "coefficient_shuffle",
    "top_atom_knockout",
    "candidate_roll",
    "candidate_derangement",
)


@dataclass(frozen=True)
class RidgeNoInterceptScalarMap:
    x_scale: np.ndarray
    weights: np.ndarray
    ridge: float
    fit_mse: float
    fit_r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        scaled = values / self.x_scale
        return scaled @ self.weights


@dataclass(frozen=True)
class EventGateRule:
    residual_weight: float
    threshold: float
    event_model: behavior_gate.RidgeScalarMap
    require_prediction_change: bool
    metadata: dict[str, Any]


def _behavior_target_matrix(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
) -> np.ndarray:
    """Build train targets for source atom directions from target residual need."""

    residual = behavior_gate._candidate_targets(rows, target_scores).reshape(-1, 1)
    target = behavior_gate._target_score_features(rows, target_scores)
    score = target[:, [0]]
    centered = target[:, [1]]
    prob = target[:, [2]]
    inv_rank = target[:, [3]]
    margin = target[:, [4]]
    return np.concatenate(
        [
            residual,
            residual * score,
            residual * centered,
            residual * prob,
            residual * inv_rank,
            residual * margin,
            residual * np.square(centered),
            residual * np.square(prob),
            residual * np.square(margin),
            residual * np.sign(centered),
        ],
        axis=1,
    )


def _fit_behavior_atom_packet_from_features(
    source_features: np.ndarray,
    behavior_targets: np.ndarray,
    *,
    fit_flat_indices: np.ndarray,
    rank: int,
    top_k: int,
    quant_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_features, dtype=np.float64)
    targets = np.asarray(behavior_targets, dtype=np.float64)
    fit = np.asarray(fit_flat_indices, dtype=np.int64)
    if source.ndim != 2:
        raise ValueError("source_features must be rank-2")
    if targets.ndim != 2:
        raise ValueError("behavior_targets must be rank-2")
    if source.shape[0] != targets.shape[0]:
        raise ValueError("source/target row mismatch")
    if fit.size == 0:
        raise ValueError("fit_flat_indices must not be empty")
    if rank < 1:
        raise ValueError("packet rank must be at least 1")

    x_mean = source[fit].mean(axis=0, keepdims=True)
    x_std = source[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    standardized_source = (source - x_mean) / x_std
    x_fit = standardized_source[fit]

    y_fit_raw = targets[fit]
    y_mean = y_fit_raw.mean(axis=0, keepdims=True)
    y_std = y_fit_raw.std(axis=0, keepdims=True).clip(min=1e-6)
    y_fit = (y_fit_raw - y_mean) / y_std

    cross_cov = x_fit.T @ y_fit / float(max(fit.size, 1))
    u, singular_values, _ = np.linalg.svd(cross_cov, full_matrices=False)
    actual_rank = int(min(rank, u.shape[1]))
    if actual_rank < 1:
        raise ValueError("could not fit a non-empty behavior atom basis")
    directions = u[:, :actual_rank]
    coeffs = standardized_source @ directions
    sparse_packet, quant_meta = preflight._sparse_topk_quantized_coordinates(
        coeffs,
        fit_flat_indices=fit,
        top_k=top_k,
        quant_bits=quant_bits,
    )

    target_map = hidden_gate._fit_ridge_matrix_map(
        sparse_packet[fit],
        y_fit_raw,
        fit_indices=np.arange(fit.size, dtype=np.int64),
        ridge=1.0,
    )
    retained = np.square(singular_values[:actual_rank])
    total = float(np.square(singular_values).sum())
    return sparse_packet, {
        "kind": "train_fit_behavior_supervised_hidden_atom_packet_coordinates",
        "fit_candidate_count": int(fit.size),
        "source_feature_dim": int(source.shape[1]),
        "behavior_target_dim": int(targets.shape[1]),
        "packet_rank": int(actual_rank),
        "requested_packet_rank": int(rank),
        "behavior_cross_cov_energy_ratio": float(retained.sum() / total) if total > 1e-12 else 0.0,
        "behavior_singular_value_max": float(singular_values[0]) if singular_values.size else 0.0,
        "behavior_singular_value_min_retained": float(singular_values[actual_rank - 1])
        if singular_values.size
        else 0.0,
        "source_std_min": float(x_std.min()),
        "source_std_max": float(x_std.max()),
        "packet_to_behavior_fit_mse": float(target_map.fit_mse),
        "packet_to_behavior_fit_r2": float(target_map.fit_r2),
        **quant_meta,
    }


def _fit_source_packet(
    rows: Sequence[arc_gate.ArcRow],
    *,
    target_scores: Sequence[Sequence[float]],
    source_model: str,
    source_device: str,
    dtype: str,
    source_max_length: int,
    source_hidden_layer: int,
    source_feature_dim: int,
    fit_candidate_indices: np.ndarray,
    ridge: float,
    packet_rank: int,
    packet_top_k: int,
    packet_bits: int,
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    flat_hidden, hidden_meta = preflight._hf_choice_hidden_features(
        list(rows),
        model_path=source_model,
        device=source_device,
        dtype=dtype,
        max_length=source_max_length,
        local_files_only=local_files_only,
        hidden_layer=source_hidden_layer,
    )
    public_flat, public_meta = preflight._public_candidate_hashed_features(list(rows), feature_dim=source_feature_dim)
    innovation, innovation_meta = preflight._public_candidate_innovation_features(
        flat_hidden,
        public_flat,
        fit_flat_indices=fit_candidate_indices,
        ridge=ridge,
    )
    behavior_targets = _behavior_target_matrix(rows, target_scores)
    sparse_packet, sparse_meta = _fit_behavior_atom_packet_from_features(
        innovation,
        behavior_targets,
        fit_flat_indices=fit_candidate_indices,
        rank=packet_rank,
        top_k=packet_top_k,
        quant_bits=packet_bits,
    )
    return sparse_packet, {
        "source_hidden": hidden_meta,
        "public": public_meta,
        "source_public_innovation": innovation_meta,
        "sparse_packet": sparse_meta,
    }


def _innovation_decoder_features(target_features: np.ndarray, packet_features: np.ndarray) -> np.ndarray:
    target = np.asarray(target_features, dtype=np.float64)
    packet = np.asarray(packet_features, dtype=np.float64)
    if target.ndim != 2 or packet.ndim != 2:
        raise ValueError("target_features and packet_features must be rank-2")
    if target.shape[0] != packet.shape[0]:
        raise ValueError("target/packet candidate mismatch")
    if target.shape[1] < 5:
        raise ValueError("target_features must include score, centered, prob, rank, and margin columns")
    centered = target[:, [1]]
    probability = target[:, [2]]
    margin = target[:, [4]]
    return np.concatenate(
        [
            packet,
            np.abs(packet),
            packet * centered,
            packet * probability,
            packet * margin,
            packet * np.square(centered),
        ],
        axis=1,
    )


def _fit_ridge_no_intercept_scalar_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_indices: np.ndarray,
    ridge: float,
) -> RidgeNoInterceptScalarMap:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("features must be rank-2 and targets rank-1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature/target row mismatch")
    if fit.size == 0:
        raise ValueError("fit_indices must not be empty")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    x_scale = x[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    scaled = x / x_scale
    x_fit = scaled[fit]
    y_fit = y[fit]
    xtx = x_fit.T @ x_fit
    if ridge > 0.0:
        xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x_fit.T @ y_fit
    weights = np.linalg.solve(xtx, xty)
    pred_fit = x_fit @ weights
    mse = float(np.mean(np.square(y_fit - pred_fit)))
    baseline = float(np.mean(np.square(y_fit)))
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - mse / baseline
    return RidgeNoInterceptScalarMap(
        x_scale=x_scale,
        weights=weights,
        ridge=float(ridge),
        fit_mse=mse,
        fit_r2=float(r2),
    )


def _same_choice_shuffle_index(
    *,
    row_index: int,
    eval_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
    source_selected: Sequence[int],
) -> int:
    row = rows[row_index]
    same_source = [
        index
        for index in eval_indices
        if index != row_index
        and len(rows[index].choices) == len(row.choices)
        and int(source_selected[index]) == int(source_selected[row_index])
    ]
    if same_source:
        return same_source[0]
    same_shape = [index for index in eval_indices if index != row_index and len(rows[index].choices) == len(row.choices)]
    if same_shape:
        return same_shape[0]
    return row_index


def _same_shape_shuffle_index(
    *,
    row_index: int,
    eval_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
) -> int:
    row = rows[row_index]
    same_shape = [index for index in eval_indices if index != row_index and len(rows[index].choices) == len(row.choices)]
    if same_shape:
        return same_shape[0]
    other = [index for index in eval_indices if index != row_index]
    if other:
        return other[0]
    return row_index


def _decode_one_row(
    row: arc_gate.ArcRow,
    *,
    rows: Sequence[arc_gate.ArcRow],
    row_index: int,
    target_features: np.ndarray,
    packet: np.ndarray,
    decoder: Any,
    decoder_mode: str,
) -> np.ndarray:
    start, end = hidden_gate._row_offsets(rows)[row_index]
    return _decode_packet_residual_rows(
        [row],
        target_features=target_features[start:end],
        packet_features=np.asarray(packet, dtype=np.float64),
        decoder=decoder,
        decoder_mode=decoder_mode,
    )[0]


def _decode_packet_residual_rows(
    rows: Sequence[arc_gate.ArcRow],
    *,
    target_features: np.ndarray,
    packet_features: np.ndarray,
    decoder: Any,
    decoder_mode: str,
) -> list[np.ndarray]:
    if decoder_mode == "target_conditioned":
        return hidden_gate._decode_residual_rows(
            rows,
            target_features=target_features,
            packet_features=packet_features,
            decoder=decoder,
        )
    if decoder_mode == "packet_innovation":
        flat_pred = decoder.predict(_innovation_decoder_features(target_features, packet_features))
        return hidden_gate._rows_from_candidate_values(rows, flat_pred)
    raise ValueError(f"unknown decoder_mode: {decoder_mode}")


def _subtract_row_baseline(residuals: Sequence[np.ndarray], baselines: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [
        np.asarray(residual, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
        for residual, baseline in zip(residuals, baselines, strict=True)
    ]


def _safe_l2(values: np.ndarray) -> float:
    return float(np.sqrt(float(np.square(values).sum())))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(left.mean()) if left.size else left
    right_centered = right - float(right.mean()) if right.size else right
    denom = _safe_l2(left_centered) * _safe_l2(right_centered)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def _event_gate_features(
    target_scores: Sequence[float],
    residual: Sequence[float],
    packet: np.ndarray,
    *,
    residual_weight: float,
) -> np.ndarray:
    target = np.asarray(target_scores, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    packet_values = np.asarray(packet, dtype=np.float64)
    if target.ndim != 1 or correction.ndim != 1:
        raise ValueError("target_scores and residual must be rank-1")
    if target.shape != correction.shape:
        raise ValueError("target/residual shape mismatch")
    weighted = target + float(residual_weight) * correction
    target_pred = behavior_gate._prediction(target)
    fused_pred = behavior_gate._prediction(weighted)
    target_probs = behavior_gate._softmax(target)
    fused_probs = behavior_gate._softmax(weighted)
    flat_packet = packet_values.reshape(-1) if packet_values.size else np.zeros(1, dtype=np.float64)
    target_margin = hidden_gate._target_margin(target)
    fused_margin = hidden_gate._target_margin(weighted)
    correction_abs = np.abs(correction)
    packet_abs = np.abs(flat_packet)
    changed = 1.0 if fused_pred != target_pred else 0.0
    return np.asarray(
        [
            1.0,
            float(target.size),
            float(target_margin),
            float(behavior_gate._entropy(target)),
            float(target_probs[target_pred]) if target_probs.size else 0.0,
            float(fused_margin),
            float(behavior_gate._entropy(weighted)),
            float(fused_probs[fused_pred]) if fused_probs.size else 0.0,
            changed,
            float(correction_abs.sum()),
            _safe_l2(correction),
            float(correction_abs.max()) if correction_abs.size else 0.0,
            float(correction.std()) if correction.size else 0.0,
            float(correction[target_pred]) if correction.size else 0.0,
            float(correction[fused_pred]) if correction.size else 0.0,
            float(correction[fused_pred] - correction[target_pred]) if correction.size else 0.0,
            _safe_corr(target, correction),
            float(packet_abs.sum()),
            _safe_l2(flat_packet),
            float(packet_abs.max()) if packet_abs.size else 0.0,
            float(np.count_nonzero(flat_packet) / max(flat_packet.size, 1)),
            float(flat_packet.mean()) if flat_packet.size else 0.0,
            float(packet_abs.mean()) if packet_abs.size else 0.0,
        ],
        dtype=np.float64,
    )


def _event_triggered_fused_scores(
    target_scores: Sequence[float],
    residual: Sequence[float],
    packet: np.ndarray,
    *,
    rule: EventGateRule,
) -> tuple[list[float], bool, float]:
    target = np.asarray(target_scores, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    fused = target + float(rule.residual_weight) * correction
    event_features = _event_gate_features(
        target,
        correction,
        packet,
        residual_weight=float(rule.residual_weight),
    ).reshape(1, -1)
    event_score = float(rule.event_model.predict(event_features)[0])
    fires = event_score >= float(rule.threshold)
    if bool(rule.require_prediction_change):
        fires = fires and behavior_gate._prediction(fused) != behavior_gate._prediction(target)
    if not fires:
        return [float(score) for score in target], False, event_score
    return [float(score) for score in fused], True, event_score


def _fuse_with_gate(
    target_scores: Sequence[float],
    residual: Sequence[float],
    packet: np.ndarray,
    *,
    gate_rule: dict[str, Any] | EventGateRule,
    gate_mode: str,
) -> tuple[list[float], bool, float | None]:
    if gate_mode == "residual_threshold":
        fused, fired = hidden_gate._fused_scores(target_scores, residual, rule=gate_rule)  # type: ignore[arg-type]
        return fused, fired, None
    if gate_mode == "event_triggered":
        fused, fired, event_score = _event_triggered_fused_scores(
            target_scores,
            residual,
            packet,
            rule=gate_rule,  # type: ignore[arg-type]
        )
        return fused, fired, event_score
    raise ValueError(f"unknown gate_mode: {gate_mode}")


def _public_gate_rule(gate_rule: dict[str, Any] | EventGateRule) -> dict[str, Any]:
    if isinstance(gate_rule, EventGateRule):
        return dict(gate_rule.metadata)
    return dict(gate_rule)


def _decode_control_residual_for_row(
    row: arc_gate.ArcRow,
    *,
    rows: Sequence[arc_gate.ArcRow],
    row_index: int,
    target_features: np.ndarray,
    packet: np.ndarray,
    decoder: Any,
    decoder_mode: str,
    subtract_zero_packet_baseline: bool,
    zero_baseline_residuals: Sequence[np.ndarray],
) -> np.ndarray:
    residual = _decode_one_row(
        row,
        rows=rows,
        row_index=row_index,
        target_features=target_features,
        packet=packet,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    if subtract_zero_packet_baseline:
        residual = np.asarray(residual, dtype=np.float64) - np.asarray(
            zero_baseline_residuals[row_index],
            dtype=np.float64,
        )
    return np.asarray(residual, dtype=np.float64)


def _event_training_packets(
    *,
    row_index: int,
    train_indices: Sequence[int],
    rows: Sequence[arc_gate.ArcRow],
    row_packets: Sequence[np.ndarray],
    target_derived_row_packets: Sequence[np.ndarray],
    source_selected: Sequence[int],
) -> dict[str, np.ndarray]:
    shuffled_index = _same_shape_shuffle_index(row_index=row_index, eval_indices=train_indices, rows=rows)
    same_choice_index = _same_choice_shuffle_index(
        row_index=row_index,
        eval_indices=train_indices,
        rows=rows,
        source_selected=source_selected,
    )
    return {
        "target_derived_packet": target_derived_row_packets[row_index],
        "zero_source": np.zeros_like(row_packets[row_index]),
        "source_row_shuffle": row_packets[shuffled_index],
        "same_source_choice_row_shuffle": row_packets[same_choice_index],
        "atom_shuffle": hidden_gate._atom_shuffle_packet(row_packets[row_index]),
        "coefficient_shuffle": hidden_gate._coefficient_shuffle_packet(row_packets[row_index]),
        "top_atom_knockout": hidden_gate._top_atom_knockout_packet(row_packets[row_index]),
        "candidate_roll": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=1),
        "candidate_derangement": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=-1),
    }


def _choose_event_gate_rule(
    *,
    rows: Sequence[arc_gate.ArcRow],
    train_indices: Sequence[int],
    target_scores: Sequence[Sequence[float]],
    source_residuals: Sequence[np.ndarray],
    target_derived_residuals: Sequence[np.ndarray],
    zero_residuals: Sequence[np.ndarray],
    row_packets: Sequence[np.ndarray],
    target_derived_row_packets: Sequence[np.ndarray],
    target_features: np.ndarray,
    decoder: Any,
    decoder_mode: str,
    subtract_zero_packet_baseline: bool,
    zero_baseline_residuals: Sequence[np.ndarray],
    source_selected: Sequence[int],
    ridge: float,
) -> EventGateRule:
    train = [int(index) for index in train_indices]
    examples: list[dict[str, Any]] = []
    for row_index in train:
        row = rows[row_index]
        target = [float(score) for score in target_scores[row_index]]
        target_pred = behavior_gate._prediction(target)
        target_correct = target_pred == int(row.answer_index)
        examples.append(
            {
                "condition": MATCHED_CONDITION,
                "row_index": row_index,
                "target": target,
                "packet": row_packets[row_index],
                "residual": np.asarray(source_residuals[row_index], dtype=np.float64),
                "target_correct": target_correct,
                "is_matched": True,
            }
        )
        corruption_packets = _event_training_packets(
            row_index=row_index,
            train_indices=train,
            rows=rows,
            row_packets=row_packets,
            target_derived_row_packets=target_derived_row_packets,
            source_selected=source_selected,
        )
        for condition, packet in corruption_packets.items():
            if condition == "target_derived_packet":
                residual = np.asarray(target_derived_residuals[row_index], dtype=np.float64)
            elif condition == "zero_source":
                residual = np.asarray(zero_residuals[row_index], dtype=np.float64)
            else:
                residual = _decode_control_residual_for_row(
                    row,
                    rows=rows,
                    row_index=row_index,
                    target_features=target_features,
                    packet=packet,
                    decoder=decoder,
                    decoder_mode=decoder_mode,
                    subtract_zero_packet_baseline=subtract_zero_packet_baseline,
                    zero_baseline_residuals=zero_baseline_residuals,
                )
            examples.append(
                {
                    "condition": condition,
                    "row_index": row_index,
                    "target": target,
                    "packet": packet,
                    "residual": np.asarray(residual, dtype=np.float64),
                    "target_correct": target_correct,
                    "is_matched": False,
                }
            )

    best_rule: EventGateRule | None = None
    best_key: tuple[Any, ...] | None = None
    for residual_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        feature_rows: list[np.ndarray] = []
        labels: list[float] = []
        for example in examples:
            feature_rows.append(
                _event_gate_features(
                    example["target"],
                    example["residual"],
                    example["packet"],
                    residual_weight=float(residual_weight),
                )
            )
            fused = np.asarray(example["target"], dtype=np.float64) + float(residual_weight) * np.asarray(
                example["residual"],
                dtype=np.float64,
            )
            fused_correct = behavior_gate._prediction(fused) == int(rows[int(example["row_index"])].answer_index)
            label = bool(example["is_matched"] and fused_correct and not bool(example["target_correct"]))
            labels.append(1.0 if label else 0.0)
        features = np.vstack(feature_rows)
        label_values = np.asarray(labels, dtype=np.float64)
        model = behavior_gate._fit_ridge_scalar_map(
            features,
            label_values,
            fit_indices=np.arange(features.shape[0], dtype=np.int64),
            ridge=float(ridge),
        )
        event_scores = np.asarray(model.predict(features), dtype=np.float64)
        thresholds = sorted(
            {
                float(event_scores.min() - 1e-6),
                float(event_scores.max() + 1e-6),
                *[float(value) for value in np.percentile(event_scores, [50, 60, 70, 80, 90])],
                *[float(value) for value in event_scores],
            }
        )
        for threshold in thresholds:
            correct = 0
            fired = 0
            helped = 0
            harmed = 0
            control_fired = 0
            margins: list[float] = []
            for example, event_score in zip(examples, event_scores, strict=True):
                row = rows[int(example["row_index"])]
                target = np.asarray(example["target"], dtype=np.float64)
                residual = np.asarray(example["residual"], dtype=np.float64)
                fused = target + float(residual_weight) * residual
                pred_changed = behavior_gate._prediction(fused) != behavior_gate._prediction(target)
                did_fire = float(event_score) >= float(threshold) and pred_changed
                if not bool(example["is_matched"]):
                    control_fired += int(did_fire)
                    continue
                selected_scores = fused if did_fire else target
                target_correct = bool(example["target_correct"])
                is_correct = behavior_gate._prediction(selected_scores) == int(row.answer_index)
                correct += int(is_correct)
                fired += int(did_fire)
                helped += int(did_fire and is_correct and not target_correct)
                harmed += int(did_fire and (not is_correct) and target_correct)
                margins.append(behavior_gate._margin(selected_scores, int(row.answer_index)))
            accuracy = correct / max(len(train), 1)
            fired_rate = fired / max(len(train), 1)
            control_count = max(len(examples) - len(train), 1)
            control_fired_rate = control_fired / control_count
            metadata = {
                "gate_mode": "event_triggered",
                "residual_weight": float(residual_weight),
                "threshold": float(threshold),
                "require_prediction_change": True,
                "train_accuracy": float(accuracy),
                "train_fired": int(fired),
                "train_fired_rate": float(fired_rate),
                "train_helped": int(helped),
                "train_harmed": int(harmed),
                "train_net_help": int(helped - harmed),
                "train_mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
                "train_event_examples": int(len(examples)),
                "train_positive_events": int(label_values.sum()),
                "train_corruption_examples": int(len(examples) - len(train)),
                "train_corruption_fired": int(control_fired),
                "train_corruption_fired_rate": float(control_fired_rate),
                "event_model_fit_mse": float(model.fit_mse),
                "event_model_fit_r2": float(model.fit_r2),
                "event_model_weight_l2": _safe_l2(np.asarray(model.weights, dtype=np.float64)),
                "event_feature_dim": int(features.shape[1]),
                "corruption_conditions": list(EVENT_GATE_CORRUPTION_CONDITIONS),
            }
            key = (
                metadata["train_accuracy"],
                metadata["train_net_help"],
                -metadata["train_harmed"],
                metadata["train_helped"],
                -metadata["train_corruption_fired"],
                -abs(metadata["train_fired_rate"] - 0.35),
                metadata["train_mean_margin"],
                -metadata["residual_weight"],
            )
            if best_rule is None or best_key is None or key > best_key:
                best_key = key
                best_rule = EventGateRule(
                    residual_weight=float(residual_weight),
                    threshold=float(threshold),
                    event_model=model,
                    require_prediction_change=True,
                    metadata=metadata,
                )
    if best_rule is None:
        raise ValueError("could not select event-triggered gate rule")
    return best_rule


def _condition_metrics(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for condition in REPORT_CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        correct = sum(1 for row in subset if row["correct"])
        fired = sum(1 for row in subset if row.get("packet_fired"))
        helped = sum(1 for row in subset if row.get("packet_helped"))
        harmed = sum(1 for row in subset if row.get("packet_harmed"))
        metrics[condition] = {
            "n": len(subset),
            "correct": int(correct),
            "accuracy": float(correct / len(subset)) if subset else 0.0,
            "mean_margin": float(statistics.fmean(float(row["margin"]) for row in subset)) if subset else 0.0,
            "packet_fired": int(fired),
            "packet_fired_rate": float(fired / len(subset)) if subset else 0.0,
            "packet_helped_vs_target": int(helped),
            "packet_harmed_vs_target": int(harmed),
            "packet_net_help_vs_target": int(helped - harmed),
        }
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    for control in CONTROL_CONDITIONS:
        correct_deltas = [
            float(group[MATCHED_CONDITION]["correct"]) - float(group[control]["correct"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and control in group
        ]
        margin_deltas = [
            float(group[MATCHED_CONDITION]["margin"]) - float(group[control]["margin"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and control in group
        ]
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{control}"] = behavior_gate._paired_bootstrap(
            correct_deltas,
            seed=seed + len(control),
            samples=bootstrap_samples,
        )
        metrics[MATCHED_CONDITION][f"mean_margin_delta_vs_{control}"] = (
            float(statistics.fmean(margin_deltas)) if margin_deltas else 0.0
        )
    return metrics


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["strict_headline"]
    lines = [
        "# ARC-Challenge Behavior-Atom Decoder Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/test disagreement rows: `{payload['train_rows']}` / `{payload['test_rows']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.6f}`",
        f"- target-only accuracy: `{headline['target_only_accuracy']:.6f}`",
        f"- best required control: `{headline['best_required_control']}`",
        f"- best required control accuracy: `{headline['best_required_control_accuracy']:.6f}`",
        f"- worst required paired CI95 low: `{headline['worst_required_ci95_low']:.6f}`",
        f"- fired rows: `{headline['matched_packet_fired']}`",
        f"- helps/harms vs target: `{headline['matched_packet_helped']}` / `{headline['matched_packet_harmed']}`",
        f"- packet bytes/row: `{payload['systems_packet_sideband']['packet_bytes_per_row']:.3f}`",
        f"- same-byte visible-text budget: `{payload['systems_packet_sideband']['same_byte_visible_text_budget']}`",
        "",
        "## Strict Controls",
        "",
        "| Control | Accuracy | Delta | CI95 low |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["strict_control_metrics"].items():
        lines.append(
            f"| `{name}` | {row['control_accuracy']:.6f} | {row['delta_accuracy']:.6f} | "
            f"{row['ci95_low']:.6f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    source_family_gate_dir: pathlib.Path,
    tiny_validation_score_cache: pathlib.Path,
    tiny_test_score_cache: pathlib.Path,
    qwen_validation_score_cache: pathlib.Path,
    qwen_test_score_cache: pathlib.Path,
    train_disagreement_limit: int,
    test_disagreement_limit: int,
    source_model: str,
    target_model: str,
    source_device: str,
    target_device: str,
    target_attn_implementation: str | None,
    dtype: str,
    source_max_length: int,
    target_max_length: int,
    source_hidden_layer: int,
    source_feature_dim: int,
    ridge: float,
    packet_rank: int,
    packet_top_k: int,
    packet_bits: int,
    local_files_only: bool,
    bootstrap_samples: int,
    same_byte_budget: int,
    subtract_zero_packet_baseline: bool,
    decoder_mode: str,
    gate_mode: str,
    min_accuracy_gap: float,
    min_ci_low: float,
    seed: int,
) -> dict[str, Any]:
    if decoder_mode not in {"target_conditioned", "packet_innovation"}:
        raise ValueError(f"unknown decoder_mode: {decoder_mode}")
    if gate_mode not in {"residual_threshold", "event_triggered"}:
        raise ValueError(f"unknown gate_mode: {gate_mode}")
    output_dir = behavior_gate._resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "strict_inputs"
    agreement_path = behavior_gate._resolve(source_family_gate_dir) / "source_cache_agreement.csv"

    validation_rows_all = arc_gate._load_rows(behavior_gate._resolve(validation_path))
    test_rows_all = arc_gate._load_rows(behavior_gate._resolve(test_path))
    train_ids = soft_gate._read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="validation",
        limit=train_disagreement_limit,
    )
    test_ids = soft_gate._read_disagreement_content_ids(
        agreement_path=agreement_path,
        split="test",
        limit=test_disagreement_limit,
    )
    train_rows = soft_gate._filter_rows_by_content_ids(validation_rows_all, train_ids)
    test_rows = soft_gate._filter_rows_by_content_ids(test_rows_all, test_ids)
    overlap = sorted({row.content_id for row in train_rows} & {row.content_id for row in test_rows})
    if overlap:
        raise ValueError(f"train/test content overlap: {overlap[:3]}")
    rows = [*train_rows, *test_rows]
    fit_row_count = len(train_rows)
    eval_indices = list(range(fit_row_count, fit_row_count + len(test_rows)))
    fit_candidate_indices = preflight._flat_candidate_indices_for_rows(rows, list(range(fit_row_count)))

    behavior_gate._write_jsonl(
        input_dir / "arc_challenge_validation_train_plus_test_disagreement.jsonl",
        [soft_gate._arc_row_payload(row) for row in rows],
    )
    behavior_gate._write_jsonl(
        input_dir / "tinyllama_source_score_cache.jsonl",
        [
            *soft_gate._select_cache_rows(cache_path=tiny_validation_score_cache, rows=train_rows),
            *soft_gate._select_cache_rows(cache_path=tiny_test_score_cache, rows=test_rows),
        ],
    )
    behavior_gate._write_jsonl(
        input_dir / "qwen_source_score_cache.jsonl",
        [
            *soft_gate._select_cache_rows(cache_path=qwen_validation_score_cache, rows=train_rows),
            *soft_gate._select_cache_rows(cache_path=qwen_test_score_cache, rows=test_rows),
        ],
    )
    tiny_cache = behavior_gate._load_score_rows(input_dir / "tinyllama_source_score_cache.jsonl")
    qwen_cache = behavior_gate._load_score_rows(input_dir / "qwen_source_score_cache.jsonl")

    target_scores, target_score_meta = behavior_gate._score_rows_with_prompt_builder(
        rows,
        model_path=target_model,
        device=target_device,
        dtype=dtype,
        max_length=target_max_length,
        local_files_only=local_files_only,
        normalization="mean",
        prompt_builder=preflight._mcq_prompt,
        attn_implementation=target_attn_implementation,
    )

    source_packet_flat, source_packet_meta = _fit_source_packet(
        rows,
        target_scores=target_scores,
        source_model=source_model,
        source_device=source_device,
        dtype=dtype,
        source_max_length=source_max_length,
        source_hidden_layer=source_hidden_layer,
        source_feature_dim=source_feature_dim,
        fit_candidate_indices=fit_candidate_indices,
        ridge=ridge,
        packet_rank=packet_rank,
        packet_top_k=packet_top_k,
        packet_bits=packet_bits,
        local_files_only=local_files_only,
    )
    sparse_meta = source_packet_meta["sparse_packet"]
    estimated_packet_bytes_per_row = float(sparse_meta["packet_bytes_per_candidate"] * max(len(row.choices) for row in rows))
    framed_packet_bytes = int(math.ceil(estimated_packet_bytes_per_row))
    same_byte_budget_used = int(same_byte_budget) if same_byte_budget > 0 else int(max(framed_packet_bytes, 1))

    def same_byte_prompt(row: arc_gate.ArcRow) -> str:
        selected = behavior_gate._source_prediction_for_row(row, tiny_cache)
        hint = row.choices[selected].encode("utf-8")[:same_byte_budget_used].decode("utf-8", errors="ignore")
        choices = "\n".join(f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True))
        return (
            "Answer the science question with the best answer.\n"
            f"Question: {row.question}\n"
            f"Choices:\n{choices}\n"
            f"Source model selected this visible hint: {hint}\n"
            "Answer:"
        )

    same_byte_scores, same_byte_meta = behavior_gate._score_rows_with_prompt_builder(
        test_rows,
        model_path=target_model,
        device=target_device,
        dtype=dtype,
        max_length=target_max_length,
        local_files_only=local_files_only,
        normalization="mean",
        prompt_builder=same_byte_prompt,
        attn_implementation=target_attn_implementation,
    )

    target_features = hidden_gate._target_score_features(rows, target_scores)
    behavior_targets = behavior_gate._candidate_targets(rows, target_scores)
    if decoder_mode == "packet_innovation":
        decoder = _fit_ridge_no_intercept_scalar_map(
            _innovation_decoder_features(target_features, source_packet_flat),
            behavior_targets,
            fit_indices=fit_candidate_indices,
            ridge=ridge,
        )
    else:
        decoder = behavior_gate._fit_ridge_scalar_map(
            hidden_gate._decoder_features(target_features, source_packet_flat),
            behavior_targets,
            fit_indices=fit_candidate_indices,
            ridge=ridge,
        )
    target_only_decoder = behavior_gate._fit_ridge_scalar_map(
        target_features,
        behavior_targets,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_packet_map = hidden_gate._fit_ridge_matrix_map(
        target_features,
        source_packet_flat,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_derived_packet_flat = target_packet_map.predict(target_features)

    source_residuals = _decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=source_packet_flat,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    target_decoder_residuals = hidden_gate._rows_from_candidate_values(rows, target_only_decoder.predict(target_features))
    target_derived_residuals = _decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=target_derived_packet_flat,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    zero_packet_flat = np.zeros_like(source_packet_flat)
    zero_residuals = _decode_packet_residual_rows(
        rows,
        target_features=target_features,
        packet_features=zero_packet_flat,
        decoder=decoder,
        decoder_mode=decoder_mode,
    )
    zero_baseline_residuals = [np.asarray(residual, dtype=np.float64) for residual in zero_residuals]
    if subtract_zero_packet_baseline:
        source_residuals = _subtract_row_baseline(source_residuals, zero_baseline_residuals)
        target_derived_residuals = _subtract_row_baseline(target_derived_residuals, zero_baseline_residuals)
        zero_residuals = [np.zeros_like(residual, dtype=np.float64) for residual in zero_residuals]
    row_packets = hidden_gate._row_packet_arrays(rows, source_packet_flat)
    target_derived_row_packets = hidden_gate._row_packet_arrays(rows, target_derived_packet_flat)
    source_selected_all = [behavior_gate._source_prediction_for_row(row, tiny_cache) for row in rows]
    if gate_mode == "event_triggered":
        gate_rule: dict[str, Any] | EventGateRule = _choose_event_gate_rule(
            rows=rows,
            train_indices=list(range(fit_row_count)),
            target_scores=target_scores,
            source_residuals=source_residuals,
            target_derived_residuals=target_derived_residuals,
            zero_residuals=zero_residuals,
            row_packets=row_packets,
            target_derived_row_packets=target_derived_row_packets,
            target_features=target_features,
            decoder=decoder,
            decoder_mode=decoder_mode,
            subtract_zero_packet_baseline=subtract_zero_packet_baseline,
            zero_baseline_residuals=zero_baseline_residuals,
            source_selected=source_selected_all,
            ridge=ridge,
        )
    else:
        gate_rule = hidden_gate._choose_gate_rule(
            train_rows,
            target_scores[:fit_row_count],
            source_residuals[:fit_row_count],
        )
    prediction_rows: list[dict[str, Any]] = []
    for eval_position, row in enumerate(test_rows):
        row_index = fit_row_count + eval_position
        target = [float(score) for score in target_scores[row_index]]
        shuffled_index = _same_shape_shuffle_index(
            row_index=row_index,
            eval_indices=eval_indices,
            rows=rows,
        )
        same_choice_index = _same_choice_shuffle_index(
            row_index=row_index,
            eval_indices=eval_indices,
            rows=rows,
            source_selected=source_selected_all,
        )
        raw_source_scores = behavior_gate._source_scores_for_row(row, tiny_cache)
        source_selected = int(source_selected_all[row_index])
        qwen_selected = behavior_gate._source_prediction_for_row(row, qwen_cache)
        candidate_packets = {
            MATCHED_CONDITION: row_packets[row_index],
            "target_derived_packet": target_derived_row_packets[row_index],
            "zero_source": np.zeros_like(row_packets[row_index]),
            "source_row_shuffle": row_packets[shuffled_index],
            "same_source_choice_row_shuffle": row_packets[same_choice_index],
            "atom_shuffle": hidden_gate._atom_shuffle_packet(row_packets[row_index]),
            "coefficient_shuffle": hidden_gate._coefficient_shuffle_packet(row_packets[row_index]),
            "top_atom_knockout": hidden_gate._top_atom_knockout_packet(row_packets[row_index]),
            "candidate_roll": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=1),
            "candidate_derangement": hidden_gate._candidate_roll_packet(row_packets[row_index], shift=-1),
        }
        condition_scores: dict[str, tuple[list[float], bool, np.ndarray, float | None]] = {}
        for condition, packet in candidate_packets.items():
            if condition == MATCHED_CONDITION:
                residual = source_residuals[row_index]
            elif condition == "target_derived_packet":
                residual = target_derived_residuals[row_index]
            elif condition == "zero_source":
                residual = zero_residuals[row_index]
            else:
                residual = _decode_one_row(
                    row,
                    rows=rows,
                    row_index=row_index,
                    target_features=target_features,
                    packet=packet,
                    decoder=decoder,
                    decoder_mode=decoder_mode,
                )
                if subtract_zero_packet_baseline:
                    residual = np.asarray(residual, dtype=np.float64) - np.asarray(
                        zero_baseline_residuals[row_index],
                        dtype=np.float64,
                    )
            fused, fired, event_score = _fuse_with_gate(
                target,
                residual,
                packet,
                gate_rule=gate_rule,
                gate_mode=gate_mode,
            )
            condition_scores[condition] = (fused, fired, np.asarray(residual, dtype=np.float64), event_score)
        target_decoder_scores, target_decoder_fired, target_decoder_event_score = _fuse_with_gate(
            target,
            target_decoder_residuals[row_index],
            np.zeros_like(row_packets[row_index]),
            gate_rule=gate_rule,
            gate_mode=gate_mode,
        )
        condition_scores.update(
            {
                "target_only": (target, False, np.zeros(len(row.choices), dtype=np.float64), None),
                "target_decoder_only": (
                    target_decoder_scores,
                    target_decoder_fired,
                    np.asarray(target_decoder_residuals[row_index], dtype=np.float64),
                    target_decoder_event_score,
                ),
                "packet_only_source_index": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "source_rank_control": (
                    behavior_gate._source_rank_scores(raw_source_scores),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "source_score_control": (
                    behavior_gate._centered_source_score_control(raw_source_scores),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "source_score_quantized_control": (
                    ecoc_gate._source_score_quantized_control(raw_source_scores, bits=4),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "same_byte_visible_text": (
                    same_byte_scores[eval_position],
                    False,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
                "qwen_substituted_packet": (
                    behavior_gate._source_index_scores(len(row.choices), qwen_selected),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                    None,
                ),
            }
        )
        target_correct = behavior_gate._prediction(target) == int(row.answer_index)
        for condition in REPORT_CONDITIONS:
            scores, fired, residual, event_score = condition_scores[condition]
            pred = behavior_gate._prediction(scores)
            correct = pred == int(row.answer_index)
            prediction_rows.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "condition": condition,
                    "answer_index": int(row.answer_index),
                    "answer_label": row.answer_label,
                    "prediction_index": int(pred),
                    "prediction_label": row.choice_labels[pred],
                    "correct": bool(correct),
                    "scores": [float(score) for score in scores],
                    "margin": float(behavior_gate._margin(scores, row.answer_index)),
                    "entropy": float(behavior_gate._entropy(scores)),
                    "packet_fired": bool(fired),
                    "packet_helped": bool(fired and correct and not target_correct),
                    "packet_harmed": bool(fired and (not correct) and target_correct),
                    "packet_residual": [float(value) for value in residual],
                    "event_score": float(event_score) if event_score is not None else None,
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "qwen_substituted_index": int(qwen_selected),
                    "qwen_substituted_label": row.choice_labels[qwen_selected],
                    "source_scores": [float(score) for score in raw_source_scores],
                    "gate_rule": _public_gate_rule(gate_rule),
                    "control_origin": condition,
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed, bootstrap_samples=bootstrap_samples)
    oracle = ecoc_gate._oracle_diagnostics(prediction_rows)
    matched = metrics[MATCHED_CONDITION]
    strict_control_metrics: dict[str, dict[str, float]] = {}
    for control in STRICT_REQUIRED_CONTROLS:
        paired = matched[f"paired_accuracy_vs_{control}"]
        strict_control_metrics[control] = {
            "control_accuracy": float(metrics[control]["accuracy"]),
            "delta_accuracy": float(matched["accuracy"] - metrics[control]["accuracy"]),
            "ci95_low": float(paired["ci95_low"]),
            "ci95_high": float(paired["ci95_high"]),
        }
    best_required_control = max(STRICT_REQUIRED_CONTROLS, key=lambda name: metrics[name]["accuracy"])
    worst_ci_low = min(row["ci95_low"] for row in strict_control_metrics.values())
    strict_pass = all(
        row["delta_accuracy"] >= float(min_accuracy_gap) and row["ci95_low"] > float(min_ci_low)
        for row in strict_control_metrics.values()
    )
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "source_private_arc_challenge_behavior_atom_decoder_gate",
        "date": dt.date.today().isoformat(),
        "created_utc": created,
        "pass_gate": bool(strict_pass),
        "implementation_gate_only": False,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "strict_required_controls": list(STRICT_REQUIRED_CONTROLS),
        "strict_control_metrics": strict_control_metrics,
        "strict_headline": {
            "matched_accuracy": float(matched["accuracy"]),
            "target_only_accuracy": float(metrics["target_only"]["accuracy"]),
            "best_required_control": best_required_control,
            "best_required_control_accuracy": float(metrics[best_required_control]["accuracy"]),
            "worst_required_ci95_low": float(worst_ci_low),
            "matched_packet_fired": int(matched["packet_fired"]),
            "matched_packet_fired_rate": float(matched["packet_fired_rate"]),
            "matched_packet_helped": int(matched["packet_helped_vs_target"]),
            "matched_packet_harmed": int(matched["packet_harmed_vs_target"]),
            "matched_packet_net_help": int(matched["packet_net_help_vs_target"]),
        },
        "condition_metrics": metrics,
        "oracle_diagnostics": oracle,
        "systems_packet_sideband": {
            "source_private": True,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_exposed": False,
            "raw_logits_or_scores_exposed": False,
            "native_serving_throughput_measured": False,
            "packet_bytes_per_row": float(estimated_packet_bytes_per_row),
            "framed_packet_bytes_per_row": int(framed_packet_bytes),
            "same_byte_visible_text_budget": int(same_byte_budget_used),
            "cache_line_bytes_per_row_64b": int(math.ceil(max(framed_packet_bytes, 1) / 64.0) * 64),
            "dma_bytes_per_row_128b": int(math.ceil(max(framed_packet_bytes, 1) / 128.0) * 128),
            "decode_flops_proxy_per_row": int(max(len(row.choices) for row in rows) * packet_rank),
            "sparse_packet_metadata": sparse_meta,
            "subtract_zero_packet_baseline": bool(subtract_zero_packet_baseline),
            "decoder_mode": decoder_mode,
            "gate_mode": gate_mode,
            "note": (
                "Byte counts cover behavior-supervised source-hidden atom IDs plus quantized coefficients only. "
                "They are not native GPU throughput, HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "feature_metadata": {
            **source_packet_meta,
            "target_score_metadata": target_score_meta,
            "same_byte_score_metadata": same_byte_meta,
            "target_conditioned_decoder": {
                "decoder_mode": decoder_mode,
                "ridge": decoder.ridge,
                "fit_mse": decoder.fit_mse,
                "fit_r2": decoder.fit_r2,
                "selected_gate_rule": _public_gate_rule(gate_rule),
            },
            "target_only_decoder": {
                "ridge": target_only_decoder.ridge,
                "fit_mse": target_only_decoder.fit_mse,
                "fit_r2": target_only_decoder.fit_r2,
            },
            "target_derived_packet_map": {
                "ridge": target_packet_map.ridge,
                "fit_mse": target_packet_map.fit_mse,
                "fit_r2": target_packet_map.fit_r2,
            },
        },
        "inputs": {
            "validation_path": behavior_gate._display(validation_path),
            "test_path": behavior_gate._display(test_path),
            "source_family_gate_dir": behavior_gate._display(source_family_gate_dir),
            "agreement_path": behavior_gate._display(agreement_path),
            "tiny_validation_score_cache": behavior_gate._display(tiny_validation_score_cache),
            "tiny_test_score_cache": behavior_gate._display(tiny_test_score_cache),
            "qwen_validation_score_cache": behavior_gate._display(qwen_validation_score_cache),
            "qwen_test_score_cache": behavior_gate._display(qwen_test_score_cache),
            "source_model": str(source_model),
            "target_model": str(target_model),
            "train_disagreement_limit": int(train_disagreement_limit),
            "test_disagreement_limit": int(test_disagreement_limit),
            "packet_rank": int(packet_rank),
            "packet_top_k": int(packet_top_k),
            "packet_bits": int(packet_bits),
            "same_byte_budget": int(same_byte_budget_used),
            "subtract_zero_packet_baseline": bool(subtract_zero_packet_baseline),
            "decoder_mode": decoder_mode,
            "gate_mode": gate_mode,
        },
        "interpretation": (
            "This gate tests whether source-hidden innovations become more useful when the sparse atom basis is "
            "trained toward target residual behavior rather than unsupervised PCA variance. It passes only if the "
            "matched packet beats target-only, target-decoder-only, target-derived packets, same-source-choice and "
            "generic wrong-row packets, atom/coefficient destruction, candidate roll/derangement, source-index/"
            "rank/score, same-byte text, and Qwen-substitution controls with positive paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_behavior_atom_decoder_gate.json"
    md_path = output_dir / "arc_challenge_behavior_atom_decoder_gate.md"
    audit_path = output_dir / "prediction_audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    behavior_gate._write_jsonl(audit_path, prediction_rows)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {"path": behavior_gate._display(json_path), "sha256": behavior_gate._sha256_file(json_path), "bytes": json_path.stat().st_size},
            {"path": behavior_gate._display(md_path), "sha256": behavior_gate._sha256_file(md_path), "bytes": md_path.stat().st_size},
            {"path": behavior_gate._display(audit_path), "sha256": behavior_gate._sha256_file(audit_path), "bytes": audit_path.stat().st_size},
        ],
        "inputs": payload["inputs"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gc.collect()
    print(json.dumps({"headline": payload["strict_headline"], "pass_gate": payload["pass_gate"]}, sort_keys=True))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--source-family-gate-dir", type=pathlib.Path, default=DEFAULT_SOURCE_FAMILY_GATE_DIR)
    parser.add_argument("--tiny-validation-score-cache", type=pathlib.Path, default=DEFAULT_TINY_VALIDATION_SCORE_CACHE)
    parser.add_argument("--tiny-test-score-cache", type=pathlib.Path, default=DEFAULT_TINY_TEST_SCORE_CACHE)
    parser.add_argument("--qwen-validation-score-cache", type=pathlib.Path, default=DEFAULT_QWEN_VALIDATION_SCORE_CACHE)
    parser.add_argument("--qwen-test-score-cache", type=pathlib.Path, default=DEFAULT_QWEN_TEST_SCORE_CACHE)
    parser.add_argument("--train-disagreement-limit", type=int, default=16)
    parser.add_argument("--test-disagreement-limit", type=int, default=16)
    parser.add_argument("--source-model", default=str(DEFAULT_TINY_MODEL))
    parser.add_argument("--target-model", default=str(DEFAULT_QWEN3_MODEL))
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=160)
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--packet-rank", type=int, default=8)
    parser.add_argument("--packet-top-k", type=int, default=2)
    parser.add_argument("--packet-bits", type=int, default=4)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--same-byte-budget",
        type=int,
        default=0,
        help="Visible-text byte budget. 0 means use framed packet bytes per row.",
    )
    parser.add_argument("--subtract-zero-packet-baseline", choices=("true", "false"), default="true")
    parser.add_argument("--decoder-mode", choices=("target_conditioned", "packet_innovation"), default="target_conditioned")
    parser.add_argument("--gate-mode", choices=("residual_threshold", "event_triggered"), default="residual_threshold")
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_gate(
        output_dir=args.output_dir,
        validation_path=args.validation_path,
        test_path=args.test_path,
        source_family_gate_dir=args.source_family_gate_dir,
        tiny_validation_score_cache=args.tiny_validation_score_cache,
        tiny_test_score_cache=args.tiny_test_score_cache,
        qwen_validation_score_cache=args.qwen_validation_score_cache,
        qwen_test_score_cache=args.qwen_test_score_cache,
        train_disagreement_limit=int(args.train_disagreement_limit),
        test_disagreement_limit=int(args.test_disagreement_limit),
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        source_device=str(args.source_device),
        target_device=str(args.target_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        source_max_length=int(args.source_max_length),
        target_max_length=int(args.target_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        source_feature_dim=int(args.source_feature_dim),
        ridge=float(args.ridge),
        packet_rank=int(args.packet_rank),
        packet_top_k=int(args.packet_top_k),
        packet_bits=int(args.packet_bits),
        local_files_only=str(args.local_files_only).lower() == "true",
        bootstrap_samples=int(args.bootstrap_samples),
        same_byte_budget=int(args.same_byte_budget),
        subtract_zero_packet_baseline=str(args.subtract_zero_packet_baseline).lower() == "true",
        decoder_mode=str(args.decoder_mode),
        gate_mode=str(args.gate_mode),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
