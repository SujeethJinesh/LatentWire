from __future__ import annotations

"""Strict ARC-Challenge hidden-atom residual decoder Sparse Resonance Packet gate.

This branch tests the next receiver/message shape after score-syndrome packets
failed.  The source packet is a sparse top-k PCA atom code over
answer-key-forbidden source candidate hidden innovations.  The receiver uses a
small target-conditioned residual decoder rather than a direct score override
or soft-prefix reconstruction.
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
from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as soft_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_hidden_atom_decoder_gate_20260504_"
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

MATCHED_CONDITION = "matched_hidden_atom_decoder_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_decoder_only",
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
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


@dataclass(frozen=True)
class RidgeMatrixMap:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    weights: np.ndarray
    ridge: float
    fit_mse: float
    fit_r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        standardized = (values - self.x_mean) / self.x_std
        return standardized @ self.weights + self.y_mean


def _fit_ridge_matrix_map(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_indices: np.ndarray,
    ridge: float,
) -> RidgeMatrixMap:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and targets must be rank-2")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature/target row mismatch")
    if fit.size == 0:
        raise ValueError("fit_indices must not be empty")
    x_mean = x[fit].mean(axis=0, keepdims=True)
    x_std = x[fit].std(axis=0, keepdims=True).clip(min=1e-6)
    standardized = (x - x_mean) / x_std
    x_fit = standardized[fit]
    y_mean = y[fit].mean(axis=0, keepdims=True)
    y_fit = y[fit] - y_mean
    kernel = x_fit @ x_fit.T
    if ridge > 0.0:
        kernel = kernel + float(ridge) * np.eye(kernel.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(kernel, y_fit)
    weights = x_fit.T @ alpha
    pred_fit = x_fit @ weights + y_mean
    mse = float(np.mean(np.square(y[fit] - pred_fit)))
    baseline = float(np.mean(np.square(y[fit] - y_mean)))
    r2 = 0.0 if baseline <= 1e-12 else 1.0 - mse / baseline
    return RidgeMatrixMap(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        weights=weights,
        ridge=float(ridge),
        fit_mse=mse,
        fit_r2=float(r2),
    )


def _target_score_features(rows: Sequence[arc_gate.ArcRow], target_scores: Sequence[Sequence[float]]) -> np.ndarray:
    return behavior_gate._target_score_features(rows, target_scores)


def _decoder_features(target_features: np.ndarray, packet_features: np.ndarray) -> np.ndarray:
    target = np.asarray(target_features, dtype=np.float64)
    packet = np.asarray(packet_features, dtype=np.float64)
    if target.ndim != 2 or packet.ndim != 2:
        raise ValueError("target_features and packet_features must be rank-2")
    if target.shape[0] != packet.shape[0]:
        raise ValueError("target/packet candidate mismatch")
    target_summary = target[:, [1, 2, 4]] if target.shape[1] >= 5 else target
    return np.concatenate([target, packet, packet * target_summary[:, :1]], axis=1)


def _row_offsets(rows: Sequence[arc_gate.ArcRow]) -> list[tuple[int, int]]:
    return behavior_gate._row_offsets(rows)


def _rows_from_candidate_values(rows: Sequence[arc_gate.ArcRow], values: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for start, end in _row_offsets(rows):
        row = np.asarray(values[start:end], dtype=np.float64).reshape(-1)
        out.append(row - float(row.mean()) if row.size else row)
    return out


def _row_packet_arrays(rows: Sequence[arc_gate.ArcRow], packet_features: np.ndarray) -> list[np.ndarray]:
    packets: list[np.ndarray] = []
    for start, end in _row_offsets(rows):
        packets.append(np.asarray(packet_features[start:end], dtype=np.float64))
    return packets


def _flat_from_row_packets(row_packets: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(packet, dtype=np.float64) for packet in row_packets], axis=0)


def _fit_source_packet(
    rows: Sequence[arc_gate.ArcRow],
    *,
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
    sparse_packet, sparse_meta = preflight._sparse_pca_packet_features(
        innovation,
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


def _top_atom_knockout_packet(row_packet: np.ndarray) -> np.ndarray:
    packet = np.asarray(row_packet, dtype=np.float64).copy()
    if packet.size:
        flat_index = int(np.argmax(np.abs(packet)))
        packet.reshape(-1)[flat_index] = 0.0
    return packet


def _candidate_roll_packet(row_packet: np.ndarray, *, shift: int) -> np.ndarray:
    return np.roll(np.asarray(row_packet, dtype=np.float64), int(shift), axis=0)


def _atom_shuffle_packet(row_packet: np.ndarray) -> np.ndarray:
    return np.roll(np.asarray(row_packet, dtype=np.float64), 1, axis=1)


def _coefficient_shuffle_packet(row_packet: np.ndarray) -> np.ndarray:
    return np.asarray(row_packet, dtype=np.float64)[:, ::-1]


def _packet_l1(row_residual: Sequence[float]) -> float:
    return float(np.sum(np.abs(np.asarray(row_residual, dtype=np.float64))))


def _fused_scores(
    target_scores: Sequence[float],
    residual: Sequence[float],
    *,
    rule: dict[str, Any],
) -> tuple[list[float], bool]:
    target = np.asarray(target_scores, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if target.shape != correction.shape:
        raise ValueError("target/residual shape mismatch")
    fused = target + float(rule["residual_weight"]) * correction
    target_pred = behavior_gate._prediction(target)
    fused_pred = behavior_gate._prediction(fused)
    fires = _packet_l1(correction) >= float(rule["min_residual_l1"])
    fires = fires and behavior_gate._margin(target, target_pred) <= float(rule["max_target_margin"])
    if bool(rule.get("require_prediction_change", False)):
        fires = fires and fused_pred != target_pred
    if not fires:
        return [float(score) for score in target], False
    return [float(score) for score in fused], True


def _choose_gate_rule(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
    residuals: Sequence[np.ndarray],
) -> dict[str, Any]:
    l1_values = [_packet_l1(residual) for residual in residuals]
    margin_values = [_target_margin(scores) for scores in target_scores]
    l1_thresholds = sorted({0.0, *[float(value) for value in np.percentile(l1_values, [25, 50, 75])]})
    margin_thresholds = sorted({float("inf"), *[float(value) for value in np.percentile(margin_values, [25, 50, 75])]})
    candidates: list[dict[str, Any]] = []
    for residual_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        for min_residual_l1 in l1_thresholds:
            for max_target_margin in margin_thresholds:
                for require_prediction_change in (False, True):
                    candidates.append(
                        {
                            "residual_weight": float(residual_weight),
                            "min_residual_l1": float(min_residual_l1),
                            "max_target_margin": float(max_target_margin),
                            "require_prediction_change": bool(require_prediction_change),
                        }
                    )
    candidates.append(
        {
            "residual_weight": 0.0,
            "min_residual_l1": float("inf"),
            "max_target_margin": -1.0,
            "require_prediction_change": True,
        }
    )
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for rule in candidates:
        correct = 0
        fired = 0
        helped = 0
        harmed = 0
        margins: list[float] = []
        for row, scores, residual in zip(rows, target_scores, residuals, strict=True):
            target_pred = behavior_gate._prediction(scores)
            target_correct = target_pred == int(row.answer_index)
            fused, did_fire = _fused_scores(scores, residual, rule=rule)
            pred = behavior_gate._prediction(fused)
            is_correct = pred == int(row.answer_index)
            correct += int(is_correct)
            fired += int(did_fire)
            helped += int(did_fire and is_correct and not target_correct)
            harmed += int(did_fire and (not is_correct) and target_correct)
            margins.append(behavior_gate._margin(fused, int(row.answer_index)))
        accuracy = correct / max(len(rows), 1)
        fired_rate = fired / max(len(rows), 1)
        candidate = {
            **rule,
            "train_accuracy": float(accuracy),
            "train_fired": int(fired),
            "train_fired_rate": float(fired_rate),
            "train_helped": int(helped),
            "train_harmed": int(harmed),
            "train_net_help": int(helped - harmed),
            "train_mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
        }
        key = (
            candidate["train_accuracy"],
            candidate["train_net_help"],
            -candidate["train_harmed"],
            candidate["train_helped"],
            -abs(candidate["train_fired_rate"] - 0.35),
            candidate["train_mean_margin"],
            -candidate["residual_weight"],
        )
        if best is None or best_key is None or key > best_key:
            best = candidate
            best_key = key
    if best is None:
        raise ValueError("could not select hidden atom decoder gate rule")
    return best


def _target_margin(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    return float(values[order[0]] - values[order[1]])


def _decode_residual_rows(
    rows: Sequence[arc_gate.ArcRow],
    *,
    target_features: np.ndarray,
    packet_features: np.ndarray,
    decoder: behavior_gate.RidgeScalarMap,
) -> list[np.ndarray]:
    flat_pred = decoder.predict(_decoder_features(target_features, packet_features))
    return _rows_from_candidate_values(rows, flat_pred)


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
        "# ARC-Challenge Hidden-Atom Decoder Packet Gate",
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
    min_accuracy_gap: float,
    min_ci_low: float,
    seed: int,
) -> dict[str, Any]:
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

    def same_byte_prompt(row: arc_gate.ArcRow) -> str:
        selected = behavior_gate._source_prediction_for_row(row, tiny_cache)
        hint = row.choices[selected].encode("utf-8")[:same_byte_budget].decode("utf-8", errors="ignore")
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

    source_packet_flat, source_packet_meta = _fit_source_packet(
        rows,
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
    target_features = _target_score_features(rows, target_scores)
    behavior_targets = behavior_gate._candidate_targets(rows, target_scores)
    decoder = behavior_gate._fit_ridge_scalar_map(
        _decoder_features(target_features, source_packet_flat),
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
    target_packet_map = _fit_ridge_matrix_map(
        target_features,
        source_packet_flat,
        fit_indices=fit_candidate_indices,
        ridge=ridge,
    )
    target_derived_packet_flat = target_packet_map.predict(target_features)

    source_residuals = _decode_residual_rows(
        rows,
        target_features=target_features,
        packet_features=source_packet_flat,
        decoder=decoder,
    )
    gate_rule = _choose_gate_rule(
        train_rows,
        target_scores[:fit_row_count],
        source_residuals[:fit_row_count],
    )
    target_decoder_residuals = _rows_from_candidate_values(rows, target_only_decoder.predict(target_features))
    target_derived_residuals = _decode_residual_rows(
        rows,
        target_features=target_features,
        packet_features=target_derived_packet_flat,
        decoder=decoder,
    )
    zero_packet_flat = np.zeros_like(source_packet_flat)
    zero_residuals = _decode_residual_rows(
        rows,
        target_features=target_features,
        packet_features=zero_packet_flat,
        decoder=decoder,
    )

    row_packets = _row_packet_arrays(rows, source_packet_flat)
    target_derived_row_packets = _row_packet_arrays(rows, target_derived_packet_flat)
    prediction_rows: list[dict[str, Any]] = []
    eval_offset = fit_row_count
    for eval_position, row in enumerate(test_rows):
        row_index = eval_offset + eval_position
        target = [float(score) for score in target_scores[row_index]]
        shuffled_index = eval_offset + ((eval_position + 1) % len(test_rows))
        raw_source_scores = behavior_gate._source_scores_for_row(row, tiny_cache)
        source_selected = behavior_gate._source_prediction_for_row(row, tiny_cache)
        qwen_selected = behavior_gate._source_prediction_for_row(row, qwen_cache)
        candidate_packets = {
            MATCHED_CONDITION: row_packets[row_index],
            "target_derived_packet": target_derived_row_packets[row_index],
            "zero_source": np.zeros_like(row_packets[row_index]),
            "source_row_shuffle": row_packets[shuffled_index],
            "atom_shuffle": _atom_shuffle_packet(row_packets[row_index]),
            "coefficient_shuffle": _coefficient_shuffle_packet(row_packets[row_index]),
            "top_atom_knockout": _top_atom_knockout_packet(row_packets[row_index]),
            "candidate_roll": _candidate_roll_packet(row_packets[row_index], shift=1),
            "candidate_derangement": _candidate_roll_packet(row_packets[row_index], shift=-1),
        }
        condition_scores: dict[str, tuple[list[float], bool, np.ndarray]] = {}
        for condition, packet in candidate_packets.items():
            if condition == MATCHED_CONDITION:
                residual = source_residuals[row_index]
            elif condition == "target_derived_packet":
                residual = target_derived_residuals[row_index]
            elif condition == "zero_source":
                residual = zero_residuals[row_index]
            else:
                flat_packet = packet
                residual = _decode_residual_rows(
                    [row],
                    target_features=target_features[_row_offsets(rows)[row_index][0] : _row_offsets(rows)[row_index][1]],
                    packet_features=flat_packet,
                    decoder=decoder,
                )[0]
            fused, fired = _fused_scores(target, residual, rule=gate_rule)
            condition_scores[condition] = (fused, fired, np.asarray(residual, dtype=np.float64))
        target_decoder_scores, target_decoder_fired = _fused_scores(target, target_decoder_residuals[row_index], rule=gate_rule)
        condition_scores.update(
            {
                "target_only": (target, False, np.zeros(len(row.choices), dtype=np.float64)),
                "target_decoder_only": (
                    target_decoder_scores,
                    target_decoder_fired,
                    np.asarray(target_decoder_residuals[row_index], dtype=np.float64),
                ),
                "packet_only_source_index": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                ),
                "source_rank_control": (
                    behavior_gate._source_rank_scores(raw_source_scores),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                ),
                "source_score_control": (
                    behavior_gate._centered_source_score_control(raw_source_scores),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                ),
                "source_score_quantized_control": (
                    ecoc_gate._source_score_quantized_control(raw_source_scores, bits=4),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                ),
                "same_byte_visible_text": (
                    same_byte_scores[eval_position],
                    False,
                    np.zeros(len(row.choices), dtype=np.float64),
                ),
                "qwen_substituted_packet": (
                    behavior_gate._source_index_scores(len(row.choices), qwen_selected),
                    True,
                    np.zeros(len(row.choices), dtype=np.float64),
                ),
            }
        )
        target_correct = behavior_gate._prediction(target) == int(row.answer_index)
        for condition in REPORT_CONDITIONS:
            scores, fired, residual = condition_scores[condition]
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
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "qwen_substituted_index": int(qwen_selected),
                    "qwen_substituted_label": row.choice_labels[qwen_selected],
                    "source_scores": [float(score) for score in raw_source_scores],
                    "gate_rule": gate_rule,
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
    sparse_meta = source_packet_meta["sparse_packet"]
    estimated_packet_bytes_per_row = float(sparse_meta["packet_bytes_per_candidate"] * max(len(row.choices) for row in rows))
    framed_packet_bytes = int(math.ceil(estimated_packet_bytes_per_row))
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "source_private_arc_challenge_hidden_atom_decoder_gate",
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
            "cache_line_bytes_per_row_64b": int(math.ceil(max(framed_packet_bytes, 1) / 64.0) * 64),
            "dma_bytes_per_row_128b": int(math.ceil(max(framed_packet_bytes, 1) / 128.0) * 128),
            "decode_flops_proxy_per_row": int(max(len(row.choices) for row in rows) * packet_rank),
            "sparse_packet_metadata": sparse_meta,
            "note": (
                "Byte counts cover sparse source-hidden atom IDs plus quantized coefficients only. They are not "
                "native GPU throughput, HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "feature_metadata": {
            **source_packet_meta,
            "target_score_metadata": target_score_meta,
            "same_byte_score_metadata": same_byte_meta,
            "target_conditioned_decoder": {
                "ridge": decoder.ridge,
                "fit_mse": decoder.fit_mse,
                "fit_r2": decoder.fit_r2,
                "selected_gate_rule": gate_rule,
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
            "same_byte_budget": int(same_byte_budget),
        },
        "interpretation": (
            "This gate tests whether compact source-hidden PCA atom packets become useful when decoded by a "
            "target-conditioned residual receiver instead of direct soft-prefix reconstruction. It passes only if "
            "the matched packet beats target-only, target-decoder-only, target-derived packets, source-row/atom/"
            "coefficient destruction, candidate roll/derangement, source-index/rank/score, same-byte text, and "
            "Qwen-substitution controls with positive paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_hidden_atom_decoder_gate.json"
    md_path = output_dir / "arc_challenge_hidden_atom_decoder_gate.md"
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
    parser.add_argument("--same-byte-budget", type=int, default=4096)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=31)
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
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
