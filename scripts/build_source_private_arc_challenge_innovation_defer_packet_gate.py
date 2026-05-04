from __future__ import annotations

"""Strict ARC-Challenge event-triggered innovation/defer SRP gate.

This gate keeps the 2-byte confidence/ECOC packet transport, but replaces the
hand-binned receiver rule with a tiny train/calibration value model. The target
fires only when the learned receiver predicts that overriding to the source
packet is more likely to help than harm.
"""

import argparse
import copy
import datetime as dt
import json
import math
import pathlib
import statistics
import sys
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
    "results/source_private_arc_challenge_innovation_defer_packet_gate_20260504_"
    "tinyllama_to_qwen3_disagreement"
)
DEFAULT_VALIDATION = soft_gate.DEFAULT_VALIDATION
DEFAULT_TEST = soft_gate.DEFAULT_TEST
DEFAULT_SOURCE_FAMILY_GATE_DIR = soft_gate.DEFAULT_SOURCE_FAMILY_GATE_DIR
DEFAULT_TINY_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_TINY_VALIDATION_SCORE_CACHE
DEFAULT_TINY_TEST_SCORE_CACHE = soft_gate.DEFAULT_TINY_TEST_SCORE_CACHE
DEFAULT_QWEN_VALIDATION_SCORE_CACHE = soft_gate.DEFAULT_QWEN_VALIDATION_SCORE_CACHE
DEFAULT_QWEN_TEST_SCORE_CACHE = soft_gate.DEFAULT_QWEN_TEST_SCORE_CACHE
DEFAULT_QWEN3_MODEL = behavior_gate.DEFAULT_QWEN3_MODEL

MATCHED_CONDITION = "matched_innovation_defer_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "header_shuffle",
    "reliability_bin_shuffle",
    "ecoc_bit_shuffle",
    "parity_flip",
    "candidate_roll",
    "candidate_derangement",
    "packet_only_source_index",
    "top1_without_gate",
    "top2_without_gate",
    "source_rank_control",
    "source_score_control",
    "source_score_quantized_control",
    "same_byte_visible_text",
    "qwen_substituted_packet",
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
STRICT_REQUIRED_CONTROLS = CONTROL_CONDITIONS


def _packet_features(target_scores: Sequence[float], packet: dict[str, Any]) -> np.ndarray:
    target = np.asarray(target_scores, dtype=np.float64)
    probs = behavior_gate._softmax(target)
    decoded = int(ecoc_gate._decode_ecoc_candidate(packet)) % max(len(target), 1)
    target_pred = behavior_gate._prediction(target)
    order = sorted(range(len(target)), key=lambda index: (-float(target[index]), index))
    rank_by_candidate = {candidate: rank for rank, candidate in enumerate(order)}
    margin = ecoc_gate._target_margin(target)
    entropy = behavior_gate._entropy(target)
    source_target_gap = float(target[decoded] - target[target_pred]) if target.size else 0.0
    source_prob = float(probs[decoded]) if probs.size else 0.0
    top_prob = float(probs[target_pred]) if probs.size else 0.0
    top2_index = int(packet["top2_index"]) % max(len(target), 1)
    top2_gap = float(target[top2_index] - target[target_pred]) if target.size else 0.0
    return np.asarray(
        [
            1.0,
            float(margin),
            float(entropy),
            float(top_prob),
            float(source_prob),
            float(source_target_gap),
            float(top2_gap),
            float(rank_by_candidate.get(decoded, len(target)) / max(len(target) - 1, 1)),
            float(decoded != target_pred),
            float(packet["margin_bin"]),
            float(packet["entropy_bin"]),
            float(packet["margin"]),
            float(packet["entropy"]),
            float(ecoc_gate._parity_ok(packet)),
            float(packet["margin_bin"]) * float(decoded != target_pred),
            float(source_target_gap) * float(packet["margin_bin"]),
            float(entropy) * float(packet["entropy_bin"]),
        ],
        dtype=np.float64,
    )


def _row_value_targets(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
    packets: Sequence[dict[str, Any]],
) -> np.ndarray:
    targets: list[float] = []
    for row, scores, packet in zip(rows, target_scores, packets, strict=True):
        target_correct = behavior_gate._prediction(scores) == int(row.answer_index)
        decoded_correct = ecoc_gate._decode_ecoc_candidate(packet) == int(row.answer_index)
        targets.append(float(decoded_correct) - float(target_correct))
    return np.asarray(targets, dtype=np.float64)


def _fit_value_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_indices: np.ndarray,
    ridge: float,
) -> behavior_gate.RidgeScalarMap:
    return behavior_gate._fit_ridge_scalar_map(
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        fit_indices=np.asarray(fit_indices, dtype=np.int64),
        ridge=float(ridge),
    )


def _apply_packet(
    target_scores: Sequence[float],
    packet: dict[str, Any],
    *,
    value_model: behavior_gate.RidgeScalarMap,
    threshold: float,
    require_disagree: bool,
    require_parity: bool,
) -> tuple[list[float], bool, int, float]:
    decoded = int(ecoc_gate._decode_ecoc_candidate(packet))
    predicted_gain = float(value_model.predict(_packet_features(target_scores, packet)[None, :])[0])
    target_pred = behavior_gate._prediction(target_scores)
    fires = predicted_gain > float(threshold)
    if require_parity:
        fires = fires and ecoc_gate._parity_ok(packet)
    if require_disagree:
        fires = fires and decoded != target_pred
    if fires:
        return ecoc_gate._force_candidate_scores(target_scores, decoded), True, decoded, predicted_gain
    return [float(score) for score in target_scores], False, decoded, predicted_gain


def _choose_defer_rule(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
    packets: Sequence[dict[str, Any]],
    *,
    value_model: behavior_gate.RidgeScalarMap,
) -> dict[str, Any]:
    features = np.asarray([_packet_features(scores, packet) for scores, packet in zip(target_scores, packets, strict=True)])
    predicted = value_model.predict(features)
    thresholds = sorted({float(value) for value in predicted})
    candidates = [min(thresholds) - 1e-6, *thresholds, max(thresholds) + 1e-6] if thresholds else [0.0]
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for threshold in candidates:
        for require_disagree in (False, True):
            correct = 0
            fired = 0
            helped = 0
            harmed = 0
            margins: list[float] = []
            for row, scores, packet in zip(rows, target_scores, packets, strict=True):
                target_correct = behavior_gate._prediction(scores) == int(row.answer_index)
                fused, did_fire, _decoded, _gain = _apply_packet(
                    scores,
                    packet,
                    value_model=value_model,
                    threshold=threshold,
                    require_disagree=require_disagree,
                    require_parity=True,
                )
                pred = behavior_gate._prediction(fused)
                is_correct = pred == int(row.answer_index)
                correct += int(is_correct)
                fired += int(did_fire)
                helped += int(did_fire and is_correct and not target_correct)
                harmed += int(did_fire and (not is_correct) and target_correct)
                margins.append(behavior_gate._margin(fused, row.answer_index))
            accuracy = correct / max(len(rows), 1)
            fired_rate = fired / max(len(rows), 1)
            row_result = {
                "threshold": float(threshold),
                "require_disagree": bool(require_disagree),
                "require_parity": True,
                "calibration_accuracy": float(accuracy),
                "calibration_fired": int(fired),
                "calibration_fired_rate": float(fired_rate),
                "calibration_helped": int(helped),
                "calibration_harmed": int(harmed),
                "calibration_net_help": int(helped - harmed),
                "calibration_mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
            }
            key = (
                row_result["calibration_accuracy"],
                row_result["calibration_net_help"],
                -row_result["calibration_harmed"],
                row_result["calibration_helped"],
                -abs(row_result["calibration_fired_rate"] - 0.35),
                row_result["calibration_mean_margin"],
            )
            if best is None or best_key is None or key > best_key:
                best = row_result
                best_key = key
    if best is None:
        raise ValueError("could not choose defer rule")
    return best


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


def _oracle_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return ecoc_gate._oracle_diagnostics(prediction_rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["strict_headline"]
    lines = [
        "# ARC-Challenge Innovation-Defer Packet Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- train/value/calibration/test rows: `{payload['train_rows']}` / "
        f"`{payload['value_fit_rows']}` / `{payload['calibration_rows']}` / `{payload['test_rows']}`",
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
    target_model: str,
    target_device: str,
    target_attn_implementation: str | None,
    dtype: str,
    target_max_length: int,
    ridge: float,
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
    fit_row_count = max(1, len(train_rows) // 2)
    calibration_count = len(train_rows) - fit_row_count
    if calibration_count <= 0:
        raise ValueError("train_disagreement_limit must leave at least one calibration row")

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

    train_source_margins = []
    train_source_entropies = []
    for row in train_rows:
        scores = behavior_gate._source_scores_for_row(row, tiny_cache)
        train_source_margins.append(float(sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1]))
        train_source_entropies.append(behavior_gate._entropy(scores))
    margin_edges = ecoc_gate._fit_bin_edges(train_source_margins, bins=4)
    entropy_edges = ecoc_gate._fit_bin_edges(train_source_entropies, bins=4)

    source_packets = [
        ecoc_gate._source_packet_from_scores(
            behavior_gate._source_scores_for_row(row, tiny_cache),
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
        )
        for row in rows
    ]
    qwen_packets = [
        ecoc_gate._source_packet_from_scores(
            behavior_gate._source_scores_for_row(row, qwen_cache),
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
        )
        for row in rows
    ]
    target_packets = [
        ecoc_gate._source_packet_from_scores(scores, margin_edges=margin_edges, entropy_edges=entropy_edges)
        for scores in target_scores
    ]

    train_features = np.asarray(
        [_packet_features(scores, packet) for scores, packet in zip(target_scores[: len(train_rows)], source_packets[: len(train_rows)], strict=True)],
        dtype=np.float64,
    )
    train_targets = _row_value_targets(train_rows, target_scores[: len(train_rows)], source_packets[: len(train_rows)])
    value_model = _fit_value_model(
        train_features,
        train_targets,
        fit_indices=np.arange(fit_row_count, dtype=np.int64),
        ridge=ridge,
    )
    defer_rule = _choose_defer_rule(
        train_rows[fit_row_count:],
        target_scores[fit_row_count : len(train_rows)],
        source_packets[fit_row_count : len(train_rows)],
        value_model=value_model,
    )

    prediction_rows: list[dict[str, Any]] = []
    eval_offset = len(train_rows)
    for eval_position, row in enumerate(test_rows):
        row_index = eval_offset + eval_position
        target = [float(score) for score in target_scores[row_index]]
        source_packet = source_packets[row_index]
        target_packet = target_packets[row_index]
        qwen_packet = qwen_packets[row_index]
        shuffled_index = eval_offset + ((eval_position + 1) % len(test_rows))
        shuffled_packet = source_packets[shuffled_index]
        raw_source_scores = behavior_gate._source_scores_for_row(row, tiny_cache)
        source_selected = behavior_gate._source_prediction_for_row(row, tiny_cache)
        source_order = sorted(range(len(raw_source_scores)), key=lambda index: (-float(raw_source_scores[index]), index))
        source_top2 = int(source_order[1]) if len(source_order) > 1 else int(source_order[0])

        def apply(packet: dict[str, Any]) -> tuple[list[float], bool, int, float]:
            return _apply_packet(
                target,
                packet,
                value_model=value_model,
                threshold=float(defer_rule["threshold"]),
                require_disagree=bool(defer_rule["require_disagree"]),
                require_parity=bool(defer_rule["require_parity"]),
            )

        packet_variants = {
            MATCHED_CONDITION: source_packet,
            "target_derived_packet": target_packet,
            "source_row_shuffle": shuffled_packet,
            "header_shuffle": ecoc_gate._mutate_packet(source_packet, header_from=shuffled_packet),
            "reliability_bin_shuffle": ecoc_gate._mutate_packet(source_packet, reliability_from=shuffled_packet),
            "ecoc_bit_shuffle": ecoc_gate._mutate_packet(source_packet, code_roll=1),
            "parity_flip": ecoc_gate._mutate_packet(source_packet, parity_flip=True),
            "candidate_roll": ecoc_gate._mutate_packet(source_packet, candidate_roll=1),
            "qwen_substituted_packet": qwen_packet,
        }
        condition_scores: dict[str, tuple[list[float], bool, int, float]] = {
            condition: apply(packet) for condition, packet in packet_variants.items()
        }
        condition_scores.update(
            {
                "target_only": (target, False, behavior_gate._prediction(target), 0.0),
                "zero_source": (target, False, behavior_gate._prediction(target), 0.0),
                "candidate_derangement": (
                    list(np.roll(condition_scores[MATCHED_CONDITION][0], 1)),
                    bool(condition_scores[MATCHED_CONDITION][1]),
                    int((condition_scores[MATCHED_CONDITION][2] + 1) % len(row.choices)),
                    float(condition_scores[MATCHED_CONDITION][3]),
                ),
                "packet_only_source_index": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    int(source_selected),
                    0.0,
                ),
                "top1_without_gate": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    int(source_selected),
                    0.0,
                ),
                "top2_without_gate": (
                    ecoc_gate._top2_without_header_scores(len(row.choices), source_selected, source_top2),
                    True,
                    int(source_selected),
                    0.0,
                ),
                "source_rank_control": (behavior_gate._source_rank_scores(raw_source_scores), True, int(source_selected), 0.0),
                "source_score_control": (
                    behavior_gate._centered_source_score_control(raw_source_scores),
                    True,
                    int(source_selected),
                    0.0,
                ),
                "source_score_quantized_control": (
                    ecoc_gate._source_score_quantized_control(raw_source_scores, bits=4),
                    True,
                    int(source_selected),
                    0.0,
                ),
                "same_byte_visible_text": (
                    same_byte_scores[eval_position],
                    False,
                    behavior_gate._prediction(same_byte_scores[eval_position]),
                    0.0,
                ),
            }
        )
        target_correct = behavior_gate._prediction(target) == int(row.answer_index)
        for condition in REPORT_CONDITIONS:
            scores, fired, decoded, predicted_gain = condition_scores[condition]
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
                    "decoded_candidate_index": int(decoded),
                    "decoded_candidate_label": row.choice_labels[int(decoded) % len(row.choice_labels)],
                    "predicted_packet_gain": float(predicted_gain),
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "source_packet": source_packet,
                    "defer_rule": defer_rule,
                    "control_origin": condition,
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed, bootstrap_samples=bootstrap_samples)
    oracle = _oracle_diagnostics(prediction_rows)
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
    packet_meta = {
        "kind": "innovation_defer_confidence_ecoc_packet",
        "codeword_bits": ecoc_gate.CODEWORD_BITS,
        "header_bits": ecoc_gate.HEADER_BITS,
        "packet_bits_per_row": ecoc_gate.PACKET_BITS_PER_ROW,
        "packet_bytes_per_row": ecoc_gate.PACKET_BITS_PER_ROW / 8.0,
        "framed_packet_bytes_per_row": int(math.ceil(ecoc_gate.PACKET_BITS_PER_ROW / 8.0)),
        "cache_line_bytes_per_row_64b": 64,
        "dma_bytes_per_row_128b": 128,
        "margin_edges": [float(edge) for edge in margin_edges],
        "entropy_edges": [float(edge) for edge in entropy_edges],
    }
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "source_private_arc_challenge_innovation_defer_packet_gate",
        "date": dt.date.today().isoformat(),
        "created_utc": created,
        "pass_gate": bool(strict_pass),
        "implementation_gate_only": False,
        "train_rows": int(len(train_rows)),
        "value_fit_rows": int(fit_row_count),
        "calibration_rows": int(calibration_count),
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
            "native_serving_throughput_measured": False,
            "packet_bytes_per_row": float(packet_meta["packet_bytes_per_row"]),
            "framed_packet_bytes_per_row": int(packet_meta["framed_packet_bytes_per_row"]),
            "cache_line_bytes_per_row_64b": int(packet_meta["cache_line_bytes_per_row_64b"]),
            "dma_bytes_per_row_128b": int(packet_meta["dma_bytes_per_row_128b"]),
            "sparse_packet_metadata": packet_meta,
            "note": (
                "Byte counts cover the source packet sideband only. The learned defer receiver is target-local "
                "logic, not native GPU throughput, HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "feature_metadata": {
            "source_packet": packet_meta,
            "value_model": {
                "ridge": float(value_model.ridge),
                "fit_mse": float(value_model.fit_mse),
                "fit_r2": float(value_model.fit_r2),
                "feature_dim": int(train_features.shape[1]),
                "fit_split": "first_half_validation_disagreement",
                "calibration_split": "second_half_validation_disagreement",
            },
            "selected_defer_rule": defer_rule,
            "target_score_metadata": target_score_meta,
            "same_byte_score_metadata": same_byte_meta,
            "source_score_cache_kind": "answer_key_forbidden_source_choice_scores",
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
            "target_model": str(target_model),
            "train_disagreement_limit": int(train_disagreement_limit),
            "test_disagreement_limit": int(test_disagreement_limit),
            "same_byte_budget": int(same_byte_budget),
        },
        "interpretation": (
            "This gate tests whether a target-local learned defer controller can decide when a fixed 2-byte "
            "source-private packet should override target-only behavior. It passes only if the learned packet "
            "beats target-only, target-derived, shuffled/header-destroyed, source-choice, source-score, same-byte "
            "text, and Qwen-substitution controls with positive paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_innovation_defer_packet_gate.json"
    md_path = output_dir / "arc_challenge_innovation_defer_packet_gate.md"
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
    parser.add_argument("--train-disagreement-limit", type=int, default=64)
    parser.add_argument("--test-disagreement-limit", type=int, default=64)
    parser.add_argument("--target-model", default=str(DEFAULT_QWEN3_MODEL))
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--ridge", type=float, default=3.0)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--same-byte-budget", type=int, default=4096)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=29)
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
        target_model=str(args.target_model),
        target_device=str(args.target_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        target_max_length=int(args.target_max_length),
        ridge=float(args.ridge),
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
