from __future__ import annotations

"""Strict ARC-Challenge confidence/ECOC Sparse Resonance Packet gate.

This gate tests the next branch after hidden-coordinate and always-on behavior
residual packets failed: a tiny source-private side-information packet with a
candidate codeword plus reliability header. The receiver applies the packet
only when the target model is uncertain and the packet passes a train-calibrated
harm-control rule.
"""

import argparse
import copy
import datetime as dt
import json
import math
import pathlib
import random
import statistics
import sys
from typing import Any, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_behavior_residual_packet_gate as behavior_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as soft_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402
from scripts import run_source_private_arc_openbookqa_soft_prefix_preflight as preflight  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_confidence_ecoc_packet_gate_20260504_"
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

MATCHED_CONDITION = "matched_confidence_ecoc_packet"
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
    "top1_without_header",
    "top2_without_header",
    "source_rank_control",
    "source_score_control",
    "source_score_quantized_control",
    "same_byte_visible_text",
    "qwen_substituted_packet",
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)
STRICT_REQUIRED_CONTROLS = CONTROL_CONDITIONS

CODEWORD_BITS = 8
HEADER_BITS = 8
PACKET_BITS_PER_ROW = CODEWORD_BITS + HEADER_BITS


def _fit_bin_edges(values: Sequence[float], *, bins: int) -> list[float]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    clean = np.asarray([float(value) for value in values], dtype=np.float64)
    if clean.size == 0:
        return [0.0 for _ in range(bins - 1)]
    quantiles = [100.0 * index / float(bins) for index in range(1, bins)]
    edges = [float(value) for value in np.percentile(clean, quantiles)]
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = float(np.nextafter(edges[index - 1], math.inf))
    return edges


def _bin_value(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="right"))


def _ecoc_codeword(candidate_index: int) -> tuple[int, ...]:
    codebook = (
        (1, 1, 1, 1, 0, 0, 0, 0),
        (1, 1, 0, 0, 1, 1, 0, 0),
        (1, 0, 1, 0, 1, 0, 1, 0),
        (1, 0, 0, 1, 0, 1, 1, 0),
        (0, 1, 1, 0, 1, 0, 0, 1),
        (0, 1, 0, 1, 0, 1, 0, 1),
        (0, 0, 1, 1, 0, 0, 1, 1),
        (0, 0, 0, 0, 1, 1, 1, 1),
    )
    return codebook[int(candidate_index) % len(codebook)]


def _packet_parity(*, codeword: Sequence[int], top2_index: int, margin_bin: int, entropy_bin: int) -> int:
    return int((sum(int(bit) for bit in codeword) + int(top2_index) + int(margin_bin) + int(entropy_bin)) % 2)


def _source_packet_from_scores(
    scores: Sequence[float],
    *,
    margin_edges: Sequence[float],
    entropy_edges: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty rank-1 sequence")
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else int(order[0])
    margin = float(values[top1] - values[top2]) if len(order) > 1 else 0.0
    entropy = behavior_gate._entropy(values)
    margin_bin = _bin_value(margin, margin_edges)
    entropy_bin = _bin_value(entropy, entropy_edges)
    codeword = _ecoc_codeword(top1)
    parity = _packet_parity(
        codeword=codeword,
        top2_index=top2,
        margin_bin=margin_bin,
        entropy_bin=entropy_bin,
    )
    return {
        "kind": "confidence_ecoc_packet",
        "candidate_count": int(values.size),
        "top1_index": int(top1),
        "top2_index": int(top2),
        "margin": float(margin),
        "entropy": float(entropy),
        "margin_bin": int(margin_bin),
        "entropy_bin": int(entropy_bin),
        "codeword": [int(bit) for bit in codeword],
        "parity": int(parity),
    }


def _decode_ecoc_candidate(packet: dict[str, Any]) -> int:
    candidate_count = int(packet["candidate_count"])
    codeword = tuple(int(bit) for bit in packet["codeword"])
    best = 0
    best_distance: int | None = None
    for candidate in range(candidate_count):
        target = _ecoc_codeword(candidate)
        distance = sum(int(left != right) for left, right in zip(codeword, target, strict=True))
        if best_distance is None or (distance, candidate) < (best_distance, best):
            best = int(candidate)
            best_distance = int(distance)
    return int(best)


def _parity_ok(packet: dict[str, Any]) -> bool:
    expected = _packet_parity(
        codeword=packet["codeword"],
        top2_index=int(packet["top2_index"]),
        margin_bin=int(packet["margin_bin"]),
        entropy_bin=int(packet["entropy_bin"]),
    )
    return int(packet["parity"]) == int(expected)


def _target_margin(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    return float(values[order[0]] - values[order[1]])


def _target_margin_edges(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
    *,
    bins: int,
) -> list[float]:
    return _fit_bin_edges([_target_margin(scores) for scores in target_scores[: len(rows)]], bins=bins)


def _gate_fires(packet: dict[str, Any], target_scores: Sequence[float], rule: dict[str, Any]) -> bool:
    if bool(rule.get("require_parity", True)) and not _parity_ok(packet):
        return False
    decoded = _decode_ecoc_candidate(packet)
    target_pred = behavior_gate._prediction(target_scores)
    if bool(rule.get("require_disagree", False)) and decoded == target_pred:
        return False
    target_margin = _target_margin(target_scores)
    target_uncertain = target_margin <= float(rule["max_target_margin"])
    source_confident = int(packet["margin_bin"]) >= int(rule["min_margin_bin"])
    source_low_entropy = int(packet["entropy_bin"]) <= int(rule["max_entropy_bin"])
    return bool(target_uncertain and source_confident and source_low_entropy)


def _force_candidate_scores(target_scores: Sequence[float], candidate_index: int) -> list[float]:
    scores = [float(score) for score in target_scores]
    if not scores:
        return scores
    selected = int(candidate_index) % len(scores)
    scores[selected] = max(scores) + max(1.0, abs(_target_margin(scores)) + 1.0)
    return scores


def _apply_packet(
    target_scores: Sequence[float],
    packet: dict[str, Any],
    rule: dict[str, Any],
) -> tuple[list[float], bool, int]:
    decoded = _decode_ecoc_candidate(packet)
    if _gate_fires(packet, target_scores, rule):
        return _force_candidate_scores(target_scores, decoded), True, int(decoded)
    return [float(score) for score in target_scores], False, int(decoded)


def _source_score_quantized_control(raw_scores: Sequence[float], *, bits: int) -> list[float]:
    values = np.asarray(raw_scores, dtype=np.float64)
    centered = values - float(values.mean()) if values.size else values
    scale = float(np.max(np.abs(centered))) if centered.size else 0.0
    levels = int((2 ** max(int(bits) - 1, 0)) - 1)
    if scale <= 1e-12 or levels < 1:
        return [0.0 for _ in centered]
    quantized = np.clip(np.rint(centered / (scale / float(levels))), -levels, levels)
    dequantized = quantized * (scale / float(levels))
    dequantized = dequantized - float(dequantized.mean())
    return [float(value) for value in dequantized]


def _top2_without_header_scores(choice_count: int, top1: int, top2: int) -> list[float]:
    scores = [0.0 for _ in range(choice_count)]
    scores[int(top1) % choice_count] = 1.0
    scores[int(top2) % choice_count] = 0.5
    return scores


def _mutate_packet(
    packet: dict[str, Any],
    *,
    code_roll: int = 0,
    candidate_roll: int = 0,
    header_from: dict[str, Any] | None = None,
    reliability_from: dict[str, Any] | None = None,
    parity_flip: bool = False,
) -> dict[str, Any]:
    out = copy.deepcopy(packet)
    if candidate_roll:
        count = int(out["candidate_count"])
        out["top1_index"] = int((int(out["top1_index"]) + int(candidate_roll)) % count)
        out["top2_index"] = int((int(out["top2_index"]) + int(candidate_roll)) % count)
        out["codeword"] = [int(bit) for bit in _ecoc_codeword(int(out["top1_index"]))]
    if code_roll:
        bits = list(out["codeword"])
        shift = int(code_roll) % len(bits)
        out["codeword"] = bits[shift:] + bits[:shift]
    if header_from is not None:
        for key in ("top2_index", "margin_bin", "entropy_bin", "margin", "entropy"):
            out[key] = copy.deepcopy(header_from[key])
    if reliability_from is not None:
        for key in ("margin_bin", "entropy_bin", "margin", "entropy"):
            out[key] = copy.deepcopy(reliability_from[key])
    out["parity"] = _packet_parity(
        codeword=out["codeword"],
        top2_index=int(out["top2_index"]),
        margin_bin=int(out["margin_bin"]),
        entropy_bin=int(out["entropy_bin"]),
    )
    if parity_flip:
        out["parity"] = int(1 - int(out["parity"]))
    return out


def _choose_gate_rule(
    rows: Sequence[arc_gate.ArcRow],
    target_scores: Sequence[Sequence[float]],
    packets: Sequence[dict[str, Any]],
    *,
    target_margin_edges: Sequence[float],
) -> dict[str, Any]:
    target_thresholds = [0.0, *[float(edge) for edge in target_margin_edges]]
    candidates: list[dict[str, Any]] = []
    max_margin_bin = max(int(packet["margin_bin"]) for packet in packets[: len(rows)]) if rows else 0
    max_entropy_bin = max(int(packet["entropy_bin"]) for packet in packets[: len(rows)]) if rows else 0
    for min_margin_bin in range(max_margin_bin + 1):
        for max_entropy_bin_value in range(max_entropy_bin + 1):
            for max_target_margin in target_thresholds:
                for require_disagree in (False, True):
                    candidates.append(
                        {
                            "min_margin_bin": int(min_margin_bin),
                            "max_entropy_bin": int(max_entropy_bin_value),
                            "max_target_margin": float(max_target_margin),
                            "require_disagree": bool(require_disagree),
                            "require_parity": True,
                        }
                    )
    candidates.append(
        {
            "min_margin_bin": 999,
            "max_entropy_bin": -1,
            "max_target_margin": -1.0,
            "require_disagree": False,
            "require_parity": True,
        }
    )
    best: dict[str, Any] | None = None
    for rule in candidates:
        correct = 0
        fired = 0
        helped = 0
        harmed = 0
        margins: list[float] = []
        for row, scores, packet in zip(rows, target_scores, packets, strict=True):
            target_pred = behavior_gate._prediction(scores)
            target_correct = target_pred == int(row.answer_index)
            fused, did_fire, _decoded = _apply_packet(scores, packet, rule)
            pred = behavior_gate._prediction(fused)
            is_correct = pred == int(row.answer_index)
            correct += int(is_correct)
            fired += int(did_fire)
            helped += int(did_fire and is_correct and not target_correct)
            harmed += int(did_fire and (not is_correct) and target_correct)
            margins.append(behavior_gate._margin(fused, row.answer_index))
        accuracy = correct / max(len(rows), 1)
        fired_rate = fired / max(len(rows), 1)
        net_help = helped - harmed
        row_result = {
            **rule,
            "train_accuracy": float(accuracy),
            "train_fired": int(fired),
            "train_fired_rate": float(fired_rate),
            "train_helped": int(helped),
            "train_harmed": int(harmed),
            "train_net_help": int(net_help),
            "train_mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
        }
        key = (
            row_result["train_accuracy"],
            row_result["train_net_help"],
            -row_result["train_harmed"],
            row_result["train_helped"],
            -abs(row_result["train_fired_rate"] - 0.35),
            row_result["train_mean_margin"],
        )
        if best is None:
            best = row_result
            best_key = key
        elif key > best_key:
            best = row_result
            best_key = key
    if best is None:
        raise ValueError("could not select confidence gate rule")
    return best


def _oracle_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    source_condition = "packet_only_source_index"
    qwen_condition = "qwen_substituted_packet"
    target_condition = "target_only"
    rows = [
        group for group in by_id.values() if target_condition in group and source_condition in group and qwen_condition in group
    ]
    target_correct = sum(1 for group in rows if group[target_condition]["correct"])
    source_correct = sum(1 for group in rows if group[source_condition]["correct"])
    qwen_correct = sum(1 for group in rows if group[qwen_condition]["correct"])
    source_helpable = sum(
        1 for group in rows if (not group[target_condition]["correct"]) and group[source_condition]["correct"]
    )
    source_harm_risk = sum(
        1 for group in rows if group[target_condition]["correct"] and (not group[source_condition]["correct"])
    )
    source_or_target = sum(
        1 for group in rows if group[target_condition]["correct"] or group[source_condition]["correct"]
    )
    qwen_or_target = sum(1 for group in rows if group[target_condition]["correct"] or group[qwen_condition]["correct"])
    n = max(len(rows), 1)
    return {
        "n": int(len(rows)),
        "target_only_correct": int(target_correct),
        "source_top1_correct": int(source_correct),
        "qwen_substitution_correct": int(qwen_correct),
        "source_helpable_rows": int(source_helpable),
        "source_harm_risk_rows": int(source_harm_risk),
        "source_or_target_oracle_correct": int(source_or_target),
        "qwen_or_target_oracle_correct": int(qwen_or_target),
        "target_only_accuracy": float(target_correct / n),
        "source_top1_accuracy": float(source_correct / n),
        "qwen_substitution_accuracy": float(qwen_correct / n),
        "source_or_target_oracle_accuracy": float(source_or_target / n),
        "qwen_or_target_oracle_accuracy": float(qwen_or_target / n),
    }


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
        "# ARC-Challenge Confidence/ECOC Packet Gate",
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
    target_model: str,
    target_device: str,
    target_attn_implementation: str | None,
    dtype: str,
    target_max_length: int,
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
    margin_edges = _fit_bin_edges(train_source_margins, bins=4)
    entropy_edges = _fit_bin_edges(train_source_entropies, bins=4)
    target_edges = _target_margin_edges(train_rows, target_scores[:fit_row_count], bins=4)

    source_packets = [
        _source_packet_from_scores(
            behavior_gate._source_scores_for_row(row, tiny_cache),
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
        )
        for row in rows
    ]
    qwen_packets = [
        _source_packet_from_scores(
            behavior_gate._source_scores_for_row(row, qwen_cache),
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
        )
        for row in rows
    ]
    target_packets = [
        _source_packet_from_scores(scores, margin_edges=margin_edges, entropy_edges=entropy_edges)
        for scores in target_scores
    ]
    gate_rule = _choose_gate_rule(
        train_rows,
        target_scores[:fit_row_count],
        source_packets[:fit_row_count],
        target_margin_edges=target_edges,
    )

    prediction_rows: list[dict[str, Any]] = []
    eval_offset = fit_row_count
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

        packet_variants = {
            MATCHED_CONDITION: source_packet,
            "target_derived_packet": target_packet,
            "source_row_shuffle": shuffled_packet,
            "header_shuffle": _mutate_packet(source_packet, header_from=shuffled_packet),
            "reliability_bin_shuffle": _mutate_packet(source_packet, reliability_from=shuffled_packet),
            "ecoc_bit_shuffle": _mutate_packet(source_packet, code_roll=1),
            "parity_flip": _mutate_packet(source_packet, parity_flip=True),
            "candidate_roll": _mutate_packet(source_packet, candidate_roll=1),
            "qwen_substituted_packet": qwen_packet,
        }
        condition_scores: dict[str, tuple[list[float], bool, int]] = {}
        for condition, packet in packet_variants.items():
            condition_scores[condition] = _apply_packet(target, packet, gate_rule)
        condition_scores.update(
            {
                "target_only": (target, False, behavior_gate._prediction(target)),
                "zero_source": (target, False, behavior_gate._prediction(target)),
                "candidate_derangement": (
                    list(np.roll(condition_scores[MATCHED_CONDITION][0], 1)),
                    bool(condition_scores[MATCHED_CONDITION][1]),
                    int((condition_scores[MATCHED_CONDITION][2] + 1) % len(row.choices)),
                ),
                "packet_only_source_index": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    int(source_selected),
                ),
                "top1_without_header": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    int(source_selected),
                ),
                "top2_without_header": (
                    _top2_without_header_scores(len(row.choices), source_selected, source_top2),
                    True,
                    int(source_selected),
                ),
                "source_rank_control": (behavior_gate._source_rank_scores(raw_source_scores), True, int(source_selected)),
                "source_score_control": (
                    behavior_gate._centered_source_score_control(raw_source_scores),
                    True,
                    int(source_selected),
                ),
                "source_score_quantized_control": (
                    _source_score_quantized_control(raw_source_scores, bits=4),
                    True,
                    int(source_selected),
                ),
                "same_byte_visible_text": (
                    same_byte_scores[eval_position],
                    False,
                    behavior_gate._prediction(same_byte_scores[eval_position]),
                ),
            }
        )
        target_correct = behavior_gate._prediction(target) == int(row.answer_index)
        for condition in REPORT_CONDITIONS:
            scores, fired, decoded = condition_scores[condition]
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
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "source_packet": source_packet,
                    "gate_rule": gate_rule,
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
        "kind": "confidence_ecoc_packet",
        "codeword_bits": CODEWORD_BITS,
        "header_bits": HEADER_BITS,
        "packet_bits_per_row": PACKET_BITS_PER_ROW,
        "packet_bytes_per_row": PACKET_BITS_PER_ROW / 8.0,
        "framed_packet_bytes_per_row": int(math.ceil(PACKET_BITS_PER_ROW / 8.0)),
        "cache_line_bytes_per_row_64b": 64,
        "dma_bytes_per_row_128b": 128,
        "margin_edges": [float(edge) for edge in margin_edges],
        "entropy_edges": [float(edge) for edge in entropy_edges],
        "target_margin_edges": [float(edge) for edge in target_edges],
    }
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "source_private_arc_challenge_confidence_ecoc_packet_gate",
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
            "native_serving_throughput_measured": False,
            "packet_bytes_per_row": float(packet_meta["packet_bytes_per_row"]),
            "framed_packet_bytes_per_row": int(packet_meta["framed_packet_bytes_per_row"]),
            "cache_line_bytes_per_row_64b": int(packet_meta["cache_line_bytes_per_row_64b"]),
            "dma_bytes_per_row_128b": int(packet_meta["dma_bytes_per_row_128b"]),
            "sparse_packet_metadata": packet_meta,
            "note": (
                "Byte counts cover the 8-bit candidate codeword plus 8-bit reliability/header sideband only. "
                "They are not native GPU throughput, HBM traffic, or an end-to-end serving measurement."
            ),
        },
        "feature_metadata": {
            "source_packet": packet_meta,
            "selected_gate_rule": gate_rule,
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
            "This gate tests whether a 2-byte confidence/ECOC side-information packet can safely improve target "
            "answers only on rows where the target is uncertain and the source packet is reliable. It passes only "
            "if the selective packet beats target-only, target-derived, shuffled/header-destroyed, source-choice, "
            "source-score, same-byte text, and Qwen-substitution controls with positive paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_confidence_ecoc_packet_gate.json"
    md_path = output_dir / "arc_challenge_confidence_ecoc_packet_gate.md"
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
    parser.add_argument("--train-disagreement-limit", type=int, default=16)
    parser.add_argument("--test-disagreement-limit", type=int, default=16)
    parser.add_argument("--target-model", default=str(DEFAULT_QWEN3_MODEL))
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--same-byte-budget", type=int, default=4096)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-ci-low", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=23)
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
