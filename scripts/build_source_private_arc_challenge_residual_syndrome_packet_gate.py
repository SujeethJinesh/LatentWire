from __future__ import annotations

"""Strict ARC-Challenge residual/syndrome Sparse Resonance Packet gate.

This gate tests the next source-private branch after candidate-identity ECOC
and learned defer packets failed.  The source sends a small syndrome over
pairwise answer-preference bits plus a reliability header; the receiver decodes
that syndrome using its own target scores as side information and applies only
the decoded source/target pairwise residual.
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
    "results/source_private_arc_challenge_residual_syndrome_packet_gate_20260504_"
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

MATCHED_CONDITION = "matched_residual_syndrome_packet"
CONTROL_CONDITIONS = (
    "target_only",
    "target_derived_packet",
    "zero_source",
    "source_row_shuffle",
    "header_shuffle",
    "syndrome_bit_shuffle",
    "parity_flip",
    "wrong_parity_matrix",
    "target_side_info_removed",
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

MARGIN_BINS = 4
ENTROPY_BINS = 4


def _pair_indices(candidate_count: int) -> list[tuple[int, int]]:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    return [(left, right) for left in range(candidate_count) for right in range(left + 1, candidate_count)]


def _pair_bits(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.asarray([int(values[left] >= values[right]) for left, right in _pair_indices(int(values.size))], dtype=np.int8)


def _pair_flip_costs(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    diffs = np.asarray([abs(float(values[left] - values[right])) for left, right in _pair_indices(int(values.size))], dtype=np.float64)
    scale = float(np.median(diffs)) if diffs.size else 0.0
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return 1.0 + diffs / scale


def _parity_matrix(bit_count: int, syndrome_bits: int, *, salt: int = 0) -> np.ndarray:
    if bit_count < 1:
        raise ValueError("bit_count must be positive")
    if syndrome_bits < 1:
        raise ValueError("syndrome_bits must be positive")
    rows = int(syndrome_bits)
    cols = int(bit_count)
    matrix = np.zeros((rows, cols), dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            value = ((row + 1) * 17 + (col + 3) * 31 + (row + col + 5) * 7 + int(salt) * 43) % 11
            matrix[row, col] = int(value in {0, 1, 3, 7, 9})
    for row in range(rows):
        if int(matrix[row].sum()) == 0:
            matrix[row, row % cols] = 1
    for col in range(cols):
        if int(matrix[:, col].sum()) == 0:
            matrix[col % rows, col] = 1
    return matrix


def _syndrome(bits: Sequence[int], matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.int8)
    if matrix.shape[1] != values.size:
        raise ValueError("parity matrix and bit vector shape mismatch")
    return np.asarray((matrix.astype(np.int64) @ values.astype(np.int64)) % 2, dtype=np.int8)


def _packet_parity(*, syndrome_bits: Sequence[int], margin_bin: int, entropy_bin: int, candidate_count: int) -> int:
    return int((sum(int(bit) for bit in syndrome_bits) + int(margin_bin) + int(entropy_bin) + int(candidate_count)) % 2)


def _source_packet_from_scores(
    scores: Sequence[float],
    *,
    margin_edges: Sequence[float],
    entropy_edges: Sequence[float],
    syndrome_bits: int,
    parity_salt: int = 0,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("scores must be a rank-1 sequence with at least two candidates")
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    margin = float(values[order[0]] - values[order[1]])
    entropy = behavior_gate._entropy(values)
    pair_bits = _pair_bits(values)
    matrix = _parity_matrix(int(pair_bits.size), syndrome_bits, salt=parity_salt)
    packet_syndrome = _syndrome(pair_bits, matrix)
    margin_bin = ecoc_gate._bin_value(margin, margin_edges)
    entropy_bin = ecoc_gate._bin_value(entropy, entropy_edges)
    parity = _packet_parity(
        syndrome_bits=packet_syndrome,
        margin_bin=margin_bin,
        entropy_bin=entropy_bin,
        candidate_count=int(values.size),
    )
    return {
        "kind": "pairwise_residual_syndrome_packet",
        "candidate_count": int(values.size),
        "pair_bit_count": int(pair_bits.size),
        "syndrome_bit_count": int(syndrome_bits),
        "syndrome": [int(bit) for bit in packet_syndrome],
        "margin": float(margin),
        "entropy": float(entropy),
        "margin_bin": int(margin_bin),
        "entropy_bin": int(entropy_bin),
        "parity": int(parity),
    }


def _zero_packet(candidate_count: int, *, syndrome_bits: int) -> dict[str, Any]:
    syndrome = [0 for _ in range(int(syndrome_bits))]
    return {
        "kind": "pairwise_residual_syndrome_packet",
        "candidate_count": int(candidate_count),
        "pair_bit_count": len(_pair_indices(int(candidate_count))),
        "syndrome_bit_count": int(syndrome_bits),
        "syndrome": syndrome,
        "margin": 0.0,
        "entropy": 0.0,
        "margin_bin": 0,
        "entropy_bin": 0,
        "parity": _packet_parity(syndrome_bits=syndrome, margin_bin=0, entropy_bin=0, candidate_count=int(candidate_count)),
    }


def _parity_ok(packet: dict[str, Any]) -> bool:
    expected = _packet_parity(
        syndrome_bits=packet["syndrome"],
        margin_bin=int(packet["margin_bin"]),
        entropy_bin=int(packet["entropy_bin"]),
        candidate_count=int(packet["candidate_count"]),
    )
    return int(packet["parity"]) == int(expected)


def _enumerate_binary_patterns(bit_count: int) -> np.ndarray:
    if bit_count > 20:
        raise ValueError("bit_count too large for exhaustive syndrome decode")
    total = 1 << int(bit_count)
    values = np.arange(total, dtype=np.uint32)[:, None]
    shifts = np.arange(int(bit_count), dtype=np.uint32)[None, :]
    return ((values >> shifts) & 1).astype(np.int8)


def _decode_pair_bits(
    packet: dict[str, Any],
    target_scores: Sequence[float],
    *,
    parity_salt: int = 0,
    use_target_side_info: bool = True,
) -> np.ndarray:
    candidate_count = int(packet["candidate_count"])
    bit_count = int(packet["pair_bit_count"])
    matrix = _parity_matrix(bit_count, int(packet["syndrome_bit_count"]), salt=parity_salt)
    patterns = _enumerate_binary_patterns(bit_count)
    syndrome = np.asarray(packet["syndrome"], dtype=np.int8)
    mask = np.all((patterns.astype(np.int64) @ matrix.T.astype(np.int64)) % 2 == syndrome[None, :], axis=1)
    candidates = patterns[mask]
    if candidates.size == 0:
        return _pair_bits(target_scores)
    if use_target_side_info:
        prior = _pair_bits(target_scores)
        costs = _pair_flip_costs(target_scores)
    else:
        prior = np.zeros(bit_count, dtype=np.int8)
        costs = np.ones(bit_count, dtype=np.float64)
    distances = ((candidates != prior[None, :]).astype(np.float64) * costs[None, :]).sum(axis=1)
    best = int(np.argmin(distances))
    decoded = candidates[best]
    if len(_pair_indices(candidate_count)) != decoded.size:
        raise ValueError("decoded bit count mismatch")
    return np.asarray(decoded, dtype=np.int8)


def _pair_residual_votes(decoded_bits: Sequence[int], target_scores: Sequence[float]) -> np.ndarray:
    target_bits = _pair_bits(target_scores)
    decoded = np.asarray(decoded_bits, dtype=np.int8)
    candidate_count = len(target_scores)
    votes = np.zeros(candidate_count, dtype=np.float64)
    for bit, target_bit, (left, right) in zip(decoded, target_bits, _pair_indices(candidate_count), strict=True):
        if int(bit) == int(target_bit):
            continue
        if int(bit):
            votes[left] += 1.0
            votes[right] -= 1.0
        else:
            votes[right] += 1.0
            votes[left] -= 1.0
    return votes - float(votes.mean()) if votes.size else votes


def _target_margin(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    return float(values[order[0]] - values[order[1]])


def _gate_fires(packet: dict[str, Any], target_scores: Sequence[float], rule: dict[str, Any]) -> bool:
    if bool(rule.get("require_parity", True)) and not _parity_ok(packet):
        return False
    if int(packet["margin_bin"]) < int(rule["min_margin_bin"]):
        return False
    if int(packet["entropy_bin"]) > int(rule["max_entropy_bin"]):
        return False
    if _target_margin(target_scores) > float(rule["max_target_margin"]):
        return False
    if bool(rule.get("require_innovation", False)):
        decoded = _decode_pair_bits(packet, target_scores)
        if not np.any(decoded != _pair_bits(target_scores)):
            return False
    return True


def _apply_packet(
    target_scores: Sequence[float],
    packet: dict[str, Any],
    rule: dict[str, Any],
    *,
    parity_salt: int = 0,
    use_target_side_info: bool = True,
) -> tuple[list[float], bool, list[int]]:
    decoded = _decode_pair_bits(
        packet,
        target_scores,
        parity_salt=parity_salt,
        use_target_side_info=use_target_side_info,
    )
    if not _gate_fires(packet, target_scores, rule):
        return [float(score) for score in target_scores], False, [int(bit) for bit in decoded]
    residual = _pair_residual_votes(decoded, target_scores)
    fused = np.asarray(target_scores, dtype=np.float64) + float(rule["residual_weight"]) * residual
    return [float(score) for score in fused], True, [int(bit) for bit in decoded]


def _mutate_packet(
    packet: dict[str, Any],
    *,
    syndrome_roll: int = 0,
    header_from: dict[str, Any] | None = None,
    parity_flip: bool = False,
) -> dict[str, Any]:
    out = copy.deepcopy(packet)
    if syndrome_roll:
        bits = list(out["syndrome"])
        shift = int(syndrome_roll) % len(bits)
        out["syndrome"] = bits[shift:] + bits[:shift]
    if header_from is not None:
        for key in ("margin_bin", "entropy_bin", "margin", "entropy"):
            out[key] = copy.deepcopy(header_from[key])
    out["parity"] = _packet_parity(
        syndrome_bits=out["syndrome"],
        margin_bin=int(out["margin_bin"]),
        entropy_bin=int(out["entropy_bin"]),
        candidate_count=int(out["candidate_count"]),
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
    max_margin_bin = max(int(packet["margin_bin"]) for packet in packets[: len(rows)]) if rows else 0
    max_entropy_bin = max(int(packet["entropy_bin"]) for packet in packets[: len(rows)]) if rows else 0
    target_thresholds = [float("inf"), *[float(edge) for edge in target_margin_edges]]
    candidates: list[dict[str, Any]] = []
    for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        for min_margin_bin in range(max_margin_bin + 1):
            for max_entropy_bin_value in range(max_entropy_bin + 1):
                for max_target_margin in target_thresholds:
                    for require_innovation in (False, True):
                        candidates.append(
                            {
                                "residual_weight": float(weight),
                                "min_margin_bin": int(min_margin_bin),
                                "max_entropy_bin": int(max_entropy_bin_value),
                                "max_target_margin": float(max_target_margin),
                                "require_innovation": bool(require_innovation),
                                "require_parity": True,
                            }
                        )
    candidates.append(
        {
            "residual_weight": 0.0,
            "min_margin_bin": 999,
            "max_entropy_bin": -1,
            "max_target_margin": -1.0,
            "require_innovation": True,
            "require_parity": True,
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
        row_result = {
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
            row_result["train_accuracy"],
            row_result["train_net_help"],
            -row_result["train_harmed"],
            row_result["train_helped"],
            -abs(row_result["train_fired_rate"] - 0.35),
            row_result["train_mean_margin"],
            -row_result["residual_weight"],
        )
        if best is None or best_key is None or key > best_key:
            best = row_result
            best_key = key
    if best is None:
        raise ValueError("could not select residual syndrome gate rule")
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


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["strict_headline"]
    lines = [
        "# ARC-Challenge Residual/Syndrome Packet Gate",
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
    syndrome_bits: int,
    local_files_only: bool,
    bootstrap_samples: int,
    same_byte_budget: int | None,
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

    train_source_margins = []
    train_source_entropies = []
    for row in train_rows:
        scores = behavior_gate._source_scores_for_row(row, tiny_cache)
        order = sorted(scores, reverse=True)
        train_source_margins.append(float(order[0] - order[1]))
        train_source_entropies.append(behavior_gate._entropy(scores))
    margin_edges = ecoc_gate._fit_bin_edges(train_source_margins, bins=MARGIN_BINS)
    entropy_edges = ecoc_gate._fit_bin_edges(train_source_entropies, bins=ENTROPY_BINS)
    target_margin_edges = ecoc_gate._fit_bin_edges(
        [_target_margin(scores) for scores in target_scores[:fit_row_count]],
        bins=4,
    )

    source_packets = [
        _source_packet_from_scores(
            behavior_gate._source_scores_for_row(row, tiny_cache),
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
            syndrome_bits=syndrome_bits,
        )
        for row in rows
    ]
    qwen_packets = [
        _source_packet_from_scores(
            behavior_gate._source_scores_for_row(row, qwen_cache),
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
            syndrome_bits=syndrome_bits,
        )
        for row in rows
    ]
    target_packets = [
        _source_packet_from_scores(
            scores,
            margin_edges=margin_edges,
            entropy_edges=entropy_edges,
            syndrome_bits=syndrome_bits,
        )
        for scores in target_scores
    ]

    gate_rule = _choose_gate_rule(
        train_rows,
        target_scores[:fit_row_count],
        source_packets[:fit_row_count],
        target_margin_edges=target_margin_edges,
    )
    header_bits = int(math.ceil(math.log2(MARGIN_BINS)) + math.ceil(math.log2(ENTROPY_BINS)) + 1)
    packet_bits_per_row = int(syndrome_bits) + header_bits
    framed_packet_bytes = int(math.ceil(packet_bits_per_row / 8.0))
    visible_text_budget = int(same_byte_budget) if same_byte_budget is not None else framed_packet_bytes

    def same_byte_prompt(row: arc_gate.ArcRow) -> str:
        selected = behavior_gate._source_prediction_for_row(row, tiny_cache)
        hint = row.choices[selected].encode("utf-8")[:visible_text_budget].decode("utf-8", errors="ignore")
        choices = "\n".join(f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True))
        return (
            "Answer the science question with the best answer.\n"
            f"Question: {row.question}\n"
            f"Choices:\n{choices}\n"
            f"Source model selected this byte-limited visible hint: {hint}\n"
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

        packet_variants = {
            MATCHED_CONDITION: (source_packet, {}, True),
            "target_derived_packet": (target_packet, {}, True),
            "zero_source": (_zero_packet(len(row.choices), syndrome_bits=syndrome_bits), {}, True),
            "source_row_shuffle": (shuffled_packet, {}, True),
            "header_shuffle": (_mutate_packet(source_packet, header_from=shuffled_packet), {}, True),
            "syndrome_bit_shuffle": (_mutate_packet(source_packet, syndrome_roll=1), {}, True),
            "parity_flip": (_mutate_packet(source_packet, parity_flip=True), {}, True),
            "wrong_parity_matrix": (source_packet, {"parity_salt": 1}, True),
            "target_side_info_removed": (source_packet, {}, False),
            "qwen_substituted_packet": (qwen_packet, {}, True),
        }
        condition_scores: dict[str, tuple[list[float], bool, list[int]]] = {}
        for condition, (packet, decode_kwargs, use_target_side_info) in packet_variants.items():
            condition_scores[condition] = _apply_packet(
                target,
                packet,
                gate_rule,
                parity_salt=int(decode_kwargs.get("parity_salt", 0)),
                use_target_side_info=bool(use_target_side_info),
            )
        matched_scores = condition_scores[MATCHED_CONDITION][0]
        condition_scores.update(
            {
                "target_only": (target, False, [int(bit) for bit in _pair_bits(target)]),
                "candidate_roll": (
                    list(np.roll(matched_scores, 1)),
                    bool(condition_scores[MATCHED_CONDITION][1]),
                    list(condition_scores[MATCHED_CONDITION][2]),
                ),
                "candidate_derangement": (
                    list(np.roll(matched_scores, -1)),
                    bool(condition_scores[MATCHED_CONDITION][1]),
                    list(condition_scores[MATCHED_CONDITION][2]),
                ),
                "packet_only_source_index": (
                    behavior_gate._source_index_scores(len(row.choices), source_selected),
                    True,
                    [int(bit) for bit in _pair_bits(raw_source_scores)],
                ),
                "source_rank_control": (
                    behavior_gate._source_rank_scores(raw_source_scores),
                    True,
                    [int(bit) for bit in _pair_bits(raw_source_scores)],
                ),
                "source_score_control": (
                    behavior_gate._centered_source_score_control(raw_source_scores),
                    True,
                    [int(bit) for bit in _pair_bits(raw_source_scores)],
                ),
                "source_score_quantized_control": (
                    ecoc_gate._source_score_quantized_control(raw_source_scores, bits=4),
                    True,
                    [int(bit) for bit in _pair_bits(raw_source_scores)],
                ),
                "same_byte_visible_text": (
                    same_byte_scores[eval_position],
                    False,
                    [int(bit) for bit in _pair_bits(same_byte_scores[eval_position])],
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
                    "decoded_pair_bits": [int(bit) for bit in decoded],
                    "source_selected_index": int(source_selected),
                    "source_selected_label": row.choice_labels[source_selected],
                    "source_packet": source_packet,
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
    packet_meta = {
        "kind": "pairwise_residual_syndrome_packet",
        "syndrome_bits": int(syndrome_bits),
        "header_bits": int(header_bits),
        "packet_bits_per_row": int(packet_bits_per_row),
        "packet_bytes_per_row": float(packet_bits_per_row / 8.0),
        "framed_packet_bytes_per_row": int(framed_packet_bytes),
        "cache_line_bytes_per_row_64b": int(math.ceil(max(framed_packet_bytes, 1) / 64.0) * 64),
        "dma_bytes_per_row_128b": int(math.ceil(max(framed_packet_bytes, 1) / 128.0) * 128),
        "margin_edges": [float(edge) for edge in margin_edges],
        "entropy_edges": [float(edge) for edge in entropy_edges],
        "target_margin_edges": [float(edge) for edge in target_margin_edges],
        "decode_enumeration_patterns_max": int(2 ** max(len(_pair_indices(len(row.choices))) for row in rows)),
        "decode_flops_proxy_per_row": int((2 ** max(len(_pair_indices(len(row.choices))) for row in rows)) * int(syndrome_bits)),
    }
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "gate": "source_private_arc_challenge_residual_syndrome_packet_gate",
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
            "packet_bytes_per_row": float(packet_meta["packet_bytes_per_row"]),
            "framed_packet_bytes_per_row": int(packet_meta["framed_packet_bytes_per_row"]),
            "cache_line_bytes_per_row_64b": int(packet_meta["cache_line_bytes_per_row_64b"]),
            "dma_bytes_per_row_128b": int(packet_meta["dma_bytes_per_row_128b"]),
            "decode_flops_proxy_per_row": int(packet_meta["decode_flops_proxy_per_row"]),
            "sparse_packet_metadata": packet_meta,
            "note": (
                "Byte counts cover the syndrome plus reliability/check header only. They are not native GPU "
                "throughput, HBM traffic, or an end-to-end serving measurement."
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
            "same_byte_budget": int(visible_text_budget),
        },
        "interpretation": (
            "This gate tests whether a fixed-byte pairwise residual/syndrome packet can transmit source "
            "innovation without exposing source text, KV, hidden states, or raw score vectors. It passes only "
            "if the decoded packet beats target-only, target-derived, wrong-row, syndrome-destroyed, "
            "source-choice/source-score, same-byte visible text, and Qwen-substitution controls with positive "
            "paired uncertainty."
        ),
    }
    json_path = output_dir / "arc_challenge_residual_syndrome_packet_gate.json"
    md_path = output_dir / "arc_challenge_residual_syndrome_packet_gate.md"
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
    parser.add_argument("--train-disagreement-limit", type=int, default=32)
    parser.add_argument("--test-disagreement-limit", type=int, default=32)
    parser.add_argument("--target-model", default=str(DEFAULT_QWEN3_MODEL))
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--target-attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--target-max-length", type=int, default=192)
    parser.add_argument("--syndrome-bits", type=int, default=4)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--same-byte-budget", type=int, default=None)
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
        syndrome_bits=int(args.syndrome_bits),
        local_files_only=str(args.local_files_only).lower() == "true",
        bootstrap_samples=int(args.bootstrap_samples),
        same_byte_budget=args.same_byte_budget,
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_ci_low=float(args.min_ci_low),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
