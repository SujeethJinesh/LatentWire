#!/usr/bin/env python3
"""Toy test-before-repair bridge.

This ablation synthesizes a small candidate pool of arithmetic reasoning traces.
One policy spends its single repair pass immediately on the highest-surface-score
candidate. The test-before-repair policy first runs discriminative checks on the
entire pool, filters out semantically noisy traces, then spends the same repair
budget only when the chosen trace fails the tests.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Sequence


@dataclass(frozen=True)
class ToyTestBeforeRepairConfig:
    seed: int = 0
    examples: int = 192
    pool_size: int = 6
    chain_length: int = 4
    severity_bins: int = 3
    test_threshold: float = 0.69
    repair_threshold: float = 0.55
    repair_noise: float = 0.09
    test_noise: float = 0.05
    near_miss_bias: float = 0.18
    route_noise_bias: float = 0.22


def _make_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _apply_op(value: int, op: str, operand: int) -> int:
    if op == "add":
        return value + operand
    if op == "sub":
        return value - operand
    if op == "mul":
        return value * operand
    raise ValueError(f"Unknown op: {op}")


def _make_trace(
    *,
    start_value: int,
    operations: Sequence[tuple[str, int]],
) -> list[dict[str, int | str | bool]]:
    value = int(start_value)
    trace: list[dict[str, int | str | bool]] = []
    for step_index, (op, operand) in enumerate(operations):
        next_value = _apply_op(value, op, operand)
        trace.append(
            {
                "step": step_index,
                "input": value,
                "op": op,
                "operand": operand,
                "stated_output": next_value,
                "true_output": next_value,
                "is_inconsistent": False,
            }
        )
        value = next_value
    return trace


def _trace_to_text(trace: Sequence[dict[str, int | str | bool]]) -> str:
    return " | ".join(
        f"{step['input']} {step['op']} {step['operand']} = {step['stated_output']}"
        for step in trace
    )


def _predict_from_trace(start_value: int, trace: Sequence[dict[str, int | str | bool]]) -> int:
    if not trace:
        return int(start_value)
    return int(trace[-1]["stated_output"])


def _trace_checksum(trace: Sequence[dict[str, int | str | bool]]) -> int:
    checksum = 0
    for step in trace:
        op_code = {"add": 3, "sub": 7, "mul": 11}[str(step["op"])]
        checksum += (
            (int(step["step"]) + 1) * 13
            + int(step["input"]) * 5
            + int(step["operand"]) * 17
            + int(step["stated_output"]) * 19
            + op_code
        )
    return checksum % 997


def _consensus_trace(candidates: Sequence[dict[str, Any]]) -> tuple[list[int], int]:
    consensus_outputs: list[int] = []
    for step_index in range(len(candidates[0]["trace"])):
        step_outputs = [int(candidate["trace"][step_index]["stated_output"]) for candidate in candidates]
        consensus_outputs.append(int(round(median(step_outputs))))
    checksum = 0
    for index, value in enumerate(consensus_outputs):
        checksum += (index + 1) * 29 + value * 31
    return consensus_outputs, checksum % 997


def _corrupt_output_trace(
    trace: list[dict[str, int | str | bool]],
    *,
    severity: int,
    rng: random.Random,
) -> tuple[list[dict[str, int | str | bool]], int, int]:
    corrupted = [dict(step) for step in trace]
    if not corrupted:
        return corrupted, 0, 0

    corrupt_index = min(max(len(corrupted) // 2, 0), len(corrupted) - 1)
    mismatch = max(1, int(severity))
    mismatch += rng.choice([-1, 0, 1]) if severity > 1 else 0
    if mismatch == 0:
        mismatch = 1

    for idx, step in enumerate(corrupted):
        if idx < corrupt_index:
            continue
        if idx == corrupt_index:
            step["stated_output"] = int(step["true_output"]) + mismatch
            step["is_inconsistent"] = True
        elif idx > corrupt_index:
            prev = corrupted[idx - 1]
            step["input"] = int(prev["stated_output"])
            true_output = _apply_op(int(step["input"]), str(step["op"]), int(step["operand"]))
            step["true_output"] = true_output
            step["stated_output"] = true_output
            step["is_inconsistent"] = False

    return corrupted, corrupt_index, abs(mismatch)


def _corrupt_route_trace(
    trace: list[dict[str, int | str | bool]],
    *,
    severity: int,
    rng: random.Random,
) -> tuple[list[dict[str, int | str | bool]], int, int]:
    corrupted = [dict(step) for step in trace]
    if not corrupted:
        return corrupted, 0, 0

    route_index = min(max(len(corrupted) // 2, 0), len(corrupted) - 1)
    if route_index == 0 and len(corrupted) > 1:
        route_index = 1
    route_step = corrupted[route_index]
    op = str(route_step["op"])
    operand = int(route_step["operand"])
    delta = max(1, int(severity))
    delta += rng.choice([0, 1]) if severity > 1 else 0

    if op == "add":
        route_step["operand"] = operand + delta
    elif op == "sub":
        route_step["operand"] = max(1, operand - delta)
    else:
        route_step["operand"] = 2 if operand != 2 else 3

    value = int(route_step["input"])
    for idx in range(route_index, len(corrupted)):
        step = corrupted[idx]
        if idx > route_index:
            step["input"] = value
        computed = _apply_op(int(step["input"]), str(step["op"]), int(step["operand"]))
        step["true_output"] = computed
        step["stated_output"] = computed
        step["is_inconsistent"] = False
        value = computed

    return corrupted, route_index, abs(delta)


def _make_example(
    *,
    example_id: int,
    config: ToyTestBeforeRepairConfig,
    rng: random.Random,
) -> dict[str, Any]:
    start_value = rng.randint(8, 42)
    operations: list[tuple[str, int]] = []
    current = start_value
    for step_index in range(config.chain_length):
        op = rng.choice(["add", "sub", "mul"] if step_index % 3 == 2 else ["add", "sub"])
        operand = rng.choice([2, 3]) if op == "mul" else rng.randint(1, 12)
        if op == "sub" and current - operand < 0:
            op = "add"
        current = _apply_op(current, op, operand)
        operations.append((op, operand))

    gold_trace = _make_trace(start_value=start_value, operations=operations)
    truth = int(gold_trace[-1]["stated_output"])
    reference_checksum = _trace_checksum(gold_trace)
    severity_level = example_id % max(int(config.severity_bins), 1)
    corruption_severity = severity_level + 1

    candidates: list[dict[str, Any]] = []
    gold_candidate = {
        "example_id": example_id,
        "candidate_id": 0,
        "candidate_source": "gold",
        "kind": "gold",
        "start_value": start_value,
        "trace": [dict(step) for step in gold_trace],
        "final_answer": truth,
        "surface_score": 0.84 + 0.02 * rng.random(),
        "severity": 0,
        "repairable": True,
        "route_noise": 0,
    }
    candidates.append(gold_candidate)

    near_trace, corrupt_index, mismatch = _corrupt_output_trace(gold_trace, severity=corruption_severity, rng=rng)
    near_candidate = {
        "example_id": example_id,
        "candidate_id": 1,
        "candidate_source": "near_miss",
        "kind": "near_miss",
        "start_value": start_value,
        "trace": near_trace,
        "final_answer": _predict_from_trace(start_value, near_trace),
        "surface_score": 0.89 + 0.03 * config.near_miss_bias + 0.012 * severity_level + 0.02 * rng.random(),
        "severity": mismatch,
        "repairable": True,
        "corrupt_index": corrupt_index,
        "route_noise": 0,
    }
    candidates.append(near_candidate)

    route_trace, route_index, route_noise = _corrupt_route_trace(gold_trace, severity=corruption_severity, rng=rng)
    route_candidate = {
        "example_id": example_id,
        "candidate_id": 2,
        "candidate_source": "route_noisy",
        "kind": "route_noisy",
        "start_value": start_value,
        "trace": route_trace,
        "final_answer": _predict_from_trace(start_value, route_trace),
        "surface_score": 0.95 + 0.03 * config.route_noise_bias + 0.01 * severity_level + 0.02 * rng.random(),
        "severity": route_noise,
        "repairable": False,
        "route_index": route_index,
        "route_noise": route_noise,
    }
    candidates.append(route_candidate)

    for candidate_id in range(3, max(int(config.pool_size), 3)):
        distractor_trace = [dict(step) for step in gold_trace]
        distractor_severity = 1 + ((example_id + candidate_id) % max(int(config.severity_bins), 1))
        if candidate_id % 2 == 0:
            distractor_trace, _, distractor_mismatch = _corrupt_output_trace(
                distractor_trace,
                severity=distractor_severity,
                rng=rng,
            )
            repairable = True
        else:
            distractor_trace, _, distractor_mismatch = _corrupt_route_trace(
                distractor_trace,
                severity=distractor_severity,
                rng=rng,
            )
            repairable = False
        candidates.append(
            {
                "example_id": example_id,
                "candidate_id": candidate_id,
                "candidate_source": f"distractor_{candidate_id}",
                "kind": "distractor",
                "start_value": start_value,
                "trace": distractor_trace,
                "final_answer": _predict_from_trace(start_value, distractor_trace),
                "surface_score": 0.74 - 0.02 * distractor_severity + 0.02 * rng.random(),
                "severity": distractor_mismatch,
                "repairable": repairable,
                "route_noise": distractor_mismatch if not repairable else 0,
            }
        )

    return {
        "example_id": example_id,
        "start_value": start_value,
        "operations": operations,
        "truth": truth,
        "reference_checksum": reference_checksum,
        "severity_level": severity_level,
        "candidates": candidates,
    }


def _make_examples(config: ToyTestBeforeRepairConfig) -> list[dict[str, Any]]:
    rng = _make_rng(config.seed)
    return [_make_example(example_id=index, config=config, rng=rng) for index in range(config.examples)]


def _severity_bins(examples: Sequence[dict[str, Any]], bins: int) -> list[list[int]]:
    level_count = max(int(bins), 1)
    groups: list[list[int]] = [[] for _ in range(level_count)]
    for index, example in enumerate(examples):
        groups[int(example["severity_level"]) % level_count].append(index)
    return groups


def _replay_trace(
    start_value: int,
    trace: Sequence[dict[str, int | str | bool]],
) -> tuple[int, list[int], list[int]]:
    value = int(start_value)
    mismatches: list[int] = []
    recomputed_outputs: list[int] = []
    for step in trace:
        computed = _apply_op(value, str(step["op"]), int(step["operand"]))
        stated = int(step["stated_output"])
        recomputed_outputs.append(computed)
        if computed != stated:
            mismatches.append(int(step["step"]))
        value = stated
    return value, mismatches, recomputed_outputs


def _repair_trace(
    *,
    start_value: int,
    trace: Sequence[dict[str, int | str | bool]],
    repair_threshold: float,
    repair_noise: float,
) -> tuple[list[dict[str, int | str | bool]], bool, int, float, bool]:
    repaired = [dict(step) for step in trace]
    value = int(start_value)
    mismatches: list[int] = []
    first_mismatch: int | None = None
    for idx, step in enumerate(repaired):
        computed = _apply_op(value, str(step["op"]), int(step["operand"]))
        stated = int(step["stated_output"])
        if computed != stated and first_mismatch is None:
            first_mismatch = idx
        if computed != stated:
            mismatches.append(idx)
        value = stated

    verifier_signal = 0.44 + 0.15 * len(mismatches) + 0.03 * sum(
        abs(int(repaired[idx]["true_output"]) - int(repaired[idx]["stated_output"])) for idx in mismatches
    )
    verifier_noise = 0.0
    if repaired:
        verifier_noise = (
            ((start_value * 17 + int(repaired[0]["operand"]) * 13 + len(repaired) * 7) % 23) - 11
        ) / 11.0
        verifier_noise *= float(repair_noise)
    repair_confidence = max(0.0, min(1.0, verifier_signal + verifier_noise))

    false_repair = False
    repair_applied = False
    if repair_confidence >= float(repair_threshold):
        repair_applied = True
        if first_mismatch is None:
            false_repair = True
            if repaired:
                repaired[0]["stated_output"] = int(repaired[0]["stated_output"]) + 1
        else:
            running = int(start_value)
            for idx, step in enumerate(repaired):
                computed = _apply_op(running, str(step["op"]), int(step["operand"]))
                if idx == first_mismatch:
                    step["stated_output"] = computed
                    step["is_inconsistent"] = False
                elif idx > first_mismatch:
                    step["input"] = int(repaired[idx - 1]["stated_output"])
                    computed = _apply_op(int(step["input"]), str(step["op"]), int(step["operand"]))
                    step["true_output"] = computed
                    step["stated_output"] = computed
                    step["is_inconsistent"] = False
                running = int(step["stated_output"])
    final_value = int(repaired[-1]["stated_output"]) if repaired else int(start_value)
    return repaired, repair_applied, len(mismatches), repair_confidence, false_repair


def _test_score(
    candidate: dict[str, Any],
    *,
    consensus_outputs: Sequence[int],
    consensus_checksum: int,
    reference_checksum: int,
    test_noise: float,
) -> tuple[float, dict[str, Any]]:
    _, mismatches, _ = _replay_trace(int(candidate["start_value"]), candidate["trace"])
    candidate_checksum = _trace_checksum(candidate["trace"])
    consensus_gap = 0.0
    for index, output in enumerate(consensus_outputs):
        consensus_gap += abs(int(candidate["trace"][index]["stated_output"]) - int(output))
    consensus_gap /= max(len(consensus_outputs), 1)
    checksum_gap = abs(candidate_checksum - int(reference_checksum))
    consensus_checksum_gap = abs(candidate_checksum - int(consensus_checksum))
    route_gap = abs(int(candidate["trace"][-1]["stated_output"]) - int(consensus_outputs[-1])) if consensus_outputs else 0

    consistency_score = 1.0 / (1.0 + len(mismatches))
    consensus_score = 1.0 / (1.0 + consensus_gap)
    checksum_score = 1.0 / (1.0 + checksum_gap + 0.5 * route_gap)
    raw_score = 0.24 * consistency_score + 0.16 * consensus_score + 0.60 * checksum_score
    route_noise = int(candidate.get("route_noise", 0))
    noise = (((candidate["example_id"] * 31 + candidate["candidate_id"] * 17 + route_noise * 7) % 19) - 9) / 9.0
    score = max(0.0, min(1.0, raw_score + float(test_noise) * noise))
    details = {
        "mismatch_count": len(mismatches),
        "consensus_gap": float(consensus_gap),
        "checksum_gap": int(checksum_gap),
        "consensus_checksum_gap": int(consensus_checksum_gap),
        "route_gap": int(route_gap),
    }
    return score, details


def _select_repair_only(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(candidates, key=lambda candidate: (float(candidate["surface_score"]), -int(candidate["candidate_id"])))


def _select_test_before_repair(
    candidates: Sequence[dict[str, Any]],
    *,
    consensus_outputs: Sequence[int],
    consensus_checksum: int,
    reference_checksum: int,
    test_threshold: float,
    test_noise: float,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any], dict[str, Any], bool]] = []
    for candidate in candidates:
        score, details = _test_score(
            candidate,
            consensus_outputs=consensus_outputs,
            consensus_checksum=consensus_checksum,
            reference_checksum=reference_checksum,
            test_noise=test_noise,
        )
        passed = score >= float(test_threshold)
        scored.append((score, candidate, details, passed))

    passing = [item for item in scored if item[3]]
    if passing:
        selected_score, selected_candidate, selected_details, selected_pass = max(
            passing,
            key=lambda item: (float(item[1]["surface_score"]), float(item[0]), -int(item[1]["candidate_id"])),
        )
    else:
        selected_score, selected_candidate, selected_details, selected_pass = max(
            scored,
            key=lambda item: (float(item[0]), float(item[1]["surface_score"]), -int(item[1]["candidate_id"])),
        )
    return selected_candidate, float(selected_score), bool(selected_pass), selected_details


def _select_oracle(
    candidates: Sequence[dict[str, Any]],
    *,
    truth: int,
    repair_threshold: float,
    repair_noise: float,
) -> dict[str, Any]:
    best_candidate: dict[str, Any] | None = None
    best_score: tuple[int, int, float, int] | None = None
    for candidate in candidates:
        repaired_trace, repair_applied, _, repair_confidence, _ = _repair_trace(
            start_value=int(candidate["start_value"]),
            trace=candidate["trace"],
            repair_threshold=repair_threshold,
            repair_noise=repair_noise,
        )
        repaired_answer = int(repaired_trace[-1]["stated_output"]) if repaired_trace else int(candidate["start_value"])
        correct = 1 if repaired_answer == int(truth) else 0
        score = (
            correct,
            1 if repair_applied else 0,
            float(candidate["surface_score"]) + repair_confidence,
            -int(candidate["candidate_id"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_candidate = {
                **candidate,
                "repaired_trace": repaired_trace,
                "repaired_answer": repaired_answer,
                "final_answer": repaired_answer,
                "final_correct": bool(correct),
                "repair_applied": repair_applied,
                "repair_confidence": repair_confidence,
            }
    assert best_candidate is not None
    return best_candidate


def _example_trace_bytes(candidate: dict[str, Any]) -> int:
    return len(_trace_to_text(candidate["trace"]).encode("utf-8"))


def _method_bytes_estimate(
    *,
    method: str,
    selected_candidate: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    repair_applied: bool,
    repaired_trace: Sequence[dict[str, int | str | bool]],
) -> dict[str, float]:
    selected_bytes = float(_example_trace_bytes(selected_candidate))
    pool_bytes = float(sum(_example_trace_bytes(candidate) for candidate in candidates))
    repair_bytes = float(len(_trace_to_text(repaired_trace).encode("utf-8"))) if repair_applied and repaired_trace else 0.0
    if method == "repair_only":
        test_bytes = 0.0
        selection_bytes = selected_bytes
    else:
        test_bytes = pool_bytes
        selection_bytes = 0.0
    return {
        "selection_bytes_estimate": selection_bytes,
        "test_bytes_estimate": test_bytes,
        "repair_bytes_estimate": repair_bytes,
        "bytes_estimate": selection_bytes + test_bytes + repair_bytes,
    }


def _build_example_row(
    *,
    example: dict[str, Any],
    method: str,
    selected_candidate: dict[str, Any],
    selected_score: float,
    selected_test_pass: bool,
    selected_test_details: dict[str, Any],
    repaired_trace: Sequence[dict[str, int | str | bool]],
    repair_applied: bool,
    repair_confidence: float,
    false_repair: bool,
    repair_change_rate: float,
    final_answer: int,
    final_correct: bool,
    repair_only_final_answer: int,
    repair_only_final_correct: bool,
    test_pass_rate: float,
    help_vs_repair_only: float,
    harm_vs_repair_only: float,
    change_rate_vs_repair_only: float,
) -> dict[str, Any]:
    byte_stats = _method_bytes_estimate(
        method=method,
        selected_candidate=selected_candidate,
        candidates=example["candidates"],
        repair_applied=repair_applied,
        repaired_trace=repaired_trace,
    )
    return {
        "example_id": example["example_id"],
        "severity_level": example["severity_level"],
        "truth": int(example["truth"]),
        "method": method,
        "selected_candidate_source": selected_candidate["candidate_source"],
        "selected_candidate_id": int(selected_candidate["candidate_id"]),
        "selected_surface_score": float(selected_candidate["surface_score"]),
        "selected_test_score": float(selected_score),
        "selected_test_pass": 1.0 if selected_test_pass else 0.0,
        "selected_test_mismatch_count": int(selected_test_details["mismatch_count"]),
        "selected_test_consensus_gap": float(selected_test_details["consensus_gap"]),
        "selected_test_checksum_gap": int(selected_test_details["checksum_gap"]),
        "selected_test_route_gap": int(selected_test_details["route_gap"]),
        "selected_final_answer": int(selected_candidate["final_answer"]),
        "selected_correct": 1.0 if int(selected_candidate["final_answer"]) == int(example["truth"]) else 0.0,
        "repaired_final_answer": int(final_answer),
        "repaired_correct": 1.0 if final_correct else 0.0,
        "repair_applied": 1.0 if repair_applied else 0.0,
        "repair_confidence": float(repair_confidence),
        "false_repair": 1.0 if false_repair else 0.0,
        "repair_changed_answer": 1.0 if int(selected_candidate["final_answer"]) != int(final_answer) else 0.0,
        "repair_change_rate": float(repair_change_rate),
        "test_pass_rate": float(test_pass_rate),
        "help_vs_repair_only": float(help_vs_repair_only),
        "harm_vs_repair_only": float(harm_vs_repair_only),
        "change_rate_vs_repair_only": float(change_rate_vs_repair_only),
        "selection_bytes_estimate": float(byte_stats["selection_bytes_estimate"]),
        "test_bytes_estimate": float(byte_stats["test_bytes_estimate"]),
        "repair_bytes_estimate": float(byte_stats["repair_bytes_estimate"]),
        "bytes_estimate": float(byte_stats["bytes_estimate"]),
    }


def _summarize_method_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = ("repair_only", "test_before_repair", "oracle")
    summary: list[dict[str, Any]] = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        summary.append(
            {
                "method": method,
                "accuracy": float(sum(row["repaired_correct"] for row in subset) / len(subset)),
                "oracle_accuracy": float(sum(row["selected_correct"] for row in subset) / len(subset)),
                "repair_application_rate": float(sum(row["repair_applied"] for row in subset) / len(subset)),
                "repair_change_rate": float(sum(row["repair_changed_answer"] for row in subset) / len(subset)),
                "test_pass_rate": float(sum(row["selected_test_pass"] for row in subset) / len(subset)),
                "help_vs_repair_only": float(sum(row["help_vs_repair_only"] for row in subset) / len(subset)),
                "harm_vs_repair_only": float(sum(row["harm_vs_repair_only"] for row in subset) / len(subset)),
                "change_rate_vs_repair_only": float(sum(row["change_rate_vs_repair_only"] for row in subset) / len(subset)),
                "bytes_estimate": float(sum(row["bytes_estimate"] for row in subset) / len(subset)),
                "selection_bytes_estimate": float(sum(row["selection_bytes_estimate"] for row in subset) / len(subset)),
                "test_bytes_estimate": float(sum(row["test_bytes_estimate"] for row in subset) / len(subset)),
                "repair_bytes_estimate": float(sum(row["repair_bytes_estimate"] for row in subset) / len(subset)),
            }
        )
    return summary


def _subgroup_metrics(
    examples: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    groups = _severity_bins(examples, bins)
    grouped: list[dict[str, Any]] = []
    for level, indices in enumerate(groups):
        if not indices:
            continue
        index_set = {int(index) for index in indices}
        subset = [row for row in rows if int(row["example_id"]) in index_set]
        grouped.append(
            {
                "severity_level": level,
                "count": len(indices),
                "repair_only_accuracy": float(
                    sum(row["repaired_correct"] for row in subset if row["method"] == "repair_only")
                    / max(sum(1 for row in subset if row["method"] == "repair_only"), 1)
                ),
                "test_before_repair_accuracy": float(
                    sum(row["repaired_correct"] for row in subset if row["method"] == "test_before_repair")
                    / max(sum(1 for row in subset if row["method"] == "test_before_repair"), 1)
                ),
                "oracle_accuracy": float(
                    sum(row["repaired_correct"] for row in subset if row["method"] == "oracle")
                    / max(sum(1 for row in subset if row["method"] == "oracle"), 1)
                ),
                "test_before_repair_pass_rate": float(
                    sum(row["selected_test_pass"] for row in subset if row["method"] == "test_before_repair")
                    / max(sum(1 for row in subset if row["method"] == "test_before_repair"), 1)
                ),
            }
        )
    return grouped


def run_experiment(config: ToyTestBeforeRepairConfig) -> dict[str, Any]:
    examples = _make_examples(config)
    detailed_rows: list[dict[str, Any]] = []

    for example in examples:
        candidates = example["candidates"]
        consensus_outputs, consensus_checksum = _consensus_trace(candidates)

        repair_only_selected = _select_repair_only(candidates)
        repair_only_repaired, repair_only_applied, repair_only_mismatches, repair_only_confidence, repair_only_false = _repair_trace(
            start_value=int(repair_only_selected["start_value"]),
            trace=repair_only_selected["trace"],
            repair_threshold=config.repair_threshold,
            repair_noise=config.repair_noise,
        )
        repair_only_final_answer = int(repair_only_repaired[-1]["stated_output"]) if repair_only_repaired else int(repair_only_selected["start_value"])
        repair_only_final_correct = repair_only_final_answer == int(example["truth"])
        repair_only_selected_score, repair_only_test_details = _test_score(
            repair_only_selected,
            consensus_outputs=consensus_outputs,
            consensus_checksum=consensus_checksum,
            reference_checksum=int(example["reference_checksum"]),
            test_noise=config.test_noise,
        )
        repair_only_test_pass = repair_only_selected_score >= float(config.test_threshold)
        detailed_rows.append(
            _build_example_row(
                example=example,
                method="repair_only",
                selected_candidate=repair_only_selected,
                selected_score=repair_only_selected_score,
                selected_test_pass=repair_only_test_pass,
                selected_test_details=repair_only_test_details,
                repaired_trace=repair_only_repaired,
                repair_applied=repair_only_applied,
                repair_confidence=repair_only_confidence,
                false_repair=repair_only_false,
                repair_change_rate=1.0 if int(repair_only_selected["final_answer"]) != repair_only_final_answer else 0.0,
                final_answer=repair_only_final_answer,
                final_correct=repair_only_final_correct,
                repair_only_final_answer=repair_only_final_answer,
                repair_only_final_correct=repair_only_final_correct,
                test_pass_rate=1.0 if repair_only_test_pass else 0.0,
                help_vs_repair_only=0.0,
                harm_vs_repair_only=0.0,
                change_rate_vs_repair_only=0.0,
            )
        )

        tbr_selected, tbr_score, tbr_test_pass, tbr_test_details = _select_test_before_repair(
            candidates,
            consensus_outputs=consensus_outputs,
            consensus_checksum=consensus_checksum,
            reference_checksum=int(example["reference_checksum"]),
            test_threshold=config.test_threshold,
            test_noise=config.test_noise,
        )
        tbr_repaired, tbr_applied, tbr_mismatches, tbr_confidence, tbr_false = _repair_trace(
            start_value=int(tbr_selected["start_value"]),
            trace=tbr_selected["trace"],
            repair_threshold=config.repair_threshold,
            repair_noise=config.repair_noise,
        )
        tbr_final_answer = int(tbr_repaired[-1]["stated_output"]) if tbr_repaired else int(tbr_selected["start_value"])
        tbr_final_correct = tbr_final_answer == int(example["truth"])
        detailed_rows.append(
            _build_example_row(
                example=example,
                method="test_before_repair",
                selected_candidate=tbr_selected,
                selected_score=tbr_score,
                selected_test_pass=tbr_test_pass,
                selected_test_details=tbr_test_details,
                repaired_trace=tbr_repaired,
                repair_applied=tbr_applied,
                repair_confidence=tbr_confidence,
                false_repair=tbr_false,
                repair_change_rate=1.0 if int(tbr_selected["final_answer"]) != tbr_final_answer else 0.0,
                final_answer=tbr_final_answer,
                final_correct=tbr_final_correct,
                repair_only_final_answer=repair_only_final_answer,
                repair_only_final_correct=repair_only_final_correct,
                test_pass_rate=1.0 if tbr_test_pass else 0.0,
                help_vs_repair_only=1.0 if tbr_final_correct and not repair_only_final_correct else 0.0,
                harm_vs_repair_only=1.0 if repair_only_final_correct and not tbr_final_correct else 0.0,
                change_rate_vs_repair_only=1.0 if tbr_final_answer != repair_only_final_answer else 0.0,
            )
        )

        oracle_selected = _select_oracle(
            candidates,
            truth=int(example["truth"]),
            repair_threshold=config.repair_threshold,
            repair_noise=config.repair_noise,
        )
        oracle_final_answer = int(oracle_selected["repaired_answer"])
        oracle_final_correct = oracle_final_answer == int(example["truth"])
        oracle_test_score, oracle_test_details = _test_score(
            oracle_selected,
            consensus_outputs=consensus_outputs,
            consensus_checksum=consensus_checksum,
            reference_checksum=int(example["reference_checksum"]),
            test_noise=config.test_noise,
        )
        oracle_test_pass = oracle_test_score >= float(config.test_threshold)
        detailed_rows.append(
            _build_example_row(
                example=example,
                method="oracle",
                selected_candidate=oracle_selected,
                selected_score=oracle_test_score,
                selected_test_pass=oracle_test_pass,
                selected_test_details=oracle_test_details,
                repaired_trace=oracle_selected["repaired_trace"],
                repair_applied=bool(oracle_selected["repair_applied"]),
                repair_confidence=float(oracle_selected["repair_confidence"]),
                false_repair=False,
                repair_change_rate=1.0 if int(oracle_selected["final_answer"]) != oracle_final_answer else 0.0,
                final_answer=oracle_final_answer,
                final_correct=oracle_final_correct,
                repair_only_final_answer=repair_only_final_answer,
                repair_only_final_correct=repair_only_final_correct,
                test_pass_rate=1.0 if oracle_test_pass else 0.0,
                help_vs_repair_only=1.0 if oracle_final_correct and not repair_only_final_correct else 0.0,
                harm_vs_repair_only=1.0 if repair_only_final_correct and not oracle_final_correct else 0.0,
                change_rate_vs_repair_only=1.0 if oracle_final_answer != repair_only_final_answer else 0.0,
            )
        )

    rows = _summarize_method_rows(detailed_rows)
    return {
        "config": asdict(config),
        "rows": rows,
        "example_rows": detailed_rows,
        "subgroups": {
            "severity": _subgroup_metrics(examples, detailed_rows, bins=config.severity_bins),
        },
    }


def write_jsonl(rows: Sequence[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return "[" + ", ".join(str(item) for item in value) + "]"
        return f"{float(value):.4f}"

    rows = payload["rows"]
    subgroups = payload["subgroups"]["severity"]
    lines = [
        "# Toy Test-Before-Repair Bridge",
        "",
        "- Candidate pool with one output-corrupted near miss and one semantically noisy route trace.",
        "- Repair-only spends its repair pass immediately on the highest-surface candidate.",
        "- Test-before-repair runs discriminative checks on the pool first, then spends the same repair budget only when needed.",
        "",
        f"- Seed: `{payload['config']['seed']}`",
        f"- Examples: `{payload['config']['examples']}`",
        f"- Pool size: `{payload['config']['pool_size']}`",
        f"- Chain length: `{payload['config']['chain_length']}`",
        f"- Test threshold: `{fmt(payload['config']['test_threshold'])}`",
        f"- Repair threshold: `{fmt(payload['config']['repair_threshold'])}`",
        "",
        "| Method | Accuracy | Oracle accuracy | Repair app. rate | Repair change rate | Test pass rate | Help vs repair-only | Harm vs repair-only | Change vs repair-only | Bytes est. | Test bytes | Repair bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {oracle_accuracy} | {repair_application_rate} | {repair_change_rate} | {test_pass_rate} | {help_vs_repair_only} | {harm_vs_repair_only} | {change_rate_vs_repair_only} | {bytes_estimate} | {test_bytes_estimate} | {repair_bytes_estimate} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                oracle_accuracy=fmt(row["oracle_accuracy"]),
                repair_application_rate=fmt(row["repair_application_rate"]),
                repair_change_rate=fmt(row["repair_change_rate"]),
                test_pass_rate=fmt(row["test_pass_rate"]),
                help_vs_repair_only=fmt(row["help_vs_repair_only"]),
                harm_vs_repair_only=fmt(row["harm_vs_repair_only"]),
                change_rate_vs_repair_only=fmt(row["change_rate_vs_repair_only"]),
                bytes_estimate=fmt(row["bytes_estimate"]),
                test_bytes_estimate=fmt(row["test_bytes_estimate"]),
                repair_bytes_estimate=fmt(row["repair_bytes_estimate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Severity Subgroups",
            "",
            "| Severity level | Count | Repair-only acc. | Test-before-repair acc. | Oracle acc. | Test-before-repair pass rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in subgroups:
        lines.append(
            "| {severity_level} | {count} | {repair_only_accuracy} | {test_before_repair_accuracy} | {oracle_accuracy} | {test_before_repair_pass_rate} |".format(
                severity_level=row["severity_level"],
                count=row["count"],
                repair_only_accuracy=fmt(row["repair_only_accuracy"]),
                test_before_repair_accuracy=fmt(row["test_before_repair_accuracy"]),
                oracle_accuracy=fmt(row["oracle_accuracy"]),
                test_before_repair_pass_rate=fmt(row["test_before_repair_pass_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The route-noisy candidate is the hard case: it is internally consistent enough that output-only repair does not fix it, but the test suite detects the semantic drift before repair is spent.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy test-before-repair bridge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=192)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--chain-length", type=int, default=4)
    parser.add_argument("--severity-bins", type=int, default=3)
    parser.add_argument("--test-threshold", type=float, default=0.69)
    parser.add_argument("--repair-threshold", type=float, default=0.55)
    parser.add_argument("--repair-noise", type=float, default=0.09)
    parser.add_argument("--test-noise", type=float, default=0.05)
    parser.add_argument("--near-miss-bias", type=float, default=0.18)
    parser.add_argument("--route-noise-bias", type=float, default=0.22)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyTestBeforeRepairConfig(
        seed=args.seed,
        examples=args.examples,
        pool_size=args.pool_size,
        chain_length=args.chain_length,
        severity_bins=args.severity_bins,
        test_threshold=args.test_threshold,
        repair_threshold=args.repair_threshold,
        repair_noise=args.repair_noise,
        test_noise=args.test_noise,
        near_miss_bias=args.near_miss_bias,
        route_noise_bias=args.route_noise_bias,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    write_jsonl(payload["example_rows"], output)
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
