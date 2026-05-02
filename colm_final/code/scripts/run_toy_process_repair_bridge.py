#!/usr/bin/env python3
"""Toy process-aware answer repair bridge.

This bounded ablation synthesizes arithmetic reasoning traces with one near-miss
candidate that contains an inconsistent intermediate step. A rerank-only
selector is biased toward the high-confidence near miss, while a process-aware
verifier can detect the inconsistency, apply a minimal repair, and recover the
correct final answer.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ToyProcessRepairConfig:
    seed: int = 0
    examples: int = 192
    pool_size: int = 5
    chain_length: int = 4
    severity_bins: int = 3
    repair_threshold: float = 0.55
    repair_noise: float = 0.11
    rerank_noise: float = 0.06
    near_miss_bias: float = 0.16


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


def _corrupt_trace(
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
            step["stated_output"] = _apply_op(int(step["input"]), str(step["op"]), int(step["operand"]))
            step["is_inconsistent"] = False

    final_output = int(corrupted[-1]["stated_output"])
    return corrupted, corrupt_index, abs(mismatch)


def _trace_to_text(trace: Sequence[dict[str, int | str | bool]]) -> str:
    parts = []
    for step in trace:
        parts.append(
            f"{step['input']} {step['op']} {step['operand']} = {step['stated_output']}"
        )
    return " | ".join(parts)


def _replay_trace(
    start_value: int,
    trace: Sequence[dict[str, int | str | bool]],
) -> tuple[int, list[int], list[int]]:
    value = int(start_value)
    mismatches: list[int] = []
    repaired_outputs: list[int] = []
    for step in trace:
        computed = _apply_op(value, str(step["op"]), int(step["operand"]))
        stated = int(step["stated_output"])
        repaired_outputs.append(computed)
        if computed != stated:
            mismatches.append(int(step["step"]))
        value = stated
    return value, mismatches, repaired_outputs


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
    repair_applied = False
    first_mismatch: int | None = None
    for idx, step in enumerate(repaired):
        computed = _apply_op(value, str(step["op"]), int(step["operand"]))
        stated = int(step["stated_output"])
        if computed != stated and first_mismatch is None:
            first_mismatch = idx
        if computed != stated:
            mismatches.append(idx)
        value = stated

    verifier_signal = 0.46 + 0.16 * len(mismatches) + 0.03 * sum(
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


def _predict_from_trace(start_value: int, trace: Sequence[dict[str, int | str | bool]]) -> int:
    if not trace:
        return int(start_value)
    return int(trace[-1]["stated_output"])


def _candidate_score(candidate: dict[str, Any], *, rerank_noise: float) -> float:
    base = float(candidate["surface_score"])
    deterministic_noise = ((candidate["example_id"] * 19 + candidate["candidate_id"] * 7) % 13 - 6) / 100.0
    return base + rerank_noise * deterministic_noise


def _select_by_rerank(candidates: Sequence[dict[str, Any]], *, rerank_noise: float) -> dict[str, Any]:
    return max(candidates, key=lambda candidate: (_candidate_score(candidate, rerank_noise=rerank_noise), -candidate["candidate_id"]))


def _make_example(
    *,
    example_id: int,
    config: ToyProcessRepairConfig,
    rng: random.Random,
) -> dict[str, Any]:
    start_value = rng.randint(7, 40)
    operations: list[tuple[str, int]] = []
    current = start_value
    for step_index in range(config.chain_length):
        op = rng.choice(["add", "sub", "mul"] if step_index % 3 == 2 else ["add", "sub"])
        if op == "mul":
            operand = rng.choice([2, 3])
        else:
            operand = rng.randint(1, 12)
        if op == "sub" and current - operand < 0:
            op = "add"
        current = _apply_op(current, op, operand)
        operations.append((op, operand))

    gold_trace = _make_trace(start_value=start_value, operations=operations)
    truth = int(gold_trace[-1]["stated_output"])
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
        "surface_score": 0.86 + 0.025 * rng.random() + (0.01 if example_id % 4 == 0 else 0.0),
        "severity": 0,
        "repairable": True,
    }
    candidates.append(gold_candidate)

    near_trace, corrupt_index, mismatch = _corrupt_trace(gold_trace, severity=corruption_severity, rng=rng)
    near_candidate = {
        "example_id": example_id,
        "candidate_id": 1,
        "candidate_source": "near_miss",
        "kind": "near_miss",
        "start_value": start_value,
        "trace": near_trace,
        "final_answer": _predict_from_trace(start_value, near_trace),
        "surface_score": 0.82 + 0.18 * config.near_miss_bias + 0.012 * severity_level + 0.025 * rng.random(),
        "severity": mismatch,
        "repairable": True,
        "corrupt_index": corrupt_index,
    }
    candidates.append(near_candidate)

    for candidate_id in range(2, max(int(config.pool_size), 2)):
        distractor_trace = [dict(step) for step in gold_trace]
        distractor_severity = 1 + ((example_id + candidate_id) % max(int(config.severity_bins), 1))
        distractor_trace, _, distractor_mismatch = _corrupt_trace(distractor_trace, severity=distractor_severity, rng=rng)
        candidates.append(
            {
                "example_id": example_id,
                "candidate_id": candidate_id,
                "candidate_source": f"distractor_{candidate_id}",
                "kind": "distractor",
                "start_value": start_value,
                "trace": distractor_trace,
                "final_answer": _predict_from_trace(start_value, distractor_trace),
                "surface_score": 0.72 - 0.04 * distractor_severity + 0.035 * rng.random(),
                "severity": distractor_mismatch,
                "repairable": True,
            }
        )

    return {
        "example_id": example_id,
        "start_value": start_value,
        "operations": operations,
        "truth": truth,
        "severity_level": severity_level,
        "candidates": candidates,
    }


def _make_examples(config: ToyProcessRepairConfig) -> list[dict[str, Any]]:
    rng = _make_rng(config.seed)
    return [_make_example(example_id=index, config=config, rng=rng) for index in range(config.examples)]


def _severity_bins(examples: Sequence[dict[str, Any]], bins: int) -> list[list[int]]:
    level_count = max(int(bins), 1)
    groups: list[list[int]] = [[] for _ in range(level_count)]
    for index, example in enumerate(examples):
        groups[int(example["severity_level"]) % level_count].append(index)
    return groups


def _evaluate_rerank_only(examples: Sequence[dict[str, Any]], config: ToyProcessRepairConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_correct = []
    selected_severity = []
    for example in examples:
        selected = _select_by_rerank(example["candidates"], rerank_noise=config.rerank_noise)
        correct = 1.0 if int(selected["final_answer"]) == int(example["truth"]) else 0.0
        rows.append(
            {
                "example_id": example["example_id"],
                "selected_candidate_source": selected["candidate_source"],
                "selected_final_answer": int(selected["final_answer"]),
                "selected_correct": correct,
                "selected_severity": int(selected["severity"]),
                "repair_applied": 0.0,
                "false_repair": 0.0,
                "repair_confidence": 0.0,
            }
        )
        selected_correct.append(correct)
        selected_severity.append(int(selected["severity"]))
    return {
        "rows": rows,
        "accuracy": float(sum(selected_correct) / max(len(selected_correct), 1)),
        "repair_application_rate": 0.0,
        "false_repair_rate": 0.0,
        "mean_selected_severity": float(sum(selected_severity) / max(len(selected_severity), 1)),
    }


def _evaluate_process_repair(examples: Sequence[dict[str, Any]], config: ToyProcessRepairConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_correct = []
    repaired_correct = []
    repair_applied_flags = []
    false_repairs = []
    selected_severity = []

    for example in examples:
        selected = _select_by_rerank(example["candidates"], rerank_noise=config.rerank_noise)
        selected_correct_flag = 1.0 if int(selected["final_answer"]) == int(example["truth"]) else 0.0
        repaired_trace, repair_applied, mismatches, repair_confidence, false_repair = _repair_trace(
            start_value=int(selected["start_value"]),
            trace=selected["trace"],
            repair_threshold=config.repair_threshold,
            repair_noise=config.repair_noise,
        )
        repaired_answer = int(repaired_trace[-1]["stated_output"]) if repaired_trace else int(selected["start_value"])
        repaired_correct_flag = 1.0 if repaired_answer == int(example["truth"]) else 0.0

        rows.append(
            {
                "example_id": example["example_id"],
                "selected_candidate_source": selected["candidate_source"],
                "selected_final_answer": int(selected["final_answer"]),
                "repaired_final_answer": repaired_answer,
                "selected_correct": selected_correct_flag,
                "repaired_correct": repaired_correct_flag,
                "repair_applied": 1.0 if repair_applied else 0.0,
                "false_repair": 1.0 if false_repair else 0.0,
                "repair_confidence": repair_confidence,
                "selected_severity": int(selected["severity"]),
                "detected_mismatches": int(mismatches),
                "selected_trace": _trace_to_text(selected["trace"]),
                "repaired_trace": _trace_to_text(repaired_trace),
            }
        )

        selected_correct.append(selected_correct_flag)
        repaired_correct.append(repaired_correct_flag)
        repair_applied_flags.append(1.0 if repair_applied else 0.0)
        false_repairs.append(1.0 if false_repair else 0.0)
        selected_severity.append(int(selected["severity"]))

    return {
        "rows": rows,
        "accuracy": float(sum(repaired_correct) / max(len(repaired_correct), 1)),
        "repair_application_rate": float(sum(repair_applied_flags) / max(len(repair_applied_flags), 1)),
        "false_repair_rate": float(sum(false_repairs) / max(len(false_repairs), 1)),
        "mean_selected_severity": float(sum(selected_severity) / max(len(selected_severity), 1)),
    }


def _evaluate_oracle(examples: Sequence[dict[str, Any]], config: ToyProcessRepairConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = []
    for example in examples:
        best_candidate = None
        best_score = None
        for candidate in example["candidates"]:
            repaired_trace, _, _, repair_confidence, false_repair = _repair_trace(
                start_value=int(candidate["start_value"]),
                trace=candidate["trace"],
                repair_threshold=config.repair_threshold,
                repair_noise=config.repair_noise,
            )
            repaired_answer = int(repaired_trace[-1]["stated_output"]) if repaired_trace else int(candidate["start_value"])
            score = (
                float(candidate["surface_score"]),
                repair_confidence,
                -int(candidate["severity"]),
                -int(candidate["candidate_id"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = repaired_answer
        oracle_correct = 1.0 if int(best_candidate) == int(example["truth"]) else 0.0
        rows.append(
            {
                "example_id": example["example_id"],
                "oracle_final_answer": int(best_candidate),
                "oracle_correct": oracle_correct,
            }
        )
        correct.append(oracle_correct)
    return {
        "rows": rows,
        "accuracy": float(sum(correct) / max(len(correct), 1)),
        "repair_application_rate": 1.0,
        "false_repair_rate": 0.0,
    }


def _subgroup_metrics(
    examples: Sequence[dict[str, Any]],
    rerank_rows: Sequence[dict[str, Any]],
    repair_rows: Sequence[dict[str, Any]],
    oracle_rows: Sequence[dict[str, Any]],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    groups = _severity_bins(examples, bins)
    rows: list[dict[str, Any]] = []
    for level, indices in enumerate(groups):
        if not indices:
            continue
        rerank_subset = [rerank_rows[i] for i in indices]
        repair_subset = [repair_rows[i] for i in indices]
        oracle_subset = [oracle_rows[i] for i in indices]
        rows.append(
            {
                "severity_level": level,
                "count": len(indices),
                "rerank_only_accuracy": float(sum(row["selected_correct"] for row in rerank_subset) / len(indices)),
                "repair_accuracy": float(sum(row["repaired_correct"] for row in repair_subset) / len(indices)),
                "oracle_accuracy": float(sum(row["oracle_correct"] for row in oracle_subset) / len(indices)),
                "repair_application_rate": float(sum(row["repair_applied"] for row in repair_subset) / len(indices)),
                "false_repair_rate": float(sum(row["false_repair"] for row in repair_subset) / len(indices)),
            }
        )
    return rows


def run_experiment(config: ToyProcessRepairConfig) -> dict[str, Any]:
    examples = _make_examples(config)
    rerank = _evaluate_rerank_only(examples, config)
    repair = _evaluate_process_repair(examples, config)
    oracle = _evaluate_oracle(examples, config)

    rows = [
        {
            "method": "rerank_only",
            "accuracy": rerank["accuracy"],
            "oracle_accuracy": oracle["accuracy"],
            "repair_application_rate": rerank["repair_application_rate"],
            "false_repair_rate": rerank["false_repair_rate"],
            "mean_selected_severity": rerank["mean_selected_severity"],
        },
        {
            "method": "process_aware_repair",
            "accuracy": repair["accuracy"],
            "oracle_accuracy": oracle["accuracy"],
            "repair_application_rate": repair["repair_application_rate"],
            "false_repair_rate": repair["false_repair_rate"],
            "mean_selected_severity": repair["mean_selected_severity"],
        },
        {
            "method": "oracle",
            "accuracy": oracle["accuracy"],
            "oracle_accuracy": oracle["accuracy"],
            "repair_application_rate": oracle["repair_application_rate"],
            "false_repair_rate": oracle["false_repair_rate"],
            "mean_selected_severity": 0.0,
        },
    ]

    return {
        "config": asdict(config),
        "rows": rows,
        "subgroups": {
            "severity": _subgroup_metrics(examples, rerank["rows"], repair["rows"], oracle["rows"], bins=config.severity_bins),
        },
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    rows = payload["rows"]
    subgroups = payload["subgroups"]["severity"]
    lines = [
        "# Toy Process Repair Bridge",
        "",
        f"- Seed: `{payload['config']['seed']}`",
        f"- Examples: `{payload['config']['examples']}`",
        f"- Pool size: `{payload['config']['pool_size']}`",
        f"- Chain length: `{payload['config']['chain_length']}`",
        f"- Repair threshold: `{fmt(payload['config']['repair_threshold'])}`",
        "",
        "| Method | Accuracy | Oracle accuracy | Repair application rate | False repair rate | Mean selected severity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {oracle_accuracy} | {repair_application_rate} | {false_repair_rate} | {mean_selected_severity} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                oracle_accuracy=fmt(row["oracle_accuracy"]),
                repair_application_rate=fmt(row["repair_application_rate"]),
                false_repair_rate=fmt(row["false_repair_rate"]),
                mean_selected_severity=fmt(row["mean_selected_severity"]),
            )
        )
    lines.extend(
        [
            "",
            "## Severity Subgroups",
            "",
            "| Severity level | Count | Rerank-only accuracy | Repair accuracy | Oracle accuracy | Repair application rate | False repair rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in subgroups:
        lines.append(
            "| {severity_level} | {count} | {rerank_only_accuracy} | {repair_accuracy} | {oracle_accuracy} | {repair_application_rate} | {false_repair_rate} |".format(
                severity_level=row["severity_level"],
                count=row["count"],
                rerank_only_accuracy=fmt(row["rerank_only_accuracy"]),
                repair_accuracy=fmt(row["repair_accuracy"]),
                oracle_accuracy=fmt(row["oracle_accuracy"]),
                repair_application_rate=fmt(row["repair_application_rate"]),
                false_repair_rate=fmt(row["false_repair_rate"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy process-aware answer repair bridge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=192)
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--chain-length", type=int, default=4)
    parser.add_argument("--severity-bins", type=int, default=3)
    parser.add_argument("--repair-threshold", type=float, default=0.55)
    parser.add_argument("--repair-noise", type=float, default=0.11)
    parser.add_argument("--rerank-noise", type=float, default=0.06)
    parser.add_argument("--near-miss-bias", type=float, default=0.16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyProcessRepairConfig(
        seed=args.seed,
        examples=args.examples,
        pool_size=args.pool_size,
        chain_length=args.chain_length,
        severity_bins=args.severity_bins,
        repair_threshold=args.repair_threshold,
        repair_noise=args.repair_noise,
        rerank_noise=args.rerank_noise,
        near_miss_bias=args.near_miss_bias,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
