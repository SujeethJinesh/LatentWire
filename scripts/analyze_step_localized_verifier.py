#!/usr/bin/env python3
"""Replay step-localized verifier policies over process-repair telemetry."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_process_gate_features import _equation_stats, extract_process_features


TARGET_METHOD = "target_alone"
SELECTED_METHOD = "selected_route_no_repair"
TARGET_SELF_METHOD = "target_self_repair"
REPAIR_METHOD = "process_repair_selected_route"


@dataclass(frozen=True)
class StepVerifierSummary:
    source: str
    rows: list[dict[str, Any]]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_index_method(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["index"]), {})[str(record["method"])] = record
    return grouped


def _eligible_examples(records: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    grouped = _group_by_index_method(records)
    examples: list[dict[str, dict[str, Any]]] = []
    for idx in sorted(grouped):
        methods = grouped[idx]
        if SELECTED_METHOD in methods and REPAIR_METHOD in methods and TARGET_METHOD in methods:
            examples.append(methods)
    return examples


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _split_steps(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip(" -*\t")
        if stripped:
            lines.append(stripped)
    if lines:
        return lines
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def extract_step_verifier_features(selected: dict[str, Any]) -> dict[str, Any]:
    text = str(selected.get("prediction") or selected.get("repair_pre_prediction") or "")
    steps = _split_steps(text)
    process = extract_process_features(selected)
    invalid_steps: list[int] = []
    equation_steps = 0
    valid_equation_steps = 0
    for idx, step in enumerate(steps):
        equation_count, valid_equations = _equation_stats(step)
        if equation_count:
            equation_steps += 1
            if valid_equations == equation_count:
                valid_equation_steps += 1
            else:
                invalid_steps.append(idx)
    first_error_step = invalid_steps[0] if invalid_steps else None
    has_invalid_equation = bool(invalid_steps)
    missing_answer_marker = process["answer_marker_score"] < 0.5
    unfinished_tail = process["finished_tail_score"] < 0.5
    no_equations = equation_steps == 0
    step_validity = valid_equation_steps / equation_steps if equation_steps else 0.0
    localized_error_count = int(has_invalid_equation) + int(missing_answer_marker) + int(unfinished_tail)
    if no_equations:
        localized_error_count += 1
    scalar_meta_score = (
        _to_float(selected.get("candidate_format_score"))
        + _to_float(selected.get("candidate_completion_score"))
        + 0.5 * _to_float(selected.get("candidate_vote_margin"))
        + 0.25 * _to_float(selected.get("candidate_answer_agreement"))
    )
    step_localized_score = (
        2.0 * step_validity
        + process["answer_marker_score"]
        + process["finished_tail_score"]
        + min(2.0, process["reasoning_step_count"] / 3.0)
        - localized_error_count
    )
    critique_plus_repair_score = -float(localized_error_count)
    return {
        "step_count": len(steps),
        "equation_step_count": equation_steps,
        "valid_equation_step_count": valid_equation_steps,
        "step_validity": step_validity,
        "first_error_step": first_error_step,
        "first_error_step_1indexed": None if first_error_step is None else first_error_step + 1,
        "has_invalid_equation": has_invalid_equation,
        "missing_answer_marker": missing_answer_marker,
        "unfinished_tail": unfinished_tail,
        "no_equations": no_equations,
        "localized_error_count": localized_error_count,
        "scalar_meta_score": scalar_meta_score,
        "step_localized_score": step_localized_score,
        "critique_plus_repair_score": critique_plus_repair_score,
        **process,
    }


def _bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values or samples <= 0:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_idx = int(0.025 * (len(means) - 1))
    high_idx = int(0.975 * (len(means) - 1))
    return means[low_idx], means[high_idx]


def _thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return []
    return unique


def _policy_uses_repair(features: dict[str, Any], policy: str, threshold: float | None) -> bool:
    if policy == "never_repair_selected":
        return False
    if policy == "repair_all_selected":
        return True
    if policy == "oracle_precheck_analysis_only":
        raise ValueError("oracle policy handled separately")
    if policy == "scalar_meta_gate":
        if threshold is None:
            raise ValueError("threshold required")
        return features["scalar_meta_score"] < threshold
    if policy == "step_localized_gate":
        if threshold is None:
            raise ValueError("threshold required")
        return features["step_localized_score"] < threshold
    if policy == "critique_plus_repair_gate":
        return bool(features["localized_error_count"])
    raise ValueError(f"unknown policy: {policy}")


def _summarize_policy(
    examples: list[dict[str, dict[str, Any]]],
    feature_rows: list[dict[str, Any]],
    *,
    policy: str,
    threshold: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected_correct = [bool(methods[SELECTED_METHOD].get("correct")) for methods in examples]
    repair_correct = [bool(methods[REPAIR_METHOD].get("correct")) for methods in examples]
    target_correct = [bool(methods[TARGET_METHOD].get("correct")) for methods in examples]
    target_self_correct = [
        bool(methods.get(TARGET_SELF_METHOD, {}).get("correct"))
        for methods in examples
        if TARGET_SELF_METHOD in methods
    ]
    if policy == "oracle_precheck_analysis_only":
        use_repair = [not value for value in selected_correct]
    else:
        use_repair = [
            _policy_uses_repair(features, policy, threshold) for features in feature_rows
        ]
    final_correct = [
        repair if repair_flag else selected
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    ]
    n = len(examples)
    final_values = [float(value) for value in final_correct]
    repair_values = [float(value) for value in repair_correct]
    target_self_accuracy = (
        sum(target_self_correct) / len(target_self_correct) if target_self_correct else None
    )
    acc_ci_low, acc_ci_high = _bootstrap_ci(
        final_values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    delta_repair_ci_low, delta_repair_ci_high = _bootstrap_ci(
        [final - repair for final, repair in zip(final_values, repair_values, strict=True)],
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    repaired_help = sum(
        repair_flag and (not selected) and repair
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    )
    missed_help = sum(
        (not repair_flag) and (not selected) and repair
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    )
    false_repair = sum(
        repair_flag and selected
        for repair_flag, selected in zip(use_repair, selected_correct, strict=True)
    )
    false_skip = sum(
        (not repair_flag) and (not selected)
        for repair_flag, selected in zip(use_repair, selected_correct, strict=True)
    )
    extra_repair_chars = [
        _to_float(methods[REPAIR_METHOD].get("repair_prompt_chars")) if repair_flag else 0.0
        for repair_flag, methods in zip(use_repair, examples, strict=True)
    ]
    extra_repair_tokens = [
        _to_float(methods[REPAIR_METHOD].get("generated_tokens")) if repair_flag else 0.0
        for repair_flag, methods in zip(use_repair, examples, strict=True)
    ]
    accuracy = sum(final_correct) / n
    repair_all_accuracy = sum(repair_correct) / n
    return {
        "policy": policy,
        "threshold": threshold,
        "n": n,
        "accuracy": accuracy,
        "accuracy_ci_low": acc_ci_low,
        "accuracy_ci_high": acc_ci_high,
        "target_accuracy": sum(target_correct) / n,
        "selected_no_repair_accuracy": sum(selected_correct) / n,
        "repair_all_accuracy": repair_all_accuracy,
        "target_self_repair_accuracy": target_self_accuracy,
        "delta_vs_repair_all": accuracy - repair_all_accuracy,
        "delta_vs_repair_all_ci_low": delta_repair_ci_low,
        "delta_vs_repair_all_ci_high": delta_repair_ci_high,
        "delta_vs_target_self_repair": (
            accuracy - target_self_accuracy if target_self_accuracy is not None else None
        ),
        "repair_rate": sum(use_repair) / n,
        "saved_repair_rate": 1.0 - (sum(use_repair) / n),
        "repair_call_count": int(sum(use_repair)),
        "repaired_help_count": int(repaired_help),
        "missed_help_count": int(missed_help),
        "false_repair_count": int(false_repair),
        "false_skip_count": int(false_skip),
        "avg_extra_repair_prompt_chars": sum(extra_repair_chars) / n,
        "avg_extra_repair_generated_tokens": sum(extra_repair_tokens) / n,
        "localized_error_rate": sum(bool(row["localized_error_count"]) for row in feature_rows) / n,
        "invalid_equation_rate": sum(bool(row["has_invalid_equation"]) for row in feature_rows) / n,
        "missing_answer_marker_rate": sum(bool(row["missing_answer_marker"]) for row in feature_rows) / n,
        "unfinished_tail_rate": sum(bool(row["unfinished_tail"]) for row in feature_rows) / n,
    }


def summarize_source(
    path: pathlib.Path,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> StepVerifierSummary:
    examples = _eligible_examples(load_jsonl(path))
    feature_rows = [extract_step_verifier_features(methods[SELECTED_METHOD]) for methods in examples]
    rows = [
        _summarize_policy(
            examples,
            feature_rows,
            policy="never_repair_selected",
            threshold=None,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1,
        ),
        _summarize_policy(
            examples,
            feature_rows,
            policy="repair_all_selected",
            threshold=None,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 2,
        ),
        _summarize_policy(
            examples,
            feature_rows,
            policy="oracle_precheck_analysis_only",
            threshold=None,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 3,
        ),
        _summarize_policy(
            examples,
            feature_rows,
            policy="critique_plus_repair_gate",
            threshold=None,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 4,
        ),
    ]
    for policy, key in (
        ("scalar_meta_gate", "scalar_meta_score"),
        ("step_localized_gate", "step_localized_score"),
    ):
        for offset, threshold in enumerate(_thresholds([float(row[key]) for row in feature_rows])):
            rows.append(
                _summarize_policy(
                    examples,
                    feature_rows,
                    policy=policy,
                    threshold=threshold,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + 1000 + offset,
                )
            )
    rows.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            float(row["repair_rate"]),
            str(row["policy"]),
            float(row["threshold"] if row["threshold"] is not None else -9999.0),
        )
    )
    return StepVerifierSummary(source=path.name, rows=rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _ci(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"[{float(low):.4f}, {float(high):.4f}]"


def format_markdown(summaries: Sequence[StepVerifierSummary], *, top_k: int = 14) -> str:
    lines = [
        "# Step-Localized Verifier Replay",
        "",
        "Policies are replayed on existing selected-route text and logged repair",
        "outputs. No model calls are made. `oracle_precheck_analysis_only` is an",
        "analysis ceiling and must not be used as a method row.",
    ]
    for summary in summaries:
        lines.extend(
            [
                "",
                f"## {summary.source}",
                "",
                "| Policy | Threshold | Accuracy | Acc CI | Repair rate | Saved repair | Delta vs repair-all | Extra repair chars | Help | Missed help | False repair | False skip |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary.rows[:top_k]:
            lines.append(
                "| {policy} | {threshold} | {accuracy} | {acc_ci} | {repair_rate} | {saved} | {delta} | {chars} | {help_count} | {missed} | {false_repair} | {false_skip} |".format(
                    policy=row["policy"],
                    threshold=_fmt(row["threshold"]),
                    accuracy=_fmt(row["accuracy"]),
                    acc_ci=_ci(row["accuracy_ci_low"], row["accuracy_ci_high"]),
                    repair_rate=_fmt(row["repair_rate"]),
                    saved=_fmt(row["saved_repair_rate"]),
                    delta=_fmt(row["delta_vs_repair_all"]),
                    chars=_fmt(row["avg_extra_repair_prompt_chars"]),
                    help_count=int(row["repaired_help_count"]),
                    missed=int(row["missed_help_count"]),
                    false_repair=int(row["false_repair_count"]),
                    false_skip=int(row["false_skip_count"]),
                )
            )
    return "\n".join(lines) + "\n"


def build_json(summaries: Sequence[StepVerifierSummary]) -> dict[str, Any]:
    return {"sources": [{"source": summary.source, "rows": summary.rows} for summary in summaries]}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay step-localized verifier policies.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--top-k", type=int, default=14)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    summaries = [
        summarize_source(
            pathlib.Path(path),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + idx * 10_000,
        )
        for idx, path in enumerate(args.inputs)
    ]
    payload = build_json(summaries)
    output_json = pathlib.Path(args.output_json)
    output_md = pathlib.Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(format_markdown(summaries, top_k=args.top_k), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
