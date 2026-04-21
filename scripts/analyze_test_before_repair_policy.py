#!/usr/bin/env python3
"""Analyze test-before-repair gates on process-repair telemetry."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_process_gate_features import extract_process_features


TARGET_METHOD = "target_alone"
SELECTED_METHOD = "selected_route_no_repair"
TARGET_SELF_METHOD = "target_self_repair"
REPAIR_METHOD = "process_repair_selected_route"

GATE_FIELDS = {
    "format_gate": "candidate_format_score",
    "completion_gate": "candidate_completion_score",
    "format_delta_gate": "selected_candidate_format_delta_vs_target",
    "vote_margin_gate": "candidate_vote_margin",
    "process_gate": "process_completeness_score",
    "format_plus_process_gate": "format_plus_process_score",
    "valid_equation_gate": "valid_equation_count",
}


@dataclass(frozen=True)
class SourcePolicySummary:
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
        idx = int(record["index"])
        grouped.setdefault(idx, {})[str(record["method"])] = record
    return grouped


def _as_float(row: dict[str, Any], field: str, default: float = float("-inf")) -> float:
    value = row.get(field)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_passes(selected: dict[str, Any], policy: str, threshold: float) -> bool:
    if policy == "format_gate":
        return _as_float(selected, "candidate_format_score") >= threshold
    if policy == "completion_gate":
        return _as_float(selected, "candidate_completion_score") >= threshold
    if policy == "format_delta_gate":
        return _as_float(selected, "selected_candidate_format_delta_vs_target") >= threshold
    if policy == "vote_margin_gate":
        return _as_float(selected, "candidate_vote_margin") >= threshold
    if policy == "format_and_completion_gate":
        return (
            _as_float(selected, "candidate_format_score") >= threshold
            and _as_float(selected, "candidate_completion_score") >= threshold
        )
    if policy in GATE_FIELDS:
        return _as_float(selected, GATE_FIELDS[policy]) >= threshold
    raise ValueError(f"Unknown policy: {policy}")


def _thresholds(records: list[dict[str, Any]], field: str) -> list[float]:
    values = sorted({_as_float(row, field) for row in records if row.get(field) is not None})
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return []
    thresholds = set(values)
    if len(values) > 2:
        thresholds.update(
            {
                values[len(values) // 4],
                values[len(values) // 2],
                values[(3 * len(values)) // 4],
            }
        )
    return sorted(thresholds)


def _eligible_examples(records: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    grouped = _group_by_index_method(records)
    examples: list[dict[str, dict[str, Any]]] = []
    for idx in sorted(grouped):
        methods = grouped[idx]
        if SELECTED_METHOD in methods and REPAIR_METHOD in methods and TARGET_METHOD in methods:
            examples.append(methods)
    return examples


def _with_process_features(
    examples: list[dict[str, dict[str, Any]]],
) -> list[dict[str, dict[str, Any]]]:
    enriched: list[dict[str, dict[str, Any]]] = []
    for methods in examples:
        methods_copy = {method: row.copy() for method, row in methods.items()}
        selected = methods_copy[SELECTED_METHOD]
        selected.update(extract_process_features(selected))
        enriched.append(methods_copy)
    return enriched


def _summarize_policy(
    examples: list[dict[str, dict[str, Any]]],
    *,
    policy: str,
    threshold: float | None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No complete examples found")

    selected_correct = [bool(methods[SELECTED_METHOD].get("correct")) for methods in examples]
    repair_correct = [bool(methods[REPAIR_METHOD].get("correct")) for methods in examples]
    target_correct = [bool(methods[TARGET_METHOD].get("correct")) for methods in examples]
    target_self_correct = [
        bool(methods.get(TARGET_SELF_METHOD, {}).get("correct"))
        for methods in examples
        if TARGET_SELF_METHOD in methods
    ]
    target_self_accuracy = (
        sum(target_self_correct) / len(target_self_correct) if target_self_correct else None
    )

    use_repair: list[bool] = []
    for methods in examples:
        selected = methods[SELECTED_METHOD]
        if policy == "never_repair_selected":
            use_repair.append(False)
        elif policy == "repair_all_selected":
            use_repair.append(True)
        elif policy == "oracle_precheck_analysis_only":
            use_repair.append(not bool(selected.get("correct")))
        else:
            if threshold is None:
                raise ValueError("threshold is required for gated policies")
            use_repair.append(not _candidate_passes(selected, policy, threshold))

    final_correct = [
        repair if repair_flag else selected
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    ]
    repaired_help = sum(
        repair_flag and (not selected) and repair
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    )
    repaired_harm = sum(
        repair_flag and selected and (not repair)
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    )
    missed_help = sum(
        (not repair_flag) and (not selected) and repair
        for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
    )
    saved_repair_correct = sum(
        (not repair_flag) and selected
        for repair_flag, selected in zip(use_repair, selected_correct, strict=True)
    )
    n = len(examples)
    repair_all_accuracy = sum(repair_correct) / n
    selected_accuracy = sum(selected_correct) / n
    target_accuracy = sum(target_correct) / n
    repair_rate = sum(use_repair) / n

    return {
        "policy": policy,
        "threshold": threshold,
        "n": n,
        "accuracy": sum(final_correct) / n,
        "target_accuracy": target_accuracy,
        "selected_no_repair_accuracy": selected_accuracy,
        "repair_all_accuracy": repair_all_accuracy,
        "target_self_repair_accuracy": target_self_accuracy,
        "delta_vs_target": (sum(final_correct) / n) - target_accuracy,
        "delta_vs_selected_no_repair": (sum(final_correct) / n) - selected_accuracy,
        "delta_vs_repair_all": (sum(final_correct) / n) - repair_all_accuracy,
        "delta_vs_target_self_repair": (
            (sum(final_correct) / n) - target_self_accuracy
            if target_self_accuracy is not None
            else None
        ),
        "repair_application_rate": repair_rate,
        "repair_saved_rate_vs_repair_all": 1.0 - repair_rate,
        "repaired_help_count": int(repaired_help),
        "repaired_harm_count": int(repaired_harm),
        "missed_help_count": int(missed_help),
        "saved_repair_correct_count": int(saved_repair_correct),
    }


def summarize_source(path: pathlib.Path) -> SourcePolicySummary:
    records = load_jsonl(path)
    examples = _with_process_features(_eligible_examples(records))
    selected_rows = [methods[SELECTED_METHOD] for methods in examples]

    rows = [
        _summarize_policy(examples, policy="never_repair_selected", threshold=None),
        _summarize_policy(examples, policy="repair_all_selected", threshold=None),
        _summarize_policy(examples, policy="oracle_precheck_analysis_only", threshold=None),
    ]
    gate_fields = dict(GATE_FIELDS)
    gate_fields["format_and_completion_gate"] = "candidate_format_score"
    for policy, field in gate_fields.items():
        for threshold in _thresholds(selected_rows, field):
            rows.append(_summarize_policy(examples, policy=policy, threshold=threshold))
    rows.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            float(row["repair_application_rate"]),
            str(row["policy"]),
            float(row["threshold"] if row["threshold"] is not None else -9999.0),
        )
    )
    return SourcePolicySummary(source=path.name, rows=rows)


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_markdown(summaries: Sequence[SourcePolicySummary], *, top_k: int = 12) -> str:
    lines = [
        "# Test-Before-Repair Policy Analysis",
        "",
        "Policies are evaluated on existing held-out process-repair telemetry. Gated",
        "policies skip target-side repair when the selected route passes a non-oracle",
        "test, otherwise they use the already logged selected-route repair output.",
        "",
        "`oracle_precheck_analysis_only` is an upper bound for a perfect pre-repair",
        "test and must not be used as a method row.",
    ]
    for summary in summaries:
        lines.extend(
            [
                "",
                f"## {summary.source}",
                "",
                "| Policy | Threshold | Accuracy | Repair rate | Saved repair | Delta vs repair-all | Delta vs target self | Repaired help | Missed help | Saved correct |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary.rows[:top_k]:
            lines.append(
                "| {policy} | {threshold} | {accuracy} | {repair_rate} | {saved_rate} | {delta_repair} | {delta_self} | {help_count} | {missed_help} | {saved_correct} |".format(
                    policy=row["policy"],
                    threshold=_format_float(row["threshold"]),
                    accuracy=_format_float(row["accuracy"]),
                    repair_rate=_format_float(row["repair_application_rate"]),
                    saved_rate=_format_float(row["repair_saved_rate_vs_repair_all"]),
                    delta_repair=_format_float(row["delta_vs_repair_all"]),
                    delta_self=_format_float(row["delta_vs_target_self_repair"]),
                    help_count=int(row["repaired_help_count"]),
                    missed_help=int(row["missed_help_count"]),
                    saved_correct=int(row["saved_repair_correct_count"]),
                )
            )
    return "\n".join(lines) + "\n"


def build_json(summaries: Sequence[SourcePolicySummary]) -> dict[str, Any]:
    return {
        "sources": [
            {
                "source": summary.source,
                "rows": summary.rows,
            }
            for summary in summaries
        ]
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze test-before-repair policies.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    summaries = [summarize_source(pathlib.Path(path)) for path in args.inputs]
    payload = build_json(summaries)
    markdown = format_markdown(summaries, top_k=args.top_k)

    output_json = pathlib.Path(args.output_json)
    output_md = pathlib.Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
