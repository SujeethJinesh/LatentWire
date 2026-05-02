"""Analyze process-repair telemetry and write paper-friendly attribution tables."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

BASELINE_METHOD = "target_alone"
TARGET_SELF_REPAIR_METHOD = "target_self_repair"
REPAIR_METHODS = ("selected_route_no_repair", "target_self_repair", "process_repair_selected_route")


@dataclass(frozen=True)
class SourceSummary:
    source: str
    total_rows: int
    method_summaries: dict[str, dict[str, Any]]
    paired_summaries: dict[str, dict[str, Any]]
    target_self_paired_summaries: dict[str, dict[str, Any]]


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_method(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)
    return grouped


def _paired_index_rows(
    records: list[dict[str, Any]],
    *,
    method: str,
    baseline_method: str = BASELINE_METHOD,
) -> list[dict[str, Any]]:
    by_method = _group_by_method(records)
    method_rows = {int(row["index"]): row for row in by_method.get(method, [])}
    baseline_rows = {int(row["index"]): row for row in by_method.get(baseline_method, [])}
    indices = sorted(set(method_rows) & set(baseline_rows))
    paired: list[dict[str, Any]] = []
    for idx in indices:
        method_row = method_rows[idx]
        baseline_row = baseline_rows[idx]
        paired.append(
            {
                "index": idx,
                "method": method,
                "correct": bool(method_row.get("correct")),
                "baseline_correct": bool(baseline_row.get("correct")),
                "flip": _paired_flip_label(
                    bool(method_row.get("correct")),
                    bool(baseline_row.get("correct")),
                ),
                "example_id": method_row.get("example_id") or baseline_row.get("example_id"),
            }
        )
    return paired


def _paired_flip_label(method_correct: bool, baseline_correct: bool) -> str:
    if method_correct and baseline_correct:
        return "both_correct"
    if method_correct and not baseline_correct:
        return "method_only"
    if baseline_correct and not method_correct:
        return "baseline_only"
    return "both_wrong"


def _paired_prediction_metrics(
    records: list[dict[str, Any]],
    method: str,
    baseline: str,
    *,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    by_method: dict[str, dict[int, bool]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), {})[int(record["index"])] = bool(record["correct"])
    method_rows = by_method.get(method, {})
    baseline_rows = by_method.get(baseline, {})
    indices = sorted(set(method_rows) & set(baseline_rows))
    if not indices:
        return {}

    diffs = [
        (1.0 if method_rows[idx] else 0.0) - (1.0 if baseline_rows[idx] else 0.0)
        for idx in indices
    ]
    method_only = sum(method_rows[idx] and not baseline_rows[idx] for idx in indices)
    baseline_only = sum(baseline_rows[idx] and not method_rows[idx] for idx in indices)
    denom = method_only + baseline_only
    if denom:
        chi2 = (max(abs(method_only - baseline_only) - 1.0, 0.0) ** 2) / denom
        p_value = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        chi2 = 0.0
        p_value = 1.0

    rng = random.Random(0)
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lo_idx = int(0.025 * (len(boot) - 1))
    hi_idx = int(0.975 * (len(boot) - 1))
    return {
        "paired_n": float(len(indices)),
        "delta_accuracy": float(sum(diffs) / len(diffs)),
        "method_only": float(method_only),
        "baseline_only": float(baseline_only),
        "both_correct": float(sum(method_rows[idx] and baseline_rows[idx] for idx in indices)),
        "both_wrong": float(sum((not method_rows[idx]) and (not baseline_rows[idx]) for idx in indices)),
        "mcnemar_chi2": float(chi2 if denom else 0.0),
        "mcnemar_p": float(p_value),
        "bootstrap_delta_low": float(boot[lo_idx]),
        "bootstrap_delta_high": float(boot[hi_idx]),
    }


def _bootstrap_interval(
    values: list[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    lo_idx = int(0.025 * (len(samples) - 1))
    hi_idx = int(0.975 * (len(samples) - 1))
    return float(samples[lo_idx]), float(samples[hi_idx])


def _accuracy_summary(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    correct = [1.0 if bool(row.get("correct")) else 0.0 for row in rows]
    accuracy = sum(correct) / len(correct) if correct else 0.0
    lo, hi = _bootstrap_interval(correct, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "count": len(rows),
        "accuracy": float(accuracy),
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
    }


def _rate_summary(
    rows: list[dict[str, Any]],
    *,
    field: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    values = [1.0 if bool(row.get(field)) else 0.0 for row in rows]
    if not values:
        return {"rate": None, "rate_ci_low": None, "rate_ci_high": None}
    rate = sum(values) / len(values)
    lo, hi = _bootstrap_interval(values, n_bootstrap=n_bootstrap, seed=seed)
    return {"rate": float(rate), "rate_ci_low": lo, "rate_ci_high": hi}


def _method_summary(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    summary = _accuracy_summary(rows, n_bootstrap=n_bootstrap, seed=seed)
    pre_rows = [row for row in rows if "repair_pre_correct" in row]
    if pre_rows:
        pre = _rate_summary(
            pre_rows,
            field="repair_pre_correct",
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        )
        changed = _rate_summary(
            pre_rows,
            field="repair_changed_answer",
            n_bootstrap=n_bootstrap,
            seed=seed + 2,
        )
        help_values = [
            1.0 if (not bool(row.get("repair_pre_correct"))) and bool(row.get("correct")) else 0.0
            for row in pre_rows
        ]
        harm_values = [
            1.0 if bool(row.get("repair_pre_correct")) and not bool(row.get("correct")) else 0.0
            for row in pre_rows
        ]
        target_values = [
            1.0 if row.get("repair_selected_candidate_source") == "target" else 0.0
            for row in pre_rows
        ]
        oracle_values = [
            1.0 if bool(row.get("repair_full_oracle_correct")) else 0.0
            for row in pre_rows
        ]
        help_rate = _bootstrap_interval(help_values, n_bootstrap=n_bootstrap, seed=seed + 3)
        harm_rate = _bootstrap_interval(harm_values, n_bootstrap=n_bootstrap, seed=seed + 4)
        target_rate = _bootstrap_interval(target_values, n_bootstrap=n_bootstrap, seed=seed + 5)
        oracle_rate = _bootstrap_interval(oracle_values, n_bootstrap=n_bootstrap, seed=seed + 6)
        summary.update(
            {
                "pre_repair_accuracy": pre["rate"],
                "pre_repair_accuracy_ci_low": pre["rate_ci_low"],
                "pre_repair_accuracy_ci_high": pre["rate_ci_high"],
                "changed_answer_rate": changed["rate"],
                "changed_answer_rate_ci_low": changed["rate_ci_low"],
                "changed_answer_rate_ci_high": changed["rate_ci_high"],
                "repair_help_rate": sum(help_values) / len(help_values),
                "repair_help_rate_ci_low": help_rate[0],
                "repair_help_rate_ci_high": help_rate[1],
                "repair_harm_rate": sum(harm_values) / len(harm_values),
                "repair_harm_rate_ci_low": harm_rate[0],
                "repair_harm_rate_ci_high": harm_rate[1],
                "target_selection_rate": sum(target_values) / len(target_values),
                "target_selection_rate_ci_low": target_rate[0],
                "target_selection_rate_ci_high": target_rate[1],
                "full_oracle": sum(oracle_values) / len(oracle_values),
                "full_oracle_ci_low": oracle_rate[0],
                "full_oracle_ci_high": oracle_rate[1],
            }
        )
    else:
        summary.update(
            {
                "pre_repair_accuracy": None,
                "pre_repair_accuracy_ci_low": None,
                "pre_repair_accuracy_ci_high": None,
                "changed_answer_rate": None,
                "changed_answer_rate_ci_low": None,
                "changed_answer_rate_ci_high": None,
                "repair_help_rate": None,
                "repair_help_rate_ci_low": None,
                "repair_help_rate_ci_high": None,
                "repair_harm_rate": None,
                "repair_harm_rate_ci_low": None,
                "repair_harm_rate_ci_high": None,
                "target_selection_rate": None,
                "target_selection_rate_ci_low": None,
                "target_selection_rate_ci_high": None,
                "full_oracle": None,
                "full_oracle_ci_low": None,
                "full_oracle_ci_high": None,
            }
        )
    return summary


def _paired_summary(
    records: list[dict[str, Any]],
    *,
    method: str,
    baseline_method: str = BASELINE_METHOD,
    n_bootstrap: int,
) -> dict[str, Any] | None:
    if method == baseline_method:
        return None
    metrics = _paired_prediction_metrics(records, method, baseline_method, n_bootstrap=n_bootstrap)
    if not metrics:
        return None
    paired_rows = _paired_index_rows(records, method=method, baseline_method=baseline_method)
    return {
        "paired_n": int(metrics["paired_n"]),
        "baseline_method": baseline_method,
        "delta_accuracy": float(metrics["delta_accuracy"]),
        "delta_accuracy_ci_low": float(metrics["bootstrap_delta_low"]),
        "delta_accuracy_ci_high": float(metrics["bootstrap_delta_high"]),
        "method_only": int(metrics["method_only"]),
        "baseline_only": int(metrics["baseline_only"]),
        "both_correct": int(metrics["both_correct"]),
        "both_wrong": int(metrics["both_wrong"]),
        "paired_rows": paired_rows,
    }


def summarize_source(
    path: pathlib.Path,
    *,
    n_bootstrap: int = 2000,
) -> SourceSummary:
    records = _load_jsonl(path)
    grouped = _group_by_method(records)
    method_summaries = {
        method: _method_summary(rows, n_bootstrap=n_bootstrap, seed=0)
        for method, rows in sorted(grouped.items())
    }
    paired_summaries = {
        method: summary
        for method in sorted(grouped)
        if (summary := _paired_summary(records, method=method, n_bootstrap=n_bootstrap)) is not None
    }
    target_self_paired_summaries = {
        method: summary
        for method in sorted(grouped)
        if TARGET_SELF_REPAIR_METHOD in grouped
        and method not in {BASELINE_METHOD, TARGET_SELF_REPAIR_METHOD}
        and (
            summary := _paired_summary(
                records,
                method=method,
                baseline_method=TARGET_SELF_REPAIR_METHOD,
                n_bootstrap=n_bootstrap,
            )
        )
        is not None
    }
    return SourceSummary(
        source=path.name,
        total_rows=len(records),
        method_summaries=method_summaries,
        paired_summaries=paired_summaries,
        target_self_paired_summaries=target_self_paired_summaries,
    )


def _format_rate(value: Any, low: Any, high: Any) -> str:
    if value is None:
        return "-"
    if low is None or high is None:
        return f"{float(value):.4f}"
    return f"{float(value):.4f} [{float(low):.4f}, {float(high):.4f}]"


def _format_count(value: Any) -> str:
    return "-" if value is None else str(int(value))


def _method_order(method: str) -> tuple[int, str]:
    if method == BASELINE_METHOD:
        return (0, method)
    if method in REPAIR_METHODS:
        return (1 + REPAIR_METHODS.index(method), method)
    return (len(REPAIR_METHODS) + 1, method)


def format_markdown(summaries: list[SourceSummary]) -> str:
    lines = [
        "# Process Repair Attribution",
        "",
        "Deterministic bootstrap intervals use example-level resampling within each source file.",
    ]
    for summary in summaries:
        lines.extend(
            [
                "",
                f"## {summary.source}",
                "",
                "| Method | N | Accuracy | Pre-repair | Changed answer | Repair help | Repair harm | Target selected | Full oracle | Method-only | Baseline-only | Both correct | Both wrong | Delta vs target | Delta vs target self-repair |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in sorted(summary.method_summaries, key=_method_order):
            method_stats = summary.method_summaries[method]
            paired_stats = summary.paired_summaries.get(method, {})
            target_self_stats = summary.target_self_paired_summaries.get(method, {})
            lines.append(
                "| {method} | {n} | {acc} | {pre} | {changed} | {help_rate} | {harm_rate} | {target} | {oracle} | {method_only} | {baseline_only} | {both_correct} | {both_wrong} | {delta} | {target_self_delta} |".format(
                    method=method,
                    n=int(method_stats["count"]),
                    acc=_format_rate(
                        method_stats["accuracy"],
                        method_stats["accuracy_ci_low"],
                        method_stats["accuracy_ci_high"],
                    ),
                    pre=_format_rate(
                        method_stats["pre_repair_accuracy"],
                        method_stats["pre_repair_accuracy_ci_low"],
                        method_stats["pre_repair_accuracy_ci_high"],
                    ),
                    changed=_format_rate(
                        method_stats["changed_answer_rate"],
                        method_stats["changed_answer_rate_ci_low"],
                        method_stats["changed_answer_rate_ci_high"],
                    ),
                    help_rate=_format_rate(
                        method_stats["repair_help_rate"],
                        method_stats["repair_help_rate_ci_low"],
                        method_stats["repair_help_rate_ci_high"],
                    ),
                    harm_rate=_format_rate(
                        method_stats["repair_harm_rate"],
                        method_stats["repair_harm_rate_ci_low"],
                        method_stats["repair_harm_rate_ci_high"],
                    ),
                    target=_format_rate(
                        method_stats["target_selection_rate"],
                        method_stats["target_selection_rate_ci_low"],
                        method_stats["target_selection_rate_ci_high"],
                    ),
                    oracle=_format_rate(
                        method_stats["full_oracle"],
                        method_stats["full_oracle_ci_low"],
                        method_stats["full_oracle_ci_high"],
                    ),
                    method_only=_format_count(paired_stats.get("method_only")),
                    baseline_only=_format_count(paired_stats.get("baseline_only")),
                    both_correct=_format_count(paired_stats.get("both_correct")),
                    both_wrong=_format_count(paired_stats.get("both_wrong")),
                    delta=_format_rate(
                        paired_stats.get("delta_accuracy"),
                        paired_stats.get("delta_accuracy_ci_low"),
                        paired_stats.get("delta_accuracy_ci_high"),
                    ),
                    target_self_delta=_format_rate(
                        target_self_stats.get("delta_accuracy"),
                        target_self_stats.get("delta_accuracy_ci_low"),
                        target_self_stats.get("delta_accuracy_ci_high"),
                    ),
                )
            )
    return "\n".join(lines) + "\n"


def build_json(summaries: list[SourceSummary]) -> dict[str, Any]:
    return {
        "sources": [
            {
                "source": summary.source,
                "total_rows": summary.total_rows,
                "method_summaries": summary.method_summaries,
                "paired_summaries": summary.paired_summaries,
                "target_self_paired_summaries": summary.target_self_paired_summaries,
            }
            for summary in summaries
        ]
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze process-repair telemetry attribution.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more telemetry JSONL files.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = [summarize_source(pathlib.Path(path), n_bootstrap=args.n_bootstrap) for path in args.inputs]
    json_payload = build_json(summaries)
    md_text = format_markdown(summaries)

    output_json = pathlib.Path(args.output_json)
    output_md = pathlib.Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(md_text, encoding="utf-8")


if __name__ == "__main__":
    main()
