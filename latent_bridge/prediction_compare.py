"""Utilities for paired comparisons across prediction JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluate import paired_prediction_metrics


BASELINE_METHODS = {"target_alone", "text_to_text", "source_alone", "routing"}


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def methods_in_records(records: list[dict[str, Any]]) -> set[str]:
    return {str(record["method"]) for record in records}


def common_methods(
    candidate_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    method_prefix: str | None = None,
    include_baseline_methods: bool = False,
) -> list[str]:
    methods = methods_in_records(candidate_records) & methods_in_records(baseline_records)
    if method_prefix is not None:
        methods = {method for method in methods if method.startswith(method_prefix)}
    if not include_baseline_methods:
        methods = methods - BASELINE_METHODS
    return sorted(methods)


def compare_prediction_records(
    candidate_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    method: str,
    baseline_method: str | None = None,
    candidate_label: str = "candidate",
    baseline_label: str = "baseline",
    n_bootstrap: int = 1000,
) -> dict[str, float | str]:
    baseline_method = baseline_method or method
    candidate_rows = {
        int(record["index"]): bool(record["correct"])
        for record in candidate_records
        if str(record["method"]) == method
    }
    baseline_rows = {
        int(record["index"]): bool(record["correct"])
        for record in baseline_records
        if str(record["method"]) == baseline_method
    }
    if not candidate_rows:
        raise ValueError(f"Method not found in candidate records: {method}")
    if not baseline_rows:
        raise ValueError(f"Method not found in baseline records: {baseline_method}")

    indices = sorted(set(candidate_rows) & set(baseline_rows))
    if not indices:
        raise ValueError(f"No paired examples for method: {method}")

    paired_records = []
    for idx in indices:
        paired_records.append(
            {"index": idx, "method": candidate_label, "correct": candidate_rows[idx]}
        )
        paired_records.append(
            {"index": idx, "method": baseline_label, "correct": baseline_rows[idx]}
        )

    stats = paired_prediction_metrics(
        paired_records,
        candidate_label,
        baseline_label,
        n_bootstrap=n_bootstrap,
    )
    candidate_accuracy = sum(candidate_rows[idx] for idx in indices) / len(indices)
    baseline_accuracy = sum(baseline_rows[idx] for idx in indices) / len(indices)
    return {
        "method": method if baseline_method == method else f"{method} vs {baseline_method}",
        "candidate_method": method,
        "baseline_method": baseline_method,
        "candidate_label": candidate_label,
        "baseline_label": baseline_label,
        "candidate_accuracy": float(candidate_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        **stats,
    }


def compare_prediction_files(
    candidate_path: str | Path,
    baseline_path: str | Path,
    *,
    methods: list[str] | None = None,
    method_prefix: str | None = None,
    include_baseline_methods: bool = False,
    candidate_label: str = "candidate",
    baseline_label: str = "baseline",
    n_bootstrap: int = 1000,
) -> list[dict[str, float | str]]:
    candidate_records = load_prediction_records(candidate_path)
    baseline_records = load_prediction_records(baseline_path)
    selected_methods = methods or common_methods(
        candidate_records,
        baseline_records,
        method_prefix=method_prefix,
        include_baseline_methods=include_baseline_methods,
    )
    if not selected_methods:
        raise ValueError("No methods selected for comparison")
    return [
        compare_prediction_records(
            candidate_records,
            baseline_records,
            method=method,
            candidate_label=candidate_label,
            baseline_label=baseline_label,
            n_bootstrap=n_bootstrap,
        )
        for method in selected_methods
    ]


def write_jsonl(rows: list[dict[str, float | str]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def format_markdown(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "| Method | Candidate Acc | Baseline Acc | Delta | Cand Only | Base Only | 95% Bootstrap Delta | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {cand:.4f} | {base:.4f} | {delta:+.4f} | {cand_only:.0f} | "
            "{base_only:.0f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} |".format(
                method=row["method"],
                cand=float(row["candidate_accuracy"]),
                base=float(row["baseline_accuracy"]),
                delta=float(row["delta_accuracy"]),
                cand_only=float(row["method_only"]),
                base_only=float(row["baseline_only"]),
                lo=float(row["bootstrap_delta_low"]),
                hi=float(row["bootstrap_delta_high"]),
                p=float(row["mcnemar_p"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_markdown(rows: list[dict[str, float | str]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown(rows), encoding="utf-8")
