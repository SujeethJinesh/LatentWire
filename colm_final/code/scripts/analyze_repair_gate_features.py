#!/usr/bin/env python3
"""Audit non-oracle repair-gate features on process-repair telemetry."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TARGET_METHOD = "target_alone"
SELECTED_METHOD = "selected_route_no_repair"
REPAIR_METHOD = "process_repair_selected_route"

DEFAULT_FEATURES = (
    "candidate_format_score",
    "selected_candidate_format_delta_vs_target",
    "candidate_completion_score",
    "candidate_numeric_consistency_score",
    "candidate_vote_margin",
    "candidate_vote_count",
    "candidate_answer_agreement",
    "candidate_unique_predictions",
    "candidate_unique_numeric_mention_count",
    "candidate_numeric_mention_count",
)


@dataclass(frozen=True)
class FeatureAudit:
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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [(score, label) for score, label in zip(scores, labels, strict=True) if label]
    negatives = [(score, label) for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for p_score, _ in positives:
        for n_score, _ in negatives:
            total += 1.0
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / total


def _best_preserving_gate(
    values: list[float],
    selected_correct: list[bool],
    repair_correct: list[bool],
) -> dict[str, Any]:
    thresholds = sorted(set(values))
    if not thresholds:
        return {
            "threshold": None,
            "accuracy": None,
            "repair_rate": None,
            "saved_repair_rate": None,
            "missed_help": None,
        }
    repair_all_accuracy = sum(repair_correct) / len(repair_correct)
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        # Pass means skip repair if the feature is high enough.
        use_repair = [value < threshold for value in values]
        final_correct = [
            repair if repair_flag else selected
            for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
        ]
        accuracy = sum(final_correct) / len(final_correct)
        missed_help = sum(
            (not repair_flag) and (not selected) and repair
            for repair_flag, repair, selected in zip(use_repair, repair_correct, selected_correct, strict=True)
        )
        repair_rate = sum(use_repair) / len(use_repair)
        candidate = {
            "threshold": threshold,
            "accuracy": accuracy,
            "repair_rate": repair_rate,
            "saved_repair_rate": 1.0 - repair_rate,
            "missed_help": int(missed_help),
            "delta_vs_repair_all": accuracy - repair_all_accuracy,
        }
        if accuracy >= repair_all_accuracy and (
            best is None
            or candidate["repair_rate"] < best["repair_rate"]
            or (
                candidate["repair_rate"] == best["repair_rate"]
                and candidate["threshold"] > best["threshold"]
            )
        ):
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": None,
        "accuracy": None,
        "repair_rate": None,
        "saved_repair_rate": None,
        "missed_help": None,
        "delta_vs_repair_all": None,
    }


def audit_source(path: pathlib.Path, features: Sequence[str] = DEFAULT_FEATURES) -> FeatureAudit:
    records = load_jsonl(path)
    examples = _eligible_examples(records)
    rows: list[dict[str, Any]] = []
    for feature in features:
        values: list[float] = []
        selected_correct: list[bool] = []
        repair_correct: list[bool] = []
        target_correct: list[bool] = []
        for methods in examples:
            selected = methods[SELECTED_METHOD]
            value = _to_float(selected.get(feature))
            if value is None:
                continue
            values.append(value)
            selected_correct.append(bool(selected.get("correct")))
            repair_correct.append(bool(methods[REPAIR_METHOD].get("correct")))
            target_correct.append(bool(methods[TARGET_METHOD].get("correct")))
        if not values:
            continue
        repair_help = [
            (not selected) and repair
            for selected, repair in zip(selected_correct, repair_correct, strict=True)
        ]
        repair_harm = [
            selected and (not repair)
            for selected, repair in zip(selected_correct, repair_correct, strict=True)
        ]
        best_gate = _best_preserving_gate(values, selected_correct, repair_correct)
        selected_true_values = [value for value, label in zip(values, selected_correct, strict=True) if label]
        selected_false_values = [value for value, label in zip(values, selected_correct, strict=True) if not label]
        help_values = [value for value, label in zip(values, repair_help, strict=True) if label]
        no_help_values = [value for value, label in zip(values, repair_help, strict=True) if not label]
        row = {
            "feature": feature,
            "n": len(values),
            "selected_correct_auroc": _auroc(values, selected_correct),
            "repair_help_auroc_high_means_help": _auroc(values, repair_help),
            "repair_harm_auroc_high_means_harm": _auroc(values, repair_harm),
            "selected_correct_mean": _mean(selected_true_values),
            "selected_wrong_mean": _mean(selected_false_values),
            "repair_help_mean": _mean(help_values),
            "no_repair_help_mean": _mean(no_help_values),
            "target_accuracy": sum(target_correct) / len(target_correct),
            "selected_no_repair_accuracy": sum(selected_correct) / len(selected_correct),
            "repair_all_accuracy": sum(repair_correct) / len(repair_correct),
            "repair_help_count": int(sum(repair_help)),
            "repair_harm_count": int(sum(repair_harm)),
            **{f"best_gate_{key}": value for key, value in best_gate.items()},
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -(row["best_gate_saved_repair_rate"] or -1.0),
            -(row["selected_correct_auroc"] or 0.0),
            row["feature"],
        )
    )
    return FeatureAudit(source=path.name, rows=rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_markdown(audits: Sequence[FeatureAudit], *, top_k: int = 10) -> str:
    lines = [
        "# Repair Gate Feature Audit",
        "",
        "This audit uses only selected-route telemetry fields. The best gate threshold",
        "uses the convention: repair when the feature is below the threshold, skip",
        "repair otherwise. Rows are sorted by repair saved while preserving repair-all",
        "accuracy when such a threshold exists.",
    ]
    for audit in audits:
        lines.extend(
            [
                "",
                f"## {audit.source}",
                "",
                "| Feature | N | Selected-correct AUROC | Help AUROC | Selected mean | Wrong mean | Help mean | Best threshold | Best acc | Saved repair | Missed help |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in audit.rows[:top_k]:
            lines.append(
                "| {feature} | {n} | {sel_auc} | {help_auc} | {sel_mean} | {wrong_mean} | {help_mean} | {threshold} | {acc} | {saved} | {missed} |".format(
                    feature=row["feature"],
                    n=int(row["n"]),
                    sel_auc=_fmt(row["selected_correct_auroc"]),
                    help_auc=_fmt(row["repair_help_auroc_high_means_help"]),
                    sel_mean=_fmt(row["selected_correct_mean"]),
                    wrong_mean=_fmt(row["selected_wrong_mean"]),
                    help_mean=_fmt(row["repair_help_mean"]),
                    threshold=_fmt(row["best_gate_threshold"]),
                    acc=_fmt(row["best_gate_accuracy"]),
                    saved=_fmt(row["best_gate_saved_repair_rate"]),
                    missed="-" if row["best_gate_missed_help"] is None else str(int(row["best_gate_missed_help"])),
                )
            )
    return "\n".join(lines) + "\n"


def build_json(audits: Sequence[FeatureAudit]) -> dict[str, Any]:
    return {
        "sources": [
            {
                "source": audit.source,
                "rows": audit.rows,
            }
            for audit in audits
        ]
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit repair gate telemetry features.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--features", nargs="*", default=list(DEFAULT_FEATURES))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    audits = [audit_source(pathlib.Path(path), features=args.features) for path in args.inputs]
    payload = build_json(audits)
    markdown = format_markdown(audits, top_k=args.top_k)
    output_json = pathlib.Path(args.output_json)
    output_md = pathlib.Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
