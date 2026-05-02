#!/usr/bin/env python3
"""Audit text-derived process gates on process-repair telemetry."""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TARGET_METHOD = "target_alone"
SELECTED_METHOD = "selected_route_no_repair"
REPAIR_METHOD = "process_repair_selected_route"

PROCESS_FEATURES = (
    "process_completeness_score",
    "format_plus_process_score",
    "equation_valid_fraction",
    "valid_equation_count",
    "equation_count",
    "answer_marker_score",
    "finished_tail_score",
    "prediction_tail_match_score",
    "reasoning_step_count",
)

_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_EQUATION_RE = re.compile(
    r"(?P<lhs>[-+*/().\d\s]+[*+/\-][-+*/().\d\s]*)=\s*(?P<rhs>[-+]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class ProcessGateAudit:
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


def _safe_eval_numeric(expr: str) -> float | None:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def visit(item: ast.AST) -> float | None:
        if isinstance(item, ast.Expression):
            return visit(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = visit(item.operand)
            if value is None:
                return None
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(
            item.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(item.left)
            right = visit(item.right)
            if left is None or right is None:
                return None
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if isinstance(item.op, ast.Mult):
                return left * right
            if abs(right) < 1e-12:
                return None
            return left / right
        return None

    return visit(node)


def _normalize_math_text(text: str) -> str:
    normalized = text.replace(",", "")
    normalized = normalized.replace("\\times", "*").replace("\\cdot", "*")
    normalized = normalized.replace("x", "*").replace("X", "*")
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = normalized.replace("$", " ")
    return normalized


def _equation_stats(text: str) -> tuple[int, int]:
    normalized = _normalize_math_text(text)
    total = 0
    valid = 0
    for match in _EQUATION_RE.finditer(normalized):
        lhs = match.group("lhs").strip()
        rhs_text = match.group("rhs").strip()
        lhs_value = _safe_eval_numeric(lhs)
        rhs_value = _safe_eval_numeric(rhs_text)
        if lhs_value is None or rhs_value is None:
            continue
        total += 1
        tolerance = max(1e-6, 1e-6 * max(abs(lhs_value), abs(rhs_value), 1.0))
        if abs(lhs_value - rhs_value) <= tolerance:
            valid += 1
    return total, valid


def _line_step_count(text: str) -> int:
    useful_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("---")
    ]
    bulletish = sum(line.startswith(("-", "*", "1.", "2.", "3.")) for line in useful_lines)
    equations = text.count("=")
    connective = len(re.findall(r"\b(therefore|so|then|because|thus|hence)\b", text, re.I))
    return int(max(len(useful_lines), bulletish + equations + connective))


def _last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_process_features(selected: dict[str, Any]) -> dict[str, float]:
    text = str(selected.get("prediction") or selected.get("repair_pre_prediction") or "")
    stripped = text.strip()
    equation_count, valid_equation_count = _equation_stats(stripped)
    equation_valid_fraction = (
        valid_equation_count / equation_count if equation_count else 0.0
    )
    lower = stripped.lower()
    answer_marker_score = float(
        any(marker in lower for marker in ("final answer", "answer:", "####", "<"))
    )
    finished_tail_score = float(
        bool(stripped)
        and not stripped.endswith(("-", ":", ",", "=", "+", "*", "/", "\\"))
        and not re.search(r"\b(is|are|equals|becomes|remaining|therefore)\s*$", lower)
    )
    normalized_prediction = str(selected.get("normalized_prediction") or "")
    tail_number = _last_number(stripped)
    prediction_tail_match_score = float(
        bool(normalized_prediction) and tail_number == normalized_prediction
    )
    reasoning_step_count = float(_line_step_count(stripped))
    equation_score = min(2.0, float(valid_equation_count))
    process_completeness_score = (
        2.0 * answer_marker_score
        + 2.0 * finished_tail_score
        + equation_score
        + prediction_tail_match_score
        + min(2.0, reasoning_step_count / 3.0)
        + equation_valid_fraction
    )
    candidate_format_score = _to_float(selected.get("candidate_format_score")) or 0.0
    return {
        "process_completeness_score": float(process_completeness_score),
        "format_plus_process_score": float(candidate_format_score + process_completeness_score),
        "equation_valid_fraction": float(equation_valid_fraction),
        "valid_equation_count": float(valid_equation_count),
        "equation_count": float(equation_count),
        "answer_marker_score": answer_marker_score,
        "finished_tail_score": finished_tail_score,
        "prediction_tail_match_score": prediction_tail_match_score,
        "reasoning_step_count": reasoning_step_count,
    }


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
    return sum(values) / len(values) if values else None


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def _best_preserving_gate(
    values: list[float],
    selected_correct: list[bool],
    repair_correct: list[bool],
) -> dict[str, Any]:
    thresholds = sorted(set(values))
    repair_all_accuracy = sum(repair_correct) / len(repair_correct)
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        use_repair = [value < threshold for value in values]
        final_correct = [
            repair if repair_flag else selected
            for repair_flag, repair, selected in zip(
                use_repair, repair_correct, selected_correct, strict=True
            )
        ]
        accuracy = sum(final_correct) / len(final_correct)
        missed_help = sum(
            (not repair_flag) and (not selected) and repair
            for repair_flag, repair, selected in zip(
                use_repair, repair_correct, selected_correct, strict=True
            )
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
    return best or {
        "threshold": None,
        "accuracy": None,
        "repair_rate": None,
        "saved_repair_rate": None,
        "missed_help": None,
        "delta_vs_repair_all": None,
    }


def audit_source(
    path: pathlib.Path,
    features: Sequence[str] = PROCESS_FEATURES,
) -> ProcessGateAudit:
    records = load_jsonl(path)
    examples = _eligible_examples(records)
    rows: list[dict[str, Any]] = []
    selected_feature_rows = [
        extract_process_features(methods[SELECTED_METHOD])
        for methods in examples
    ]
    selected_correct = [bool(methods[SELECTED_METHOD].get("correct")) for methods in examples]
    repair_correct = [bool(methods[REPAIR_METHOD].get("correct")) for methods in examples]
    target_correct = [bool(methods[TARGET_METHOD].get("correct")) for methods in examples]
    repair_help = [
        (not selected) and repair
        for selected, repair in zip(selected_correct, repair_correct, strict=True)
    ]
    repair_harm = [
        selected and (not repair)
        for selected, repair in zip(selected_correct, repair_correct, strict=True)
    ]

    for feature in features:
        values = [row[feature] for row in selected_feature_rows if feature in row]
        if len(values) != len(examples) or not values:
            continue
        best_gate = _best_preserving_gate(values, selected_correct, repair_correct)
        selected_true_values = [
            value for value, label in zip(values, selected_correct, strict=True) if label
        ]
        selected_false_values = [
            value for value, label in zip(values, selected_correct, strict=True) if not label
        ]
        help_values = [value for value, label in zip(values, repair_help, strict=True) if label]
        no_help_values = [value for value, label in zip(values, repair_help, strict=True) if not label]
        rows.append(
            {
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
        )
    rows.sort(
        key=lambda row: (
            -(row["best_gate_saved_repair_rate"] or -1.0),
            -(row["selected_correct_auroc"] or 0.0),
            row["feature"],
        )
    )
    return ProcessGateAudit(source=path.name, rows=rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_markdown(audits: Sequence[ProcessGateAudit], *, top_k: int = 12) -> str:
    lines = [
        "# Process Gate Feature Audit",
        "",
        "This audit derives non-oracle process features from the selected-route",
        "solution text, then applies the same high-score-means-skip-repair gate",
        "used by the cheaper metadata audit.",
    ]
    for audit in audits:
        lines.extend(
            [
                "",
                f"## {audit.source}",
                "",
                "| Feature | N | Selected AUROC | Help AUROC | Selected mean | Wrong mean | Help mean | Best threshold | Best acc | Saved repair | Missed help |",
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
                    missed=(
                        "-"
                        if row["best_gate_missed_help"] is None
                        else str(int(row["best_gate_missed_help"]))
                    ),
                )
            )
    return "\n".join(lines) + "\n"


def build_json(audits: Sequence[ProcessGateAudit]) -> dict[str, Any]:
    return {"sources": [{"source": audit.source, "rows": audit.rows} for audit in audits]}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit process-text repair gate features.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--features", nargs="*", default=list(PROCESS_FEATURES))
    parser.add_argument("--top-k", type=int, default=12)
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
