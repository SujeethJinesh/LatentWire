#!/usr/bin/env python3
"""Rank SVAMP32 gate-sweep rows against the clean residual target set.

This is a matched-only candidate-search helper.  It does not establish source
necessity; it identifies which exact gate rows are worth spending controls on.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.harness_common import _has_numeric_extraction


@dataclass(frozen=True)
class SweepConfig:
    expected_n: int = 32
    min_numeric_coverage: int = 31
    min_clean_residual_recovered: int = 2
    target_self_correct: int = 14


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ordered_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row["example_id"]) for row in records]


def _correct_ids(records: Sequence[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in records if bool(row.get("correct"))}


def _numeric_coverage(records: Sequence[dict[str, Any]]) -> int:
    return sum(int(_has_numeric_extraction(str(row.get("prediction", "")))) for row in records)


def _group_by_raw_method(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["method"]), []).append(dict(row))
    return grouped


def _clean_ids(target_set_payload: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in target_set_payload.get("ids", {}).get("clean_residual_targets", [])
    }


def _teacher_only_ids(target_records: Sequence[dict[str, Any]], teacher_records: Sequence[dict[str, Any]]) -> set[str]:
    return _correct_ids(teacher_records) - _correct_ids(target_records)


def _validate_reference(
    *,
    target_records: Sequence[dict[str, Any]],
    teacher_records: Sequence[dict[str, Any]],
    target_set_payload: dict[str, Any],
    config: SweepConfig,
) -> tuple[list[str], set[str]]:
    issues: list[str] = []
    reference_ids = _ordered_ids(target_records)
    if len(reference_ids) != config.expected_n:
        issues.append(f"target n={len(reference_ids)} != expected_n={config.expected_n}")
    if len(reference_ids) != len(set(reference_ids)):
        issues.append("target has duplicate example_id values")
    if _ordered_ids(teacher_records) != reference_ids:
        issues.append("teacher exact ordered ID parity is false")
    teacher_only = _teacher_only_ids(target_records, teacher_records)
    target_set_teacher_only = {
        str(value) for value in target_set_payload.get("ids", {}).get("teacher_only", [])
    }
    clean = _clean_ids(target_set_payload)
    if teacher_only != target_set_teacher_only:
        issues.append("target_set.ids.teacher_only does not match target/teacher rows")
    if not clean:
        issues.append("target_set has no clean_residual_targets")
    if not clean.issubset(teacher_only):
        issues.append("clean_residual_targets is not a subset of teacher-only IDs")
    return issues, clean


def analyze_sweep(
    *,
    target_records: Sequence[dict[str, Any]],
    teacher_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    target_set_payload: dict[str, Any],
    config: SweepConfig,
) -> dict[str, Any]:
    reference_ids = _ordered_ids(target_records)
    validation_issues, clean = _validate_reference(
        target_records=target_records,
        teacher_records=teacher_records,
        target_set_payload=target_set_payload,
        config=config,
    )
    if validation_issues:
        raise ValueError("SVAMP32 gate-sweep validation failed: " + "; ".join(validation_issues))

    target_correct = _correct_ids(target_records)
    teacher_correct = _correct_ids(teacher_records)
    teacher_only = teacher_correct - target_correct
    rows: list[dict[str, Any]] = []
    for method, records in _group_by_raw_method(candidate_records).items():
        ids = _ordered_ids(records)
        duplicate_ids = sorted({example_id for example_id in ids if ids.count(example_id) > 1})
        correct = _correct_ids(records)
        teacher_recovered = correct & teacher_only
        clean_recovered = correct & clean
        losses_vs_target = target_correct - correct
        wins_vs_target = correct - target_correct
        exact_parity = ids == reference_ids
        numeric_coverage = _numeric_coverage(records)
        status = "matched_candidate_for_controls" if (
            len(records) == config.expected_n
            and not duplicate_ids
            and exact_parity
            and numeric_coverage >= config.min_numeric_coverage
            and len(clean_recovered) >= config.min_clean_residual_recovered
        ) else "matched_candidate_below_clean_gate"
        rows.append(
            {
                "method": method,
                "status": status,
                "n": len(records),
                "correct": len(correct),
                "accuracy": float(len(correct) / max(len(records), 1)),
                "delta_vs_target": len(correct) - len(target_correct),
                "delta_vs_target_self_repair": len(correct) - config.target_self_correct,
                "exact_ordered_id_parity": exact_parity,
                "duplicate_example_ids": duplicate_ids,
                "numeric_extraction_coverage": numeric_coverage,
                "teacher_only_recovered_count": len(teacher_recovered),
                "teacher_only_recovered_ids": sorted(teacher_recovered),
                "clean_residual_recovered_count": len(clean_recovered),
                "clean_residual_recovered_ids": sorted(clean_recovered),
                "wins_vs_target_count": len(wins_vs_target),
                "wins_vs_target_ids": sorted(wins_vs_target),
                "losses_vs_target_count": len(losses_vs_target),
                "losses_vs_target_ids": sorted(losses_vs_target),
                "oracle_target_self_plus_clean_candidate_bound": (
                    config.target_self_correct + len(clean_recovered)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["clean_residual_recovered_count"]),
            int(row["correct"]),
            -int(row["losses_vs_target_count"]),
            str(row["method"]),
        ),
        reverse=True,
    )
    passing = [row["method"] for row in rows if row["status"] == "matched_candidate_for_controls"]
    return {
        "date": str(date.today()),
        "status": (
            "matched_gate_candidate_for_controls" if passing else "no_matched_gate_candidate_for_controls"
        ),
        "passing_methods": passing,
        "config": {
            "expected_n": config.expected_n,
            "min_numeric_coverage": config.min_numeric_coverage,
            "min_clean_residual_recovered": config.min_clean_residual_recovered,
            "target_self_correct": config.target_self_correct,
        },
        "reference": {
            "target_correct": len(target_correct),
            "teacher_correct": len(teacher_correct),
            "teacher_only_count": len(teacher_only),
            "clean_residual_target_count": len(clean),
            "clean_residual_target_ids": sorted(clean),
        },
        "rows": rows,
    }


def write_markdown(payload: dict[str, Any], output_md: pathlib.Path) -> None:
    ref = payload["reference"]
    lines = [
        "# SVAMP32 Gate Sweep Clean-Target Readout",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- target: `{ref['target_correct']}/32`",
        f"- C2C teacher: `{ref['teacher_correct']}/32`",
        f"- teacher-only IDs: `{ref['teacher_only_count']}`",
        f"- clean residual targets: `{ref['clean_residual_target_count']}`",
        "",
        "## Rows",
        "",
        "| Method | Status | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses | Oracle self+clean bound |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {method} | `{status}` | {correct}/32 | {clean} | {teacher_only} | {delta:+d} | {losses} | {bound}/32 |".format(
                method=row["method"],
                status=row["status"],
                correct=int(row["correct"]),
                clean=int(row["clean_residual_recovered_count"]),
                teacher_only=int(row["teacher_only_recovered_count"]),
                delta=int(row["delta_vs_target_self_repair"]),
                losses=int(row["losses_vs_target_count"]),
                bound=int(row["oracle_target_self_plus_clean_candidate_bound"]),
            )
        )
    lines.extend(["", "## Clean IDs By Row", ""])
    for row in payload["rows"]:
        ids = ", ".join(f"`{value}`" for value in row["clean_residual_recovered_ids"]) or "none"
        lines.append(f"- `{row['method']}`: {ids}")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--expected-n", type=int, default=SweepConfig.expected_n)
    parser.add_argument("--min-numeric-coverage", type=int, default=SweepConfig.min_numeric_coverage)
    parser.add_argument(
        "--min-clean-residual-recovered",
        type=int,
        default=SweepConfig.min_clean_residual_recovered,
    )
    parser.add_argument("--target-self-correct", type=int, default=SweepConfig.target_self_correct)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    target_path = _resolve(args.target_jsonl)
    teacher_path = _resolve(args.teacher_jsonl)
    candidate_path = _resolve(args.candidate_jsonl)
    target_set_path = _resolve(args.target_set_json)
    config = SweepConfig(
        expected_n=args.expected_n,
        min_numeric_coverage=args.min_numeric_coverage,
        min_clean_residual_recovered=args.min_clean_residual_recovered,
        target_self_correct=args.target_self_correct,
    )
    payload = analyze_sweep(
        target_records=_read_jsonl(target_path),
        teacher_records=_read_jsonl(teacher_path),
        candidate_records=_read_jsonl(candidate_path),
        target_set_payload=_load_json(target_set_path),
        config=config,
    )
    payload["artifacts"] = {
        "target_jsonl": _display_path(target_path),
        "teacher_jsonl": _display_path(teacher_path),
        "candidate_jsonl": _display_path(candidate_path),
        "target_set_json": _display_path(target_set_path),
    }
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, output_md)
    return payload


if __name__ == "__main__":
    main()
