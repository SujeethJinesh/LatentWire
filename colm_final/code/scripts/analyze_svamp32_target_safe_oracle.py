#!/usr/bin/env python3
"""Score target-safe candidate oracles against source-destroying controls."""

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

from scripts import harness_common as harness


@dataclass(frozen=True)
class RowSpec:
    label: str
    path: pathlib.Path
    method: str


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


def _parse_spec(spec: str) -> RowSpec:
    if "=" not in spec:
        raise ValueError(f"Expected label=path=...,method=... spec, got {spec!r}")
    label, raw_fields = spec.split("=", 1)
    fields: dict[str, str] = {}
    for item in raw_fields.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value in row spec {spec!r}; got {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    if not label or not fields.get("path") or not fields.get("method"):
        raise ValueError(f"Spec needs label, path, and method: {spec!r}")
    return RowSpec(label=label, path=_resolve(fields["path"]), method=fields["method"])


def _records_for_method(path: pathlib.Path, method: str) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(_read_jsonl(path))
    if method not in grouped:
        raise KeyError(f"Method {method!r} not found in {path}: {sorted(grouped)}")
    return [dict(row) for row in grouped[method]]


def _ordered_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row["example_id"]) for row in records]


def _subset_reference_order(
    records: Sequence[dict[str, Any]],
    reference_ids: Sequence[str],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in seen:
            duplicates.add(example_id)
        seen.add(example_id)
        by_id[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Missing reference IDs: {missing}")
    return [by_id[example_id] for example_id in reference_ids]


def _correct_ids(records: Sequence[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in records if bool(row.get("correct"))}


def _load_spec_rows(
    specs: Sequence[str],
    reference_ids: Sequence[str],
) -> tuple[list[RowSpec], dict[str, list[dict[str, Any]]]]:
    parsed = [_parse_spec(spec) for spec in specs]
    rows: dict[str, list[dict[str, Any]]] = {}
    for spec in parsed:
        original = _records_for_method(spec.path, spec.method)
        rows[spec.label] = _subset_reference_order(original, reference_ids)
    return parsed, rows


def _clean_ids(path: pathlib.Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(value) for value in payload.get("ids", {}).get("clean_residual_targets", [])}


def _summary_for_rows(
    *,
    label: str,
    spec: RowSpec | None,
    records: Sequence[dict[str, Any]],
    reference_ids: Sequence[str],
    target_correct: set[str],
    teacher_correct: set[str],
    teacher_only: set[str],
    clean_residual: set[str],
) -> dict[str, Any]:
    correct = _correct_ids(records)
    return {
        "label": label,
        "method": spec.method if spec else None,
        "path": _display_path(spec.path) if spec else None,
        "n": len(records),
        "correct": len(correct),
        "accuracy": float(len(correct) / max(len(records), 1)),
        "exact_ordered_id_parity": _ordered_ids(records) == list(reference_ids),
        "numeric_extraction_coverage": int(
            sum(
                int(harness._has_numeric_extraction(str(row.get("prediction", ""))))
                for row in records
            )
        ),
        "wins_vs_target_count": len(correct - target_correct),
        "wins_vs_target_ids": sorted(correct - target_correct),
        "losses_vs_target_count": len(target_correct - correct),
        "losses_vs_target_ids": sorted(target_correct - correct),
        "teacher_overlap_correct_count": len(correct & teacher_correct),
        "teacher_only_recovered_count": len(correct & teacher_only),
        "teacher_only_recovered_ids": sorted(correct & teacher_only),
        "clean_residual_recovered_count": len(correct & clean_residual),
        "clean_residual_recovered_ids": sorted(correct & clean_residual),
    }


def _oracle_records(
    *,
    label: str,
    baseline: Sequence[dict[str, Any]],
    arms: dict[str, Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, base in enumerate(baseline):
        arm_labels = ["baseline"] + list(arms)
        arm_rows = [base] + [list(records)[index] for records in arms.values()]
        correct_labels = [
            arm_label
            for arm_label, row in zip(arm_labels, arm_rows, strict=True)
            if bool(row.get("correct"))
        ]
        chosen_label = correct_labels[0] if correct_labels else "baseline"
        chosen = arm_rows[arm_labels.index(chosen_label)]
        row = dict(chosen)
        row["method"] = label
        row["oracle_arm_labels"] = arm_labels
        row["oracle_correct_labels"] = correct_labels
        row["oracle_selected_label"] = chosen_label
        row["oracle_selected_source_method"] = chosen.get("method")
        rows.append(row)
    return rows


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    target_spec = _parse_spec(args.target)
    teacher_spec = _parse_spec(args.teacher)
    baseline_spec = _parse_spec(args.baseline)
    target_records = _records_for_method(target_spec.path, target_spec.method)
    reference_ids = _ordered_ids(target_records)
    teacher_records = _subset_reference_order(
        _records_for_method(teacher_spec.path, teacher_spec.method),
        reference_ids,
    )
    baseline_records = _subset_reference_order(
        _records_for_method(baseline_spec.path, baseline_spec.method),
        reference_ids,
    )
    candidate_specs, candidates = _load_spec_rows(args.candidate, reference_ids)
    control_specs, controls = _load_spec_rows(args.control, reference_ids)

    target_correct = _correct_ids(target_records)
    teacher_correct = _correct_ids(teacher_records)
    teacher_only = teacher_correct - target_correct
    clean_residual = _clean_ids(pathlib.Path(args.target_set_json) if args.target_set_json else None)

    live_oracle = _oracle_records(
        label="target_safe_candidate_oracle",
        baseline=baseline_records,
        arms=candidates,
    )
    control_oracle = _oracle_records(
        label="target_safe_control_oracle",
        baseline=baseline_records,
        arms=controls,
    )
    live_correct = _correct_ids(live_oracle)
    control_correct = _correct_ids(control_oracle)
    clean_live = live_correct & clean_residual
    clean_control = control_correct & clean_residual
    clean_source_necessary = clean_live - clean_control

    baseline_summary = _summary_for_rows(
        label=baseline_spec.label,
        spec=baseline_spec,
        records=baseline_records,
        reference_ids=reference_ids,
        target_correct=target_correct,
        teacher_correct=teacher_correct,
        teacher_only=teacher_only,
        clean_residual=clean_residual,
    )
    candidate_summaries = [
        _summary_for_rows(
            label=spec.label,
            spec=spec,
            records=candidates[spec.label],
            reference_ids=reference_ids,
            target_correct=target_correct,
            teacher_correct=teacher_correct,
            teacher_only=teacher_only,
            clean_residual=clean_residual,
        )
        for spec in candidate_specs
    ]
    control_summaries = [
        _summary_for_rows(
            label=spec.label,
            spec=spec,
            records=controls[spec.label],
            reference_ids=reference_ids,
            target_correct=target_correct,
            teacher_correct=teacher_correct,
            teacher_only=teacher_only,
            clean_residual=clean_residual,
        )
        for spec in control_specs
    ]
    live_summary = _summary_for_rows(
        label="target_safe_candidate_oracle",
        spec=None,
        records=live_oracle,
        reference_ids=reference_ids,
        target_correct=target_correct,
        teacher_correct=teacher_correct,
        teacher_only=teacher_only,
        clean_residual=clean_residual,
    )
    control_summary = _summary_for_rows(
        label="target_safe_control_oracle",
        spec=None,
        records=control_oracle,
        reference_ids=reference_ids,
        target_correct=target_correct,
        teacher_correct=teacher_correct,
        teacher_only=teacher_only,
        clean_residual=clean_residual,
    )

    passes = (
        len(clean_source_necessary) >= args.min_clean_source_necessary
        and int(live_summary["correct"]) >= int(baseline_summary["correct"])
        and int(live_summary["losses_vs_target_count"]) <= args.max_losses_vs_target
    )
    if len(candidate_summaries) == 0:
        status = "no_candidates"
    elif passes:
        status = "oracle_has_enough_clean_source_signal"
    else:
        status = "oracle_lacks_clean_source_signal"

    return {
        "date": str(date.today()),
        "status": status,
        "reference_n": len(reference_ids),
        "config": {
            "min_clean_source_necessary": args.min_clean_source_necessary,
            "max_losses_vs_target": args.max_losses_vs_target,
        },
        "reference": {
            "target_correct": len(target_correct),
            "teacher_correct": len(teacher_correct),
            "teacher_only_count": len(teacher_only),
            "teacher_only_ids": sorted(teacher_only),
            "clean_residual_count": len(clean_residual),
            "clean_residual_ids": sorted(clean_residual),
        },
        "baseline": baseline_summary,
        "candidates": candidate_summaries,
        "controls": control_summaries,
        "candidate_oracle": live_summary,
        "control_oracle": control_summary,
        "clean_source_necessary": {
            "count": len(clean_source_necessary),
            "ids": sorted(clean_source_necessary),
            "candidate_clean_ids": sorted(clean_live),
            "control_clean_ids": sorted(clean_control),
        },
        "artifacts": {
            "target": _display_path(target_spec.path),
            "teacher": _display_path(teacher_spec.path),
            "baseline": _display_path(baseline_spec.path),
            "candidates": {
                spec.label: _display_path(spec.path) for spec in candidate_specs
            },
            "controls": {spec.label: _display_path(spec.path) for spec in control_specs},
            "target_set_json": args.target_set_json,
        },
    }


def write_markdown(payload: dict[str, Any], output_md: pathlib.Path) -> None:
    lines = [
        "# SVAMP32 Target-Safe Oracle Replay",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- target: `{payload['reference']['target_correct']}/32`",
        f"- teacher: `{payload['reference']['teacher_correct']}/32`",
        f"- teacher-only IDs: `{payload['reference']['teacher_only_count']}`",
        f"- clean residual IDs: `{payload['reference']['clean_residual_count']}`",
        "",
        "## Gate",
        "",
        f"- minimum clean source-necessary IDs: `{payload['config']['min_clean_source_necessary']}`",
        f"- maximum target losses: `{payload['config']['max_losses_vs_target']}`",
        "",
        "## Rows",
        "",
        "| Row | Correct | C2C-only | Clean residual | Wins vs target | Losses vs target | Numeric coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_rows = [payload["baseline"], *payload["candidates"], *payload["controls"]]
    all_rows.extend([payload["candidate_oracle"], payload["control_oracle"]])
    for row in all_rows:
        lines.append(
            "| {label} | {correct}/32 | {teacher_only} | {clean} | {wins} | {losses} | {numeric}/32 |".format(
                label=row["label"],
                correct=int(row["correct"]),
                teacher_only=int(row["teacher_only_recovered_count"]),
                clean=int(row["clean_residual_recovered_count"]),
                wins=int(row["wins_vs_target_count"]),
                losses=int(row["losses_vs_target_count"]),
                numeric=int(row["numeric_extraction_coverage"]),
            )
        )
    lines.extend(
        [
            "",
            "## Clean Source-Necessary Accounting",
            "",
            f"- candidate oracle clean IDs: `{payload['clean_source_necessary']['candidate_clean_ids']}`",
            f"- control oracle clean IDs: `{payload['clean_source_necessary']['control_clean_ids']}`",
            f"- clean source-necessary IDs: `{payload['clean_source_necessary']['ids']}`",
            f"- clean source-necessary count: `{payload['clean_source_necessary']['count']}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in payload["artifacts"].items():
        if isinstance(value, dict):
            lines.append(f"- {key}:")
            for subkey, subvalue in value.items():
                lines.append(f"  - `{subkey}`: `{subvalue}`")
        else:
            lines.append(f"- {key}: `{value}`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="label=path=...,method=...")
    parser.add_argument("--teacher", required=True, help="label=path=...,method=...")
    parser.add_argument("--baseline", required=True, help="target-safe fallback row")
    parser.add_argument("--candidate", action="append", default=[], help="repeatable label=path=...,method=...")
    parser.add_argument("--control", action="append", default=[], help="repeatable label=path=...,method=...")
    parser.add_argument("--target-set-json")
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--max-losses-vs-target", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = analyze(args)
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
