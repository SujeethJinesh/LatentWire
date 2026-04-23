#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

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
    if method in grouped:
        return [dict(row) for row in grouped[method]]
    raise KeyError(f"Method {method!r} not found in {path}: {sorted(grouped)}")


def _ordered_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["example_id"]) for record in records]


def _subset_reference_order(records: list[dict[str, Any]], reference_ids: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in seen:
            duplicates.add(example_id)
        seen.add(example_id)
        by_id[example_id] = row
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Missing reference IDs: {missing}")
    return [by_id[example_id] for example_id in reference_ids]


def _correct_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in records if bool(row.get("correct"))}


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["example_id"]): row for row in records}


def _compact_cell(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"present": False, "correct": False, "normalized_prediction": None}
    return {
        "present": True,
        "correct": bool(row.get("correct")),
        "normalized_prediction": row.get("normalized_prediction"),
        "selected_candidate_source": row.get("selected_candidate_source"),
        "candidate_source": row.get("candidate_source"),
        "repair_changed_answer": row.get("repair_changed_answer"),
    }


def _row_summary(
    *,
    spec: RowSpec,
    records: list[dict[str, Any]],
    original_records: list[dict[str, Any]],
    reference_ids: list[str],
    target_correct: set[str],
    teacher_correct: set[str],
    teacher_only_ids: set[str],
) -> dict[str, Any]:
    correct = _correct_ids(records)
    wins_vs_target = correct - target_correct
    losses_vs_target = target_correct - correct
    recovered_teacher_only = correct & teacher_only_ids
    original_ids = _ordered_ids(original_records)
    return {
        "label": spec.label,
        "method": spec.method,
        "path": _display_path(spec.path),
        "n": len(records),
        "correct": len(correct),
        "accuracy": float(len(correct) / max(len(records), 1)),
        "artifact_n": len(original_records),
        "exact_ordered_id_parity": _ordered_ids(records) == reference_ids,
        "artifact_prefix_id_parity": original_ids[: len(reference_ids)] == reference_ids,
        "artifact_contains_reference_ids": set(reference_ids).issubset(set(original_ids)),
        "numeric_extraction_coverage": int(
            sum(int(harness._has_numeric_extraction(str(row.get("prediction", "")))) for row in records)
        ),
        "empty_predictions": int(sum(int(not str(row.get("prediction", "")).strip()) for row in records)),
        "wins_vs_target_count": len(wins_vs_target),
        "wins_vs_target_ids": sorted(wins_vs_target),
        "losses_vs_target_count": len(losses_vs_target),
        "losses_vs_target_ids": sorted(losses_vs_target),
        "teacher_overlap_correct_count": len(correct & teacher_correct),
        "teacher_only_recovered_count": len(recovered_teacher_only),
        "teacher_only_recovered_ids": sorted(recovered_teacher_only),
        "non_teacher_wins_vs_target_count": len(wins_vs_target - teacher_only_ids),
        "non_teacher_wins_vs_target_ids": sorted(wins_vs_target - teacher_only_ids),
    }


def _candidate_control_overlap(
    candidate_summary: dict[str, Any],
    control_summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_teacher_ids = set(candidate_summary["teacher_only_recovered_ids"])
    overlap: dict[str, dict[str, Any]] = {}
    for label, control in control_summaries.items():
        control_teacher_ids = set(control["teacher_only_recovered_ids"])
        retained = candidate_teacher_ids & control_teacher_ids
        overlap[label] = {
            "control_teacher_only_recovered_count": len(control_teacher_ids),
            "candidate_teacher_only_retained_count": len(retained),
            "candidate_teacher_only_retained_ids": sorted(retained),
            "candidate_teacher_only_retention_ratio": float(
                len(retained) / max(len(candidate_teacher_ids), 1)
            ),
        }
    return overlap


def _candidate_status(
    candidate_summary: dict[str, Any],
    control_overlap: dict[str, dict[str, Any]],
    *,
    require_clean_controls: bool,
) -> str:
    recovered = int(candidate_summary["teacher_only_recovered_count"])
    if recovered == 0:
        return "no_teacher_only_recovery"
    max_retained = max(
        (int(row["candidate_teacher_only_retained_count"]) for row in control_overlap.values()),
        default=0,
    )
    if max_retained >= recovered:
        return "teacher_recovery_explained_by_controls"
    if max_retained > 0:
        return "partial_control_overlap"
    if require_clean_controls and not control_overlap:
        return "teacher_recovery_without_controls"
    return "teacher_recovery_not_explained_by_controls"


def _teacher_only_provenance(
    *,
    teacher_only_ids: set[str],
    target_records: list[dict[str, Any]],
    teacher_records: list[dict[str, Any]],
    sources: dict[str, list[dict[str, Any]]],
    controls: dict[str, list[dict[str, Any]]],
    candidates: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    target_by_id = _by_id(target_records)
    teacher_by_id = _by_id(teacher_records)
    source_by_id = {label: _by_id(records) for label, records in sources.items()}
    control_by_id = {label: _by_id(records) for label, records in controls.items()}
    candidate_by_id = {label: _by_id(records) for label, records in candidates.items()}
    rows: list[dict[str, Any]] = []
    for example_id in sorted(teacher_only_ids):
        sources_row = {
            label: _compact_cell(records.get(example_id))
            for label, records in source_by_id.items()
        }
        controls_row = {
            label: _compact_cell(records.get(example_id))
            for label, records in control_by_id.items()
        }
        candidates_row = {
            label: _compact_cell(records.get(example_id))
            for label, records in candidate_by_id.items()
        }
        rows.append(
            {
                "example_id": example_id,
                "target": _compact_cell(target_by_id.get(example_id)),
                "teacher": _compact_cell(teacher_by_id.get(example_id)),
                "sources": sources_row,
                "controls": controls_row,
                "candidates": candidates_row,
                "source_correct_labels": sorted(
                    label for label, cell in sources_row.items() if bool(cell["correct"])
                ),
                "control_correct_labels": sorted(
                    label for label, cell in controls_row.items() if bool(cell["correct"])
                ),
                "candidate_correct_labels": sorted(
                    label for label, cell in candidates_row.items() if bool(cell["correct"])
                ),
            }
        )
    return rows


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    teacher = payload["teacher_summary"]
    target = payload["target_summary"]
    lines = [
        "# C2C Teacher Innovation Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- gate: `{payload['gate']['status']}`",
        f"- target: `{target['correct']} / {target['n']}`",
        f"- teacher: `{teacher['correct']} / {teacher['n']}`",
        f"- teacher-only IDs: `{payload['teacher_only_count']}`",
        "",
        "## Row Summary",
        "",
        "| Role | Label | Correct | Wins vs target | Teacher-only recovered | Losses vs target | Numeric coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("sources", "controls", "candidates"):
        for row in payload[role]:
            lines.append(
                f"| {role[:-1]} | `{row['label']}` | {row['correct']}/{row['n']} | "
                f"{row['wins_vs_target_count']} | {row['teacher_only_recovered_count']} | "
                f"{row['losses_vs_target_count']} | {row['numeric_extraction_coverage']}/{row['n']} |"
            )

    lines.extend(["", "## Candidate Control Overlap", ""])
    if payload["candidates"]:
        lines.extend(["| Candidate | Status | Teacher-only recovered | Max retained by control |", "|---|---|---:|---:|"])
        for candidate in payload["candidates"]:
            overlaps = payload["candidate_control_overlap"][candidate["label"]]
            max_retained = max(
                (item["candidate_teacher_only_retained_count"] for item in overlaps.values()),
                default=0,
            )
            lines.append(
                f"| `{candidate['label']}` | `{candidate['teacher_probe_status']}` | "
                f"{candidate['teacher_only_recovered_count']} | {max_retained} |"
            )
    else:
        lines.append("- No candidates provided.")

    lines.extend(["", "## Teacher-Only Provenance", ""])
    lines.extend(
        [
            "| Example ID | Teacher norm | Source-correct | Control-correct | Candidate-correct |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in payload["teacher_only_provenance"]:
        lines.append(
            f"| {row['example_id']} | {row['teacher']['normalized_prediction']} | "
            f"{', '.join(row['source_correct_labels']) or 'none'} | "
            f"{', '.join(row['control_correct_labels']) or 'none'} | "
            f"{', '.join(row['candidate_correct_labels']) or 'none'} |"
        )

    lines.extend(["", "## Gate Notes", ""])
    for note in payload["gate"]["notes"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Artifacts", ""])
    for role in ("target", "teacher"):
        row = payload[f"{role}_summary"]
        lines.append(f"- {role}: `{row['path']}` method `{row['method']}`")
    for role in ("sources", "controls", "candidates"):
        for row in payload[role]:
            lines.append(f"- {role[:-1]}.{row['label']}: `{row['path']}` method `{row['method']}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_spec_rows(specs: list[str], reference_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summaries: list[dict[str, Any]] = []
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    for raw_spec in specs:
        spec = _parse_spec(raw_spec)
        original_records = _records_for_method(spec.path, spec.method)
        records = _subset_reference_order(original_records, reference_ids)
        rows_by_label[spec.label] = records
        summaries.append(
            {
                "spec": spec,
                "records": records,
                "original_records": original_records,
            }
        )
    return summaries, rows_by_label


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_spec = _parse_spec(args.target)
    teacher_spec = _parse_spec(args.teacher)
    target_original = _records_for_method(target_spec.path, target_spec.method)
    reference_ids = _ordered_ids(target_original)
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Target row has duplicate example_id values")
    teacher_original = _records_for_method(teacher_spec.path, teacher_spec.method)
    target_records = _subset_reference_order(target_original, reference_ids)
    teacher_records = _subset_reference_order(teacher_original, reference_ids)

    target_correct = _correct_ids(target_records)
    teacher_correct = _correct_ids(teacher_records)
    teacher_only_ids = teacher_correct - target_correct

    source_specs, source_records = _load_spec_rows(args.source, reference_ids)
    control_specs, control_records = _load_spec_rows(args.control, reference_ids)
    candidate_specs, candidate_records = _load_spec_rows(args.candidate, reference_ids)

    def summarize_loaded(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            _row_summary(
                spec=item["spec"],
                records=item["records"],
                original_records=item["original_records"],
                reference_ids=reference_ids,
                target_correct=target_correct,
                teacher_correct=teacher_correct,
                teacher_only_ids=teacher_only_ids,
            )
            for item in items
        ]

    source_summaries = summarize_loaded(source_specs)
    control_summaries_list = summarize_loaded(control_specs)
    candidate_summaries = summarize_loaded(candidate_specs)
    control_summaries = {row["label"]: row for row in control_summaries_list}
    candidate_control_overlap: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in candidate_summaries:
        overlap = _candidate_control_overlap(candidate, control_summaries)
        candidate_control_overlap[candidate["label"]] = overlap
        candidate["teacher_probe_status"] = _candidate_status(
            candidate,
            overlap,
            require_clean_controls=bool(args.require_controls),
        )

    target_summary = _row_summary(
        spec=target_spec,
        records=target_records,
        original_records=target_original,
        reference_ids=reference_ids,
        target_correct=target_correct,
        teacher_correct=teacher_correct,
        teacher_only_ids=teacher_only_ids,
    )
    teacher_summary = _row_summary(
        spec=teacher_spec,
        records=teacher_records,
        original_records=teacher_original,
        reference_ids=reference_ids,
        target_correct=target_correct,
        teacher_correct=teacher_correct,
        teacher_only_ids=teacher_only_ids,
    )

    notes: list[str] = []
    statuses = {row["label"]: row["teacher_probe_status"] for row in candidate_summaries}
    if not candidate_summaries:
        gate_status = "no_candidates"
        notes.append("No candidate rows were provided.")
    elif any(status == "teacher_recovery_not_explained_by_controls" for status in statuses.values()):
        gate_status = "candidate_recovers_teacher_innovations_without_control_explanation"
        notes.append("At least one candidate recovers teacher-only IDs not recovered by provided controls.")
    elif any(status == "partial_control_overlap" for status in statuses.values()):
        gate_status = "candidate_teacher_recovery_partially_control_explained"
        notes.append("At least one candidate recovers teacher-only IDs, but controls recover some of them.")
    elif any(status == "teacher_recovery_without_controls" for status in statuses.values()):
        gate_status = "candidate_teacher_recovery_requires_controls"
        notes.append("At least one candidate recovers teacher-only IDs, but no controls were provided.")
    elif any(status == "teacher_recovery_explained_by_controls" for status in statuses.values()):
        gate_status = "candidate_teacher_recovery_explained_by_controls"
        notes.append("Teacher-only recovery is fully reproduced by at least one provided control.")
    else:
        gate_status = "no_existing_candidate_recovers_teacher_innovations"
        notes.append("No current candidate recovers teacher-only IDs.")
    if not control_summaries_list:
        notes.append("No controls were provided, so source-specificity cannot be claimed.")
    if teacher_summary["teacher_only_recovered_count"] < int(args.min_teacher_only):
        notes.append(
            f"Teacher surface has fewer than the requested {args.min_teacher_only} teacher-only IDs."
        )

    payload = {
        "date": str(date.today()),
        "reference_n": len(reference_ids),
        "min_teacher_only": int(args.min_teacher_only),
        "teacher_only_count": len(teacher_only_ids),
        "teacher_only_ids": sorted(teacher_only_ids),
        "target_summary": target_summary,
        "teacher_summary": teacher_summary,
        "sources": source_summaries,
        "controls": control_summaries_list,
        "candidates": candidate_summaries,
        "candidate_control_overlap": candidate_control_overlap,
        "teacher_only_provenance": _teacher_only_provenance(
            teacher_only_ids=teacher_only_ids,
            target_records=target_records,
            teacher_records=teacher_records,
            sources=source_records,
            controls=control_records,
            candidates=candidate_records,
        ),
        "gate": {
            "status": gate_status,
            "candidate_statuses": statuses,
            "notes": notes,
        },
    }

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe which rows recover C2C-only teacher innovations on a frozen ID slice."
    )
    parser.add_argument("--target", required=True, help="label=path=...,method=... target row")
    parser.add_argument("--teacher", required=True, help="label=path=...,method=... teacher row")
    parser.add_argument("--source", action="append", default=[], help="repeatable label=path=...,method=...")
    parser.add_argument("--control", action="append", default=[], help="repeatable label=path=...,method=...")
    parser.add_argument("--candidate", action="append", default=[], help="repeatable label=path=...,method=...")
    parser.add_argument("--min-teacher-only", type=int, default=5)
    parser.add_argument("--require-controls", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(_parse_args(argv))


if __name__ == "__main__":
    main()
