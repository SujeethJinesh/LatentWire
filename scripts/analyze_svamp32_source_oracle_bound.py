#!/usr/bin/env python3
"""Audit source informativeness and oracle-selection headroom on SVAMP32 rows."""

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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_spec(spec: str) -> RowSpec:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected label=path=...,method=... spec, got {spec!r}"
        )
    label, raw_fields = spec.split("=", 1)
    fields: dict[str, str] = {}
    for item in raw_fields.split(","):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Expected key=value in row spec {spec!r}; got {item!r}"
            )
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    if not label or not fields.get("path") or not fields.get("method"):
        raise argparse.ArgumentTypeError(
            f"Spec needs label, path, and method: {spec!r}"
        )
    return RowSpec(label=label, path=_resolve(fields["path"]), method=fields["method"])


def _records_for_method(spec: RowSpec) -> list[dict[str, Any]]:
    records = _read_jsonl(spec.path)
    raw_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        raw_grouped.setdefault(str(row["method"]), []).append(row)
    if spec.method in raw_grouped:
        return [dict(row) for row in raw_grouped[spec.method]]
    grouped = harness.group_by_method(records)
    if spec.method not in grouped:
        raise KeyError(
            f"Method {spec.method!r} not found in {spec.path}; "
            f"raw_available={sorted(raw_grouped)}, normalized_available={sorted(grouped)}"
        )
    return [dict(row) for row in grouped[spec.method]]


def _ordered_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row["example_id"]) for row in records]


def _subset_reference_order(
    records: Sequence[dict[str, Any]],
    reference_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in by_id:
            duplicates.add(example_id)
        by_id[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Missing reference IDs: {missing}")
    return [by_id[example_id] for example_id in reference_ids]


def _correct_ids(records: Sequence[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in records if bool(row.get("correct"))}


def _numeric_coverage(records: Sequence[dict[str, Any]]) -> int:
    return sum(
        int(harness._has_numeric_extraction(str(row.get("prediction", ""))))
        for row in records
    )


def _load_target_set(path: pathlib.Path | None) -> dict[str, set[str]]:
    if path is None:
        return {"teacher_only": set(), "clean_residual": set()}
    payload = _read_json(path)
    ids = payload.get("ids", {})
    return {
        "teacher_only": {str(value) for value in ids.get("teacher_only", [])},
        "clean_residual": {
            str(value) for value in ids.get("clean_residual_targets", [])
        },
    }


def _summarize_row(
    *,
    spec: RowSpec,
    records: Sequence[dict[str, Any]],
    reference_ids: Sequence[str],
    target_correct: set[str],
    teacher_correct: set[str],
    teacher_only: set[str],
    clean_residual: set[str],
) -> dict[str, Any]:
    correct = _correct_ids(records)
    wins_vs_target = correct - target_correct
    losses_vs_target = target_correct - correct
    teacher_only_recovered = correct & teacher_only
    clean_recovered = correct & clean_residual
    return {
        "label": spec.label,
        "method": spec.method,
        "path": _display_path(spec.path),
        "n": len(records),
        "correct": len(correct),
        "accuracy": float(len(correct) / max(len(records), 1)),
        "exact_ordered_id_parity": _ordered_ids(records) == list(reference_ids),
        "numeric_extraction_coverage": _numeric_coverage(records),
        "teacher_overlap_correct_count": len(correct & teacher_correct),
        "teacher_only_recovered_count": len(teacher_only_recovered),
        "teacher_only_recovered_ids": sorted(teacher_only_recovered),
        "clean_residual_recovered_count": len(clean_recovered),
        "clean_residual_recovered_ids": sorted(clean_recovered),
        "wins_vs_target_count": len(wins_vs_target),
        "wins_vs_target_ids": sorted(wins_vs_target),
        "losses_vs_target_count": len(losses_vs_target),
        "losses_vs_target_ids": sorted(losses_vs_target),
    }


def _compact_cell(records_by_label: dict[str, set[str]], example_id: str) -> list[str]:
    return sorted(label for label, correct in records_by_label.items() if example_id in correct)


def analyze(
    *,
    target_spec: RowSpec,
    teacher_spec: RowSpec,
    source_specs: Sequence[RowSpec],
    baseline_specs: Sequence[RowSpec],
    candidate_specs: Sequence[RowSpec],
    target_set_path: pathlib.Path | None,
    expected_n: int,
    run_date: str,
) -> dict[str, Any]:
    target_original = _records_for_method(target_spec)
    if len(target_original) != expected_n:
        raise ValueError(f"target n={len(target_original)} != expected_n={expected_n}")
    reference_ids = _ordered_ids(target_original)
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("target has duplicate example_id values")

    teacher_records = _subset_reference_order(_records_for_method(teacher_spec), reference_ids)
    target_records = [dict(row) for row in target_original]
    target_correct = _correct_ids(target_records)
    teacher_correct = _correct_ids(teacher_records)
    teacher_only = teacher_correct - target_correct
    target_sets = _load_target_set(target_set_path)
    if target_sets["teacher_only"] and target_sets["teacher_only"] != teacher_only:
        raise ValueError("target_set.ids.teacher_only does not match target/teacher rows")
    clean_residual = target_sets["clean_residual"]
    if clean_residual and not clean_residual.issubset(teacher_only):
        raise ValueError("clean_residual_targets must be a subset of teacher-only IDs")

    loaded_sources = {
        spec.label: _subset_reference_order(_records_for_method(spec), reference_ids)
        for spec in source_specs
    }
    loaded_baselines = {
        spec.label: _subset_reference_order(_records_for_method(spec), reference_ids)
        for spec in baseline_specs
    }
    loaded_candidates = {
        spec.label: _subset_reference_order(_records_for_method(spec), reference_ids)
        for spec in candidate_specs
    }

    source_correct_by_label = {
        label: _correct_ids(records) for label, records in loaded_sources.items()
    }
    candidate_correct_by_label = {
        label: _correct_ids(records) for label, records in loaded_candidates.items()
    }

    source_summaries = [
        _summarize_row(
            spec=spec,
            records=loaded_sources[spec.label],
            reference_ids=reference_ids,
            target_correct=target_correct,
            teacher_correct=teacher_correct,
            teacher_only=teacher_only,
            clean_residual=clean_residual,
        )
        for spec in source_specs
    ]
    baseline_summaries = [
        _summarize_row(
            spec=spec,
            records=loaded_baselines[spec.label],
            reference_ids=reference_ids,
            target_correct=target_correct,
            teacher_correct=teacher_correct,
            teacher_only=teacher_only,
            clean_residual=clean_residual,
        )
        for spec in baseline_specs
    ]
    candidate_summaries = [
        _summarize_row(
            spec=spec,
            records=loaded_candidates[spec.label],
            reference_ids=reference_ids,
            target_correct=target_correct,
            teacher_correct=teacher_correct,
            teacher_only=teacher_only,
            clean_residual=clean_residual,
        )
        for spec in candidate_specs
    ]

    base_correct_by_label = {"target": target_correct}
    base_correct_by_label.update(
        {label: _correct_ids(records) for label, records in loaded_baselines.items()}
    )
    row_correct_by_label = {
        **source_correct_by_label,
        **candidate_correct_by_label,
    }

    oracle_bounds: list[dict[str, Any]] = []
    for base_label, base_correct in base_correct_by_label.items():
        for row_label, row_correct in row_correct_by_label.items():
            union = base_correct | row_correct
            oracle_bounds.append(
                {
                    "base": base_label,
                    "row": row_label,
                    "base_correct": len(base_correct),
                    "row_correct": len(row_correct),
                    "oracle_correct": len(union),
                    "oracle_delta_vs_base": len(union) - len(base_correct),
                    "row_unique_over_base_count": len(row_correct - base_correct),
                    "row_unique_over_base_ids": sorted(row_correct - base_correct),
                    "base_unique_over_row_count": len(base_correct - row_correct),
                    "base_unique_over_row_ids": sorted(base_correct - row_correct),
                    "clean_residual_added_count": len((row_correct - base_correct) & clean_residual),
                    "clean_residual_added_ids": sorted((row_correct - base_correct) & clean_residual),
                }
            )

    candidate_provenance: list[dict[str, Any]] = []
    for summary in candidate_summaries:
        recovered = set(summary["teacher_only_recovered_ids"])
        clean_recovered = set(summary["clean_residual_recovered_ids"])
        candidate_provenance.append(
            {
                "candidate": summary["label"],
                "teacher_only_recovered_count": len(recovered),
                "teacher_only_recovered_ids": sorted(recovered),
                "teacher_only_recovered_source_correct": {
                    example_id: _compact_cell(source_correct_by_label, example_id)
                    for example_id in sorted(recovered)
                },
                "clean_residual_recovered_count": len(clean_recovered),
                "clean_residual_recovered_ids": sorted(clean_recovered),
                "clean_residual_recovered_source_correct": {
                    example_id: _compact_cell(source_correct_by_label, example_id)
                    for example_id in sorted(clean_recovered)
                },
            }
        )

    clean_id_provenance = [
        {
            "example_id": example_id,
            "source_correct": _compact_cell(source_correct_by_label, example_id),
            "candidate_correct": _compact_cell(candidate_correct_by_label, example_id),
        }
        for example_id in sorted(clean_residual)
    ]
    source_clean_union = set().union(*source_correct_by_label.values()) if source_correct_by_label else set()
    candidate_clean_union = (
        set().union(*candidate_correct_by_label.values()) if candidate_correct_by_label else set()
    )
    self_repair_label = "target_self_repair"
    self_repair_correct = base_correct_by_label.get(self_repair_label)
    if self_repair_correct is None and baseline_specs:
        self_repair_label = baseline_specs[0].label
        self_repair_correct = base_correct_by_label.get(self_repair_label)
    best_self_oracle = None
    best_self_source_oracle: dict[str, Any] | None = None
    best_self_candidate_oracle: dict[str, Any] | None = None
    if self_repair_correct is not None:
        best_self_oracle = max(
            (
                len(self_repair_correct | row_correct)
                for row_correct in row_correct_by_label.values()
            ),
            default=len(self_repair_correct),
        )
        if source_correct_by_label:
            label, correct_ids = max(
                source_correct_by_label.items(),
                key=lambda item: (len(self_repair_correct | item[1]), item[0]),
            )
            best_self_source_oracle = {
                "label": label,
                "correct": len(self_repair_correct | correct_ids),
                "delta_vs_self_repair": len(self_repair_correct | correct_ids)
                - len(self_repair_correct),
            }
        if candidate_correct_by_label:
            label, correct_ids = max(
                candidate_correct_by_label.items(),
                key=lambda item: (len(self_repair_correct | item[1]), item[0]),
            )
            best_self_candidate_oracle = {
                "label": label,
                "correct": len(self_repair_correct | correct_ids),
                "delta_vs_self_repair": len(self_repair_correct | correct_ids)
                - len(self_repair_correct),
            }

    return {
        "date": run_date,
        "artifacts": {
            "target": _display_path(target_spec.path),
            "teacher": _display_path(teacher_spec.path),
            "target_set": _display_path(target_set_path) if target_set_path else None,
        },
        "config": {"expected_n": expected_n},
        "reference": {
            "target_correct": len(target_correct),
            "teacher_correct": len(teacher_correct),
            "teacher_only_count": len(teacher_only),
            "teacher_only_ids": sorted(teacher_only),
            "clean_residual_count": len(clean_residual),
            "clean_residual_ids": sorted(clean_residual),
            "source_union_clean_residual_correct_count": len(source_clean_union & clean_residual),
            "source_union_clean_residual_correct_ids": sorted(source_clean_union & clean_residual),
            "candidate_union_clean_residual_correct_count": len(candidate_clean_union & clean_residual),
            "candidate_union_clean_residual_correct_ids": sorted(candidate_clean_union & clean_residual),
            "best_oracle_with_self_repair": best_self_oracle,
            "best_source_oracle_with_self_repair": best_self_source_oracle,
            "best_candidate_oracle_with_self_repair": best_self_candidate_oracle,
            "self_repair_label": self_repair_label if self_repair_correct is not None else None,
        },
        "rows": {
            "target": _summarize_row(
                spec=target_spec,
                records=target_records,
                reference_ids=reference_ids,
                target_correct=target_correct,
                teacher_correct=teacher_correct,
                teacher_only=teacher_only,
                clean_residual=clean_residual,
            ),
            "teacher": _summarize_row(
                spec=teacher_spec,
                records=teacher_records,
                reference_ids=reference_ids,
                target_correct=target_correct,
                teacher_correct=teacher_correct,
                teacher_only=teacher_only,
                clean_residual=clean_residual,
            ),
            "sources": source_summaries,
            "baselines": baseline_summaries,
            "candidates": candidate_summaries,
        },
        "oracle_bounds": oracle_bounds,
        "candidate_provenance": candidate_provenance,
        "clean_id_provenance": clean_id_provenance,
    }


def write_markdown(payload: dict[str, Any], output: pathlib.Path) -> None:
    ref = payload["reference"]
    lines = [
        "# SVAMP32 Source Informativeness And Oracle Bound",
        "",
        f"- date: `{payload['date']}`",
        f"- target: `{ref['target_correct']}/32`",
        f"- C2C teacher: `{ref['teacher_correct']}/32`",
        f"- teacher-only IDs: `{ref['teacher_only_count']}`",
        f"- clean residual IDs: `{ref['clean_residual_count']}`",
        f"- source-union clean residual correct: `{ref['source_union_clean_residual_correct_count']}/{ref['clean_residual_count']}`",
        f"- candidate-union clean residual correct: `{ref['candidate_union_clean_residual_correct_count']}/{ref['clean_residual_count']}`",
        f"- best oracle with `{ref['self_repair_label']}`: `{ref['best_oracle_with_self_repair']}/32`",
        (
            f"- best source-row oracle with `{ref['self_repair_label']}`: "
            f"`{ref['best_source_oracle_with_self_repair']['label']}` "
            f"`{ref['best_source_oracle_with_self_repair']['correct']}/32`"
            if ref.get("best_source_oracle_with_self_repair")
            else "- best source-row oracle with self-repair: `n/a`"
        ),
        (
            f"- best candidate-row oracle with `{ref['self_repair_label']}`: "
            f"`{ref['best_candidate_oracle_with_self_repair']['label']}` "
            f"`{ref['best_candidate_oracle_with_self_repair']['correct']}/32`"
            if ref.get("best_candidate_oracle_with_self_repair")
            else "- best candidate-row oracle with self-repair: `n/a`"
        ),
        "",
        "## Source Rows",
        "",
        "| Label | Correct | Teacher-only correct | Clean residual correct | Wins vs target | Losses vs target |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]["sources"]:
        lines.append(
            "| {label} | {correct}/32 | {teacher_only} | {clean} | {wins} | {losses} |".format(
                label=row["label"],
                correct=row["correct"],
                teacher_only=row["teacher_only_recovered_count"],
                clean=row["clean_residual_recovered_count"],
                wins=row["wins_vs_target_count"],
                losses=row["losses_vs_target_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Label | Correct | Teacher-only correct | Clean residual correct | Wins vs target | Losses vs target |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]["baselines"]:
        lines.append(
            "| {label} | {correct}/32 | {teacher_only} | {clean} | {wins} | {losses} |".format(
                label=row["label"],
                correct=row["correct"],
                teacher_only=row["teacher_only_recovered_count"],
                clean=row["clean_residual_recovered_count"],
                wins=row["wins_vs_target_count"],
                losses=row["losses_vs_target_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Rows",
            "",
            "| Label | Correct | Teacher-only correct | Clean residual correct | Wins vs target | Losses vs target |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]["candidates"]:
        lines.append(
            "| {label} | {correct}/32 | {teacher_only} | {clean} | {wins} | {losses} |".format(
                label=row["label"],
                correct=row["correct"],
                teacher_only=row["teacher_only_recovered_count"],
                clean=row["clean_residual_recovered_count"],
                wins=row["wins_vs_target_count"],
                losses=row["losses_vs_target_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Oracle Bounds",
            "",
            "| Base | Row | Oracle correct | Delta vs base | Clean residual added |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        payload["oracle_bounds"],
        key=lambda item: (
            str(item["base"]),
            -int(item["oracle_correct"]),
            str(item["row"]),
        ),
    ):
        lines.append(
            "| {base} | {row} | {oracle}/32 | {delta:+d} | {clean} |".format(
                base=row["base"],
                row=row["row"],
                oracle=row["oracle_correct"],
                delta=row["oracle_delta_vs_base"],
                clean=row["clean_residual_added_count"],
            )
        )
    lines.extend(["", "## Clean Residual Provenance", ""])
    for row in payload["clean_id_provenance"]:
        sources = ", ".join(f"`{label}`" for label in row["source_correct"]) or "none"
        candidates = ", ".join(f"`{label}`" for label in row["candidate_correct"]) or "none"
        lines.append(
            f"- `{row['example_id']}`: source_correct={sources}; candidate_correct={candidates}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=_parse_spec)
    parser.add_argument("--teacher", required=True, type=_parse_spec)
    parser.add_argument("--source", action="append", type=_parse_spec, default=[])
    parser.add_argument("--baseline", action="append", type=_parse_spec, default=[])
    parser.add_argument("--candidate", action="append", type=_parse_spec, default=[])
    parser.add_argument("--target-set-json")
    parser.add_argument("--expected-n", type=int, default=32)
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = analyze(
        target_spec=args.target,
        teacher_spec=args.teacher,
        source_specs=args.source,
        baseline_specs=args.baseline,
        candidate_specs=args.candidate,
        target_set_path=_resolve(args.target_set_json) if args.target_set_json else None,
        expected_n=args.expected_n,
        run_date=args.date,
    )
    output_json = _resolve(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, _resolve(args.output_md))


if __name__ == "__main__":
    main()
