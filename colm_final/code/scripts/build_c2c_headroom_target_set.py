#!/usr/bin/env python3
"""Build a clean C2C-headroom target set from a teacher-innovation probe."""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class HeadroomConfig:
    source_labels: tuple[str, ...] = ("source", "t2t")
    control_labels: tuple[str, ...] = ()
    candidate_labels: tuple[str, ...] = ()
    min_teacher_only: int = 5
    min_clean_teacher_only: int = 2


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in rows}


def _id_set(row: dict[str, Any] | None, field: str = "teacher_only_recovered_ids") -> set[str]:
    if not row:
        return set()
    return {str(item) for item in row.get(field, [])}


def _labels_correct_for(provenance: list[dict[str, Any]], group: str, example_id: str) -> list[str]:
    for row in provenance:
        if str(row["example_id"]) == example_id:
            return [str(item) for item in row.get(f"{group}_correct_labels", [])]
    return []


def build_target_set(
    probe_payload: dict[str, Any],
    *,
    config: HeadroomConfig,
) -> dict[str, Any]:
    teacher_only_ids = {str(item) for item in probe_payload.get("teacher_only_ids", [])}
    sources = _rows_by_label(list(probe_payload.get("sources", [])))
    controls = _rows_by_label(list(probe_payload.get("controls", [])))
    candidates = _rows_by_label(list(probe_payload.get("candidates", [])))
    provenance = list(probe_payload.get("teacher_only_provenance", []))

    source_explained_ids = set().union(
        *(_id_set(sources.get(label)) for label in config.source_labels),
    )
    control_explained_ids = set().union(
        *(_id_set(controls.get(label)) for label in config.control_labels),
    )
    candidate_explained_ids = set().union(
        *(_id_set(candidates.get(label)) for label in config.candidate_labels),
    )
    explained_ids = source_explained_ids | control_explained_ids
    clean_teacher_only_ids = teacher_only_ids - explained_ids

    rows: list[dict[str, Any]] = []
    for example_id in sorted(teacher_only_ids):
        labels: list[str] = []
        if example_id in source_explained_ids:
            labels.append("source_or_text")
        if example_id in control_explained_ids:
            labels.append("control_explained")
        if example_id in candidate_explained_ids:
            labels.append("candidate_recovered")
        if example_id in clean_teacher_only_ids:
            labels.append("clean_c2c_headroom_target")
        rows.append(
            {
                "example_id": example_id,
                "labels": labels,
                "source_correct_labels": _labels_correct_for(provenance, "source", example_id),
                "control_correct_labels": _labels_correct_for(provenance, "control", example_id),
                "candidate_correct_labels": _labels_correct_for(provenance, "candidate", example_id),
            }
        )

    teacher_only_count = len(teacher_only_ids)
    clean_count = len(clean_teacher_only_ids)
    status = (
        "clean_headroom_available"
        if teacher_only_count >= config.min_teacher_only
        and clean_count >= config.min_clean_teacher_only
        else "insufficient_clean_headroom"
    )
    return {
        "date": str(date.today()),
        "status": status,
        "config": {
            "source_labels": list(config.source_labels),
            "control_labels": list(config.control_labels),
            "candidate_labels": list(config.candidate_labels),
            "min_teacher_only": int(config.min_teacher_only),
            "min_clean_teacher_only": int(config.min_clean_teacher_only),
        },
        "summary": {
            "reference_n": int(probe_payload.get("reference_n", 0)),
            "target_correct": int(probe_payload.get("target_summary", {}).get("correct", 0)),
            "teacher_correct": int(probe_payload.get("teacher_summary", {}).get("correct", 0)),
            "teacher_only_count": teacher_only_count,
            "source_explained_teacher_only_count": len(source_explained_ids),
            "control_explained_teacher_only_count": len(control_explained_ids),
            "candidate_explained_teacher_only_count": len(candidate_explained_ids),
            "clean_teacher_only_count": clean_count,
            "target_only_vs_teacher_count": int(
                probe_payload.get("teacher_summary", {}).get("losses_vs_target_count", 0)
            ),
        },
        "ids": {
            "teacher_only": sorted(teacher_only_ids),
            "source_explained": sorted(source_explained_ids),
            "control_explained": sorted(control_explained_ids),
            "candidate_explained": sorted(candidate_explained_ids),
            "clean_teacher_only": sorted(clean_teacher_only_ids),
        },
        "teacher_only_rows": rows,
    }


def write_markdown(payload: dict[str, Any], output_md: pathlib.Path) -> None:
    summary = payload["summary"]
    lines = [
        "# C2C Headroom Target Set",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- target: `{summary['target_correct']}/{summary['reference_n']}`",
        f"- C2C teacher: `{summary['teacher_correct']}/{summary['reference_n']}`",
        f"- C2C-only IDs: `{summary['teacher_only_count']}`",
        f"- source/text explained C2C-only IDs: `{summary['source_explained_teacher_only_count']}`",
        f"- clean C2C-only targets: `{summary['clean_teacher_only_count']}`",
        f"- target-only vs C2C IDs to preserve: `{summary['target_only_vs_teacher_count']}`",
        "",
        "## ID Sets",
        "",
        f"- source/text explained: {', '.join(f'`{item}`' for item in payload['ids']['source_explained']) or 'none'}",
        f"- clean C2C headroom targets: {', '.join(f'`{item}`' for item in payload['ids']['clean_teacher_only']) or 'none'}",
        "",
        "## Teacher-Only Provenance",
        "",
        "| Example ID | Labels | Source correct | Control correct | Candidate correct |",
        "|---|---|---|---|---|",
    ]
    for row in payload["teacher_only_rows"]:
        lines.append(
            "| {example_id} | {labels} | {sources} | {controls} | {candidates} |".format(
                example_id=row["example_id"],
                labels=", ".join(f"`{item}`" for item in row["labels"]) or "none",
                sources=", ".join(f"`{item}`" for item in row["source_correct_labels"]) or "none",
                controls=", ".join(f"`{item}`" for item in row["control_correct_labels"]) or "none",
                candidates=", ".join(f"`{item}`" for item in row["candidate_correct_labels"]) or "none",
            )
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--source-label", action="append", default=None)
    parser.add_argument("--control-label", action="append", default=None)
    parser.add_argument("--candidate-label", action="append", default=None)
    parser.add_argument("--min-teacher-only", type=int, default=HeadroomConfig.min_teacher_only)
    parser.add_argument(
        "--min-clean-teacher-only",
        type=int,
        default=HeadroomConfig.min_clean_teacher_only,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = HeadroomConfig(
        source_labels=tuple(args.source_label or HeadroomConfig.source_labels),
        control_labels=tuple(args.control_label or HeadroomConfig.control_labels),
        candidate_labels=tuple(args.candidate_label or HeadroomConfig.candidate_labels),
        min_teacher_only=int(args.min_teacher_only),
        min_clean_teacher_only=int(args.min_clean_teacher_only),
    )
    payload = build_target_set(_load_json(pathlib.Path(args.probe_json)), config=config)
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
