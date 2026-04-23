#!/usr/bin/env python3
"""Build the SVAMP32 residual innovation target set for the next method branch."""

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


@dataclass(frozen=True)
class TargetSetConfig:
    target_self_label: str = "target_self_repair"
    source_labels: tuple[str, ...] = ("source", "t2t")
    source_control_labels: tuple[str, ...] = ("zero_source", "shuffled_source")
    min_correct: int = 16
    min_teacher_only: int = 5
    min_unique_vs_target_self: int = 2


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
    config: TargetSetConfig,
) -> dict[str, Any]:
    controls = _rows_by_label(list(probe_payload.get("controls", [])))
    sources = _rows_by_label(list(probe_payload.get("sources", [])))
    candidates = _rows_by_label(list(probe_payload.get("candidates", [])))
    provenance = list(probe_payload.get("teacher_only_provenance", []))

    teacher_only_ids = {str(item) for item in probe_payload.get("teacher_only_ids", [])}
    target_self_row = controls.get(config.target_self_label)
    target_self_ids = _id_set(target_self_row)
    source_explained_ids = set().union(
        *[_id_set(sources.get(label)) for label in config.source_labels],
    )
    source_control_explained_ids = set().union(
        *[_id_set(controls.get(label)) for label in config.source_control_labels],
    )
    candidate_explained_ids = set().union(
        *[_id_set(row) for row in candidates.values()],
    )
    clean_residual_ids = (
        teacher_only_ids
        - target_self_ids
        - source_explained_ids
        - source_control_explained_ids
    )
    target_self_correct = int(target_self_row.get("correct", 0)) if target_self_row else 0
    target_self_teacher_only = len(target_self_ids)
    oracle_target_self_plus_teacher = target_self_correct + len(teacher_only_ids - target_self_ids)
    required_clean_residual = max(
        0,
        config.min_correct - target_self_correct,
        config.min_teacher_only - target_self_teacher_only,
        config.min_unique_vs_target_self,
    )
    can_clear_gate_if_preserving_self = len(clean_residual_ids) >= required_clean_residual

    rows: list[dict[str, Any]] = []
    for example_id in sorted(teacher_only_ids):
        labels: list[str] = []
        if example_id in target_self_ids:
            labels.append("target_self_repair")
        if example_id in source_explained_ids:
            labels.append("source_or_text")
        if example_id in source_control_explained_ids:
            labels.append("source_control")
        if example_id in candidate_explained_ids:
            labels.append("current_candidate")
        if example_id in clean_residual_ids:
            labels.append("clean_c2c_residual_target")
        rows.append(
            {
                "example_id": example_id,
                "labels": labels,
                "source_correct_labels": _labels_correct_for(provenance, "source", example_id),
                "control_correct_labels": _labels_correct_for(provenance, "control", example_id),
                "candidate_correct_labels": _labels_correct_for(provenance, "candidate", example_id),
            }
        )

    return {
        "date": str(date.today()),
        "status": (
            "residual_headroom_available"
            if can_clear_gate_if_preserving_self
            else "insufficient_clean_residual_headroom"
        ),
        "config": {
            "target_self_label": config.target_self_label,
            "source_labels": list(config.source_labels),
            "source_control_labels": list(config.source_control_labels),
            "min_correct": config.min_correct,
            "min_teacher_only": config.min_teacher_only,
            "min_unique_vs_target_self": config.min_unique_vs_target_self,
        },
        "summary": {
            "target_correct": int(probe_payload.get("target_summary", {}).get("correct", 0)),
            "teacher_correct": int(probe_payload.get("teacher_summary", {}).get("correct", 0)),
            "teacher_only_count": len(teacher_only_ids),
            "target_self_repair_correct": target_self_correct,
            "target_self_repair_teacher_only_count": target_self_teacher_only,
            "source_explained_teacher_only_count": len(source_explained_ids),
            "source_control_explained_teacher_only_count": len(source_control_explained_ids),
            "current_candidate_teacher_only_count": len(candidate_explained_ids),
            "clean_residual_teacher_only_count": len(clean_residual_ids),
            "oracle_target_self_plus_teacher_correct": oracle_target_self_plus_teacher,
            "required_clean_residual_to_clear_gate_if_preserving_self": required_clean_residual,
            "can_clear_gate_if_preserving_target_self": can_clear_gate_if_preserving_self,
        },
        "ids": {
            "teacher_only": sorted(teacher_only_ids),
            "target_self_repair": sorted(target_self_ids),
            "source_explained": sorted(source_explained_ids),
            "source_control_explained": sorted(source_control_explained_ids),
            "current_candidate": sorted(candidate_explained_ids),
            "clean_residual_targets": sorted(clean_residual_ids),
        },
        "teacher_only_rows": rows,
    }


def write_markdown(payload: dict[str, Any], output_md: pathlib.Path) -> None:
    summary = payload["summary"]
    lines = [
        "# SVAMP32 Innovation Target Set",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- target: `{summary['target_correct']}/32`",
        f"- C2C teacher: `{summary['teacher_correct']}/32`",
        f"- target_self_repair: `{summary['target_self_repair_correct']}/32`",
        f"- C2C-only IDs: `{summary['teacher_only_count']}`",
        f"- clean residual C2C-only targets: `{summary['clean_residual_teacher_only_count']}`",
        f"- oracle target_self_repair plus C2C teacher: `{summary['oracle_target_self_plus_teacher_correct']}/32`",
        f"- required clean residual wins if preserving target_self_repair: `{summary['required_clean_residual_to_clear_gate_if_preserving_self']}`",
        "",
        "## Interpretation",
        "",
    ]
    if summary["can_clear_gate_if_preserving_target_self"]:
        lines.append(
            "A target-self-preserving connector can clear the current paper gate "
            "by adding the required number of clean residual C2C-only wins."
        )
    else:
        lines.append(
            "The clean residual target set is too small to clear the current paper gate "
            "unless the gate or benchmark surface changes."
        )
    lines.extend(
        [
            "",
            "## ID Sets",
            "",
            f"- target_self_repair C2C-only: {', '.join(f'`{item}`' for item in payload['ids']['target_self_repair']) or 'none'}",
            f"- source/source-control explained C2C-only: {', '.join(f'`{item}`' for item in sorted(set(payload['ids']['source_explained']) | set(payload['ids']['source_control_explained']))) or 'none'}",
            f"- clean residual targets: {', '.join(f'`{item}`' for item in payload['ids']['clean_residual_targets']) or 'none'}",
            "",
            "## Teacher-Only Provenance",
            "",
            "| Example ID | Labels | Source correct | Control correct | Candidate correct |",
            "|---|---|---|---|---|",
        ]
    )
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
    parser.add_argument("--target-self-label", default=TargetSetConfig.target_self_label)
    parser.add_argument("--source-label", action="append", default=None)
    parser.add_argument("--source-control-label", action="append", default=None)
    parser.add_argument("--min-correct", type=int, default=TargetSetConfig.min_correct)
    parser.add_argument("--min-teacher-only", type=int, default=TargetSetConfig.min_teacher_only)
    parser.add_argument(
        "--min-unique-vs-target-self",
        type=int,
        default=TargetSetConfig.min_unique_vs_target_self,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = TargetSetConfig(
        target_self_label=args.target_self_label,
        source_labels=tuple(args.source_label or TargetSetConfig.source_labels),
        source_control_labels=tuple(
            args.source_control_label or TargetSetConfig.source_control_labels
        ),
        min_correct=args.min_correct,
        min_teacher_only=args.min_teacher_only,
        min_unique_vs_target_self=args.min_unique_vs_target_self,
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
