#!/usr/bin/env python3
"""Apply the current SVAMP32 paper gate to a C2C teacher-probe artifact."""

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


@dataclass(frozen=True)
class GateConfig:
    target_self_label: str = "target_self_repair"
    source_control_labels: tuple[str, ...] = ("zero_source", "shuffled_source")
    min_correct: int = 16
    min_delta_vs_target_self: int = 1
    min_teacher_only: int = 5
    min_unique_vs_target_self: int = 2
    max_losses_vs_target: int = 1
    max_source_control_retained: int = 1


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows_by_label(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(row["label"])] = row
    return out


def _as_id_set(row: dict[str, Any], field: str) -> set[str]:
    return {str(value) for value in row.get(field, [])}


def _control_overlap(
    probe_payload: dict[str, Any],
    candidate_label: str,
    control_label: str,
) -> dict[str, Any]:
    return (
        probe_payload.get("candidate_control_overlap", {})
        .get(candidate_label, {})
        .get(control_label, {})
    )


def evaluate_candidate(
    probe_payload: dict[str, Any],
    candidate: dict[str, Any],
    *,
    controls: dict[str, dict[str, Any]],
    config: GateConfig,
) -> dict[str, Any]:
    label = str(candidate["label"])
    target_self = controls.get(config.target_self_label)
    target_self_present = target_self is not None
    target_self_correct = int(target_self.get("correct", 0)) if target_self else 0
    target_self_teacher_ids = (
        _as_id_set(target_self, "teacher_only_recovered_ids") if target_self else set()
    )
    candidate_teacher_ids = _as_id_set(candidate, "teacher_only_recovered_ids")
    unique_vs_target_self = candidate_teacher_ids - target_self_teacher_ids

    source_control_details: dict[str, dict[str, Any]] = {}
    max_source_retained = 0
    retained_by_source_controls: set[str] = set()
    missing_source_controls: list[str] = []
    for control_label in config.source_control_labels:
        overlap = _control_overlap(probe_payload, label, control_label)
        control_present = control_label in controls
        overlap_present = bool(overlap)
        if not control_present or not overlap_present:
            missing_source_controls.append(control_label)
        retained = {str(value) for value in overlap.get("candidate_teacher_only_retained_ids", [])}
        retained_count = int(overlap.get("candidate_teacher_only_retained_count", len(retained)))
        max_source_retained = max(max_source_retained, retained_count)
        retained_by_source_controls.update(retained)
        source_control_details[control_label] = {
            "candidate_teacher_only_retained_count": retained_count,
            "candidate_teacher_only_retained_ids": sorted(retained),
            "control_present": control_present,
            "overlap_present": overlap_present,
            "present": control_present and overlap_present,
        }

    correct = int(candidate.get("correct", 0))
    losses_vs_target = int(candidate.get("losses_vs_target_count", 0))
    criteria = {
        "target_self_repair_present": target_self_present,
        "source_controls_present": not missing_source_controls,
        "min_correct": correct >= config.min_correct,
        "beats_target_self_repair": (correct - target_self_correct) >= config.min_delta_vs_target_self,
        "min_teacher_only": len(candidate_teacher_ids) >= config.min_teacher_only,
        "min_unique_vs_target_self_repair": (
            len(unique_vs_target_self) >= config.min_unique_vs_target_self
        ),
        "max_losses_vs_target": losses_vs_target <= config.max_losses_vs_target,
        "max_source_control_retained": (
            max_source_retained <= config.max_source_control_retained
        ),
    }
    failing = [name for name, passed in criteria.items() if not passed]
    return {
        "candidate_label": label,
        "status": "passes_paper_gate" if not failing else "fails_paper_gate",
        "correct": correct,
        "target_self_repair_correct": target_self_correct,
        "delta_vs_target_self_repair": correct - target_self_correct,
        "teacher_only_recovered_count": len(candidate_teacher_ids),
        "teacher_only_recovered_ids": sorted(candidate_teacher_ids),
        "unique_vs_target_self_repair_count": len(unique_vs_target_self),
        "unique_vs_target_self_repair_ids": sorted(unique_vs_target_self),
        "retained_by_source_controls_count": len(retained_by_source_controls),
        "retained_by_source_controls_ids": sorted(retained_by_source_controls),
        "max_source_control_retained_count": max_source_retained,
        "losses_vs_target_count": losses_vs_target,
        "criteria": criteria,
        "failing_criteria": failing,
        "target_self_repair_present": target_self_present,
        "missing_source_control_labels": missing_source_controls,
        "source_control_details": source_control_details,
    }


def evaluate_gate(probe_payload: dict[str, Any], *, config: GateConfig) -> dict[str, Any]:
    controls = _rows_by_label(probe_payload.get("controls", []))
    candidate_rows = list(probe_payload.get("candidates", []))
    candidates = [
        evaluate_candidate(probe_payload, row, controls=controls, config=config)
        for row in candidate_rows
    ]
    passing = [row["candidate_label"] for row in candidates if row["status"] == "passes_paper_gate"]
    target_self = controls.get(config.target_self_label, {})
    return {
        "date": str(date.today()),
        "status": "candidate_passes_target_self_repair_gate" if passing else "no_candidate_passes_target_self_repair_gate",
        "passing_candidates": passing,
        "config": {
            "target_self_label": config.target_self_label,
            "source_control_labels": list(config.source_control_labels),
            "min_correct": config.min_correct,
            "min_delta_vs_target_self": config.min_delta_vs_target_self,
            "min_teacher_only": config.min_teacher_only,
            "min_unique_vs_target_self": config.min_unique_vs_target_self,
            "max_losses_vs_target": config.max_losses_vs_target,
            "max_source_control_retained": config.max_source_control_retained,
        },
        "reference": {
            "target_correct": int(probe_payload.get("target_summary", {}).get("correct", 0)),
            "teacher_correct": int(probe_payload.get("teacher_summary", {}).get("correct", 0)),
            "teacher_only_count": int(probe_payload.get("teacher_only_count", 0)),
            "target_self_repair_correct": int(target_self.get("correct", 0)),
            "target_self_repair_teacher_only": int(
                target_self.get("teacher_only_recovered_count", 0)
            ),
        },
        "candidates": candidates,
    }


def _fmt_bool(value: bool) -> str:
    return "pass" if value else "fail"


def write_markdown(payload: dict[str, Any], output_md: pathlib.Path) -> None:
    lines = [
        "# SVAMP32 Target-Self-Repair Paper Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- target: `{payload['reference']['target_correct']}/32`",
        f"- C2C teacher: `{payload['reference']['teacher_correct']}/32`",
        f"- target_self_repair: `{payload['reference']['target_self_repair_correct']}/32`",
        f"- target_self_repair C2C-only recovered: `{payload['reference']['target_self_repair_teacher_only']}/{payload['reference']['teacher_only_count']}`",
        "",
        "## Gate",
        "",
        f"- minimum correct: `{payload['config']['min_correct']}/32`",
        f"- minimum delta vs target_self_repair: `+{payload['config']['min_delta_vs_target_self']}`",
        f"- minimum C2C-only recovered: `{payload['config']['min_teacher_only']}/{payload['reference']['teacher_only_count']}`",
        f"- minimum C2C-only unique vs target_self_repair: `{payload['config']['min_unique_vs_target_self']}`",
        f"- maximum target losses: `{payload['config']['max_losses_vs_target']}`",
        f"- maximum retained by any source control: `{payload['config']['max_source_control_retained']}`",
        "",
        "## Candidate Decisions",
        "",
        "| Candidate | Status | Correct | Delta vs self-repair | C2C-only | Unique vs self-repair | Max source-control retained | Target losses | Failing criteria |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            "| {candidate} | `{status}` | {correct}/32 | {delta:+d} | {teacher_only} | {unique} | {retained} | {losses} | {failing} |".format(
                candidate=row["candidate_label"],
                status=row["status"],
                correct=int(row["correct"]),
                delta=int(row["delta_vs_target_self_repair"]),
                teacher_only=int(row["teacher_only_recovered_count"]),
                unique=int(row["unique_vs_target_self_repair_count"]),
                retained=int(row["max_source_control_retained_count"]),
                losses=int(row["losses_vs_target_count"]),
                failing=", ".join(row["failing_criteria"]) or "none",
            )
        )
    lines.extend(["", "## Criteria Detail", ""])
    for row in payload["candidates"]:
        lines.append(f"### {row['candidate_label']}")
        for name, passed in row["criteria"].items():
            lines.append(f"- `{name}`: `{_fmt_bool(bool(passed))}`")
        lines.append(
            "- C2C-only unique vs target_self_repair: "
            + ", ".join(f"`{item}`" for item in row["unique_vs_target_self_repair_ids"])
            if row["unique_vs_target_self_repair_ids"]
            else "- C2C-only unique vs target_self_repair: none"
        )
        lines.append(
            "- C2C-only retained by source controls: "
            + ", ".join(f"`{item}`" for item in row["retained_by_source_controls_ids"])
            if row["retained_by_source_controls_ids"]
            else "- C2C-only retained by source controls: none"
        )
        lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--target-self-label", default=GateConfig.target_self_label)
    parser.add_argument("--source-control-label", action="append", default=None)
    parser.add_argument("--min-correct", type=int, default=GateConfig.min_correct)
    parser.add_argument(
        "--min-delta-vs-target-self",
        type=int,
        default=GateConfig.min_delta_vs_target_self,
    )
    parser.add_argument("--min-teacher-only", type=int, default=GateConfig.min_teacher_only)
    parser.add_argument(
        "--min-unique-vs-target-self",
        type=int,
        default=GateConfig.min_unique_vs_target_self,
    )
    parser.add_argument(
        "--max-losses-vs-target",
        type=int,
        default=GateConfig.max_losses_vs_target,
    )
    parser.add_argument(
        "--max-source-control-retained",
        type=int,
        default=GateConfig.max_source_control_retained,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    source_controls = tuple(args.source_control_label or GateConfig.source_control_labels)
    config = GateConfig(
        target_self_label=args.target_self_label,
        source_control_labels=source_controls,
        min_correct=args.min_correct,
        min_delta_vs_target_self=args.min_delta_vs_target_self,
        min_teacher_only=args.min_teacher_only,
        min_unique_vs_target_self=args.min_unique_vs_target_self,
        max_losses_vs_target=args.max_losses_vs_target,
        max_source_control_retained=args.max_source_control_retained,
    )
    payload = evaluate_gate(_load_json(pathlib.Path(args.probe_json)), config=config)
    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
