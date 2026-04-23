#!/usr/bin/env python3
"""Analyze the target-self-preserving source-innovation sidecar bound.

This is an oracle-bound analysis, not a deployable router.  It asks whether a
candidate has enough clean, source-necessary C2C-only residual wins that a
future target-self-preserving sidecar is worth implementing.
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


@dataclass(frozen=True)
class SidecarGateConfig:
    target_self_label: str = "target_self_repair"
    source_control_labels: tuple[str, ...] = ("translated_kv_zero",)
    candidate_label: str | None = None
    min_correct: int = 16
    min_clean_source_necessary: int = 2
    min_delta_vs_target_self: int = 1
    min_numeric_coverage: int = 31
    expected_n: int = 32


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _rows_by_label(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in rows}


def _as_id_set(row: dict[str, Any] | None, field: str) -> set[str]:
    if not row:
        return set()
    return {str(value) for value in row.get(field, [])}


def _clean_residual_ids(target_set_payload: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in target_set_payload.get("ids", {}).get("clean_residual_targets", [])
    }


def _row_provenance_issues(
    row: dict[str, Any],
    *,
    role: str,
    reference_n: int,
    min_numeric_coverage: int,
) -> list[str]:
    label = str(row.get("label", role))
    prefix = f"{role}.{label}"
    issues: list[str] = []
    if int(row.get("n", -1)) != reference_n:
        issues.append(f"{prefix}: n={row.get('n')} != reference_n={reference_n}")
    if int(row.get("artifact_n", -1)) != reference_n:
        issues.append(
            f"{prefix}: artifact_n={row.get('artifact_n')} != reference_n={reference_n}"
        )
    if not bool(row.get("exact_ordered_id_parity")):
        issues.append(f"{prefix}: exact_ordered_id_parity=false")
    if int(row.get("numeric_extraction_coverage", 0)) < min_numeric_coverage:
        issues.append(
            f"{prefix}: numeric_extraction_coverage={row.get('numeric_extraction_coverage')} "
            f"< {min_numeric_coverage}"
        )
    return issues


def validate_provenance(
    probe_payload: dict[str, Any],
    target_set_payload: dict[str, Any],
    *,
    config: SidecarGateConfig,
) -> None:
    reference_n = int(probe_payload.get("reference_n", 0))
    issues: list[str] = []
    if reference_n != config.expected_n:
        issues.append(f"probe.reference_n={reference_n} != expected_n={config.expected_n}")
    for role, rows in (
        ("target", [probe_payload.get("target_summary", {})]),
        ("teacher", [probe_payload.get("teacher_summary", {})]),
        ("source", list(probe_payload.get("sources", []))),
        ("control", list(probe_payload.get("controls", []))),
        ("candidate", list(probe_payload.get("candidates", []))),
    ):
        for row in rows:
            issues.extend(
                _row_provenance_issues(
                    row,
                    role=role,
                    reference_n=reference_n,
                    min_numeric_coverage=config.min_numeric_coverage,
                )
            )

    probe_teacher_only = {str(value) for value in probe_payload.get("teacher_only_ids", [])}
    target_teacher_only = {
        str(value) for value in target_set_payload.get("ids", {}).get("teacher_only", [])
    }
    clean_ids = _clean_residual_ids(target_set_payload)
    if probe_teacher_only != target_teacher_only:
        issues.append("target_set.ids.teacher_only does not match probe teacher_only_ids")
    if not clean_ids:
        issues.append("target_set has no clean_residual_targets")
    if not clean_ids.issubset(probe_teacher_only):
        issues.append("target_set clean_residual_targets is not a subset of probe teacher_only_ids")
    if issues:
        raise ValueError("SVAMP32 sidecar-bound provenance validation failed: " + "; ".join(issues))


def _candidate_control_retained_ids(
    probe_payload: dict[str, Any],
    candidate_label: str,
    control_labels: Sequence[str],
) -> tuple[set[str], dict[str, dict[str, Any]], list[str]]:
    overlaps = probe_payload.get("candidate_control_overlap", {}).get(candidate_label, {})
    controls_by_label = _rows_by_label(probe_payload.get("controls", []))
    retained_union: set[str] = set()
    details: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for control_label in control_labels:
        row = overlaps.get(control_label, {})
        retained = {str(value) for value in row.get("candidate_teacher_only_retained_ids", [])}
        present = control_label in controls_by_label and bool(row)
        if not present:
            missing.append(control_label)
        retained_union.update(retained)
        details[control_label] = {
            "present": present,
            "candidate_teacher_only_retained_count": int(
                row.get("candidate_teacher_only_retained_count", len(retained))
            ),
            "candidate_teacher_only_retained_ids": sorted(retained),
            "control_teacher_only_recovered_count": int(
                row.get(
                    "control_teacher_only_recovered_count",
                    len(_as_id_set(controls_by_label.get(control_label), "teacher_only_recovered_ids")),
                )
            ),
        }
    return retained_union, details, missing


def _select_candidate(
    candidates: list[dict[str, Any]],
    candidate_label: str | None,
) -> dict[str, Any]:
    if candidate_label is None:
        if len(candidates) != 1:
            labels = [str(row.get("label")) for row in candidates]
            raise ValueError(f"--candidate-label is required; candidates present: {labels}")
        return candidates[0]
    by_label = _rows_by_label(candidates)
    if candidate_label not in by_label:
        raise KeyError(f"Candidate {candidate_label!r} not found: {sorted(by_label)}")
    return by_label[candidate_label]


def evaluate_sidecar_bound(
    probe_payload: dict[str, Any],
    target_set_payload: dict[str, Any],
    *,
    config: SidecarGateConfig,
) -> dict[str, Any]:
    validate_provenance(probe_payload, target_set_payload, config=config)
    controls = _rows_by_label(probe_payload.get("controls", []))
    target_self = controls.get(config.target_self_label)
    if target_self is None:
        raise KeyError(f"Target-self control {config.target_self_label!r} not found")
    candidate = _select_candidate(list(probe_payload.get("candidates", [])), config.candidate_label)
    candidate_label = str(candidate["label"])

    clean_ids = _clean_residual_ids(target_set_payload)
    candidate_teacher_ids = _as_id_set(candidate, "teacher_only_recovered_ids")
    target_self_teacher_ids = _as_id_set(target_self, "teacher_only_recovered_ids")
    source_retained_ids, source_control_details, missing_controls = _candidate_control_retained_ids(
        probe_payload,
        candidate_label,
        config.source_control_labels,
    )

    clean_candidate_ids = candidate_teacher_ids & clean_ids
    clean_source_necessary_ids = clean_candidate_ids - source_retained_ids
    clean_new_vs_target_self_ids = clean_source_necessary_ids - target_self_teacher_ids

    target_self_correct = int(target_self.get("correct", 0))
    oracle_correct = target_self_correct + len(clean_new_vs_target_self_ids)
    delta_vs_target_self = oracle_correct - target_self_correct
    criteria = {
        "source_controls_present": not missing_controls,
        "min_correct": oracle_correct >= config.min_correct,
        "min_delta_vs_target_self": delta_vs_target_self >= config.min_delta_vs_target_self,
        "min_clean_source_necessary": (
            len(clean_new_vs_target_self_ids) >= config.min_clean_source_necessary
        ),
        "target_losses_prevented_by_sidecar": True,
    }
    failing = [name for name, passed in criteria.items() if not passed]
    status = (
        "oracle_sidecar_bound_clears_gate_not_method"
        if not failing
        else "oracle_sidecar_bound_fails_gate"
    )
    return {
        "date": str(date.today()),
        "status": status,
        "interpretation": (
            "This is an oracle upper bound for a target-self-preserving sidecar. "
            "It is not a deployable method because it assumes perfect knowledge of "
            "which clean candidate wins to add."
        ),
        "config": {
            "target_self_label": config.target_self_label,
            "candidate_label": candidate_label,
            "source_control_labels": list(config.source_control_labels),
            "min_correct": config.min_correct,
            "min_clean_source_necessary": config.min_clean_source_necessary,
            "min_delta_vs_target_self": config.min_delta_vs_target_self,
            "min_numeric_coverage": config.min_numeric_coverage,
            "expected_n": config.expected_n,
        },
        "artifacts": {
            "probe_json": None,
            "target_set_json": None,
        },
        "reference": {
            "n": int(probe_payload.get("reference_n", 0)),
            "target_correct": int(probe_payload.get("target_summary", {}).get("correct", 0)),
            "teacher_correct": int(probe_payload.get("teacher_summary", {}).get("correct", 0)),
            "teacher_only_count": int(probe_payload.get("teacher_only_count", 0)),
            "target_self_repair_correct": target_self_correct,
            "target_self_repair_teacher_only_count": len(target_self_teacher_ids),
            "clean_residual_target_count": len(clean_ids),
        },
        "candidate": {
            "label": candidate_label,
            "matched_correct": int(candidate.get("correct", 0)),
            "matched_teacher_only_recovered_count": len(candidate_teacher_ids),
            "matched_teacher_only_recovered_ids": sorted(candidate_teacher_ids),
            "matched_clean_residual_recovered_count": len(clean_candidate_ids),
            "matched_clean_residual_recovered_ids": sorted(clean_candidate_ids),
            "retained_by_source_controls_count": len(source_retained_ids),
            "retained_by_source_controls_ids": sorted(source_retained_ids),
            "clean_source_necessary_count": len(clean_source_necessary_ids),
            "clean_source_necessary_ids": sorted(clean_source_necessary_ids),
            "clean_new_vs_target_self_count": len(clean_new_vs_target_self_ids),
            "clean_new_vs_target_self_ids": sorted(clean_new_vs_target_self_ids),
            "source_control_details": source_control_details,
            "missing_source_control_labels": missing_controls,
        },
        "oracle_sidecar_bound": {
            "correct": oracle_correct,
            "delta_vs_target_self": delta_vs_target_self,
            "target_losses_vs_target_self": 0,
            "clean_source_necessary_count": len(clean_new_vs_target_self_ids),
            "clean_source_necessary_ids": sorted(clean_new_vs_target_self_ids),
            "criteria": criteria,
            "failing_criteria": failing,
        },
    }


def write_markdown(payload: dict[str, Any], output_md: pathlib.Path) -> None:
    ref = payload["reference"]
    cand = payload["candidate"]
    bound = payload["oracle_sidecar_bound"]
    lines = [
        "# SVAMP32 Source-Innovation Sidecar Bound",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- target: `{ref['target_correct']}/{ref['n']}`",
        f"- C2C teacher: `{ref['teacher_correct']}/{ref['n']}`",
        f"- target_self_repair: `{ref['target_self_repair_correct']}/{ref['n']}`",
        f"- clean residual target set: `{ref['clean_residual_target_count']}`",
        f"- candidate: `{cand['label']}`",
        "",
        "## Bound",
        "",
        f"- oracle target_self_repair + clean source sidecar: `{bound['correct']}/{ref['n']}`",
        f"- delta vs target_self_repair: `{bound['delta_vs_target_self']:+d}`",
        f"- target losses vs target_self_repair: `{bound['target_losses_vs_target_self']}`",
        f"- clean source-necessary IDs: `{bound['clean_source_necessary_count']}`",
        f"- failing criteria: `{', '.join(bound['failing_criteria']) or 'none'}`",
        "",
        "## Candidate Accounting",
        "",
        f"- matched candidate correct: `{cand['matched_correct']}/{ref['n']}`",
        f"- matched C2C-only recovered: `{cand['matched_teacher_only_recovered_count']}`",
        f"- matched clean residual recovered: `{cand['matched_clean_residual_recovered_count']}`",
        f"- retained by source controls: `{cand['retained_by_source_controls_count']}`",
        "- clean source-necessary IDs: "
        + (
            ", ".join(f"`{item}`" for item in cand["clean_source_necessary_ids"])
            if cand["clean_source_necessary_ids"]
            else "none"
        ),
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--candidate-label")
    parser.add_argument("--target-self-label", default=SidecarGateConfig.target_self_label)
    parser.add_argument("--source-control-label", action="append", default=None)
    parser.add_argument("--min-correct", type=int, default=SidecarGateConfig.min_correct)
    parser.add_argument(
        "--min-clean-source-necessary",
        type=int,
        default=SidecarGateConfig.min_clean_source_necessary,
    )
    parser.add_argument(
        "--min-delta-vs-target-self",
        type=int,
        default=SidecarGateConfig.min_delta_vs_target_self,
    )
    parser.add_argument(
        "--min-numeric-coverage",
        type=int,
        default=SidecarGateConfig.min_numeric_coverage,
    )
    parser.add_argument("--expected-n", type=int, default=SidecarGateConfig.expected_n)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    probe_path = _resolve(args.probe_json)
    target_set_path = _resolve(args.target_set_json)
    config = SidecarGateConfig(
        target_self_label=args.target_self_label,
        source_control_labels=tuple(
            args.source_control_label or SidecarGateConfig.source_control_labels
        ),
        candidate_label=args.candidate_label,
        min_correct=args.min_correct,
        min_clean_source_necessary=args.min_clean_source_necessary,
        min_delta_vs_target_self=args.min_delta_vs_target_self,
        min_numeric_coverage=args.min_numeric_coverage,
        expected_n=args.expected_n,
    )
    payload = evaluate_sidecar_bound(
        _load_json(probe_path),
        _load_json(target_set_path),
        config=config,
    )
    payload["artifacts"] = {
        "probe_json": _display_path(probe_path),
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
