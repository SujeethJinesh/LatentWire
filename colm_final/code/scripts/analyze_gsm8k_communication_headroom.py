#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _parse_spec(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise ValueError(f"Expected label=path spec, got {spec!r}")
    label, path = spec.split("=", 1)
    if not label:
        raise ValueError(f"Missing label in spec {spec!r}")
    return label, _resolve(path)


def _records_for_method(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(records)
    if method in grouped:
        return grouped[method]
    if len(grouped) == 1:
        return next(iter(grouped.values()))
    raise KeyError(f"Method {method!r} not found in {sorted(grouped)}")


def _ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["example_id"]) for record in records]


def _correct_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record["example_id"]) for record in records if bool(record.get("correct"))}


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["example_id"]): record for record in records}


def _selector_trace_stat(record: dict[str, Any], *, key: str, stat: str) -> float | None:
    values = [
        float(item[key])
        for item in record.get("selector_trace", [])
        if isinstance(item, dict) and item.get(key) is not None
    ]
    if not values:
        return None
    if stat == "min":
        return min(values)
    if stat == "avg":
        return sum(values) / len(values)
    if stat == "max":
        return max(values)
    raise ValueError(f"Unknown selector trace stat: {stat}")


def _score(record: dict[str, Any], score_field: str | None) -> float | None:
    if not score_field:
        return None
    derived_selector_fields = {
        "selector_gap_min": ("score_gap", "min"),
        "selector_gap_avg": ("score_gap", "avg"),
        "selector_gap_max": ("score_gap", "max"),
        "selector_entropy_min": ("score_entropy", "min"),
        "selector_entropy_avg_trace": ("score_entropy", "avg"),
        "selector_entropy_max": ("score_entropy", "max"),
        "selector_top_min": ("score_top", "min"),
        "selector_top_avg": ("score_top", "avg"),
        "selector_top_max": ("score_top", "max"),
    }
    if score_field in derived_selector_fields:
        key, stat = derived_selector_fields[score_field]
        return _selector_trace_stat(record, key=key, stat=stat)
    value = record.get(score_field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_row(record: dict[str, Any] | None, *, score_field: str | None = None) -> dict[str, Any]:
    if record is None:
        return {"present": False, "correct": False}
    return {
        "present": True,
        "correct": bool(record.get("correct")),
        "normalized_prediction": record.get("normalized_prediction"),
        "prediction": str(record.get("prediction", "")),
        "score": _score(record, score_field),
    }


def _summary(
    *,
    label: str,
    records: list[dict[str, Any]],
    reference_ids: list[str],
    target_records: list[dict[str, Any]],
    text_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    correct = len(_correct_ids(records))
    summary = {
        "label": label,
        "n": len(records),
        "correct": correct,
        "accuracy": float(correct / max(len(records), 1)),
        "ordered_id_parity": _ids(records) == reference_ids,
        "set_id_parity": set(_ids(records)) == set(reference_ids),
        "empty_predictions": int(
            sum(int(not str(record.get("prediction", "")).strip()) for record in records)
        ),
        "numeric_extraction_coverage": int(
            sum(
                int(harness._has_numeric_extraction(str(record.get("prediction", ""))))
                for record in records
            )
        ),
        "paired_vs_target": harness.paired_vs_baseline(records, target_records),
    }
    if text_records is not None:
        summary["paired_vs_text"] = harness.paired_vs_baseline(records, text_records)
    return summary


def _oracle_row(label: str, row_sets: dict[str, set[str]]) -> dict[str, Any]:
    union_ids: set[str] = set()
    for ids in row_sets.values():
        union_ids.update(ids)
    return {
        "label": label,
        "correct": len(union_ids),
        "components": {name: len(ids) for name, ids in row_sets.items()},
    }


def _candidate_analysis(
    *,
    label: str,
    records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    text_records: list[dict[str, Any]] | None,
    source_records: dict[str, list[dict[str, Any]]],
    control_records: dict[str, list[dict[str, Any]]],
    reference_ids: list[str],
    score_field: str | None,
    score_margin: float,
) -> dict[str, Any]:
    target_correct = _correct_ids(target_records)
    candidate_correct = _correct_ids(records)
    candidate_wins = candidate_correct - target_correct
    candidate_losses = target_correct - candidate_correct
    text_correct = _correct_ids(text_records) if text_records is not None else set()
    source_sets = {label: _correct_ids(rows) for label, rows in source_records.items()}
    control_sets = {label: _correct_ids(rows) for label, rows in control_records.items()}
    candidate_by_id = _by_id(records)
    target_by_id = _by_id(target_records)
    text_by_id = _by_id(text_records) if text_records is not None else {}
    source_by_id = {label: _by_id(rows) for label, rows in source_records.items()}
    control_by_id = {label: _by_id(rows) for label, rows in control_records.items()}

    source_overlap: dict[str, Any] = {}
    for source_label, source_correct in source_sets.items():
        source_only_wins = source_correct - target_correct
        source_overlap[source_label] = {
            "source_correct": len(source_correct),
            "source_only_win_count_vs_target": len(source_only_wins),
            "source_only_win_ids_vs_target": sorted(source_only_wins),
            "candidate_win_source_correct_count": len(candidate_wins & source_correct),
            "candidate_win_source_correct_ids": sorted(candidate_wins & source_correct),
            "candidate_win_source_wrong_count": len(candidate_wins - source_correct),
            "candidate_win_source_only_overlap_count": len(candidate_wins & source_only_wins),
            "candidate_win_source_only_overlap_ids": sorted(candidate_wins & source_only_wins),
        }

    control_overlap: dict[str, Any] = {}
    for control_label, control_correct in control_sets.items():
        retained = candidate_wins & control_correct
        control_wins = control_correct - target_correct
        control_overlap[control_label] = {
            "control_correct": len(control_correct),
            "control_win_count_vs_target": len(control_wins),
            "control_win_ids_vs_target": sorted(control_wins),
            "candidate_win_retention_count": len(retained),
            "candidate_win_retention_ids": sorted(retained),
            "candidate_win_retention_ratio": float(len(retained) / max(len(candidate_wins), 1)),
            "control_correct_candidate_wrong_ids": sorted(control_correct - candidate_correct),
        }

    source_any = set().union(*source_sets.values()) if source_sets else set()
    max_control_retention = max(
        (item["candidate_win_retention_count"] for item in control_overlap.values()),
        default=0,
    )
    max_source_on_wins = max(
        (item["candidate_win_source_correct_count"] for item in source_overlap.values()),
        default=0,
    )
    if not candidate_wins:
        gate_status = "no_candidate_wins_vs_target"
    elif max_control_retention >= max(1, len(candidate_wins) // 2):
        gate_status = "candidate_wins_not_source_specific_under_controls"
    elif max_source_on_wins == 0 and len(target_correct | source_any) <= len(target_correct):
        gate_status = "no_measured_source_alone_headroom"
    elif max_source_on_wins == 0:
        gate_status = "candidate_wins_not_explained_by_source_alone"
    else:
        gate_status = "source_headroom_available_for_method_probe"

    win_provenance: list[dict[str, Any]] = []
    for example_id in sorted(candidate_wins):
        sources = {
            source_label: _compact_row(rows.get(example_id), score_field=score_field)
            for source_label, rows in source_by_id.items()
        }
        controls = {
            control_label: _compact_row(rows.get(example_id), score_field=score_field)
            for control_label, rows in control_by_id.items()
        }
        candidate_cell = _compact_row(candidate_by_id.get(example_id), score_field=score_field)
        control_scores = [
            float(row["score"])
            for row in controls.values()
            if row.get("score") is not None
        ]
        candidate_score = candidate_cell.get("score")
        max_control_score = max(control_scores) if control_scores else None
        score_delta = (
            float(candidate_score) - float(max_control_score)
            if candidate_score is not None and max_control_score is not None
            else None
        )
        win_provenance.append(
            {
                "example_id": example_id,
                "candidate": candidate_cell,
                "target": _compact_row(target_by_id.get(example_id), score_field=score_field),
                "text": (
                    _compact_row(text_by_id.get(example_id), score_field=score_field)
                    if text_records is not None
                    else None
                ),
                "sources": sources,
                "controls": controls,
                "source_correct_labels": sorted(
                    label for label, row in sources.items() if bool(row.get("correct"))
                ),
                "control_correct_labels": sorted(
                    label for label, row in controls.items() if bool(row.get("correct"))
                ),
                "score_contrast": {
                    "candidate_score": candidate_score,
                    "max_control_score": max_control_score,
                    "candidate_minus_max_control": score_delta,
                    "passes_margin": (
                        bool(score_delta is not None and score_delta > score_margin)
                        if score_field
                        else None
                    ),
                },
            }
        )

    scored_win_rows = [
        row
        for row in win_provenance
        if row["score_contrast"]["candidate_minus_max_control"] is not None
    ]
    score_contrast = {
        "score_field": score_field,
        "score_margin": score_margin,
        "evaluated_candidate_win_count": len(scored_win_rows),
        "passed_candidate_win_count": int(
            sum(int(bool(row["score_contrast"]["passes_margin"])) for row in scored_win_rows)
        ),
        "passed_candidate_win_ids": [
            row["example_id"]
            for row in scored_win_rows
            if bool(row["score_contrast"]["passes_margin"])
        ],
        "exact_equal_score_count": int(
            sum(
                int(abs(float(row["score_contrast"]["candidate_minus_max_control"])) <= 1e-12)
                for row in scored_win_rows
            )
        ),
    }

    return {
        "label": label,
        "summary": _summary(
            label=label,
            records=records,
            reference_ids=reference_ids,
            target_records=target_records,
            text_records=text_records,
        ),
        "candidate_win_count_vs_target": len(candidate_wins),
        "candidate_win_ids_vs_target": sorted(candidate_wins),
        "candidate_loss_count_vs_target": len(candidate_losses),
        "candidate_loss_ids_vs_target": sorted(candidate_losses),
        "candidate_win_text_correct_count": len(candidate_wins & text_correct),
        "candidate_win_text_correct_ids": sorted(candidate_wins & text_correct),
        "source_overlap": source_overlap,
        "control_overlap": control_overlap,
        "oracle": {
            "target_or_candidate": len(target_correct | candidate_correct),
            "target_or_text": len(target_correct | text_correct) if text_records is not None else None,
            "target_or_any_source": len(target_correct | source_any) if source_sets else None,
            "target_or_candidate_or_any_source": (
                len(target_correct | candidate_correct | source_any) if source_sets else None
            ),
        },
        "candidate_win_provenance": win_provenance,
        "score_contrast": score_contrast,
        "gate_status": gate_status,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K Communication Headroom Diagnostic",
        "",
        f"- date: `{payload['date']}`",
        f"- gate: `{payload['gate']['status']}`",
        f"- target: `{payload['target_summary']['correct']} / {payload['target_summary']['n']}`",
    ]
    if payload.get("text_summary") is not None:
        text_summary = payload["text_summary"]
        lines.append(f"- text_to_text: `{text_summary['correct']} / {text_summary['n']}`")
    lines.extend(["", "## Candidate Summary", ""])
    lines.extend(
        [
            "| Candidate | Correct | Pair vs target | Wins vs target | Max control retained wins | Max source-correct wins | Score-contrast wins | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in payload["candidates"]:
        paired = item["summary"]["paired_vs_target"]
        max_control = max(
            (
                control["candidate_win_retention_count"]
                for control in item["control_overlap"].values()
            ),
            default=0,
        )
        max_source = max(
            (
                source["candidate_win_source_correct_count"]
                for source in item["source_overlap"].values()
            ),
            default=0,
        )
        score_contrast = item["score_contrast"]
        score_cell = (
            f"{score_contrast['passed_candidate_win_count']}/"
            f"{score_contrast['evaluated_candidate_win_count']}"
            if score_contrast["score_field"]
            else "n/a"
        )
        lines.append(
            f"| {item['label']} | {item['summary']['correct']}/{item['summary']['n']} | "
            f"{paired['win']}/{paired['loss']}/{paired['tie']} | "
            f"{item['candidate_win_count_vs_target']} | {max_control} | {max_source} | "
            f"{score_cell} | `{item['gate_status']}` |"
        )

    lines.extend(["", "## Source Rows", ""])
    if payload["sources"]:
        lines.extend(["| Source | Correct | Pair vs target | Oracle target/source |", "|---|---:|---:|---:|"])
        for item in payload["sources"]:
            paired = item["summary"]["paired_vs_target"]
            lines.append(
                f"| {item['label']} | {item['summary']['correct']}/{item['summary']['n']} | "
                f"{paired['win']}/{paired['loss']}/{paired['tie']} | "
                f"{item['oracle_target_or_source']} |"
            )
    else:
        lines.append("- No source rows were provided.")

    lines.extend(["", "## Control Retention", ""])
    for item in payload["candidates"]:
        lines.append(f"### {item['label']}")
        if not item["control_overlap"]:
            lines.append("- No controls were provided.")
            continue
        lines.extend(["| Control | Correct | Wins vs target | Retained candidate wins |", "|---|---:|---:|---:|"])
        for label, control in item["control_overlap"].items():
            lines.append(
                f"| {label} | {control['control_correct']} | "
                f"{control['control_win_count_vs_target']} | "
                f"{control['candidate_win_retention_count']}/{item['candidate_win_count_vs_target']} |"
            )

    lines.extend(["", "## Candidate-Win Provenance", ""])
    for item in payload["candidates"]:
        lines.append(f"### {item['label']}")
        if not item["candidate_win_provenance"]:
            lines.append("- No target-relative candidate wins.")
            continue
        lines.extend(
            [
                "| Example ID | Candidate norm | Source-correct labels | Control-correct labels | Candidate score minus max control |",
                "|---|---:|---|---|---:|",
            ]
        )
        for row in item["candidate_win_provenance"]:
            candidate_norm = row["candidate"].get("normalized_prediction")
            source_labels = ", ".join(row["source_correct_labels"]) or "none"
            control_labels = ", ".join(row["control_correct_labels"]) or "none"
            score_delta = row["score_contrast"]["candidate_minus_max_control"]
            score_delta_cell = "n/a" if score_delta is None else f"{score_delta:.6g}"
            lines.append(
                f"| {row['example_id']} | {candidate_norm} | {source_labels} | "
                f"{control_labels} | {score_delta_cell} |"
            )

    lines.extend(["", "## Gate Notes", ""])
    for note in payload["gate"]["notes"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Artifact Paths", ""])
    for key, value in payload["artifacts"].items():
        if isinstance(value, dict):
            for label, path_value in value.items():
                lines.append(f"- {key}.{label}: `{path_value}`")
        else:
            lines.append(f"- {key}: `{value}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = _resolve(args.baseline_predictions)
    baseline_records = _read_jsonl(baseline_path)
    target_records = _records_for_method(baseline_records, args.target_method)
    text_records = None
    try:
        text_records = _records_for_method(baseline_records, args.text_method)
    except KeyError:
        if args.require_text_method:
            raise

    reference_ids = _ids(target_records)
    target_correct = _correct_ids(target_records)
    target_summary = _summary(
        label=args.target_method,
        records=target_records,
        reference_ids=reference_ids,
        target_records=target_records,
        text_records=text_records,
    )
    text_summary = (
        _summary(
            label=args.text_method,
            records=text_records,
            reference_ids=reference_ids,
            target_records=target_records,
            text_records=text_records,
        )
        if text_records is not None
        else None
    )

    sources: list[dict[str, Any]] = []
    source_records: dict[str, list[dict[str, Any]]] = {}
    source_artifacts: dict[str, str] = {}
    for spec in args.source:
        label, path = _parse_spec(spec)
        records = _records_for_method(_read_jsonl(path), args.source_method)
        correct = _correct_ids(records)
        source_records[label] = records
        source_artifacts[label] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        sources.append(
            {
                "label": label,
                "summary": _summary(
                    label=label,
                    records=records,
                    reference_ids=reference_ids,
                    target_records=target_records,
                    text_records=text_records,
                ),
                "oracle_target_or_source": len(target_correct | correct),
            }
        )

    control_records: dict[str, list[dict[str, Any]]] = {}
    control_artifacts: dict[str, str] = {}
    for spec in args.control:
        label, path = _parse_spec(spec)
        records = _records_for_method(_read_jsonl(path), args.control_method)
        control_records[label] = records
        control_artifacts[label] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    candidates: list[dict[str, Any]] = []
    candidate_artifacts: dict[str, str] = {}
    for spec in args.candidate:
        label, path = _parse_spec(spec)
        records = _records_for_method(_read_jsonl(path), args.candidate_method)
        candidate_artifacts[label] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        candidates.append(
            _candidate_analysis(
                label=label,
                records=records,
                target_records=target_records,
                text_records=text_records,
                source_records=source_records,
                control_records=control_records,
                reference_ids=reference_ids,
                score_field=args.score_field,
                score_margin=float(args.score_margin),
            )
        )

    notes: list[str] = []
    candidate_statuses = {item["label"]: item["gate_status"] for item in candidates}
    if any(status == "candidate_wins_not_source_specific_under_controls" for status in candidate_statuses.values()):
        status = "control_retention_blocks_positive_claim"
        notes.append("At least one candidate has target-relative wins retained by zero/shuffle controls.")
    elif any(status == "source_headroom_available_for_method_probe" for status in candidate_statuses.values()):
        status = "source_headroom_available"
        notes.append("At least one candidate has wins that overlap source-correct rows and avoid control retention.")
    elif candidates:
        status = "no_actionable_source_headroom_for_current_candidates"
        notes.append("Current candidates do not expose source-specific headroom under the provided rows.")
    else:
        status = "no_candidates"
        notes.append("No candidate rows were provided.")
    if sources and all(item["oracle_target_or_source"] <= target_summary["correct"] for item in sources):
        notes.append("Source-alone rows add no measured oracle headroom over target-alone on this slice.")
    if args.score_field:
        for item in candidates:
            contrast = item["score_contrast"]
            notes.append(
                f"{item['label']} score contrast `{args.score_field}` kept "
                f"{contrast['passed_candidate_win_count']} / "
                f"{contrast['evaluated_candidate_win_count']} target-relative wins at margin "
                f"{float(args.score_margin):g}; exact-equal score wins: "
                f"{contrast['exact_equal_score_count']}."
            )

    payload = {
        "date": str(date.today()),
        "score_field": args.score_field,
        "score_margin": float(args.score_margin),
        "artifacts": {
            "baseline_predictions": str(
                baseline_path.relative_to(ROOT) if baseline_path.is_relative_to(ROOT) else baseline_path
            ),
            "candidates": candidate_artifacts,
            "sources": source_artifacts,
            "controls": control_artifacts,
        },
        "target_summary": target_summary,
        "text_summary": text_summary,
        "sources": sources,
        "candidates": candidates,
        "oracle": [
            _oracle_row("target", {"target": target_correct}),
            *(
                [_oracle_row("target_or_text", {"target": target_correct, "text": _correct_ids(text_records)})]
                if text_records is not None
                else []
            ),
            *[
                _oracle_row(
                    f"target_or_source.{label}",
                    {"target": target_correct, label: _correct_ids(records)},
                )
                for label, records in source_records.items()
            ],
        ],
        "gate": {
            "status": status,
            "candidate_statuses": candidate_statuses,
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
        description=(
            "Analyze GSM8K target/source/candidate/control overlap to estimate "
            "communication headroom and source-control leakage."
        )
    )
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--target-method", default="target_alone")
    parser.add_argument("--text-method", default="text_to_text")
    parser.add_argument("--require-text-method", action="store_true")
    parser.add_argument("--source", action="append", default=[], help="Repeatable label=path source row.")
    parser.add_argument("--source-method", default="source_alone")
    parser.add_argument("--candidate", action="append", default=[], help="Repeatable label=path candidate row.")
    parser.add_argument("--candidate-method", default="rotalign_kv")
    parser.add_argument("--control", action="append", default=[], help="Repeatable label=path source-control row.")
    parser.add_argument("--control-method", default="rotalign_kv")
    parser.add_argument("--score-field")
    parser.add_argument(
        "--score-margin",
        type=float,
        default=0.0,
        help="Strict margin for candidate score > max control score on target-relative wins.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    if not args.candidate:
        parser.error("at least one --candidate label=path is required")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(_parse_args(argv))


if __name__ == "__main__":
    main()
