#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _records_for_method(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(records)
    if method in grouped:
        return grouped[method]
    if len(grouped) == 1:
        return next(iter(grouped.values()))
    raise KeyError(f"Method {method!r} not found in {sorted(grouped)}")


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["example_id"]): record for record in records}


def _ordered_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["example_id"]) for record in records]


def _correct_count(records: list[dict[str, Any]]) -> int:
    return int(sum(int(bool(record.get("correct"))) for record in records))


def _numeric_coverage(records: list[dict[str, Any]]) -> int:
    return int(
        sum(
            int(harness._has_numeric_extraction(str(record.get("prediction", ""))))
            for record in records
        )
    )


def _paired_counts(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, int]:
    return harness.paired_vs_baseline(candidate, baseline)


def _correct_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record["example_id"]) for record in records if bool(record.get("correct"))}


def _telemetry_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_indices = [record.get("source_control_source_index") for record in records]
    own_indices = [record.get("index") for record in records]
    has_source_indices = all(index is not None for index in source_indices)
    return {
        "source_prompt_control": dict(Counter(str(record.get("source_prompt_control", "missing")) for record in records)),
        "source_kv_control": dict(Counter(str(record.get("source_kv_control", "missing")) for record in records)),
        "translated_kv_control": dict(Counter(str(record.get("translated_kv_control", "missing")) for record in records)),
        "source_index_telemetry_present": has_source_indices,
        "source_index_deranged": bool(
            has_source_indices
            and len(records) > 1
            and all(int(src_idx) != int(idx) for src_idx, idx in zip(source_indices, own_indices))
        ),
    }


def _row_summary(
    *,
    label: str,
    records: list[dict[str, Any]],
    reference_ids: list[str],
    target_records: list[dict[str, Any]],
    live_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {
        "label": label,
        "n": len(records),
        "correct": _correct_count(records),
        "accuracy": float(_correct_count(records) / max(len(records), 1)),
        "numeric_extraction_coverage": _numeric_coverage(records),
        "empty_predictions": int(sum(int(not str(row.get("prediction", "")).strip()) for row in records)),
        "ordered_id_parity": _ordered_ids(records) == reference_ids,
        "set_id_parity": set(_ordered_ids(records)) == set(reference_ids),
        "paired_vs_target": _paired_counts(records, target_records),
        "telemetry": _telemetry_summary(records),
    }
    if live_records is not None:
        live_by_id = _by_id(live_records)
        target_correct = _correct_ids(target_records)
        live_correct = _correct_ids(live_records)
        row_correct = _correct_ids(records)
        live_win_ids = sorted(example_id for example_id in live_correct if example_id not in target_correct)
        control_win_ids = sorted(example_id for example_id in row_correct if example_id not in target_correct)
        summary.update(
            {
                "paired_vs_live": _paired_counts(records, live_records),
                "live_win_count": len(live_win_ids),
                "live_win_retention_count": len(row_correct.intersection(live_win_ids)),
                "live_win_retention_ids": sorted(row_correct.intersection(live_win_ids)),
                "control_only_win_count_vs_target": len(control_win_ids),
                "control_only_win_ids_vs_target": control_win_ids,
                "live_correct_control_wrong": sorted(
                    example_id for example_id in live_correct if example_id not in row_correct
                ),
                "control_correct_live_wrong": sorted(
                    example_id for example_id in row_correct if not bool(live_by_id[example_id].get("correct"))
                ),
            }
        )
    return summary


def _source_control_gate(
    *,
    target_correct: int,
    live_summary: dict[str, Any],
    control_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for summary in control_summaries:
        paired = summary["paired_vs_target"]
        checks.append(
            {
                "label": summary["label"],
                "accuracy_collapses_to_target_or_below_plus_one": int(summary["correct"]) <= int(target_correct) + 1,
                "wins_not_above_losses_vs_target": int(paired["win"]) <= int(paired["loss"]),
                "low_live_win_retention": int(summary.get("live_win_retention_count", 0)) <= 1,
                "valid_artifact": bool(summary["ordered_id_parity"])
                and int(summary["empty_predictions"]) == 0
                and int(summary["numeric_extraction_coverage"]) == int(summary["n"]),
            }
        )
    decisive_labels = {
        str(summary["label"])
        for summary in control_summaries
        if "shuffle" in str(summary["label"])
    }
    if not decisive_labels:
        decisive_labels = {str(summary["label"]) for summary in control_summaries}

    def _negative_control_collapses(row: dict[str, Any]) -> bool:
        return (
            bool(row["accuracy_collapses_to_target_or_below_plus_one"])
            and bool(row["wins_not_above_losses_vs_target"])
            and bool(row["low_live_win_retention"])
        )

    passed = all(_negative_control_collapses(row) for row in checks) and all(
        bool(row["valid_artifact"]) for row in checks if str(row["label"]) in decisive_labels
    )
    return {
        "status": "source_controls_support_matched_source_signal" if passed else "source_controls_do_not_clear_gate",
        "checks": checks,
        "decisive_control_labels": sorted(decisive_labels),
        "live_correct": live_summary["correct"],
        "target_correct": target_correct,
        "required_next_if_pass": "repeat on next finite valid seed, then strict cross-family falsification",
        "required_next_if_fail": "demote live row to target-cache/control artifact and pivot method",
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K Source-Control Readout",
        "",
        f"- date: `{payload['date']}`",
        f"- live label: `{payload['live_label']}`",
        f"- target correct: `{payload['target_summary']['correct']} / {payload['target_summary']['n']}`",
        f"- live correct: `{payload['live_summary']['correct']} / {payload['live_summary']['n']}`",
        f"- gate: `{payload['gate']['status']}`",
        "",
        "## Controls",
        "",
        "| Row | Correct | Pair vs target | Pair vs live | Live-win retention | Numeric coverage | Deranged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in payload["control_summaries"]:
        target_pair = summary["paired_vs_target"]
        live_pair = summary.get("paired_vs_live", {"win": 0, "loss": 0, "tie": summary["n"]})
        lines.append(
            f"| {summary['label']} | {summary['correct']}/{summary['n']} | "
            f"{target_pair['win']}/{target_pair['loss']}/{target_pair['tie']} | "
            f"{live_pair['win']}/{live_pair['loss']}/{live_pair['tie']} | "
            f"{summary.get('live_win_retention_count', 0)}/{summary.get('live_win_count', 0)} | "
            f"{summary['numeric_extraction_coverage']}/{summary['n']} | "
            f"{summary['telemetry']['source_index_deranged']} |"
        )
    lines.extend(["", "## Gate Checks", ""])
    for check in payload["gate"]["checks"]:
        lines.append(
            f"- `{check['label']}`: collapse=`{check['accuracy_collapses_to_target_or_below_plus_one']}`, "
            f"wins<=losses=`{check['wins_not_above_losses_vs_target']}`, "
            f"low retention=`{check['low_live_win_retention']}`, valid=`{check['valid_artifact']}`"
        )
    lines.extend(["", "## Artifact Paths", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_records = _read_jsonl(ROOT / args.baseline_predictions)
    target_records = _records_for_method(baseline_records, args.target_method)
    live_records = _records_for_method(_read_jsonl(ROOT / args.live_predictions), args.live_method)
    reference_ids = _ordered_ids(target_records)
    live_summary = _row_summary(
        label=args.live_label,
        records=live_records,
        reference_ids=reference_ids,
        target_records=target_records,
    )
    target_summary = _row_summary(
        label=args.target_method,
        records=target_records,
        reference_ids=reference_ids,
        target_records=target_records,
    )

    control_summaries: list[dict[str, Any]] = []
    control_artifacts: dict[str, str] = {}
    for spec in args.control:
        label, path = spec.split("=", 1)
        control_artifacts[label] = path
        control_records = _records_for_method(_read_jsonl(ROOT / path), args.control_method)
        control_summaries.append(
            _row_summary(
                label=label,
                records=control_records,
                reference_ids=reference_ids,
                target_records=target_records,
                live_records=live_records,
            )
        )

    payload = {
        "date": str(date.today()),
        "live_label": args.live_label,
        "artifacts": {
            "baseline_predictions": args.baseline_predictions,
            "live_predictions": args.live_predictions,
            **{f"control_{label}": path for label, path in control_artifacts.items()},
        },
        "target_summary": target_summary,
        "live_summary": live_summary,
        "control_summaries": control_summaries,
        "gate": _source_control_gate(
            target_correct=int(target_summary["correct"]),
            live_summary=live_summary,
            control_summaries=control_summaries,
        ),
    }
    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GSM8K live predictions against source controls.")
    parser.add_argument("--baseline-predictions", required=True, help="JSONL containing target_alone records.")
    parser.add_argument("--target-method", default="target_alone")
    parser.add_argument("--live-predictions", required=True)
    parser.add_argument("--live-method", default="rotalign_kv_gate_0.10")
    parser.add_argument("--live-label", default="live")
    parser.add_argument("--control", action="append", default=[], help="Control spec label=path. Repeatable.")
    parser.add_argument("--control-method", default="rotalign_kv")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    if not args.control:
        parser.error("at least one --control label=path is required")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(_parse_args(argv))


if __name__ == "__main__":
    main()
