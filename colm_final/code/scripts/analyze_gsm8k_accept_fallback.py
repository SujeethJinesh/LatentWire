#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return harness.read_jsonl(path)


def _records_for_method(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(records)
    if method not in grouped:
        raise KeyError(f"Method {method!r} not found in {sorted(grouped)}")
    return grouped[method]


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["example_id"]): record for record in records}


def _ordered_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["example_id"]) for record in records]


def _numeric_prediction(record: dict[str, Any]) -> str | None:
    normalized = record.get("normalized_prediction")
    if normalized is not None:
        return str(normalized)
    return harness._extract_prediction_numeric_answer(str(record.get("prediction", "")))


def _is_degenerate_numeric(record: dict[str, Any], *, max_numeric_len: int) -> bool:
    numeric = _numeric_prediction(record)
    if numeric is not None and len(numeric) > max_numeric_len:
        return True
    prediction = str(record.get("prediction", "")).strip()
    return bool(re.fullmatch(r"(.)\1{15,}", prediction))


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


def _score(record: dict[str, Any], score_field: str) -> float | None:
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


def _threshold_for_quantile(records: list[dict[str, Any]], *, score_field: str, quantile: float) -> float:
    values = sorted(score for record in records if (score := _score(record, score_field)) is not None)
    if not values:
        raise ValueError(f"No numeric scores found for {score_field!r}")
    q = min(max(float(quantile), 0.0), 1.0)
    index = int(q * (len(values) - 1))
    return values[index]


def _paired_counts(records: list[dict[str, Any]], target_records: list[dict[str, Any]]) -> dict[str, int]:
    return harness.paired_vs_baseline(records, target_records)


def _result_record(
    *,
    candidate_record: dict[str, Any],
    target_record: dict[str, Any],
    accepted: bool,
    policy: str,
) -> dict[str, Any]:
    chosen = candidate_record if accepted else target_record
    row = dict(chosen)
    row["method"] = policy
    row["accepted_candidate"] = bool(accepted)
    row["fallback_to_target"] = not accepted
    row["candidate_numeric_prediction"] = _numeric_prediction(candidate_record)
    row["target_numeric_prediction"] = _numeric_prediction(target_record)
    return row


def _policy_accepts(
    *,
    policy: dict[str, Any],
    candidate_record: dict[str, Any],
    target_record: dict[str, Any],
    score_field: str,
    max_numeric_len: int,
) -> bool:
    if policy["kind"] == "target_only":
        return False

    candidate_numeric = _numeric_prediction(candidate_record)
    target_numeric = _numeric_prediction(target_record)
    if candidate_numeric is None:
        return False
    if _is_degenerate_numeric(candidate_record, max_numeric_len=max_numeric_len):
        return False

    if policy["kind"] in {"numeric", "score_quantile"} and candidate_numeric == target_numeric:
        return False

    if policy["kind"] == "numeric":
        return True

    if policy["kind"] == "score_quantile":
        score = _score(candidate_record, score_field)
        return score is not None and score >= float(policy["threshold"])

    raise ValueError(f"Unknown policy kind: {policy['kind']}")


def _apply_policy(
    *,
    label: str,
    records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    policy: dict[str, Any],
    score_field: str,
    max_numeric_len: int,
    reference_live_win_ids: set[str],
) -> dict[str, Any]:
    target_by_id = _by_id(target_records)
    accepted_ids: list[str] = []
    accepted_correct_ids: list[str] = []
    accepted_wrong_ids: list[str] = []
    replay_records: list[dict[str, Any]] = []
    for candidate_record in records:
        example_id = str(candidate_record["example_id"])
        target_record = target_by_id[example_id]
        accepted = _policy_accepts(
            policy=policy,
            candidate_record=candidate_record,
            target_record=target_record,
            score_field=score_field,
            max_numeric_len=max_numeric_len,
        )
        if accepted:
            accepted_ids.append(example_id)
            if bool(candidate_record["correct"]):
                accepted_correct_ids.append(example_id)
            else:
                accepted_wrong_ids.append(example_id)
        replay_records.append(
            _result_record(
                candidate_record=candidate_record,
                target_record=target_record,
                accepted=accepted,
                policy=policy["name"],
            )
        )

    paired = _paired_counts(replay_records, target_records)
    correct = sum(int(bool(record["correct"])) for record in replay_records)
    numeric_coverage = sum(
        int(harness._has_numeric_extraction(str(record.get("prediction", ""))))
        for record in replay_records
    )
    live_win_retention_ids = sorted(set(accepted_ids) & set(reference_live_win_ids))
    return {
        "label": label,
        "policy": policy["name"],
        "correct": int(correct),
        "n": len(replay_records),
        "accuracy": float(correct / max(len(replay_records), 1)),
        "paired_vs_target": paired,
        "accepted_count": len(accepted_ids),
        "accepted_correct_count": len(accepted_correct_ids),
        "accepted_wrong_count": len(accepted_wrong_ids),
        "accepted_ids": accepted_ids,
        "accepted_correct_ids": accepted_correct_ids,
        "accepted_wrong_ids": accepted_wrong_ids,
        "live_win_retention_count": len(live_win_retention_ids),
        "live_win_retention_ids": live_win_retention_ids,
        "numeric_extraction_coverage": int(numeric_coverage),
    }


def _source_live_win_ids(records: list[dict[str, Any]], target_records: list[dict[str, Any]]) -> set[str]:
    target_by_id = _by_id(target_records)
    return {
        str(record["example_id"])
        for record in records
        if bool(record["correct"]) and not bool(target_by_id[str(record["example_id"])]["correct"])
    }


def _parse_labeled_path(spec: str) -> tuple[str, pathlib.Path]:
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"Expected label=path, got {spec!r}")
    return label, pathlib.Path(path)


def _gate(
    *,
    policy_name: str,
    target_correct: int,
    candidate_summaries: list[dict[str, Any]],
    control_summaries: list[dict[str, Any]],
    max_control_accepts: int,
    max_control_live_win_retention: int,
) -> dict[str, Any]:
    seed0 = next((item for item in candidate_summaries if item["label"] == "seed0"), None)
    finite_repeats = [item for item in candidate_summaries if item["label"] != "seed0"]
    checks = {
        "seed0_beats_target": bool(seed0 and int(seed0["correct"]) > target_correct),
        "seed0_no_harms": bool(seed0 and int(seed0["paired_vs_target"]["loss"]) == 0),
        "finite_repeats_nonnegative": all(
            int(item["correct"]) >= target_correct and int(item["paired_vs_target"]["loss"]) == 0
            for item in finite_repeats
        ),
        "controls_do_not_beat_target": all(
            int(item["paired_vs_target"]["win"]) == 0 for item in control_summaries
        ),
        "controls_accept_rarely": all(
            int(item["accepted_count"]) <= max_control_accepts for item in control_summaries
        ),
        "controls_do_not_retain_live_wins": all(
            int(item["live_win_retention_count"]) <= max_control_live_win_retention
            for item in control_summaries
        ),
    }
    passed = all(checks.values())
    return {
        "policy": policy_name,
        "status": "accept_fallback_clears_offline_gate" if passed else "accept_fallback_does_not_clear_gate",
        "checks": checks,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K Accept/Fallback Replay",
        "",
        f"- date: `{payload['date']}`",
        f"- target correct: `{payload['target_summary']['correct']} / {payload['target_summary']['n']}`",
        f"- score field: `{payload['config']['score_field']}`",
        f"- threshold reference: `{payload['config']['threshold_reference_label']}`",
        "",
        "## Policy Gates",
        "",
        "| Policy | Status | Seed0 | Repeats | Controls |",
        "|---|---|---:|---:|---:|",
    ]
    for gate in payload["gates"]:
        checks = gate["checks"]
        lines.append(
            f"| `{gate['policy']}` | {gate['status']} | "
            f"{'pass' if checks['seed0_beats_target'] and checks['seed0_no_harms'] else 'fail'} | "
            f"{'pass' if checks['finite_repeats_nonnegative'] else 'fail'} | "
            f"{'pass' if checks['controls_do_not_beat_target'] and checks['controls_accept_rarely'] and checks['controls_do_not_retain_live_wins'] else 'fail'} |"
        )
    lines.extend(["", "## Summaries", ""])
    for group_name in ("candidates", "controls"):
        lines.extend(
            [
                f"### {group_name.title()}",
                "",
                "| Label | Policy | Correct | Pair vs target | Accepted | Live-win retention |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in payload[group_name]:
            pair = item["paired_vs_target"]
            lines.append(
                f"| `{item['label']}` | `{item['policy']}` | {item['correct']}/{item['n']} | "
                f"{pair['win']}/{pair['loss']}/{pair['tie']} | {item['accepted_count']} | "
                f"{item['live_win_retention_count']} |"
            )
        lines.append("")
    lines.extend(["## Thresholds", ""])
    for policy in payload["policies"]:
        if "threshold" in policy:
            lines.append(f"- `{policy['name']}`: `{policy['threshold']:.6f}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_records = _records_for_method(_read_jsonl(ROOT / args.baseline_predictions), args.target_method)
    reference_ids = _ordered_ids(target_records)

    candidates: dict[str, list[dict[str, Any]]] = {}
    for spec in args.candidate:
        label, path = _parse_labeled_path(spec)
        records = _records_for_method(_read_jsonl(ROOT / path), args.candidate_method)
        if _ordered_ids(records) != reference_ids:
            raise ValueError(f"Candidate {label!r} does not match target ordered IDs")
        candidates[label] = records

    controls: dict[str, list[dict[str, Any]]] = {}
    for spec in args.control:
        label, path = _parse_labeled_path(spec)
        records = _records_for_method(_read_jsonl(ROOT / path), args.control_method)
        if _ordered_ids(records) != reference_ids:
            raise ValueError(f"Control {label!r} does not match target ordered IDs")
        controls[label] = records

    if args.threshold_reference not in candidates:
        raise ValueError(f"Threshold reference {args.threshold_reference!r} not in candidates")

    policies: list[dict[str, Any]] = [
        {"name": "target_only", "kind": "target_only"},
        {"name": "numeric_changed", "kind": "numeric"},
    ]
    for quantile in args.score_quantile:
        threshold = _threshold_for_quantile(
            candidates[args.threshold_reference],
            score_field=args.score_field,
            quantile=quantile,
        )
        suffix = f"{quantile:g}".replace(".", "p")
        policies.append(
            {
                "name": f"{args.score_field}_ge_q{suffix}_numeric_changed",
                "kind": "score_quantile",
                "quantile": float(quantile),
                "threshold": float(threshold),
            }
        )

    reference_live_win_ids = _source_live_win_ids(candidates[args.threshold_reference], target_records)
    candidate_summaries: list[dict[str, Any]] = []
    control_summaries: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    target_correct = sum(int(bool(record["correct"])) for record in target_records)
    for policy in policies:
        policy_candidate_summaries = [
            _apply_policy(
                label=label,
                records=records,
                target_records=target_records,
                policy=policy,
                score_field=args.score_field,
                max_numeric_len=args.max_numeric_len,
                reference_live_win_ids=reference_live_win_ids,
            )
            for label, records in candidates.items()
        ]
        policy_control_summaries = [
            _apply_policy(
                label=label,
                records=records,
                target_records=target_records,
                policy=policy,
                score_field=args.score_field,
                max_numeric_len=args.max_numeric_len,
                reference_live_win_ids=reference_live_win_ids,
            )
            for label, records in controls.items()
        ]
        candidate_summaries.extend(policy_candidate_summaries)
        control_summaries.extend(policy_control_summaries)
        gates.append(
            _gate(
                policy_name=policy["name"],
                target_correct=target_correct,
                candidate_summaries=policy_candidate_summaries,
                control_summaries=policy_control_summaries,
                max_control_accepts=args.max_control_accepts,
                max_control_live_win_retention=args.max_control_live_win_retention,
            )
        )

    payload = {
        "date": str(date.today()),
        "config": {
            "baseline_predictions": str(args.baseline_predictions),
            "target_method": args.target_method,
            "candidate_method": args.candidate_method,
            "control_method": args.control_method,
            "score_field": args.score_field,
            "threshold_reference_label": args.threshold_reference,
            "score_quantiles": list(args.score_quantile),
            "max_numeric_len": int(args.max_numeric_len),
            "max_control_accepts": int(args.max_control_accepts),
            "max_control_live_win_retention": int(args.max_control_live_win_retention),
        },
        "target_summary": {
            "correct": int(target_correct),
            "n": len(target_records),
            "accuracy": float(target_correct / max(len(target_records), 1)),
        },
        "reference_live_win_ids": sorted(reference_live_win_ids),
        "policies": policies,
        "candidates": candidate_summaries,
        "controls": control_summaries,
        "gates": gates,
    }
    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay GSM8K candidate rows with target fallback under non-oracle accept policies.")
    parser.add_argument("--baseline-predictions", required=True, type=pathlib.Path)
    parser.add_argument("--target-method", default="target_alone")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate spec label=path. Repeatable.")
    parser.add_argument("--candidate-method", default="rotalign_kv")
    parser.add_argument("--control", action="append", default=[], help="Control spec label=path. Repeatable.")
    parser.add_argument("--control-method", default="rotalign_kv")
    parser.add_argument("--threshold-reference", default="seed0")
    parser.add_argument("--score-field", default="selector_entropy_avg")
    parser.add_argument(
        "--score-quantile",
        type=float,
        action="append",
        dest="score_quantile",
        default=[],
        help="Quantile threshold on --score-field. Repeatable.",
    )
    parser.add_argument(
        "--selector-entropy-quantile",
        type=float,
        action="append",
        dest="score_quantile",
        default=[],
        help="Backward-compatible alias for --score-quantile.",
    )
    parser.add_argument("--max-numeric-len", type=int, default=12)
    parser.add_argument("--max-control-accepts", type=int, default=1)
    parser.add_argument("--max-control-live-win-retention", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-md", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    if not args.score_quantile:
        args.score_quantile = [0.6, 0.7, 0.75, 0.8, 0.9]
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(_parse_args(argv))


if __name__ == "__main__":
    main()
