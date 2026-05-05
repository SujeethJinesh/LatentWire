#!/usr/bin/env python3
"""Audit generated-answer packets on the repaired SVAMP32 C2C replay.

This is a boundary diagnostic, not a candidate ICLR method. It asks what happens
if the packet target is aligned directly to the C2C generated numeric answer.
If the packet works only because it transmits the generated answer or its public
candidate index, the result should strengthen claim boundaries rather than be
promoted as source-private model-to-model communication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_svamp32_c2c_candidate_pool_delta_packet_gate as pool_gate


@dataclass(frozen=True)
class RowAudit:
    index: int
    example_id: str
    answer: tuple[str, ...]
    gold: str
    candidate_values: tuple[str, ...]
    c2c_value: str | None
    c2c_index: int | None
    target_value: str | None
    source_value: str | None
    text_value: str | None


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


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_correct(value: str | None, gold: str) -> bool:
    return value is not None and str(value) == str(gold)


def _generated_prediction_number(row: dict[str, Any]) -> str | None:
    value = pool_gate._canonical_number(
        pool_gate._extract_prediction_numeric_answer(str(row.get("prediction", "")))
    )
    if value is not None:
        return value
    return pool_gate._prediction_number(row)


def _candidate_values(*values: str | None) -> tuple[str, ...]:
    unique = {str(value) for value in values if value is not None}
    return tuple(sorted(unique, key=lambda item: (pool_gate._numeric_float(item), item)))


def _value_at(values: Sequence[str], index: int | None) -> str | None:
    if index is None or not values:
        return None
    return str(values[int(index) % len(values)])


def _build_rows(
    *,
    target_set: dict[str, Any],
    c2c_records: dict[str, dict[str, Any]],
    target_records: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    text_records: dict[str, dict[str, Any]],
) -> list[RowAudit]:
    rows: list[RowAudit] = []
    for index, example_id in enumerate(target_set["reference_ids"]):
        example_id = str(example_id)
        c2c = c2c_records[example_id]
        target = target_records[example_id]
        source = source_records[example_id]
        text = text_records[example_id]
        c2c_value = _generated_prediction_number(c2c)
        target_value = _generated_prediction_number(target)
        source_value = _generated_prediction_number(source)
        text_value = _generated_prediction_number(text)
        values = _candidate_values(c2c_value, target_value, source_value, text_value)
        c2c_index = values.index(c2c_value) if c2c_value in values else None
        rows.append(
            RowAudit(
                index=index,
                example_id=example_id,
                answer=tuple(str(value) for value in target.get("answer", c2c.get("answer", []))),
                gold=pool_gate._gold_number(target.get("answer", c2c.get("answer", []))),
                candidate_values=values,
                c2c_value=c2c_value,
                c2c_index=c2c_index,
                target_value=target_value,
                source_value=source_value,
                text_value=text_value,
            )
        )
    return rows


def _same_index_wrong_row_indices(rows: Sequence[RowAudit]) -> list[int]:
    by_index: dict[int, list[int]] = {}
    for row in rows:
        if row.c2c_index is not None:
            by_index.setdefault(int(row.c2c_index), []).append(int(row.index))
    out: list[int] = []
    n = len(rows)
    for row in rows:
        peers = [
            idx for idx in by_index.get(int(row.c2c_index or 0), [])
            if idx != int(row.index)
        ]
        out.append(peers[0] if peers else (int(row.index) + 1) % n)
    return out


def _evaluate_rows(
    rows: Sequence[RowAudit],
    *,
    target_ids: dict[str, set[str]],
) -> dict[str, Any]:
    same_index_wrong_rows = _same_index_wrong_row_indices(rows)
    detailed_rows: list[dict[str, Any]] = []
    n = len(rows)
    for row in rows:
        next_row = rows[(int(row.index) + 1) % n]
        same_index_row = rows[same_index_wrong_rows[int(row.index)]]
        selections = {
            "generated_answer_value_packet": row.c2c_value,
            "same_byte_visible_answer_text": row.c2c_value,
            "generated_answer_index_packet": _value_at(row.candidate_values, row.c2c_index),
            "target_only": row.target_value,
            "source_alone": row.source_value,
            "text_to_text": row.text_value,
            "wrong_row_value_packet": next_row.c2c_value,
            "index_row_shuffle": _value_at(row.candidate_values, next_row.c2c_index),
            "same_source_choice_wrong_row": _value_at(row.candidate_values, same_index_row.c2c_index),
            "candidate_roll": _value_at(
                row.candidate_values,
                None if row.c2c_index is None else int(row.c2c_index) + 1,
            ),
            "candidate_derangement": _value_at(
                row.candidate_values,
                None if row.c2c_index is None else int(row.c2c_index) + max(1, len(row.candidate_values) // 2),
            ),
            "zero_packet": row.target_value,
        }
        detailed_rows.append(
            {
                "index": int(row.index),
                "example_id": row.example_id,
                "labels": [
                    label for label, ids in target_ids.items() if row.example_id in ids
                ],
                "gold_answer": row.gold,
                "candidate_values": list(row.candidate_values),
                "candidate_pool_size": len(row.candidate_values),
                "c2c_value": row.c2c_value,
                "c2c_index": row.c2c_index,
                "target_value": row.target_value,
                "source_value": row.source_value,
                "text_value": row.text_value,
                "same_index_wrong_row": same_index_row.example_id,
                "conditions": {
                    condition: {
                        "prediction": value,
                        "correct": _condition_correct(value, row.gold),
                    }
                    for condition, value in selections.items()
                },
            }
        )

    target_correct = {
        row["example_id"]
        for row in detailed_rows
        if bool(row["conditions"]["target_only"]["correct"])
    }

    def summarize(condition: str) -> dict[str, Any]:
        correct = {
            row["example_id"]
            for row in detailed_rows
            if bool(row["conditions"][condition]["correct"])
        }
        return {
            "condition": condition,
            "correct_count": len(correct),
            "correct_ids": sorted(correct),
            "teacher_only_correct_count": len(correct & target_ids["teacher_only"]),
            "teacher_only_correct_ids": sorted(correct & target_ids["teacher_only"]),
            "clean_correct_count": len(correct & target_ids["clean_residual_targets"]),
            "clean_correct_ids": sorted(correct & target_ids["clean_residual_targets"]),
            "helps_count": len(correct - target_correct),
            "helps_ids": sorted(correct - target_correct),
            "harms_count": len(target_correct - correct),
            "harms_ids": sorted(target_correct - correct),
        }

    conditions = tuple(detailed_rows[0]["conditions"]) if detailed_rows else ()
    summaries = {condition: summarize(condition) for condition in conditions}
    destructive_controls = tuple(
        condition for condition in conditions
        if condition not in {"generated_answer_value_packet", "generated_answer_index_packet"}
    )
    destructive_clean_union = set().union(
        *[
            set(summaries[condition]["clean_correct_ids"])
            for condition in destructive_controls
        ]
    )
    matched_clean = set(summaries["generated_answer_value_packet"]["clean_correct_ids"])
    publishable_source_necessary = matched_clean - destructive_clean_union
    return {
        "rows": detailed_rows,
        "condition_summaries": summaries,
        "destructive_controls": list(destructive_controls),
        "answer_label_clean_ids": sorted(matched_clean),
        "publishable_source_necessary_clean_ids": sorted(publishable_source_necessary),
    }


def _packet_contract(rows: Sequence[RowAudit]) -> dict[str, Any]:
    pool_sizes = [max(len(row.candidate_values), 1) for row in rows]
    index_bits = [max(1, math.ceil(math.log2(size))) for size in pool_sizes]
    value_bytes = [len(str(row.c2c_value or "").encode("utf-8")) for row in rows]
    avg_index_bits = sum(index_bits) / max(len(index_bits), 1)
    avg_value_bytes = sum(value_bytes) / max(len(value_bytes), 1)
    return {
        "kind": "generated_answer_value_or_candidate_index_packet",
        "not_a_method_reason": "packet exposes the C2C generated numeric answer or its public candidate index",
        "source_private": False,
        "teacher_derived_not_deployable": True,
        "avg_candidate_pool_size": sum(pool_sizes) / max(len(pool_sizes), 1),
        "max_candidate_pool_size": max(pool_sizes) if pool_sizes else 0,
        "avg_index_packet_bits_per_row": avg_index_bits,
        "avg_index_packet_bytes_per_row": avg_index_bits / 8.0,
        "avg_index_packet_cacheline_bytes_per_row": 64.0,
        "avg_visible_answer_text_bytes_per_row": avg_value_bytes,
        "avg_visible_answer_text_cacheline_bytes_per_row": 64.0,
    }


def _write_manifest(output_dir: pathlib.Path, payload: dict[str, Any], artifacts: Sequence[pathlib.Path]) -> None:
    manifest = {
        "date": payload["date"],
        "status": payload["status"],
        "artifacts": {
            _display_path(path): {"sha256": _sha256(path)}
            for path in artifacts
            if path.exists()
        },
    }
    manifest_json = output_dir / "manifest.json"
    manifest_md = output_dir / "manifest.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SVAMP32 C2C Generated-Answer Packet Audit Manifest",
        "",
        f"- date: `{manifest['date']}`",
        f"- status: `{manifest['status']}`",
        "",
        "## Artifacts",
        "",
        "| Path | SHA256 |",
        "|---|---|",
    ]
    for path, meta in manifest["artifacts"].items():
        lines.append(f"| `{path}` | `{meta['sha256']}` |")
    manifest_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summaries = payload["condition_summaries"]
    lines = [
        "# SVAMP32 C2C Generated-Answer Packet Audit",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- output JSON: `{payload['artifacts']['output_json']}`",
        "",
        "## Result",
        "",
        "| Condition | Correct | Teacher-only | Clean | Helps | Harms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in (
        "generated_answer_value_packet",
        "same_byte_visible_answer_text",
        "generated_answer_index_packet",
        "target_only",
        "source_alone",
        "text_to_text",
        "wrong_row_value_packet",
        "index_row_shuffle",
        "same_source_choice_wrong_row",
        "candidate_roll",
        "candidate_derangement",
        "zero_packet",
    ):
        summary = summaries[condition]
        lines.append(
            "| {condition} | {correct}/{n} | {teacher_only} | {clean} | {helps} | {harms} |".format(
                condition=condition,
                correct=summary["correct_count"],
                n=payload["reference_n"],
                teacher_only=summary["teacher_only_correct_count"],
                clean=summary["clean_correct_count"],
                helps=summary["helps_count"],
                harms=summary["harms_count"],
            )
        )
    contract = payload["packet_contract"]
    lines.extend(
        [
            "",
            "## Packet Contract",
            "",
            f"- kind: `{contract['kind']}`",
            f"- source-private: `{contract['source_private']}`",
            f"- teacher-derived and not deployable: `{contract['teacher_derived_not_deployable']}`",
            f"- average candidate-index bits per row: `{contract['avg_index_packet_bits_per_row']:.3f}`",
            f"- average candidate-index bytes per row: `{contract['avg_index_packet_bytes_per_row']:.3f}`",
            f"- average visible-answer text bytes per row: `{contract['avg_visible_answer_text_bytes_per_row']:.3f}`",
            f"- cacheline-rounded bytes per row: `{contract['avg_index_packet_cacheline_bytes_per_row']:.1f}`",
            "",
            "## Interpretation",
            "",
            "Direct generated-answer alignment recovers the dense C2C replay only by "
            "sending the answer itself, or an equivalent candidate index. The "
            "`same_byte_visible_answer_text` control is identical to the generated "
            "answer packet, so this is an upper-bound and leakage audit rather than "
            "a source-private communication method.",
            "",
            f"- answer-label clean IDs: `{len(payload['answer_label_clean_ids'])}`",
            f"- publishable source-necessary clean IDs after destructive controls: "
            f"`{len(payload['publishable_source_necessary_clean_ids'])}`",
            "",
            "## Decision",
            "",
            "Do not promote generated-answer value/index packets as LatentWire. The "
            "next C2C-distillation gate must use pre-answer state or a source-side "
            "packet target that is not equivalent to revealing the generated answer.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze(
    *,
    target_set_path: pathlib.Path,
    run_date: str,
    output_json: pathlib.Path,
    output_md: pathlib.Path,
) -> dict[str, Any]:
    target_set = _read_json(target_set_path)
    artifacts = target_set["artifacts"]
    c2c_spec = artifacts["source"]
    target_spec = artifacts["target"]
    control_specs = {spec["label"]: spec for spec in artifacts["controls"]}
    c2c_records = pool_gate._load_method_records(_resolve(c2c_spec["path"]), c2c_spec["method"])
    target_records = pool_gate._load_method_records(_resolve(target_spec["path"]), target_spec["method"])
    source_records = pool_gate._load_method_records(
        _resolve(control_specs["source_alone"]["path"]),
        control_specs["source_alone"]["method"],
    )
    text_records = pool_gate._load_method_records(
        _resolve(control_specs["text_to_text"]["path"]),
        control_specs["text_to_text"]["method"],
    )
    target_ids = {
        label: {str(value) for value in values}
        for label, values in target_set["ids"].items()
    }
    rows = _build_rows(
        target_set=target_set,
        c2c_records=c2c_records,
        target_records=target_records,
        source_records=source_records,
        text_records=text_records,
    )
    evaluation = _evaluate_rows(rows, target_ids=target_ids)
    matched = evaluation["condition_summaries"]["generated_answer_value_packet"]
    visible = evaluation["condition_summaries"]["same_byte_visible_answer_text"]
    status = (
        "generated_answer_packet_is_answer_leak_not_method"
        if matched["correct_ids"] == visible["correct_ids"]
        else "generated_answer_packet_audit_unexpected_visibility_mismatch"
    )
    payload = {
        "date": run_date,
        "status": status,
        "reference_n": len(rows),
        "sources": {
            "target_set_json": _display_path(target_set_path),
            "c2c_generate": c2c_spec,
            "target": target_spec,
            "controls": artifacts["controls"],
        },
        "target_ids": {label: sorted(values) for label, values in target_ids.items()},
        "packet_contract": _packet_contract(rows),
        "condition_summaries": evaluation["condition_summaries"],
        "answer_label_clean_ids": evaluation["answer_label_clean_ids"],
        "publishable_source_necessary_clean_ids": evaluation["publishable_source_necessary_clean_ids"],
        "destructive_controls": evaluation["destructive_controls"],
        "rows": evaluation["rows"],
        "artifacts": {
            "output_json": _display_path(output_json),
            "output_md": _display_path(output_md),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    _write_manifest(output_json.parent, payload, [output_json, output_md])
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-set-json",
        default="results/svamp32_c2c_mps_compat_replay_20260505/c2c_replay_target_set.json",
    )
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = analyze(
        target_set_path=_resolve(args.target_set_json),
        run_date=str(args.date),
        output_json=_resolve(args.output_json),
        output_md=_resolve(args.output_md),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_json": payload["artifacts"]["output_json"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
