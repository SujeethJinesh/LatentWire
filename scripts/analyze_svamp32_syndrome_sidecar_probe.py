#!/usr/bin/env python3
"""Probe a target-candidate syndrome sidecar bound on frozen SVAMP32 rows.

This is an oracle/bound analysis, not a deployable method. It asks whether a
compact C2C-derived numeric residue can select gold answers already present in
target-side candidate pools, while zero/shuffle/target-only/slots-only controls
fail on the clean residual IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
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


def _by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in records:
        example_id = str(row["example_id"])
        if example_id in out:
            duplicates.add(example_id)
        out[example_id] = dict(row)
    if duplicates:
        raise ValueError(f"Duplicate example_id values: {sorted(duplicates)}")
    return out


def _subset_reference_order(
    records: Sequence[dict[str, Any]],
    reference_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id = _by_id(records)
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Missing reference IDs: {missing}")
    return [by_id[example_id] for example_id in reference_ids]


def _normalize_numeric(value: Any) -> str | None:
    return harness._normalize_numeric_string(str(value))


def _numeric_mentions(text: Any) -> list[str]:
    out: list[str] = []
    for raw in harness._NUMERIC_TOKEN_RE.findall(str(text or "")):
        numeric = _normalize_numeric(raw)
        if numeric is not None:
            out.append(numeric)
    return out


def _prediction_numeric(row: dict[str, Any]) -> str | None:
    normalized = _normalize_numeric(row.get("normalized_prediction", ""))
    if normalized is not None:
        return normalized
    return harness._extract_prediction_numeric_answer(str(row.get("prediction", "")))


def _gold_numeric(row: dict[str, Any]) -> str:
    for field in ("answer", "answer_text", "target"):
        if field in row:
            numeric = harness._extract_reference_numeric_answer(str(row[field]))
            if numeric is not None:
                return numeric
    raise ValueError(f"Could not extract gold numeric answer for {row.get('example_id')}")


def _integer_value(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except InvalidOperation:
        return None
    if decimal != decimal.to_integral_value():
        return None
    return int(decimal)


def _signature(value: str | None, moduli: Sequence[int]) -> tuple[int, ...] | None:
    integer = _integer_value(value)
    if integer is None:
        return None
    return tuple(integer % modulus for modulus in moduli)


def _append_numeric(
    *,
    value: Any,
    label: str,
    ordered_values: list[str],
    labels_by_value: dict[str, list[str]],
) -> None:
    numeric = _normalize_numeric(value)
    if numeric is None:
        return
    if numeric not in labels_by_value:
        ordered_values.append(numeric)
        labels_by_value[numeric] = []
    labels_by_value[numeric].append(label)


def _append_field_values(
    *,
    row: dict[str, Any],
    field: str,
    label: str,
    ordered_values: list[str],
    labels_by_value: dict[str, list[str]],
) -> None:
    value = row.get(field)
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            _append_numeric(
                value=item,
                label=label,
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
            )
        return
    _append_numeric(
        value=value,
        label=label,
        ordered_values=ordered_values,
        labels_by_value=labels_by_value,
    )


def _candidate_pool(
    rows_by_label: dict[str, dict[str, Any]],
    candidate_labels: Sequence[str],
) -> tuple[list[str], dict[str, list[str]]]:
    ordered_values: list[str] = []
    labels_by_value: dict[str, list[str]] = {}
    for label in candidate_labels:
        row = rows_by_label[label]
        prediction_numeric = _prediction_numeric(row)
        if prediction_numeric is not None:
            _append_numeric(
                value=prediction_numeric,
                label=label,
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
            )
        for numeric in _numeric_mentions(row.get("prediction", "")):
            _append_numeric(
                value=numeric,
                label=label,
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
            )
        for field in (
            "candidate_numeric_mentions",
            "candidate_unique_predictions",
            "candidate_vote_prediction",
            "candidate_tail_numeric_mention",
            "selected_candidate_tail_numeric_mention",
            "repair_pre_normalized_prediction",
            "repair_post_normalized_prediction",
        ):
            _append_field_values(
                row=row,
                field=field,
                label=label,
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
            )
    return ordered_values, labels_by_value


def _fallback_prediction(
    *,
    rows_by_label: dict[str, dict[str, Any]],
    fallback_label: str,
    target_label: str,
) -> str | None:
    if fallback_label in rows_by_label:
        fallback = _prediction_numeric(rows_by_label[fallback_label])
        if fallback is not None:
            return fallback
    return _prediction_numeric(rows_by_label[target_label])


def _select_by_syndrome(
    *,
    ordered_values: Sequence[str],
    source_value: str | None,
    moduli: Sequence[int],
    fallback: str | None,
) -> str | None:
    source_signature = _signature(source_value, moduli)
    if source_signature is None:
        return fallback
    for value in ordered_values:
        if _signature(value, moduli) == source_signature:
            return value
    return fallback


def _select_slots_only(
    *,
    ordered_values: Sequence[str],
    labels_by_value: dict[str, list[str]],
    fallback: str | None,
) -> str | None:
    if not ordered_values:
        return fallback
    counts = Counter({value: len(labels_by_value[value]) for value in ordered_values})
    return max(ordered_values, key=lambda value: (counts[value], -ordered_values.index(value)))


def _load_target_ids(path: pathlib.Path) -> dict[str, set[str]]:
    payload = _read_json(path)
    ids = payload.get("ids", {})
    return {
        "teacher_only": {str(value) for value in ids.get("teacher_only", [])},
        "clean_residual_targets": {
            str(value) for value in ids.get("clean_residual_targets", [])
        },
        "target_self_repair": {str(value) for value in ids.get("target_self_repair", [])},
    }


def _moduli_bits(moduli: Sequence[int]) -> float:
    return float(sum(math.log2(modulus) for modulus in moduli))


def _summarize_condition(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    clean_ids: set[str],
    target_self_ids: set[str],
    teacher_only_ids: set[str],
) -> dict[str, Any]:
    correct_ids = {
        str(row["example_id"])
        for row in rows
        if bool(row["conditions"][condition]["correct"])
    }
    return {
        "condition": condition,
        "correct_count": len(correct_ids),
        "correct_ids": sorted(correct_ids),
        "clean_correct_count": len(correct_ids & clean_ids),
        "clean_correct_ids": sorted(correct_ids & clean_ids),
        "target_self_correct_count": len(correct_ids & target_self_ids),
        "target_self_correct_ids": sorted(correct_ids & target_self_ids),
        "teacher_only_correct_count": len(correct_ids & teacher_only_ids),
        "teacher_only_correct_ids": sorted(correct_ids & teacher_only_ids),
    }


def _evaluate_moduli(
    *,
    moduli: Sequence[int],
    reference_ids: Sequence[str],
    target_by_id: dict[str, dict[str, Any]],
    teacher_by_id: dict[str, dict[str, Any]],
    candidate_by_label: dict[str, dict[str, dict[str, Any]]],
    target_label: str,
    fallback_label: str,
    target_ids: dict[str, set[str]],
    shuffle_offset: int,
    min_correct: int,
    min_clean_source_necessary: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_labels = list(candidate_by_label)
    for index, example_id in enumerate(reference_ids):
        rows_by_label = {
            label: records[example_id] for label, records in candidate_by_label.items()
        }
        ordered_values, labels_by_value = _candidate_pool(rows_by_label, candidate_labels)
        fallback = _fallback_prediction(
            rows_by_label=rows_by_label,
            fallback_label=fallback_label,
            target_label=target_label,
        )
        shuffled_id = reference_ids[(index + shuffle_offset) % len(reference_ids)]
        gold = _gold_numeric(target_by_id[example_id])
        teacher_numeric = _prediction_numeric(teacher_by_id[example_id])
        shuffled_teacher_numeric = _prediction_numeric(teacher_by_id[shuffled_id])
        selections = {
            "matched": _select_by_syndrome(
                ordered_values=ordered_values,
                source_value=teacher_numeric,
                moduli=moduli,
                fallback=fallback,
            ),
            "zero_source": _select_by_syndrome(
                ordered_values=ordered_values,
                source_value="0",
                moduli=moduli,
                fallback=fallback,
            ),
            "shuffled_source": _select_by_syndrome(
                ordered_values=ordered_values,
                source_value=shuffled_teacher_numeric,
                moduli=moduli,
                fallback=fallback,
            ),
            "target_only": fallback,
            "slots_only": _select_slots_only(
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
                fallback=fallback,
            ),
        }
        rows.append(
            {
                "index": index,
                "example_id": example_id,
                "labels": [
                    label
                    for label, ids in target_ids.items()
                    if example_id in ids
                ],
                "gold_answer": gold,
                "teacher_prediction": teacher_numeric,
                "fallback_prediction": fallback,
                "candidate_pool_size": len(ordered_values),
                "candidate_pool_values": list(ordered_values),
                "candidate_pool_contains_gold": gold in set(ordered_values),
                "candidate_labels_for_gold": sorted(set(labels_by_value.get(gold, []))),
                "conditions": {
                    condition: {
                        "prediction": prediction,
                        "correct": prediction == gold,
                        "candidate_labels": sorted(
                            set(labels_by_value.get(str(prediction), []))
                        )
                        if prediction is not None
                        else [],
                    }
                    for condition, prediction in selections.items()
                },
            }
        )

    clean_ids = target_ids["clean_residual_targets"]
    target_self_ids = target_ids["target_self_repair"]
    teacher_only_ids = target_ids["teacher_only"]
    condition_summaries = {
        condition: _summarize_condition(
            rows,
            condition=condition,
            clean_ids=clean_ids,
            target_self_ids=target_self_ids,
            teacher_only_ids=teacher_only_ids,
        )
        for condition in (
            "matched",
            "zero_source",
            "shuffled_source",
            "target_only",
            "slots_only",
        )
    }
    control_clean_union = set().union(
        *[
            set(condition_summaries[condition]["clean_correct_ids"])
            for condition in (
                "zero_source",
                "shuffled_source",
                "target_only",
                "slots_only",
            )
        ]
    )
    matched_clean = set(condition_summaries["matched"]["clean_correct_ids"])
    source_necessary_clean = matched_clean - control_clean_union
    fallback_correct = condition_summaries["target_only"]["correct_count"]
    criteria = {
        "min_correct": condition_summaries["matched"]["correct_count"] >= min_correct,
        "preserve_fallback_floor": (
            condition_summaries["matched"]["correct_count"] >= fallback_correct
        ),
        "min_clean_source_necessary": (
            len(source_necessary_clean) >= min_clean_source_necessary
        ),
        "clean_controls_do_not_explain": len(source_necessary_clean) == len(matched_clean),
    }
    failing = [name for name, passed in criteria.items() if not passed]
    status = (
        "syndrome_sidecar_bound_clears_gate_not_method"
        if not failing
        else "syndrome_sidecar_bound_fails_gate"
    )
    return {
        "moduli": list(moduli),
        "syndrome_bits": _moduli_bits(moduli),
        "syndrome_bytes": int(math.ceil(_moduli_bits(moduli) / 8.0)),
        "status": status,
        "criteria": criteria,
        "failing_criteria": failing,
        "fallback_correct_count": fallback_correct,
        "candidate_pool_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"])) for row in rows
        ),
        "candidate_pool_clean_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"]))
            for row in rows
            if row["example_id"] in clean_ids
        ),
        "condition_summaries": condition_summaries,
        "control_clean_union_ids": sorted(control_clean_union),
        "source_necessary_clean_ids": sorted(source_necessary_clean),
        "rows": rows,
    }


def analyze(
    *,
    target_spec: RowSpec,
    teacher_spec: RowSpec,
    candidate_specs: Sequence[RowSpec],
    target_set_path: pathlib.Path,
    moduli_sets: Sequence[Sequence[int]],
    fallback_label: str,
    shuffle_offset: int,
    min_correct: int,
    min_clean_source_necessary: int,
    min_numeric_coverage: int,
    run_date: str,
) -> dict[str, Any]:
    target_records = _records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("target rows contain duplicate example_id values")
    teacher_records = _subset_reference_order(
        _records_for_method(teacher_spec),
        reference_ids,
    )
    target_by_id = _by_id(target_records)
    teacher_by_id = _by_id(teacher_records)
    target_ids = _load_target_ids(target_set_path)
    if target_ids["teacher_only"]:
        target_correct = {
            str(row["example_id"]) for row in target_records if bool(row.get("correct"))
        }
        teacher_correct = {
            str(row["example_id"]) for row in teacher_records if bool(row.get("correct"))
        }
        if target_ids["teacher_only"] != teacher_correct - target_correct:
            raise ValueError("target_set.ids.teacher_only does not match target/teacher rows")
        if not target_ids["clean_residual_targets"].issubset(target_ids["teacher_only"]):
            raise ValueError("clean_residual_targets must be a subset of teacher_only")

    candidate_by_label: dict[str, dict[str, dict[str, Any]]] = {
        target_spec.label: target_by_id,
    }
    for spec in candidate_specs:
        if spec.label in candidate_by_label:
            raise ValueError(f"Duplicate candidate label {spec.label!r}")
        candidate_by_label[spec.label] = _by_id(
            _subset_reference_order(_records_for_method(spec), reference_ids)
        )
    if fallback_label not in candidate_by_label:
        raise ValueError(
            f"fallback label {fallback_label!r} not in candidates: {sorted(candidate_by_label)}"
        )

    teacher_numeric_coverage = sum(
        int(_prediction_numeric(row) is not None) for row in teacher_records
    )
    candidate_numeric_coverage = {
        label: sum(
            int(_prediction_numeric(records[example_id]) is not None)
            for example_id in reference_ids
        )
        for label, records in candidate_by_label.items()
    }
    provenance_issues: list[str] = []
    if teacher_numeric_coverage < min_numeric_coverage:
        provenance_issues.append(
            f"teacher_numeric_coverage={teacher_numeric_coverage} < {min_numeric_coverage}"
        )
    for label, coverage in candidate_numeric_coverage.items():
        if coverage < min_numeric_coverage:
            provenance_issues.append(
                f"candidate.{label}.numeric_coverage={coverage} < {min_numeric_coverage}"
            )

    runs = [
        _evaluate_moduli(
            moduli=moduli,
            reference_ids=reference_ids,
            target_by_id=target_by_id,
            teacher_by_id=teacher_by_id,
            candidate_by_label=candidate_by_label,
            target_label=target_spec.label,
            fallback_label=fallback_label,
            target_ids=target_ids,
            shuffle_offset=shuffle_offset,
            min_correct=min_correct,
            min_clean_source_necessary=min_clean_source_necessary,
        )
        for moduli in moduli_sets
    ]
    clearing = [run for run in runs if run["status"].endswith("clears_gate_not_method")]
    status = (
        "syndrome_sidecar_bound_clears_gate_not_method"
        if clearing and not provenance_issues
        else "syndrome_sidecar_bound_fails_gate"
    )
    return {
        "date": run_date,
        "status": status,
        "interpretation": (
            "This is a target-candidate oracle/bound probe. It uses C2C-derived "
            "numeric residues as a proxy syndrome and does not prove that a "
            "source latent can predict those residues."
        ),
        "artifacts": {
            "target": {
                "label": target_spec.label,
                "path": _display_path(target_spec.path),
                "method": target_spec.method,
            },
            "teacher": {
                "label": teacher_spec.label,
                "path": _display_path(teacher_spec.path),
                "method": teacher_spec.method,
            },
            "target_set_json": _display_path(target_set_path),
            "candidates": [
                {
                    "label": spec.label,
                    "path": _display_path(spec.path),
                    "method": spec.method,
                }
                for spec in candidate_specs
            ],
        },
        "config": {
            "fallback_label": fallback_label,
            "shuffle_offset": shuffle_offset,
            "min_correct": min_correct,
            "min_clean_source_necessary": min_clean_source_necessary,
            "min_numeric_coverage": min_numeric_coverage,
            "moduli_sets": [list(moduli) for moduli in moduli_sets],
        },
        "reference_n": len(reference_ids),
        "reference_ids": reference_ids,
        "target_ids": {key: sorted(value) for key, value in target_ids.items()},
        "provenance": {
            "exact_ordered_id_parity": True,
            "teacher_numeric_coverage": teacher_numeric_coverage,
            "candidate_numeric_coverage": candidate_numeric_coverage,
            "issues": provenance_issues,
        },
        "runs": runs,
    }


def _parse_moduli_set(raw: str) -> list[int]:
    values = [int(part) for part in raw.replace(";", ",").split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("moduli set cannot be empty")
    if any(value <= 1 for value in values):
        raise argparse.ArgumentTypeError(f"moduli must be >1: {raw!r}")
    return values


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SVAMP32 Syndrome Sidecar Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- fallback label: `{payload['config']['fallback_label']}`",
        f"- teacher numeric coverage: `{payload['provenance']['teacher_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
        "",
        "## Moduli Sweep",
        "",
        "| Moduli | Bytes | Status | Matched | Target-Only | Target-Self Matched | Clean Gold In Pool | Clean Matched | Clean Necessary | Control Clean Union | Source-Necessary IDs |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in payload["runs"]:
        matched = run["condition_summaries"]["matched"]
        target_only = run["condition_summaries"]["target_only"]
        lines.append(
            "| {moduli} | {bytes} | {status} | {matched} | {target_only} | {target_self} | {clean_pool} | {clean} | {necessary} | {control_clean} | {ids} |".format(
                moduli=",".join(str(value) for value in run["moduli"]),
                bytes=run["syndrome_bytes"],
                status=run["status"],
                matched=matched["correct_count"],
                target_only=target_only["correct_count"],
                target_self=matched["target_self_correct_count"],
                clean_pool=run["candidate_pool_clean_gold_count"],
                clean=matched["clean_correct_count"],
                necessary=len(run["source_necessary_clean_ids"]),
                control_clean=len(run["control_clean_union_ids"]),
                ids=", ".join(f"`{value}`" for value in run["source_necessary_clean_ids"]) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=_parse_spec)
    parser.add_argument("--teacher", required=True, type=_parse_spec)
    parser.add_argument("--candidate", action="append", type=_parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target_self_repair")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--moduli-set", action="append", type=_parse_moduli_set)
    parser.add_argument("--min-correct", type=int, default=14)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    moduli_sets = args.moduli_set or [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [97],
    ]
    payload = analyze(
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=args.candidate,
        target_set_path=_resolve(args.target_set_json),
        moduli_sets=moduli_sets,
        fallback_label=args.fallback_label,
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_md, payload)
    print(json.dumps({"status": payload["status"], "output_json": _display_path(output_json)}, indent=2))
    return payload


if __name__ == "__main__":
    main()
