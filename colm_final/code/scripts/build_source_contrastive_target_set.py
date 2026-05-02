#!/usr/bin/env python3
"""Build exact-ID source-only target sets for contrastive communication gates."""

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


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_spec(raw: str) -> RowSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected label=path=...,method=... spec: {raw!r}")
    label, rest = raw.split("=", 1)
    fields: dict[str, str] = {}
    for item in rest.split(","):
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value in spec {raw!r}: {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    if not label or not fields.get("path") or not fields.get("method"):
        raise argparse.ArgumentTypeError(f"Spec needs label, path, and method: {raw!r}")
    return RowSpec(label=label, path=_resolve(fields["path"]), method=fields["method"])


def _records_for_method(spec: RowSpec) -> list[dict[str, Any]]:
    raw_records = _read_jsonl(spec.path)
    raw_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_records:
        raw_grouped.setdefault(str(row.get("method")), []).append(row)
    if spec.method in raw_grouped:
        return [dict(row) for row in raw_grouped[spec.method]]
    grouped = harness.group_by_method(raw_records)
    if spec.method not in grouped:
        raise KeyError(
            f"Method {spec.method!r} not found in {spec.path}; "
            f"raw_available={sorted(raw_grouped)}, normalized_available={sorted(grouped)}"
        )
    return [dict(row) for row in grouped[spec.method]]


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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


def _ordered_subset(records: list[dict[str, Any]], reference_ids: list[str]) -> list[dict[str, Any]]:
    by_id = _by_id(records)
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Missing reference IDs: {missing}")
    return [by_id[example_id] for example_id in reference_ids]


def _correct_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in records if bool(row.get("correct"))}


def _numeric_coverage(records: list[dict[str, Any]]) -> int:
    return int(
        sum(
            int(harness._has_numeric_extraction(str(row.get("prediction", ""))))
            for row in records
        )
    )


def _summary(label: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "n": len(records),
        "correct": len(_correct_ids(records)),
        "numeric_coverage": _numeric_coverage(records),
    }


def build_target_set(
    *,
    target_spec: RowSpec,
    source_spec: RowSpec,
    control_specs: list[RowSpec],
    baseline_specs: list[RowSpec],
    min_source_only: int,
    run_date: str,
) -> dict[str, Any]:
    target_records = _records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("target rows contain duplicate example_id values")

    source_records = _ordered_subset(_records_for_method(source_spec), reference_ids)
    controls = {
        spec.label: _ordered_subset(_records_for_method(spec), reference_ids)
        for spec in control_specs
    }
    baselines = {
        spec.label: _ordered_subset(_records_for_method(spec), reference_ids)
        for spec in baseline_specs
    }

    target_correct = _correct_ids(target_records)
    source_correct = _correct_ids(source_records)
    control_correct = {
        label: _correct_ids(records) for label, records in controls.items()
    }
    baseline_correct = {
        label: _correct_ids(records) for label, records in baselines.items()
    }
    control_union = set().union(*control_correct.values()) if control_correct else set()
    baseline_union = set().union(*baseline_correct.values()) if baseline_correct else set()
    source_only = source_correct - target_correct
    clean_source_only = source_only - control_union - baseline_union
    target_only = target_correct - source_correct
    oracle = target_correct | source_correct

    rows: list[dict[str, Any]] = []
    for example_id in reference_ids:
        labels: list[str] = []
        if example_id in source_only:
            labels.append("source_only")
        if example_id in clean_source_only:
            labels.append("clean_source_only")
        if example_id in target_only:
            labels.append("target_only")
        for label, ids in control_correct.items():
            if example_id in ids:
                labels.append(f"control:{label}")
        for label, ids in baseline_correct.items():
            if example_id in ids:
                labels.append(f"baseline:{label}")
        rows.append({"example_id": example_id, "labels": labels})

    status = (
        "source_contrastive_target_set_ready"
        if len(clean_source_only) >= int(min_source_only)
        else "insufficient_clean_source_only_ids"
    )
    return {
        "date": run_date,
        "status": status,
        "config": {
            "min_source_only": int(min_source_only),
        },
        "artifacts": {
            "target": {
                "label": target_spec.label,
                "path": _display_path(target_spec.path),
                "method": target_spec.method,
            },
            "source": {
                "label": source_spec.label,
                "path": _display_path(source_spec.path),
                "method": source_spec.method,
            },
            "controls": [
                {"label": spec.label, "path": _display_path(spec.path), "method": spec.method}
                for spec in control_specs
            ],
            "baselines": [
                {"label": spec.label, "path": _display_path(spec.path), "method": spec.method}
                for spec in baseline_specs
            ],
        },
        "reference_n": len(reference_ids),
        "reference_ids": reference_ids,
        "provenance": {
            "exact_ordered_id_parity": True,
            "target_numeric_coverage": _numeric_coverage(target_records),
            "source_numeric_coverage": _numeric_coverage(source_records),
            "control_numeric_coverage": {
                label: _numeric_coverage(records) for label, records in controls.items()
            },
            "baseline_numeric_coverage": {
                label: _numeric_coverage(records) for label, records in baselines.items()
            },
        },
        "summaries": {
            "target": _summary(target_spec.label, target_records),
            "source": _summary(source_spec.label, source_records),
            "controls": {
                label: _summary(label, records) for label, records in controls.items()
            },
            "baselines": {
                label: _summary(label, records) for label, records in baselines.items()
            },
        },
        "ids": {
            "source_only": sorted(source_only),
            "clean_source_only": sorted(clean_source_only),
            "clean_residual_targets": sorted(clean_source_only),
            "target_only": sorted(target_only),
            "control_union": sorted(control_union),
            "baseline_union": sorted(baseline_union),
            "target_self_repair": sorted(baseline_union),
        },
        "counts": {
            "target_correct": len(target_correct),
            "source_correct": len(source_correct),
            "source_only": len(source_only),
            "clean_source_only": len(clean_source_only),
            "target_only": len(target_only),
            "target_or_source_oracle": len(oracle),
        },
        "rows": rows,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    counts = payload["counts"]
    lines = [
        "# Source-Contrastive Target Set",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- target correct: `{counts['target_correct']}/{payload['reference_n']}`",
        f"- source correct: `{counts['source_correct']}/{payload['reference_n']}`",
        f"- source-only IDs: `{counts['source_only']}`",
        f"- clean source-only IDs: `{counts['clean_source_only']}`",
        f"- target-or-source oracle: `{counts['target_or_source_oracle']}/{payload['reference_n']}`",
        "",
        "## ID Sets",
        "",
        f"- clean source-only: {', '.join(f'`{item}`' for item in payload['ids']['clean_source_only']) or 'none'}",
        f"- source-only: {', '.join(f'`{item}`' for item in payload['ids']['source_only']) or 'none'}",
        f"- excluded by controls/baselines: {', '.join(f'`{item}`' for item in sorted(set(payload['ids']['control_union']) | set(payload['ids']['baseline_union']))) or 'none'}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=_parse_spec)
    parser.add_argument("--source", required=True, type=_parse_spec)
    parser.add_argument("--control", action="append", type=_parse_spec, default=[])
    parser.add_argument("--baseline", action="append", type=_parse_spec, default=[])
    parser.add_argument("--min-source-only", type=int, default=5)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = build_target_set(
        target_spec=args.target,
        source_spec=args.source,
        control_specs=list(args.control),
        baseline_specs=list(args.baseline),
        min_source_only=int(args.min_source_only),
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
