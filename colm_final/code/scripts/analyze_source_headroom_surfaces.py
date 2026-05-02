#!/usr/bin/env python3
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
class SurfaceSpec:
    label: str
    target_path: pathlib.Path
    target_method: str
    source_path: pathlib.Path
    source_method: str
    eval_file: pathlib.Path | None = None
    note: str = ""


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_surface_spec(spec: str) -> SurfaceSpec:
    if "=" not in spec:
        raise ValueError(f"Expected label=key=value,... spec, got {spec!r}")
    label, raw_fields = spec.split("=", 1)
    fields: dict[str, str] = {}
    for item in raw_fields.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected key=value in surface spec {spec!r}; got {item!r}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()

    path = fields.get("path")
    target_path = fields.get("target_path") or path
    source_path = fields.get("source_path") or path
    if not label or not target_path or not source_path:
        raise ValueError(f"Surface spec needs label plus path/target_path/source_path: {spec!r}")
    return SurfaceSpec(
        label=label,
        target_path=_resolve(target_path),
        target_method=fields.get("target_method", "target_alone"),
        source_path=_resolve(source_path),
        source_method=fields.get("source_method", "source_alone"),
        eval_file=_resolve(fields["eval_file"]) if fields.get("eval_file") else None,
        note=fields.get("note", ""),
    )


def _records_for_method(path: pathlib.Path, method: str) -> list[dict[str, Any]]:
    grouped = harness.group_by_method(_read_jsonl(path))
    if method in grouped:
        return [dict(row) for row in grouped[method]]
    if len(grouped) == 1:
        return [dict(row) for row in next(iter(grouped.values()))]
    raise KeyError(f"Method {method!r} not found in {path}: {sorted(grouped)}")


def _example_ids_from_eval(eval_file: pathlib.Path, count: int) -> list[str]:
    examples = harness.load_generation(str(eval_file))
    if len(examples) < count:
        raise ValueError(f"Eval file {eval_file} has {len(examples)} examples, need {count}")
    return [str(example["example_id"]) for example in examples[:count]]


def _ensure_ids(
    records: list[dict[str, Any]],
    *,
    eval_file: pathlib.Path | None,
) -> tuple[list[dict[str, Any]], str]:
    if all(record.get("example_id") is not None for record in records):
        return records, "example_id"
    patched = [dict(row) for row in records]
    if eval_file is not None:
        ids = _example_ids_from_eval(eval_file, len(records))
        for row, example_id in zip(patched, ids, strict=True):
            row["example_id"] = example_id
        return patched, "eval_file"
    for idx, row in enumerate(patched):
        row["example_id"] = f"index:{row.get('index', idx)}"
    return patched, "index_fallback"


def _correct_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(row["example_id"]) for row in records if bool(row.get("correct"))}


def _ordered_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(row["example_id"]) for row in records]


def _numeric_coverage(records: list[dict[str, Any]]) -> int:
    return int(
        sum(int(harness._has_numeric_extraction(str(row.get("prediction", "")))) for row in records)
    )


def _row_summary(label: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = len(_correct_ids(records))
    total = len(records)
    return {
        "label": label,
        "n": total,
        "correct": correct,
        "accuracy": float(correct / max(total, 1)),
        "empty_predictions": int(sum(int(not str(row.get("prediction", "")).strip()) for row in records)),
        "numeric_extraction_coverage": _numeric_coverage(records),
    }


def _surface_status(
    *,
    exact_id_parity: bool,
    strict_ids: bool,
    source_only_count: int,
    oracle_gain_count: int,
    min_source_only: int,
) -> str:
    if not exact_id_parity:
        return "invalid_id_mismatch"
    if not strict_ids:
        return "weak_index_only_surface"
    if source_only_count >= min_source_only and oracle_gain_count >= min_source_only:
        return "strong_source_complementary_surface"
    if source_only_count > 0:
        return "weak_source_complementary_surface"
    return "no_source_complementary_headroom"


def analyze_surface(spec: SurfaceSpec, *, min_source_only: int) -> dict[str, Any]:
    target_records, target_id_source = _ensure_ids(
        _records_for_method(spec.target_path, spec.target_method),
        eval_file=spec.eval_file,
    )
    source_records, source_id_source = _ensure_ids(
        _records_for_method(spec.source_path, spec.source_method),
        eval_file=spec.eval_file,
    )
    target_ids = _ordered_ids(target_records)
    source_ids = _ordered_ids(source_records)
    target_correct = _correct_ids(target_records)
    source_correct = _correct_ids(source_records)
    source_only = source_correct - target_correct
    target_only = target_correct - source_correct
    both_correct = source_correct & target_correct
    oracle = source_correct | target_correct
    strict_ids = target_id_source != "index_fallback" and source_id_source != "index_fallback"
    exact_id_parity = target_ids == source_ids
    oracle_gain = len(oracle) - len(target_correct)
    status = _surface_status(
        exact_id_parity=exact_id_parity,
        strict_ids=strict_ids,
        source_only_count=len(source_only),
        oracle_gain_count=oracle_gain,
        min_source_only=min_source_only,
    )
    return {
        "label": spec.label,
        "status": status,
        "note": spec.note,
        "artifacts": {
            "target_path": str(spec.target_path.relative_to(ROOT) if spec.target_path.is_relative_to(ROOT) else spec.target_path),
            "source_path": str(spec.source_path.relative_to(ROOT) if spec.source_path.is_relative_to(ROOT) else spec.source_path),
            "eval_file": (
                str(spec.eval_file.relative_to(ROOT) if spec.eval_file and spec.eval_file.is_relative_to(ROOT) else spec.eval_file)
                if spec.eval_file
                else None
            ),
        },
        "methods": {
            "target_method": spec.target_method,
            "source_method": spec.source_method,
        },
        "id_sources": {
            "target": target_id_source,
            "source": source_id_source,
            "strict_ids": strict_ids,
            "exact_ordered_id_parity": exact_id_parity,
            "set_id_parity": set(target_ids) == set(source_ids),
        },
        "target": _row_summary(spec.target_method, target_records),
        "source": _row_summary(spec.source_method, source_records),
        "overlap": {
            "both_correct_count": len(both_correct),
            "both_correct_ids": sorted(both_correct),
            "source_only_count": len(source_only),
            "source_only_ids": sorted(source_only),
            "target_only_count": len(target_only),
            "target_only_ids": sorted(target_only),
            "both_wrong_count": max(len(target_records) - len(oracle), 0),
            "target_or_source_oracle_count": len(oracle),
            "oracle_gain_vs_target_count": oracle_gain,
            "oracle_accuracy": float(len(oracle) / max(len(target_records), 1)),
        },
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Headroom Surface Scan",
        "",
        f"- date: `{payload['date']}`",
        f"- min source-only threshold: `{payload['min_source_only']}`",
        "",
        "| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["surfaces"]:
        target = row["target"]
        source = row["source"]
        overlap = row["overlap"]
        id_source = f"{row['id_sources']['target']}/{row['id_sources']['source']}"
        lines.append(
            f"| {row['label']} | `{row['status']}` | "
            f"{target['correct']}/{target['n']} | {source['correct']}/{source['n']} | "
            f"{overlap['source_only_count']} | {overlap['target_or_source_oracle_count']}/{target['n']} | "
            f"{id_source} | {row.get('note', '')} |"
        )
    lines.extend(["", "## Ranked Decision", ""])
    for row in payload["ranked_surfaces"][:5]:
        overlap = row["overlap"]
        lines.append(
            f"- `{row['label']}`: status=`{row['status']}`, "
            f"source_only=`{overlap['source_only_count']}`, "
            f"oracle=`{overlap['target_or_source_oracle_count']}/{row['target']['n']}`, "
            f"strict_ids=`{row['id_sources']['strict_ids']}`"
        )
    lines.extend(["", "## Artifact Paths", ""])
    for row in payload["surfaces"]:
        lines.append(
            f"- `{row['label']}` target `{row['artifacts']['target_path']}` "
            f"({row['methods']['target_method']}); source `{row['artifacts']['source_path']}` "
            f"({row['methods']['source_method']})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    surfaces = [
        analyze_surface(_parse_surface_spec(spec), min_source_only=int(args.min_source_only))
        for spec in args.surface
    ]
    ranked = sorted(
        surfaces,
        key=lambda row: (
            row["status"] == "strong_source_complementary_surface",
            bool(row["id_sources"]["strict_ids"]),
            int(row["overlap"]["source_only_count"]),
            int(row["overlap"]["oracle_gain_vs_target_count"]),
            float(row["source"]["accuracy"] - row["target"]["accuracy"]),
        ),
        reverse=True,
    )
    payload = {
        "date": str(date.today()),
        "min_source_only": int(args.min_source_only),
        "surfaces": surfaces,
        "ranked_surfaces": ranked,
        "recommended_next_gate": (
            "run learned connector on highest-ranked strong surface"
            if ranked and ranked[0]["status"] == "strong_source_complementary_surface"
            else "materialize a stricter stronger-source surface before training"
        ),
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
        description="Rank frozen target/source surfaces by target-complementary source headroom."
    )
    parser.add_argument(
        "--surface",
        action="append",
        default=[],
        help=(
            "Repeatable label=path=...,target_method=...,source_method=...,eval_file=... "
            "or label=target_path=...,source_path=...,target_method=...,source_method=..."
        ),
    )
    parser.add_argument("--min-source-only", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    if not args.surface:
        parser.error("at least one --surface is required")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run(_parse_args(argv))


if __name__ == "__main__":
    main()
