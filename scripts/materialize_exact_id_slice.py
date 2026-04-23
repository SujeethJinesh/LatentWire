#!/usr/bin/env python3
"""Materialize a JSONL artifact exactly ordered by a reference ID slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import harness_common as harness


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(row["example_id"]) for row in records]


def _check_unique(ids: list[str], *, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example_id in ids:
        if example_id in seen:
            duplicates.add(example_id)
        seen.add(example_id)
    if duplicates:
        raise ValueError(f"{label} has duplicate example_id values: {sorted(duplicates)}")


def materialize_exact_id_slice(
    *,
    reference_records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_ids = _ordered_ids(reference_records)
    source_ids = _ordered_ids(source_records)
    _check_unique(reference_ids, label="reference")
    _check_unique(source_ids, label="source")
    by_id = {str(row["example_id"]): row for row in source_records}
    missing = [example_id for example_id in reference_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"Source artifact is missing reference IDs: {missing}")
    return [dict(by_id[example_id]) for example_id in reference_ids]


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--source-method", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-meta-json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    reference_path = _resolve(args.reference_jsonl)
    source_path = _resolve(args.source_jsonl)
    output_path = _resolve(args.output_jsonl)
    output_meta_path = (
        _resolve(args.output_meta_json)
        if args.output_meta_json
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )
    reference_records = harness.read_jsonl(reference_path)
    grouped = harness.group_by_method(harness.read_jsonl(source_path))
    if args.source_method not in grouped:
        raise KeyError(f"Method {args.source_method!r} not found in {source_path}: {sorted(grouped)}")
    source_records = grouped[args.source_method]
    output_records = materialize_exact_id_slice(
        reference_records=reference_records,
        source_records=source_records,
    )
    write_jsonl(output_path, output_records)
    reference_ids = _ordered_ids(reference_records)
    output_ids = _ordered_ids(output_records)
    meta = {
        "date": str(date.today()),
        "reference_jsonl": _display_path(reference_path),
        "reference_sha256": _sha256(reference_path),
        "source_jsonl": _display_path(source_path),
        "source_sha256": _sha256(source_path),
        "source_method": args.source_method,
        "source_artifact_n": len(source_records),
        "output_jsonl": _display_path(output_path),
        "output_sha256": _sha256(output_path),
        "output_n": len(output_records),
        "reference_n": len(reference_records),
        "dropped_source_rows": len(source_records) - len(output_records),
        "exact_ordered_id_parity": output_ids == reference_ids,
    }
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


if __name__ == "__main__":
    main()
