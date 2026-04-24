#!/usr/bin/env python3
"""Materialize generation examples selected by stable example IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.evaluate import _generation_example_id, load_generation


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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ids_from_target_set(path: pathlib.Path, fields: Sequence[str]) -> list[str]:
    payload = _read_json(path)
    ids = payload.get("ids", {})
    values: list[str] = []
    for field in fields:
        values.extend(str(value) for value in ids.get(field, []))
    return _ordered_unique(values)


def materialize_subset(
    *,
    eval_path: pathlib.Path,
    selected_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_rows = _read_jsonl(eval_path)
    examples = load_generation(str(eval_path))
    if len(raw_rows) != len(examples):
        raise ValueError("raw JSONL row count does not match parsed generation examples")
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row, example in zip(raw_rows, examples):
        example_id = _generation_example_id(example)
        if example_id in by_id:
            duplicates.add(example_id)
        by_id[example_id] = row
    if duplicates:
        raise ValueError(f"eval file has duplicate stable IDs: {sorted(duplicates)}")
    missing = [example_id for example_id in selected_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"selected IDs missing from eval file: {missing}")
    return [dict(by_id[example_id]) for example_id in selected_ids], list(selected_ids)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--id-fields", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-meta-json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    eval_path = _resolve(args.eval_file)
    target_set_path = _resolve(args.target_set_json)
    output_path = _resolve(args.output_jsonl)
    output_meta_path = (
        _resolve(args.output_meta_json)
        if args.output_meta_json
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )
    selected_ids = _ids_from_target_set(target_set_path, args.id_fields)
    rows, ordered_ids = materialize_subset(eval_path=eval_path, selected_ids=selected_ids)
    _write_jsonl(output_path, rows)
    meta = {
        "date": str(date.today()),
        "eval_file": _display_path(eval_path),
        "eval_file_sha256": _sha256(eval_path),
        "target_set_json": _display_path(target_set_path),
        "target_set_sha256": _sha256(target_set_path),
        "id_fields": list(args.id_fields),
        "selected_ids": ordered_ids,
        "output_jsonl": _display_path(output_path),
        "output_sha256": _sha256(output_path),
        "output_n": len(rows),
    }
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


if __name__ == "__main__":
    main()
