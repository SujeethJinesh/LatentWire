#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_range(
    path: pathlib.Path,
    *,
    start_index: int,
    count: int,
) -> list[dict[str, Any]]:
    if start_index < 1:
        raise ValueError("--start-index is 1-based and must be >= 1")
    if count <= 0:
        raise ValueError("--count must be positive")

    rows: list[dict[str, Any]] = []
    stop_index = start_index + count - 1
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if line_index < start_index:
                continue
            if line_index > stop_index:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != count:
        raise ValueError(
            f"Requested rows {start_index}..{stop_index} from {path}, "
            f"but only materialized {len(rows)} rows"
        )
    return rows


def _metadata_id(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("id") is not None:
        return str(metadata["id"])
    return None


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# JSONL Range Materialization",
        "",
        f"- date: `{payload['date']}`",
        f"- source: `{payload['source_path']}`",
        f"- output: `{payload['output_path']}`",
        f"- start index: `{payload['start_index']}`",
        f"- count: `{payload['count']}`",
        f"- source sha256: `{payload['source_sha256']}`",
        f"- output sha256: `{payload['output_sha256']}`",
        "",
        "## IDs",
        "",
        f"- first metadata id: `{payload['first_metadata_id']}`",
        f"- last metadata id: `{payload['last_metadata_id']}`",
        "",
        "## Command",
        "",
        "```bash",
        payload["command"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source = _resolve(args.source)
    output = _resolve(args.output)
    manifest_json = _resolve(args.manifest_json) if args.manifest_json else output.with_suffix(output.suffix + ".manifest.json")
    manifest_md = _resolve(args.manifest_md) if args.manifest_md else output.with_suffix(output.suffix + ".manifest.md")

    rows = _read_jsonl_range(source, start_index=args.start_index, count=args.count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    payload = {
        "date": args.run_date,
        "source_path": _display_path(source),
        "output_path": _display_path(output),
        "start_index": int(args.start_index),
        "count": int(args.count),
        "first_metadata_id": _metadata_id(rows[0]),
        "last_metadata_id": _metadata_id(rows[-1]),
        "metadata_ids": [_metadata_id(row) for row in rows],
        "source_sha256": _sha256_file(source),
        "output_sha256": _sha256_file(output),
        "manifest_json": _display_path(manifest_json),
        "manifest_md": _display_path(manifest_md),
        "command": " ".join(args.command_argv),
    }
    _write_json(manifest_json, payload)
    _write_markdown(manifest_md, payload)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a 1-based contiguous JSONL row range.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--manifest-json")
    parser.add_argument("--manifest-md")
    parser.add_argument("--run-date", default=str(date.today()))
    args = parser.parse_args(argv)
    command_argv = ["./venv_arm64/bin/python", "scripts/materialize_jsonl_range.py"]
    for key, value in vars(args).items():
        if key == "command_argv" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        command_argv.extend([flag, str(value)])
    args.command_argv = command_argv
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return materialize(_parse_args(argv))


if __name__ == "__main__":
    main()
