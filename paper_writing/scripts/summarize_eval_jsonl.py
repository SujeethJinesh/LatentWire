#!/usr/bin/env python3
"""Summarize Bridged Eval JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def summarize_file(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    total = len(rows)
    if not total:
        print(f"{path} -> empty")
        return
    invalid = sum(
        1 for r in rows if r.get("bridged_extracted") in (None, "", "[invalid]")
    )
    gold_match = sum(
        1
        for r in rows
        if r.get("bridged_extracted") not in (None, "", "[invalid]")
        and r.get("bridged_extracted") == r.get("gold_extracted")
    )
    print(f"{path} -> total={total} invalid={invalid} gold_match={gold_match}")
    for candidate in rows:
        extracted = candidate.get("bridged_extracted")
        if extracted not in (None, "", "[invalid]"):
            snippet = candidate.get("bridged_full", "").split("\n")
            print("  sample:", " / ".join(snippet[:3]))
            break


def iter_jsonl(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.glob("eval_samples_step_*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval JSONL files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories.")
    args = parser.parse_args()
    for jsonl_path in iter_jsonl(args.paths):
        summarize_file(jsonl_path)


if __name__ == "__main__":
    main()
