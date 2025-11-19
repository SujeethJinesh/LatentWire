#!/usr/bin/env python3
"""Summarize Bridged Eval JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def is_valid_answer(answer) -> bool:
    """Check if an extracted answer is valid (not None, empty, or [invalid])."""
    return answer not in (None, "", "[invalid]")


def summarize_file(path: Path) -> None:
    """Summarize a single JSONL file with robust error handling."""
    rows = []
    errors = 0

    # Stream file reading instead of loading entire file into memory
    try:
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: {path}:{line_num} invalid JSON: {e}", file=sys.stderr)
                    errors += 1
    except IOError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return

    total = len(rows)
    if not total:
        print(f"{path} -> empty" + (f" ({errors} JSON errors)" if errors else ""))
        return

    invalid = sum(1 for r in rows if not is_valid_answer(r.get("bridged_extracted")))
    gold_match = sum(
        1
        for r in rows
        if is_valid_answer(r.get("bridged_extracted"))
        and r.get("bridged_extracted") == r.get("gold_extracted")
    )

    print(f"{path} -> total={total} invalid={invalid} gold_match={gold_match}" +
          (f" ({errors} JSON errors skipped)" if errors else ""))

    # Show first valid sample (truncate long output)
    for candidate in rows:
        extracted = candidate.get("bridged_extracted")
        if is_valid_answer(extracted):
            snippet = candidate.get("bridged_full", "").split("\n")
            sample_text = " / ".join(snippet[:3])
            # Truncate very long samples
            if len(sample_text) > 200:
                sample_text = sample_text[:197] + "..."
            print("  sample:", sample_text)
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
