#!/usr/bin/env python3
"""Deterministically split a JSONL dataset into search/eval subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--search-output", required=True, help="Output JSONL for held-out search/tuning")
    parser.add_argument("--eval-output", required=True, help="Output JSONL for final evaluation")
    parser.add_argument(
        "--search-count",
        type=int,
        required=True,
        help="Number of examples to place in the held-out search split",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    search_output = Path(args.search_output)
    eval_output = Path(args.eval_output)

    records = load_jsonl(input_path)
    if args.search_count <= 0 or args.search_count >= len(records):
        raise ValueError(
            f"--search-count must be between 1 and {len(records) - 1}, got {args.search_count}"
        )

    rng = random.Random(args.seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    search_indices = set(indices[: args.search_count])
    search_records = [records[idx] for idx in range(len(records)) if idx in search_indices]
    eval_records = [records[idx] for idx in range(len(records)) if idx not in search_indices]

    write_jsonl(search_output, search_records)
    write_jsonl(eval_output, eval_records)

    print(
        f"Wrote {len(search_records)} search examples to {search_output} and "
        f"{len(eval_records)} eval examples to {eval_output}"
    )


if __name__ == "__main__":
    main()
