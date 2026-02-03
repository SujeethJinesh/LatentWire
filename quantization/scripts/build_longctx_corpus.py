#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build a local long-context padding corpus (JSONL).")
    parser.add_argument("--output", default="quantization/prompts/longctx_corpus.jsonl")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="text")
    parser.add_argument("--max-chunks", type=int, default=5000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    try:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
        for row in dataset:
            text = row.get(args.field, "")
            if not text or len(text) < args.min_chars:
                continue
            chunks.append(text.strip())
            if len(chunks) >= args.max_chunks:
                break
    except Exception:
        base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
        chunks = [base.strip() for _ in range(args.max_chunks)]

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, text in enumerate(chunks):
            record = {"id": idx, "text": text}
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
