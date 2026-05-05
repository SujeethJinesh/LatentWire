from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def _choice_labels(row: dict[str, Any]) -> list[str]:
    labels = row.get("choice_labels")
    if labels:
        return [str(label) for label in labels]
    choices = row.get("choices") or []
    return [chr(ord("A") + idx) for idx in range(len(choices))]


def _answer_index(row: dict[str, Any]) -> int:
    if "answer_index" in row:
        return int(row["answer_index"])
    if "answer" in row:
        return int(row["answer"])
    raise KeyError("row must contain answer_index or answer")


def _prompt(row: dict[str, Any]) -> str:
    labels = _choice_labels(row)
    choices = row["choices"]
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    return (
        "Answer the multiple-choice question. Use only the answer letter.\n\n"
        f"Question: {row['question']}\n"
        f"Choices:\n{choice_lines}\n"
        "Answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer_index = _answer_index(row)
        labels = _choice_labels(row)
        answer_label = labels[answer_index]
        rows.append(
            {
                "prompt": _prompt(row),
                "answer_text": answer_label,
                "aliases": [answer_label, f"{answer_label}.", f"({answer_label})"],
                "source_question": row["question"],
                "row_id": row.get("id") or row.get("row_id") or row.get("content_id"),
                "content_id": row.get("content_id"),
            }
        )
        if args.limit is not None and len(rows) >= int(args.limit):
            break
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
