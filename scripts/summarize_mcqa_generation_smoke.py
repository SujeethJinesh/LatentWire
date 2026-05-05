from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


LETTER_RE = re.compile(r"\b([A-Z])\b|^\s*[\(\[]?([A-Z])[\)\].:]?", re.IGNORECASE)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_answer_letter(text: str) -> str | None:
    match = LETTER_RE.search(text.strip())
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pred_path = pathlib.Path(args.predictions)
    rows = _read_jsonl(pred_path)
    scored = []
    for row in rows:
        gold = str(row["answer"][0]).strip().upper()[:1]
        pred = _first_answer_letter(str(row.get("prediction", "")))
        scored.append(
            {
                "index": row.get("index"),
                "example_id": row.get("example_id"),
                "gold": gold,
                "parsed_prediction": pred,
                "raw_prediction": row.get("prediction", ""),
                "letter_correct": pred == gold,
                "generated_tokens": row.get("generated_tokens"),
                "latency_sec": row.get("latency_sec"),
            }
        )
    count = len(scored)
    payload = {
        "prediction_file": str(pred_path),
        "n": count,
        "letter_accuracy": sum(int(row["letter_correct"]) for row in scored) / max(count, 1),
        "unparsed": sum(int(row["parsed_prediction"] is None) for row in scored),
        "rows": scored,
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md = out_path.with_suffix(".md")
    lines = [
        "# MCQA Generation Smoke Summary",
        "",
        f"- prediction file: `{pred_path}`",
        f"- rows: `{payload['n']}`",
        f"- parsed letter accuracy: `{payload['letter_accuracy']:.3f}`",
        f"- unparsed rows: `{payload['unparsed']}`",
        "",
        "| idx | gold | parsed | correct | raw prediction |",
        "|---:|---|---|:---:|---|",
    ]
    for row in scored:
        raw = str(row["raw_prediction"]).replace("|", "\\|")
        lines.append(
            f"| {row['index']} | {row['gold']} | {row['parsed_prediction']} | "
            f"{'yes' if row['letter_correct'] else 'no'} | `{raw}` |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("n", "letter_accuracy", "unparsed")}, indent=2))


if __name__ == "__main__":
    main()
