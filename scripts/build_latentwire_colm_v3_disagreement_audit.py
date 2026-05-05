from __future__ import annotations

"""Build the COLM_v3 source/target/packet disagreement audit.

The audit is deliberately small: it uses the already-frozen acceptance-baseline
prediction files and reports whether packet gains are mostly source-choice
transport or actual repairs beyond source choice.
"""

import argparse
import csv
import gzip
import json
import pathlib
from collections import defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/source_private_colm_acceptance_baselines_20260502/disagreement_audit"
DEFAULT_INPUTS = (
    (
        "ARC-Challenge",
        ROOT / "results/source_private_colm_acceptance_baselines_20260502/arc_test_predictions.jsonl.gz",
    ),
    (
        "OpenBookQA",
        ROOT / "results/source_private_colm_acceptance_baselines_20260502/openbookqa_test_predictions.jsonl.gz",
    ),
)


def _open_jsonl(path: pathlib.Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_rows(path: pathlib.Path) -> dict[tuple[int, str], dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = defaultdict(dict)
    for row in _open_jsonl(path):
        condition = row.get("condition")
        if condition not in {"target_only", "matched_source_private_packet", "source_index_byte"}:
            continue
        key = (int(row["seed"]), str(row["content_id"]))
        grouped[key][condition] = row
    return grouped


def _summarize(benchmark: str, path: pathlib.Path) -> dict[str, Any]:
    grouped = _load_rows(path)
    usable = [rows for rows in grouped.values() if {"target_only", "matched_source_private_packet", "source_index_byte"} <= rows.keys()]
    if not usable:
        raise ValueError(f"no complete rows found in {path}")

    counts = defaultdict(int)
    for rows in usable:
        target = rows["target_only"]
        packet = rows["matched_source_private_packet"]
        source = rows["source_index_byte"]
        target_correct = bool(target["correct"])
        source_correct = bool(source["correct"])
        packet_correct = bool(packet["correct"])
        counts["n"] += 1
        counts["target_correct"] += int(target_correct)
        counts["source_correct"] += int(source_correct)
        counts["packet_correct"] += int(packet_correct)
        counts["both_target_source_correct"] += int(target_correct and source_correct)
        counts["target_only_correct"] += int(target_correct and not source_correct)
        counts["source_only_correct"] += int(source_correct and not target_correct)
        counts["both_wrong"] += int((not target_correct) and (not source_correct))
        counts["packet_follows_source"] += int(int(packet["prediction_index"]) == int(source["prediction_index"]))
        counts["packet_repairs_target"] += int((not target_correct) and packet_correct)
        counts["packet_damages_target"] += int(target_correct and not packet_correct)
        counts["packet_repairs_source"] += int((not source_correct) and packet_correct)
        counts["packet_damages_source"] += int(source_correct and not packet_correct)
        counts["source_correct_target_wrong_packet_follows_source"] += int(
            source_correct
            and not target_correct
            and int(packet["prediction_index"]) == int(source["prediction_index"])
        )
        counts["target_correct_source_wrong_packet_follows_source"] += int(
            target_correct
            and not source_correct
            and int(packet["prediction_index"]) == int(source["prediction_index"])
        )

    n = counts["n"]

    def rate(key: str) -> float:
        return counts[key] / n

    return {
        "benchmark": benchmark,
        "prediction_path": str(path.relative_to(ROOT)),
        "n_seed_items": n,
        "target_accuracy": rate("target_correct"),
        "source_index_accuracy": rate("source_correct"),
        "packet_accuracy": rate("packet_correct"),
        "both_target_source_correct": rate("both_target_source_correct"),
        "target_only_correct": rate("target_only_correct"),
        "source_only_correct": rate("source_only_correct"),
        "both_wrong": rate("both_wrong"),
        "packet_follows_source": rate("packet_follows_source"),
        "packet_repairs_target": rate("packet_repairs_target"),
        "packet_damages_target": rate("packet_damages_target"),
        "packet_repairs_source": rate("packet_repairs_source"),
        "packet_damages_source": rate("packet_damages_source"),
        "source_correct_target_wrong_packet_follows_source": rate(
            "source_correct_target_wrong_packet_follows_source"
        ),
        "target_correct_source_wrong_packet_follows_source": rate(
            "target_correct_source_wrong_packet_follows_source"
        ),
    }


def _write_md(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# COLM_v3 Disagreement Audit",
        "",
        "This audit uses the frozen acceptance-baseline predictions and asks where the packet changes the target decision.",
        "",
        "| Benchmark | seed-items | target | source-index | packet | source-only correct | target-only correct | packet follows source | repairs target | damages target | repairs source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {n_seed_items} | {target_accuracy:.3f} | {source_index_accuracy:.3f} | "
            "{packet_accuracy:.3f} | {source_only_correct:.3f} | {target_only_correct:.3f} | "
            "{packet_follows_source:.3f} | {packet_repairs_target:.3f} | {packet_damages_target:.3f} | "
            "{packet_repairs_source:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation: the packet mostly follows the source-selected candidate. It repairs target errors primarily when the source is correct, and it can damage target-correct/source-wrong rows. This supports the paper's source-choice boundary rather than a stronger evidence-synthesis claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [_summarize(name, path) for name, path in DEFAULT_INPUTS]

    (prefix.with_suffix(".json")).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_md(prefix.with_suffix(".md"), rows)
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
