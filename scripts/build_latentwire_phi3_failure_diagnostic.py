from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_agreement(path: pathlib.Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["split"], row["row_id"], row["content_id"])
            rows[key] = {
                "split": row["split"],
                "row_id": row["row_id"],
                "content_id": row["content_id"],
                "phi3_source_index": int(row["alt_source_selected_index"]),
                "qwen_source_index": int(row["qwen_source_selected_index"]),
                "answer_index": int(row["answer_index"]),
                "source_agree": row["agree"] == "True",
                "phi3_source_correct": row["alt_source_correct"] == "True",
                "qwen_source_correct": row["qwen_source_correct"] == "True",
            }
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    categories = Counter()
    for row in rows:
        if row["phi3_source_correct"] and row["qwen_source_correct"]:
            categories["both_correct"] += 1
        elif row["phi3_source_correct"]:
            categories["phi3_only_correct"] += 1
        elif row["qwen_source_correct"]:
            categories["qwen_only_correct"] += 1
        else:
            categories["both_wrong"] += 1
    return {
        "n": n,
        "phi3_source_accuracy": _mean([float(row["phi3_source_correct"]) for row in rows]),
        "qwen_source_accuracy": _mean([float(row["qwen_source_correct"]) for row in rows]),
        "phi3_qwen_choice_agreement": _mean([float(row["source_agree"]) for row in rows]),
        "category_counts": dict(categories),
        "category_rates": {key: value / n for key, value in categories.items()} if n else {},
    }


def _prediction_summary(
    predictions: list[dict[str, Any]],
    agreement: dict[tuple[str, str, str], dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, Any]:
    rows_by_condition_split_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    conditional: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        key = (pred["split"], pred["row_id"], pred["content_id"])
        if key not in agreement:
            continue
        source = agreement[key]
        seed = int(pred.get("seed", -1))
        condition = str(pred["condition"])
        rows_by_condition_split_seed[(condition, pred["split"], seed)].append(pred | {"_source": source})
        if source["phi3_source_correct"] and source["qwen_source_correct"]:
            category = "both_correct"
        elif source["phi3_source_correct"]:
            category = "phi3_only_correct"
        elif source["qwen_source_correct"]:
            category = "qwen_only_correct"
        else:
            category = "both_wrong"
        conditional[(condition, pred["split"], category)].append(pred | {"_source": source})

    seed_rows: list[dict[str, Any]] = []
    for (condition, split, seed), rows in sorted(rows_by_condition_split_seed.items()):
        follows_phi3 = [
            int(int(row["prediction_index"]) == int(row["_source"]["phi3_source_index"]))
            for row in rows
        ]
        follows_qwen = [
            int(int(row["prediction_index"]) == int(row["_source"]["qwen_source_index"]))
            for row in rows
        ]
        seed_rows.append(
            {
                "condition": condition,
                "split": split,
                "seed": seed,
                "n": len(rows),
                "accuracy": _mean([float(row["correct"]) for row in rows]),
                "follows_phi3_source": _mean([float(v) for v in follows_phi3]),
                "follows_qwen_source": _mean([float(v) for v in follows_qwen]),
            }
        )

    aggregate: dict[str, dict[str, Any]] = {}
    for key in sorted({(row["condition"], row["split"]) for row in seed_rows}):
        condition, split = key
        rows = [row for row in seed_rows if row["condition"] == condition and row["split"] == split]
        aggregate[f"{split}:{condition}"] = {
            "source_name": source_name,
            "seed_count": len(rows),
            "n_per_seed": rows[0]["n"] if rows else 0,
            "accuracy_mean": _mean([row["accuracy"] for row in rows]),
            "accuracy_min": min([row["accuracy"] for row in rows], default=0.0),
            "accuracy_max": max([row["accuracy"] for row in rows], default=0.0),
            "follows_phi3_source_mean": _mean([row["follows_phi3_source"] for row in rows]),
            "follows_qwen_source_mean": _mean([row["follows_qwen_source"] for row in rows]),
        }

    conditional_rows: list[dict[str, Any]] = []
    for (condition, split, category), rows in sorted(conditional.items()):
        conditional_rows.append(
            {
                "condition": condition,
                "split": split,
                "category": category,
                "n_prediction_rows": len(rows),
                "n_items": len({(row["row_id"], row["content_id"]) for row in rows}),
                "accuracy": _mean([float(row["correct"]) for row in rows]),
                "follows_phi3_source": _mean(
                    [float(int(int(row["prediction_index"]) == int(row["_source"]["phi3_source_index"]))) for row in rows]
                ),
                "follows_qwen_source": _mean(
                    [float(int(int(row["prediction_index"]) == int(row["_source"]["qwen_source_index"]))) for row in rows]
                ),
            }
        )

    return {
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "conditional_rows": conditional_rows,
    }


def _write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Phi-3 Failure Diagnosis",
        "",
        f"- date: `{payload['date']}`",
        f"- interpretation: `{payload['interpretation']}`",
        "",
        "## Source Cache",
        "",
        "| Split | n | Phi-3 source acc | Qwen source acc | Phi-3/Qwen choice agree | both correct | Phi-3 only | Qwen only | both wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["source_cache_summary"].items():
        counts = row["category_counts"]
        lines.append(
            f"| {split} | {row['n']} | {row['phi3_source_accuracy']:.3f} | "
            f"{row['qwen_source_accuracy']:.3f} | {row['phi3_qwen_choice_agreement']:.3f} | "
            f"{counts.get('both_correct', 0)} | {counts.get('phi3_only_correct', 0)} | "
            f"{counts.get('qwen_only_correct', 0)} | {counts.get('both_wrong', 0)} |"
        )
    lines.extend(["", "## Packet Aggregates", ""])
    lines.extend(
        [
            "| Surface | Seeds | n/seed | Acc | Follow Phi-3 | Follow Qwen |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for surface, row in payload["prediction_summary"]["aggregate"].items():
        lines.append(
            f"| {surface} | {row['seed_count']} | {row['n_per_seed']} | "
            f"{row['accuracy_mean']:.3f} | {row['follows_phi3_source_mean']:.3f} | "
            f"{row['follows_qwen_source_mean']:.3f} |"
        )
    lines.extend(["", "## Conditional Packet Accuracy", ""])
    lines.extend(
        [
            "| Split | Condition | Source category | Items | Acc | Follow Phi-3 | Follow Qwen |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["prediction_summary"]["conditional_rows"]:
        lines.append(
            f"| {row['split']} | {row['condition']} | {row['category']} | "
            f"{row['n_items']} | {row['accuracy']:.3f} | {row['follows_phi3_source']:.3f} | "
            f"{row['follows_qwen_source']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            payload["readout"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    artifact_dir = pathlib.Path(args.artifact_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agreement = _read_agreement(artifact_dir / "source_cache_agreement.csv")
    source_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agreement.values():
        source_by_split[row["split"]].append(row)

    predictions = _read_jsonl(artifact_dir / "matched_predictions.jsonl")
    predictions.extend(
        row
        for row in _read_jsonl(artifact_dir / "qwen_disagreement_predictions.jsonl")
        if row.get("condition") == "qwen_substituted_packet"
    )
    prediction_summary = _prediction_summary(predictions, agreement, source_name="phi3_mini_4k")
    source_cache_summary = {
        split: _summarize_source_rows(rows) for split, rows in sorted(source_by_split.items())
    }
    test_summary = source_cache_summary.get("test", {})
    phi3_acc = float(test_summary.get("phi3_source_accuracy", 0.0))
    qwen_acc = float(test_summary.get("qwen_source_accuracy", 0.0))
    agree = float(test_summary.get("phi3_qwen_choice_agreement", 0.0))
    phi3_packet = prediction_summary["aggregate"].get("test:matched_source_private_packet", {})
    follows_phi3 = float(phi3_packet.get("follows_phi3_source_mean", 0.0))
    readout = (
        "The Phi-3 failure is primarily a source-choice/family boundary result, not packet "
        "corruption: on the ARC test cache Phi-3's source-choice accuracy is "
        f"{phi3_acc:.3f} versus Qwen's {qwen_acc:.3f}, Phi-3 and Qwen choose the same "
        f"candidate only {agree:.3f} of the time, and the decoded packet follows the "
        f"Phi-3 source choice at {follows_phi3:.3f}. The current packet therefore "
        "faithfully transports the alternate source's candidate preference, but that "
        "preference is weaker and often different from the same-family Qwen source."
    )
    payload = {
        "date": "2026-05-05",
        "artifact_dir": str(artifact_dir),
        "source_cache_summary": source_cache_summary,
        "prediction_summary": prediction_summary,
        "interpretation": "source-choice/family boundary, not decoded-packet corruption",
        "readout": readout,
    }

    (output_dir / "phi3_failure_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(payload, output_dir / "phi3_failure_diagnostic.md")
    with (output_dir / "phi3_failure_conditional_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "condition",
                "category",
                "n_items",
                "accuracy",
                "follows_phi3_source",
                "follows_qwen_source",
            ],
        )
        writer.writeheader()
        for row in payload["prediction_summary"]["conditional_rows"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})


if __name__ == "__main__":
    main()
