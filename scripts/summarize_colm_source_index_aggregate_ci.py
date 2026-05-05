from __future__ import annotations

"""Clustered aggregate CI for packet minus source-index.

The COLM_v3 paper already reports per-seed packet-vs-source-index lower bounds.
This script adds the reviewer-requested aggregate number by consuming the frozen
prediction rows from the acceptance-baseline audit. It does not rerun models.
"""

import argparse
import csv
import gzip
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "results/source_private_colm_acceptance_baselines_20260502"
DEFAULT_SAMPLES = 10_000
PACKET_CONDITION = "matched_source_private_packet"
SOURCE_INDEX_CONDITION = "source_index_byte"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _seed_deltas(rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    packet: dict[tuple[int, str], int] = {}
    source: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["row_id"]))
        if row["condition"] == PACKET_CONDITION:
            packet[key] = int(bool(row["correct"]))
        elif row["condition"] == SOURCE_INDEX_CONDITION:
            source[key] = int(bool(row["correct"]))
    if set(packet) != set(source):
        missing_keys = sorted(set(packet) ^ set(source))[:5]
        raise ValueError(f"packet/source row mismatch; first missing keys: {missing_keys}")
    by_seed: dict[int, list[int]] = {}
    for seed, row_id in sorted(packet):
        by_seed.setdefault(seed, []).append(packet[(seed, row_id)] - source[(seed, row_id)])
    return {seed: np.asarray(values, dtype=np.float64) for seed, values in by_seed.items()}


def _cluster_bootstrap(by_seed: dict[int, np.ndarray], samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    seeds = sorted(by_seed)
    observed = float(np.mean(np.concatenate([by_seed[item] for item in seeds])))
    draws = np.empty(samples, dtype=np.float64)
    for sample_idx in range(samples):
        total = 0.0
        count = 0
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        for chosen_seed in chosen:
            values = by_seed[int(chosen_seed)]
            item_idx = rng.integers(0, len(values), size=len(values))
            total += float(values[item_idx].sum())
            count += int(len(values))
        draws[sample_idx] = total / count
    return {
        "mean": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _summarize_one(path: pathlib.Path, benchmark: str, samples: int, seed: int) -> dict[str, Any]:
    rows = _read_rows(path)
    by_seed = _seed_deltas(rows)
    ci = _cluster_bootstrap(by_seed, samples=samples, seed=seed)
    item_count = int(sum(len(values) for values in by_seed.values()) / len(by_seed))
    return {
        "benchmark": benchmark,
        "prediction_file": str(path.relative_to(ROOT)),
        "prediction_sha256": _sha256(path),
        "seed_count": len(by_seed),
        "items_per_seed": item_count,
        "paired_units": int(sum(len(values) for values in by_seed.values())),
        "bootstrap": "two_stage_seed_item_cluster_bootstrap",
        "bootstrap_samples": samples,
        **ci,
        "interpretation": (
            "Positive values would mean the packet beats explicit source-index communication. "
            "The current aggregate is non-positive or has a non-positive lower bound, so the "
            "paper should keep the source-choice boundary in the main claim."
        ),
    }


def _write_outputs(output_dir: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    json_path = output_dir / "aggregate_source_index_ci.json"
    md_path = output_dir / "aggregate_source_index_ci.md"
    csv_path = output_dir / "aggregate_source_index_ci.csv"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "benchmark",
        "seed_count",
        "items_per_seed",
        "paired_units",
        "mean",
        "ci95_low",
        "ci95_high",
        "bootstrap_samples",
        "prediction_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    lines = [
        "# Aggregate Packet-vs-Source-Index CI",
        "",
        "This artifact consumes frozen prediction rows from the COLM acceptance-baseline audit. It uses a two-stage seed/item cluster bootstrap and does not rerun models.",
        "",
        "| Benchmark | Seeds | Items/seed | Mean pkt-src | CI95 low | CI95 high |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['seed_count']} | {row['items_per_seed']} | "
            f"{row['mean']:+.4f} | {row['ci95_low']:+.4f} | {row['ci95_high']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: neither benchmark supports packet superiority over explicit source-index communication. This strengthens the workshop claim boundary rather than the positive-method claim.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = set(manifest.get("artifacts", []))
        artifacts.update([json_path.name, md_path.name, csv_path.name])
        manifest["artifacts"] = sorted(artifacts)
        artifact_hashes = manifest.setdefault("artifact_sha256", {})
        for path in (json_path, md_path, csv_path):
            artifact_hashes[path.name] = _sha256(path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260505)
    args = parser.parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    rows = [
        _summarize_one(input_dir / "arc_test_predictions.jsonl.gz", "ARC-Challenge", args.samples, args.seed + 1),
        _summarize_one(input_dir / "openbookqa_test_predictions.jsonl.gz", "OpenBookQA", args.samples, args.seed + 2),
    ]
    _write_outputs(input_dir, rows)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
