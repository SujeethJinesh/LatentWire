#!/usr/bin/env python3
"""Experiment E: threshold sensitivity for OutlierMigrate decomposition."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "experimental/outlier_migrate/decomposition_analysis"
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260512
RANK_DELTA = 2
THRESHOLDS = [0.005, 0.01, 0.02, 0.05]

PACKETS = [
    {
        "key": "phase0_granite_tiny",
        "label": "Phase 0 Granite-Tiny",
        "run_dir": ROOT / "experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z",
    },
    {
        "key": "phase1_granite_small",
        "label": "Phase 1 Granite-Small",
        "run_dir": ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
    },
    {
        "key": "phase2_nemotron3",
        "label": "Phase 2 Nemotron-3",
        "run_dir": ROOT / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
    },
    {
        "key": "phase5p_transformer",
        "label": "Phase 5' Transformer",
        "run_dir": ROOT / "experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z",
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def bootstrap_ci(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(mean(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def load_packet(packet: dict[str, Any]) -> dict[str, Any]:
    metrics = load_json(packet["run_dir"] / "metrics.json")
    rows = list(iter_rows(packet["run_dir"] / "activation_magnitudes.jsonl.gz"))
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    return {
        **packet,
        "metrics": metrics,
        "positions": tuple(int(pos) for pos in metrics["positions"]),
        "by_trace_layer": by_trace_layer,
    }


def gate_style_top_channels(by_trace_layer: dict[int, dict[int, dict[int, list[float]]]], *, top_fraction: float, base_position: int) -> dict[int, list[int]]:
    layer_indices = sorted({layer for trace in by_trace_layer.values() for layer in trace})
    selected: dict[int, list[int]] = {}
    for layer_index in layer_indices:
        base_vectors = [
            by_trace_layer[prompt_index][layer_index][base_position]
            for prompt_index in sorted(by_trace_layer)
            if layer_index in by_trace_layer[prompt_index]
        ]
        channel_count = len(base_vectors[0])
        top_k = max(1, math.ceil(channel_count * top_fraction))
        mean_magnitudes = [
            mean(float(vector[channel]) for vector in base_vectors) for channel in range(channel_count)
        ]
        selected[layer_index] = sorted(
            range(channel_count), key=lambda channel: (-mean_magnitudes[channel], channel)
        )[:top_k]
    return selected


def compute_for_threshold(packet_data: dict[str, Any], *, top_fraction: float) -> dict[str, Any]:
    by_trace_layer = packet_data["by_trace_layer"]
    positions = packet_data["positions"]
    base_position = positions[0]
    final_position = positions[-1]
    gate_top = gate_style_top_channels(by_trace_layer, top_fraction=top_fraction, base_position=base_position)
    trace_rows: list[dict[str, Any]] = []
    for prompt_index in sorted(by_trace_layer):
        trace_values: dict[str, list[float]] = defaultdict(list)
        for layer_index in sorted(by_trace_layer[prompt_index]):
            base = by_trace_layer[prompt_index][layer_index][base_position]
            final = by_trace_layer[prompt_index][layer_index][final_position]
            top_k = max(1, math.ceil(len(base) * top_fraction))
            top_boundary = top_k - 1
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            strict_selected = [channel for channel, rank in enumerate(base_ranks) if rank <= top_boundary]
            left = 0
            within = 0
            strict_original = 0
            for channel in strict_selected:
                delta = abs(final_ranks[channel] - base_ranks[channel])
                if delta > RANK_DELTA:
                    strict_original += 1
                if final_ranks[channel] > top_boundary:
                    left += 1
                elif delta > RANK_DELTA:
                    within += 1
            gate_selected = gate_top[layer_index]
            gate_migrated = sum(
                1 for channel in gate_selected if abs(final_ranks[channel] - base_ranks[channel]) > RANK_DELTA
            )
            trace_values["left_set_fraction"].append(left / len(strict_selected))
            trace_values["within_set_rank_shuffle_fraction"].append(within / len(strict_selected))
            trace_values["strict_original_fraction"].append(strict_original / len(strict_selected))
            trace_values["gate_style_migration_fraction"].append(gate_migrated / len(gate_selected))
        trace_rows.append(
            {
                "prompt_index": prompt_index,
                **{key: float(mean(values)) for key, values in trace_values.items()},
            }
        )
    aggregate: dict[str, Any] = {
        "top_channel_fraction": top_fraction,
        "top_channel_percent": top_fraction * 100.0,
        "base_position": base_position,
        "final_position": final_position,
    }
    for key in [
        "left_set_fraction",
        "within_set_rank_shuffle_fraction",
        "strict_original_fraction",
        "gate_style_migration_fraction",
    ]:
        values = [float(row[key]) for row in trace_rows]
        aggregate[key] = float(mean(values))
        aggregate[f"{key}_ci95"] = bootstrap_ci(values)
    return {"aggregate": aggregate, "trace_rows": trace_rows}


def stable_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    baseline = next(row for row in rows if math.isclose(row["top_channel_fraction"], 0.01))
    stable: list[float] = []
    for row in rows:
        left_ok = abs(row["left_set_fraction"] - baseline["left_set_fraction"]) <= 0.10
        within_ok = abs(row["within_set_rank_shuffle_fraction"] - baseline["within_set_rank_shuffle_fraction"]) <= 0.10
        if left_ok and within_ok:
            stable.append(float(row["top_channel_fraction"]))
    return stable


def write_report(path: Path, packets: list[dict[str, Any]]) -> None:
    lines = [
        "# OutlierMigrate Experiment E: Threshold Sensitivity",
        "",
        "## Scope",
        "",
        "This post-hoc analysis recomputes decomposition at top-channel thresholds 0.5%, 1%, 2%, and 5%. It reports all thresholds; no threshold is selected post hoc.",
        "",
        "## Summary",
        "",
        "| Packet | Top % | Gate-style migration | Strict set-leaving | Within-set shuffling | Stable vs top-1%? |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for packet in packets:
        stable = set(packet["stable_thresholds"])
        for row in packet["threshold_rows"]:
            lines.append(
                f"| {packet['label']} | {row['top_channel_percent']:.1f} | "
                f"{row['gate_style_migration_fraction']:.12f} | "
                f"{row['left_set_fraction']:.12f} | "
                f"{row['within_set_rank_shuffle_fraction']:.12f} | "
                f"{'yes' if row['top_channel_fraction'] in stable else 'no'} |"
            )
    lines.extend([
        "",
        "## Stability Rule",
        "",
        "A threshold is marked stable when both strict set-leaving and within-set rank-shuffling remain within 0.10 absolute fraction of the top-1% values for the same packet.",
        "",
        "## Figure",
        "",
        "`threshold_sensitivity.pdf` plots strict set-leaving and within-set rank-shuffling as a function of the top-channel threshold.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(path: Path, packets: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for packet in packets:
        rows = packet["threshold_rows"]
        x = [row["top_channel_percent"] for row in rows]
        axes[0].plot(x, [row["left_set_fraction"] for row in rows], marker="o", label=packet["label"])
        axes[1].plot(x, [row["within_set_rank_shuffle_fraction"] for row in rows], marker="o", label=packet["label"])
    axes[0].set_title("Strict Set-Leaving")
    axes[1].set_title("Within-Set Rank Shuffling")
    for axis in axes:
        axis.set_xlabel("Protected top-channel threshold (%)")
        axis.set_ylabel("Fraction")
        axis.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packets_out: list[dict[str, Any]] = []
    for packet in PACKETS:
        if not packet["run_dir"].is_dir():
            continue
        data = load_packet(packet)
        threshold_rows = [
            compute_for_threshold(data, top_fraction=threshold)["aggregate"]
            for threshold in THRESHOLDS
        ]
        packets_out.append(
            {
                "key": packet["key"],
                "label": packet["label"],
                "run_dir": str(packet["run_dir"].relative_to(ROOT)),
                "threshold_rows": threshold_rows,
                "stable_thresholds": stable_thresholds(threshold_rows),
            }
        )
    payload = {
        "schema_version": "om_experiment_e_threshold_sensitivity_v1",
        "thresholds": THRESHOLDS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rank_delta_strictly_greater_than": RANK_DELTA,
        "packets": packets_out,
    }
    write_json(args.output_dir / "threshold_sensitivity.json", payload)
    write_report(args.output_dir / "threshold_sensitivity.md", packets_out)
    write_figure(args.output_dir / "threshold_sensitivity.pdf", packets_out)
    print(json.dumps({"output": str(args.output_dir / "threshold_sensitivity.md"), "packets": len(packets_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
