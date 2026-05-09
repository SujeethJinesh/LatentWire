#!/usr/bin/env python3
"""Decompose OutlierMigrate rank migration into set-leaving and within-set motion."""

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


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase1 import check_om_phase1 as phase1_checker
from experimental.shared import check_phase0_gate as phase0_checker


DEFAULT_PHASE0_RUN = ROOT / "experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z"
DEFAULT_PHASE1_RUN = ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z"
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260508


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_activation_rows(path: Path) -> Any:
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


def bootstrap_ci(values: list[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(mean(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def decompose(run_dir: Path, *, phase: int) -> dict[str, Any]:
    if phase == 0:
        positions = phase0_checker.POSITIONS
        expected_original_payload = load_json(run_dir / "metrics.json")
        expected_original = expected_original_payload["migration_fraction"]
        expected_original_ci = expected_original_payload["bootstrap_ci95"]
        trace_count = phase0_checker.TRACE_COUNT
    elif phase == 1:
        positions = phase1_checker.POSITIONS
        expected_original_payload = load_json(run_dir / "metrics.json")
        expected_original = expected_original_payload["migration_fraction"]
        expected_original_ci = expected_original_payload["bootstrap_ci95"]
        trace_count = phase1_checker.TRACE_COUNT
    else:
        raise ValueError(f"unsupported phase: {phase}")

    rows = list(iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz"))
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]

    base_position = positions[0]
    final_position = positions[-1]
    top_fraction = float(phase1_checker.THRESHOLDS["top_channel_fraction"])
    rank_delta = int(phase1_checker.THRESHOLDS["rank_delta_strictly_greater_than"])
    trace_metrics: list[dict[str, Any]] = []
    layer_metrics: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    total_selected = 0
    total_left = 0
    total_within = 0

    for prompt_index in sorted(by_trace_layer):
        trace_left: list[float] = []
        trace_within: list[float] = []
        for layer_index in sorted(by_trace_layer[prompt_index]):
            base = by_trace_layer[prompt_index][layer_index][base_position]
            final = by_trace_layer[prompt_index][layer_index][final_position]
            channel_count = len(base)
            top_k = max(1, math.ceil(channel_count * top_fraction))
            top_boundary = top_k - 1
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            selected = [channel for channel, rank in enumerate(base_ranks) if rank <= top_boundary]
            left = 0
            within = 0
            for channel in selected:
                delta = abs(final_ranks[channel] - base_ranks[channel])
                left_set = final_ranks[channel] > top_boundary
                if left_set:
                    left += 1
                elif delta > rank_delta:
                    within += 1
            denom = len(selected)
            total_selected += denom
            total_left += left
            total_within += within
            left_fraction = left / denom
            within_fraction = within / denom
            trace_left.append(left_fraction)
            trace_within.append(within_fraction)
            layer_metrics[layer_index]["left_set_fraction"].append(left_fraction)
            layer_metrics[layer_index]["within_set_rank_shuffle_fraction"].append(within_fraction)
        trace_metrics.append(
            {
                "prompt_index": prompt_index,
                "left_set_fraction": float(mean(trace_left)),
                "within_set_rank_shuffle_fraction": float(mean(trace_within)),
                "layer_count": len(trace_left),
            }
        )

    trace_left_values = [float(row["left_set_fraction"]) for row in trace_metrics]
    trace_within_values = [float(row["within_set_rank_shuffle_fraction"]) for row in trace_metrics]
    aggregate = {
        "left_set_fraction": float(mean(trace_left_values)),
        "left_set_ci95": bootstrap_ci(trace_left_values),
        "within_set_rank_shuffle_fraction": float(mean(trace_within_values)),
        "within_set_rank_shuffle_ci95": bootstrap_ci(trace_within_values),
        "original_migration_fraction": float(expected_original),
        "original_migration_ci95": {
            "ci95_low": float(expected_original_ci["ci95_low"]),
            "ci95_high": float(expected_original_ci["ci95_high"]),
        },
        "raw_count_left_set_fraction": total_left / total_selected,
        "raw_count_within_set_rank_shuffle_fraction": total_within / total_selected,
    }
    return {
        "phase": phase,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "trace_count": trace_count,
        "observed_trace_count": len(trace_metrics),
        "base_position": base_position,
        "final_position": final_position,
        "top_channel_fraction": top_fraction,
        "strict_set_membership_selection": "per prompt and layer, channels with position-100 rank <= ceil(channel_count * 0.01) - 1",
        "top_boundary_semantics": "zero-based rank <= ceil(channel_count * 0.01) - 1",
        "rank_delta_strictly_greater_than": rank_delta,
        "aggregate": aggregate,
        "original_metrics_json_migration_fraction": expected_original,
        "original_metrics_json_ci95": expected_original_ci,
        "trace_metrics": trace_metrics,
        "layer_metrics": [
            {
                "layer_index": layer_index,
                "left_set_fraction": float(mean(values["left_set_fraction"])),
                "within_set_rank_shuffle_fraction": float(mean(values["within_set_rank_shuffle_fraction"])),
                "trace_count": len(values["left_set_fraction"]),
            }
            for layer_index, values in sorted(layer_metrics.items())
        ],
    }


def write_report(run_dir: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        f"# OutlierMigrate Phase {payload['phase']} Migration Decomposition",
        "",
        "## Scope",
        "",
        f"- Run directory: `{payload['run_dir']}`",
        f"- Base decode position: {payload['base_position']}",
        f"- Final decode position: {payload['final_position']}",
        f"- Trace count: {payload['observed_trace_count']}",
        f"- Strict top-channel set: per prompt and layer, channels with rank <= `ceil(channel_count * 0.01) - 1` at decode position {payload['base_position']}.",
        f"- Strict set-leaving definition: a strict base top-1% channel has final rank > `ceil(channel_count * 0.01) - 1`.",
        "- Within-set rank shuffling definition: a base top-1% channel remains in the final top-1% set but moves by more than 2 rank positions.",
        "- Original migration definition: the preregistered checker metric, which selects top-1% channels per layer by mean magnitude at position 100 across traces and counts movement by more than 2 rank positions.",
        "- The first two rows are post-hoc interpretability readouts using the strict set-membership definition; the third row is the unchanged gate metric from `metrics.json`.",
        "",
        "## Aggregate Readout",
        "",
        "| Metric | Fraction | 95% bootstrap CI |",
        "| --- | ---: | ---: |",
        (
            f"| Strict set-leaving | {aggregate['left_set_fraction']:.12f} | "
            f"[{aggregate['left_set_ci95']['ci95_low']:.12f}, {aggregate['left_set_ci95']['ci95_high']:.12f}] |"
        ),
        (
            f"| Within-set rank shuffling | {aggregate['within_set_rank_shuffle_fraction']:.12f} | "
            f"[{aggregate['within_set_rank_shuffle_ci95']['ci95_low']:.12f}, "
            f"{aggregate['within_set_rank_shuffle_ci95']['ci95_high']:.12f}] |"
        ),
        (
            f"| Original conflated migration | {aggregate['original_migration_fraction']:.12f} | "
            f"[{aggregate['original_migration_ci95']['ci95_low']:.12f}, "
            f"{aggregate['original_migration_ci95']['ci95_high']:.12f}] |"
        ),
        "",
        "## Consistency Check",
        "",
        f"- `metrics.json` original migration fraction: {payload['original_metrics_json_migration_fraction']:.12f}",
        f"- Reported original migration fraction: {aggregate['original_migration_fraction']:.12f}",
        "",
        "## Per-Trace Readout",
        "",
        "| Prompt index | Strict set-leaving | Within-set rank shuffling |",
        "| ---: | ---: | ---: |",
    ]
    for row in payload["trace_metrics"]:
        lines.append(
            f"| {row['prompt_index']} | {row['left_set_fraction']:.12f} | "
            f"{row['within_set_rank_shuffle_fraction']:.12f} |"
        )
    lines.extend(["", "## Machine-Readable Payload", "", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    (run_dir / "migration_decomposition.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-run-dir", type=Path, default=DEFAULT_PHASE0_RUN)
    parser.add_argument("--phase1-run-dir", type=Path, default=DEFAULT_PHASE1_RUN)
    args = parser.parse_args(argv)

    for phase, run_dir in [(0, args.phase0_run_dir), (1, args.phase1_run_dir)]:
        payload = decompose(run_dir.resolve(), phase=phase)
        write_report(run_dir.resolve(), payload)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "run_dir": str(run_dir),
                    "left_set_fraction": payload["aggregate"]["left_set_fraction"],
                    "within_set_rank_shuffle_fraction": payload["aggregate"][
                        "within_set_rank_shuffle_fraction"
                    ],
                    "original_migration_fraction": payload["aggregate"]["original_migration_fraction"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
