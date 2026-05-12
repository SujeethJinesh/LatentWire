#!/usr/bin/env python3
"""Phase 9 Step 9.0 free analytical checks for decode-position channel drift."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "experimental/outlier_migrate/phase9"
TOP_FRACTIONS = (0.01, 0.02)
RANK_DELTA = 2


@dataclass(frozen=True)
class Packet:
    key: str
    label: str
    model_id: str
    run_dir: Path


PACKETS = [
    Packet(
        key="granite4_h_small",
        label="Granite-4-H-Small",
        model_id="ibm-granite/granite-4.0-h-small",
        run_dir=ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
    ),
    Packet(
        key="nemotron3_nano",
        label="Nemotron-3-Nano",
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        run_dir=ROOT / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
    ),
    Packet(
        key="deepseek_r1_distill_qwen_1_5b",
        label="DeepSeek-R1-Distill-Qwen-1.5B",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        run_dir=ROOT / "experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z",
    ),
]

BINS = [
    ("0-500", 0, 500),
    ("500-2000", 500, 2000),
    ("2000-5000", 2000, 5000),
    ("5000-10000", 5000, 10000),
    ("10000-20000", 10000, 20000),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_packet(packet: Packet) -> dict[str, Any]:
    metrics = load_json(packet.run_dir / "metrics.json")
    model_provenance = load_json(packet.run_dir / "model_provenance.json")
    rows = list(iter_rows(packet.run_dir / "activation_magnitudes.jsonl.gz"))
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    return {
        "packet": packet,
        "metrics": metrics,
        "model_provenance": model_provenance,
        "by_trace_layer": by_trace_layer,
        "positions": tuple(int(pos) for pos in metrics["positions"]),
        "layer_count": int(metrics["layer_count"]),
    }


def strict_components_for_layer(base: list[float], target: list[float], *, top_fraction: float) -> dict[str, float]:
    top_k = max(1, math.ceil(len(base) * top_fraction))
    top_boundary = top_k - 1
    base_ranks = ranks_desc(base)
    target_ranks = ranks_desc(target)
    selected = [channel for channel, rank in enumerate(base_ranks) if rank <= top_boundary]
    left = 0
    within = 0
    original = 0
    for channel in selected:
        delta = abs(target_ranks[channel] - base_ranks[channel])
        if delta > RANK_DELTA:
            original += 1
        if target_ranks[channel] > top_boundary:
            left += 1
        elif delta > RANK_DELTA:
            within += 1
    denom = len(selected)
    return {
        "left_set_fraction": left / denom,
        "within_set_rank_shuffle_fraction": within / denom,
        "strict_original_fraction": original / denom,
    }


def compute_decomposition(packet_data: dict[str, Any]) -> dict[str, Any]:
    by_trace_layer = packet_data["by_trace_layer"]
    positions = packet_data["positions"]
    base_position = positions[0]
    final_position = positions[-1]
    trace_rows: list[dict[str, Any]] = []
    for prompt_index in sorted(by_trace_layer):
        trace_values: dict[str, list[float]] = defaultdict(list)
        for layer_index in sorted(by_trace_layer[prompt_index]):
            components = strict_components_for_layer(
                by_trace_layer[prompt_index][layer_index][base_position],
                by_trace_layer[prompt_index][layer_index][final_position],
                top_fraction=0.01,
            )
            for key, value in components.items():
                trace_values[key].append(value)
        trace_rows.append(
            {
                "prompt_index": prompt_index,
                **{key: float(mean(values)) for key, values in trace_values.items()},
            }
        )
    aggregate = {
        "left_set_fraction": float(mean(row["left_set_fraction"] for row in trace_rows)),
        "within_set_rank_shuffle_fraction": float(mean(row["within_set_rank_shuffle_fraction"] for row in trace_rows)),
        "strict_original_fraction": float(mean(row["strict_original_fraction"] for row in trace_rows)),
    }
    return {
        "base_position": base_position,
        "final_position": final_position,
        "trace_rows": trace_rows,
        "aggregate": aggregate,
    }


def bin_positions(positions: tuple[int, ...], low: int, high: int) -> list[int]:
    if low == 0:
        return [position for position in positions if 0 < position <= high]
    return [position for position in positions if low < position <= high]


def top_channels_for_bin(
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]],
    *,
    layer_index: int,
    positions: list[int],
    top_fraction: float,
) -> set[int]:
    first_prompt = next(iter(sorted(by_trace_layer)))
    channel_count = len(by_trace_layer[first_prompt][layer_index][positions[0]])
    top_k = max(1, math.ceil(channel_count * top_fraction))
    means = []
    for channel in range(channel_count):
        values: list[float] = []
        for prompt_index in sorted(by_trace_layer):
            for position in positions:
                values.append(float(by_trace_layer[prompt_index][layer_index][position][channel]))
        means.append(mean(values))
    ordered = sorted(range(channel_count), key=lambda channel: (-means[channel], channel))
    return set(ordered[:top_k])


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left.intersection(right)) / len(left.union(right))


def compute_bin_overlap(packet_data: dict[str, Any]) -> dict[str, Any]:
    by_trace_layer = packet_data["by_trace_layer"]
    positions = packet_data["positions"]
    layer_indices = sorted({layer for trace in by_trace_layer.values() for layer in trace})
    bins = [
        {"name": name, "low": low, "high": high, "positions": bin_positions(positions, low, high)}
        for name, low, high in BINS
    ]
    if any(not item["positions"] for item in bins):
        missing = [item["name"] for item in bins if not item["positions"]]
        raise RuntimeError(f"no recorded decode positions in bins: {missing}")
    rows: list[dict[str, Any]] = []
    for top_fraction in TOP_FRACTIONS:
        by_layer_bin = {
            (layer_index, item["name"]): top_channels_for_bin(
                by_trace_layer,
                layer_index=layer_index,
                positions=item["positions"],
                top_fraction=top_fraction,
            )
            for layer_index in layer_indices
            for item in bins
        }
        for left_bin, right_bin in zip(bins, bins[1:]):
            values = [
                jaccard(
                    by_layer_bin[(layer_index, left_bin["name"])],
                    by_layer_bin[(layer_index, right_bin["name"])],
                )
                for layer_index in layer_indices
            ]
            rows.append(
                {
                    "top_fraction": top_fraction,
                    "left_bin": left_bin["name"],
                    "right_bin": right_bin["name"],
                    "left_positions": left_bin["positions"],
                    "right_positions": right_bin["positions"],
                    "mean_jaccard": float(mean(values)),
                    "min_jaccard": float(min(values)),
                    "max_jaccard": float(max(values)),
                    "layer_count": len(values),
                    "layer_values": [
                        {"layer_index": layer_index, "jaccard": float(value)}
                        for layer_index, value in zip(layer_indices, values)
                    ],
                }
            )
    return {"bins": bins, "rows": rows}


def write_decomposition_markdown(path: Path, payloads: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 9 Step 9.0: Decomposition Replication",
        "",
        "This no-GPU check recomputes the strict set-leaving / within-set rank-shuffling decomposition from existing activation packets before any Phase 9 method run.",
        "",
        "Definitions:",
        "",
        "- Strict set-leaving: a channel ranked inside the top-1% boundary at decode position 100 has rank outside that boundary at decode position 20000.",
        "- Within-set rank shuffling: a channel remains inside the top-1% set at decode position 20000 but moves by more than two rank positions.",
        "- Original drift metric: the earlier rank-change metric, which conflates set-leaving and within-set shuffling.",
        "",
        "| Model | Run | Strict set-leaving | Within-set shuffling | Original drift | Gate status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for payload in payloads:
        packet: Packet = payload["packet"]
        agg = payload["decomposition"]["aggregate"]
        status = "PASS: set-leaving > 0.50" if agg["left_set_fraction"] > 0.50 else "FAIL: set-leaving <= 0.50"
        lines.append(
            f"| {packet.label} | `{packet.run_dir.relative_to(ROOT)}` | "
            f"`{agg['left_set_fraction']:.12f}` | "
            f"`{agg['within_set_rank_shuffle_fraction']:.12f}` | "
            f"`{agg['strict_original_fraction']:.12f}` | {status} |"
        )
    all_pass = all(payload["decomposition"]["aggregate"]["left_set_fraction"] > 0.50 for payload in payloads)
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Set-leaving above `0.50` on all three required models: `{'true' if all_pass else 'false'}`.",
            "- If this value is `false`, Phase 9 must stop and surface to the human because the paper premise is at risk.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_overlap_markdown(path: Path, payloads: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 9 Step 9.0: Bin-to-Bin Top-Channel Overlap",
        "",
        "This no-GPU check computes adjacent-bin Jaccard overlap for top-channel sets from existing activation packets.",
        "",
        "Bin convention: recorded decode positions are assigned to half-open intervals `(low, high]`, except the first bin is `(0, 500]`. Top-channel sets are selected per layer by mean absolute activation magnitude over all traces and recorded positions in the bin.",
        "",
        "| Model | Top fraction | Adjacent bins | Positions compared | Mean Jaccard | Min | Max |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    top1_means: list[float] = []
    for payload in payloads:
        packet: Packet = payload["packet"]
        for row in payload["overlap"]["rows"]:
            if row["top_fraction"] == 0.01:
                top1_means.append(float(row["mean_jaccard"]))
            lines.append(
                f"| {packet.label} | `{row['top_fraction']:.2f}` | "
                f"{row['left_bin']} -> {row['right_bin']} | "
                f"`{row['left_positions']}` -> `{row['right_positions']}` | "
                f"`{row['mean_jaccard']:.12f}` | `{row['min_jaccard']:.12f}` | `{row['max_jaccard']:.12f}` |"
            )
    overall_top1 = float(mean(top1_means)) if top1_means else float("nan")
    lines.extend(
        [
            "",
            "## Decision Readout",
            "",
            f"- Overall mean adjacent-bin top-1% Jaccard overlap: `{overall_top1:.12f}`.",
            "- If this is above `0.40`, M10 is likely to work.",
            "- If this is below `0.40`, M10 is unlikely to work; demote M10 and prioritize M9.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    payloads: list[dict[str, Any]] = []
    for packet in PACKETS:
        packet_data = load_packet(packet)
        payloads.append(
            {
                "packet": packet,
                "decomposition": compute_decomposition(packet_data),
                "overlap": compute_bin_overlap(packet_data),
            }
        )

    write_decomposition_markdown(args.output_dir / "step9_0_decomposition_replication.md", payloads)
    write_overlap_markdown(args.output_dir / "step9_0_bin_overlap_analysis.md", payloads)
    print(
        json.dumps(
            {
                "decomposition_output": str(args.output_dir / "step9_0_decomposition_replication.md"),
                "overlap_output": str(args.output_dir / "step9_0_bin_overlap_analysis.md"),
                "set_leaving_all_above_0_50": all(
                    payload["decomposition"]["aggregate"]["left_set_fraction"] > 0.50
                    for payload in payloads
                ),
                "overall_top1_jaccard": mean(
                    row["mean_jaccard"]
                    for payload in payloads
                    for row in payload["overlap"]["rows"]
                    if row["top_fraction"] == 0.01
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
