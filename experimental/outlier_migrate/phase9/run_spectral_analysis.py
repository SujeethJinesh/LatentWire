#!/usr/bin/env python3
"""Spectral analysis of decode-position channel trajectories.

This is a no-GPU analysis over existing activation packets. It computes FFT
summary statistics only for packets with dense, uniformly spaced decode
positions. Sparse packets are reported as not identifiable for FFT purposes.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_MD = ROOT / "experimental/outlier_migrate/phase9/spectral_analysis.md"
OUT_JSON = ROOT / "experimental/outlier_migrate/phase9/spectral_analysis.json"

PACKETS = [
    {
        "label": "Granite-Small dense M11 packet",
        "model": "ibm-granite/granite-4.0-h-small",
        "run_dir": ROOT
        / "experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z",
        "role": "dense_available",
    },
    {
        "label": "Granite-Small Phase 1 packet",
        "model": "ibm-granite/granite-4.0-h-small",
        "run_dir": ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z",
        "role": "four_model_sparse",
    },
    {
        "label": "Nemotron-3-Nano Phase 2 packet",
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "run_dir": ROOT
        / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
        "role": "four_model_sparse",
    },
    {
        "label": "DeepSeek-R1-Distill-Qwen-1.5B Phase 5' packet",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "run_dir": ROOT
        / "experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z",
        "role": "four_model_sparse",
    },
    {
        "label": "Falcon-H1 Phase 7 packet",
        "model": "tiiuae/Falcon-H1-1.5B-Base",
        "run_dir": ROOT / "experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z",
        "role": "four_model_sparse",
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def is_uniform_dense(positions: list[int]) -> bool:
    if len(positions) < 20:
        return False
    diffs = [b - a for a, b in zip(positions, positions[1:])]
    return len(set(diffs)) == 1


def top_channels(values: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((np.arange(values.shape[0]), -values))
    return order[:count]


def autocorr_length(values: np.ndarray, step_tokens: int) -> int | None:
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return None
    threshold = 1.0 / math.e
    for lag in range(1, len(centered)):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if corr < threshold:
            return lag * step_tokens
    return None


def analyze_dense_packet(packet: dict[str, Any], positions: list[int]) -> dict[str, Any]:
    run_dir = Path(packet["run_dir"])
    activation_path = run_dir / "activation_magnitudes.jsonl.gz"
    sums: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = defaultdict(int)
    channel_counts: dict[int, int] = {}

    for row in iter_rows(activation_path):
        layer = int(row["layer_index"])
        pos = int(row["decode_position"])
        if pos not in positions:
            continue
        vector = np.asarray(row["channel_magnitudes"], dtype=np.float64)
        key = (layer, pos)
        if key not in sums:
            sums[key] = np.zeros_like(vector)
        sums[key] += vector
        counts[key] += 1
        channel_counts[layer] = vector.shape[0]

    step_tokens = positions[1] - positions[0]
    low_freq_fractions: list[float] = []
    spectral_entropies: list[float] = []
    autocorr_lengths: list[int] = []
    sampled_channels = 0
    layer_count = 0

    for layer, channel_count in sorted(channel_counts.items()):
        if any((layer, pos) not in sums for pos in positions):
            continue
        means = np.stack([sums[(layer, pos)] / counts[(layer, pos)] for pos in positions])
        top_k = max(1, math.ceil(0.01 * channel_count))
        selected = top_channels(means[0], top_k)
        layer_count += 1
        for channel in selected:
            trajectory = means[:, int(channel)]
            centered = trajectory - trajectory.mean()
            spectrum = np.fft.rfft(centered)
            power = np.abs(spectrum) ** 2
            non_dc = power[1:]
            if non_dc.sum() <= 0:
                continue
            low_bins = max(1, math.ceil(0.10 * len(non_dc)))
            probs = non_dc / non_dc.sum()
            low_freq_fractions.append(float(non_dc[:low_bins].sum() / non_dc.sum()))
            spectral_entropies.append(float(-(probs * np.log2(probs + 1e-30)).sum() / math.log2(len(probs))))
            acl = autocorr_length(trajectory, step_tokens)
            if acl is not None:
                autocorr_lengths.append(acl)
            sampled_channels += 1

    return {
        "label": packet["label"],
        "model": packet["model"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "positions": positions,
        "position_count": len(positions),
        "step_tokens": step_tokens,
        "layer_count": layer_count,
        "sampled_top1_channels": sampled_channels,
        "median_low_frequency_power_fraction_first_10pct_bins": median(low_freq_fractions)
        if low_freq_fractions
        else None,
        "median_normalized_spectral_entropy": median(spectral_entropies) if spectral_entropies else None,
        "median_autocorrelation_length_tokens": median(autocorr_lengths) if autocorr_lengths else None,
        "low_frequency_power_fraction_values": low_freq_fractions,
        "normalized_spectral_entropy_values": spectral_entropies,
        "autocorrelation_length_token_values": autocorr_lengths,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for packet in PACKETS:
        run_dir = Path(packet["run_dir"])
        manifest = load_json(run_dir / "activation_magnitude_manifest.json")
        positions = [int(pos) for pos in manifest.get("positions", [])]
        if is_uniform_dense(positions):
            results.append(analyze_dense_packet(packet, positions))
        else:
            skipped.append(
                {
                    "label": packet["label"],
                    "model": packet["model"],
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "positions": positions,
                    "reason": "FFT/autocorrelation requires dense uniformly spaced trajectories; this packet is sparse.",
                }
            )

    payload = {
        "schema_version": "om_phase9_spectral_analysis_v1",
        "results": results,
        "skipped": skipped,
        "interpretation": (
            "Only the Granite M11 packet currently supports dense FFT analysis. "
            "Four-model all-20K spectral analysis is not identifiable from the existing sparse packets."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 9 Spectral Analysis of Decode-Position Channel Trajectories",
        "",
        "Generated: 2026-05-18",
        "",
        "## Identifiability",
        "",
        "The requested four-model, all-20K FFT is not fully identifiable from current packets. "
        "Granite-Small has a dense M11 packet sampled every 100 decode positions from 100 to 10000. "
        "The four-model packets for Granite-Small, Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1 "
        "are sparse grids with 6 positions, so FFT and autocorrelation estimates would be misleading.",
        "",
        "This report therefore computes the spectral readout only where the packet is dense and "
        "uniformly spaced, and records skipped packets explicitly.",
        "",
        "## Dense-Packet Results",
        "",
        "| Packet | Positions | Layers | Top-1% channels | Low-frequency power (first 10% bins) | Spectral entropy | Autocorr length (tokens) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['label']} | {row['position_count']} | {row['layer_count']} | "
            f"{row['sampled_top1_channels']} | "
            f"{fmt(row['median_low_frequency_power_fraction_first_10pct_bins'])} | "
            f"{fmt(row['median_normalized_spectral_entropy'])} | "
            f"{fmt(row['median_autocorrelation_length_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "## Skipped Packets",
            "",
            "| Packet | Positions | Reason |",
            "| --- | ---: | --- |",
        ]
    )
    for row in skipped:
        lines.append(f"| {row['label']} | {len(row['positions'])} | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The dense Granite readout provides an initial sanity check, not a four-model conclusion. "
            "If the low-frequency power fraction is high and spectral entropy is low, streaming "
            "subspace or predictor methods are more plausible on Granite. If the spectrum is broad, "
            "local trajectory prediction is unlikely to rescue the current W4A16 protection family.",
            "",
            "Because the dense packet stops at decode position 10000 and only covers Granite-Small, "
            "this analysis should not be used to claim cross-model spectral structure. A future "
            "component/dense packet would need uniform trajectories to 20000 on Nemotron, DeepSeek, "
            "and Falcon before promoting Streaming-PCA beyond a conditional follow-up.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
