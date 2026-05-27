"""Generate polished paper figures from committed result artifacts."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parent
STYLE = FIG_DIR / "outlier_migrate_matplotlib.mplstyle"


ACTIVATION_FILES = {
    "Granite-Small": REPO
    / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitudes.jsonl.gz",
    "Nemotron-3": REPO
    / "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/activation_magnitudes.jsonl.gz",
    "DeepSeek-R1-Distill": REPO
    / "experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z/activation_magnitudes.jsonl.gz",
    "Falcon-H1": REPO
    / "experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z/activation_magnitudes.jsonl.gz",
}


def top_set(values: list[float]) -> frozenset[int]:
    """Return the top 1 percent channel indices by magnitude."""
    arr = np.asarray(values, dtype=np.float32)
    k = max(1, math.ceil(arr.size * 0.01))
    idx = np.argpartition(arr, -k)[-k:]
    return frozenset(int(i) for i in idx)


def set_leaving_curves() -> dict[str, tuple[list[int], list[float]]]:
    """Compute set-leaving curves relative to decode position 100."""
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for model, path in ACTIVATION_FILES.items():
        sets: dict[tuple[int, int, int], frozenset[int]] = {}
        positions: set[int] = set()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pos = int(row["decode_position"])
                positions.add(pos)
                prompt = int(row.get("prompt_index", row.get("prompt_id", 0)))
                layer = int(row["layer_index"])
                sets[(prompt, layer, pos)] = top_set(row["channel_magnitudes"])
        xs = sorted(positions)
        by_pos: dict[int, list[float]] = defaultdict(list)
        for prompt, layer, pos in list(sets):
            if pos != 100:
                continue
            base = sets[(prompt, layer, pos)]
            for other in xs:
                current = sets.get((prompt, layer, other))
                if current is None:
                    continue
                by_pos[other].append(len(base - current) / len(base))
        curves[model] = (xs, [float(np.mean(by_pos[pos])) for pos in xs])
    return curves


def plot_set_leaving() -> None:
    curves = set_leaving_curves()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for model, (xs, ys) in curves.items():
        ax.plot(xs, ys, marker="o", linewidth=2, label=model)
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 1000, 5000, 10000, 20000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-0.02, 0.78)
    ax.set_xlabel("Decode position")
    ax.set_ylabel("Top-1% set-leaving fraction")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "set_leaving_decode_positions.pdf")
    plt.close(fig)


def plot_kl_trajectories() -> None:
    run_dir = REPO / "experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z"
    summary = json.loads((run_dir / "kl_summary.json").read_text())
    fits = json.loads((run_dir / "growth_model_fits.json").read_text())
    labels = {
        "static_1pct": "Static top-1%",
        "decdec_reactive_top1_proxy": "DecDEC proxy",
        "m11_alpha_0_5": "M11 EMA",
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for key, label in labels.items():
        curve = summary["trace_mean_curves"][key]
        xs = np.asarray([p["decode_position"] for p in curve], dtype=float)
        ys = np.asarray([p["mean_kl"] for p in curve], dtype=float)
        ax.plot(xs, ys, linewidth=1.6, label=label)
        fit = fits["regime_fits"][key]["fits"]["sublinear_sqrt"]
        ax.plot(xs, fit["intercept"] + fit["slope"] * np.sqrt(xs), "--", linewidth=1.0, alpha=0.8)
    decay = {
        key: fits["regime_fits"][key]["ar1_decay_estimate"] for key in labels
    }
    note = "\n".join(
        [
            f"AR decay: static {decay['static_1pct']:.2f}",
            f"DecDEC {decay['decdec_reactive_top1_proxy']:.2f}",
            f"M11 {decay['m11_alpha_0_5']:.2f}",
        ]
    )
    ax.text(0.985, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlabel("Decode position")
    ax.set_ylabel(r"Mean KL(BF16 $\parallel$ Q)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kl_accumulation_trajectories.pdf")
    plt.close(fig)


def plot_method_recovery() -> None:
    rows = [
        ("M2 bins", "Granite", -0.867, -4.25, 0.357),
        ("M10 scales", "Granite", 0.234, -3.47, 0.490),
        ("M11 EMA", "Granite", 0.048, -8.94, 0.448),
        ("M18 act+K", "Granite", -0.344, -14.1, 0.731),
        ("DecDEC proxy", "Granite", -0.070, -8.50, 0.391),
        ("M11b top-5", "Granite", 0.449, -2.28, 1.00),
        ("M26 core", "Granite", 0.178, -1.78, 0.986),
        ("ParoQuant", "Granite", 0.754, 0.477, 1.00),
        ("ParoQuant+M11b", "Granite", 0.565, -63.6, 0.932),
        ("Static top-10", "Nemotron", 0.594, 0.246, 0.814),
        ("M11b top-5", "Nemotron", 0.457, 0.313, 0.760),
        ("M11b top-10", "Nemotron", 0.815, 0.158, 0.906),
    ]
    colors = {"Granite": "#4C78A8", "Nemotron": "#F58518"}
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    y = np.arange(len(rows))
    for i, (name, model, med, lo, hi) in enumerate(rows):
        ax.errorbar(
            med,
            i,
            xerr=[[med - lo], [hi - med]],
            fmt="o",
            color=colors[model],
            capsize=3,
            markersize=5,
        )
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.axvline(0.3, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name} ({model})" for name, model, *_ in rows], fontsize=8)
    ax.set_xlim(-2.0, 1.08)
    ax.set_xlabel("Median recovery with 95% bootstrap CI")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(0.31, -0.75, "positive-method threshold", fontsize=7, va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method_recovery_comparison.pdf")
    plt.close(fig)


def plot_per_component() -> None:
    rows = [
        ("Granite", "Attention", 0.557, 0.282),
        ("Granite", "SSM/Mamba", 0.567, 0.270),
        ("Nemotron", "Attention", 0.563, 0.255),
        ("Nemotron", "MoE", 0.527, 0.278),
        ("Nemotron", "SSM/Mamba", 0.533, 0.264),
    ]
    labels = [f"{m}\n{t}" for m, t, *_ in rows]
    strict = np.asarray([r[2] for r in rows])
    shuffle = np.asarray([r[3] for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.bar(x, strict, label="Strict set-leaving", color="#4C78A8")
    ax.bar(x, shuffle, bottom=strict, label="Within-set shuffling", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of base top-1% channels")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_component_drift.pdf")
    plt.close(fig)


def main() -> None:
    if STYLE.exists():
        plt.style.use(STYLE)
    plot_set_leaving()
    plot_kl_trajectories()
    plot_method_recovery()
    plot_per_component()


if __name__ == "__main__":
    main()
