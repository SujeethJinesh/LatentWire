from __future__ import annotations

"""Build camera-ready COLM_v3 figures that are not regenerated elsewhere."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "colm_final/paper/figures"


def build_accuracy_overview() -> pathlib.Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = ["ARC-Challenge", "OpenBookQA"]
    values = {
        "Target-only": [0.265, 0.276],
        "Same-byte text": [0.300, 0.350],
        "Source index": [0.346, 0.378],
        "LatentWire": [0.344, 0.378],
    }
    colors = {
        "Target-only": "#a7a7a7",
        "Same-byte text": "#6baed6",
        "Source index": "#fdae6b",
        "LatentWire": "#31a354",
    }
    x = np.arange(len(benchmarks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(values))
    for offset, (label, vals) in zip(offsets, values.items(), strict=True):
        ax.bar(x + offset, vals, width, label=label, color=colors[label], edgecolor="black", linewidth=0.35)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 0.48)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.text(
        0.5,
        0.455,
        "Source-index is the decisive boundary: it matches or exceeds the packet.",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    out = FIGURE_DIR / "accuracy_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(build_accuracy_overview())


if __name__ == "__main__":
    main()
