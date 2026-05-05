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
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(values))
    for offset, (label, vals) in zip(offsets, values.items(), strict=True):
        ax.bar(x + offset, vals, width, label=label, color=colors[label], edgecolor="black", linewidth=0.35)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 0.43)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGURE_DIR / "accuracy_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(build_accuracy_overview())


if __name__ == "__main__":
    main()
