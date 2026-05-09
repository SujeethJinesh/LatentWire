#!/usr/bin/env python3
"""Write the descriptive Phase 3 grid-sensitivity report from a run packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase3/results"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_dir() -> Path:
    candidates = [path for path in DEFAULT_RESULTS_DIR.iterdir() if path.is_dir()] if DEFAULT_RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no Phase 3 result dirs found under {DEFAULT_RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 560
    height = 340
    left = 72
    right = 28
    top = 28
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(row["grid_size"]) for row in rows]
    ys = [float(row["median_recovery"]) for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min = min(0.0, min(ys))
    y_max = max(1.0, max(ys))

    def xmap(x: float) -> float:
        return left + (x - x_min) / max(1.0, x_max - x_min) * plot_w

    def ymap(y: float) -> float:
        return top + (y_max - y) / max(1e-9, y_max - y_min) * plot_h

    points = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f'<circle cx="{xmap(float(row["grid_size"])):.1f}" cy="{ymap(float(row["median_recovery"])):.1f}" '
        f'r="5" fill="#1f77b4"><title>{row["name"]}: {row["median_recovery"]:.6f}</title></circle>'
        for row in rows
    )
    labels = "\n".join(
        f'<text x="{xmap(float(row["grid_size"])):.1f}" y="{height - 34}" text-anchor="middle" font-size="12">{row["name"]}</text>'
        for row in rows
    )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<text x="{width/2:.1f}" y="20" text-anchor="middle" font-size="14">Phase 3 Recovery vs Protected-Grid Density</text>
<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-size="12">Median recovery</text>
<text x="{width/2:.1f}" y="{height-8}" text-anchor="middle" font-size="12">Calibration grid</text>
<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>
{circles}
{labels}
</svg>
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    grid = load_json(run_dir / "grid_sensitivity_metrics.json")
    ordered = [
        ("sparse", grid["grids"]["sparse"]),
        ("primary", grid["grids"]["primary"]),
        ("dense", grid["grids"]["dense"]),
    ]
    rows = [
        {
            "name": name,
            "positions": values["positions"],
            "grid_size": len(values["positions"]),
            "median_recovery": float(values["median_recovery"]),
            "ci95_low": float(values["bootstrap_ci95"]["ci95_low"]),
            "ci95_high": float(values["bootstrap_ci95"]["ci95_high"]),
        }
        for name, values in ordered
    ]
    csv_path = run_dir / "grid_sensitivity.csv"
    csv_path.write_text(
        "name,grid_size,positions,median_recovery,ci95_low,ci95_high\n"
        + "\n".join(
            f"{row['name']},{row['grid_size']},{' '.join(map(str, row['positions']))},"
            f"{row['median_recovery']:.12f},{row['ci95_low']:.12f},{row['ci95_high']:.12f}"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    svg_path = run_dir / "grid_sensitivity.svg"
    write_svg(svg_path, rows)
    md_path = run_dir / "grid_sensitivity.md"
    lines = [
        "# Phase 3 Position Grid Sensitivity",
        "",
        f"Run dir: `{run_dir.relative_to(ROOT)}`",
        "",
        "| Grid | Positions | Median recovery | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['positions']} | {row['median_recovery']:.12f} | "
            f"[{row['ci95_low']:.12f}, {row['ci95_high']:.12f}] |"
        )
    lines.extend(
        [
            "",
            f"CSV: `{csv_path.relative_to(ROOT)}`",
            f"SVG plot: `{svg_path.relative_to(ROOT)}`",
            "",
            "This is a descriptive robustness check, not a separate pass/kill gate.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"markdown": str(md_path), "csv": str(csv_path), "svg": str(svg_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
