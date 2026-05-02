from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _svg_text(x: float, y: float, text: str, *, size: int = 14, weight: str = "400", anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="#1f2937">{text}</text>'
    )


def _svg_box(x: float, y: float, w: float, h: float, label: str, body: str, *, fill: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{fill}" stroke="#1f2937" stroke-width="1.5"/>',
            _svg_text(x + w / 2, y + 28, label, size=16, weight="700"),
            _svg_text(x + w / 2, y + 55, body, size=13),
        ]
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2 - 8
    return "\n".join(
        [
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>',
            _svg_text(mid_x, mid_y, label, size=13),
        ]
    )


def write_setup_svg(path: pathlib.Path) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="360" viewBox="0 0 980 360">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="#374151"/>
  </marker>
</defs>
<rect width="980" height="360" fill="#ffffff"/>
{_svg_text(490, 34, "Source-private evidence communication with decoder side information", size=20, weight="700")}
{_svg_box(55, 95, 220, 95, "Public task X", "issue + code + candidates", fill="#e0f2fe")}
{_svg_box(55, 220, 220, 80, "Target state T", "prior + candidate metadata", fill="#d1fae5")}
{_svg_box(385, 95, 220, 95, "Source-private S", "hidden tool trace", fill="#fee2e2")}
{_svg_box(385, 220, 220, 80, "Message M", "1.55-2 byte REPAIR_DIAG", fill="#fef3c7")}
{_svg_box(715, 150, 220, 100, "Decoder D(X,T,M)", "candidate selection", fill="#ede9fe")}
{_arrow(275, 142, 715, 175, "public side information")}
{_arrow(275, 260, 715, 210, "target prior")}
{_arrow(495, 190, 495, 220, "source packetizes")}
{_arrow(605, 260, 715, 210, "rate-capped packet")}
{_svg_text(825, 285, "Output: selected repair candidate", size=14, weight="700")}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _extract_rate_rows(core_summary: dict[str, Any], holdout_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface, summary in [("core_seed29", core_summary), ("holdout_seed30", holdout_summary)]:
        for budget in summary["budget_summaries"]:
            metrics = budget["metrics"]
            for condition, label in [
                ("matched_repair_packet", "packet"),
                ("structured_json_matched", "json"),
                ("structured_free_text_matched", "free_text"),
                ("structured_text_matched", "hidden_log_prefix"),
            ]:
                rows.append(
                    {
                        "surface": surface,
                        "interface": label,
                        "bytes": budget["budget_bytes"],
                        "accuracy": metrics[condition]["accuracy"],
                    }
                )
        for condition, label in [("full_diag_text", "full_diag"), ("full_hidden_log", "full_log")]:
            metric = summary["budget_summaries"][0]["metrics"][condition]
            rows.append(
                {
                    "surface": surface,
                    "interface": label,
                    "bytes": metric["mean_payload_bytes"],
                    "accuracy": metric["accuracy"],
                }
            )
    return rows


def write_rate_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["surface", "interface", "bytes", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def _avg_by_interface(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["interface"], float(row["bytes"])), []).append(float(row["accuracy"]))
    result: dict[str, list[tuple[float, float]]] = {}
    for (interface, byte_count), accs in grouped.items():
        result.setdefault(interface, []).append((byte_count, sum(accs) / len(accs)))
    return {key: sorted(value) for key, value in result.items()}


def write_rate_svg(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    series = _avg_by_interface(rows)
    width, height = 820, 520
    left, right, top, bottom = 80, 40, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = 380.0

    def sx(x: float) -> float:
        return left + (min(x, max_x) / max_x) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y) * plot_h

    colors = {
        "packet": "#2563eb",
        "json": "#dc2626",
        "free_text": "#ea580c",
        "hidden_log_prefix": "#6b7280",
        "full_diag": "#059669",
        "full_log": "#7c3aed",
    }
    labels = {
        "packet": "2-byte packet",
        "json": "JSON relay",
        "free_text": "Free-text relay",
        "hidden_log_prefix": "Hidden-log prefix",
        "full_diag": "Full diagnostic text",
        "full_log": "Full hidden-log relay",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 28, "Accuracy versus communicated bytes", size=20, weight="700"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
    ]
    for y in [0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<line x1="{left}" y1="{sy(y):.1f}" x2="{left + plot_w}" y2="{sy(y):.1f}" stroke="#e5e7eb"/>')
        parts.append(_svg_text(left - 12, sy(y) + 4, f"{y:.2f}", size=12, anchor="end"))
    for x in [0, 2, 14, 32, 100, 200, 374]:
        parts.append(f'<line x1="{sx(x):.1f}" y1="{top + plot_h}" x2="{sx(x):.1f}" y2="{top + plot_h + 5}" stroke="#111827"/>')
        parts.append(_svg_text(sx(x), top + plot_h + 22, str(x), size=11))
    parts.append(_svg_text(left + plot_w / 2, height - 22, "Communicated bytes", size=14, weight="700"))
    parts.append(_svg_text(24, top + plot_h / 2, "Accuracy", size=14, weight="700", anchor="middle"))

    for interface, points in series.items():
        color = colors[interface]
        if len(points) > 1:
            coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{color}"/>')

    legend_x, legend_y = 520, 78
    for idx, interface in enumerate(["packet", "json", "free_text", "hidden_log_prefix", "full_diag", "full_log"]):
        y = legend_y + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{colors[interface]}"/>')
        parts.append(_svg_text(legend_x + 22, y + 2, labels[interface], size=12, anchor="start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--core-summary",
        type=pathlib.Path,
        default=ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/sweep_summary.json",
    )
    parser.add_argument(
        "--holdout-summary",
        type=pathlib.Path,
        default=ROOT / "results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/sweep_summary.json",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    core_summary = _load_json(args.core_summary if args.core_summary.is_absolute() else ROOT / args.core_summary)
    holdout_summary = _load_json(args.holdout_summary if args.holdout_summary.is_absolute() else ROOT / args.holdout_summary)
    rows = _extract_rate_rows(core_summary, holdout_summary)
    write_setup_svg(output_dir / "source_private_setup.svg")
    write_rate_csv(output_dir / "rate_curve.csv", rows)
    write_rate_svg(output_dir / "rate_curve.svg", rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": ["source_private_setup.svg", "rate_curve.csv", "rate_curve.svg"],
                "core_summary": str(args.core_summary),
                "holdout_summary": str(args.holdout_summary),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
