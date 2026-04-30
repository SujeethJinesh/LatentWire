from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of grid records")
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_basis_mode: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        by_basis_mode.setdefault((record["mode"], record["basis"]), []).append(record)

    bidirectional_passes: list[dict[str, Any]] = []
    for (mode, basis), rows in sorted(by_basis_mode.items()):
        directions = {row["direction"]: row for row in rows}
        if {"core_to_holdout", "holdout_to_core"}.issubset(directions) and all(
            directions[direction]["pass"] for direction in ("core_to_holdout", "holdout_to_core")
        ):
            bidirectional_passes.append(
                {
                    "mode": mode,
                    "basis": basis,
                    "core_to_holdout_source": directions["core_to_holdout"]["source"],
                    "holdout_to_core_source": directions["holdout_to_core"]["source"],
                    "min_ci95_low": min(
                        directions["core_to_holdout"]["ci95_low"],
                        directions["holdout_to_core"]["ci95_low"],
                    ),
                }
            )

    max_source = max((row["source"] for row in records), default=None)
    max_source_minus_best = max((row["source"] - row["best_control"] for row in records), default=None)
    max_ci95_low = max((row["ci95_low"] for row in records), default=None)
    best_rows = sorted(
        records,
        key=lambda row: (row["source"] - row["best_control"], row["source"], row["ci95_low"]),
        reverse=True,
    )[:8]
    return {
        "rows": len(records),
        "pass_rows": sum(1 for row in records if row["pass"]),
        "modes": sorted({row["mode"] for row in records}),
        "basis_views": sorted({row["basis"] for row in records}),
        "directions": sorted({row["direction"] for row in records}),
        "bidirectional_basis_mode_passes": bidirectional_passes,
        "bidirectional_pass_count": len(bidirectional_passes),
        "max_source_accuracy": max_source,
        "max_source_minus_best_control": max_source_minus_best,
        "max_ci95_low_vs_best_control": max_ci95_low,
        "best_rows": best_rows,
        "pass_gate": bool(bidirectional_passes),
        "interpretation": (
            "No existing public basis or diagnostic-table mode rescues bidirectional held-out-family "
            "conditional PQ innovation at n256. This weakens the hypothesis that cross-family failure is "
            "only a static-basis selection problem and points to ontology calibration or public-conditioned "
            "codebooks as the next branch."
        ),
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Conditional PQ Basis/Schema Grid",
        "",
        f"- pass gate: `{summary['pass_gate']}`",
        f"- rows: `{summary['rows']}`",
        f"- pass rows: `{summary['pass_rows']}`",
        f"- bidirectional basis/mode passes: `{summary['bidirectional_pass_count']}`",
        f"- max source accuracy: `{summary['max_source_accuracy']}`",
        f"- max source minus best control: `{summary['max_source_minus_best_control']}`",
        f"- max CI95 low vs best control: `{summary['max_ci95_low_vs_best_control']}`",
        "",
        "## Best Rows",
        "",
        "| Mode | Basis | Direction | Pass | Source | Target | Best control | CI95 low | Unquantized |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["best_rows"]:
        lines.append(
            f"| {row['mode']} | {row['basis']} | {row['direction']} | `{row['pass']}` | "
            f"{row['source']:.3f} | {row['target']:.3f} | {row['best_control']:.3f} | "
            f"{row['ci95_low']:.3f} | {row['unquantized']:.3f} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-records",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_conditional_pq_basis_schema_grid_20260430/summary_grid_records.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/source_private_conditional_pq_basis_schema_grid_20260430/summary"),
    )
    args = parser.parse_args()
    records_path = args.grid_records if args.grid_records.is_absolute() else ROOT / args.grid_records
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(records_path)
    payload = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "grid_records": str(records_path.relative_to(ROOT)),
        "records": records,
        "summary": _summarize(records),
    }
    (output_dir / "conditional_pq_basis_schema_grid_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "conditional_pq_basis_schema_grid_summary.md", payload)
    manifest = {
        "artifacts": [
            "conditional_pq_basis_schema_grid_summary.json",
            "conditional_pq_basis_schema_grid_summary.md",
            "manifest.json",
            "manifest.md",
        ],
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in [
                "conditional_pq_basis_schema_grid_summary.json",
                "conditional_pq_basis_schema_grid_summary.md",
            ]
        },
        "pass_gate": payload["summary"]["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Conditional PQ Basis/Schema Grid Manifest",
                "",
                f"- pass gate: `{payload['summary']['pass_gate']}`",
                f"- rows: `{payload['summary']['rows']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["summary"]["pass_gate"],
                "rows": payload["summary"]["rows"],
                "pass_rows": payload["summary"]["pass_rows"],
                "bidirectional_pass_count": payload["summary"]["bidirectional_pass_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
