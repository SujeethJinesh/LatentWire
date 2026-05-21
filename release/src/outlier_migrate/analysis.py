"""Small analysis helpers for release artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from outlier_migrate.data import write_json
from outlier_migrate.metrics import RecoverySummary


def rank_recovery_summaries(summaries: dict[str, RecoverySummary]) -> list[dict[str, object]]:
    """Return summaries sorted by median recovery descending."""

    rows = []
    for name, summary in summaries.items():
        row = {"method": name, **asdict(summary)}
        rows.append(row)
    return sorted(rows, key=lambda row: (-float(row["median_recovery"]), str(row["method"])))


def write_analysis_manifest(path: Path, *, status: str, rows: list[dict[str, object]]) -> None:
    """Write a release analysis manifest."""

    write_json(path, {"status": status, "rows": rows})
