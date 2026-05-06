"""Validate SinkAware native GPU handoff packets for review completeness.

This checker is intentionally about admissibility, not success. A passing
packet has enough native timing, quality, NCU, metadata, and decision artifacts
for a reviewer to inspect a promote/kill decision. It does not make a GPU speed
or quality claim by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


REQUIRED_FILES = [
    "metadata.json",
    "quality_drift.csv",
    "quality_drift_by_head.csv",
    "latency.csv",
    "ncu_summary.csv",
    "decision.md",
]

REQUIRED_METADATA_FIELDS = [
    "gpu",
    "driver",
    "cuda",
    "pytorch",
    "triton",
    "model",
    "dtype",
    "sequence_shapes",
]

REQUIRED_ROW_IDS = [
    "exact_attention",
    "exact_fixed_sink_decomposition",
    "rank2_sink_logit_predictor",
    "position_only_predictor",
]

ROW_ALIASES = {
    "exact": "exact_attention",
    "exactattention": "exact_attention",
    "exact_attention": "exact_attention",
    "exact_attention_reference": "exact_attention",
    "exact_fixed_sink_decomposition": "exact_fixed_sink_decomposition",
    "exact_sink_decomposition": "exact_fixed_sink_decomposition",
    "exact_decomposition": "exact_fixed_sink_decomposition",
    "fixed_sink_decomposition": "exact_fixed_sink_decomposition",
    "rank2": "rank2_sink_logit_predictor",
    "rank_2": "rank2_sink_logit_predictor",
    "rank2_sink_logit_predictor": "rank2_sink_logit_predictor",
    "rank2_sink_predictor": "rank2_sink_logit_predictor",
    "rank2_predictor": "rank2_sink_logit_predictor",
    "position": "position_only_predictor",
    "position_only": "position_only_predictor",
    "position_only_predictor": "position_only_predictor",
}

CSV_SCHEMAS = {
    "quality_drift.csv": {
        "required_columns": [
            "row",
            "model",
            "sequence_length",
            "batch_size",
            "layer",
            "output_rel_l2",
            "sink_mass_mae",
            "attention_l1",
        ],
        "numeric_columns": [
            "sequence_length",
            "batch_size",
            "output_rel_l2",
            "sink_mass_mae",
            "attention_l1",
        ],
    },
    "quality_drift_by_head.csv": {
        "required_columns": [
            "row",
            "model",
            "sequence_length",
            "batch_size",
            "layer",
            "head",
            "output_rel_l2",
            "sink_mass_mae",
            "attention_l1",
        ],
        "numeric_columns": [
            "sequence_length",
            "batch_size",
            "head",
            "output_rel_l2",
            "sink_mass_mae",
            "attention_l1",
        ],
    },
    "latency.csv": {
        "required_columns": [
            "row",
            "model",
            "sequence_length",
            "batch_size",
            "run_id",
            "latency_ms",
        ],
        "numeric_columns": ["sequence_length", "batch_size", "latency_ms"],
    },
    "ncu_summary.csv": {
        "required_columns": [
            "row",
            "model",
            "sequence_length",
            "batch_size",
            "dram_bytes",
            "l2_bytes",
            "achieved_occupancy",
            "registers_per_thread",
        ],
        "numeric_columns": [
            "sequence_length",
            "batch_size",
            "dram_bytes",
            "l2_bytes",
            "achieved_occupancy",
            "registers_per_thread",
        ],
    },
}

ROW_COLUMN_ALIASES = ["row", "row_id", "method", "variant"]
PLACEHOLDER_MARKERS = [
    "TODO_NATIVE_SINKAWARE_FILL",
    "TODO_NATIVE_PROFILE_FILL",
    "TODO",
    "TBD",
    "PLACEHOLDER",
    "FILL_ME",
    "SCREENSHOT_ONLY",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _contains_placeholder(text: str) -> bool:
    upper = text.upper()
    return any(marker in upper for marker in PLACEHOLDER_MARKERS)


def _normalize_token(value: object) -> str:
    text = str(value).strip().lower()
    for char in [" ", "-", "/", "."]:
        text = text.replace(char, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _canonical_row_id(value: object) -> str | None:
    token = _normalize_token(value)
    return ROW_ALIASES.get(token)


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and not _contains_placeholder(value)
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _is_number(value: object) -> bool:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in {float("inf"), float("-inf")}


def _row_value(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row:
            return row[name]
    return ""


def _load_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except csv.Error as exc:
        errors.append(f"{path.name} is invalid CSV: {exc}")
        return []

    if reader.fieldnames is None:
        errors.append(f"{path.name} has no header row")
        return []
    if not rows:
        errors.append(f"{path.name} has no data rows")
    return rows


def _validate_required_files(run_dir: Path, errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        path = run_dir / relative
        if not path.is_file():
            errors.append(f"missing required artifact: {relative}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"required artifact is empty: {relative}")
            continue
        if _contains_placeholder(_read_text(path)[:8192]):
            errors.append(f"required artifact contains placeholder markers: {relative}")


def _validate_metadata(run_dir: Path, errors: list[str], warnings: list[str]) -> None:
    path = run_dir / "metadata.json"
    if not path.is_file():
        return
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"metadata.json is invalid JSON: {exc}")
        return
    if not isinstance(metadata, dict):
        errors.append("metadata.json must contain a JSON object")
        return

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata or not _has_value(metadata[field]):
            errors.append(f"metadata.json missing required field: {field}")

    if str(metadata.get("cuda", "")).lower() in {"false", "none", "unavailable", "cpu"}:
        errors.append("metadata.json cuda must describe a native CUDA environment")
    if str(metadata.get("gpu", "")).lower() in {"", "none", "cpu", "mps"}:
        errors.append("metadata.json gpu must describe an NVIDIA GPU")
    if "mac" in str(metadata.get("gpu", "")).lower():
        errors.append("metadata.json gpu appears to describe a Mac-local run")
    if not isinstance(metadata.get("sequence_shapes", []), (list, dict)):
        warnings.append("metadata.json sequence_shapes should be a list or object")


def _validate_csv_artifact(
    run_dir: Path,
    relative: str,
    min_repeated_runs: int,
    errors: list[str],
) -> dict[str, object]:
    path = run_dir / relative
    if not path.is_file():
        return {"rows": 0, "covered_rows": [], "row_run_counts": {}}

    schema = CSV_SCHEMAS[relative]
    rows = _load_csv(path, errors)
    fieldnames = set(rows[0].keys()) if rows else set()
    missing_columns = [
        column for column in schema["required_columns"] if column not in fieldnames
    ]
    if "row" in missing_columns and any(alias in fieldnames for alias in ROW_COLUMN_ALIASES):
        missing_columns.remove("row")
    for column in missing_columns:
        errors.append(f"{relative} missing required column: {column}")

    covered_rows: set[str] = set()
    run_ids_by_row: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=2):
        raw_row_id = _row_value(row, ROW_COLUMN_ALIASES)
        canonical_row_id = _canonical_row_id(raw_row_id)
        if canonical_row_id is None:
            errors.append(f"{relative} row {index} has unknown row id: {raw_row_id!r}")
            continue
        covered_rows.add(canonical_row_id)
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            run_ids_by_row.setdefault(canonical_row_id, set()).add(run_id)
        for column in schema["required_columns"]:
            if column == "row":
                value = raw_row_id
            else:
                value = row.get(column, "")
            if not _has_value(value):
                errors.append(f"{relative} row {index} missing value for {column}")
        for column in schema["numeric_columns"]:
            if column in row and not _is_number(row[column]):
                errors.append(f"{relative} row {index} has non-numeric {column}")

    missing_rows = sorted(set(REQUIRED_ROW_IDS) - covered_rows)
    for row_id in missing_rows:
        errors.append(f"{relative} missing required row: {row_id}")

    if relative == "latency.csv":
        for row_id in REQUIRED_ROW_IDS:
            distinct_runs = len(run_ids_by_row.get(row_id, set()))
            if distinct_runs < min_repeated_runs:
                errors.append(
                    f"latency.csv row {row_id} has {distinct_runs} distinct run_id values; "
                    f"expected at least {min_repeated_runs}"
                )

    return {
        "rows": len(rows),
        "covered_rows": sorted(covered_rows),
        "row_run_counts": {key: len(value) for key, value in sorted(run_ids_by_row.items())},
    }


def _validate_decision(run_dir: Path, errors: list[str]) -> None:
    path = run_dir / "decision.md"
    if not path.is_file():
        return
    text = _read_text(path)
    lower = text.lower()
    if "promote" not in lower and "kill" not in lower:
        errors.append("decision.md must contain an explicit promote or kill decision")
    for marker in ["threshold", "rank-2", "quality"]:
        if marker not in lower:
            errors.append(f"decision.md missing required marker: {marker}")
    if "speed" not in lower and "memory" not in lower and "hbm" not in lower:
        errors.append("decision.md must discuss speed, memory, or HBM evidence")
    if "native" not in lower and "gpu" not in lower:
        errors.append("decision.md must identify the native GPU evidence basis")


def check_native_gpu_packet(
    run_dir: Path,
    min_repeated_runs: int = 3,
) -> dict[str, object]:
    """Return a JSON-serializable packet completeness result."""

    errors: list[str] = []
    warnings: list[str] = []
    run_dir = run_dir.resolve()
    csv_summaries: dict[str, object] = {}

    if not run_dir.exists():
        return {
            "status": "FAIL",
            "run_dir": str(run_dir),
            "errors": [f"run directory does not exist: {run_dir}"],
            "warnings": warnings,
        }
    if not run_dir.is_dir():
        errors.append(f"run path is not a directory: {run_dir}")

    _validate_required_files(run_dir, errors)
    _validate_metadata(run_dir, errors, warnings)
    for relative in CSV_SCHEMAS:
        csv_summaries[relative] = _validate_csv_artifact(
            run_dir, relative, min_repeated_runs, errors
        )
    _validate_decision(run_dir, errors)

    return {
        "status": "FAIL" if errors else "PASS",
        "run_dir": str(run_dir),
        "errors": errors,
        "warnings": warnings,
        "required_rows": REQUIRED_ROW_IDS,
        "csv_summaries": csv_summaries,
        "min_repeated_runs": min_repeated_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-repeated-runs", type=int, default=3)
    args = parser.parse_args()

    result = check_native_gpu_packet(args.run_dir, min_repeated_runs=args.min_repeated_runs)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
