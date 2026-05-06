"""Validate Mac-local gate result packets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BASE_REQUIRED_FILES = ("config.json", "raw_rows.jsonl", "summary.json", "decision.md")
REAL_REQUIRED_FILES = BASE_REQUIRED_FILES + ("summary.md",)
REQUIRED_SUMMARY_FIELDS = ("seed", "surface", "decision", "rows", "claim_boundary")
REAL_CONFIG_FIELDS = (
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "prompt_source",
    "prompt_ids_hash",
    "seed_list",
    "context_lengths",
    "dtype",
    "device",
    "command",
    "architecture_map_hash",
)
REAL_ROW_FIELDS = {
    "ssq_lr": (
        "model_id",
        "model_revision",
        "prompt_id",
        "layer",
        "position_bucket",
        "state_shape",
        "max_abs",
        "rms",
        "std",
        "kurtosis",
        "outlier_mass",
        "control_type",
    ),
    "horn": (
        "model_id",
        "prompt_id",
        "layer_left",
        "layer_right",
        "direction",
        "boundary_index",
        "pre_norm_position",
        "post_norm_position",
        "max_abs",
        "rms",
        "kurtosis",
        "control_type",
    ),
    "hbsm": (
        "model_id",
        "layer",
        "boundary_flag",
        "precision_perturbation",
        "kl_or_nll_drift",
        "cheap_predictor",
        "parameter_count",
        "weight_norm",
        "top_decile_flag",
        "random_top_decile",
        "train_test_split",
        "control_type",
    ),
}
REAL_CONTROL_VALUES = {
    "ssq_lr": {"bf16_no_quant"},
    "horn": {"boundary", "non_boundary", "permuted_direction"},
    "hbsm": {"perturbation_off", "random_flags", "layer_index", "parameter_count_norm", "boundary_only"},
}
SSQ_LR_POSITION_BUCKETS = {"prefill_end", "2k_or_end", "8k_or_end", "final_minus_128"}


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _missing_fields(row: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if field not in row]


def _infer_mode(summary: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    surface = str(summary.get("surface", ""))
    return "synthetic" if surface.startswith("synthetic") else "real"


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_finite_fields(
    *,
    project: str,
    row_index: int,
    row: dict[str, Any],
    fields: tuple[str, ...],
    errors: list[str],
) -> None:
    for field in fields:
        if not _finite_number(row.get(field)):
            errors.append(f"{project} row {row_index} {field} must be finite numeric")


def _validate_nonnegative_fields(
    *,
    project: str,
    row_index: int,
    row: dict[str, Any],
    fields: tuple[str, ...],
    errors: list[str],
) -> None:
    for field in fields:
        value = row.get(field)
        if _finite_number(value) and float(value) < 0.0:
            errors.append(f"{project} row {row_index} {field} must be nonnegative")


def _valid_positive_int_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    )


def _validate_real_coverage(
    *,
    project: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    model_id = str(config.get("model_id", "")).strip()
    mismatched = [
        index for index, row in enumerate(rows) if str(row.get("model_id", "")).strip() != model_id
    ]
    if mismatched:
        errors.append("row model_id values must match config.json model_id")

    if project == "ssq_lr":
        buckets = {str(row.get("position_bucket")) for row in rows}
        missing_buckets = SSQ_LR_POSITION_BUCKETS - buckets
        if missing_buckets:
            errors.append(
                "ssq_lr real packet missing position buckets: "
                + ", ".join(sorted(missing_buckets))
            )
        prompt_ids = {str(row.get("prompt_id")) for row in rows}
        if len(prompt_ids) < 12 and "resource_limit_note" not in config:
            errors.append(
                "ssq_lr real packet needs at least 12 distinct prompt_id values "
                "or config.json resource_limit_note"
            )
        for index, row in enumerate(rows):
            _validate_finite_fields(
                project="ssq_lr",
                row_index=index,
                row=row,
                fields=("max_abs", "rms", "std", "kurtosis", "outlier_mass"),
                errors=errors,
            )
            _validate_nonnegative_fields(
                project="ssq_lr",
                row_index=index,
                row=row,
                fields=("max_abs", "rms", "std"),
                errors=errors,
            )
            outlier_mass = row.get("outlier_mass")
            if _finite_number(outlier_mass) and not 0.0 <= float(outlier_mass) <= 1.0:
                errors.append(f"ssq_lr row {index} outlier_mass must be in [0, 1]")
            if not _valid_positive_int_list(row.get("state_shape")):
                errors.append(f"ssq_lr row {index} state_shape must be a non-empty positive integer list")

    if project == "horn":
        prompt_ids = {str(row.get("prompt_id")) for row in rows}
        if len(prompt_ids) < 12 and "resource_limit_note" not in config:
            errors.append(
                "horn real packet needs at least 12 distinct prompt_id values "
                "or config.json resource_limit_note"
            )
        boundary_rows = [row for row in rows if str(row.get("control_type")) == "boundary"]
        boundary_directions = {str(row.get("direction")) for row in boundary_rows}
        missing_directions = {"attention->ssm", "ssm->attention"} - boundary_directions
        if missing_directions:
            errors.append(
                "horn real packet missing boundary directions: "
                + ", ".join(sorted(missing_directions))
            )
        boundary_by_key = {
            (row.get("layer_left"), row.get("layer_right"), row.get("boundary_index")): str(row.get("direction"))
            for row in boundary_rows
        }
        for row in rows:
            if str(row.get("control_type")) != "permuted_direction":
                continue
            key = (row.get("layer_left"), row.get("layer_right"), row.get("boundary_index"))
            original = boundary_by_key.get(key)
            if original is None:
                errors.append("horn permuted_direction row must match an observed boundary tuple")
            elif str(row.get("direction")) == original:
                errors.append("horn permuted_direction row must flip the observed boundary direction")
        for index, row in enumerate(rows):
            _validate_finite_fields(
                project="horn",
                row_index=index,
                row=row,
                fields=("max_abs", "rms", "kurtosis"),
                errors=errors,
            )
            _validate_nonnegative_fields(
                project="horn",
                row_index=index,
                row=row,
                fields=("max_abs", "rms"),
                errors=errors,
            )
            boundary_index = row.get("boundary_index")
            if not isinstance(boundary_index, int) or isinstance(boundary_index, bool):
                errors.append(f"horn row {index} boundary_index must be an integer")

    if project == "hbsm":
        flags = {bool(row.get("boundary_flag")) for row in rows}
        if flags != {False, True}:
            errors.append("hbsm real packet needs both boundary_flag=true and boundary_flag=false")
        splits = {str(row.get("train_test_split")) for row in rows}
        if not {"train", "test"}.issubset(splits) and "resource_limit_note" not in config:
            errors.append("hbsm real packet needs both train and test split rows or config.json resource_limit_note")
        top_decile_count = sum(1 for row in rows if row.get("top_decile_flag") is True)
        random_top_decile_count = sum(1 for row in rows if row.get("random_top_decile") is True)
        if top_decile_count != random_top_decile_count:
            errors.append("hbsm random_top_decile true count must match top_decile_flag true count")
        for index, row in enumerate(rows):
            for field in ("kl_or_nll_drift", "cheap_predictor", "parameter_count", "weight_norm"):
                if not _finite_number(row.get(field)):
                    errors.append(f"hbsm row {index} {field} must be finite numeric")
            _validate_nonnegative_fields(
                project="hbsm",
                row_index=index,
                row=row,
                fields=("parameter_count", "weight_norm"),
                errors=errors,
            )
            for field in ("top_decile_flag", "random_top_decile"):
                if not isinstance(row.get(field), bool):
                    errors.append(f"hbsm row {index} {field} must be boolean")
            drift = row.get("kl_or_nll_drift")
            if str(row.get("control_type")) == "perturbation_off" and _finite_number(drift) and abs(float(drift)) > 1e-6:
                errors.append("hbsm perturbation_off rows must have near-zero drift")


def validate_gate_packet(
    packet_dir: Path,
    *,
    expected_decision_prefix: str | None = None,
    mode: str = "auto",
    project: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    for name in BASE_REQUIRED_FILES:
        if not (packet_dir / name).exists():
            errors.append(f"missing {name}")

    summary: dict[str, Any] = {}
    config: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    resolved_mode = mode
    if (packet_dir / "config.json").exists():
        config = _load_json(packet_dir / "config.json")
    if (packet_dir / "summary.json").exists():
        summary = _load_json(packet_dir / "summary.json")
        resolved_mode = _infer_mode(summary, mode)
        for field in REQUIRED_SUMMARY_FIELDS:
            if field not in summary:
                errors.append(f"summary.json missing {field}")
        decision = str(summary.get("decision", ""))
        if expected_decision_prefix and not decision.startswith(expected_decision_prefix):
            errors.append(f"decision {decision!r} does not start with {expected_decision_prefix!r}")
        claim_boundary = [str(item) for item in summary.get("claim_boundary", [])]
        if str(summary.get("surface", "")).startswith("synthetic") and "synthetic-only" not in claim_boundary:
            errors.append("synthetic packet must include synthetic-only claim boundary")

    if resolved_mode == "real":
        for name in REAL_REQUIRED_FILES:
            if not (packet_dir / name).exists():
                errors.append(f"real packet missing {name}")
        for field in REAL_CONFIG_FIELDS:
            if field not in config:
                errors.append(f"config.json missing provenance field {field}")
        for boundary in ("synthetic-only", "not model evidence"):
            if boundary in [str(item) for item in summary.get("claim_boundary", [])]:
                errors.append(f"real packet cannot include claim boundary {boundary!r}")

    if (packet_dir / "raw_rows.jsonl").exists():
        with (packet_dir / "raw_rows.jsonl").open() as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw_rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    errors.append(f"raw_rows.jsonl line {line_number} is not JSON: {exc}")
        if not raw_rows:
            errors.append("raw_rows.jsonl contains no rows")

    summary_rows = summary.get("rows", [])
    if summary_rows and raw_rows and len(summary_rows) != len(raw_rows):
        errors.append("summary row count does not match raw_rows.jsonl")
    if resolved_mode == "real" and summary.get("row_count") != len(raw_rows):
        errors.append("real packet summary.json row_count must match raw_rows.jsonl")

    if resolved_mode == "real" and project:
        expected_fields = REAL_ROW_FIELDS.get(project)
        if expected_fields is None:
            errors.append(f"unknown real-packet project {project!r}")
        else:
            for index, row in enumerate(raw_rows):
                missing = _missing_fields(row, expected_fields)
                if missing:
                    errors.append(f"row {index} missing fields: {', '.join(missing)}")
            required_controls = REAL_CONTROL_VALUES.get(project, set())
            observed_controls = {str(row.get("control_type")) for row in raw_rows}
            missing_controls = required_controls - observed_controls
            if missing_controls:
                controls = ", ".join(sorted(missing_controls))
                errors.append(f"missing required controls: {controls}")
            _validate_real_coverage(project=project, rows=raw_rows, config=config, errors=errors)

    decision_text = (packet_dir / "decision.md").read_text() if (packet_dir / "decision.md").exists() else ""
    if summary and str(summary.get("decision")) not in decision_text:
        errors.append("decision.md does not contain summary decision")

    return {
        "packet_dir": str(packet_dir),
        "ok": not errors,
        "errors": errors,
        "row_count": len(raw_rows),
        "decision": summary.get("decision"),
        "surface": summary.get("surface"),
        "mode": resolved_mode,
        "project": project,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--expected-decision-prefix")
    parser.add_argument("--mode", choices=("auto", "synthetic", "real"), default="auto")
    parser.add_argument("--project", choices=sorted(REAL_ROW_FIELDS))
    args = parser.parse_args()
    report = validate_gate_packet(
        args.packet_dir,
        expected_decision_prefix=args.expected_decision_prefix,
        mode=args.mode,
        project=args.project,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
