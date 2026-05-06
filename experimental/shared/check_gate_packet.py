"""Validate Mac-local gate result packets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experimental.shared.hybrid_gate_evaluators import (
    evaluate_hbsm_b1,
    evaluate_horn_h1,
    evaluate_ssq_lr_s1,
)

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
REAL_SUMMARY_FIELDS = {
    "ssq_lr": (
        "gate_name",
        "gate_status",
        "gate_pass",
        "prompt_count",
        "position_buckets",
        "ssm_layer_count",
        "passing_layer_count",
        "required_passing_layer_count",
        "pass_fraction",
        "selected_s1_ratio",
        "selected_s1_ci_low",
        "holm_p_min",
        "max_abs_ratio_final_minus_128_vs_prefill_end",
        "std_ratio_final_minus_128_vs_prefill_end",
        "kurtosis_ratio_final_minus_128_vs_prefill_end",
    ),
    "horn": (
        "gate_name",
        "gate_status",
        "gate_pass",
        "prompt_count",
        "boundary_directions",
        "selected_h1_metric",
        "selected_h1_direction",
        "selected_h1_ratio",
        "selected_h1_threshold",
        "selected_h1_ci_low",
        "max_abs_direction_ratio",
        "kurtosis_direction_ratio",
        "non_boundary_control_ratio",
        "permuted_direction_ratio",
        "support_fraction",
    ),
    "hbsm": (
        "gate_name",
        "gate_status",
        "gate_pass",
        "top_decile_count",
        "random_top_decile_count",
        "train_count",
        "test_count",
        "split_counts",
        "control_types",
        "boundary_top_decile_count",
        "non_boundary_top_decile_count",
        "boundary_top_decile_rate",
        "non_boundary_top_decile_rate",
        "boundary_top_decile_enrichment",
        "fisher_p_boundary_top_decile",
        "cheap_predictor_spearman",
    ),
}
HASH_FIELDS = ("prompt_ids_hash", "architecture_map_hash")
RESOURCE_LIMITED_DECISION = "RESOURCE_LIMITED_NOT_PROMOTABLE"


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


def _valid_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_ratio_field(
    *,
    project: str,
    summary: dict[str, Any],
    field: str,
    errors: list[str],
) -> None:
    value = summary.get(field)
    if not _finite_number(value) or float(value) <= 0.0:
        errors.append(f"{project} summary {field} must be positive finite numeric")


def _resource_limited(config: dict[str, Any]) -> bool:
    return "resource_limit_note" in config


def _validate_resource_limit_decision(
    *,
    project: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    errors: list[str],
) -> None:
    if not _resource_limited(config):
        return
    decision = str(summary.get("decision", ""))
    if not decision.startswith(RESOURCE_LIMITED_DECISION):
        errors.append(
            f"{project} resource-limited real packet must use {RESOURCE_LIMITED_DECISION} decision"
        )


def _validate_hash_provenance(config: dict[str, Any], errors: list[str]) -> None:
    for field in HASH_FIELDS:
        value = config.get(field)
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) <= len("sha256:"):
            errors.append(f"config.json {field} must be a sha256:<hex-or-token> value")


def _validate_real_summary(
    *,
    project: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    required_fields = REAL_SUMMARY_FIELDS.get(project)
    if required_fields is None:
        return
    for field in required_fields:
        if field not in summary:
            errors.append(f"{project} summary.json missing {field}")

    if project == "ssq_lr":
        evaluated = evaluate_ssq_lr_s1(rows)
        prompt_count = summary.get("prompt_count")
        observed_prompt_count = len({str(row.get("prompt_id")) for row in rows})
        if prompt_count != observed_prompt_count:
            errors.append("ssq_lr summary prompt_count must match distinct row prompt_id count")
        if not _resource_limited(config) and isinstance(prompt_count, int) and prompt_count < 12:
            errors.append("ssq_lr summary prompt_count must be at least 12 unless resource-limited")
        position_buckets = summary.get("position_buckets")
        if not isinstance(position_buckets, list) or set(map(str, position_buckets)) != SSQ_LR_POSITION_BUCKETS:
            errors.append("ssq_lr summary position_buckets must match preregistered S1 buckets")
        for field in ("ssm_layer_count", "passing_layer_count"):
            if not _valid_nonnegative_int(summary.get(field)):
                errors.append(f"ssq_lr summary {field} must be a nonnegative integer")
        ssm_layer_count = summary.get("ssm_layer_count")
        passing_layer_count = summary.get("passing_layer_count")
        if isinstance(ssm_layer_count, int) and ssm_layer_count <= 0:
            errors.append("ssq_lr summary ssm_layer_count must be positive")
        if (
            isinstance(ssm_layer_count, int)
            and isinstance(passing_layer_count, int)
            and passing_layer_count > ssm_layer_count
        ):
            errors.append("ssq_lr summary passing_layer_count cannot exceed ssm_layer_count")
        pass_fraction = summary.get("pass_fraction")
        if not _finite_number(pass_fraction) or not 0.0 <= float(pass_fraction) <= 1.0:
            errors.append("ssq_lr summary pass_fraction must be in [0, 1]")
        selected_s1_ci_low = summary.get("selected_s1_ci_low")
        if not _finite_number(selected_s1_ci_low):
            errors.append("ssq_lr summary selected_s1_ci_low must be finite numeric")
        holm_p_min = summary.get("holm_p_min")
        if not _finite_number(holm_p_min) or not 0.0 <= float(holm_p_min) <= 1.0:
            errors.append("ssq_lr summary holm_p_min must be in [0, 1]")
        for field in (
            "max_abs_ratio_final_minus_128_vs_prefill_end",
            "std_ratio_final_minus_128_vs_prefill_end",
            "kurtosis_ratio_final_minus_128_vs_prefill_end",
        ):
            _validate_ratio_field(project="ssq_lr", summary=summary, field=field, errors=errors)
        for field in (
            "gate_status",
            "gate_pass",
            "required_passing_layer_count",
            "selected_s1_ratio",
        ):
            if summary.get(field) != evaluated.get(field):
                errors.append(f"ssq_lr summary {field} must match evaluator output")
        for field in (
            "passing_layer_count",
            "pass_fraction",
            "selected_s1_ci_low",
            "max_abs_ratio_final_minus_128_vs_prefill_end",
            "std_ratio_final_minus_128_vs_prefill_end",
            "kurtosis_ratio_final_minus_128_vs_prefill_end",
        ):
            value = summary.get(field)
            expected = evaluated.get(field)
            if _finite_number(value) and _finite_number(expected):
                if abs(float(value) - float(expected)) > 1e-9:
                    errors.append(f"ssq_lr summary {field} must match evaluator output")

    if project == "horn":
        evaluated = evaluate_horn_h1(rows)
        prompt_count = summary.get("prompt_count")
        observed_prompt_count = len({str(row.get("prompt_id")) for row in rows})
        if prompt_count != observed_prompt_count:
            errors.append("horn summary prompt_count must match distinct row prompt_id count")
        if not _resource_limited(config) and isinstance(prompt_count, int) and prompt_count < 12:
            errors.append("horn summary prompt_count must be at least 12 unless resource-limited")
        directions = summary.get("boundary_directions")
        if not isinstance(directions, list) or set(map(str, directions)) != {"attention->ssm", "ssm->attention"}:
            errors.append("horn summary boundary_directions must include attention->ssm and ssm->attention")
        _validate_ratio_field(project="horn", summary=summary, field="selected_h1_ratio", errors=errors)
        ci_low = summary.get("selected_h1_ci_low")
        if not _finite_number(ci_low) or float(ci_low) <= 0.0:
            errors.append("horn summary selected_h1_ci_low must be positive finite numeric")
        support_fraction = summary.get("support_fraction")
        if not _finite_number(support_fraction) or not 0.0 <= float(support_fraction) <= 1.0:
            errors.append("horn summary support_fraction must be in [0, 1]")
        for field in (
            "gate_status",
            "gate_pass",
            "selected_h1_metric",
            "selected_h1_direction",
            "selected_h1_threshold",
        ):
            if summary.get(field) != evaluated.get(field):
                errors.append(f"horn summary {field} must match evaluator output")
        for field in (
            "selected_h1_ratio",
            "selected_h1_ci_low",
            "max_abs_direction_ratio",
            "kurtosis_direction_ratio",
            "non_boundary_control_ratio",
            "permuted_direction_ratio",
            "support_fraction",
        ):
            value = summary.get(field)
            expected = evaluated.get(field)
            if _finite_number(value) and _finite_number(expected):
                if abs(float(value) - float(expected)) > 1e-9:
                    errors.append(f"horn summary {field} must match evaluator output")

    if project == "hbsm":
        evaluated = evaluate_hbsm_b1(rows)
        for field in ("top_decile_count", "random_top_decile_count", "train_count", "test_count"):
            if not _valid_nonnegative_int(summary.get(field)):
                errors.append(f"hbsm summary {field} must be a nonnegative integer")
        if summary.get("top_decile_count") != sum(1 for row in rows if row.get("top_decile_flag") is True):
            errors.append("hbsm summary top_decile_count must match rows")
        if summary.get("random_top_decile_count") != sum(1 for row in rows if row.get("random_top_decile") is True):
            errors.append("hbsm summary random_top_decile_count must match rows")
        if summary.get("train_count") != sum(1 for row in rows if str(row.get("train_test_split")) == "train"):
            errors.append("hbsm summary train_count must match rows")
        if summary.get("test_count") != sum(1 for row in rows if str(row.get("train_test_split")) == "test"):
            errors.append("hbsm summary test_count must match rows")
        enrichment = summary.get("boundary_top_decile_enrichment")
        if not _finite_number(enrichment) or float(enrichment) < 0.0:
            errors.append("hbsm summary boundary_top_decile_enrichment must be nonnegative finite numeric")
        for field in ("gate_status", "gate_pass", "control_types", "split_counts"):
            if summary.get(field) != evaluated.get(field):
                errors.append(f"hbsm summary {field} must match evaluator output")
        for field in (
            "boundary_top_decile_count",
            "non_boundary_top_decile_count",
            "boundary_top_decile_rate",
            "non_boundary_top_decile_rate",
            "boundary_top_decile_enrichment",
            "fisher_p_boundary_top_decile",
            "cheap_predictor_spearman",
        ):
            value = summary.get(field)
            expected = evaluated.get(field)
            if _finite_number(value) and _finite_number(expected):
                if abs(float(value) - float(expected)) > 1e-9:
                    errors.append(f"hbsm summary {field} must match evaluator output")


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
        buckets_by_prompt_layer: dict[tuple[str, Any], set[str]] = {}
        for row in rows:
            key = (str(row.get("prompt_id")), row.get("layer"))
            buckets_by_prompt_layer.setdefault(key, set()).add(str(row.get("position_bucket")))
        missing_matrix = [
            key for key, observed_buckets in buckets_by_prompt_layer.items() if observed_buckets != SSQ_LR_POSITION_BUCKETS
        ]
        if missing_matrix:
            errors.append("ssq_lr real packet needs every prompt/layer to cover every S1 bucket")
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
            (
                row.get("prompt_id"),
                row.get("layer_left"),
                row.get("layer_right"),
                row.get("boundary_index"),
                row.get("pre_norm_position"),
                row.get("post_norm_position"),
            ): str(row.get("direction"))
            for row in boundary_rows
        }
        for row in rows:
            if str(row.get("control_type")) != "permuted_direction":
                continue
            key = (
                row.get("prompt_id"),
                row.get("layer_left"),
                row.get("layer_right"),
                row.get("boundary_index"),
                row.get("pre_norm_position"),
                row.get("post_norm_position"),
            )
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
        flags = {row.get("boundary_flag") for row in rows if isinstance(row.get("boundary_flag"), bool)}
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
            for field in ("boundary_flag", "top_decile_flag", "random_top_decile"):
                if not isinstance(row.get(field), bool):
                    errors.append(f"hbsm row {index} {field} must be boolean")
            drift = row.get("kl_or_nll_drift")
            if str(row.get("control_type")) == "perturbation_off" and _finite_number(drift) and abs(float(drift)) > 1e-5:
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
        _validate_hash_provenance(config, errors)
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
            _validate_real_summary(project=project, rows=raw_rows, summary=summary, config=config, errors=errors)
            _validate_resource_limit_decision(project=project, config=config, summary=summary, errors=errors)

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
