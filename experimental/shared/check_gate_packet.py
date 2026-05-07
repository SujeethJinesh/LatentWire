"""Validate Mac-local gate result packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import torch

from experimental.shared.hybrid_gate_evaluators import (
    evaluate_hbsm_b1,
    evaluate_horn_h1,
    evaluate_ssq_lr_s1,
)
from experimental.shared.sensitivity_metrics import kurtosis, max_abs

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
TENSOR_PROVENANCE_FIELDS = (
    "tensor_name",
    "tensor_source_name",
    "tensor_storage_name",
    "tensor_sha256",
    "tensor_dtype",
    "tensor_shape",
)
REAL_ROW_FIELDS = {
    "ssq_lr": (
        "model_id",
        "model_revision",
        "prompt_id",
        "layer",
        "layer_kind",
        "position_bucket",
        "state_tensor_kind",
        "state_shape",
        "max_abs",
        "rms",
        "std",
        "kurtosis",
        "outlier_mass",
        "control_type",
        *TENSOR_PROVENANCE_FIELDS,
    ),
    "horn": (
        "model_id",
        "prompt_id",
        "prompt_cluster_id",
        "layer_left",
        "layer_right",
        "direction",
        "matched_boundary_direction",
        "boundary_index",
        "pre_norm_position",
        "post_norm_position",
        "max_abs",
        "rms",
        "kurtosis",
        "control_type",
        *TENSOR_PROVENANCE_FIELDS,
    ),
    "hbsm": (
        "model_id",
        "prompt_id",
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
    "hbsm": {
        "perturbation_off",
        "random_flags",
        "layer_index",
        "parameter_count_norm",
        "boundary_only",
        "kl_lens_rank",
        "activation_outlier",
    },
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
        "distribution_passing_layer_count",
        "required_passing_layer_count",
        "pass_fraction",
        "selected_s1_ratio",
        "selected_s1_ci_low",
        "holm_p_min",
        "magnitude_gate_pass",
        "distribution_effect_floor_pass",
        "distribution_gate_pass",
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
        "selected_h1_cluster_bootstrap_low",
        "max_abs_direction_ratio",
        "kurtosis_direction_ratio",
        "non_boundary_control_ratio",
        "permuted_direction_ratio",
        "non_boundary_direction_count",
        "permuted_direction_count",
        "support_fraction",
    ),
    "hbsm": (
        "gate_name",
        "gate_status",
        "gate_pass",
        "primary_row_count",
        "scoring_layer_count",
        "prompt_count",
        "expected_top_decile_count",
        "top_decile_count",
        "random_top_decile_count",
        "train_count",
        "test_count",
        "split_counts",
        "control_types",
        "boundary_top_decile_count",
        "non_boundary_top_decile_count",
        "boundary_random_top_decile_count",
        "non_boundary_random_top_decile_count",
        "boundary_top_decile_rate",
        "non_boundary_top_decile_rate",
        "boundary_random_top_decile_rate",
        "non_boundary_random_top_decile_rate",
        "boundary_top_decile_enrichment",
        "random_boundary_top_decile_enrichment",
        "fisher_p_boundary_top_decile",
        "fisher_p_random_boundary_top_decile",
        "cheap_predictor_spearman",
        "baseline_spearman",
        "cheap_predictor_margin_vs_best_baseline",
    ),
}
HASH_FIELDS = ("prompt_ids_hash", "architecture_map_hash")
TRACE_PLAN_HASH_FIELD = "trace_plan_hash"
RESOURCE_LIMITED_DECISION = "RESOURCE_LIMITED_NOT_PROMOTABLE"
SCHEMA_REHEARSAL_DECISION = "SCHEMA_REHEARSAL_NOT_PROMOTABLE"
ARCHITECTURE_MAPS_PATH = (
    Path(__file__).resolve().parent / "results/hybrid_architecture_maps_20260506/architecture_maps.json"
)
MODEL_ELIGIBILITY_PATH = (
    Path(__file__).resolve().parent / "results/hybrid_model_eligibility_20260506/raw_rows.jsonl"
)
TRACE_PLAN_CONFIG_PATH = (
    Path(__file__).resolve().parent / "results/hybrid_trace_plan_20260507/config.json"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _known_architecture_hashes() -> dict[str, tuple[str, str]]:
    try:
        maps = _load_json(ARCHITECTURE_MAPS_PATH)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(maps, list):
        return {}
    known: dict[str, tuple[str, str]] = {}
    for row in maps:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id", "")).strip()
        config_sha = str(row.get("config_sha256", "")).strip().lower()
        if model_id and re.fullmatch(r"[0-9a-f]{64}", config_sha):
            expected = (model_id, f"sha256:{config_sha}")
            known[model_id] = expected
            aliases = row.get("model_id_aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_text = str(alias).strip()
                    if alias_text:
                        known[alias_text] = expected
    return known


def _canonical_model_aliases() -> dict[str, str]:
    return {alias: canonical for alias, (canonical, _) in _known_architecture_hashes().items()}


def _known_model_revision_sets() -> dict[str, set[str]]:
    aliases = _canonical_model_aliases()
    known: dict[str, set[str]] = {}
    try:
        lines = MODEL_ELIGIBILITY_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return known
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        model_id = str(row.get("model_id", "")).strip()
        revision = str(row.get("sha", "")).strip()
        if not model_id or not re.fullmatch(r"[0-9a-f]{40}", revision):
            continue
        canonical = aliases.get(model_id, model_id)
        for key in {model_id, canonical}:
            known.setdefault(key, set()).add(revision)
    return known


def _known_trace_plan_hashes() -> dict[str, str]:
    try:
        config = _load_json(TRACE_PLAN_CONFIG_PATH)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    hashes = config.get("trace_plan_hashes", {}) if isinstance(config, dict) else {}
    if not isinstance(hashes, dict):
        return {}
    known: dict[str, str] = {}
    for project, value in hashes.items():
        digest = str(value).strip().lower()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            known[str(project)] = digest
    return known


def _load_trace_plan_rows(project: str, trace_plan_path: str | None = None) -> list[dict[str, Any]]:
    if trace_plan_path:
        path = Path(trace_plan_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        path = TRACE_PLAN_CONFIG_PATH.parent / f"{project}_trace_plan.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def _validate_tensor_provenance(
    *,
    project: str,
    row_index: int,
    row: dict[str, Any],
    errors: list[str],
) -> None:
    for field in ("tensor_name", "tensor_source_name", "tensor_storage_name", "tensor_dtype"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{project} row {row_index} {field} must be a non-empty string")
    storage_name = str(row.get("tensor_storage_name", ""))
    if storage_name and not storage_name.endswith(".pt"):
        errors.append(f"{project} row {row_index} tensor_storage_name must name a .pt tensor file")
    digest = row.get("tensor_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        errors.append(f"{project} row {row_index} tensor_sha256 must be sha256:<64 lowercase hex chars>")
    if not _valid_positive_int_list(row.get("tensor_shape")):
        errors.append(f"{project} row {row_index} tensor_shape must be a non-empty positive integer list")


def _load_tensor_artifact(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} did not load to a torch.Tensor")
    return value


def _rms(tensor: torch.Tensor) -> float:
    values = tensor.float()
    return float(torch.sqrt(torch.mean(values * values)))


def _outlier_mass(tensor: torch.Tensor) -> float:
    values = tensor.float().reshape(-1).abs()
    if values.numel() == 0:
        return 0.0
    threshold = values.mean() + 3.0 * values.std(unbiased=False)
    return float(torch.mean((values > threshold).float()))


def _close_enough(left: Any, right: float, *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    if not _finite_number(left) or not math.isfinite(right):
        return False
    return abs(float(left) - right) <= atol + rtol * abs(right)


def _validate_tensor_metrics(
    *,
    project: str,
    row_index: int,
    row: dict[str, Any],
    tensor: torch.Tensor,
    errors: list[str],
) -> None:
    tensor_shape = list(tensor.shape)
    if row.get("tensor_shape") != tensor_shape:
        errors.append(f"{project} row {row_index} tensor_shape does not match loaded tensor")
    if project == "ssq_lr" and row.get("state_shape") != tensor_shape:
        errors.append(f"{project} row {row_index} state_shape does not match loaded tensor")
    computed: dict[str, float] = {
        "max_abs": max_abs(tensor),
        "rms": _rms(tensor),
        "kurtosis": kurtosis(tensor),
    }
    if project == "ssq_lr":
        computed["std"] = float(torch.std(tensor.float(), unbiased=False))
        computed["outlier_mass"] = _outlier_mass(tensor)
    for field, expected in computed.items():
        if not _close_enough(row.get(field), expected):
            errors.append(f"{project} row {row_index} {field} does not match loaded tensor")


def _validate_tensor_artifacts(
    *,
    project: str,
    packet_dir: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    if project not in {"ssq_lr", "horn"} or _schema_rehearsal(config):
        return
    tensor_dir = packet_dir / "tensors"
    manifest_path = tensor_dir / "tensor_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"{project} real packet must include tensors/tensor_manifest.json")
        return
    try:
        manifest = _load_json(manifest_path)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"{project} tensors/tensor_manifest.json is invalid: {exc}")
        return
    if not isinstance(manifest, dict):
        errors.append(f"{project} tensors/tensor_manifest.json must be an object")
        return
    manifest_by_storage: dict[str, dict[str, Any]] = {}
    for value in manifest.values():
        if not isinstance(value, dict):
            continue
        storage_name = str(value.get("storage_name", "")).strip()
        if storage_name:
            manifest_by_storage[storage_name] = value
    for index, row in enumerate(rows):
        storage_name = str(row.get("tensor_storage_name", "")).strip()
        if not storage_name:
            continue
        storage_path = Path(storage_name)
        if storage_path.is_absolute() or ".." in storage_path.parts or storage_path.name != storage_name:
            errors.append(f"{project} row {index} tensor_storage_name must be a plain in-packet filename")
            continue
        tensor_path = tensor_dir / storage_name
        if not tensor_path.is_file():
            errors.append(f"{project} row {index} tensor artifact is missing: tensors/{storage_name}")
            continue
        actual_sha256 = _file_sha256(tensor_path)
        if actual_sha256 != str(row.get("tensor_sha256", "")).strip().lower():
            errors.append(f"{project} row {index} tensor_sha256 does not match tensors/{storage_name}")
        try:
            tensor = _load_tensor_artifact(tensor_path)
        except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            errors.append(f"{project} row {index} tensor artifact could not be loaded: {exc}")
            continue
        _validate_tensor_metrics(
            project=project,
            row_index=index,
            row=row,
            tensor=tensor,
            errors=errors,
        )
        manifest_row = manifest_by_storage.get(storage_name)
        if manifest_row is None:
            errors.append(f"{project} row {index} tensor_storage_name is absent from tensor_manifest.json")
            continue
        checks = [
            ("sha256", "tensor_sha256"),
            ("dtype", "tensor_dtype"),
            ("shape", "tensor_shape"),
        ]
        for manifest_field, row_field in checks:
            if manifest_row.get(manifest_field) != row.get(row_field):
                errors.append(
                    f"{project} row {index} {row_field} does not match tensor_manifest.json"
                )
        source_name = str(row.get("tensor_source_name", "")).strip()
        original_name = str(manifest_row.get("original_name", "")).strip()
        if source_name and original_name and source_name != original_name:
            errors.append(
                f"{project} row {index} tensor_source_name does not match tensor_manifest.json"
            )


def _validate_hbsm_source_artifact(
    *,
    packet_dir: Path,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    if _schema_rehearsal(config):
        return
    evidence_dir = packet_dir / "evidence"
    source_path = evidence_dir / "hbsm_row_packet.json"
    manifest_path = evidence_dir / "source_manifest.json"
    if not source_path.is_file():
        errors.append("hbsm real packet must include evidence/hbsm_row_packet.json")
        return
    if not manifest_path.is_file():
        errors.append("hbsm real packet must include evidence/source_manifest.json")
        return
    expected_digest = str(config.get("source_row_packet_sha256", "")).strip().lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest):
        errors.append("hbsm config.json source_row_packet_sha256 must be sha256:<64 lowercase hex chars>")
    actual_digest = _file_sha256(source_path)
    if expected_digest and actual_digest != expected_digest:
        errors.append("hbsm source_row_packet_sha256 does not match evidence/hbsm_row_packet.json")
    try:
        manifest = _load_json(manifest_path)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"hbsm evidence/source_manifest.json is invalid: {exc}")
        return
    source_manifest = manifest.get("source_row_packet") if isinstance(manifest, dict) else None
    if not isinstance(source_manifest, dict):
        errors.append("hbsm evidence/source_manifest.json must contain source_row_packet")
        return
    if source_manifest.get("storage_name") != "hbsm_row_packet.json":
        errors.append("hbsm source_manifest storage_name must be hbsm_row_packet.json")
    if source_manifest.get("sha256") != actual_digest:
        errors.append("hbsm source_manifest sha256 does not match evidence/hbsm_row_packet.json")


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


def _hbsm_aggregated_primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("control_type")) != "boundary_only":
            continue
        key = (str(row.get("model_id", "")), int(row["layer"]))
        grouped.setdefault(key, []).append(row)
    aggregated = []
    for (model_id, layer), layer_rows in sorted(grouped.items(), key=lambda item: item[0]):
        aggregated.append(
            {
                "model_id": model_id,
                "layer": layer,
                "boundary_flag": any(bool(row["boundary_flag"]) for row in layer_rows),
                "top_decile_flag": any(bool(row["top_decile_flag"]) for row in layer_rows),
                "random_top_decile": any(bool(row["random_top_decile"]) for row in layer_rows),
                "kl_or_nll_drift": sum(float(row["kl_or_nll_drift"]) for row in layer_rows) / len(layer_rows),
            }
        )
    return aggregated


def _hbsm_measured_top_decile_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    expected_top_decile_count = math.ceil(0.10 * len(rows)) if rows else 0
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["kl_or_nll_drift"]),
            str(row["model_id"]),
            int(row["layer"]),
        ),
    )
    return {
        (str(row["model_id"]), int(row["layer"]))
        for row in ranked[:expected_top_decile_count]
    }


def _horn_matched_direction(row: dict[str, Any]) -> str:
    control_type = str(row.get("control_type", ""))
    if control_type == "non_boundary":
        return str(row.get("matched_boundary_direction", "")).strip()
    return str(row.get("direction", "")).strip()


def _trace_plan_key(project: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if project == "ssq_lr":
        return (
            str(row.get("model_id", "")),
            str(row.get("prompt_id", "")),
            int(row.get("layer", -1)),
            str(row.get("position_bucket", "")),
        )
    if project == "horn":
        return (
            str(row.get("model_id", "")),
            str(row.get("prompt_id", "")),
            str(row.get("prompt_cluster_id", "")),
            str(row.get("control_type", "")),
            int(row.get("boundary_index", -9999)),
            int(row.get("layer_left", -9999)),
            int(row.get("layer_right", -9999)),
            str(row.get("direction", "")),
            str(row.get("matched_boundary_direction", "")),
            str(row.get("pre_norm_position", "")),
            str(row.get("post_norm_position", "")),
        )
    if project == "hbsm":
        return (
            str(row.get("model_id", "")),
            str(row.get("prompt_id", "")),
            str(row.get("control_type", "")),
            int(row.get("layer", -9999)),
            bool(row.get("boundary_flag")),
            str(row.get("precision_perturbation", "")),
            str(row.get("train_test_split", "")),
        )
    return tuple(sorted(row.items()))


def _validate_trace_plan_coverage(
    *,
    project: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    if _schema_rehearsal(config):
        return
    trace_plan_path = config.get("trace_plan_path")
    if not isinstance(trace_plan_path, str):
        errors.append("config.json trace_plan_path must cite the frozen trace-plan rows for real packets")
        return
    resolved_trace_plan_path = Path(trace_plan_path)
    if not resolved_trace_plan_path.is_absolute():
        resolved_trace_plan_path = REPO_ROOT / resolved_trace_plan_path
    if not _resource_limited(config):
        trace_plan_hash = str(config.get(TRACE_PLAN_HASH_FIELD, "")).strip().lower()
        try:
            actual_trace_plan_hash = _file_sha256(resolved_trace_plan_path)
        except OSError as exc:
            errors.append(f"trace plan rows unavailable for coverage validation: {exc}")
            return
        if trace_plan_hash and actual_trace_plan_hash != trace_plan_hash:
            errors.append(
                "config.json trace_plan_path must point to rows whose SHA-256 equals "
                "config.json trace_plan_hash for promotable real packets"
            )
            return
    try:
        planned_rows = _load_trace_plan_rows(
            project,
            trace_plan_path if isinstance(trace_plan_path, str) else None,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        errors.append(f"trace plan rows unavailable for coverage validation: {exc}")
        return
    model_id = str(config.get("model_id", "")).strip()
    planned_for_model = [row for row in planned_rows if str(row.get("model_id", "")).strip() == model_id]
    if not planned_for_model:
        errors.append("trace plan has no rows for config.json model_id")
        return
    expected = {_trace_plan_key(project, row) for row in planned_for_model}
    observed_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = _trace_plan_key(project, row)
        observed_counts[key] = observed_counts.get(key, 0) + 1
    observed = set(observed_counts)
    if any(count > 1 for count in observed_counts.values()):
        errors.append(f"{project} real packet has duplicate trace-plan row keys")
    if observed - expected:
        errors.append(f"{project} real packet contains rows outside the frozen trace plan")
    if expected - observed and not _resource_limited(config):
        errors.append(f"{project} real packet is missing rows from the frozen trace plan")


def _resource_limited(config: dict[str, Any]) -> bool:
    return "resource_limit_note" in config


def _schema_rehearsal(config: dict[str, Any]) -> bool:
    return config.get("schema_rehearsal") is True


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


def _validate_schema_rehearsal_decision(
    *,
    project: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    errors: list[str],
) -> None:
    if not _schema_rehearsal(config):
        return
    decision = str(summary.get("decision", ""))
    if not decision.startswith(SCHEMA_REHEARSAL_DECISION):
        errors.append(
            f"{project} schema-rehearsal packet must use {SCHEMA_REHEARSAL_DECISION} decision"
        )


def _expected_gate_status(project: str, rows: list[dict[str, Any]]) -> str | None:
    if project == "ssq_lr":
        return str(evaluate_ssq_lr_s1(rows).get("gate_status"))
    if project == "horn":
        return str(evaluate_horn_h1(rows).get("gate_status"))
    if project == "hbsm":
        return str(evaluate_hbsm_b1(rows).get("gate_status"))
    return None


def _validate_real_decision(
    *,
    project: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    summary: dict[str, Any],
    errors: list[str],
) -> None:
    if _schema_rehearsal(config):
        return
    gate_status = _expected_gate_status(project, rows)
    if gate_status is None:
        return
    decision = str(summary.get("decision", ""))
    if _resource_limited(config):
        expected = f"{RESOURCE_LIMITED_DECISION}_{gate_status}"
    else:
        expected = gate_status
    if decision != expected:
        errors.append(f"{project} summary decision must equal recomputed gate decision {expected!r}")


def _validate_hash_provenance(
    config: dict[str, Any],
    *,
    project: str | None,
    errors: list[str],
) -> None:
    for field in HASH_FIELDS:
        value = config.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value):
            errors.append(f"config.json {field} must be a sha256:<64-hex-digest> value")
    if not _schema_rehearsal(config):
        value = config.get(TRACE_PLAN_HASH_FIELD)
        if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value):
            errors.append(
                f"config.json {TRACE_PLAN_HASH_FIELD} must be a sha256:<64-hex-digest> value"
            )
        elif project:
            known = _known_trace_plan_hashes()
            if not known:
                errors.append("trace-plan provenance artifact is unavailable")
            else:
                expected = known.get(project)
                if expected is None:
                    errors.append("project must match a trace plan in the shared trace-plan artifact")
                elif value.lower() != expected:
                    errors.append(
                        "config.json trace_plan_hash must match shared trace-plan hash for project"
                    )


def _validate_architecture_map_provenance(config: dict[str, Any], errors: list[str]) -> None:
    if _schema_rehearsal(config):
        return
    known = _known_architecture_hashes()
    model_id = str(config.get("model_id", "")).strip()
    architecture_map_hash = str(config.get("architecture_map_hash", "")).strip().lower()
    if not known:
        errors.append("architecture map provenance artifact is unavailable")
        return
    resolved = known.get(model_id)
    if resolved is None:
        errors.append("config.json model_id must match a model in the shared architecture map artifact")
    elif architecture_map_hash != resolved[1]:
        errors.append(
            "config.json architecture_map_hash must match shared architecture map config hash for model_id"
        )


def _validate_model_revision_provenance(config: dict[str, Any], errors: list[str]) -> None:
    if _schema_rehearsal(config):
        return
    known = _known_model_revision_sets()
    if not known:
        errors.append("model eligibility revision artifact is unavailable")
        return
    identity = str(config.get("served_model_id") or config.get("model_id", "")).strip()
    canonical = str(config.get("model_id", "")).strip()
    allowed = set(known.get(identity, set())) | set(known.get(canonical, set()))
    if not allowed:
        errors.append("config.json model_id/served_model_id must match a model in the eligibility artifact")
        return
    model_revision = str(config.get("model_revision", "")).strip()
    tokenizer_revision = str(config.get("tokenizer_revision", "")).strip()
    if model_revision not in allowed:
        errors.append("config.json model_revision must match the registered eligibility snapshot sha")
    if tokenizer_revision not in allowed:
        errors.append("config.json tokenizer_revision must match the registered eligibility snapshot sha")


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
        for field in ("ssm_layer_count", "passing_layer_count", "distribution_passing_layer_count"):
            if not _valid_nonnegative_int(summary.get(field)):
                errors.append(f"ssq_lr summary {field} must be a nonnegative integer")
        ssm_layer_count = summary.get("ssm_layer_count")
        passing_layer_count = summary.get("passing_layer_count")
        distribution_passing_layer_count = summary.get("distribution_passing_layer_count")
        if isinstance(ssm_layer_count, int) and ssm_layer_count <= 0:
            errors.append("ssq_lr summary ssm_layer_count must be positive")
        if (
            isinstance(ssm_layer_count, int)
            and isinstance(passing_layer_count, int)
            and passing_layer_count > ssm_layer_count
        ):
            errors.append("ssq_lr summary passing_layer_count cannot exceed ssm_layer_count")
        if (
            isinstance(ssm_layer_count, int)
            and isinstance(distribution_passing_layer_count, int)
            and distribution_passing_layer_count > ssm_layer_count
        ):
            errors.append("ssq_lr summary distribution_passing_layer_count cannot exceed ssm_layer_count")
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
            "magnitude_gate_pass",
            "distribution_effect_floor_pass",
            "distribution_gate_pass",
            "required_passing_layer_count",
            "selected_s1_ratio",
        ):
            if summary.get(field) != evaluated.get(field):
                errors.append(f"ssq_lr summary {field} must match evaluator output")
        for field in (
            "passing_layer_count",
            "distribution_passing_layer_count",
            "pass_fraction",
            "selected_s1_ci_low",
            "holm_p_min",
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
        cluster_low = summary.get("selected_h1_cluster_bootstrap_low")
        if not _finite_number(cluster_low) or float(cluster_low) <= 0.0:
            errors.append("horn summary selected_h1_cluster_bootstrap_low must be positive finite numeric")
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
            "selected_h1_cluster_bootstrap_low",
            "max_abs_direction_ratio",
            "kurtosis_direction_ratio",
            "non_boundary_control_ratio",
            "permuted_direction_ratio",
            "non_boundary_direction_count",
            "permuted_direction_count",
            "support_fraction",
        ):
            value = summary.get(field)
            expected = evaluated.get(field)
            if _finite_number(value) and _finite_number(expected):
                if abs(float(value) - float(expected)) > 1e-9:
                    errors.append(f"horn summary {field} must match evaluator output")

    if project == "hbsm":
        evaluated = evaluate_hbsm_b1(rows)
        for field in (
            "primary_row_count",
            "scoring_layer_count",
            "prompt_count",
            "expected_top_decile_count",
            "top_decile_count",
            "random_top_decile_count",
            "train_count",
            "test_count",
        ):
            if not _valid_nonnegative_int(summary.get(field)):
                errors.append(f"hbsm summary {field} must be a nonnegative integer")
        enrichment = summary.get("boundary_top_decile_enrichment")
        if not _finite_number(enrichment) or float(enrichment) < 0.0:
            errors.append("hbsm summary boundary_top_decile_enrichment must be nonnegative finite numeric")
        random_enrichment = summary.get("random_boundary_top_decile_enrichment")
        if not _finite_number(random_enrichment) or float(random_enrichment) < 0.0:
            errors.append("hbsm summary random_boundary_top_decile_enrichment must be nonnegative finite numeric")
        for field in (
            "gate_status",
            "gate_pass",
            "primary_row_count",
            "scoring_layer_count",
            "prompt_count",
            "expected_top_decile_count",
            "top_decile_count",
            "random_top_decile_count",
            "train_count",
            "test_count",
            "control_types",
            "split_counts",
            "baseline_spearman",
        ):
            if summary.get(field) != evaluated.get(field):
                errors.append(f"hbsm summary {field} must match evaluator output")
        for field in (
            "boundary_top_decile_count",
            "non_boundary_top_decile_count",
            "boundary_random_top_decile_count",
            "non_boundary_random_top_decile_count",
            "boundary_top_decile_rate",
            "non_boundary_top_decile_rate",
            "boundary_random_top_decile_rate",
            "non_boundary_random_top_decile_rate",
            "boundary_top_decile_enrichment",
            "random_boundary_top_decile_enrichment",
            "fisher_p_boundary_top_decile",
            "fisher_p_random_boundary_top_decile",
            "cheap_predictor_spearman",
            "cheap_predictor_margin_vs_best_baseline",
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
            _validate_tensor_provenance(project="ssq_lr", row_index=index, row=row, errors=errors)
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
            if row.get("state_shape") != row.get("tensor_shape"):
                errors.append(f"ssq_lr row {index} state_shape must match tensor_shape")
            layer_kind = str(row.get("layer_kind", "")).strip().lower()
            if layer_kind not in {"ssm", "mamba", "mamba2", "ssm_recurrent"}:
                errors.append(
                    f"ssq_lr row {index} layer_kind must identify an SSM/Mamba recurrent layer"
                )
            state_tensor_kind = str(row.get("state_tensor_kind", "")).strip().lower()
            if state_tensor_kind not in {
                "recurrent_ssm_state",
                "ssm_recurrent_state",
                "mamba_recurrent_state",
                "mamba2_recurrent_state",
            }:
                errors.append(
                    f"ssq_lr row {index} state_tensor_kind must identify recurrent SSM state"
                )

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
        boundary_prompt_directions: dict[str, set[str]] = {}
        for row in boundary_rows:
            boundary_prompt_directions.setdefault(str(row.get("prompt_id")), set()).add(
                str(row.get("direction"))
            )
        incomplete_boundary_prompts = [
            prompt_id
            for prompt_id, observed in boundary_prompt_directions.items()
            if not {"attention->ssm", "ssm->attention"}.issubset(observed)
        ]
        if incomplete_boundary_prompts:
            errors.append("horn boundary rows must include both directions for every prompt")
        for control_type in ("non_boundary", "permuted_direction"):
            control_directions = {
                _horn_matched_direction(row)
                for row in rows
                if str(row.get("control_type")) == control_type
            }
            if not {"attention->ssm", "ssm->attention"}.issubset(control_directions):
                errors.append(
                    f"horn {control_type} controls must match both attention->ssm and ssm->attention boundary directions"
                )
        non_boundary_prompt_directions: dict[str, set[str]] = {}
        for row in rows:
            if str(row.get("control_type")) == "non_boundary":
                non_boundary_prompt_directions.setdefault(str(row.get("prompt_id")), set()).add(
                    _horn_matched_direction(row)
                )
        incomplete_non_boundary_prompts = [
            prompt_id
            for prompt_id in boundary_prompt_directions
            if not {"attention->ssm", "ssm->attention"}.issubset(
                non_boundary_prompt_directions.get(prompt_id, set())
            )
        ]
        if incomplete_non_boundary_prompts:
            errors.append("horn non_boundary controls must match both boundary directions for every prompt")
        boundary_by_key = {
            (
                row.get("prompt_id"),
                row.get("layer_left"),
                row.get("layer_right"),
                row.get("boundary_index"),
                row.get("pre_norm_position"),
                row.get("post_norm_position"),
            ): row
            for row in boundary_rows
        }
        permuted_keys = {
            (
                row.get("prompt_id"),
                row.get("layer_left"),
                row.get("layer_right"),
                row.get("boundary_index"),
                row.get("pre_norm_position"),
                row.get("post_norm_position"),
            )
            for row in rows
            if str(row.get("control_type")) == "permuted_direction"
        }
        if set(boundary_by_key) - permuted_keys:
            errors.append("horn every observed boundary tuple must have a paired permuted_direction row")
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
            else:
                actual_direction = str(row.get("direction", "")).strip()
                matched_direction = str(row.get("matched_boundary_direction", actual_direction)).strip()
                if actual_direction == str(original.get("direction")):
                    errors.append("horn permuted_direction row must flip the observed boundary direction")
                if matched_direction and matched_direction != actual_direction:
                    errors.append("horn permuted_direction matched_boundary_direction must equal flipped direction")
                source = str(row.get("tensor_source_name", "")).strip()
                alias = str(row.get("tensor_alias_of", "")).strip()
                original_source = str(original.get("tensor_source_name", "")).strip()
                if not alias:
                    errors.append("horn permuted_direction row must record tensor_alias_of")
                if source and original_source and source != original_source:
                    errors.append("horn permuted_direction row must reuse the observed boundary tensor source")
                if row.get("tensor_sha256") != original.get("tensor_sha256"):
                    errors.append("horn permuted_direction row must reuse the observed boundary tensor hash")
                for field in ("max_abs", "rms", "kurtosis"):
                    value = row.get(field)
                    original_value = original.get(field)
                    if _finite_number(value) and _finite_number(original_value):
                        if abs(float(value) - float(original_value)) > 1e-9:
                            errors.append(
                                "horn permuted_direction row must reuse the observed boundary metrics"
                            )
        for index, row in enumerate(rows):
            _validate_tensor_provenance(project="horn", row_index=index, row=row, errors=errors)
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
        primary_rows = [row for row in rows if str(row.get("control_type")) == "boundary_only"]
        flags = {row.get("boundary_flag") for row in primary_rows if isinstance(row.get("boundary_flag"), bool)}
        if flags != {False, True}:
            errors.append("hbsm real packet needs boundary_only rows with both boundary_flag=true and boundary_flag=false")
        prompt_ids = {str(row.get("prompt_id")) for row in primary_rows if row.get("prompt_id") is not None}
        if len(prompt_ids) < 12 and "resource_limit_note" not in config:
            errors.append(
                "hbsm real packet needs at least 12 distinct boundary_only prompt_id values "
                "or config.json resource_limit_note"
            )
        prompt_flags: dict[str, set[bool]] = {}
        for row in primary_rows:
            if isinstance(row.get("boundary_flag"), bool):
                prompt_flags.setdefault(str(row.get("prompt_id")), set()).add(bool(row.get("boundary_flag")))
        incomplete_prompts = [prompt_id for prompt_id, observed in prompt_flags.items() if observed != {False, True}]
        if incomplete_prompts:
            errors.append("hbsm real packet needs every boundary_only prompt to include boundary and non-boundary layers")
        splits = {str(row.get("train_test_split")) for row in primary_rows}
        if not {"train", "test"}.issubset(splits) and "resource_limit_note" not in config:
            errors.append("hbsm boundary_only rows need both train and test splits or config.json resource_limit_note")
        scoring_rows = _hbsm_aggregated_primary_rows(primary_rows)
        top_decile_count = sum(1 for row in scoring_rows if row.get("top_decile_flag") is True)
        random_top_decile_count = sum(1 for row in scoring_rows if row.get("random_top_decile") is True)
        expected_top_decile_count = math.ceil(0.10 * len(scoring_rows)) if scoring_rows else 0
        measured_top_keys = _hbsm_measured_top_decile_keys(scoring_rows)
        supplied_top_keys = {
            (str(row["model_id"]), int(row["layer"]))
            for row in scoring_rows
            if row.get("top_decile_flag") is True
        }
        if supplied_top_keys != measured_top_keys:
            errors.append("hbsm boundary_only top_decile_flag must match measured kl_or_nll_drift top decile")
        mismatched_prompt_top_flags = [
            (str(row.get("model_id")), int(row["layer"]), str(row.get("prompt_id")))
            for row in primary_rows
            if isinstance(row.get("top_decile_flag"), bool)
            and bool(row["top_decile_flag"]) != (
                (str(row.get("model_id")), int(row["layer"])) in measured_top_keys
            )
        ]
        if mismatched_prompt_top_flags:
            errors.append(
                "hbsm every boundary_only prompt row top_decile_flag must match measured kl_or_nll_drift top decile"
            )
        if top_decile_count != expected_top_decile_count:
            errors.append(
                "hbsm boundary_only top_decile_flag true count must equal ceil(10% of aggregated primary layers)"
            )
        if random_top_decile_count != expected_top_decile_count:
            errors.append(
                "hbsm boundary_only random_top_decile true count must equal ceil(10% of aggregated primary layers)"
            )
        scoring_keys = {(str(row["model_id"]), int(row["layer"])) for row in scoring_rows}
        for control_type in sorted(REAL_CONTROL_VALUES["hbsm"] - {"boundary_only"}):
            control_keys = {
                (str(row.get("model_id", "")), int(row["layer"]))
                for row in rows
                if str(row.get("control_type")) == control_type and "layer" in row
            }
            if control_keys != scoring_keys:
                errors.append(
                    f"hbsm {control_type} controls must cover the same model/layer scoring set as boundary_only"
                )
        allowed_perturbations = {
            "fp4_sim_weight_or_activation_perturbation",
            "mxfp4_e2m1",
            "perturbation_off",
            "random_flags",
            "layer_index",
            "parameter_count_norm",
            "kl_lens_rank",
            "activation_outlier",
        }
        for index, row in enumerate(rows):
            precision_perturbation = str(row.get("precision_perturbation", "")).strip()
            if precision_perturbation not in allowed_perturbations:
                errors.append(f"hbsm row {index} precision_perturbation must be a preregistered perturbation label")
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
        _validate_hash_provenance(config, project=project, errors=errors)
        _validate_architecture_map_provenance(config, errors)
        _validate_model_revision_provenance(config, errors)
        for boundary in ("synthetic-only", "not model evidence"):
            if (
                boundary in [str(item) for item in summary.get("claim_boundary", [])]
                and not _schema_rehearsal(config)
            ):
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
            missing_row_fields = False
            for index, row in enumerate(raw_rows):
                missing = _missing_fields(row, expected_fields)
                if missing:
                    missing_row_fields = True
                    errors.append(f"row {index} missing fields: {', '.join(missing)}")
            required_controls = REAL_CONTROL_VALUES.get(project, set())
            observed_controls = {str(row.get("control_type")) for row in raw_rows}
            missing_controls = required_controls - observed_controls
            if missing_controls:
                controls = ", ".join(sorted(missing_controls))
                errors.append(f"missing required controls: {controls}")
            unknown_controls = observed_controls - required_controls
            if unknown_controls:
                controls = ", ".join(sorted(unknown_controls))
                errors.append(f"unknown controls: {controls}")
            if not missing_row_fields:
                _validate_trace_plan_coverage(project=project, rows=raw_rows, config=config, errors=errors)
                _validate_tensor_artifacts(
                    project=project,
                    packet_dir=packet_dir,
                    rows=raw_rows,
                    config=config,
                    errors=errors,
                )
                if project == "hbsm":
                    _validate_hbsm_source_artifact(
                        packet_dir=packet_dir,
                        config=config,
                        errors=errors,
                    )
                _validate_real_coverage(project=project, rows=raw_rows, config=config, errors=errors)
                _validate_real_summary(project=project, rows=raw_rows, summary=summary, config=config, errors=errors)
                _validate_real_decision(
                    project=project,
                    rows=raw_rows,
                    config=config,
                    summary=summary,
                    errors=errors,
                )
            _validate_resource_limit_decision(project=project, config=config, summary=summary, errors=errors)
            _validate_schema_rehearsal_decision(project=project, config=config, summary=summary, errors=errors)

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
