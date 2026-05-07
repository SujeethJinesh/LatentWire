"""Check native HybridKernel profiler run artifacts for review completeness.

This verifier is intentionally about admissibility, not success. A passing run
directory has enough metadata, native profiler artifacts, and repeated reduced
metrics for a reviewer to inspect the promote/kill decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze


REQUIRED_FILES = [
    "metadata/environment.txt",
    "metadata/environment.json",
    "metadata/architecture_map.json",
    "metadata/model_provenance.json",
    "metadata/native_control_matrix.json",
    "metadata/profile_scope.json",
    "metadata/reduction_input_manifest.json",
    "profiler_metrics.json",
    "profiler_analysis_gate.json",
    "profiler_analysis_gate.md",
    "readout.md",
]

ENVIRONMENT_MARKERS = ["nvidia-smi", "nsys", "ncu", "python"]

READOUT_MARKERS = [
    "Distinct boundary conversion/materialization kernel?",
    "Boundary idle or launch gap?",
    "Extra DRAM/L2 traffic near boundary?",
    "End-to-end impact estimate clears 3%?",
    "Same-family controls available?",
    "Cross-family falsification attempted?",
]

READOUT_TEMPLATE_PLACEHOLDERS = [
    "kernel names and timestamps",
    "median and paired deltas",
    "ncu bytes vs matched controls",
    "formula and confidence interval",
    "model/control rows",
    "yes/no",
]

NSYS_PATTERNS = ["*.nsys-rep", "*.sqlite", "*.qdrep"]
NCU_PATTERNS = ["*.ncu-rep"]
MIN_NATIVE_ARTIFACT_BYTES = 1024
MIN_NATIVE_LOG_BYTES = 8
SKELETON_TODO_MARKER = "TODO_NATIVE_PROFILE_FILL"
PLACEHOLDER_ARTIFACT_MARKERS = [
    b"placeholder",
    b"native profiler export bytes",
    SKELETON_TODO_MARKER.lower().encode("utf-8"),
]
SQLITE_HEADER = b"SQLite format 3\x00"
SERVER_LOG_EVIDENCE_MARKERS = ["vllm", "nsys", "ncu", "cuda"]
CLIENT_LOG_EVIDENCE_MARKERS = ['"requests"', '"model"', '"status"']

ALLOWED_PROFILED_PROCESSES = {"vllm_server", "single_process_vllm_benchmark"}
PROFILED_PROCESS_FIELDS = ["profiled_process", "nsys_profiled_process", "ncu_profiled_process"]
NO_BOUNDARY_SIGNAL_MODE = "no_boundary_signal_kill"
NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL = "not_run_no_boundary_signal"
NO_BOUNDARY_SIGNAL_EVIDENCE_MARKERS = [
    "no suspicious boundary",
    "no boundary-local",
    "no boundary signal",
    "no distinct boundary",
]
REQUIRED_METRIC_PROVENANCE_FIELDS = [
    "row_role",
    "control_family",
    "boundary_direction",
    "nsys_artifact",
    "nsys_artifact_sha256",
    "ncu_artifact",
    "ncu_artifact_sha256",
    "kernel_names",
    "boundary_indices",
    "control_window_ids",
    "time_window_ms",
    "recoverable_fraction_basis",
    "reduction_command",
    "reduction_notes",
]
REQUIRED_REDUCTION_MANIFEST_FIELDS = [
    "run_id",
    "row_role",
    "model",
    "reduction_source_path",
    "source_nsys_artifact",
    "source_nsys_artifact_sha256",
    "source_time_window_ms",
    "source_ncu_artifact",
    "source_ncu_artifact_sha256",
    "reduction_command",
    "reduction_script_sha256",
    "reduction_notes",
]
REDUCTION_MANIFEST_VERSION = "hybridkernel_reduction_inputs_v1"
ALLOWED_ROW_ROLES = {"primary_hybrid", "same_family_control", "cross_family_falsification"}
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
DEFAULT_CROSS_FAMILY_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
ENVIRONMENT_PROVENANCE_VERSION = "hybridkernel_environment_v1"
MODEL_PROVENANCE_VERSION = "hybridkernel_model_provenance_v1"
REPLACEMENT_METADATA_PATH = "metadata/cross_family_control_replacement_template.json"
MAX_RECOVERABLE_FRACTION = 0.60
REQUIRED_REPLACEMENT_ROW_FIELDS = [
    "row_role",
    "model",
    "model_revision",
    "served_dtype",
    "control_family",
    "control_model_or_segment",
    "boundary_direction",
    "boundary_indices",
    "required_repeats",
    "request_shape",
    "feasibility_reason",
    "architecture_map_path",
    "architecture_map_sha256",
    "operator_initials",
    "preregistration_timestamp_utc",
]
REPLACEMENT_MATRIX_MATCH_FIELDS = [
    "model",
    "control_family",
    "control_model_or_segment",
    "boundary_direction",
]
MUTABLE_REVISION_NAMES = {"main", "master", "head", "latest", "dev", "trunk"}
REQUIRED_ENVIRONMENT_FIELDS = [
    "timestamp_utc",
    "hostname",
    "nvidia_smi",
    "nsys_version",
    "ncu_version",
    "python_version",
]
REQUIRED_ENVIRONMENT_PACKAGES = ["vllm", "torch", "triton", "transformers"]


def _matching_artifacts(root: Path, patterns: list[str]) -> list[Path]:
    artifacts: list[Path] = []
    for pattern in patterns:
        artifacts.extend(sorted(root.glob(pattern)))
    return artifacts


def _has_server_profiler_log(log_files: list[Path]) -> bool:
    return any(
        "server" in path.name.lower()
        and ("nsys" in path.name.lower() or "ncu" in path.name.lower())
        for path in log_files
    )


def _has_client_replay_log(log_files: list[Path]) -> bool:
    return any("client" in path.name.lower() for path in log_files)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_native_logs(log_files: list[Path], errors: list[str]) -> None:
    for log_file in log_files:
        name = log_file.name.lower()
        size = log_file.stat().st_size
        if size < MIN_NATIVE_LOG_BYTES:
            errors.append(
                f"profiling log is too small to be reviewable native evidence: "
                f"{log_file.name} has {size} bytes"
            )
        text = _read_text(log_file).lower()
        if SKELETON_TODO_MARKER.lower() in text:
            errors.append(f"profiling log still contains native run-packet TODO markers: {log_file.name}")
        if "placeholder" in text:
            errors.append(f"profiling log appears to be placeholder evidence: {log_file.name}")
        if "server" in name and ("nsys" in name or "ncu" in name):
            if not any(marker in text for marker in SERVER_LOG_EVIDENCE_MARKERS):
                errors.append(
                    f"server profiler log lacks Nsight/vLLM/CUDA evidence markers: {log_file.name}"
                )
        if "client" in name:
            if not all(marker in text for marker in CLIENT_LOG_EVIDENCE_MARKERS):
                errors.append(
                    f"client replay log lacks profiler_driver JSON evidence markers: {log_file.name}"
                )
            try:
                payload = json.loads(_read_text(log_file))
            except json.JSONDecodeError:
                errors.append(f"client replay log is not valid profiler_driver JSON: {log_file.name}")
            else:
                requests = payload.get("requests") if isinstance(payload, dict) else None
                model = payload.get("model") if isinstance(payload, dict) else None
                if not isinstance(model, str) or not model.strip():
                    errors.append(
                        f"client replay log JSON must contain non-empty top-level model: {log_file.name}"
                    )
                dry_run = payload.get("dry_run") if isinstance(payload, dict) else None
                if dry_run is not False:
                    errors.append(
                        f"client replay log JSON must contain dry_run=false for native evidence: "
                        f"{log_file.name}"
                    )
                if payload.get("token_counts_required") is not True:
                    errors.append(
                        f"client replay log JSON must contain token_counts_required=true for native evidence: "
                        f"{log_file.name}"
                    )
                token_count_source = payload.get("token_count_source")
                if not isinstance(token_count_source, str) or not token_count_source.strip():
                    errors.append(
                        f"client replay log JSON must contain non-empty token_count_source: {log_file.name}"
                    )
                if not isinstance(requests, list) or not requests:
                    errors.append(
                        f"client replay log JSON must contain non-empty requests list: {log_file.name}"
                    )
                elif not all(isinstance(row, dict) and row.get("status") for row in requests):
                    errors.append(
                        f"client replay log requests must contain status fields: {log_file.name}"
                    )
                elif not all(str(row.get("status")) == "ok" for row in requests):
                    errors.append(
                        f"client replay log requests must all have status=ok for native evidence: {log_file.name}"
                    )
                else:
                    for index, row in enumerate(requests):
                        prompt_counts = row.get("prompt_token_counts")
                        prompt_sha256 = row.get("prompt_sha256")
                        payload_sha256 = str(row.get("payload_sha256", "")).strip()
                        batch_size = row.get("batch_size")
                        prompt_total = row.get("prompt_token_count_total")
                        requested_decode = row.get("requested_decode_tokens")
                        expected_completion_tokens = row.get("expected_completion_tokens_total")
                        if not _positive_int(batch_size):
                            errors.append(
                                f"client replay request {index} must contain positive batch_size"
                            )
                        if (
                            not isinstance(prompt_sha256, list)
                            or not prompt_sha256
                            or not all(isinstance(value, str) and SHA256_PATTERN.match(value) for value in prompt_sha256)
                        ):
                            errors.append(
                                f"client replay request {index} must contain prompt_sha256 rows"
                            )
                        elif _positive_int(batch_size) and len(prompt_sha256) != batch_size:
                            errors.append(
                                f"client replay request {index} prompt_sha256 length must equal batch_size"
                            )
                        if not SHA256_PATTERN.match(payload_sha256):
                            errors.append(
                                f"client replay request {index} must contain payload_sha256"
                            )
                        if (
                            not isinstance(prompt_counts, list)
                            or not prompt_counts
                            or not all(_positive_int(value) for value in prompt_counts)
                        ):
                            errors.append(
                                f"client replay request {index} must contain positive prompt_token_counts"
                            )
                        elif _positive_int(batch_size) and len(prompt_counts) != batch_size:
                            errors.append(
                                f"client replay request {index} prompt_token_counts length must equal batch_size"
                            )
                        elif len(set(prompt_counts)) != 1:
                            errors.append(
                                f"client replay request {index} prompt_token_counts must be uniform for fixed-shape replay"
                            )
                        if not _positive_int(prompt_total):
                            errors.append(
                                f"client replay request {index} must contain positive prompt_token_count_total"
                            )
                        elif isinstance(prompt_counts, list) and all(
                            _positive_int(value) for value in prompt_counts
                        ) and prompt_total != sum(prompt_counts):
                            errors.append(
                                f"client replay request {index} prompt_token_count_total must equal sum(prompt_token_counts)"
                            )
                        if not _positive_int(requested_decode):
                            errors.append(
                                f"client replay request {index} must contain positive requested_decode_tokens"
                            )
                        if not _positive_int(expected_completion_tokens):
                            errors.append(
                                f"client replay request {index} must contain positive expected_completion_tokens_total"
                            )
                        if (
                            _positive_int(batch_size)
                            and _positive_int(requested_decode)
                        ):
                            computed_completion_tokens = batch_size * requested_decode
                            if (
                                _positive_int(expected_completion_tokens)
                                and expected_completion_tokens != computed_completion_tokens
                            ):
                                errors.append(
                                    f"client replay request {index} expected_completion_tokens_total "
                                    "must equal batch_size * requested_decode_tokens"
                                )
                        else:
                            computed_completion_tokens = None
                        response_usage = row.get("response_usage")
                        if not isinstance(response_usage, dict):
                            errors.append(
                                f"client replay request {index} must contain response_usage with completion_tokens"
                            )
                        else:
                            completion_tokens = response_usage.get("completion_tokens")
                            if not _positive_int(completion_tokens):
                                errors.append(
                                    f"client replay request {index} response_usage.completion_tokens must be positive"
                                )
                            elif (
                                isinstance(computed_completion_tokens, int)
                                and completion_tokens != computed_completion_tokens
                            ):
                                errors.append(
                                    f"client replay request {index} completion_tokens must equal "
                                    "batch_size * requested_decode_tokens"
                                )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _reject_skeleton_todo(path: Path, label: str, errors: list[str]) -> None:
    if path.is_file() and SKELETON_TODO_MARKER in _read_text(path):
        errors.append(f"{label} still contains native run-packet skeleton TODO markers")


def _is_unfilled_text(value: object) -> bool:
    text = str(value or "").strip()
    return (
        not text
        or SKELETON_TODO_MARKER in text
        or "TO_FILL" in text
        or "placeholder" in text.lower()
    )


def _validate_profiled_process(field: str, value: str, errors: list[str]) -> None:
    if value not in ALLOWED_PROFILED_PROCESSES:
        errors.append(
            f"profile_scope.json {field} must identify the profiled CUDA-serving process, "
            "not just an HTTP client"
        )


def _profile_scope_models(profile_scope: dict[str, object], errors: list[str]) -> set[str]:
    models: set[str] = set()
    top_level_model = str(profile_scope.get("model", "")).strip()
    if top_level_model:
        models.add(top_level_model)
    model_scopes = profile_scope.get("model_scopes")
    if model_scopes is None:
        return models
    if not isinstance(model_scopes, list) or not model_scopes:
        errors.append("profile_scope.json model_scopes must be a non-empty list when present")
        return models
    for index, scope in enumerate(model_scopes):
        if not isinstance(scope, dict):
            errors.append(f"profile_scope.json model_scopes[{index}] must be an object")
            continue
        model = str(scope.get("model", "")).strip()
        if not model:
            errors.append(f"profile_scope.json model_scopes[{index}].model must be non-empty")
        else:
            models.add(model)
        command = str(scope.get("vllm_command", "")).strip()
        if "vllm" not in command.lower():
            errors.append(
                f"profile_scope.json model_scopes[{index}].vllm_command must mention vLLM"
            )
        row_roles = scope.get("row_roles")
        if "row_role" in scope:
            errors.append(
                f"profile_scope.json model_scopes[{index}] must use row_roles list, not row_role string"
            )
        if not isinstance(row_roles, list) or not row_roles:
            errors.append(f"profile_scope.json model_scopes[{index}].row_roles must be a non-empty list")
        else:
            for role in row_roles:
                if role not in ALLOWED_ROW_ROLES:
                    errors.append(
                        f"profile_scope.json model_scopes[{index}].row_roles contains unknown role {role!r}"
                    )
    return models


def _profile_scope_model_roles(
    profile_scope: dict[str, object],
) -> set[tuple[str, str]]:
    roles: set[tuple[str, str]] = set()
    model_scopes = profile_scope.get("model_scopes")
    if not isinstance(model_scopes, list):
        return roles
    for scope in model_scopes:
        if not isinstance(scope, dict):
            continue
        model = str(scope.get("model", "")).strip()
        row_roles = scope.get("row_roles")
        if not model or not isinstance(row_roles, list):
            continue
        for role in row_roles:
            if isinstance(role, str) and role in ALLOWED_ROW_ROLES:
                roles.add((model, role))
    return roles


def _is_utf8_text_sample(sample: bytes) -> bool:
    if not sample:
        return True
    try:
        decoded = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if "\x00" in decoded:
        return False
    printable = sum(1 for char in decoded if char.isprintable() or char in "\r\n\t")
    return printable / max(len(decoded), 1) > 0.95


def _validate_profiler_export_bytes(
    *,
    path: Path,
    label: str,
    relative: Path | str,
    errors: list[str],
) -> None:
    head = path.read_bytes()[:4096]
    lowered = head.lower()
    if any(marker in lowered for marker in PLACEHOLDER_ARTIFACT_MARKERS):
        errors.append(f"{label} artifact appears to be a placeholder, not a native profiler export: {relative}")
    if path.suffix == ".sqlite":
        if not head.startswith(SQLITE_HEADER):
            errors.append(f"{label} SQLite artifact does not have a SQLite header: {relative}")
        return
    if _is_utf8_text_sample(head):
        errors.append(f"{label} artifact appears to be plain text, not a native profiler export: {relative}")


def _validate_native_artifacts(
    *,
    label: str,
    root: Path,
    patterns: list[str],
    min_bytes: int,
    errors: list[str],
) -> None:
    artifacts = _matching_artifacts(root, patterns)
    if not artifacts:
        errors.append(f"missing {label} artifact under {root.name}/")
        return

    for artifact in artifacts:
        size = artifact.stat().st_size
        if size < min_bytes:
            errors.append(
                f"{label} artifact is too small to be a reviewable native profiler export: "
                f"{artifact.relative_to(root.parent)} has {size} bytes, expected at least "
                f"{min_bytes}"
            )
        _validate_profiler_export_bytes(
            path=artifact,
            label=label,
            relative=artifact.relative_to(root.parent),
            errors=errors,
        )


def _validate_metric_artifact_path(
    *,
    row_index: int,
    field: str,
    value: str,
    expected_sha256: str,
    run_dir: Path,
    allowed_suffixes: set[str],
    require_native_artifacts: bool,
    min_bytes: int,
    packet_mode: str,
    errors: list[str],
) -> None:
    if (
        field == "ncu_artifact"
        and packet_mode == NO_BOUNDARY_SIGNAL_MODE
        and value == NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL
    ):
        if expected_sha256 != NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL:
            errors.append(
                f"metric row {row_index} ncu_artifact_sha256 must be "
                f"{NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL} when ncu_artifact is "
                f"{NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL}"
            )
        return
    artifact_path = Path(value)
    if artifact_path.is_absolute():
        errors.append(f"metric row {row_index} {field} must be relative to the run directory")
        return
    resolved = (run_dir / artifact_path).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError:
        errors.append(f"metric row {row_index} {field} must stay inside the run directory")
        return
    if resolved.suffix not in allowed_suffixes:
        errors.append(
            f"metric row {row_index} {field} must point to one of "
            f"{sorted(allowed_suffixes)}"
        )
    if not require_native_artifacts:
        return
    if not resolved.is_file():
        errors.append(f"metric row {row_index} {field} does not exist: {value}")
        return
    expected_sha256 = expected_sha256.strip().lower()
    if not SHA256_PATTERN.match(expected_sha256):
        errors.append(
            f"metric row {row_index} {field}_sha256 must be sha256:<64 lowercase hex chars>"
        )
    else:
        actual_sha256 = _file_sha256(resolved)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"metric row {row_index} {field}_sha256 mismatch for {value}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    size = resolved.stat().st_size
    if size < min_bytes:
        errors.append(
            f"metric row {row_index} {field} is too small to be native evidence: "
            f"{value} has {size} bytes, expected at least {min_bytes}"
        )
    before = len(errors)
    _validate_profiler_export_bytes(
        path=resolved,
        label=f"metric row {row_index} {field}",
        relative=value,
        errors=errors,
    )
    if len(errors) > before:
        return


def _is_pending_metric_row(raw: dict[str, object]) -> bool:
    measured_fields = [
        "total_step_ms",
        "attention_ssm_boundary_ms",
        "matched_non_boundary_ms",
        "recoverable_fraction",
    ]
    return all(raw.get(field) is None for field in measured_fields)


def _validate_metric_provenance(
    payload: dict[str, object],
    *,
    run_dir: Path,
    require_native_artifacts: bool,
    min_native_artifact_bytes: int,
    packet_mode: str,
    errors: list[str],
) -> set[str]:
    roles: set[str] = set()
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        errors.append("profiler_metrics.json rows must be a list")
        return roles
    for idx, row in enumerate(rows):
        if not isinstance(row, dict) or _is_pending_metric_row(row):
            continue
        missing = [field for field in REQUIRED_METRIC_PROVENANCE_FIELDS if field not in row]
        if missing:
            errors.append(f"metric row {idx} missing provenance fields: {', '.join(missing)}")
            continue
        row_role = str(row.get("row_role", "")).strip()
        roles.add(row_role)
        if row_role not in ALLOWED_ROW_ROLES:
            errors.append(
                f"metric row {idx} row_role must be one of {sorted(ALLOWED_ROW_ROLES)}"
            )
        recoverable_fraction = row.get("recoverable_fraction")
        if (
            not isinstance(recoverable_fraction, (int, float))
            or isinstance(recoverable_fraction, bool)
            or recoverable_fraction < 0
            or recoverable_fraction > MAX_RECOVERABLE_FRACTION
        ):
            errors.append(
                f"metric row {idx} recoverable_fraction must be in "
                f"[0, {MAX_RECOVERABLE_FRACTION:.2f}] unless a new gate is preregistered"
            )
        for field in [
            "control_family",
            "boundary_direction",
            "nsys_artifact",
            "ncu_artifact",
            "recoverable_fraction_basis",
            "reduction_command",
            "reduction_notes",
        ]:
            value = str(row.get(field, "")).strip()
            if _is_unfilled_text(value):
                errors.append(f"metric row {idx} {field} must be filled with non-placeholder text")
        nsys_value = str(row.get("nsys_artifact", "")).strip()
        nsys_sha256 = str(row.get("nsys_artifact_sha256", "")).strip()
        ncu_value = str(row.get("ncu_artifact", "")).strip()
        ncu_sha256 = str(row.get("ncu_artifact_sha256", "")).strip()
        if nsys_value and "TODO_NATIVE_PROFILE_FILL" not in nsys_value and "placeholder" not in nsys_value.lower():
            _validate_metric_artifact_path(
                row_index=idx,
                field="nsys_artifact",
                value=nsys_value,
                expected_sha256=nsys_sha256,
                run_dir=run_dir,
                allowed_suffixes={".nsys-rep", ".sqlite", ".qdrep"},
                require_native_artifacts=require_native_artifacts,
                min_bytes=min_native_artifact_bytes,
                packet_mode=packet_mode,
                errors=errors,
            )
        if ncu_value and "TODO_NATIVE_PROFILE_FILL" not in ncu_value and "placeholder" not in ncu_value.lower():
            _validate_metric_artifact_path(
                row_index=idx,
                field="ncu_artifact",
                value=ncu_value,
                expected_sha256=ncu_sha256,
                run_dir=run_dir,
                allowed_suffixes={".ncu-rep"},
                require_native_artifacts=require_native_artifacts,
                min_bytes=min_native_artifact_bytes,
                packet_mode=packet_mode,
                errors=errors,
            )
        kernel_names = row.get("kernel_names")
        if not isinstance(kernel_names, list) or not kernel_names or not all(
            isinstance(name, str) and name.strip() for name in kernel_names
        ):
            errors.append(f"metric row {idx} kernel_names must be a non-empty string list")
        boundary_indices = row.get("boundary_indices")
        if not isinstance(boundary_indices, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in boundary_indices
        ):
            errors.append(f"metric row {idx} boundary_indices must be a list of integers")
        elif row_role in {"primary_hybrid", "cross_family_falsification"} and not boundary_indices:
            errors.append(
                f"metric row {idx} {row_role} rows must name boundary_indices"
            )
        control_window_ids = row.get("control_window_ids")
        if not isinstance(control_window_ids, list) or not all(
            isinstance(value, str) and value.strip() and not _is_unfilled_text(value)
            for value in control_window_ids
        ):
            errors.append(f"metric row {idx} control_window_ids must be a list of stable non-placeholder strings")
        elif row_role == "same_family_control" and not control_window_ids:
            errors.append(f"metric row {idx} same_family_control rows must name control_window_ids")
        time_window = row.get("time_window_ms")
        if not isinstance(time_window, dict):
            errors.append(f"metric row {idx} time_window_ms must be an object")
        else:
            start = time_window.get("start")
            end = time_window.get("end")
            if not isinstance(start, (int, float)) or isinstance(start, bool):
                errors.append(f"metric row {idx} time_window_ms.start must be numeric")
            if not isinstance(end, (int, float)) or isinstance(end, bool):
                errors.append(f"metric row {idx} time_window_ms.end must be numeric")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end <= start:
                errors.append(f"metric row {idx} time_window_ms.end must exceed start")
        if packet_mode != NO_BOUNDARY_SIGNAL_MODE:
            selection = row.get("ncu_launch_selection")
            if not isinstance(selection, dict):
                errors.append(f"metric row {idx} ncu_launch_selection must be an object")
            else:
                kernel_regex = str(selection.get("kernel_regex", "")).strip()
                if not kernel_regex or "TODO_NATIVE_PROFILE_FILL" in kernel_regex:
                    errors.append(f"metric row {idx} ncu_launch_selection.kernel_regex must be filled")
                for field in ("launch_skip", "launch_count"):
                    value = selection.get(field)
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        errors.append(f"metric row {idx} ncu_launch_selection.{field} must be a nonnegative integer")
                if isinstance(selection.get("launch_count"), int) and selection.get("launch_count") == 0:
                    errors.append(f"metric row {idx} ncu_launch_selection.launch_count must be positive")
                source_artifact = str(selection.get("source_nsys_artifact", "")).strip()
                if source_artifact != nsys_value:
                    errors.append(
                        f"metric row {idx} ncu_launch_selection.source_nsys_artifact must match nsys_artifact"
                    )
                source_window = selection.get("source_time_window_ms")
                if source_window != time_window:
                    errors.append(
                        f"metric row {idx} ncu_launch_selection.source_time_window_ms must match time_window_ms"
                    )
                notes = str(selection.get("derivation_notes", "")).strip()
                if not notes or "TODO_NATIVE_PROFILE_FILL" in notes or "placeholder" in notes.lower():
                    errors.append(f"metric row {idx} ncu_launch_selection.derivation_notes must be filled")
    return roles


def _reduction_manifest_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("run_id", "")).strip(),
        str(row.get("row_role", "")).strip(),
        str(row.get("model", "")).strip(),
    )


def _validate_reduction_manifest_text(
    *,
    row_index: int,
    field: str,
    value: object,
    errors: list[str],
) -> str:
    text = str(value or "").strip()
    if not text or SKELETON_TODO_MARKER in text or "placeholder" in text.lower():
        errors.append(
            f"reduction_input_manifest.json row {row_index} {field} must be "
            "filled with non-placeholder text"
        )
    return text


def _resolve_packet_relative_path(run_dir: Path, value: str, label: str, errors: list[str]) -> Path | None:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label} must be a relative path inside the run packet")
        return None
    resolved = (run_dir / relative).resolve()
    if run_dir.resolve() not in (resolved, *resolved.parents):
        errors.append(f"{label} must stay inside the run packet")
        return None
    if not resolved.is_file():
        errors.append(f"{label} file does not exist inside run packet: {value}")
        return None
    text = _read_text(resolved)
    if SKELETON_TODO_MARKER in text or "TO_FILL" in text:
        errors.append(f"{label} file still contains unfilled template markers: {value}")
    return resolved


def _validate_reduction_input_manifest(
    *,
    path: Path,
    run_dir: Path,
    raw_rows: list[dict[str, object]],
    packet_mode: str,
    errors: list[str],
) -> None:
    if not path.is_file():
        return
    _reject_skeleton_todo(path, "metadata/reduction_input_manifest.json", errors)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"reduction_input_manifest.json is invalid: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("reduction_input_manifest.json must be a JSON object")
        return
    manifest_version = str(payload.get("manifest_version", "")).strip()
    if manifest_version != REDUCTION_MANIFEST_VERSION:
        errors.append(
            "reduction_input_manifest.json manifest_version must be "
            f"{REDUCTION_MANIFEST_VERSION!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("reduction_input_manifest.json must contain non-empty rows")
        return

    metric_by_key: dict[tuple[str, str, str], tuple[int, dict[str, object]]] = {}
    for metric_index, metric_row in enumerate(raw_rows):
        key = _reduction_manifest_key(metric_row)
        if not all(key):
            continue
        if key in metric_by_key:
            errors.append(
                "profiler_metrics.json has duplicate reduction manifest key "
                f"{key!r}; run_id, row_role, and model must identify one row"
            )
        else:
            metric_by_key[key] = (metric_index, metric_row)

    manifest_keys: set[tuple[str, str, str]] = set()
    for manifest_index, manifest_row in enumerate(rows):
        if not isinstance(manifest_row, dict):
            errors.append(f"reduction_input_manifest.json row {manifest_index} must be an object")
            continue
        missing = [
            field for field in REQUIRED_REDUCTION_MANIFEST_FIELDS if field not in manifest_row
        ]
        if missing:
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} missing fields: "
                + ", ".join(missing)
            )
            continue

        key = _reduction_manifest_key(manifest_row)
        if not all(key):
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} must contain non-empty "
                "run_id, row_role, and model"
            )
            continue
        if key in manifest_keys:
            errors.append(f"reduction_input_manifest.json duplicates row key {key!r}")
            continue
        manifest_keys.add(key)

        metric_entry = metric_by_key.get(key)
        if metric_entry is None:
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} does not match any "
                f"non-pending profiler_metrics.json row: {key!r}"
            )
            continue
        metric_index, metric_row = metric_entry

        for field in [
            "reduction_source_path",
            "source_nsys_artifact",
            "source_nsys_artifact_sha256",
            "source_ncu_artifact",
            "source_ncu_artifact_sha256",
            "reduction_command",
            "reduction_notes",
        ]:
            _validate_reduction_manifest_text(
                row_index=manifest_index,
                field=field,
                value=manifest_row.get(field),
                errors=errors,
            )

        reduction_source_path = str(manifest_row.get("reduction_source_path", "")).strip()
        source_path = _resolve_packet_relative_path(
            run_dir,
            reduction_source_path,
            f"reduction_input_manifest.json row {manifest_index} reduction_source_path",
            errors,
        )

        nsys_artifact = str(manifest_row.get("source_nsys_artifact", "")).strip()
        nsys_sha256 = str(manifest_row.get("source_nsys_artifact_sha256", "")).strip()
        if nsys_artifact != str(metric_row.get("nsys_artifact", "")).strip():
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} source_nsys_artifact "
                f"must match profiler_metrics.json row {metric_index} nsys_artifact"
            )
        if nsys_sha256 != str(metric_row.get("nsys_artifact_sha256", "")).strip():
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} source_nsys_artifact_sha256 "
                f"must match profiler_metrics.json row {metric_index} nsys_artifact_sha256"
            )
        elif not SHA256_PATTERN.match(nsys_sha256):
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} "
                "source_nsys_artifact_sha256 must be sha256:<64 lowercase hex chars>"
            )

        ncu_artifact = str(manifest_row.get("source_ncu_artifact", "")).strip()
        ncu_sha256 = str(manifest_row.get("source_ncu_artifact_sha256", "")).strip()
        metric_ncu_artifact = str(metric_row.get("ncu_artifact", "")).strip()
        metric_ncu_sha256 = str(metric_row.get("ncu_artifact_sha256", "")).strip()
        if ncu_artifact != metric_ncu_artifact:
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} source_ncu_artifact "
                f"must match profiler_metrics.json row {metric_index} ncu_artifact"
            )
        if ncu_sha256 != metric_ncu_sha256:
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} source_ncu_artifact_sha256 "
                f"must match profiler_metrics.json row {metric_index} ncu_artifact_sha256"
            )
        elif (
            packet_mode != NO_BOUNDARY_SIGNAL_MODE
            or ncu_sha256 != NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL
        ) and not SHA256_PATTERN.match(ncu_sha256):
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} "
                "source_ncu_artifact_sha256 must be sha256:<64 lowercase hex chars>"
            )

        source_window = manifest_row.get("source_time_window_ms")
        if source_window != metric_row.get("time_window_ms"):
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} "
                f"source_time_window_ms must match profiler_metrics.json row {metric_index} "
                "time_window_ms"
            )
        reduction_command = str(manifest_row.get("reduction_command", "")).strip()
        if reduction_command != str(metric_row.get("reduction_command", "")).strip():
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} reduction_command "
                f"must match profiler_metrics.json row {metric_index} reduction_command"
            )
        reduction_notes = str(manifest_row.get("reduction_notes", "")).strip()
        if reduction_notes != str(metric_row.get("reduction_notes", "")).strip():
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} reduction_notes "
                f"must match profiler_metrics.json row {metric_index} reduction_notes"
            )
        reduction_script_sha256 = str(manifest_row.get("reduction_script_sha256", "")).strip()
        if not SHA256_PATTERN.match(reduction_script_sha256):
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} "
                "reduction_script_sha256 must be sha256:<64 lowercase hex chars>"
            )
        elif source_path is not None and _file_sha256(source_path) != reduction_script_sha256:
            errors.append(
                f"reduction_input_manifest.json row {manifest_index} "
                "reduction_script_sha256 must match reduction_source_path file"
            )

    missing_keys = set(metric_by_key) - manifest_keys
    for key in sorted(missing_keys):
        errors.append(
            "reduction_input_manifest.json is missing the non-pending "
            f"profiler_metrics.json row {key!r}"
        )


def _validate_repeated_artifact_identity(
    *,
    metric_rows: list[dict[str, float | str]],
    raw_rows: list[dict[str, object]],
    min_repeated_runs: int,
    require_native_artifacts: bool,
    packet_mode: str,
    errors: list[str],
) -> None:
    if not require_native_artifacts:
        return
    by_config: dict[str, list[tuple[dict[str, float | str], dict[str, object]]]] = {}
    for reduced, raw in zip(metric_rows, raw_rows, strict=True):
        by_config.setdefault(str(reduced["config_key"]), []).append((reduced, raw))
    for config_key, config_rows in by_config.items():
        run_ids = {str(reduced["run_id"]) for reduced, _ in config_rows}
        if len(config_rows) < min_repeated_runs or len(run_ids) < min_repeated_runs:
            continue
        nsys_artifacts = {
            str(raw.get("nsys_artifact", "")).strip()
            for _, raw in config_rows
            if str(raw.get("nsys_artifact", "")).strip()
        }
        if len(nsys_artifacts) < min_repeated_runs:
            errors.append(
                f"same model/config group {config_key} reuses Nsight Systems artifacts; "
                f"expected at least {min_repeated_runs} distinct nsys_artifact paths"
            )
        if packet_mode != NO_BOUNDARY_SIGNAL_MODE:
            ncu_artifacts = {
                str(raw.get("ncu_artifact", "")).strip()
                for _, raw in config_rows
                if str(raw.get("ncu_artifact", "")).strip()
            }
            if len(ncu_artifacts) < min_repeated_runs:
                errors.append(
                    f"same model/config group {config_key} reuses Nsight Compute artifacts; "
                    f"expected at least {min_repeated_runs} distinct ncu_artifact paths"
                )
        windows = set()
        for _, raw in config_rows:
            time_window = raw.get("time_window_ms")
            if isinstance(time_window, dict):
                windows.add((time_window.get("start"), time_window.get("end")))
        if len(windows) < min_repeated_runs:
            errors.append(
                f"same model/config group {config_key} reuses profiler time windows; "
                f"expected at least {min_repeated_runs} distinct time_window_ms intervals"
            )


def _validate_global_artifact_identity(
    *,
    raw_rows: list[dict[str, object]],
    require_native_artifacts: bool,
    packet_mode: str,
    errors: list[str],
) -> None:
    if not require_native_artifacts:
        return
    artifact_fields = [("nsys_artifact", "Nsight Systems")]
    if packet_mode != NO_BOUNDARY_SIGNAL_MODE:
        artifact_fields.append(("ncu_artifact", "Nsight Compute"))
    for field, label in artifact_fields:
        seen: dict[str, int] = {}
        for idx, raw in enumerate(raw_rows):
            value = str(raw.get(field, "")).strip()
            if (
                not value
                or "TODO_NATIVE_PROFILE_FILL" in value
                or "placeholder" in value.lower()
            ):
                continue
            first_idx = seen.get(value)
            if first_idx is not None:
                errors.append(
                    f"metric rows {first_idx} and {idx} reuse the same {label} artifact "
                    f"`{value}`; native primary/control/falsification rows must cite "
                    "distinct profiler artifacts"
                )
            else:
                seen[value] = idx


def _models_from_architecture_map(path: Path, errors: list[str]) -> set[str]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"architecture_map.json is invalid: {exc}")
        return set()
    rows = payload if isinstance(payload, list) else payload.get("rows", []) if isinstance(payload, dict) else []
    models = {
        str(row.get("model") or row.get("model_id", "")).strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("model") or row.get("model_id", "")).strip()
    }
    if not models:
        errors.append("architecture_map.json must contain at least one model/model_id entry")
    return models


def _native_control_specs(path: Path, errors: list[str]) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"native_control_matrix.json is invalid: {exc}")
        return {}
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        errors.append("native_control_matrix.json must contain non-empty rows")
        return {}
    request_shape = payload.get("request_shape") if isinstance(payload, dict) else None
    if request_shape is not None and not isinstance(request_shape, dict):
        errors.append("native_control_matrix.json request_shape must be an object when present")
        request_shape = None
    if isinstance(request_shape, dict):
        for field in (
            "batch_size",
            "prefill_tokens",
            "decode_tokens",
            "requests",
            "dtype",
            "cuda_graph_enabled",
        ):
            if field not in request_shape:
                errors.append(f"native_control_matrix.json request_shape must include {field}")
    specs: dict[str, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"native_control_matrix.json row {index} must be an object")
            continue
        role = str(row.get("row_role", "")).strip()
        if role not in ALLOWED_ROW_ROLES:
            errors.append(f"native_control_matrix.json row {index} has invalid row_role")
            continue
        spec = dict(row)
        if request_shape is not None:
            spec["_request_shape"] = dict(request_shape)
        specs.setdefault(role, []).append(spec)
    return specs


def _validate_environment_json(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    _reject_skeleton_todo(path, "metadata/environment.json", errors)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"environment.json is invalid: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("environment.json must be a JSON object")
        return
    if str(payload.get("environment_version", "")).strip() != ENVIRONMENT_PROVENANCE_VERSION:
        errors.append(
            "environment.json environment_version must be "
            f"{ENVIRONMENT_PROVENANCE_VERSION!r}"
        )
    for field in REQUIRED_ENVIRONMENT_FIELDS:
        if _is_unfilled_text(payload.get(field)):
            errors.append(f"environment.json {field} must be filled")
    packages = payload.get("packages")
    if not isinstance(packages, dict):
        errors.append("environment.json packages must be an object")
        return
    for package in REQUIRED_ENVIRONMENT_PACKAGES:
        value = packages.get(package)
        if _is_unfilled_text(value) or "unavailable" in str(value).lower():
            errors.append(f"environment.json packages.{package} must be filled with an installed version")


def _validate_model_snapshot_manifest(
    *,
    run_dir: Path,
    row_index: int,
    row: dict[str, object],
    errors: list[str],
) -> None:
    manifest_path_value = str(row.get("snapshot_manifest_path", "")).strip()
    manifest_sha256 = str(row.get("snapshot_manifest_sha256", "")).strip()
    if _is_unfilled_text(manifest_path_value):
        errors.append(f"model_provenance.json row {row_index} snapshot_manifest_path must be filled")
        return
    if not SHA256_PATTERN.match(manifest_sha256):
        errors.append(
            f"model_provenance.json row {row_index} snapshot_manifest_sha256 "
            "must be sha256:<64 lowercase hex chars>"
        )
    manifest_path = _resolve_packet_relative_path(
        run_dir,
        manifest_path_value,
        f"model_provenance.json row {row_index} snapshot_manifest_path",
        errors,
    )
    if manifest_path is None:
        return
    if SHA256_PATTERN.match(manifest_sha256) and _file_sha256(manifest_path) != manifest_sha256:
        errors.append(
            f"model_provenance.json row {row_index} snapshot_manifest_sha256 "
            "must match snapshot_manifest_path file"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            f"model_provenance.json row {row_index} snapshot manifest is invalid: {exc}"
        )
        return
    if not isinstance(manifest, dict):
        errors.append(f"model_provenance.json row {row_index} snapshot manifest must be a JSON object")
        return
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append(f"model_provenance.json row {row_index} snapshot manifest must list files")
        return
    for file_index, file_row in enumerate(files):
        if not isinstance(file_row, dict):
            errors.append(
                f"model_provenance.json row {row_index} snapshot manifest file {file_index} must be an object"
            )
            continue
        if _is_unfilled_text(file_row.get("path")):
            errors.append(
                f"model_provenance.json row {row_index} snapshot manifest file {file_index} path must be filled"
            )
        digest = str(file_row.get("sha256", "")).strip()
        if not SHA256_PATTERN.match(digest):
            errors.append(
                f"model_provenance.json row {row_index} snapshot manifest file {file_index} "
                "sha256 must be sha256:<64 lowercase hex chars>"
            )


def _validate_model_provenance(
    path: Path,
    required_models: set[str],
    errors: list[str],
    *,
    run_dir: Path,
) -> None:
    if not path.is_file():
        return
    _reject_skeleton_todo(path, "metadata/model_provenance.json", errors)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"model_provenance.json is invalid: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append("model_provenance.json must be a JSON object")
        return
    if str(payload.get("provenance_version", "")).strip() != MODEL_PROVENANCE_VERSION:
        errors.append(
            "model_provenance.json provenance_version must be "
            f"{MODEL_PROVENANCE_VERSION!r}"
        )
    rows = payload.get("models")
    if not isinstance(rows, list) or not rows:
        errors.append("model_provenance.json must contain non-empty models rows")
        return
    covered: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"model_provenance.json row {index} must be an object")
            continue
        for field in [
            "model_id",
            "served_model_id",
            "model_revision",
            "tokenizer_revision",
            "cache_source",
        ]:
            value = str(row.get(field, "")).strip()
            if _is_unfilled_text(value):
                errors.append(f"model_provenance.json row {index} {field} must be filled")
            if field in {"model_revision", "tokenizer_revision"}:
                lowered = value.lower()
                if lowered in MUTABLE_REVISION_NAMES or lowered.startswith("refs/heads/"):
                    errors.append(
                        f"model_provenance.json row {index} {field} must not be a mutable branch alias"
                    )
        for field in ("model_revision_is_immutable", "tokenizer_revision_is_immutable"):
            if row.get(field) is not True:
                errors.append(f"model_provenance.json row {index} {field} must be true")
        _validate_model_snapshot_manifest(
            run_dir=run_dir,
            row_index=index,
            row=row,
            errors=errors,
        )
        trust_remote_code = row.get("trust_remote_code")
        if not isinstance(trust_remote_code, bool):
            errors.append(f"model_provenance.json row {index} trust_remote_code must be boolean")
        local_files_only = row.get("local_files_only")
        if not isinstance(local_files_only, bool):
            errors.append(f"model_provenance.json row {index} local_files_only must be boolean")
        for field in ("model_id", "served_model_id"):
            value = str(row.get(field, "")).strip()
            if value:
                covered.add(value)
    missing = required_models - covered
    if missing:
        errors.append(
            "model_provenance.json models do not cover profiled/metric models: "
            + ", ".join(sorted(missing))
        )


def _validate_cross_family_replacement(
    *,
    run_dir: Path,
    control_specs: dict[str, list[dict[str, object]]],
    raw_rows: list[dict[str, object]],
    errors: list[str],
) -> None:
    cross_models = {
        str(row.get("model", "")).strip()
        for row in raw_rows
        if str(row.get("row_role", "")).strip() == "cross_family_falsification"
    }
    replacement_models = {model for model in cross_models if model and model != DEFAULT_CROSS_FAMILY_MODEL}
    if not replacement_models:
        return
    replacement_path = run_dir / REPLACEMENT_METADATA_PATH
    if not replacement_path.is_file():
        errors.append(
            "cross-family replacement requires a filled "
            f"{REPLACEMENT_METADATA_PATH} copied into packet metadata before profiling"
        )
        return
    text = _read_text(replacement_path)
    if "TO_FILL" in text or "TEMPLATE_ONLY_NOT_NATIVE_EVIDENCE" in text or SKELETON_TODO_MARKER in text:
        errors.append(
            f"{REPLACEMENT_METADATA_PATH} must be filled before a replacement cross-family model is admissible"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"{REPLACEMENT_METADATA_PATH} is invalid: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append(f"{REPLACEMENT_METADATA_PATH} must be a JSON object")
        return
    replacement_row = payload.get("replacement_row")
    if not isinstance(replacement_row, dict):
        errors.append(f"{REPLACEMENT_METADATA_PATH} must contain replacement_row")
        return
    for field in REQUIRED_REPLACEMENT_ROW_FIELDS:
        if field not in replacement_row or _is_unfilled_text(replacement_row.get(field)):
            errors.append(f"{REPLACEMENT_METADATA_PATH} replacement_row.{field} must be filled")
    replacement_model = str(replacement_row.get("model", "")).strip()
    if replacement_model not in replacement_models:
        errors.append(
            f"{REPLACEMENT_METADATA_PATH} replacement_row.model must match the "
            "non-Qwen cross-family model in native_control_matrix/profiler_metrics"
        )
    if str(replacement_row.get("row_role", "")).strip() != "cross_family_falsification":
        errors.append(f"{REPLACEMENT_METADATA_PATH} replacement_row.row_role must be cross_family_falsification")
    if replacement_row.get("required_repeats") != 3:
        errors.append(f"{REPLACEMENT_METADATA_PATH} replacement_row.required_repeats must be 3")
    boundary_indices = replacement_row.get("boundary_indices")
    if (
        not isinstance(boundary_indices, list)
        or not boundary_indices
        or any(isinstance(value, bool) or not isinstance(value, int) for value in boundary_indices)
    ):
        errors.append(f"{REPLACEMENT_METADATA_PATH} replacement_row.boundary_indices must be non-empty integer list")
    architecture_map_path = str(replacement_row.get("architecture_map_path", "")).strip()
    architecture_map_sha256 = str(replacement_row.get("architecture_map_sha256", "")).strip().lower()
    if architecture_map_path:
        resolved_architecture_map = run_dir / architecture_map_path
        if not resolved_architecture_map.is_file():
            errors.append(
                f"{REPLACEMENT_METADATA_PATH} replacement_row.architecture_map_path "
                "must point to a packet file"
            )
        elif not SHA256_PATTERN.match(architecture_map_sha256):
            errors.append(
                f"{REPLACEMENT_METADATA_PATH} replacement_row.architecture_map_sha256 "
                "must be sha256:<64 lowercase hex chars>"
            )
        else:
            actual_sha256 = _file_sha256(resolved_architecture_map)
            if actual_sha256 != architecture_map_sha256:
                errors.append(
                    f"{REPLACEMENT_METADATA_PATH} replacement_row.architecture_map_sha256 "
                    f"mismatch for {architecture_map_path}: expected {architecture_map_sha256}, "
                    f"got {actual_sha256}"
                )
    preregistration_timestamp = str(
        replacement_row.get("preregistration_timestamp_utc", "")
    ).strip()
    if preregistration_timestamp and not re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
        preregistration_timestamp,
    ):
        errors.append(
            f"{REPLACEMENT_METADATA_PATH} replacement_row.preregistration_timestamp_utc "
            "must use YYYY-MM-DDTHH:MM:SSZ"
        )
    request_shape = replacement_row.get("request_shape")
    if not isinstance(request_shape, dict):
        errors.append(f"{REPLACEMENT_METADATA_PATH} replacement_row.request_shape must be an object")
    elif control_specs.get("cross_family_falsification"):
        matching_specs = [
            spec
            for spec in control_specs.get("cross_family_falsification", [])
            if str(spec.get("model", "")).strip() == replacement_model
        ]
        if not matching_specs:
            errors.append(
                f"{REPLACEMENT_METADATA_PATH} replacement_row.model must also appear "
                "in native_control_matrix.json as a cross_family_falsification row"
            )
        elif not any(
            all(
                str(spec.get(field, "")).strip()
                == str(replacement_row.get(field, "")).strip()
                for field in REPLACEMENT_MATRIX_MATCH_FIELDS
            )
            for spec in matching_specs
        ):
            errors.append(
                f"{REPLACEMENT_METADATA_PATH} replacement_row must match the copied "
                "native_control_matrix.json cross-family row"
            )
        spec_shapes = [
            spec.get("_request_shape")
            for spec in matching_specs
            if isinstance(spec.get("_request_shape"), dict)
        ]
        expected_shape = dict(request_shape)
        expected_shape.setdefault("dtype", replacement_row.get("served_dtype", ""))
        if spec_shapes and not any(
            all(expected_shape.get(key) == spec_shape.get(key) for key in expected_shape)
            for spec_shape in spec_shapes
        ):
            errors.append(
                f"{REPLACEMENT_METADATA_PATH} replacement_row.request_shape must match native_control_matrix.json"
            )


def _validate_native_control_matrix_rows(
    *,
    raw_rows: list[dict[str, object]],
    control_specs: dict[str, list[dict[str, object]]],
    errors: list[str],
) -> None:
    if not control_specs:
        return
    for index, row in enumerate(raw_rows):
        role = str(row.get("row_role", "")).strip()
        specs = control_specs.get(role, [])
        if not specs:
            errors.append(f"metric row {index} row_role is not predeclared in native_control_matrix.json")
            continue
        allowed_models: set[str] = set()
        allowed_families: set[str] = set()
        allowed_controls: set[str] = set()
        allowed_directions: set[str] = set()
        request_shapes: list[dict[str, object]] = []
        for spec in specs:
            for field in ("model", "fallback_model_if_vram_allows"):
                value = str(spec.get(field, "")).strip()
                if value:
                    allowed_models.add(value)
            family = str(spec.get("control_family", "")).strip()
            if family:
                allowed_families.add(family)
            control = str(spec.get("control_model_or_segment", "")).strip()
            if control:
                allowed_controls.add(control)
            direction = str(spec.get("boundary_direction", "")).strip()
            if direction:
                allowed_directions.add(direction)
            request_shape = spec.get("_request_shape")
            if isinstance(request_shape, dict):
                request_shapes.append(request_shape)
        model = str(row.get("model", "")).strip()
        if allowed_models and model not in allowed_models:
            errors.append(
                f"metric row {index} model is not allowed by native_control_matrix.json for {role}"
            )
        family = str(row.get("control_family", "")).strip()
        if allowed_families and family not in allowed_families:
            errors.append(
                f"metric row {index} control_family is not allowed by native_control_matrix.json for {role}"
            )
        control = str(row.get("control_model_or_segment", "")).strip()
        if allowed_controls and control not in allowed_controls:
            errors.append(
                f"metric row {index} control_model_or_segment is not allowed by "
                f"native_control_matrix.json for {role}"
            )
        direction = str(row.get("boundary_direction", "")).strip()
        if allowed_directions and direction not in allowed_directions:
            errors.append(
                f"metric row {index} boundary_direction is not allowed by "
                f"native_control_matrix.json for {role}"
            )
        if request_shapes:
            dtype = str(row.get("dtype", "")).strip()
            batch_shape = row.get("batch_shape")
            if not isinstance(batch_shape, dict):
                errors.append(
                    f"metric row {index} batch_shape must match native_control_matrix.json "
                    "request_shape"
                )
                continue
            row_shape = {
                "batch_size": batch_shape.get("batch_size"),
                "prefill_tokens": batch_shape.get("prefill_tokens"),
                "decode_tokens": batch_shape.get("decode_tokens"),
                "requests": batch_shape.get("requests"),
                "dtype": dtype,
                "cuda_graph_enabled": row.get("cuda_graph_enabled"),
            }
            if not any(
                all(row_shape.get(key) == spec_shape.get(key) for key in row_shape)
                for spec_shape in request_shapes
            ):
                errors.append(
                    f"metric row {index} request_shape does not match "
                    f"native_control_matrix.json for {role}: {row_shape}"
                )


def _validate_no_boundary_signal_packet(
    *,
    raw_rows: list[dict[str, object]],
    computed_analysis: dict[str, object] | None,
    errors: list[str],
) -> None:
    """Require explicit negative Nsight Systems evidence when NCU is skipped."""

    if not raw_rows:
        return
    if computed_analysis is not None:
        status = str(computed_analysis.get("status", ""))
        if not status.startswith("KILL or shelve"):
            errors.append(
                "no_boundary_signal_kill packets must analyze as a clean kill "
                "before Nsight Compute can be skipped"
            )
    for index, row in enumerate(raw_rows):
        ncu_artifact = str(row.get("ncu_artifact", "")).strip()
        ncu_sha256 = str(row.get("ncu_artifact_sha256", "")).strip()
        if ncu_artifact != NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL:
            errors.append(
                f"metric row {index} ncu_artifact must be "
                f"{NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL} in no_boundary_signal_kill mode"
            )
        if ncu_sha256 != NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL:
            errors.append(
                f"metric row {index} ncu_artifact_sha256 must be "
                f"{NO_BOUNDARY_SIGNAL_ARTIFACT_SENTINEL} in no_boundary_signal_kill mode"
            )
        evidence_text = " ".join(
            [
                str(row.get("recoverable_fraction_basis", "")),
                str(row.get("reduction_notes", "")),
                " ".join(
                    value
                    for value in row.get("kernel_names", [])
                    if isinstance(value, str)
                )
                if isinstance(row.get("kernel_names"), list)
                else "",
            ]
        ).lower()
        if not any(marker in evidence_text for marker in NO_BOUNDARY_SIGNAL_EVIDENCE_MARKERS):
            errors.append(
                f"metric row {index} must state explicit no-boundary-signal Nsight Systems "
                "evidence in recoverable_fraction_basis, reduction_notes, or kernel_names "
                "before skipping Nsight Compute"
            )


def check_run_artifacts(
    run_dir: Path,
    min_repeated_runs: int = 3,
    require_native_artifacts: bool = True,
    min_native_artifact_bytes: int = MIN_NATIVE_ARTIFACT_BYTES,
    packet_mode: str = "full",
    require_full_matrix: bool = False,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        errors.append(f"run directory does not exist: {run_dir}")
        return {"status": "FAIL", "run_dir": str(run_dir), "errors": errors, "warnings": warnings}
    if not run_dir.is_dir():
        errors.append(f"run path is not a directory: {run_dir}")

    for relative in REQUIRED_FILES:
        if not (run_dir / relative).is_file():
            errors.append(f"missing required artifact: {relative}")

    logs_dir = run_dir / "logs"
    log_files = _matching_artifacts(logs_dir, ["*.log", "*.txt"])
    if not logs_dir.is_dir() or not log_files:
        errors.append("missing profiling logs under logs/*.log or logs/*.txt")
    else:
        if not _has_server_profiler_log(log_files):
            errors.append("missing Nsight server profiler log under logs/")
        if not _has_client_replay_log(log_files):
            errors.append("missing client replay log under logs/")
        if require_native_artifacts:
            _validate_native_logs(log_files, errors)

    if require_native_artifacts:
        _validate_native_artifacts(
            label="Nsight Systems",
            root=run_dir / "nsys",
            patterns=NSYS_PATTERNS,
            min_bytes=min_native_artifact_bytes,
            errors=errors,
        )
        if packet_mode == NO_BOUNDARY_SIGNAL_MODE:
            warnings.append(
                "Nsight Compute artifact is optional in no_boundary_signal_kill mode"
            )
        else:
            _validate_native_artifacts(
                label="Nsight Compute",
                root=run_dir / "ncu",
                patterns=NCU_PATTERNS,
                min_bytes=min_native_artifact_bytes,
                errors=errors,
            )

    environment_path = run_dir / "metadata/environment.txt"
    _reject_skeleton_todo(environment_path, "metadata/environment.txt", errors)
    if environment_path.is_file():
        environment = _read_text(environment_path).lower()
        for marker in ENVIRONMENT_MARKERS:
            if marker not in environment:
                errors.append(f"environment metadata does not mention {marker}")
    _validate_environment_json(run_dir / "metadata/environment.json", errors)

    architecture_models = _models_from_architecture_map(
        run_dir / "metadata/architecture_map.json", errors
    )
    control_specs = _native_control_specs(
        run_dir / "metadata/native_control_matrix.json", errors
    )

    readout_path = run_dir / "readout.md"
    _reject_skeleton_todo(readout_path, "readout.md", errors)
    if readout_path.is_file():
        readout = _read_text(readout_path)
        readout_lower = readout.lower()
        for marker in READOUT_MARKERS:
            if marker not in readout:
                errors.append(f"readout.md missing decision row: {marker}")
        for placeholder in READOUT_TEMPLATE_PLACEHOLDERS:
            if placeholder in readout_lower:
                errors.append(
                    f"readout.md still contains unfilled template placeholder: {placeholder}"
                )

    profile_scope_path = run_dir / "metadata/profile_scope.json"
    profile_model = ""
    profile_models: set[str] = set()
    profile_model_roles: set[tuple[str, str]] = set()
    _reject_skeleton_todo(profile_scope_path, "metadata/profile_scope.json", errors)
    if profile_scope_path.is_file():
        try:
            profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
            profile_model = str(profile_scope.get("model", "")).strip()
            if not profile_model:
                errors.append("profile_scope.json must contain non-empty model")
            if isinstance(profile_scope, dict):
                profile_models = _profile_scope_models(profile_scope, errors)
                profile_model_roles = _profile_scope_model_roles(profile_scope)
            profiled_process_fields = (
                ["profiled_process", "nsys_profiled_process"]
                if packet_mode == NO_BOUNDARY_SIGNAL_MODE
                else PROFILED_PROCESS_FIELDS
            )
            for field in profiled_process_fields:
                _validate_profiled_process(field, str(profile_scope.get(field, "")), errors)
            trace_scope = str(profile_scope.get("trace_scope", "")).lower()
            nsys_trace_scope = str(profile_scope.get("nsys_trace_scope", trace_scope)).lower()
            ncu_trace_scope = str(profile_scope.get("ncu_trace_scope", trace_scope)).lower()
            vllm_command = str(profile_scope.get("vllm_command", ""))
            if "server" not in trace_scope and "single_process" not in trace_scope:
                errors.append("profile_scope.json trace_scope must cover server-side CUDA work")
            if "server" not in nsys_trace_scope and "single_process" not in nsys_trace_scope:
                errors.append("profile_scope.json nsys_trace_scope must cover server-side CUDA work")
            if (
                packet_mode != NO_BOUNDARY_SIGNAL_MODE
                and "server" not in ncu_trace_scope
                and "single_process" not in ncu_trace_scope
            ):
                errors.append("profile_scope.json ncu_trace_scope must cover server-side CUDA work")
            if "vllm" not in vllm_command.lower():
                errors.append("profile_scope.json vllm_command must mention vLLM")
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"profile_scope.json is invalid: {exc}")

    metrics_status = "unavailable"
    metrics_rows = 0
    model_run_counts: dict[str, int] = {}
    model_distinct_run_counts: dict[str, int] = {}
    model_config_run_counts: dict[str, int] = {}
    model_config_distinct_run_counts: dict[str, int] = {}
    computed_analysis: dict[str, object] | None = None
    metric_models: set[str] = set()
    metric_model_roles: set[tuple[str, str]] = set()
    metric_roles: set[str] = set()
    client_models: set[str] = set()
    client_replay_shapes: set[tuple[str, int, int, int, int]] = set()
    client_replay_run_shapes: set[tuple[str, str, int, int, int, int]] = set()
    for log_file in log_files:
        if "client" not in log_file.name.lower():
            continue
        try:
            payload = json.loads(_read_text(log_file))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and str(payload.get("model", "")).strip():
            client_model = str(payload["model"]).strip()
            client_run_id = str(payload.get("run_id", "")).strip()
            client_models.add(client_model)
            if not client_run_id:
                errors.append(
                    "client replay log must contain top-level run_id matching "
                    f"a profiler_metrics.json row: {log_file.name}"
                )
            requests = payload.get("requests")
            if isinstance(requests, list) and requests:
                log_shapes: list[tuple[int, int, int]] = []
                for request in requests:
                    if not isinstance(request, dict):
                        continue
                    batch_size = request.get("batch_size")
                    prompt_counts = request.get("prompt_token_counts")
                    requested_decode = request.get("requested_decode_tokens")
                    if (
                        _positive_int(batch_size)
                        and isinstance(prompt_counts, list)
                        and len(prompt_counts) == batch_size
                        and all(_positive_int(value) for value in prompt_counts)
                        and len(set(prompt_counts)) == 1
                        and _positive_int(requested_decode)
                    ):
                        log_shapes.append((batch_size, int(prompt_counts[0]), requested_decode))
                distinct_log_shapes = set(log_shapes)
                if len(distinct_log_shapes) > 1:
                    errors.append(
                        "client replay log must use one fixed request shape: "
                        f"{log_file.name}"
                    )
                elif len(log_shapes) == len(requests) and distinct_log_shapes:
                    batch_size, prefill_tokens, requested_decode = log_shapes[0]
                    client_replay_shapes.add(
                        (client_model, batch_size, prefill_tokens, requested_decode, len(requests))
                    )
                    if client_run_id:
                        client_replay_run_shapes.add(
                            (
                                client_model,
                                client_run_id,
                                batch_size,
                                prefill_tokens,
                                requested_decode,
                                len(requests),
                            )
                        )
    metrics_path = run_dir / "profiler_metrics.json"
    _reject_skeleton_todo(metrics_path, "profiler_metrics.json", errors)
    if metrics_path.is_file():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_roles = _validate_metric_provenance(
                payload,
                run_dir=run_dir,
                require_native_artifacts=require_native_artifacts,
                min_native_artifact_bytes=min_native_artifact_bytes,
                packet_mode=packet_mode,
                errors=errors,
            )
            result = analyze(payload)
            computed_analysis = result
            metrics_status = str(result["status"])
            rows = result["rows"]
            metrics_rows = len(rows)
            raw_native_rows = [
                row
                for row in payload.get("rows", [])
                if isinstance(row, dict) and not _is_pending_metric_row(row)
            ]
            _validate_reduction_input_manifest(
                path=run_dir / "metadata/reduction_input_manifest.json",
                run_dir=run_dir,
                raw_rows=raw_native_rows,
                packet_mode=packet_mode,
                errors=errors,
            )
            _validate_native_control_matrix_rows(
                raw_rows=raw_native_rows,
                control_specs=control_specs,
                errors=errors,
            )
            _validate_cross_family_replacement(
                run_dir=run_dir,
                control_specs=control_specs,
                raw_rows=raw_native_rows,
                errors=errors,
            )
            if packet_mode == NO_BOUNDARY_SIGNAL_MODE:
                _validate_no_boundary_signal_packet(
                    raw_rows=raw_native_rows,
                    computed_analysis=computed_analysis,
                    errors=errors,
                )
            if len(raw_native_rows) == len(rows):
                _validate_repeated_artifact_identity(
                    metric_rows=rows,
                    raw_rows=raw_native_rows,
                    min_repeated_runs=min_repeated_runs,
                    require_native_artifacts=require_native_artifacts,
                    packet_mode=packet_mode,
                    errors=errors,
                )
                _validate_global_artifact_identity(
                    raw_rows=raw_native_rows,
                    require_native_artifacts=require_native_artifacts,
                    packet_mode=packet_mode,
                    errors=errors,
                )
            metric_models = {str(row["model"]) for row in rows}
            metric_model_roles = {
                (str(row["model"]), str(row["row_role"]))
                for row in rows
                if str(row.get("row_role", "")) in ALLOWED_ROW_ROLES
            }
            if client_replay_shapes:
                for row in rows:
                    model = str(row["model"])
                    if model not in client_models:
                        errors.append(
                            "profiler_metrics.json model lacks matching client replay log: "
                            f"{model}"
                        )
                        continue
                    replay_shape = (
                        model,
                        int(float(row["batch_size"])),
                        int(float(row["prefill_tokens"])),
                        int(float(row["decode_tokens"])),
                        int(float(row["requests"])),
                    )
                    if replay_shape not in client_replay_shapes:
                        errors.append(
                            "client replay shape does not match profiler_metrics.json "
                            f"batch_shape for model {model}: expected {replay_shape}"
                        )
                    replay_run_shape = (
                        model,
                        str(row["run_id"]),
                        int(float(row["batch_size"])),
                        int(float(row["prefill_tokens"])),
                        int(float(row["decode_tokens"])),
                        int(float(row["requests"])),
                    )
                    if replay_run_shape not in client_replay_run_shapes:
                        errors.append(
                            "client replay log does not match profiler_metrics.json "
                            f"run_id and batch_shape: expected {replay_run_shape}"
                        )
            counts = Counter(str(row["model"]) for row in rows)
            model_run_counts = dict(counts)
            config_counts = Counter(str(row["config_key"]) for row in rows)
            model_config_run_counts = dict(config_counts)
            model_to_run_ids: dict[str, set[str]] = {}
            config_to_run_ids: dict[str, set[str]] = {}
            for row in rows:
                model_to_run_ids.setdefault(str(row["model"]), set()).add(str(row["run_id"]))
                config_to_run_ids.setdefault(str(row["config_key"]), set()).add(str(row["run_id"]))
            model_distinct_run_counts = {
                model: len(run_ids) for model, run_ids in model_to_run_ids.items()
            }
            model_config_distinct_run_counts = {
                config: len(run_ids) for config, run_ids in config_to_run_ids.items()
            }
            if metrics_rows == 0:
                errors.append("profiler_metrics.json has no valid native rows")
            if "primary_hybrid" not in metric_roles:
                errors.append("profiler_metrics.json must contain primary_hybrid metric rows")
            required_review_roles = {"same_family_control", "cross_family_falsification"}
            missing_review_roles = required_review_roles - metric_roles
            if (
                (require_full_matrix or result["status"].startswith("PROMOTE"))
                and missing_review_roles
            ):
                errors.append(
                    "promotable profiler packet must include control/falsification rows: "
                    + ", ".join(sorted(missing_review_roles))
                )
            elif missing_review_roles:
                warnings.append(
                    "profiler packet is admissible only as a profiling audit until it includes "
                    "control/falsification rows: "
                    + ", ".join(sorted(missing_review_roles))
                )
            if require_full_matrix or result["status"].startswith("PROMOTE"):
                for role in ("primary_hybrid", "same_family_control", "cross_family_falsification"):
                    role_rows = [row for row in rows if str(row.get("row_role")) == role]
                    role_run_ids = {str(row.get("run_id")) for row in role_rows}
                    if len(role_rows) < min_repeated_runs:
                        errors.append(
                            f"full matrix requires at least {min_repeated_runs} {role} rows"
                        )
                    if len(role_run_ids) < min_repeated_runs:
                        errors.append(
                            f"full matrix requires at least {min_repeated_runs} distinct {role} run_id values"
                        )
            if counts and max(counts.values()) < min_repeated_runs:
                errors.append(
                    f"no model has at least {min_repeated_runs} repeated native rows"
                )
            if config_counts and max(config_counts.values()) < min_repeated_runs:
                errors.append(
                    f"no same model/config group has at least {min_repeated_runs} "
                    "repeated native rows"
                )
            if model_distinct_run_counts and max(model_distinct_run_counts.values()) < min_repeated_runs:
                errors.append(
                    f"no model has at least {min_repeated_runs} distinct repeated run_id values"
                )
            if (
                model_config_distinct_run_counts
                and max(model_config_distinct_run_counts.values()) < min_repeated_runs
            ):
                errors.append(
                    f"no same model/config group has at least {min_repeated_runs} "
                    "distinct repeated run_id values"
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"profiler_metrics.json is invalid: {exc}")

    if profile_model and metric_models and profile_model not in metric_models:
        errors.append("profile_scope.json model does not match profiler_metrics.json models")
    if profile_models and metric_models and not metric_models.issubset(profile_models):
        errors.append("profile_scope.json models do not cover profiler_metrics.json models")
    if len(metric_models) > 1 and not profile_models.issuperset(metric_models):
        errors.append(
            "multi-model profiler_metrics.json requires profile_scope.json model_scopes "
            "covering every metric model"
        )
    if profile_model_roles and metric_model_roles and not metric_model_roles.issubset(profile_model_roles):
        errors.append(
            "profile_scope.json model_scopes do not cover profiler_metrics.json model/row_role pairs"
        )
    if client_models and metric_models and not client_models.issubset(metric_models):
        errors.append("client replay model does not match profiler_metrics.json models")
    if architecture_models and metric_models and not metric_models.issubset(architecture_models):
        errors.append("architecture_map.json models do not cover profiler_metrics.json models")
    required_provenance_models = set(metric_models) | set(client_models)
    if profile_model:
        required_provenance_models.add(profile_model)
    _validate_model_provenance(
        run_dir / "metadata/model_provenance.json",
        required_provenance_models,
        errors,
        run_dir=run_dir,
    )

    analysis_json_path = run_dir / "profiler_analysis_gate.json"
    if analysis_json_path.is_file():
        try:
            saved_analysis = json.loads(analysis_json_path.read_text(encoding="utf-8"))
            if computed_analysis is None:
                warnings.append(
                    "profiler_analysis_gate.json cannot be cross-checked because "
                    "profiler_metrics.json was missing or invalid"
                )
            else:
                for field in ["status", "decision", "summary"]:
                    if saved_analysis.get(field) != computed_analysis.get(field):
                        errors.append(
                            f"profiler_analysis_gate.json {field} does not match "
                            "profiler_metrics.json"
                        )
                saved_rows = saved_analysis.get("rows")
                if not isinstance(saved_rows, list):
                    errors.append("profiler_analysis_gate.json rows must be present")
                elif len(saved_rows) != metrics_rows:
                    errors.append(
                        "profiler_analysis_gate.json row count does not match "
                        "profiler_metrics.json"
                    )
                elif saved_rows != computed_analysis.get("rows"):
                    errors.append(
                        "profiler_analysis_gate.json rows do not match profiler_metrics.json"
                    )
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"profiler_analysis_gate.json is invalid: {exc}")

    analysis_md_path = run_dir / "profiler_analysis_gate.md"
    if analysis_md_path.is_file():
        analysis_md = _read_text(analysis_md_path)
        if "# HybridKernel Profiler Analysis Gate" not in analysis_md:
            errors.append("profiler_analysis_gate.md missing gate title")
        if computed_analysis is not None and str(computed_analysis["status"]) not in analysis_md:
            errors.append("profiler_analysis_gate.md status does not match profiler_metrics.json")

    status = "FAIL" if errors else "PASS"
    return {
        "status": status,
        "run_dir": str(run_dir),
        "errors": errors,
        "warnings": warnings,
        "metrics_status": metrics_status,
        "metrics_rows": metrics_rows,
        "model_run_counts": model_run_counts,
        "model_distinct_run_counts": model_distinct_run_counts,
        "model_config_run_counts": model_config_run_counts,
        "model_config_distinct_run_counts": model_config_distinct_run_counts,
        "min_repeated_runs": min_repeated_runs,
        "native_artifacts_required": require_native_artifacts,
        "min_native_artifact_bytes": min_native_artifact_bytes,
        "packet_mode": packet_mode,
        "require_full_matrix": require_full_matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-repeated-runs", type=int, default=3)
    parser.add_argument("--min-native-artifact-bytes", type=int, default=MIN_NATIVE_ARTIFACT_BYTES)
    parser.add_argument(
        "--allow-missing-native-artifacts",
        action="store_true",
        help="Schema-test-only escape hatch; rejected with --require-full-matrix.",
    )
    parser.add_argument("--packet-mode", choices=("full", NO_BOUNDARY_SIGNAL_MODE), default="full")
    parser.add_argument(
        "--require-full-matrix",
        action="store_true",
        help="Fail unless primary, same-family control, and cross-family falsification rows are present.",
    )
    args = parser.parse_args()
    if args.allow_missing_native_artifacts and args.require_full_matrix:
        parser.error(
            "--allow-missing-native-artifacts is schema-test-only and cannot be used "
            "with --require-full-matrix"
        )

    result = check_run_artifacts(
        args.run_dir,
        min_repeated_runs=args.min_repeated_runs,
        require_native_artifacts=not args.allow_missing_native_artifacts,
        min_native_artifact_bytes=args.min_native_artifact_bytes,
        packet_mode=args.packet_mode,
        require_full_matrix=args.require_full_matrix,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
