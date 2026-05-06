"""Check native HybridKernel profiler run artifacts for review completeness.

This verifier is intentionally about admissibility, not success. A passing run
directory has enough metadata, native profiler artifacts, and repeated reduced
metrics for a reviewer to inspect the promote/kill decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze


REQUIRED_FILES = [
    "metadata/environment.txt",
    "metadata/architecture_map.json",
    "metadata/profile_scope.json",
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
    SKELETON_TODO_MARKER.lower().encode("utf-8"),
]
SERVER_LOG_EVIDENCE_MARKERS = ["vllm", "nsys", "ncu", "cuda"]
CLIENT_LOG_EVIDENCE_MARKERS = ['"requests"', '"model"', '"status"']

ALLOWED_PROFILED_PROCESSES = {"vllm_server", "single_process_vllm_benchmark"}
PROFILED_PROCESS_FIELDS = ["profiled_process", "nsys_profiled_process", "ncu_profiled_process"]
NO_BOUNDARY_SIGNAL_MODE = "no_boundary_signal_kill"
REQUIRED_METRIC_PROVENANCE_FIELDS = [
    "row_role",
    "control_family",
    "boundary_direction",
    "nsys_artifact",
    "ncu_artifact",
    "kernel_names",
    "boundary_indices",
    "time_window_ms",
    "reduction_notes",
]
ALLOWED_ROW_ROLES = {"primary_hybrid", "same_family_control", "cross_family_falsification"}


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _reject_skeleton_todo(path: Path, label: str, errors: list[str]) -> None:
    if path.is_file() and SKELETON_TODO_MARKER in _read_text(path):
        errors.append(f"{label} still contains native run-packet skeleton TODO markers")


def _validate_profiled_process(field: str, value: str, errors: list[str]) -> None:
    if value not in ALLOWED_PROFILED_PROCESSES:
        errors.append(
            f"profile_scope.json {field} must identify the profiled CUDA-serving process, "
            "not just an HTTP client"
        )


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
        head = artifact.read_bytes()[:4096].lower()
        if any(marker in head for marker in PLACEHOLDER_ARTIFACT_MARKERS):
            errors.append(
                f"{label} artifact appears to be a placeholder, not a native profiler export: "
                f"{artifact.relative_to(root.parent)}"
            )


def _validate_metric_artifact_path(
    *,
    row_index: int,
    field: str,
    value: str,
    run_dir: Path,
    allowed_suffixes: set[str],
    require_native_artifacts: bool,
    min_bytes: int,
    packet_mode: str,
    errors: list[str],
) -> None:
    if field == "ncu_artifact" and packet_mode == NO_BOUNDARY_SIGNAL_MODE and value == "not_run_no_boundary_signal":
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
    size = resolved.stat().st_size
    if size < min_bytes:
        errors.append(
            f"metric row {row_index} {field} is too small to be native evidence: "
            f"{value} has {size} bytes, expected at least {min_bytes}"
        )
    head = resolved.read_bytes()[:4096].lower()
    if any(marker in head for marker in PLACEHOLDER_ARTIFACT_MARKERS):
        errors.append(f"metric row {row_index} {field} appears to be placeholder evidence: {value}")


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
        for field in ["control_family", "boundary_direction", "nsys_artifact", "ncu_artifact", "reduction_notes"]:
            value = str(row.get(field, "")).strip()
            if not value or "TODO_NATIVE_PROFILE_FILL" in value or "placeholder" in value.lower():
                errors.append(f"metric row {idx} {field} must be filled with non-placeholder text")
        nsys_value = str(row.get("nsys_artifact", "")).strip()
        ncu_value = str(row.get("ncu_artifact", "")).strip()
        if nsys_value and "TODO_NATIVE_PROFILE_FILL" not in nsys_value and "placeholder" not in nsys_value.lower():
            _validate_metric_artifact_path(
                row_index=idx,
                field="nsys_artifact",
                value=nsys_value,
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
        elif row_role == "primary_hybrid" and not boundary_indices:
            errors.append(f"metric row {idx} primary_hybrid rows must name boundary_indices")
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
    return roles


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


def check_run_artifacts(
    run_dir: Path,
    min_repeated_runs: int = 3,
    require_native_artifacts: bool = True,
    min_native_artifact_bytes: int = MIN_NATIVE_ARTIFACT_BYTES,
    packet_mode: str = "full",
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

    architecture_models = _models_from_architecture_map(
        run_dir / "metadata/architecture_map.json", errors
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
    _reject_skeleton_todo(profile_scope_path, "metadata/profile_scope.json", errors)
    if profile_scope_path.is_file():
        try:
            profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
            profile_model = str(profile_scope.get("model", "")).strip()
            if not profile_model:
                errors.append("profile_scope.json must contain non-empty model")
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
    metric_roles: set[str] = set()
    client_models: set[str] = set()
    for log_file in log_files:
        if "client" not in log_file.name.lower():
            continue
        try:
            payload = json.loads(_read_text(log_file))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and str(payload.get("model", "")).strip():
            client_models.add(str(payload["model"]).strip())
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
            metric_models = {str(row["model"]) for row in rows}
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
                result["status"].startswith("PROMOTE")
                and missing_review_roles
            ):
                errors.append(
                    "promotion profiler packet must include control/falsification rows: "
                    + ", ".join(sorted(missing_review_roles))
                )
            elif missing_review_roles:
                warnings.append(
                    "profiler packet is admissible only as a profiling audit until it includes "
                    "control/falsification rows: "
                    + ", ".join(sorted(missing_review_roles))
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
    if client_models and metric_models and not client_models.issubset(metric_models):
        errors.append("client replay model does not match profiler_metrics.json models")
    if architecture_models and metric_models and not metric_models.issubset(architecture_models):
        errors.append("architecture_map.json models do not cover profiler_metrics.json models")

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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-repeated-runs", type=int, default=3)
    parser.add_argument("--min-native-artifact-bytes", type=int, default=MIN_NATIVE_ARTIFACT_BYTES)
    parser.add_argument("--allow-missing-native-artifacts", action="store_true")
    parser.add_argument("--packet-mode", choices=("full", NO_BOUNDARY_SIGNAL_MODE), default="full")
    args = parser.parse_args()

    result = check_run_artifacts(
        args.run_dir,
        min_repeated_runs=args.min_repeated_runs,
        require_native_artifacts=not args.allow_missing_native_artifacts,
        min_native_artifact_bytes=args.min_native_artifact_bytes,
        packet_mode=args.packet_mode,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
