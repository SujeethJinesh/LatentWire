#!/usr/bin/env python3
"""Check a Decode Microkernel Consolidation Phase 0 result packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SOURCE_PACKET_REL = (
    "experimental/hybridkernel/phase2/results/"
    "hybridkernel_profiler_gate_20260507T212428Z"
)
EXPECTED_SOURCE_PACKET = ROOT / EXPECTED_SOURCE_PACKET_REL
SOURCE_PACKET_ID = "hybridkernel_profiler_gate_20260507T212428Z"

THRESHOLDS: dict[str, Any] = {
    "min_admitted_rows": 8,
    "expected_source_rows": 9,
    "min_rows_by_role": {
        "primary_hybrid": 3,
        "same_family_control": 2,
        "cross_family_falsification": 3,
    },
    "median_launches_per_decode_token_min": {
        "primary_hybrid": 500.0,
        "same_family_control": 500.0,
        "cross_family_falsification": 300.0,
    },
    "per_row_launches_per_decode_token_min": 250.0,
    "median_candidate_time_fraction_min": 0.65,
    "median_candidate_launch_fraction_min": 0.20,
    "median_top3_time_fraction_min": {
        "primary_hybrid": 0.60,
        "same_family_control": 0.60,
        "cross_family_falsification": 0.85,
    },
    "required_classes_by_role": {
        "primary_hybrid": ["gemv", "moe"],
        "same_family_control": ["gemv", "moe"],
        "cross_family_falsification": ["gemv", "selective_scan"],
    },
}

REQUIRED_PACKET_FILES = [
    "environment.json",
    "input_artifact_manifest.json",
    "metrics.json",
    "command_metadata.json",
    "logs/stdout.log",
    "logs/stderr.log",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_relative(path_text: str) -> Path | None:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def reduction_manifest_rows() -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = load_json(EXPECTED_SOURCE_PACKET / "metadata/reduction_input_manifest.json")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("source reduction manifest rows must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source reduction manifest row must be an object")
        key = (
            str(row.get("run_id", "")),
            str(row.get("row_role", "")),
            str(row.get("model", "")),
        )
        if not all(key):
            raise ValueError("source reduction manifest row has empty key")
        result[key] = row
    return result


def validate_client_log(path: Path, row: dict[str, Any], infra_reasons: list[str]) -> None:
    run_id = str(row.get("run_id"))
    try:
        payload = load_json(path)
    except Exception as exc:
        infra_reasons.append(f"{run_id}: invalid client log JSON: {exc!r}")
        return
    if str(payload.get("run_id")) != run_id:
        infra_reasons.append(f"{run_id}: client log run_id mismatch")
    if str(payload.get("model")) != str(row.get("model")):
        infra_reasons.append(f"{run_id}: client log model mismatch")
    requests = payload.get("requests", [])
    if not isinstance(requests, list) or not requests:
        infra_reasons.append(f"{run_id}: client log has no requests")
        return
    decode_tokens = 0
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or request.get("status") != "ok":
            infra_reasons.append(f"{run_id}: client request {index} is not status=ok")
            continue
        expected = request.get("expected_completion_tokens_total")
        usage = request.get("response_usage") or {}
        observed = usage.get("completion_tokens")
        if expected is None:
            infra_reasons.append(f"{run_id}: client request {index} missing expected completion tokens")
            continue
        if observed is not None and int(observed) != int(expected):
            infra_reasons.append(f"{run_id}: client request {index} observed/requested decode mismatch")
        decode_tokens += int(expected)
    if decode_tokens != int(row.get("decode_tokens_total", -1)):
        infra_reasons.append(f"{run_id}: client requested decode total does not match metrics")


def validate_sanitized_sqlite(path: Path, row: dict[str, Any], infra_reasons: list[str]) -> None:
    run_id = str(row.get("run_id"))
    try:
        import sqlite3

        with sqlite3.connect(path) as conn:
            tables = {
                name
                for (name,) in conn.execute(
                    "select name from sqlite_master where type='table'"
                )
            }
            required = {"KERNEL_SUMMARY", "TARGET_INFO_SESSION_START_TIME", "SANITIZATION_NOTE"}
            missing = sorted(required - tables)
            if missing:
                infra_reasons.append(f"{run_id}: sanitized SQLite missing tables {missing}")
                return
            kernel_count, launches, total_ns = conn.execute(
                "select count(*), coalesce(sum(launches), 0), coalesce(sum(total_ns), 0) "
                "from KERNEL_SUMMARY"
            ).fetchone()
            if int(kernel_count) != int(row.get("kernel_name_count", -1)):
                infra_reasons.append(f"{run_id}: KERNEL_SUMMARY count does not match metrics")
            if int(launches) != int(row.get("total_kernel_launches", -1)):
                infra_reasons.append(f"{run_id}: KERNEL_SUMMARY launches do not match metrics")
            expected_total_ms = round(float(total_ns) / 1e6, 6)
            if abs(expected_total_ms - float(row.get("total_kernel_time_ms", -1.0))) > 1e-6:
                infra_reasons.append(f"{run_id}: KERNEL_SUMMARY total time does not match metrics")
    except Exception as exc:
        infra_reasons.append(f"{run_id}: unreadable sanitized SQLite: {exc!r}")


def validate_source_audit(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    command: dict[str, Any],
    admitted_rows: list[dict[str, Any]],
    infra_reasons: list[str],
) -> None:
    if metrics.get("source_packet") != EXPECTED_SOURCE_PACKET_REL:
        infra_reasons.append("metrics.source_packet does not match fixed preregistered input packet")
    if metrics.get("source_packet_id") != SOURCE_PACKET_ID:
        infra_reasons.append("metrics.source_packet_id does not match fixed packet id")
    expected_metrics_path = EXPECTED_SOURCE_PACKET / "profiler_metrics.json"
    if metrics.get("source_packet_sha256") != file_sha256(expected_metrics_path):
        infra_reasons.append("metrics.source_packet_sha256 does not match fixed profiler_metrics.json")
    if manifest.get("source_packet") != EXPECTED_SOURCE_PACKET_REL:
        infra_reasons.append("manifest.source_packet does not match fixed preregistered input packet")
    if manifest.get("source_packet_id") != SOURCE_PACKET_ID:
        infra_reasons.append("manifest.source_packet_id does not match fixed packet id")
    command_input = Path(str(command.get("input_packet", "")))
    try:
        command_input_resolved = command_input.resolve()
    except OSError:
        command_input_resolved = command_input
    if command_input_resolved != EXPECTED_SOURCE_PACKET.resolve():
        infra_reasons.append("command_metadata.input_packet does not match fixed preregistered input packet")

    manifest_files = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    for required in [
        "artifact_check.json",
        "profiler_metrics.json",
        "metadata/environment.json",
        "metadata/reduction_input_manifest.json",
    ]:
        item = manifest_files.get(required)
        path = EXPECTED_SOURCE_PACKET / required
        if item is None:
            infra_reasons.append(f"manifest missing fixed input artifact {required}")
        elif item.get("sha256") != file_sha256(path):
            infra_reasons.append(f"manifest hash mismatch for fixed input artifact {required}")

    try:
        reduction_by_key = reduction_manifest_rows()
    except Exception as exc:
        infra_reasons.append(f"cannot load source reduction manifest: {exc!r}")
        reduction_by_key = {}

    for row in admitted_rows:
        run_id = str(row.get("run_id"))
        nsys_rel = source_relative(str(row.get("source_nsys_artifact", "")))
        client_rel = source_relative(str(row.get("source_client_log", "")))
        if nsys_rel is None or client_rel is None:
            infra_reasons.append(f"{run_id}: source paths must be relative and stay inside packet")
            continue
        nsys_path = EXPECTED_SOURCE_PACKET / nsys_rel
        client_path = EXPECTED_SOURCE_PACKET / client_rel
        if not nsys_path.is_file():
            infra_reasons.append(f"{run_id}: missing source_nsys_artifact {nsys_rel}")
        elif file_sha256(nsys_path) != row.get("source_nsys_artifact_sha256"):
            infra_reasons.append(f"{run_id}: source_nsys_artifact_sha256 mismatch")
        else:
            validate_sanitized_sqlite(nsys_path, row, infra_reasons)
        if not client_path.is_file():
            infra_reasons.append(f"{run_id}: missing source_client_log {client_rel}")
        elif file_sha256(client_path) != row.get("source_client_log_sha256"):
            infra_reasons.append(f"{run_id}: source_client_log_sha256 mismatch")
        else:
            validate_client_log(client_path, row, infra_reasons)
        key = (run_id, str(row.get("row_role")), str(row.get("model")))
        reduction_row = reduction_by_key.get(key)
        if reduction_row is None:
            infra_reasons.append(f"{run_id}: missing source reduction manifest row")
        else:
            if str(reduction_row.get("source_nsys_artifact")) != str(nsys_rel):
                infra_reasons.append(f"{run_id}: reduction manifest source_nsys_artifact mismatch")
            if str(reduction_row.get("source_nsys_artifact_sha256")) != row.get(
                "source_nsys_artifact_sha256"
            ):
                infra_reasons.append(f"{run_id}: reduction manifest source_nsys_artifact_sha256 mismatch")


def rows_by_role(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_role.setdefault(str(row.get("row_role")), []).append(row)
    return by_role


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra_reasons: list[str] = []
    kill_reasons: list[str] = []

    for rel_path in REQUIRED_PACKET_FILES:
        if not (run_dir / rel_path).exists():
            infra_reasons.append(f"missing required packet file: {rel_path}")
    if infra_reasons:
        return {"decision": "INFRA", "run_dir": str(run_dir), "reasons": infra_reasons}

    try:
        metrics = load_json(run_dir / "metrics.json")
        manifest = load_json(run_dir / "input_artifact_manifest.json")
        command = load_json(run_dir / "command_metadata.json")
    except Exception as exc:
        return {"decision": "INFRA", "run_dir": str(run_dir), "reasons": [f"bad JSON: {exc!r}"]}

    if metrics.get("thresholds") != THRESHOLDS:
        infra_reasons.append("threshold metadata mismatch")
    if metrics.get("schema_version") != "dmc_phase0_metrics_v1":
        infra_reasons.append("metrics schema_version is not dmc_phase0_metrics_v1")
    if metrics.get("upstream_artifact_check_status") != "PASS":
        infra_reasons.append("upstream artifact_check status is not PASS")
    if metrics.get("upstream_packet_mode") != "no_boundary_signal_kill":
        infra_reasons.append("upstream packet mode is not no_boundary_signal_kill")
    if not command.get("no_gpu_inference"):
        infra_reasons.append("command metadata does not assert no_gpu_inference")

    manifest_files = manifest.get("files", [])
    missing_inputs = [
        row.get("path")
        for row in manifest_files
        if not row.get("exists") or row.get("sha256") is None or row.get("bytes") in (None, 0)
    ]
    if missing_inputs:
        infra_reasons.append(f"missing or empty input artifacts: {missing_inputs}")

    admitted_rows = metrics.get("admitted_rows", [])
    excluded_rows = metrics.get("excluded_rows", [])
    if not isinstance(admitted_rows, list) or not isinstance(excluded_rows, list):
        infra_reasons.append("admitted_rows and excluded_rows must be lists")
        admitted_rows = []
        excluded_rows = []

    validate_source_audit(
        metrics=metrics,
        manifest=manifest,
        command=command,
        admitted_rows=admitted_rows,
        infra_reasons=infra_reasons,
    )

    expected = int(THRESHOLDS["expected_source_rows"])
    if len(admitted_rows) + len(excluded_rows) != expected:
        infra_reasons.append(
            f"expected {expected} total source rows, got {len(admitted_rows) + len(excluded_rows)}"
        )
    if len(admitted_rows) < int(THRESHOLDS["min_admitted_rows"]):
        infra_reasons.append(
            f"admitted row count {len(admitted_rows)} below {THRESHOLDS['min_admitted_rows']}"
        )

    grouped = rows_by_role(admitted_rows)
    for role, minimum in THRESHOLDS["min_rows_by_role"].items():
        actual = len(grouped.get(role, []))
        if actual < minimum:
            infra_reasons.append(f"{role} admitted rows {actual} below required {minimum}")

    for row in admitted_rows:
        run_id = row.get("run_id")
        if row.get("decode_tokens_total", 0) <= 0:
            infra_reasons.append(f"{run_id}: nonpositive decode_tokens_total")
        if row.get("total_kernel_launches", 0) <= 0:
            infra_reasons.append(f"{run_id}: nonpositive total_kernel_launches")
        if row.get("total_kernel_time_ms", 0) <= 0:
            infra_reasons.append(f"{run_id}: nonpositive total_kernel_time_ms")
        if row.get("kernel_name_count", 0) <= 0:
            infra_reasons.append(f"{run_id}: empty kernel_name_count")

    if infra_reasons:
        return {"decision": "INFRA", "run_dir": str(run_dir), "reasons": infra_reasons}

    for row in admitted_rows:
        value = float(row["launches_per_decode_token"])
        minimum = float(THRESHOLDS["per_row_launches_per_decode_token_min"])
        if value < minimum:
            kill_reasons.append(
                f"{row['run_id']}: launches_per_decode_token {value:.6f} below {minimum:.6f}"
            )

    role_metrics: dict[str, dict[str, Any]] = {}
    for role, role_rows in grouped.items():
        present: set[str] = set()
        for row in role_rows:
            present.update(row["candidate_classes_present"])
        role_metrics[role] = {
            "rows": len(role_rows),
            "median_launches_per_decode_token": median(
                float(row["launches_per_decode_token"]) for row in role_rows
            ),
            "median_candidate_time_fraction": median(
                float(row["candidate_time_fraction"]) for row in role_rows
            ),
            "median_candidate_launch_fraction": median(
                float(row["candidate_launch_fraction"]) for row in role_rows
            ),
            "median_top3_time_fraction": median(float(row["top3_time_fraction"]) for row in role_rows),
            "candidate_classes_present": sorted(present),
        }

    for role, minimum in THRESHOLDS["median_launches_per_decode_token_min"].items():
        actual = role_metrics[role]["median_launches_per_decode_token"]
        if actual < minimum:
            kill_reasons.append(
                f"{role}: median_launches_per_decode_token {actual:.6f} below {minimum:.6f}"
            )
    for role, values in role_metrics.items():
        actual_time = values["median_candidate_time_fraction"]
        actual_launch = values["median_candidate_launch_fraction"]
        if actual_time < THRESHOLDS["median_candidate_time_fraction_min"]:
            kill_reasons.append(
                f"{role}: median_candidate_time_fraction {actual_time:.6f} below "
                f"{THRESHOLDS['median_candidate_time_fraction_min']:.6f}"
            )
        if actual_launch < THRESHOLDS["median_candidate_launch_fraction_min"]:
            kill_reasons.append(
                f"{role}: median_candidate_launch_fraction {actual_launch:.6f} below "
                f"{THRESHOLDS['median_candidate_launch_fraction_min']:.6f}"
            )

    for role, minimum in THRESHOLDS["median_top3_time_fraction_min"].items():
        actual = role_metrics[role]["median_top3_time_fraction"]
        if actual < minimum:
            kill_reasons.append(f"{role}: median_top3_time_fraction {actual:.6f} below {minimum:.6f}")

    for role, required_classes in THRESHOLDS["required_classes_by_role"].items():
        present = set(role_metrics[role]["candidate_classes_present"])
        missing = sorted(set(required_classes) - present)
        if missing:
            kill_reasons.append(f"{role}: missing required candidate classes {missing}")

    if kill_reasons:
        return {
            "decision": "KILL",
            "run_dir": str(run_dir),
            "reasons": kill_reasons,
            "role_metrics": role_metrics,
        }
    return {
        "decision": "PASS",
        "run_dir": str(run_dir),
        "reasons": [],
        "role_metrics": role_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    result = evaluate(args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
