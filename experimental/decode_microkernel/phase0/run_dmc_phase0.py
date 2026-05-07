#!/usr/bin/env python3
"""Run the Decode Microkernel Consolidation Phase 0 measurement gate.

This is an offline reducer. It consumes the fixed HybridKernel killed-branch
profiler packet and measures decode-time launch density plus kernel-family
concentration. It never launches vLLM, CUDA, Nsight, or GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PACKET = (
    ROOT
    / "experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z"
)
DEFAULT_RESULTS_DIR = ROOT / "experimental/decode_microkernel/phase0/results"
SCHEMA_VERSION = "dmc_phase0_metrics_v1"
SOURCE_PACKET_ID = "hybridkernel_profiler_gate_20260507T212428Z"
EXPECTED_SOURCE_PACKET_REL = (
    "experimental/hybridkernel/phase2/results/"
    "hybridkernel_profiler_gate_20260507T212428Z"
)

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


@dataclass(frozen=True)
class KernelRow:
    name: str
    launches: int
    total_ns: int


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=ROOT,
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_sha() -> str | None:
    result = command_output(["git", "rev-parse", "HEAD"])
    if result["returncode"] == 0:
        return str(result["stdout"]).strip()
    return None


def classify_kernel(name: str) -> set[str]:
    lower = name.lower()
    classes: set[str] = set()
    if "gemv" in lower:
        classes.add("gemv")
    if "moe" in lower or "expert" in lower:
        classes.add("moe")
    if "selective_scan" in lower or "devicescan" in lower:
        classes.add("selective_scan")
    return classes


def decode_tokens_from_client_log(path: Path) -> tuple[int, dict[str, Any]]:
    payload = load_json(path)
    requests = payload.get("requests", [])
    decode_tokens = 0
    ok_requests = 0
    for row in requests:
        if row.get("status") != "ok":
            continue
        ok_requests += 1
        requested_completion = row.get("expected_completion_tokens_total")
        usage = row.get("response_usage") or {}
        observed_completion = usage.get("completion_tokens")
        if requested_completion is None:
            raise ValueError("client request missing expected_completion_tokens_total")
        if observed_completion is not None and int(observed_completion) != int(requested_completion):
            raise ValueError(
                "client request response_usage.completion_tokens does not match "
                "expected_completion_tokens_total"
            )
        decode_tokens += int(requested_completion)
    return decode_tokens, {
        "run_id": payload.get("run_id"),
        "model": payload.get("model"),
        "started_at_utc": payload.get("started_at_utc"),
        "ended_at_utc": payload.get("ended_at_utc"),
        "ok_requests": ok_requests,
        "total_requests": len(requests),
    }


def kernel_summary(sqlite_path: Path) -> list[KernelRow]:
    with sqlite3.connect(sqlite_path) as conn:
        has_table = conn.execute(
            "select 1 from sqlite_master where type='table' and name='KERNEL_SUMMARY'"
        ).fetchone()
        if not has_table:
            raise ValueError("missing KERNEL_SUMMARY table")
        rows = conn.execute(
            "select name, launches, total_ns from KERNEL_SUMMARY order by total_ns desc"
        ).fetchall()
    return [KernelRow(str(name), int(launches), int(total_ns)) for name, launches, total_ns in rows]


def role_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_role.setdefault(str(row["row_role"]), []).append(row)

    summary: dict[str, Any] = {}
    for role, role_rows in sorted(by_role.items()):
        present: set[str] = set()
        for row in role_rows:
            present.update(row["candidate_classes_present"])
        summary[role] = {
            "rows": len(role_rows),
            "median_launches_per_decode_token": median(
                row["launches_per_decode_token"] for row in role_rows
            ),
            "median_candidate_time_fraction": median(
                row["candidate_time_fraction"] for row in role_rows
            ),
            "median_candidate_launch_fraction": median(
                row["candidate_launch_fraction"] for row in role_rows
            ),
            "median_top3_time_fraction": median(row["top3_time_fraction"] for row in role_rows),
            "candidate_classes_present": sorted(present),
        }
    return summary


def build_row(
    *,
    source_packet: Path,
    source_row: dict[str, Any],
    reduction_manifest_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    run_id = str(source_row["run_id"])
    rel_sqlite = Path(str(source_row["nsys_artifact"]))
    sqlite_path = source_packet / rel_sqlite
    client_rel = Path("logs") / f"client_{run_id}.log"
    client_path = source_packet / client_rel
    try:
        if not sqlite_path.exists():
            raise FileNotFoundError(f"missing sanitized SQLite artifact: {rel_sqlite}")
        if not client_path.exists():
            raise FileNotFoundError(f"missing client log: {client_rel}")
        expected_sha = str(source_row.get("nsys_artifact_sha256", ""))
        actual_sha = file_sha256(sqlite_path)
        if expected_sha != actual_sha:
            raise ValueError(f"sanitized SQLite hash mismatch: expected {expected_sha}, got {actual_sha}")
        manifest_key = (
            run_id,
            str(source_row.get("row_role", "")),
            str(source_row.get("model", "")),
        )
        manifest_row = reduction_manifest_by_key.get(manifest_key)
        if manifest_row is None:
            raise ValueError(f"missing reduction manifest row for {manifest_key!r}")
        if str(manifest_row.get("source_nsys_artifact")) != str(rel_sqlite):
            raise ValueError(
                "reduction manifest source_nsys_artifact does not match profiler_metrics row"
            )
        if str(manifest_row.get("source_nsys_artifact_sha256")) != expected_sha:
            raise ValueError(
                "reduction manifest source_nsys_artifact_sha256 does not match profiler_metrics row"
            )
        decode_tokens, client_summary = decode_tokens_from_client_log(client_path)
        if decode_tokens <= 0:
            raise ValueError("client log has no positive successful decode token count")
        kernels = kernel_summary(sqlite_path)
        if not kernels:
            raise ValueError("KERNEL_SUMMARY is empty")
    except Exception as exc:
        return None, {
            "run_id": run_id,
            "row_role": source_row.get("row_role"),
            "model": source_row.get("model"),
            "reason": str(exc),
        }

    total_launches = sum(row.launches for row in kernels)
    total_time_ns = sum(row.total_ns for row in kernels)
    candidate_launches = 0
    candidate_time_ns = 0
    classes_present: set[str] = set()
    class_breakdown: dict[str, dict[str, float | int]] = {
        "gemv": {"launches": 0, "time_ms": 0.0},
        "moe": {"launches": 0, "time_ms": 0.0},
        "selective_scan": {"launches": 0, "time_ms": 0.0},
    }
    for kernel in kernels:
        classes = classify_kernel(kernel.name)
        if not classes:
            continue
        candidate_launches += kernel.launches
        candidate_time_ns += kernel.total_ns
        classes_present.update(classes)
        for class_name in classes:
            class_breakdown[class_name]["launches"] = int(class_breakdown[class_name]["launches"]) + kernel.launches
            class_breakdown[class_name]["time_ms"] = round(
                float(class_breakdown[class_name]["time_ms"]) + kernel.total_ns / 1e6,
                6,
            )

    by_launch = sorted(kernels, key=lambda row: row.launches, reverse=True)
    top3_time_ns = sum(row.total_ns for row in kernels[:3])
    top5_launches = sum(row.launches for row in by_launch[:5])
    metric_row = {
        "run_id": run_id,
        "row_role": source_row["row_role"],
        "model": source_row["model"],
        "source_nsys_artifact": str(rel_sqlite),
        "source_nsys_artifact_sha256": file_sha256(sqlite_path),
        "source_client_log": str(client_rel),
        "source_client_log_sha256": file_sha256(client_path),
        "decode_tokens_total": decode_tokens,
        "client_summary": client_summary,
        "kernel_name_count": len(kernels),
        "total_kernel_launches": total_launches,
        "total_kernel_time_ms": round(total_time_ns / 1e6, 6),
        "launches_per_decode_token": total_launches / decode_tokens,
        "top3_time_fraction": top3_time_ns / total_time_ns,
        "top5_launch_fraction": top5_launches / total_launches,
        "candidate_launch_fraction": candidate_launches / total_launches,
        "candidate_time_fraction": candidate_time_ns / total_time_ns,
        "candidate_classes_present": sorted(classes_present),
        "candidate_class_breakdown": class_breakdown,
        "top_kernels_by_time": [
            {
                "name": row.name,
                "launches": row.launches,
                "total_ms": round(row.total_ns / 1e6, 6),
            }
            for row in kernels[:10]
        ],
        "top_kernels_by_launches": [
            {
                "name": row.name,
                "launches": row.launches,
                "total_ms": round(row.total_ns / 1e6, 6),
            }
            for row in by_launch[:10]
        ],
    }
    return metric_row, None


def build_environment(source_packet: Path) -> dict[str, Any]:
    upstream_env_path = source_packet / "metadata/environment.json"
    return {
        "schema_version": "dmc_phase0_environment_v1",
        "created_at_utc": utc_now(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cwd": str(Path.cwd()),
        "source_packet_environment_path": str(upstream_env_path.relative_to(ROOT)),
        "source_packet_environment_sha256": file_sha256(upstream_env_path)
        if upstream_env_path.exists()
        else None,
        "source_packet_environment": load_json(upstream_env_path)
        if upstream_env_path.exists()
        else None,
    }


def build_manifest(source_packet: Path, source_metrics: dict[str, Any]) -> dict[str, Any]:
    files = [
        "artifact_check.json",
        "profiler_metrics.json",
        "metadata/environment.json",
        "metadata/reduction_input_manifest.json",
    ]
    for row in source_metrics.get("rows", []):
        files.append(str(row["nsys_artifact"]))
        files.append(f"logs/client_{row['run_id']}.log")
    unique_files = sorted(set(files))
    return {
        "schema_version": "dmc_phase0_input_manifest_v1",
        "source_packet": str(source_packet.relative_to(ROOT)),
        "source_packet_id": SOURCE_PACKET_ID,
        "created_at_utc": utc_now(),
        "files": [
            {
                "path": item,
                "sha256": file_sha256(source_packet / item) if (source_packet / item).exists() else None,
                "bytes": (source_packet / item).stat().st_size if (source_packet / item).exists() else None,
                "exists": (source_packet / item).exists(),
            }
            for item in unique_files
        ],
    }


def reduction_manifest_rows(source_packet: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = load_json(source_packet / "metadata/reduction_input_manifest.json")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("metadata/reduction_input_manifest.json rows must be a list")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("metadata/reduction_input_manifest.json row must be an object")
        key = (
            str(row.get("run_id", "")),
            str(row.get("row_role", "")),
            str(row.get("model", "")),
        )
        if not all(key):
            raise ValueError("metadata/reduction_input_manifest.json row has an empty key")
        if key in result:
            raise ValueError(f"duplicate reduction manifest row key {key!r}")
        result[key] = row
    return result


def ensure_fixed_source_packet(source_packet: Path) -> None:
    expected = DEFAULT_INPUT_PACKET.resolve()
    if source_packet.resolve() != expected:
        raise ValueError(
            "Decode Microkernel Phase 0 is preregistered to use only "
            f"{EXPECTED_SOURCE_PACKET_REL}; got {source_packet}"
        )


def choose_run_dir(results_dir: Path, run_id: str | None) -> Path:
    if run_id is None:
        run_id = "dmc_phase0_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return results_dir / run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-packet", type=Path, default=DEFAULT_INPUT_PACKET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    started_at = utc_now()
    source_packet = args.input_packet.resolve()
    run_dir = choose_run_dir(args.results_dir.resolve(), args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    try:
        ensure_fixed_source_packet(source_packet)
        artifact_check = load_json(source_packet / "artifact_check.json")
        source_metrics = load_json(source_packet / "profiler_metrics.json")
        reduction_manifest_by_key = reduction_manifest_rows(source_packet)
        if source_metrics.get("packet_mode") != "no_boundary_signal_kill":
            raise ValueError("source packet mode is not no_boundary_signal_kill")
        admitted_rows: list[dict[str, Any]] = []
        excluded_rows: list[dict[str, Any]] = []
        for source_row in source_metrics.get("rows", []):
            row, exclusion = build_row(
                source_packet=source_packet,
                source_row=source_row,
                reduction_manifest_by_key=reduction_manifest_by_key,
            )
            if row is not None:
                admitted_rows.append(row)
            if exclusion is not None:
                excluded_rows.append(exclusion)

        metrics = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "source_packet": str(source_packet.relative_to(ROOT)),
            "source_packet_id": SOURCE_PACKET_ID,
            "source_packet_sha256": file_sha256(source_packet / "profiler_metrics.json"),
            "upstream_artifact_check_status": artifact_check.get("status"),
            "upstream_packet_mode": source_metrics.get("packet_mode"),
            "thresholds": THRESHOLDS,
            "candidate_class_definitions": {
                "gemv": "kernel name contains gemv",
                "moe": "kernel name contains moe or expert",
                "selective_scan": "kernel name contains selective_scan or DeviceScan",
            },
            "admitted_rows": admitted_rows,
            "excluded_rows": excluded_rows,
            "role_summary": role_summary(admitted_rows),
            "interpretation": (
                "Offline Phase 0 candidate gate only. A PASS means launch-density and "
                "concentration justify authoring a Phase 1 consolidation method; it is "
                "not a latency speedup or paper result."
            ),
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "environment.json", build_environment(source_packet))
        write_json(run_dir / "input_artifact_manifest.json", build_manifest(source_packet, source_metrics))
        stdout_lines.append(
            json.dumps(
                {
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "admitted_rows": len(admitted_rows),
                    "excluded_rows": len(excluded_rows),
                    "roles": metrics["role_summary"],
                },
                sort_keys=True,
            )
        )
        exit_code = 0
    except Exception as exc:
        stderr_lines.append(repr(exc))
        write_json(
            run_dir / "metrics.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "source_packet": str(source_packet),
                "thresholds": THRESHOLDS,
                "admitted_rows": [],
                "excluded_rows": [],
                "infra_error": repr(exc),
            },
        )
        exit_code = 2

    if not (run_dir / "environment.json").exists():
        try:
            write_json(run_dir / "environment.json", build_environment(source_packet))
        except Exception as exc:  # pragma: no cover - only exercised on broken packets.
            write_json(
                run_dir / "environment.json",
                {
                    "schema_version": "dmc_phase0_environment_v1",
                    "created_at_utc": utc_now(),
                    "environment_error": repr(exc),
                },
            )
    if not (run_dir / "input_artifact_manifest.json").exists():
        try:
            source_metrics_for_manifest = (
                load_json(source_packet / "profiler_metrics.json")
                if (source_packet / "profiler_metrics.json").exists()
                else {"rows": []}
            )
            write_json(
                run_dir / "input_artifact_manifest.json",
                build_manifest(source_packet, source_metrics_for_manifest),
            )
        except Exception as exc:  # pragma: no cover - only exercised on broken packets.
            write_json(
                run_dir / "input_artifact_manifest.json",
                {
                    "schema_version": "dmc_phase0_input_manifest_v1",
                    "source_packet": str(source_packet),
                    "created_at_utc": utc_now(),
                    "manifest_error": repr(exc),
                    "files": [],
                },
            )

    command_metadata = {
        "schema_version": "dmc_phase0_command_v1",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "cwd": str(Path.cwd()),
        "git_sha": git_sha(),
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "input_packet": str(source_packet),
        "run_dir": str(run_dir),
        "no_gpu_inference": True,
        "path_python": shutil.which("python"),
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
        },
    }
    write_json(run_dir / "command_metadata.json", command_metadata)
    (run_dir / "logs/stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (run_dir / "logs/stderr.log").write_text("\n".join(stderr_lines) + "\n", encoding="utf-8")
    if stdout_lines:
        print(stdout_lines[-1])
    if stderr_lines:
        print(stderr_lines[-1], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
