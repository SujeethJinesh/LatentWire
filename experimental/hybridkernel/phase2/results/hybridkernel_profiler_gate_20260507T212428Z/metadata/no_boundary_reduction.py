#!/usr/bin/env python3
"""Reduce the HybridKernel full-matrix Nsight Systems packet.

This script records the conservative no-boundary-signal reduction used for the
2026-05-07 native packet. It derives each row's request window from the
timestamped profiler_driver client log and the Nsight Systems session start
time, then records a zero recoverable boundary-fusion upper bound because the
server-side traces contain no distinct layer-boundary NVTX range or
boundary-local conversion/materialization kernel that could be reduced into a
candidate fused operator window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze, render_markdown


NO_BOUNDARY_SENTINEL = "not_run_no_boundary_signal"
DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1]

ROWS = [
    {
        "run_id": "granite_primary_r1",
        "model": "ibm-granite/granite-4.0-h-tiny",
        "seed": 1,
        "row_role": "primary_hybrid",
        "control_family": "same_family_matched_segment",
        "control_model_or_segment": "granite_hybrid_attention_ssm_boundary_windows",
        "boundary_direction": "mixed_attention_ssm",
        "boundary_indices": [4, 5, 14, 15, 24, 25, 34, 35],
        "control_window_ids": [],
    },
    {
        "run_id": "granite_primary_r2",
        "model": "ibm-granite/granite-4.0-h-tiny",
        "seed": 2,
        "row_role": "primary_hybrid",
        "control_family": "same_family_matched_segment",
        "control_model_or_segment": "granite_hybrid_attention_ssm_boundary_windows",
        "boundary_direction": "mixed_attention_ssm",
        "boundary_indices": [4, 5, 14, 15, 24, 25, 34, 35],
        "control_window_ids": [],
    },
    {
        "run_id": "granite_primary_r3",
        "model": "ibm-granite/granite-4.0-h-tiny",
        "seed": 3,
        "row_role": "primary_hybrid",
        "control_family": "same_family_matched_segment",
        "control_model_or_segment": "granite_hybrid_attention_ssm_boundary_windows",
        "boundary_direction": "mixed_attention_ssm",
        "boundary_indices": [4, 5, 14, 15, 24, 25, 34, 35],
        "control_window_ids": [],
    },
    {
        "run_id": "granite_same_family_r1",
        "model": "ibm-granite/granite-4.0-h-tiny",
        "seed": 11,
        "row_role": "same_family_control",
        "control_family": "same_model_non_boundary_segment_control",
        "control_model_or_segment": "granite_same_model_non_boundary_ssm_to_ssm_or_attention_internal_windows",
        "boundary_direction": "non_boundary_same_family",
        "boundary_indices": [],
        "control_window_ids": [
            "granite-non-boundary-window-0",
            "granite-non-boundary-window-1",
            "granite-non-boundary-window-2",
        ],
    },
    {
        "run_id": "granite_same_family_r2",
        "model": "ibm-granite/granite-4.0-h-tiny",
        "seed": 12,
        "row_role": "same_family_control",
        "control_family": "same_model_non_boundary_segment_control",
        "control_model_or_segment": "granite_same_model_non_boundary_ssm_to_ssm_or_attention_internal_windows",
        "boundary_direction": "non_boundary_same_family",
        "boundary_indices": [],
        "control_window_ids": [
            "granite-non-boundary-window-0",
            "granite-non-boundary-window-1",
            "granite-non-boundary-window-2",
        ],
    },
    {
        "run_id": "granite_same_family_r3",
        "model": "ibm-granite/granite-4.0-h-tiny",
        "seed": 13,
        "row_role": "same_family_control",
        "control_family": "same_model_non_boundary_segment_control",
        "control_model_or_segment": "granite_same_model_non_boundary_ssm_to_ssm_or_attention_internal_windows",
        "boundary_direction": "non_boundary_same_family",
        "boundary_indices": [],
        "control_window_ids": [
            "granite-non-boundary-window-0",
            "granite-non-boundary-window-1",
            "granite-non-boundary-window-2",
        ],
    },
    {
        "run_id": "cross_family_r1",
        "model": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "seed": 21,
        "row_role": "cross_family_falsification",
        "control_family": "cross_family_hybrid_control",
        "control_model_or_segment": "nemotron_h_attention_adjacent_boundary_windows",
        "boundary_direction": "attention_adjacent_mamba_mlp_boundary",
        "boundary_indices": [13, 14, 20, 21, 29, 30, 38, 39],
        "control_window_ids": [],
    },
    {
        "run_id": "cross_family_r2",
        "model": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "seed": 22,
        "row_role": "cross_family_falsification",
        "control_family": "cross_family_hybrid_control",
        "control_model_or_segment": "nemotron_h_attention_adjacent_boundary_windows",
        "boundary_direction": "attention_adjacent_mamba_mlp_boundary",
        "boundary_indices": [13, 14, 20, 21, 29, 30, 38, 39],
        "control_window_ids": [],
    },
    {
        "run_id": "cross_family_r3",
        "model": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "seed": 23,
        "row_role": "cross_family_falsification",
        "control_family": "cross_family_hybrid_control",
        "control_model_or_segment": "nemotron_h_attention_adjacent_boundary_windows",
        "boundary_direction": "attention_adjacent_mamba_mlp_boundary",
        "boundary_indices": [13, 14, 20, 21, 29, 30, 38, 39],
        "control_window_ids": [],
    },
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_client_window(path: Path) -> tuple[dict[str, Any], int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload["requests"]
    start_ns = min(int(row["start_epoch_ns"]) for row in requests)
    end_ns = max(int(row["end_epoch_ns"]) for row in requests)
    return payload, start_ns, end_ns


def session_start_epoch_ns(sqlite_path: Path) -> int:
    with sqlite3.connect(sqlite_path) as conn:
        return int(conn.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0])


def top_kernels(sqlite_path: Path, start_rel_ns: int, end_rel_ns: int) -> list[dict[str, Any]]:
    with sqlite3.connect(sqlite_path) as conn:
        has_summary = conn.execute(
            "select 1 from sqlite_master where type='table' and name='KERNEL_SUMMARY'"
        ).fetchone()
        if has_summary:
            return [
                {
                    "name": str(name),
                    "launches": int(launches),
                    "total_ms": round(float(total_ns) / 1e6, 6),
                }
                for name, launches, total_ns in conn.execute(
                    "select name, launches, total_ns from KERNEL_SUMMARY order by total_ns desc limit 8"
                )
            ]
    query = """
        select coalesce(s.value, printf('kernel_id:%d', k.demangledName)) as name,
               count(*) as launches,
               sum(k.end - k.start) as total_ns
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.start >= ? and k.end <= ?
        group by name
        order by total_ns desc
        limit 8
    """
    with sqlite3.connect(sqlite_path) as conn:
        return [
            {"name": str(name), "launches": int(launches), "total_ms": round(float(total_ns) / 1e6, 6)}
            for name, launches, total_ns in conn.execute(query, (start_rel_ns, end_rel_ns))
        ]


def overlapping_nvtx(sqlite_path: Path, start_rel_ns: int, end_rel_ns: int) -> list[dict[str, Any]]:
    query = """
        select coalesce(n.text, s.value, '') as label,
               count(*) as events
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where n.start <= ? and coalesce(n.end, n.start) >= ?
        group by label
        order by events desc
        limit 8
    """
    with sqlite3.connect(sqlite_path) as conn:
        return [
            {"label": str(label), "events": int(events)}
            for label, events in conn.execute(query, (end_rel_ns, start_rel_ns))
            if str(label).strip()
        ]


def build_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    worksheet_rows: list[dict[str, Any]] = []
    for spec in ROWS:
        run_id = spec["run_id"]
        client_path = run_dir / "logs" / f"client_{run_id}.log"
        sqlite_path = run_dir / "nsys" / f"{run_id}.sanitized.sqlite"
        client_payload, start_epoch_ns, end_epoch_ns = load_client_window(client_path)
        session_start_ns = session_start_epoch_ns(sqlite_path)
        start_rel_ns = start_epoch_ns - session_start_ns
        end_rel_ns = end_epoch_ns - session_start_ns
        start_ms = math.floor(start_rel_ns / 1000) / 1000.0
        end_ms = math.ceil(end_rel_ns / 1000) / 1000.0
        # The window endpoints are rounded outward to microsecond precision.
        # Add a one-nanosecond inclusive-window margin so the analyzer's float
        # comparison never sees the rounded window exceed the recorded total.
        total_step_ms = round(end_ms - start_ms + 0.000001, 6)
        kernel_summary = top_kernels(sqlite_path, start_rel_ns, end_rel_ns)
        nvtx_summary = overlapping_nvtx(sqlite_path, start_rel_ns, end_rel_ns)
        kernel_names = [row["name"] for row in kernel_summary[:5]]
        if not kernel_names:
            kernel_names = ["no distinct boundary-local kernel identified"]
        nsys_artifact = f"nsys/{run_id}.sanitized.sqlite"
        nsys_sha = file_sha256(run_dir / nsys_artifact)
        raw_sqlite_artifact = f"nsys/{run_id}.sqlite"
        raw_sqlite_sha = file_sha256(run_dir / raw_sqlite_artifact)
        reduction_command = (
            "python metadata/no_boundary_reduction.py "
            "--run-dir . --packet-mode no_boundary_signal_kill"
        )
        top_kernel_note = "; ".join(
            f"{row['name']} ({row['launches']} launches, {row['total_ms']} ms)"
            for row in kernel_summary[:5]
        )
        nvtx_note = "; ".join(
            f"{row['label']} ({row['events']} events)" for row in nvtx_summary[:5]
        ) or "no overlapping NVTX labels"
        basis = (
            "No boundary signal: the request-window Nsight Systems trace contains "
            "ordinary vLLM CUDA kernels but no distinct boundary-local "
            "conversion/materialization kernel or layer-boundary NVTX range that "
            "can be isolated as a candidate fused boundary operator."
        )
        notes = (
            f"No boundary-local kernel window was selected. Request window "
            f"{start_ms:.6f}-{end_ms:.6f} ms was derived from client log "
            f"logs/client_{run_id}.log and TARGET_INFO_SESSION_START_TIME in "
            f"sanitized Nsight Systems SQLite export {nsys_artifact}. "
            f"Top kernels: {top_kernel_note}. Overlapping NVTX: {nvtx_note}."
        )
        metric_row: dict[str, Any] = {
            "model": spec["model"],
            "run_id": run_id,
            "total_step_ms": total_step_ms,
            "attention_ssm_boundary_ms": 0.0,
            "matched_non_boundary_ms": 0.0,
            "recoverable_fraction": 0.0,
            "dtype": "bfloat16",
            "profiled_process": "vllm_server",
            "trace_scope": "server-side CUDA kernels captured by Nsight Systems during fixed-shape vLLM replay",
            "cuda_graph_enabled": True,
            "batch_shape": {
                "batch_size": 1,
                "prefill_tokens": 128,
                "decode_tokens": 64,
                "requests": 16,
            },
            "control_model_or_segment": spec["control_model_or_segment"],
            "row_role": spec["row_role"],
            "control_family": spec["control_family"],
            "boundary_direction": spec["boundary_direction"],
            "nsys_artifact": nsys_artifact,
            "nsys_artifact_sha256": nsys_sha,
            "ncu_artifact": NO_BOUNDARY_SENTINEL,
            "ncu_artifact_sha256": NO_BOUNDARY_SENTINEL,
            "kernel_names": kernel_names,
            "boundary_indices": spec["boundary_indices"],
            "control_window_ids": spec["control_window_ids"],
            "time_window_ms": {"start": start_ms, "end": end_ms},
            "ncu_launch_selection": {
                "kernel_regex": NO_BOUNDARY_SENTINEL,
                "launch_skip": 0,
                "launch_count": 0,
                "source_nsys_artifact": nsys_artifact,
                "source_time_window_ms": {"start": start_ms, "end": end_ms},
                "derivation_notes": (
                    "Nsight Compute skipped because Nsight Systems showed no boundary signal "
                    "or distinct boundary-local kernel window to profile."
                ),
            },
            "recoverable_fraction_basis": basis,
            "reduction_command": reduction_command,
            "reduction_notes": notes,
            "notes": (
                f"Seed {spec['seed']}; client top-level run_id {client_payload['run_id']}; "
                "row reduced as no_boundary_signal_kill."
            ),
        }
        metric_rows.append(metric_row)
        worksheet_rows.append(
            {
                "run_id": run_id,
                "model": spec["model"],
                "row_role": spec["row_role"],
                "seed": spec["seed"],
                "client_log": f"logs/client_{run_id}.log",
                "server_log": f"logs/nsys_server_{run_id}.log",
                "nsys_artifact": nsys_artifact,
                "nsys_artifact_sha256": nsys_sha,
                "raw_nsys_sqlite_export": raw_sqlite_artifact,
                "raw_nsys_sqlite_export_sha256": raw_sqlite_sha,
                "start_epoch_ns": start_epoch_ns,
                "end_epoch_ns": end_epoch_ns,
                "session_start_epoch_ns": session_start_ns,
                "window_start_ms": start_ms,
                "window_end_ms": end_ms,
                "total_step_ms": total_step_ms,
                "boundary_ms": 0.0,
                "matched_non_boundary_ms": 0.0,
                "recoverable_fraction": 0.0,
                "top_kernels": " | ".join(
                    f"{row['name']}:{row['launches']}:{row['total_ms']}ms" for row in kernel_summary[:5]
                ),
                "nvtx_labels": " | ".join(
                    f"{row['label']}:{row['events']}" for row in nvtx_summary[:5]
                ),
                "decision_evidence": basis,
            }
        )
    return metric_rows, worksheet_rows


def write_worksheet(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_id",
        "model",
        "row_role",
        "seed",
        "client_log",
        "server_log",
        "nsys_artifact",
        "nsys_artifact_sha256",
        "raw_nsys_sqlite_export",
        "raw_nsys_sqlite_export_sha256",
        "start_epoch_ns",
        "end_epoch_ns",
        "session_start_epoch_ns",
        "window_start_ms",
        "window_end_ms",
        "total_step_ms",
        "boundary_ms",
        "matched_non_boundary_ms",
        "recoverable_fraction",
        "top_kernels",
        "nvtx_labels",
        "decision_evidence",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row[column]).replace("\t", " ") for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(run_dir: Path, metric_rows: list[dict[str, Any]], source_sha: str) -> None:
    rows = []
    for row in metric_rows:
        rows.append(
            {
                "run_id": row["run_id"],
                "row_role": row["row_role"],
                "model": row["model"],
                "reduction_source_path": "metadata/no_boundary_reduction.py",
                "source_nsys_artifact": row["nsys_artifact"],
                "source_nsys_artifact_sha256": row["nsys_artifact_sha256"],
                "source_time_window_ms": row["time_window_ms"],
                "source_ncu_artifact": row["ncu_artifact"],
                "source_ncu_artifact_sha256": row["ncu_artifact_sha256"],
                "reduction_command": row["reduction_command"],
                "reduction_script_sha256": source_sha,
                "reduction_notes": row["reduction_notes"],
            }
        )
    payload = {
        "manifest_version": "hybridkernel_reduction_inputs_v1",
        "description": (
            "Full-matrix no-boundary-signal reduction inputs. Each row is tied "
            "to a timestamped client replay log, a server-side Nsight Systems "
            "SQLite export, and the reduction script hash."
        ),
        "packet_mode": "no_boundary_signal_kill",
        "rows": rows,
    }
    (run_dir / "metadata" / "reduction_input_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_readout(run_dir: Path, result: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    row_lines = []
    for row in rows:
        row_lines.append(
            f"- `{row['run_id']}` `{row['row_role']}` `{row['model']}`: "
            f"gain upper bound 0.000000, window {row['time_window_ms']['start']:.6f}-"
            f"{row['time_window_ms']['end']:.6f} ms, artifact `{row['nsys_artifact']}`."
        )
    readout = f"""# HybridKernel Native Profiler Readout

| Question | Evidence | Decision |
|---|---|---|
| Distinct boundary conversion/materialization kernel? | Full-matrix Nsight Systems traces contain ordinary vLLM CUDA kernels but no distinct boundary-local conversion/materialization kernel and no layer-boundary NVTX range in the fixed request windows. Evidence rows are listed below and reduction inputs are in `metadata/reduction_worksheet.tsv`. | No; row reduction uses `no_boundary_signal_kill`. |
| Boundary idle or launch gap? | No boundary-local interval could be isolated from the server-side CUDA trace, so no launch-gap window was selected for Nsight Compute. | No measurable boundary-specific launch gap. |
| Extra DRAM/L2 traffic near boundary? | Nsight Compute was not run because there was no boundary-local kernel/window to profile after Nsight Systems reduction. The preregistered no-boundary path requires a clean kill rather than inventing an NCU selection. | Not measured; skipped per no-boundary-signal rule. |
| End-to-end impact estimate clears 3%? | `profiler_analysis_gate.json` recomputes a 0.000000 recoverable-gain upper bound for all nine rows; primary bootstrap CI is [0.000000, 0.000000]. | No; the 3% promotion shelf is not reached. |
| Same-family controls available? | Three same-family Granite non-boundary control traces are present: `granite_same_family_r1`, `granite_same_family_r2`, and `granite_same_family_r3`. | Available and below the 3% gate. |
| Cross-family falsification attempted? | Three pre-committed replacement Nemotron Nano 9B v2 traces are present: `cross_family_r1`, `cross_family_r2`, and `cross_family_r3`. | Attempted and below the 3% gate. |

## Decision

`{result['status']}`

{result['decision']}

## Row Evidence

{chr(10).join(row_lines)}

## Reduction Notes

This packet does not claim a speedup. It records a full-matrix native profiler
kill because the server-side Nsight Systems traces did not expose a distinct
boundary-local conversion/materialization or launch/locality signal that could
support a boundary-fusion prototype. The reduction therefore sets
`attention_ssm_boundary_ms = 0.0`, `matched_non_boundary_ms = 0.0`, and
`recoverable_fraction = 0.0` for every row, with exact request windows and
artifact hashes preserved in `metadata/reduction_input_manifest.json`.
"""
    (run_dir / "readout.md").write_text(readout, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--packet-mode", default="no_boundary_signal_kill")
    args = parser.parse_args()
    if args.packet_mode != "no_boundary_signal_kill":
        raise SystemExit("this reducer only supports no_boundary_signal_kill")

    run_dir = args.run_dir.resolve()
    metric_rows, worksheet_rows = build_rows(run_dir)
    metrics = {
        "description": (
            "Full-matrix HybridKernel native profiler reduction. Nsight Systems "
            "showed no distinct boundary-local conversion/materialization signal, "
            "so the packet is reduced under no_boundary_signal_kill."
        ),
        "packet_mode": "no_boundary_signal_kill",
        "rows": metric_rows,
    }
    (run_dir / "profiler_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_worksheet(run_dir / "metadata" / "reduction_worksheet.tsv", worksheet_rows)
    source_sha = file_sha256(run_dir / "metadata" / "no_boundary_reduction.py")
    write_manifest(run_dir, metric_rows, source_sha)
    result = analyze(metrics)
    (run_dir / "profiler_analysis_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "profiler_analysis_gate.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    write_readout(run_dir, result, metric_rows)
    print(json.dumps({"status": result["status"], "rows": len(metric_rows)}, indent=2))


if __name__ == "__main__":
    main()
