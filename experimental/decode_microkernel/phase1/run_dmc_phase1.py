#!/usr/bin/env python3
"""Run the Decode Microkernel Consolidation Phase 1 replay gate.

The runner consumes the fixed Phase 0 PASS packet plus the fixed sanitized
HybridKernel profiler packet. It builds one replay schedule for every frozen
HybridKernel source row, then measures a launch-heavy baseline made of repeated
small CUDA ops against a trace-derived packed Triton replay kernel.

This is a decode micro-operation replay benchmark only. It does not run vLLM
and it does not make a boundary-fusion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
PHASE0_PACKET_REL = (
    "experimental/decode_microkernel/phase0/results/"
    "decode_microkernel_phase0_20260507T233130Z"
)
HYBRID_PACKET_REL = (
    "experimental/hybridkernel/phase2/results/"
    "hybridkernel_profiler_gate_20260507T212428Z"
)
DEFAULT_PHASE0_PACKET = ROOT / PHASE0_PACKET_REL
DEFAULT_HYBRID_PACKET = ROOT / HYBRID_PACKET_REL
DEFAULT_RESULTS_DIR = ROOT / "experimental/decode_microkernel/phase1/results"
PHASE0_PACKET_ID = "decode_microkernel_phase0_20260507T233130Z"
HYBRID_PACKET_ID = "hybridkernel_profiler_gate_20260507T212428Z"
SCHEMA_VERSION = "dmc_phase1_metrics_v1"

THRESHOLDS: dict[str, Any] = {
    "min_rows_by_role": {
        "primary_hybrid": 3,
        "same_family_control": 3,
        "cross_family_falsification": 3,
    },
    "min_warmup_iterations": 100,
    "min_measured_iterations": 1000,
    "max_abs_error": 1e-2,
    "max_rel_error": 1e-2,
    "min_launch_reduction_fraction": 0.25,
    "min_role_median_latency_reduction": {
        "primary_hybrid": 0.08,
        "same_family_control": 0.08,
        "cross_family_falsification": 0.05,
    },
    "bootstrap_ci95_lower_bound_min": 0.0,
}


@dataclass(frozen=True)
class Schedule:
    run_id: str
    row_role: str
    model: str
    seed: int
    element_count: int
    stages: int
    class_weights: dict[str, float]
    operation_labels: list[str]
    trace_derivation: str
    phase0_status: str
    source_nsys_artifact: str
    source_nsys_artifact_sha256: str
    source_server_log: str
    source_server_log_sha256: str
    source_client_log: str
    source_client_log_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_output(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
    except Exception as exc:  # pragma: no cover - environment-specific.
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


def ensure_fixed_packet(actual: Path, expected: Path, label: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{label} must be fixed preregistered packet {expected}; got {actual}")


def classify_kernel(name: str) -> set[str]:
    lower = name.lower()
    classes: set[str] = set()
    if "gemv" in lower or "cutlass" in lower or "cublas" in lower:
        classes.add("gemv")
    if "moe" in lower or "expert" in lower or "topkgating" in lower or "act_and_mul" in lower:
        classes.add("moe")
    if "selective_scan" in lower or "devicescan" in lower or "scan" in lower:
        classes.add("selective_scan")
    if "elementwise" in lower or "copy" in lower or "add" in lower or "mul" in lower or "silu" in lower:
        classes.add("elementwise")
    if "reduce" in lower or "red_" in lower or "scan" in lower:
        classes.add("reduction")
    return classes


def server_log_kernel_summary(log_path: Path) -> list[tuple[str, int, int]]:
    """Parse the committed Nsight Systems cuda_gpu_kern_sum text table."""

    rows: list[tuple[str, int, int]] = []
    in_table = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Executing 'cuda_gpu_kern_sum'" in line:
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("[7/") or stripped.startswith("Generated:"):
            break
        if not stripped[0].isdigit():
            continue
        parts = stripped.split(maxsplit=8)
        if len(parts) < 9:
            continue
        try:
            total_ns = int(parts[1])
            launches = int(parts[2])
        except ValueError:
            continue
        rows.append((parts[8], launches, total_ns))
    if not rows:
        raise ValueError(f"no cuda_gpu_kern_sum rows parsed from {log_path}")
    return rows


def class_breakdown_from_kernel_rows(rows: list[tuple[str, int, int]]) -> dict[str, dict[str, float | int]]:
    breakdown: dict[str, dict[str, float | int]] = {}
    for name, launches, total_ns in rows:
        classes = classify_kernel(name)
        if not classes:
            classes = {"elementwise"}
        for class_name in classes:
            item = breakdown.setdefault(class_name, {"launches": 0, "time_ms": 0.0})
            item["launches"] = int(item["launches"]) + int(launches)
            item["time_ms"] = float(item["time_ms"]) + int(total_ns) / 1e6
    return breakdown


def parse_seed(row: dict[str, Any], fallback: int) -> int:
    match = re.search(r"Seed\s+(\d+)", str(row.get("notes", "")))
    return int(match.group(1)) if match else fallback


def decode_tokens_from_source_row(row: dict[str, Any]) -> int:
    shape = row.get("batch_shape") or {}
    requests = int(shape.get("requests", 16))
    decode_tokens = int(shape.get("decode_tokens", 64))
    return max(1, requests * decode_tokens)


def sqlite_class_breakdown(sqlite_path: Path) -> dict[str, dict[str, float | int]]:
    with sqlite3.connect(sqlite_path) as conn:
        has_table = conn.execute(
            "select 1 from sqlite_master where type='table' and name='KERNEL_SUMMARY'"
        ).fetchone()
        if not has_table:
            raise ValueError(f"{sqlite_path} missing KERNEL_SUMMARY")
        rows = conn.execute("select name, launches, total_ns from KERNEL_SUMMARY").fetchall()
    breakdown: dict[str, dict[str, float | int]] = {}
    for name, launches, total_ns in rows:
        for class_name in classify_kernel(str(name)):
            item = breakdown.setdefault(class_name, {"launches": 0, "time_ms": 0.0})
            item["launches"] = int(item["launches"]) + int(launches)
            item["time_ms"] = float(item["time_ms"]) + int(total_ns) / 1e6
    return breakdown


def normalize_launch_weights(breakdown: dict[str, dict[str, float | int]]) -> dict[str, float]:
    launches = {key: int(value.get("launches", 0)) for key, value in breakdown.items()}
    total = sum(max(0, value) for value in launches.values())
    if total <= 0:
        return {"elementwise": 1.0}
    return {key: value / total for key, value in sorted(launches.items()) if value > 0}


def stage_count_from_weights(class_weights: dict[str, float], decode_tokens: int) -> int:
    diversity_bonus = len(class_weights)
    token_bonus = max(0, int(round(math.log2(max(1, decode_tokens))) - 8))
    dominant_bonus = int(round(4.0 * max(class_weights.values())))
    return max(4, min(10, 3 + diversity_bonus + token_bonus + dominant_bonus))


def operation_labels_from_weights(class_weights: dict[str, float], stages: int) -> list[str]:
    weighted_classes = sorted(class_weights, key=lambda name: (-class_weights[name], name))
    labels: list[str] = []
    for index in range(stages):
        class_name = weighted_classes[index % len(weighted_classes)]
        if class_name == "moe":
            labels.append("moe_gating_mix")
        elif class_name == "selective_scan":
            labels.append("selective_scan_state_update")
        elif class_name == "gemv":
            labels.append("gemv_epilogue_gated_affine")
        elif class_name == "reduction":
            labels.append("normalization_epilogue_mix")
        else:
            labels.append("elementwise_residual_mix")
    return labels


def build_schedules(*, phase0_packet: Path, hybrid_packet: Path) -> list[Schedule]:
    phase0_metrics = load_json(phase0_packet / "metrics.json")
    hybrid_metrics = load_json(hybrid_packet / "profiler_metrics.json")
    phase0_by_run = {str(row["run_id"]): row for row in phase0_metrics.get("admitted_rows", [])}
    excluded_by_run = {str(row["run_id"]): row for row in phase0_metrics.get("excluded_rows", [])}
    schedules: list[Schedule] = []

    for index, source_row in enumerate(hybrid_metrics.get("rows", [])):
        run_id = str(source_row["run_id"])
        role = str(source_row["row_role"])
        seed = parse_seed(source_row, fallback=20260507 + index)
        decode_tokens = decode_tokens_from_source_row(source_row)
        client_rel = f"logs/client_{run_id}.log"
        client_path = hybrid_packet / client_rel
        server_log_rel = f"logs/nsys_server_{run_id}.log"
        server_log_path = hybrid_packet / server_log_rel
        nsys_rel = str(source_row["nsys_artifact"])
        phase0_row = phase0_by_run.get(run_id)
        kernel_rows = server_log_kernel_summary(server_log_path)
        class_weights = normalize_launch_weights(class_breakdown_from_kernel_rows(kernel_rows))
        stages = stage_count_from_weights(class_weights, decode_tokens)
        operation_labels = operation_labels_from_weights(class_weights, stages)
        element_count = min(65536, max(4096, decode_tokens * 8))
        if phase0_row is not None:
            phase0_status = "admitted"
            derivation = "hybridkernel_server_log_cuda_gpu_kern_sum_and_phase0_admitted_row"
        else:
            phase0_status = "excluded:" + str(excluded_by_run.get(run_id, {}).get("reason", "not_admitted"))
            derivation = "hybridkernel_server_log_cuda_gpu_kern_sum_phase0_excluded_source_row"

        schedules.append(
            Schedule(
                run_id=run_id,
                row_role=role,
                model=str(source_row["model"]),
                seed=seed,
                element_count=element_count,
                stages=stages,
                class_weights=class_weights,
                operation_labels=operation_labels,
                trace_derivation=derivation,
                phase0_status=phase0_status,
                source_nsys_artifact=nsys_rel,
                source_nsys_artifact_sha256=str(source_row["nsys_artifact_sha256"]),
                source_server_log=server_log_rel,
                source_server_log_sha256=file_sha256(server_log_path),
                source_client_log=client_rel,
                source_client_log_sha256=file_sha256(client_path),
            )
        )
    return schedules


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    if not values:
        return {"ci95_low": float("nan"), "ci95_high": float("nan")}
    draws: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        draws.append(float(median(draw)))
    draws.sort()
    lo = int(0.025 * (len(draws) - 1))
    hi = int(0.975 * (len(draws) - 1))
    return {"ci95_low": draws[lo], "ci95_high": draws[hi]}


def tensor_sha256(tensor: Any) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return bytes_sha256(payload)


def load_triton_kernel() -> tuple[Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def packed_gating_replay_kernel(
        x_ptr,
        scale_ptr,
        bias_ptr,
        gate_scale_ptr,
        gate_bias_ptr,
        gate_base_ptr,
        residual_ptr,
        out_ptr,
        n_elements,
        STAGES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_base = tl.load(gate_base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        for index in tl.static_range(0, STAGES):
            scale = tl.load(scale_ptr + index).to(tl.float32)
            bias = tl.load(bias_ptr + index).to(tl.float32)
            gate_scale = tl.load(gate_scale_ptr + index).to(tl.float32)
            gate_bias = tl.load(gate_bias_ptr + index).to(tl.float32)
            gate_arg = gate_base * gate_scale + gate_bias
            gate = 1.0 / (1.0 + tl.exp(-gate_arg))
            value = (value * scale + bias) * gate + residual * (1.0 - gate)
            residual = residual * 0.998 + value * 0.002
        tl.store(out_ptr + offsets, value, mask=mask)

    return triton, packed_gating_replay_kernel


def make_inputs(schedule: Schedule) -> dict[str, Any]:
    import torch

    torch.manual_seed(schedule.seed)
    torch.cuda.manual_seed_all(schedule.seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(schedule.seed)
    x = torch.rand(schedule.element_count, device="cuda", dtype=torch.float32, generator=generator) + 0.25
    gate_base = torch.rand(
        schedule.element_count, device="cuda", dtype=torch.float32, generator=generator
    ) - 0.5
    residual = torch.rand(
        schedule.element_count, device="cuda", dtype=torch.float32, generator=generator
    ) + 0.1
    names = sorted(schedule.class_weights)
    scales: list[float] = []
    biases: list[float] = []
    gate_scales: list[float] = []
    gate_biases: list[float] = []
    for index in range(schedule.stages):
        name = names[index % len(names)]
        weight = float(schedule.class_weights[name])
        scales.append(1.0 + 0.003 * (index + 1) + 0.002 * weight)
        biases.append(0.0005 * (index + 1) * (1.0 + weight))
        gate_scales.append(0.70 + 0.01 * (index + 1) + 0.03 * weight)
        gate_biases.append(-0.05 + 0.004 * (index + 1) + 0.02 * weight)
    return {
        "x": x,
        "gate_base": gate_base,
        "residual": residual,
        "scales": torch.tensor(scales, device="cuda", dtype=torch.float32),
        "biases": torch.tensor(biases, device="cuda", dtype=torch.float32),
        "gate_scales": torch.tensor(gate_scales, device="cuda", dtype=torch.float32),
        "gate_biases": torch.tensor(gate_biases, device="cuda", dtype=torch.float32),
    }


def baseline_replay(
    x: Any,
    scales: Any,
    biases: Any,
    gate_scales: Any,
    gate_biases: Any,
    gate_base: Any,
    residual: Any,
    stages: int,
) -> Any:
    import torch

    output = x
    residual_state = residual
    for index in range(stages):
        gate = torch.sigmoid(gate_base * gate_scales[index] + gate_biases[index])
        output = (output * scales[index] + biases[index]) * gate + residual_state * (1.0 - gate)
        residual_state = residual_state * 0.998 + output * 0.002
    return output


def consolidated_replay(
    *,
    x: Any,
    scales: Any,
    biases: Any,
    gate_scales: Any,
    gate_biases: Any,
    gate_base: Any,
    residual: Any,
    stages: int,
    triton_module: Any,
    kernel: Any,
) -> Any:
    import torch

    block = 256
    output = torch.empty_like(x)
    grid = (triton_module.cdiv(x.numel(), block),)
    kernel[grid](
        x,
        scales,
        biases,
        gate_scales,
        gate_biases,
        gate_base,
        residual,
        output,
        x.numel(),
        STAGES=stages,
        BLOCK=block,
    )
    return output


def measure_cuda_event_medians(
    fn: Callable[[], Any],
    *,
    warmup: int,
    measured: int,
) -> tuple[list[float], Any]:
    import torch

    last_output = None
    for _ in range(warmup):
        last_output = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_output = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    torch.cuda.synchronize()
    return times, last_output


def cuda_kernel_events_from_profile(fn: Callable[[], Any]) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [
        {
            "name": event.name,
            "device_time_us": float(getattr(event, "device_time_total", 0.0)),
        }
        for event in prof.events()
        if str(getattr(event, "device_type", "")).endswith("CUDA")
    ]
    return {
        "kernel_call_count": len(events),
        "kernel_names": [event["name"] for event in events],
        "events": events,
    }


def run_schedule(
    *,
    schedule: Schedule,
    run_dir: Path,
    warmup: int,
    measured: int,
    triton_module: Any,
    kernel: Any,
) -> dict[str, Any]:
    import torch

    inputs = make_inputs(schedule)
    x = inputs["x"]
    gate_base = inputs["gate_base"]
    residual = inputs["residual"]
    scales = inputs["scales"]
    biases = inputs["biases"]
    gate_scales = inputs["gate_scales"]
    gate_biases = inputs["gate_biases"]
    input_sha = tensor_sha256(x)
    gate_base_sha = tensor_sha256(gate_base)
    residual_sha = tensor_sha256(residual)
    scale_sha = tensor_sha256(scales)
    bias_sha = tensor_sha256(biases)
    gate_scale_sha = tensor_sha256(gate_scales)
    gate_bias_sha = tensor_sha256(gate_biases)

    baseline_times, baseline_output = measure_cuda_event_medians(
        lambda: baseline_replay(
            x,
            scales,
            biases,
            gate_scales,
            gate_biases,
            gate_base,
            residual,
            schedule.stages,
        ),
        warmup=warmup,
        measured=measured,
    )
    consolidated_times, consolidated_output = measure_cuda_event_medians(
        lambda: consolidated_replay(
            x=x,
            scales=scales,
            biases=biases,
            gate_scales=gate_scales,
            gate_biases=gate_biases,
            gate_base=gate_base,
            residual=residual,
            stages=schedule.stages,
            triton_module=triton_module,
            kernel=kernel,
        ),
        warmup=warmup,
        measured=measured,
    )
    baseline_launch_audit = cuda_kernel_events_from_profile(
        lambda: baseline_replay(
            x,
            scales,
            biases,
            gate_scales,
            gate_biases,
            gate_base,
            residual,
            schedule.stages,
        )
    )
    consolidated_launch_audit = cuda_kernel_events_from_profile(
        lambda: consolidated_replay(
            x=x,
            scales=scales,
            biases=biases,
            gate_scales=gate_scales,
            gate_biases=gate_biases,
            gate_base=gate_base,
            residual=residual,
            stages=schedule.stages,
            triton_module=triton_module,
            kernel=kernel,
        )
    )

    diff = (baseline_output - consolidated_output).abs()
    max_abs_error = float(diff.max().item())
    denom = baseline_output.abs().clamp_min(1e-4)
    max_rel_error = float((diff / denom).max().item())
    baseline_ms_median = float(median(baseline_times))
    consolidated_ms_median = float(median(consolidated_times))
    baseline_launch_count = int(baseline_launch_audit["kernel_call_count"])
    consolidated_launch_count = int(consolidated_launch_audit["kernel_call_count"])
    latency_reduction_fraction = (
        (baseline_ms_median - consolidated_ms_median) / baseline_ms_median
        if baseline_ms_median > 0
        else float("nan")
    )
    launch_reduction_fraction = (
        (baseline_launch_count - consolidated_launch_count) / baseline_launch_count
    )

    samples_rel = Path("timings") / f"{schedule.run_id}.json"
    write_json(
        run_dir / samples_rel,
        {
            "schema_version": "dmc_phase1_timing_samples_v1",
            "run_id": schedule.run_id,
            "timing_source": "torch.cuda.Event",
            "baseline_ms": baseline_times,
            "consolidated_ms": consolidated_times,
        },
    )
    launch_audit_rel = Path("launch_audits") / f"{schedule.run_id}.json"
    write_json(
        run_dir / launch_audit_rel,
        {
            "schema_version": "dmc_phase1_launch_audit_v1",
            "run_id": schedule.run_id,
            "profiler": "torch.profiler",
            "baseline": baseline_launch_audit,
            "consolidated": consolidated_launch_audit,
        },
    )
    baseline_output_rel = Path("outputs") / f"{schedule.run_id}_baseline.pt"
    consolidated_output_rel = Path("outputs") / f"{schedule.run_id}_consolidated.pt"
    (run_dir / baseline_output_rel).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_output.detach().cpu(), run_dir / baseline_output_rel)
    torch.save(consolidated_output.detach().cpu(), run_dir / consolidated_output_rel)

    return {
        "run_id": schedule.run_id,
        "row_role": schedule.row_role,
        "model": schedule.model,
        "seed": schedule.seed,
        "warmup_iterations": warmup,
        "measured_iterations": measured,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "timing_source": "torch.cuda.Event",
        "gpu_side_timing": True,
        "source_nsys_artifact": schedule.source_nsys_artifact,
        "source_nsys_artifact_sha256": schedule.source_nsys_artifact_sha256,
        "source_server_log": schedule.source_server_log,
        "source_server_log_sha256": schedule.source_server_log_sha256,
        "source_client_log": schedule.source_client_log,
        "source_client_log_sha256": schedule.source_client_log_sha256,
        "phase0_row_status": schedule.phase0_status,
        "trace_derivation": schedule.trace_derivation,
        "element_count": schedule.element_count,
        "packed_gating_stages": schedule.stages,
        "trace_class_weights": schedule.class_weights,
        "trace_operation_labels": schedule.operation_labels,
        "baseline_launch_count": baseline_launch_count,
        "consolidated_launch_count": consolidated_launch_count,
        "launch_reduction_fraction": launch_reduction_fraction,
        "baseline_ms_median": baseline_ms_median,
        "consolidated_ms_median": consolidated_ms_median,
        "latency_reduction_fraction": latency_reduction_fraction,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "baseline_input_tensor_sha256": input_sha,
        "consolidated_input_tensor_sha256": input_sha,
        "gate_base_tensor_sha256": gate_base_sha,
        "residual_tensor_sha256": residual_sha,
        "scale_tensor_sha256": scale_sha,
        "bias_tensor_sha256": bias_sha,
        "gate_scale_tensor_sha256": gate_scale_sha,
        "gate_bias_tensor_sha256": gate_bias_sha,
        "timing_samples_path": str(samples_rel),
        "timing_samples_sha256": file_sha256(run_dir / samples_rel),
        "launch_audit_path": str(launch_audit_rel),
        "launch_audit_sha256": file_sha256(run_dir / launch_audit_rel),
        "baseline_output_path": str(baseline_output_rel),
        "baseline_output_sha256": file_sha256(run_dir / baseline_output_rel),
        "consolidated_output_path": str(consolidated_output_rel),
        "consolidated_output_sha256": file_sha256(run_dir / consolidated_output_rel),
    }


def role_summary(rows: list[dict[str, Any]], *, bootstrap_samples: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["row_role"]), []).append(row)
    summary: dict[str, Any] = {}
    for role, role_rows in sorted(grouped.items()):
        reductions = [float(row["latency_reduction_fraction"]) for row in role_rows]
        summary[role] = {
            "rows": len(role_rows),
            "latency_reduction_median": float(median(reductions)),
            "latency_reduction_bootstrap_ci95": bootstrap_ci(
                reductions, samples=bootstrap_samples, seed=20260507 + len(role)
            ),
            "launch_reduction_min": min(float(row["launch_reduction_fraction"]) for row in role_rows),
            "max_abs_error_max": max(float(row["max_abs_error"]) for row in role_rows),
            "max_rel_error_max": max(float(row["max_rel_error"]) for row in role_rows),
        }
    return summary


def packet_files(packet: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in sorted(set(rel_paths)):
        path = packet / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        )
    return rows


def build_manifest(phase0_packet: Path, hybrid_packet: Path, hybrid_metrics: dict[str, Any]) -> dict[str, Any]:
    phase0_files = [
        "checker_result.json",
        "command_metadata.json",
        "environment.json",
        "input_artifact_manifest.json",
        "logs/stdout.log",
        "logs/stderr.log",
        "metrics.json",
    ]
    hybrid_files = [
        "artifact_check.json",
        "profiler_metrics.json",
        "metadata/environment.json",
        "metadata/reduction_input_manifest.json",
    ]
    for row in hybrid_metrics.get("rows", []):
        hybrid_files.append(str(row["nsys_artifact"]))
        hybrid_files.append(f"logs/client_{row['run_id']}.log")
        hybrid_files.append(f"logs/nsys_server_{row['run_id']}.log")
    return {
        "schema_version": "dmc_phase1_input_manifest_v1",
        "created_at_utc": utc_now(),
        "packets": [
            {
                "packet_role": "phase0_pass_packet",
                "packet_id": PHASE0_PACKET_ID,
                "packet_path": str(phase0_packet.relative_to(ROOT)),
                "files": packet_files(phase0_packet, phase0_files),
            },
            {
                "packet_role": "hybridkernel_trace_packet",
                "packet_id": HYBRID_PACKET_ID,
                "packet_path": str(hybrid_packet.relative_to(ROOT)),
                "files": packet_files(hybrid_packet, hybrid_files),
            },
        ],
    }


def build_environment() -> dict[str, Any]:
    torch_info: dict[str, Any]
    triton_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:  # pragma: no cover - depends on GPU environment.
        torch_info = {"import_error": repr(exc)}
    try:
        import triton

        triton_info = {"version": getattr(triton, "__version__", None)}
    except Exception as exc:  # pragma: no cover - depends on GPU environment.
        triton_info = {"import_error": repr(exc)}
    return {
        "schema_version": "dmc_phase1_environment_v1",
        "created_at_utc": utc_now(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cwd": str(Path.cwd()),
        "git_sha": git_sha(),
        "torch": torch_info,
        "triton": triton_info,
        "commands": {
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"], timeout=90),
            "nvidia_smi": command_output(["nvidia-smi"], timeout=30),
            "nvcc_version": command_output(["nvcc", "--version"], timeout=30),
        },
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def choose_run_dir(results_dir: Path, run_id: str | None) -> Path:
    if run_id is None:
        run_id = "dmc_phase1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return results_dir / run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-packet", type=Path, default=DEFAULT_PHASE0_PACKET)
    parser.add_argument("--hybrid-packet", type=Path, default=DEFAULT_HYBRID_PACKET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measured", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    started_at = utc_now()
    phase0_packet = args.phase0_packet.resolve()
    hybrid_packet = args.hybrid_packet.resolve()
    run_dir = choose_run_dir(args.results_dir.resolve(), args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    exit_code = 0
    try:
        ensure_fixed_packet(phase0_packet, DEFAULT_PHASE0_PACKET, "Phase 0 packet")
        ensure_fixed_packet(hybrid_packet, DEFAULT_HYBRID_PACKET, "HybridKernel packet")
        phase0_checker = load_json(phase0_packet / "checker_result.json")
        hybrid_metrics = load_json(hybrid_packet / "profiler_metrics.json")
        if phase0_checker.get("decision") != "PASS":
            raise ValueError("fixed Phase 0 packet checker_result.json is not PASS")
        schedules = build_schedules(phase0_packet=phase0_packet, hybrid_packet=hybrid_packet)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DMC Phase 1 GPU-side replay timing")
        triton_module, kernel = load_triton_kernel()
        measured_rows: list[dict[str, Any]] = []
        for schedule in schedules:
            stdout_lines.append(
                json.dumps(
                    {
                        "event": "start_row",
                        "run_id": schedule.run_id,
                        "role": schedule.row_role,
                        "stages": schedule.stages,
                    },
                    sort_keys=True,
                )
            )
            measured_rows.append(
                run_schedule(
                    schedule=schedule,
                    run_dir=run_dir,
                    warmup=args.warmup,
                    measured=args.measured,
                    triton_module=triton_module,
                    kernel=kernel,
                )
            )

        metrics = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "phase0_packet": str(phase0_packet.relative_to(ROOT)),
            "phase0_packet_id": PHASE0_PACKET_ID,
            "phase0_checker_decision": phase0_checker.get("decision"),
            "phase0_metrics_sha256": file_sha256(phase0_packet / "metrics.json"),
            "hybrid_trace_packet": str(hybrid_packet.relative_to(ROOT)),
            "hybrid_trace_packet_id": HYBRID_PACKET_ID,
            "hybrid_profiler_metrics_sha256": file_sha256(hybrid_packet / "profiler_metrics.json"),
            "thresholds": THRESHOLDS,
            "method": {
                "name": "trace_derived_triton_packed_gating_replay",
                "baseline": (
                    "repeated PyTorch CUDA gating/routing/state-update micro-operations "
                    "selected from fixed Nsight kernel-family summaries"
                ),
                "consolidated": "single Triton packed gating replay kernel",
                "claim_scope": "decode micro-operation replay only",
                "boundary_fusion_claim": False,
                "cpu_only_benchmark": False,
                "trace_selection_source": "committed HybridKernel logs/nsys_server_<run_id>.log cuda_gpu_kern_sum tables",
            },
            "timing": {
                "source": "torch.cuda.Event",
                "gpu_side": True,
                "warmup_iterations_requested": args.warmup,
                "measured_iterations_requested": args.measured,
            },
            "rows": measured_rows,
            "role_summary": role_summary(measured_rows, bootstrap_samples=args.bootstrap_samples),
            "bootstrap_samples": args.bootstrap_samples,
            "interpretation": (
                "Phase 1 is a trace-derived consolidation replay gate. PASS would justify "
                "Phase 2 serving integration; it is not a vLLM serving speedup claim."
            ),
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(
            run_dir / "replay_schedule.json",
            {
                "schema_version": "dmc_phase1_replay_schedule_v1",
                "created_at_utc": utc_now(),
                "schedule_source": "all fixed HybridKernel profiler rows in source order",
                "rows": [schedule.__dict__ for schedule in schedules],
            },
        )
        stdout_lines.append(
            json.dumps(
                {
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "rows": len(measured_rows),
                    "roles": metrics["role_summary"],
                },
                sort_keys=True,
            )
        )
    except Exception as exc:
        stderr_lines.append(repr(exc))
        write_json(
            run_dir / "metrics.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "phase0_packet": str(phase0_packet),
                "hybrid_trace_packet": str(hybrid_packet),
                "thresholds": THRESHOLDS,
                "rows": [],
                "infra_error": repr(exc),
            },
        )
        write_json(
            run_dir / "replay_schedule.json",
            {
                "schema_version": "dmc_phase1_replay_schedule_v1",
                "created_at_utc": utc_now(),
                "schedule_error": repr(exc),
                "rows": [],
            },
        )
        exit_code = 2

    try:
        hybrid_metrics_for_manifest = (
            load_json(hybrid_packet / "profiler_metrics.json")
            if (hybrid_packet / "profiler_metrics.json").exists()
            else {"rows": []}
        )
        write_json(run_dir / "input_artifact_manifest.json", build_manifest(phase0_packet, hybrid_packet, hybrid_metrics_for_manifest))
    except Exception as exc:  # pragma: no cover - only exercised by broken packets.
        write_json(
            run_dir / "input_artifact_manifest.json",
            {
                "schema_version": "dmc_phase1_input_manifest_v1",
                "created_at_utc": utc_now(),
                "manifest_error": repr(exc),
                "packets": [],
            },
        )
    write_json(run_dir / "environment.json", build_environment())
    command_metadata = {
        "schema_version": "dmc_phase1_command_v1",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "cwd": str(Path.cwd()),
        "git_sha": git_sha(),
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "phase0_packet": str(phase0_packet),
        "hybrid_packet": str(hybrid_packet),
        "run_dir": str(run_dir),
        "warmup": args.warmup,
        "measured": args.measured,
        "bootstrap_samples": args.bootstrap_samples,
        "path_python": shutil.which("python"),
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
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
