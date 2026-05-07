#!/usr/bin/env python3
"""Check a Decode Microkernel Consolidation Phase 1 result packet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PHASE0_PACKET_REL = (
    "experimental/decode_microkernel/phase0/results/"
    "decode_microkernel_phase0_20260507T233130Z"
)
HYBRID_PACKET_REL = (
    "experimental/hybridkernel/phase2/results/"
    "hybridkernel_profiler_gate_20260507T212428Z"
)
PHASE0_PACKET = ROOT / PHASE0_PACKET_REL
HYBRID_PACKET = ROOT / HYBRID_PACKET_REL
PHASE0_PACKET_ID = "decode_microkernel_phase0_20260507T233130Z"
HYBRID_PACKET_ID = "hybridkernel_profiler_gate_20260507T212428Z"

PASS_DECISION = "PASS_DMC_PHASE1_CONSOLIDATED_REPLAY"
KILL_DECISION = "KILL_DMC_PHASE1_NO_REPLAY_GAIN"
INFRA_DECISION = "FAIL_INFRA_DMC_PHASE1"

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

REQUIRED_PACKET_FILES = [
    "environment.json",
    "input_artifact_manifest.json",
    "metrics.json",
    "replay_schedule.json",
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


def is_close(left: float, right: float, *, tol: float = 1e-9) -> bool:
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def source_relative(path_text: str) -> Path | None:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def packet_by_role(manifest: dict[str, Any], role: str) -> dict[str, Any] | None:
    for packet in manifest.get("packets", []):
        if isinstance(packet, dict) and packet.get("packet_role") == role:
            return packet
    return None


def validate_manifest_packet(
    *,
    packet: dict[str, Any] | None,
    packet_role: str,
    packet_id: str,
    packet_path: str,
    absolute_packet: Path,
    required_files: list[str],
    infra_reasons: list[str],
) -> None:
    if packet is None:
        infra_reasons.append(f"manifest missing {packet_role}")
        return
    if packet.get("packet_id") != packet_id:
        infra_reasons.append(f"manifest {packet_role} packet_id mismatch")
    if packet.get("packet_path") != packet_path:
        infra_reasons.append(f"manifest {packet_role} packet_path mismatch")
    files = {
        str(item.get("path")): item
        for item in packet.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    for rel in required_files:
        path = absolute_packet / rel
        item = files.get(rel)
        if item is None:
            infra_reasons.append(f"manifest {packet_role} missing file {rel}")
        elif not path.is_file():
            infra_reasons.append(f"fixed source artifact missing on disk: {packet_path}/{rel}")
        else:
            if item.get("exists") is not True:
                infra_reasons.append(f"manifest {packet_role}/{rel} exists flag is not true")
            if item.get("bytes") != path.stat().st_size:
                infra_reasons.append(f"manifest {packet_role}/{rel} byte count mismatch")
            if item.get("sha256") != file_sha256(path):
                infra_reasons.append(f"manifest {packet_role}/{rel} sha256 mismatch")


def phase0_checker_decision() -> str:
    checker_path = ROOT / "experimental/decode_microkernel/phase0/check_dmc_phase0.py"
    spec = importlib.util.spec_from_file_location("dmc_phase0_checker_for_phase1", checker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Phase 0 checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate(PHASE0_PACKET.resolve())
    return str(result.get("decision"))


def validate_fixed_inputs(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    command: dict[str, Any],
    rows: list[dict[str, Any]],
    infra_reasons: list[str],
) -> None:
    if metrics.get("phase0_packet") != PHASE0_PACKET_REL:
        infra_reasons.append("metrics.phase0_packet does not match fixed preregistered Phase 0 packet")
    if metrics.get("phase0_packet_id") != PHASE0_PACKET_ID:
        infra_reasons.append("metrics.phase0_packet_id mismatch")
    if metrics.get("phase0_checker_decision") != "PASS":
        infra_reasons.append("metrics.phase0_checker_decision is not PASS")
    if metrics.get("phase0_metrics_sha256") != file_sha256(PHASE0_PACKET / "metrics.json"):
        infra_reasons.append("metrics.phase0_metrics_sha256 mismatch")
    if metrics.get("hybrid_trace_packet") != HYBRID_PACKET_REL:
        infra_reasons.append("metrics.hybrid_trace_packet does not match fixed trace packet")
    if metrics.get("hybrid_trace_packet_id") != HYBRID_PACKET_ID:
        infra_reasons.append("metrics.hybrid_trace_packet_id mismatch")
    if metrics.get("hybrid_profiler_metrics_sha256") != file_sha256(HYBRID_PACKET / "profiler_metrics.json"):
        infra_reasons.append("metrics.hybrid_profiler_metrics_sha256 mismatch")

    for field, expected in [
        ("phase0_packet", PHASE0_PACKET.resolve()),
        ("hybrid_packet", HYBRID_PACKET.resolve()),
    ]:
        try:
            actual = Path(str(command.get(field, ""))).resolve()
        except OSError:
            actual = Path(str(command.get(field, "")))
        if actual != expected:
            infra_reasons.append(f"command_metadata.{field} does not match fixed preregistered packet")

    try:
        checker_result = load_json(PHASE0_PACKET / "checker_result.json")
        if checker_result.get("decision") != "PASS":
            infra_reasons.append("fixed Phase 0 checker_result.json is not PASS")
        if phase0_checker_decision() != "PASS":
            infra_reasons.append("fresh Phase 0 checker evaluation is not PASS")
    except Exception as exc:
        infra_reasons.append(f"cannot revalidate fixed Phase 0 PASS packet: {exc!r}")

    phase0_required = [
        "checker_result.json",
        "command_metadata.json",
        "environment.json",
        "input_artifact_manifest.json",
        "logs/stdout.log",
        "logs/stderr.log",
        "metrics.json",
    ]
    hybrid_metrics = load_json(HYBRID_PACKET / "profiler_metrics.json")
    hybrid_required = [
        "artifact_check.json",
        "profiler_metrics.json",
        "metadata/environment.json",
        "metadata/reduction_input_manifest.json",
    ]
    for source_row in hybrid_metrics.get("rows", []):
        hybrid_required.append(str(source_row["nsys_artifact"]))
        hybrid_required.append(f"logs/client_{source_row['run_id']}.log")
    validate_manifest_packet(
        packet=packet_by_role(manifest, "phase0_pass_packet"),
        packet_role="phase0_pass_packet",
        packet_id=PHASE0_PACKET_ID,
        packet_path=PHASE0_PACKET_REL,
        absolute_packet=PHASE0_PACKET,
        required_files=phase0_required,
        infra_reasons=infra_reasons,
    )
    validate_manifest_packet(
        packet=packet_by_role(manifest, "hybridkernel_trace_packet"),
        packet_role="hybridkernel_trace_packet",
        packet_id=HYBRID_PACKET_ID,
        packet_path=HYBRID_PACKET_REL,
        absolute_packet=HYBRID_PACKET,
        required_files=hybrid_required,
        infra_reasons=infra_reasons,
    )

    expected_by_run = {str(row["run_id"]): row for row in hybrid_metrics.get("rows", [])}
    actual_run_ids = [str(row.get("run_id")) for row in rows]
    if sorted(actual_run_ids) != sorted(expected_by_run):
        infra_reasons.append("metrics rows are not exactly the fixed 9 HybridKernel source rows")
    if len(set(actual_run_ids)) != len(actual_run_ids):
        infra_reasons.append("metrics rows contain duplicate run_id values")
    for row in rows:
        run_id = str(row.get("run_id"))
        source_row = expected_by_run.get(run_id)
        if source_row is None:
            continue
        if row.get("row_role") != source_row.get("row_role"):
            infra_reasons.append(f"{run_id}: row_role does not match fixed trace row")
        if row.get("model") != source_row.get("model"):
            infra_reasons.append(f"{run_id}: model does not match fixed trace row")
        nsys_rel = source_relative(str(row.get("source_nsys_artifact", "")))
        client_rel = source_relative(str(row.get("source_client_log", "")))
        if nsys_rel is None or client_rel is None:
            infra_reasons.append(f"{run_id}: source artifact paths must be packet-relative")
            continue
        if str(nsys_rel) != str(source_row.get("nsys_artifact")):
            infra_reasons.append(f"{run_id}: source_nsys_artifact does not match fixed trace row")
        nsys_path = HYBRID_PACKET / nsys_rel
        client_path = HYBRID_PACKET / client_rel
        if not nsys_path.is_file():
            infra_reasons.append(f"{run_id}: missing source_nsys_artifact")
        elif row.get("source_nsys_artifact_sha256") != file_sha256(nsys_path):
            infra_reasons.append(f"{run_id}: source_nsys_artifact_sha256 mismatch")
        if not client_path.is_file():
            infra_reasons.append(f"{run_id}: missing source_client_log")
        elif row.get("source_client_log_sha256") != file_sha256(client_path):
            infra_reasons.append(f"{run_id}: source_client_log_sha256 mismatch")


def load_timing_samples(run_dir: Path, row: dict[str, Any], infra_reasons: list[str]) -> dict[str, Any] | None:
    run_id = str(row.get("run_id"))
    rel = source_relative(str(row.get("timing_samples_path", "")))
    if rel is None:
        infra_reasons.append(f"{run_id}: timing_samples_path must be relative")
        return None
    path = run_dir / rel
    if not path.is_file():
        infra_reasons.append(f"{run_id}: missing timing samples file")
        return None
    if row.get("timing_samples_sha256") != file_sha256(path):
        infra_reasons.append(f"{run_id}: timing_samples_sha256 mismatch")
    try:
        samples = load_json(path)
    except Exception as exc:
        infra_reasons.append(f"{run_id}: invalid timing samples JSON: {exc!r}")
        return None
    if samples.get("timing_source") != "torch.cuda.Event":
        infra_reasons.append(f"{run_id}: timing samples are not CUDA event samples")
    return samples


def validate_row_infra(run_dir: Path, row: dict[str, Any], infra_reasons: list[str]) -> None:
    run_id = str(row.get("run_id"))
    if row.get("timing_source") != "torch.cuda.Event" or row.get("gpu_side_timing") is not True:
        infra_reasons.append(f"{run_id}: row timing is not GPU-side CUDA event timing")
    if int(row.get("warmup_iterations", -1)) < THRESHOLDS["min_warmup_iterations"]:
        infra_reasons.append(f"{run_id}: warmup iterations below preregistered minimum")
    if int(row.get("measured_iterations", -1)) < THRESHOLDS["min_measured_iterations"]:
        infra_reasons.append(f"{run_id}: measured iterations below preregistered minimum")
    if not isinstance(row.get("seed"), int):
        infra_reasons.append(f"{run_id}: fixed integer seed missing")
    if row.get("baseline_input_tensor_sha256") != row.get("consolidated_input_tensor_sha256"):
        infra_reasons.append(f"{run_id}: baseline and consolidated input tensor hashes differ")
    if not str(row.get("baseline_input_tensor_sha256", "")).startswith("sha256:"):
        infra_reasons.append(f"{run_id}: input tensor hash missing")
    if int(row.get("baseline_launch_count", 0)) <= 0 or int(row.get("consolidated_launch_count", 0)) <= 0:
        infra_reasons.append(f"{run_id}: launch counts must be positive")
    if int(row.get("baseline_launch_count", 0)) <= int(row.get("consolidated_launch_count", 0)):
        infra_reasons.append(f"{run_id}: consolidated path does not have fewer launches")
    if float(row.get("baseline_ms_median", 0.0)) <= 0.0:
        infra_reasons.append(f"{run_id}: nonpositive baseline_ms_median")
    if float(row.get("consolidated_ms_median", 0.0)) <= 0.0:
        infra_reasons.append(f"{run_id}: nonpositive consolidated_ms_median")

    samples = load_timing_samples(run_dir, row, infra_reasons)
    if samples is None:
        return
    baseline_samples = samples.get("baseline_ms")
    consolidated_samples = samples.get("consolidated_ms")
    if not isinstance(baseline_samples, list) or not isinstance(consolidated_samples, list):
        infra_reasons.append(f"{run_id}: timing samples must include baseline_ms and consolidated_ms lists")
        return
    measured = int(row.get("measured_iterations", -1))
    if len(baseline_samples) != measured or len(consolidated_samples) != measured:
        infra_reasons.append(f"{run_id}: timing sample count does not match measured_iterations")
        return
    if len(baseline_samples) < THRESHOLDS["min_measured_iterations"]:
        infra_reasons.append(f"{run_id}: timing sample count below preregistered minimum")
    if any(float(value) <= 0.0 or not math.isfinite(float(value)) for value in baseline_samples):
        infra_reasons.append(f"{run_id}: invalid baseline timing sample")
    if any(float(value) <= 0.0 or not math.isfinite(float(value)) for value in consolidated_samples):
        infra_reasons.append(f"{run_id}: invalid consolidated timing sample")
    if not is_close(float(row.get("baseline_ms_median")), float(median(float(v) for v in baseline_samples))):
        infra_reasons.append(f"{run_id}: baseline_ms_median does not match timing samples")
    if not is_close(
        float(row.get("consolidated_ms_median")),
        float(median(float(v) for v in consolidated_samples)),
    ):
        infra_reasons.append(f"{run_id}: consolidated_ms_median does not match timing samples")


def rows_by_role(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("row_role")), []).append(row)
    return grouped


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        draws.append(float(median(draw)))
    draws.sort()
    lo = int(0.025 * (len(draws) - 1))
    hi = int(0.975 * (len(draws) - 1))
    return {"ci95_low": draws[lo], "ci95_high": draws[hi]}


def validate_role_summary(
    *,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    infra_reasons: list[str],
) -> dict[str, dict[str, Any]]:
    grouped = rows_by_role(rows)
    role_metrics: dict[str, dict[str, Any]] = {}
    for role, minimum in THRESHOLDS["min_rows_by_role"].items():
        actual = len(grouped.get(role, []))
        if actual < minimum:
            infra_reasons.append(f"{role}: row count {actual} below required {minimum}")
    bootstrap_samples = int(metrics.get("bootstrap_samples", 0))
    if bootstrap_samples <= 0:
        infra_reasons.append("bootstrap_samples must be positive")
        bootstrap_samples = 1
    for role, role_rows in grouped.items():
        reductions = [float(row["latency_reduction_fraction"]) for row in role_rows]
        role_metrics[role] = {
            "rows": len(role_rows),
            "latency_reduction_median": float(median(reductions)),
            "latency_reduction_bootstrap_ci95": bootstrap_ci(
                reductions, samples=bootstrap_samples, seed=20260507 + len(role)
            ),
            "launch_reduction_min": min(float(row["launch_reduction_fraction"]) for row in role_rows),
            "max_abs_error_max": max(float(row["max_abs_error"]) for row in role_rows),
            "max_rel_error_max": max(float(row["max_rel_error"]) for row in role_rows),
        }
    recorded = metrics.get("role_summary", {})
    if not isinstance(recorded, dict):
        infra_reasons.append("metrics.role_summary must be an object")
        return role_metrics
    for role, values in role_metrics.items():
        recorded_values = recorded.get(role)
        if not isinstance(recorded_values, dict):
            infra_reasons.append(f"metrics.role_summary missing {role}")
            continue
        for key in ["rows", "latency_reduction_median", "launch_reduction_min", "max_abs_error_max", "max_rel_error_max"]:
            if key == "rows":
                if int(recorded_values.get(key, -1)) != int(values[key]):
                    infra_reasons.append(f"{role}: role_summary {key} mismatch")
            elif not is_close(float(recorded_values.get(key, float("nan"))), float(values[key])):
                infra_reasons.append(f"{role}: role_summary {key} mismatch")
        recorded_ci = recorded_values.get("latency_reduction_bootstrap_ci95", {})
        computed_ci = values["latency_reduction_bootstrap_ci95"]
        for key in ["ci95_low", "ci95_high"]:
            if not is_close(float(recorded_ci.get(key, float("nan"))), float(computed_ci[key])):
                infra_reasons.append(f"{role}: role_summary bootstrap {key} mismatch")
    return role_metrics


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra_reasons: list[str] = []
    kill_reasons: list[str] = []

    for rel_path in REQUIRED_PACKET_FILES:
        if not (run_dir / rel_path).exists():
            infra_reasons.append(f"missing required packet file: {rel_path}")
    if infra_reasons:
        return {"decision": INFRA_DECISION, "run_dir": str(run_dir), "reasons": infra_reasons}

    try:
        metrics = load_json(run_dir / "metrics.json")
        manifest = load_json(run_dir / "input_artifact_manifest.json")
        command = load_json(run_dir / "command_metadata.json")
        schedule = load_json(run_dir / "replay_schedule.json")
    except Exception as exc:
        return {
            "decision": INFRA_DECISION,
            "run_dir": str(run_dir),
            "reasons": [f"bad JSON in required artifacts: {exc!r}"],
        }

    if metrics.get("schema_version") != "dmc_phase1_metrics_v1":
        infra_reasons.append("metrics schema_version is not dmc_phase1_metrics_v1")
    if manifest.get("schema_version") != "dmc_phase1_input_manifest_v1":
        infra_reasons.append("input manifest schema_version mismatch")
    if schedule.get("schema_version") != "dmc_phase1_replay_schedule_v1":
        infra_reasons.append("replay_schedule schema_version mismatch")
    if command.get("schema_version") != "dmc_phase1_command_v1":
        infra_reasons.append("command_metadata schema_version mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra_reasons.append("threshold metadata mismatch")
    method = metrics.get("method", {})
    if method.get("name") != "trace_derived_triton_packed_affine_replay":
        infra_reasons.append("method.name is not the preregistered consolidated replay method")
    if method.get("boundary_fusion_claim") is not False:
        infra_reasons.append("method metadata must explicitly reject boundary-fusion claim scope")
    if method.get("cpu_only_benchmark") is not False:
        infra_reasons.append("method metadata must explicitly reject CPU-only benchmarking")
    timing = metrics.get("timing", {})
    if timing.get("source") != "torch.cuda.Event" or timing.get("gpu_side") is not True:
        infra_reasons.append("metrics timing source is not GPU-side CUDA event timing")

    rows = metrics.get("rows", [])
    if not isinstance(rows, list):
        infra_reasons.append("metrics.rows must be a list")
        rows = []
    schedule_rows = schedule.get("rows", [])
    if not isinstance(schedule_rows, list):
        infra_reasons.append("replay_schedule.rows must be a list")
        schedule_rows = []
    if len(schedule_rows) != len(rows):
        infra_reasons.append("replay schedule row count does not match measured rows")

    validate_fixed_inputs(
        metrics=metrics,
        manifest=manifest,
        command=command,
        rows=rows,
        infra_reasons=infra_reasons,
    )

    for row in rows:
        try:
            validate_row_infra(run_dir, row, infra_reasons)
            baseline = float(row["baseline_ms_median"])
            consolidated = float(row["consolidated_ms_median"])
            baseline_launches = int(row["baseline_launch_count"])
            consolidated_launches = int(row["consolidated_launch_count"])
            expected_latency = (baseline - consolidated) / baseline
            expected_launch = (baseline_launches - consolidated_launches) / baseline_launches
            if not is_close(float(row["latency_reduction_fraction"]), expected_latency):
                infra_reasons.append(f"{row.get('run_id')}: latency_reduction_fraction formula mismatch")
            if not is_close(float(row["launch_reduction_fraction"]), expected_launch):
                infra_reasons.append(f"{row.get('run_id')}: launch_reduction_fraction formula mismatch")
            if not str(row.get("trace_derivation", "")).startswith(
                ("phase0_", "source_", "role_template_from_phase0_")
            ):
                infra_reasons.append(f"{row.get('run_id')}: trace derivation is not auditable")
        except Exception as exc:
            infra_reasons.append(f"{row.get('run_id')}: malformed metric row: {exc!r}")

    role_metrics = validate_role_summary(rows=rows, metrics=metrics, infra_reasons=infra_reasons)

    if infra_reasons:
        return {"decision": INFRA_DECISION, "run_dir": str(run_dir), "reasons": infra_reasons}

    for row in rows:
        run_id = str(row["run_id"])
        if float(row["max_abs_error"]) > THRESHOLDS["max_abs_error"]:
            kill_reasons.append(
                f"{run_id}: max_abs_error {float(row['max_abs_error']):.8f} exceeds "
                f"{THRESHOLDS['max_abs_error']:.8f}"
            )
        if float(row["max_rel_error"]) > THRESHOLDS["max_rel_error"]:
            kill_reasons.append(
                f"{run_id}: max_rel_error {float(row['max_rel_error']):.8f} exceeds "
                f"{THRESHOLDS['max_rel_error']:.8f}"
            )
        if float(row["launch_reduction_fraction"]) < THRESHOLDS["min_launch_reduction_fraction"]:
            kill_reasons.append(
                f"{run_id}: launch_reduction_fraction {float(row['launch_reduction_fraction']):.8f} "
                f"below {THRESHOLDS['min_launch_reduction_fraction']:.8f}"
            )

    for role, minimum in THRESHOLDS["min_role_median_latency_reduction"].items():
        actual = float(role_metrics[role]["latency_reduction_median"])
        if actual < float(minimum):
            kill_reasons.append(
                f"{role}: median latency reduction {actual:.8f} below {float(minimum):.8f}"
            )
        ci_low = float(role_metrics[role]["latency_reduction_bootstrap_ci95"]["ci95_low"])
        if ci_low <= float(THRESHOLDS["bootstrap_ci95_lower_bound_min"]):
            kill_reasons.append(
                f"{role}: bootstrap CI lower bound {ci_low:.8f} is not greater than 0"
            )

    if kill_reasons:
        return {
            "decision": KILL_DECISION,
            "run_dir": str(run_dir),
            "reasons": kill_reasons,
            "role_metrics": role_metrics,
        }
    return {
        "decision": PASS_DECISION,
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
    return 0 if result["decision"] == PASS_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
