#!/usr/bin/env python3
"""Check a Decode Microkernel Consolidation Phase 2 serving result packet."""

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
PHASE1_PACKET_REL = (
    "experimental/decode_microkernel/phase1/results/"
    "dmc_phase1_20260508T000525Z"
)
PHASE1_PACKET = ROOT / PHASE1_PACKET_REL
PHASE1_PACKET_ID = "dmc_phase1_20260508T000525Z"
PHASE1_PASS_DECISION = "PASS_DMC_PHASE1_CONSOLIDATED_REPLAY"

PASS_DECISION = "PASS_DMC_PHASE2_SERVING_GAIN"
KILL_DECISION = "KILL_DMC_PHASE2_NO_SERVING_GAIN"
INFRA_DECISION = "FAIL_INFRA_DMC_PHASE2"

ROLE_MODELS = {
    "primary": "ibm-granite/granite-4.0-h-tiny",
    "same_family": "ibm-granite/granite-4.0-h-small",
    "cross_family": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
}

THRESHOLDS: dict[str, Any] = {
    "min_rows_by_role": {"primary": 12, "same_family": 12, "cross_family": 12},
    "min_row_launch_reduction_fraction": 0.10,
    "min_role_median_decode_latency_reduction": {
        "primary": 0.05,
        "same_family": 0.05,
        "cross_family": 0.03,
    },
    "bootstrap_ci95_lower_bound_min_exclusive": 0.0,
    "require_positive_median_tokens_per_second_gain": True,
}

REQUIRED_PACKET_FILES = [
    "environment.json",
    "input_artifact_manifest.json",
    "metrics.json",
    "command_metadata.json",
    "prompt_manifest.json",
    "model_diagnostics.json",
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


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_close(left: float, right: float, *, tol: float = 1e-9) -> bool:
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def source_relative(path_text: str) -> Path | None:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        return None
    return path


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    payload = json.dumps(prompts, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return bytes_sha256(payload)


def percentile_nearest_rank(values: list[float], percent: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percent / 100.0 * len(ordered)) - 1))
    return float(ordered[index])


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


def phase1_checker_decision() -> str:
    checker_path = ROOT / "experimental/decode_microkernel/phase1/check_dmc_phase1.py"
    spec = importlib.util.spec_from_file_location("dmc_phase1_checker_for_phase2", checker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Phase 1 checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate(PHASE1_PACKET.resolve())
    return str(result.get("decision"))


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


def validate_fixed_inputs(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    command: dict[str, Any],
    infra_reasons: list[str],
) -> None:
    if metrics.get("phase1_packet") != PHASE1_PACKET_REL:
        infra_reasons.append("metrics.phase1_packet does not match fixed preregistered Phase 1 packet")
    if metrics.get("phase1_packet_id") != PHASE1_PACKET_ID:
        infra_reasons.append("metrics.phase1_packet_id mismatch")
    if metrics.get("phase1_checker_decision") != PHASE1_PASS_DECISION:
        infra_reasons.append("metrics.phase1_checker_decision is not the fixed Phase 1 PASS decision")
    if metrics.get("phase1_checker_result_sha256") != file_sha256(PHASE1_PACKET / "checker_result.json"):
        infra_reasons.append("metrics.phase1_checker_result_sha256 mismatch")
    if metrics.get("phase1_metrics_sha256") != file_sha256(PHASE1_PACKET / "metrics.json"):
        infra_reasons.append("metrics.phase1_metrics_sha256 mismatch")
    try:
        actual = Path(str(command.get("phase1_packet", ""))).resolve()
        if actual != PHASE1_PACKET.resolve():
            infra_reasons.append("command_metadata.phase1_packet does not match fixed preregistered packet")
    except OSError:
        infra_reasons.append("command_metadata.phase1_packet is not a valid path")
    try:
        checker_result = load_json(PHASE1_PACKET / "checker_result.json")
        if checker_result.get("decision") != PHASE1_PASS_DECISION:
            infra_reasons.append("fixed Phase 1 checker_result.json is not PASS_DMC_PHASE1_CONSOLIDATED_REPLAY")
        if phase1_checker_decision() != PHASE1_PASS_DECISION:
            infra_reasons.append("fresh Phase 1 checker evaluation is not PASS_DMC_PHASE1_CONSOLIDATED_REPLAY")
    except Exception as exc:
        infra_reasons.append(f"cannot revalidate fixed Phase 1 PASS packet: {exc!r}")

    phase1_required = [
        "checker_result.json",
        "command_metadata.json",
        "environment.json",
        "input_artifact_manifest.json",
        "logs/stdout.log",
        "logs/stderr.log",
        "metrics.json",
        "replay_schedule.json",
    ]
    validate_manifest_packet(
        packet=packet_by_role(manifest, "phase1_pass_packet"),
        packet_role="phase1_pass_packet",
        packet_id=PHASE1_PACKET_ID,
        packet_path=PHASE1_PACKET_REL,
        absolute_packet=PHASE1_PACKET,
        required_files=phase1_required,
        infra_reasons=infra_reasons,
    )


def validate_prompt_manifest(
    *, prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra_reasons: list[str]
) -> set[int]:
    if prompt_manifest.get("schema_version") != "dmc_phase2_prompt_manifest_v1":
        infra_reasons.append("prompt_manifest schema_version mismatch")
    if prompt_manifest.get("source") != "AIME-2025":
        infra_reasons.append("prompt_manifest.source is not AIME-2025")
    if prompt_manifest.get("selection") != "deterministic_indices_0_11":
        infra_reasons.append("prompt_manifest.selection is not deterministic_indices_0_11")
    prompts = prompt_manifest.get("prompts", [])
    if not isinstance(prompts, list):
        infra_reasons.append("prompt_manifest.prompts must be a list")
        prompts = []
    indices: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for item in prompts:
        if not isinstance(item, dict):
            infra_reasons.append("prompt_manifest contains a non-object prompt row")
            continue
        try:
            index = int(item["index"])
        except Exception:
            infra_reasons.append("prompt_manifest row missing integer index")
            continue
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            infra_reasons.append(f"prompt {index}: missing prompt text")
        indices.add(index)
        normalized.append(
            {
                "index": index,
                "prompt_id": str(item.get("prompt_id", index)),
                "prompt": item.get("prompt"),
                "answer": item.get("answer"),
            }
        )
    if sorted(indices) != list(range(12)):
        infra_reasons.append("prompt indices are not exactly 0-11")
    if int(prompt_manifest.get("prompt_count", -1)) != 12 or len(prompts) != 12:
        infra_reasons.append("prompt_manifest must contain exactly 12 prompts")
    if normalized and prompt_manifest.get("prompt_sha256") != prompt_payload_sha256(normalized):
        infra_reasons.append("prompt_manifest.prompt_sha256 does not match prompt payload")
    if metrics.get("prompt_source") != "AIME-2025":
        infra_reasons.append("metrics.prompt_source is not AIME-2025")
    if metrics.get("prompt_selection") != "deterministic_indices_0_11":
        infra_reasons.append("metrics.prompt_selection mismatch")
    if metrics.get("prompt_sha256") != prompt_manifest.get("prompt_sha256"):
        infra_reasons.append("metrics.prompt_sha256 does not match prompt manifest")
    if not str(metrics.get("prompt_sha256", "")).startswith("sha256:"):
        infra_reasons.append("metrics.prompt_sha256 missing")
    return indices


def validate_environment(environment: dict[str, Any], infra_reasons: list[str]) -> None:
    if environment.get("schema_version") != "dmc_phase2_environment_v1":
        infra_reasons.append("environment schema_version mismatch")
    torch_env = environment.get("torch", {})
    if torch_env.get("cuda_available") is not True:
        infra_reasons.append("environment torch.cuda_available is not true")
    if not torch_env.get("devices"):
        infra_reasons.append("environment does not record a CUDA device")
    vllm_env = environment.get("packages", {}).get("vllm", {})
    if vllm_env.get("available") is not True:
        infra_reasons.append("environment does not record an importable vLLM package")


def validate_model_diagnostics(model_diag: dict[str, Any], infra_reasons: list[str]) -> None:
    if model_diag.get("schema_version") != "dmc_phase2_model_diagnostics_v1":
        infra_reasons.append("model_diagnostics schema_version mismatch")
    models = model_diag.get("models", {})
    if not isinstance(models, dict):
        infra_reasons.append("model_diagnostics.models must be an object")
        return
    for role, model_id in ROLE_MODELS.items():
        row = models.get(role)
        if not isinstance(row, dict):
            infra_reasons.append(f"model_diagnostics missing {role}")
            continue
        if row.get("model_id") != model_id:
            infra_reasons.append(f"model_diagnostics {role} model_id mismatch")


def validate_hashed_artifact(
    run_dir: Path,
    *,
    row: dict[str, Any],
    rel_field: str,
    sha_field: str,
    infra_reasons: list[str],
) -> Path | None:
    run_id = str(row.get("run_id"))
    rel = source_relative(str(row.get(rel_field, "")))
    if rel is None:
        infra_reasons.append(f"{run_id}: {rel_field} must be a relative path")
        return None
    path = run_dir / rel
    if not path.is_file():
        infra_reasons.append(f"{run_id}: missing artifact {rel_field}")
        return None
    if row.get(sha_field) != file_sha256(path):
        infra_reasons.append(f"{run_id}: {sha_field} mismatch")
    return path


def validate_row_logs(run_dir: Path, row: dict[str, Any], infra_reasons: list[str]) -> None:
    for prefix in ["baseline", "dmc"]:
        command_field = f"{prefix}_command_line"
        if not row.get(command_field):
            infra_reasons.append(f"{row.get('run_id')}: missing {command_field}")
        for stream in ["stdout", "stderr"]:
            validate_hashed_artifact(
                run_dir,
                row=row,
                rel_field=f"{prefix}_{stream}_path",
                sha_field=f"{prefix}_{stream}_sha256",
                infra_reasons=infra_reasons,
            )


def validate_launch_audit(run_dir: Path, row: dict[str, Any], infra_reasons: list[str]) -> None:
    path = validate_hashed_artifact(
        run_dir,
        row=row,
        rel_field="launch_audit_path",
        sha_field="launch_audit_sha256",
        infra_reasons=infra_reasons,
    )
    if path is None:
        return
    try:
        audit = load_json(path)
    except Exception as exc:
        infra_reasons.append(f"{row.get('run_id')}: invalid launch audit JSON: {exc!r}")
        return
    if audit.get("schema_version") != "dmc_phase2_launch_audit_v1":
        infra_reasons.append(f"{row.get('run_id')}: launch audit schema mismatch")
    if audit.get("profiler") not in {"nsight_systems", "torch_profiler", "cuda_launch_trace"}:
        infra_reasons.append(f"{row.get('run_id')}: unsupported launch audit profiler")
    baseline = audit.get("baseline", {})
    dmc = audit.get("dmc", {})
    if int(baseline.get("launch_count", -1)) != int(row.get("baseline_launch_count", -2)):
        infra_reasons.append(f"{row.get('run_id')}: baseline launch count does not match launch audit")
    if int(dmc.get("launch_count", -1)) != int(row.get("dmc_launch_count", -2)):
        infra_reasons.append(f"{row.get('run_id')}: DMC launch count does not match launch audit")
    if not baseline.get("evidence") or not dmc.get("evidence"):
        infra_reasons.append(f"{row.get('run_id')}: launch audit missing evidence")


def validate_latency_samples(run_dir: Path, row: dict[str, Any], infra_reasons: list[str]) -> None:
    path = validate_hashed_artifact(
        run_dir,
        row=row,
        rel_field="per_token_latency_path",
        sha_field="per_token_latency_sha256",
        infra_reasons=infra_reasons,
    )
    if path is None:
        return
    try:
        samples = load_json(path)
    except Exception as exc:
        infra_reasons.append(f"{row.get('run_id')}: invalid per-token latency JSON: {exc!r}")
        return
    if samples.get("schema_version") != "dmc_phase2_per_token_latency_v1":
        infra_reasons.append(f"{row.get('run_id')}: per-token latency schema mismatch")
    for prefix in ["baseline", "dmc"]:
        values = samples.get(f"{prefix}_ms", [])
        if not isinstance(values, list) or not values:
            infra_reasons.append(f"{row.get('run_id')}: missing {prefix} per-token latency samples")
            continue
        if any(float(value) <= 0.0 or not math.isfinite(float(value)) for value in values):
            infra_reasons.append(f"{row.get('run_id')}: invalid {prefix} per-token latency sample")
            continue
        recorded = row.get(f"{prefix}_per_token_latency_ms", {})
        for key, percent in [("p50", 50.0), ("p95", 95.0), ("p99", 99.0)]:
            actual = percentile_nearest_rank([float(v) for v in values], percent)
            if not is_close(float(recorded.get(key, float("nan"))), actual):
                infra_reasons.append(f"{row.get('run_id')}: {prefix} {key} latency does not match samples")


def validate_row_infra(
    run_dir: Path,
    row: dict[str, Any],
    prompt_indices: set[int],
    prompt_sha256: str,
    infra_reasons: list[str],
) -> None:
    run_id = str(row.get("run_id"))
    role = str(row.get("role"))
    if role not in ROLE_MODELS:
        infra_reasons.append(f"{run_id}: invalid role {role}")
    elif row.get("model") != ROLE_MODELS[role]:
        infra_reasons.append(f"{run_id}: model substitution for role {role}")
    try:
        prompt_index = int(row.get("prompt_index"))
        if prompt_index not in prompt_indices:
            infra_reasons.append(f"{run_id}: prompt_index not in fixed prompt set")
    except Exception:
        infra_reasons.append(f"{run_id}: prompt_index missing or non-integer")
    if row.get("prompt_sha256") != prompt_sha256:
        infra_reasons.append(f"{run_id}: prompt_sha256 mismatch")
    if not isinstance(row.get("seed"), int):
        infra_reasons.append(f"{run_id}: missing integer seed")
    if not isinstance(row.get("sampling_parameters"), dict) or not row.get("sampling_parameters"):
        infra_reasons.append(f"{run_id}: missing sampling parameters")
    if int(row.get("decode_length", 0)) <= 0:
        infra_reasons.append(f"{run_id}: decode_length must be positive")
    if not row.get("dtype"):
        infra_reasons.append(f"{run_id}: missing dtype")
    if not row.get("cuda_graph_state"):
        infra_reasons.append(f"{run_id}: missing CUDA graph state")
    backend = row.get("serving_backend", {})
    if not isinstance(backend, dict) or str(backend.get("name", "")).lower() != "vllm":
        infra_reasons.append(f"{run_id}: row is not marked as a vLLM serving row")
    if not isinstance(row.get("baseline_output_text"), str) or not isinstance(row.get("dmc_output_text"), str):
        infra_reasons.append(f"{run_id}: missing output text")
    baseline_tokens = row.get("baseline_token_ids")
    dmc_tokens = row.get("dmc_token_ids")
    if not isinstance(baseline_tokens, list) or not isinstance(dmc_tokens, list):
        infra_reasons.append(f"{run_id}: missing generated token IDs")
    elif not all(isinstance(token, int) for token in baseline_tokens + dmc_tokens):
        infra_reasons.append(f"{run_id}: generated token IDs must be integers")
    else:
        expected_match = baseline_tokens == dmc_tokens
        if row.get("generated_token_match") is not expected_match:
            infra_reasons.append(f"{run_id}: generated_token_match does not match token ID equality")
    for field in ["baseline_decode_ms", "dmc_decode_ms", "baseline_tokens_per_second", "dmc_tokens_per_second"]:
        try:
            value = float(row[field])
            if value <= 0.0 or not math.isfinite(value):
                infra_reasons.append(f"{run_id}: {field} must be positive and finite")
        except Exception:
            infra_reasons.append(f"{run_id}: missing numeric {field}")
    for prefix in ["baseline", "dmc"]:
        lat = row.get(f"{prefix}_per_token_latency_ms", {})
        if not isinstance(lat, dict):
            infra_reasons.append(f"{run_id}: missing {prefix} per-token latency summary")
            continue
        try:
            p50, p95, p99 = float(lat["p50"]), float(lat["p95"]), float(lat["p99"])
            if not (0.0 < p50 <= p95 <= p99):
                infra_reasons.append(f"{run_id}: invalid {prefix} per-token latency ordering")
        except Exception:
            infra_reasons.append(f"{run_id}: missing {prefix} p50/p95/p99 latency")
    try:
        baseline_launch = int(row["baseline_launch_count"])
        dmc_launch = int(row["dmc_launch_count"])
        if baseline_launch <= 0 or dmc_launch <= 0:
            infra_reasons.append(f"{run_id}: launch counts must be positive")
    except Exception:
        infra_reasons.append(f"{run_id}: missing launch counts")
    validate_launch_audit(run_dir, row, infra_reasons)
    validate_latency_samples(run_dir, row, infra_reasons)
    validate_row_logs(run_dir, row, infra_reasons)


def rows_by_role(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("role")), []).append(row)
    return grouped


def validate_role_summary(
    *, rows: list[dict[str, Any]], metrics: dict[str, Any], infra_reasons: list[str]
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
        reductions = [float(row["decode_latency_reduction_fraction"]) for row in role_rows]
        tps_gains = [float(row["tokens_per_second_gain_fraction"]) for row in role_rows]
        role_metrics[role] = {
            "rows": len(role_rows),
            "decode_latency_reduction_median": float(median(reductions)),
            "decode_latency_reduction_bootstrap_ci95": bootstrap_ci(
                reductions, samples=bootstrap_samples, seed=20260508 + len(role)
            ),
            "tokens_per_second_gain_median": float(median(tps_gains)),
            "launch_reduction_min": min(float(row["launch_reduction_fraction"]) for row in role_rows),
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
        for key in [
            "rows",
            "decode_latency_reduction_median",
            "tokens_per_second_gain_median",
            "launch_reduction_min",
        ]:
            if key == "rows":
                if int(recorded_values.get(key, -1)) != int(values[key]):
                    infra_reasons.append(f"{role}: role_summary {key} mismatch")
            elif not is_close(float(recorded_values.get(key, float("nan"))), float(values[key])):
                infra_reasons.append(f"{role}: role_summary {key} mismatch")
        recorded_ci = recorded_values.get("decode_latency_reduction_bootstrap_ci95", {})
        computed_ci = values["decode_latency_reduction_bootstrap_ci95"]
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
        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        environment = load_json(run_dir / "environment.json")
        model_diag = load_json(run_dir / "model_diagnostics.json")
    except Exception as exc:
        return {
            "decision": INFRA_DECISION,
            "run_dir": str(run_dir),
            "reasons": [f"bad JSON in required artifacts: {exc!r}"],
        }

    if metrics.get("schema_version") != "dmc_phase2_metrics_v1":
        infra_reasons.append("metrics schema_version is not dmc_phase2_metrics_v1")
    if manifest.get("schema_version") != "dmc_phase2_input_manifest_v1":
        infra_reasons.append("input manifest schema_version mismatch")
    if command.get("schema_version") != "dmc_phase2_command_v1":
        infra_reasons.append("command_metadata schema_version mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra_reasons.append("threshold metadata mismatch")
    if metrics.get("required_models") != ROLE_MODELS:
        infra_reasons.append("required model metadata mismatch")
    method = metrics.get("method", {})
    if method.get("name") != "dmc_vllm_serving_integration":
        infra_reasons.append("method.name is not dmc_vllm_serving_integration")
    if method.get("serving_backend") != "vllm":
        infra_reasons.append("method.serving_backend is not vllm")
    if method.get("serving_integration_implemented") is not True:
        infra_reasons.append("method does not declare a real serving integration")
    if method.get("serving_rows_are_real") is not True:
        infra_reasons.append("method does not declare real serving rows")
    if method.get("phase1_replay_latency_used_as_proxy") is not False:
        infra_reasons.append("Phase 1 replay latency is forbidden as a serving proxy")
    if method.get("cpu_only_timing") is not False:
        infra_reasons.append("CPU-only timing is forbidden")
    if method.get("model_substitution") is not False:
        infra_reasons.append("model substitution is forbidden")
    if method.get("boundary_fusion_claim") is not False:
        infra_reasons.append("boundary-fusion claim is forbidden for DMC Phase 2")

    validate_fixed_inputs(metrics=metrics, manifest=manifest, command=command, infra_reasons=infra_reasons)
    prompt_indices = validate_prompt_manifest(
        prompt_manifest=prompt_manifest, metrics=metrics, infra_reasons=infra_reasons
    )
    validate_environment(environment, infra_reasons)
    validate_model_diagnostics(model_diag, infra_reasons)

    rows = metrics.get("rows", [])
    if not isinstance(rows, list):
        infra_reasons.append("metrics.rows must be a list")
        rows = []
    run_ids = [str(row.get("run_id")) for row in rows if isinstance(row, dict)]
    if len(run_ids) != len(set(run_ids)):
        infra_reasons.append("metrics rows contain duplicate run_id values")
    expected_prompt_sha = str(metrics.get("prompt_sha256"))
    for row in rows:
        if not isinstance(row, dict):
            infra_reasons.append("metrics.rows contains a non-object row")
            continue
        try:
            validate_row_infra(run_dir, row, prompt_indices, expected_prompt_sha, infra_reasons)
            baseline = float(row["baseline_decode_ms"])
            dmc = float(row["dmc_decode_ms"])
            baseline_tps = float(row["baseline_tokens_per_second"])
            dmc_tps = float(row["dmc_tokens_per_second"])
            baseline_launch = int(row["baseline_launch_count"])
            dmc_launch = int(row["dmc_launch_count"])
            expected_latency = (baseline - dmc) / baseline
            expected_tps = (dmc_tps - baseline_tps) / baseline_tps
            expected_launch = (baseline_launch - dmc_launch) / baseline_launch
            if not is_close(float(row.get("decode_latency_reduction_fraction")), expected_latency):
                infra_reasons.append(f"{row.get('run_id')}: decode latency reduction formula mismatch")
            if not is_close(float(row.get("tokens_per_second_gain_fraction")), expected_tps):
                infra_reasons.append(f"{row.get('run_id')}: tokens/sec gain formula mismatch")
            if not is_close(float(row.get("launch_reduction_fraction")), expected_launch):
                infra_reasons.append(f"{row.get('run_id')}: launch reduction formula mismatch")
        except Exception as exc:
            infra_reasons.append(f"{row.get('run_id')}: malformed metric row: {exc!r}")

    role_metrics = validate_role_summary(rows=rows, metrics=metrics, infra_reasons=infra_reasons)

    if infra_reasons:
        return {"decision": INFRA_DECISION, "run_dir": str(run_dir), "reasons": infra_reasons}

    for row in rows:
        run_id = str(row["run_id"])
        if row["generated_token_match"] is not True:
            kill_reasons.append(f"{run_id}: generated_token_match is false")
        if row["baseline_output_text"] != row["dmc_output_text"]:
            kill_reasons.append(f"{run_id}: baseline and DMC output text differ")
        if float(row["launch_reduction_fraction"]) < THRESHOLDS["min_row_launch_reduction_fraction"]:
            kill_reasons.append(
                f"{run_id}: launch_reduction_fraction {float(row['launch_reduction_fraction']):.8f} "
                f"below {THRESHOLDS['min_row_launch_reduction_fraction']:.8f}"
            )

    for role, minimum in THRESHOLDS["min_role_median_decode_latency_reduction"].items():
        actual = float(role_metrics[role]["decode_latency_reduction_median"])
        if actual < float(minimum):
            kill_reasons.append(
                f"{role}: median decode latency reduction {actual:.8f} below {float(minimum):.8f}"
            )
        ci_low = float(role_metrics[role]["decode_latency_reduction_bootstrap_ci95"]["ci95_low"])
        if ci_low <= float(THRESHOLDS["bootstrap_ci95_lower_bound_min_exclusive"]):
            kill_reasons.append(
                f"{role}: bootstrap CI lower bound {ci_low:.8f} is not greater than 0"
            )
        tps_gain = float(role_metrics[role]["tokens_per_second_gain_median"])
        if tps_gain <= 0.0:
            kill_reasons.append(f"{role}: median tokens/sec gain {tps_gain:.8f} is not positive")

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
