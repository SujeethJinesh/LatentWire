#!/usr/bin/env python3
"""Check Cross-Layer Quantization Error Compounding result packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experimental/cross_layer_error/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_11.jsonl"

MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
PROMPT_SOURCE = "AIME-2025"
PROMPT_SELECTION = "deterministic_indices_0_11"
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_FILE = "aime2025-I.jsonl"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
DEPTHS = (1, 5, 10, 15)
TRACE_COUNT = 12
DECODE_POSITION = 1000
SCHEMA_VERSION = "cle_theoretical_v1"

PASS = "PASS_CLE_BOUND_TIGHT"
KILL = "KILL_CLE_BOUND_LOOSE"
FAIL_INFRA = "FAIL_INFRA_CLE"

THRESHOLDS: dict[str, Any] = {
    "tight_ratio_min": 1.0,
    "tight_ratio_max": 2.0,
    "loose_ratio_gt": 5.0,
    "bootstrap_samples": 1000,
    "bootstrap_ci": 0.95,
    "functional_constant_predicted_relative_range_max": 0.05,
    "functional_measured_growth_factor_min": 1.5,
    "functional_opposite_endpoint_relative_change_min": 0.25,
}

REQUIRED_FILES = [
    "derivation.md",
    "environment.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "quantization_config.json",
    "predicted_bounds.json",
    "trace_tokens.jsonl",
    "logits_manifest.json",
    "raw_drift_rows.jsonl",
    "drift_metrics.json",
    "bootstrap_ci.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_SELF_EXCLUDES = {"artifact_hashes.json", "checker_result.json", "artifact_check.json"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_close(left: float, right: float, *, tol: float = 1e-6) -> bool:
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    payload = "".join(str(row["prompt"]) for row in ordered).encode("utf-8")
    return bytes_sha256(payload)


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no CLE result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                yield row


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            infra.append("artifact_hashes contains a non-object row")
            continue
        rel = str(row.get("path", ""))
        seen.add(rel)
        if rel in HASHED_SELF_EXCLUDES:
            infra.append(f"artifact_hashes must not include checker/self artifact {rel}")
            continue
        path = run_dir / rel
        if not path.is_file():
            infra.append(f"hashed artifact missing on disk: {rel}")
            continue
        if row.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if row.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")

    disk_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and str(path.relative_to(run_dir)) not in HASHED_SELF_EXCLUDES
    }
    for rel in sorted(disk_files.difference(seen)):
        infra.append(f"artifact_hashes missing on-disk artifact {rel}")


def validate_prompt_manifest(prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra: list[str]) -> dict[int, str]:
    if prompt_manifest.get("source") != PROMPT_SOURCE:
        infra.append("prompt_manifest.source must be AIME-2025")
    if prompt_manifest.get("selection") != PROMPT_SELECTION:
        infra.append("prompt_manifest.selection must be deterministic_indices_0_11")
    if int(prompt_manifest.get("prompt_count", -1)) != TRACE_COUNT:
        infra.append("prompt_manifest.prompt_count must be 12")
    if DEFAULT_PROMPT_FILE.is_file():
        if prompt_manifest.get("prompt_file_sha256") != file_sha256(DEFAULT_PROMPT_FILE):
            infra.append("prompt_manifest.prompt_file_sha256 must match canonical AIME-2025 indices 0-11")
    prompts = prompt_manifest.get("prompts", [])
    if not isinstance(prompts, list):
        infra.append("prompt_manifest.prompts must be a list")
        prompts = []
    if len(prompts) != TRACE_COUNT:
        infra.append("prompt_manifest.prompts must contain exactly 12 rows")
    prompt_ids: dict[int, str] = {}
    observed_indices: list[int] = []
    for row in prompts:
        if not isinstance(row, dict):
            infra.append("prompt_manifest contains a non-object prompt row")
            continue
        try:
            index = int(row["index"])
        except Exception:
            infra.append("prompt_manifest row missing integer index")
            continue
        observed_indices.append(index)
        prompt_ids[index] = str(row.get("prompt_id", ""))
        if row.get("source_dataset") != EXPECTED_PROMPT_SOURCE_DATASET:
            infra.append(f"prompt {index}: source_dataset mismatch")
        if row.get("source_file") != EXPECTED_PROMPT_SOURCE_FILE:
            infra.append(f"prompt {index}: source_file mismatch")
        if row.get("source_commit") != EXPECTED_PROMPT_SOURCE_COMMIT:
            infra.append(f"prompt {index}: source_commit mismatch")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            infra.append(f"prompt {index}: missing prompt text")
    if observed_indices != list(range(TRACE_COUNT)):
        infra.append("prompt indices must be exactly 0-11 in deterministic order")
    if prompts:
        expected_sha = prompt_payload_sha256([row for row in prompts if isinstance(row, dict)])
        if prompt_manifest.get("prompt_sha256") != expected_sha:
            infra.append("prompt_manifest.prompt_sha256 mismatch")
        if metrics.get("prompt_sha256") != expected_sha:
            infra.append("drift_metrics.prompt_sha256 must match prompt_manifest")
    if metrics.get("prompt_source") != PROMPT_SOURCE:
        infra.append("drift_metrics.prompt_source must be AIME-2025")
    if metrics.get("prompt_selection") != PROMPT_SELECTION:
        infra.append("drift_metrics.prompt_selection mismatch")
    return prompt_ids


def validate_run_event_order(events: list[dict[str, Any]], infra: list[str]) -> None:
    names = [str(row.get("event", "")) for row in events]
    for required in [
        "run_started",
        "bound_predictions_written",
        "derivation_locked",
        "measurement_started",
        "trace_tokens_written",
        "run_completed",
    ]:
        if required not in names:
            infra.append(f"run_events missing {required}")
    if "derivation_locked" in names and "measurement_started" in names:
        if names.index("derivation_locked") > names.index("measurement_started"):
            infra.append("derivation_locked must occur before measurement_started")
    if "bound_predictions_written" in names and "measurement_started" in names:
        if names.index("bound_predictions_written") > names.index("measurement_started"):
            infra.append("bound_predictions_written must occur before measurement_started")
    if "trace_tokens_written" in names and "measurement_started" in names:
        if names.index("trace_tokens_written") < names.index("measurement_started"):
            infra.append("trace_tokens_written must occur after measurement_started")


def validate_trace_tokens(
    trace_rows: list[dict[str, Any]],
    prompt_ids: dict[int, str],
    infra: list[str],
) -> dict[int, list[int]]:
    if len(trace_rows) != TRACE_COUNT:
        infra.append(f"trace_tokens.jsonl must contain {TRACE_COUNT} rows")
    by_prompt: dict[int, list[int]] = {}
    seen: set[int] = set()
    for row in trace_rows:
        try:
            prompt_index = int(row["prompt_index"])
            token_count = int(row["token_count"])
            token_ids = [int(token) for token in row["token_ids"]]
        except Exception as exc:
            infra.append(f"trace token row {row.get('_line_number')} missing typed fields: {exc!r}")
            continue
        if prompt_index in seen:
            infra.append(f"duplicate trace token row for prompt {prompt_index}")
        seen.add(prompt_index)
        if prompt_index not in range(TRACE_COUNT):
            infra.append(f"trace token prompt_index out of range: {prompt_index}")
        if row.get("prompt_id") != prompt_ids.get(prompt_index):
            infra.append(f"trace token prompt_id mismatch for prompt {prompt_index}")
        if int(row.get("decode_position", -1)) != DECODE_POSITION:
            infra.append(f"trace token row {prompt_index} decode_position must be 1000")
        if token_count != DECODE_POSITION or len(token_ids) != DECODE_POSITION:
            infra.append(f"trace token row {prompt_index} must contain exactly 1000 token ids")
        if any(token < 0 for token in token_ids):
            infra.append(f"trace token row {prompt_index} contains negative token ids")
        expected_hash = bytes_sha256(json.dumps(token_ids, separators=(",", ":")).encode("utf-8"))
        if row.get("token_ids_sha256") != expected_hash:
            infra.append(f"trace token row {prompt_index} token_ids_sha256 mismatch")
        if row.get("source") != "bf16_greedy_prefix":
            infra.append(f"trace token row {prompt_index} source must be bf16_greedy_prefix")
        by_prompt[prompt_index] = token_ids
    expected = set(range(TRACE_COUNT))
    missing = sorted(expected.difference(seen))
    if missing:
        infra.append(f"missing trace token rows for prompts {missing}")
    return by_prompt


def validate_predicted_bounds(
    run_dir: Path,
    predicted_bounds: dict[str, Any],
    infra: list[str],
) -> dict[int, dict[str, Any]]:
    if predicted_bounds.get("schema_version") != f"{SCHEMA_VERSION}_predicted_bounds":
        infra.append("predicted_bounds schema_version mismatch")
    if predicted_bounds.get("model_id") != MODEL_ID:
        infra.append("predicted_bounds.model_id mismatch")
    if list(predicted_bounds.get("depths", [])) != list(DEPTHS):
        infra.append("predicted_bounds.depths must be [1,5,10,15]")
    if predicted_bounds.get("created_before_measurement") is not True:
        infra.append("predicted_bounds.created_before_measurement must be true")
    if predicted_bounds.get("derivation_sha256") != file_sha256(run_dir / "derivation.md"):
        infra.append("predicted_bounds.derivation_sha256 must match derivation.md")
    if not isinstance(predicted_bounds.get("bound_formula"), str) or not predicted_bounds["bound_formula"].strip():
        infra.append("predicted_bounds.bound_formula must describe F")
    rows = predicted_bounds.get("bounds_by_depth", [])
    if not isinstance(rows, list):
        infra.append("predicted_bounds.bounds_by_depth must be a list")
        rows = []
    by_depth: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            infra.append("predicted_bounds contains a non-object depth row")
            continue
        try:
            depth = int(row["depth"])
            value = float(row["predicted_bound_l2"])
        except Exception:
            infra.append("predicted_bounds row missing numeric depth/predicted_bound_l2")
            continue
        if depth not in DEPTHS:
            infra.append(f"predicted_bounds contains non-preregistered depth {depth}")
        if depth in by_depth:
            infra.append(f"predicted_bounds duplicate depth {depth}")
        by_depth[depth] = row
        if not math.isfinite(value) or value < 0.0:
            infra.append(f"predicted bound for depth {depth} must be finite and nonnegative")
        if len(row.get("quantized_layers", [])) != depth:
            infra.append(f"predicted bound depth {depth} must record {depth} quantized layers")
        for field in ["sigma_block_sum", "sigma_outlier_sum", "layer_error_l2_sum"]:
            try:
                numeric = float(row.get(field))
            except Exception:
                infra.append(f"predicted bound depth {depth} missing numeric {field}")
                continue
            if not math.isfinite(numeric) or numeric < 0.0:
                infra.append(f"predicted bound depth {depth} invalid {field}")
    if sorted(by_depth) != list(DEPTHS):
        infra.append("predicted_bounds must contain exactly depths [1,5,10,15]")
    return by_depth


def load_logit_entries(run_dir: Path, logits_manifest: dict[str, Any], infra: list[str]) -> dict[tuple[str, int, int | None], dict[str, Any]]:
    if logits_manifest.get("schema_version") != f"{SCHEMA_VERSION}_logits_manifest":
        infra.append("logits_manifest schema_version mismatch")
    if logits_manifest.get("dtype") != "float32_le":
        infra.append("logits_manifest.dtype must be float32_le")
    entries = logits_manifest.get("entries", [])
    if not isinstance(entries, list):
        infra.append("logits_manifest.entries must be a list")
        entries = []
    by_key: dict[tuple[str, int, int | None], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            infra.append("logits_manifest contains non-object entry")
            continue
        role = str(entry.get("role", ""))
        try:
            prompt_index = int(entry["prompt_index"])
            depth = int(entry["depth"]) if role == "fp4" else None
            shape = [int(dim) for dim in entry.get("shape", [])]
        except Exception:
            infra.append("logits_manifest entry missing typed prompt/depth/shape")
            continue
        if role not in {"bf16", "fp4"}:
            infra.append(f"logits_manifest invalid role {role}")
            continue
        if prompt_index not in range(TRACE_COUNT):
            infra.append(f"logits_manifest prompt_index out of range: {prompt_index}")
        if role == "fp4" and depth not in DEPTHS:
            infra.append(f"logits_manifest fp4 depth outside preregistered grid: {depth}")
        if role == "bf16" and "depth" in entry:
            infra.append("bf16 logits_manifest entries must not carry depth")
        if len(shape) != 1 or shape[0] <= 0:
            infra.append(f"logits_manifest {role} prompt {prompt_index} shape must be one-dimensional")
        rel = Path(str(entry.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts:
            infra.append(f"logits_manifest path must be packet-relative: {rel}")
            continue
        path = run_dir / rel
        if not path.is_file():
            infra.append(f"logits artifact missing: {rel}")
            continue
        if entry.get("bytes") != path.stat().st_size:
            infra.append(f"logits artifact byte mismatch: {rel}")
        if entry.get("sha256") != file_sha256(path):
            infra.append(f"logits artifact sha256 mismatch: {rel}")
        key = (role, prompt_index, depth)
        if key in by_key:
            infra.append(f"duplicate logits_manifest entry {key}")
        by_key[key] = entry
    expected_count = TRACE_COUNT + TRACE_COUNT * len(DEPTHS)
    if len(entries) != expected_count:
        infra.append(f"logits_manifest must contain {expected_count} entries")
    return by_key


def read_logits(path: Path, shape: list[int]) -> Any:
    import numpy as np

    values = np.fromfile(path, dtype="<f4")
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} float32 values, expected {expected}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains nonfinite logits")
    return values.reshape(shape)


def validate_and_recompute_rows(
    run_dir: Path,
    raw_rows: list[dict[str, Any]],
    logit_entries: dict[tuple[str, int, int | None], dict[str, Any]],
    prompt_ids: dict[int, str],
    infra: list[str],
) -> list[dict[str, Any]]:
    expected_rows = TRACE_COUNT * len(DEPTHS)
    if len(raw_rows) != expected_rows:
        infra.append(f"raw_drift_rows must contain {expected_rows} rows")
    seen: set[tuple[int, int]] = set()
    recomputed: list[dict[str, Any]] = []
    for row in raw_rows:
        try:
            prompt_index = int(row["prompt_index"])
            depth = int(row["depth"])
            observed_l2 = float(row["l2_drift"])
        except Exception as exc:
            infra.append(f"raw row {row.get('_line_number')}: missing typed fields: {exc!r}")
            continue
        key = (prompt_index, depth)
        if key in seen:
            infra.append(f"duplicate raw drift row for prompt/depth {key}")
        seen.add(key)
        if prompt_index not in range(TRACE_COUNT):
            infra.append(f"raw drift prompt_index out of range: {prompt_index}")
        if depth not in DEPTHS:
            infra.append(f"raw drift depth outside preregistered grid: {depth}")
        if row.get("prompt_id") != prompt_ids.get(prompt_index):
            infra.append(f"raw drift prompt_id mismatch for prompt {prompt_index}")
        if int(row.get("decode_position", -1)) != DECODE_POSITION:
            infra.append(f"raw drift row {key} decode_position must be 1000")
        if row.get("quantization_format") != "nvfp4_e2m1_weight_sim":
            infra.append(f"raw drift row {key} quantization_format mismatch")
        if not math.isfinite(observed_l2) or observed_l2 < 0.0:
            infra.append(f"raw drift row {key} l2_drift must be finite and nonnegative")
            continue
        bf16_entry = logit_entries.get(("bf16", prompt_index, None))
        fp4_entry = logit_entries.get(("fp4", prompt_index, depth))
        if bf16_entry is None or fp4_entry is None:
            infra.append(f"missing logits_manifest entries for raw row {key}")
            continue
        if row.get("bf16_logits_path") != bf16_entry.get("path"):
            infra.append(f"raw drift row {key} bf16_logits_path mismatch")
        if row.get("fp4_logits_path") != fp4_entry.get("path"):
            infra.append(f"raw drift row {key} fp4_logits_path mismatch")
        try:
            bf16 = read_logits(run_dir / str(bf16_entry["path"]), list(bf16_entry["shape"]))
            fp4 = read_logits(run_dir / str(fp4_entry["path"]), list(fp4_entry["shape"]))
            if list(bf16_entry["shape"]) != list(fp4_entry["shape"]):
                infra.append(f"logits shape mismatch for raw row {key}")
                continue
            drift = float(math.sqrt(float(((bf16 - fp4) ** 2).sum())))
        except Exception as exc:
            infra.append(f"cannot recompute logits drift for row {key}: {exc!r}")
            continue
        if not is_close(observed_l2, drift):
            infra.append(f"raw drift row {key} l2_drift does not match logits artifacts")
        bf16_argmax = int(bf16.argmax())
        fp4_argmax = int(fp4.argmax())
        if "bf16_argmax_token_id" in row and int(row["bf16_argmax_token_id"]) != bf16_argmax:
            infra.append(f"raw drift row {key} bf16_argmax_token_id mismatch")
        if "fp4_argmax_token_id" in row and int(row["fp4_argmax_token_id"]) != fp4_argmax:
            infra.append(f"raw drift row {key} fp4_argmax_token_id mismatch")
        recomputed.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": prompt_ids.get(prompt_index),
                "depth": depth,
                "l2_drift": drift,
                "bf16_argmax_token_id": bf16_argmax,
                "fp4_argmax_token_id": fp4_argmax,
            }
        )
    expected_keys = {(prompt_index, depth) for prompt_index in range(TRACE_COUNT) for depth in DEPTHS}
    missing = sorted(expected_keys.difference(seen))
    if missing:
        infra.append(f"missing raw drift rows: {missing[:6]}")
    return sorted(recomputed, key=lambda row: (int(row["depth"]), int(row["prompt_index"])))


def _bootstrap_ci(values: list[float], *, samples: int, rng: random.Random) -> dict[str, float]:
    boot: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(mean(sample)))
    boot.sort()
    return {
        "ci95_low": float(boot[int(0.025 * (len(boot) - 1))]),
        "ci95_high": float(boot[int(0.975 * (len(boot) - 1))]),
    }


def functional_form_readout(depth_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = [float(row["predicted_bound_l2"]) for row in depth_metrics]
    measured = [float(row["mean_l2_drift"]) for row in depth_metrics]
    eps = 1e-12
    predicted_range = (max(predicted) - min(predicted)) / max(abs(max(predicted)), eps)
    measured_growth_factor = measured[-1] / max(measured[0], eps)
    predicted_growth_factor = predicted[-1] / max(predicted[0], eps)
    constant_predicted_vs_growing_measured = (
        predicted_range <= THRESHOLDS["functional_constant_predicted_relative_range_max"]
        and measured_growth_factor >= THRESHOLDS["functional_measured_growth_factor_min"]
    )
    pred_delta = predicted[-1] - predicted[0]
    measured_delta = measured[-1] - measured[0]
    pred_rel = abs(pred_delta) / max(abs(predicted[0]), eps)
    measured_rel = abs(measured_delta) / max(abs(measured[0]), eps)
    opposite_endpoint_direction = (
        pred_delta * measured_delta < 0.0
        and pred_rel >= THRESHOLDS["functional_opposite_endpoint_relative_change_min"]
        and measured_rel >= THRESHOLDS["functional_opposite_endpoint_relative_change_min"]
    )
    return {
        "predicted_growth_factor_depth15_over_depth1": float(predicted_growth_factor),
        "measured_growth_factor_depth15_over_depth1": float(measured_growth_factor),
        "predicted_relative_range": float(predicted_range),
        "constant_predicted_vs_growing_measured": bool(constant_predicted_vs_growing_measured),
        "opposite_endpoint_direction": bool(opposite_endpoint_direction),
        "functional_form_mismatch": bool(
            constant_predicted_vs_growing_measured or opposite_endpoint_direction
        ),
    }


def compute_metrics(
    rows: list[dict[str, Any]],
    predicted_by_depth: dict[int, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_depth: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_depth[int(row["depth"])].append(float(row["l2_drift"]))
    rng = random.Random(seed)
    depth_metrics: list[dict[str, Any]] = []
    for depth in DEPTHS:
        values = by_depth[depth]
        if len(values) != TRACE_COUNT:
            raise ValueError(f"depth {depth} has {len(values)} rows, expected {TRACE_COUNT}")
        mean_drift = float(mean(values))
        predicted = float(predicted_by_depth[depth]["predicted_bound_l2"])
        ratio = math.inf if mean_drift <= 0.0 and predicted > 0.0 else predicted / mean_drift if mean_drift > 0.0 else math.nan
        depth_metrics.append(
            {
                "depth": depth,
                "mean_l2_drift": mean_drift,
                "bootstrap_ci95": _bootstrap_ci(values, samples=bootstrap_samples, rng=rng),
                "predicted_bound_l2": predicted,
                "predicted_to_measured_ratio": float(ratio),
                "prompt_count": len(values),
            }
        )
    ratios = [float(row["predicted_to_measured_ratio"]) for row in depth_metrics]
    functional = functional_form_readout(depth_metrics)
    return {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "metric_name": "bf16_vs_nvfp4_l2_logit_drift_bound",
        "model_id": MODEL_ID,
        "prompt_source": PROMPT_SOURCE,
        "prompt_selection": PROMPT_SELECTION,
        "depths": list(DEPTHS),
        "decode_position": DECODE_POSITION,
        "trace_count": TRACE_COUNT,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "depth_metrics": depth_metrics,
        "functional_form": functional,
        "thresholds": THRESHOLDS,
        "decision_inputs": {
            "all_ratios_in_tight_band": all(
                THRESHOLDS["tight_ratio_min"] <= ratio <= THRESHOLDS["tight_ratio_max"]
                for ratio in ratios
            ),
            "any_ratio_gt_loose_threshold": any(ratio > THRESHOLDS["loose_ratio_gt"] for ratio in ratios),
            "any_ratio_below_upper_bound_floor": any(ratio < THRESHOLDS["tight_ratio_min"] for ratio in ratios),
            "functional_form_mismatch": bool(functional["functional_form_mismatch"]),
        },
    }


def compare_nested(observed: Any, computed: Any, infra: list[str], path: str) -> None:
    if path.endswith(".created_at_utc"):
        return
    if isinstance(computed, dict):
        if not isinstance(observed, dict):
            infra.append(f"{path} type mismatch")
            return
        for key, value in computed.items():
            if key not in observed:
                infra.append(f"{path}.{key} missing")
                continue
            compare_nested(observed[key], value, infra, f"{path}.{key}")
        return
    if isinstance(computed, list):
        if not isinstance(observed, list) or len(observed) != len(computed):
            infra.append(f"{path} list length mismatch")
            return
        for index, value in enumerate(computed):
            compare_nested(observed[index], value, infra, f"{path}[{index}]")
        return
    if isinstance(computed, float):
        try:
            actual = float(observed)
        except Exception:
            infra.append(f"{path} numeric mismatch")
            return
        if math.isinf(computed):
            if not math.isinf(actual) or math.copysign(1.0, actual) != math.copysign(1.0, computed):
                infra.append(f"{path} mismatch: observed={observed!r} computed={computed!r}")
        elif math.isnan(computed):
            if not math.isnan(actual):
                infra.append(f"{path} mismatch: observed={observed!r} computed=nan")
        elif not is_close(actual, computed):
            infra.append(f"{path} mismatch: observed={observed!r} computed={computed!r}")
        return
    if observed != computed:
        infra.append(f"{path} mismatch: observed={observed!r} computed={computed!r}")


def decision_from_metrics(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    inputs = metrics["decision_inputs"]
    depth_metrics = metrics["depth_metrics"]
    ratios = {
        int(row["depth"]): float(row["predicted_to_measured_ratio"]) for row in depth_metrics
    }
    if inputs["all_ratios_in_tight_band"]:
        return PASS, [
            "all predicted/measured ratios are within the preregistered tight band [1.0, 2.0]"
        ]
    if inputs["any_ratio_gt_loose_threshold"] or inputs["functional_form_mismatch"]:
        triggers: list[str] = []
        loose_depths = [
            depth for depth, ratio in ratios.items() if ratio > THRESHOLDS["loose_ratio_gt"]
        ]
        if loose_depths:
            triggers.append(f"ratio > 5.0 at depths {loose_depths}")
        if inputs["functional_form_mismatch"]:
            triggers.append("functional form mismatch")
        return KILL, triggers
    return KILL, [
        "complete packet is not tight at every depth; FAIL_INFRA_CLE is reserved for malformed packets"
    ]


def infra_result(run_dir: Path, reasons: list[str]) -> dict[str, Any]:
    result = {
        "decision": FAIL_INFRA,
        "run_dir": str(run_dir),
        "artifact_complete": False,
        "reasons": reasons,
    }
    if run_dir.is_dir():
        write_json(run_dir / "checker_result.json", result)
        write_json(
            run_dir / "artifact_check.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_artifact_check",
                "decision": FAIL_INFRA,
                "run_dir": str(run_dir),
                "required_files": REQUIRED_FILES,
                "artifact_complete": False,
                "reasons": reasons,
            },
        )
    return result


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required artifact: {rel}")
    try:
        environment = load_json(run_dir / "environment.json")
        model = load_json(run_dir / "model_provenance.json")
        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        command = load_json(run_dir / "command_metadata.json")
        random_seed = load_json(run_dir / "random_seed.json")
        quant_config = load_json(run_dir / "quantization_config.json")
        predicted_bounds = load_json(run_dir / "predicted_bounds.json")
        trace_rows = list(iter_jsonl(run_dir / "trace_tokens.jsonl"))
        logits_manifest = load_json(run_dir / "logits_manifest.json")
        drift_metrics = load_json(run_dir / "drift_metrics.json")
        bootstrap_ci = load_json(run_dir / "bootstrap_ci.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
        events = list(iter_jsonl(run_dir / "run_events.jsonl"))
        raw_rows = list(iter_jsonl(run_dir / "raw_drift_rows.jsonl"))
    except Exception as exc:
        return infra_result(run_dir, [*infra, f"bad or unreadable packet artifacts: {exc!r}"])

    if environment.get("schema_version") != f"{SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if model.get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if model.get("local_files_only") is not True:
        infra.append("model_provenance.local_files_only must be true")
    if not model.get("hf_snapshot_commit") or not model.get("snapshot_path"):
        infra.append("model provenance must record local HF snapshot commit and path")
    if command.get("branch") != "cross_layer_error":
        infra.append("command_metadata.branch must be cross_layer_error")
    if command.get("model_id") != MODEL_ID:
        infra.append("command_metadata.model_id mismatch")
    if list(command.get("depths", [])) != list(DEPTHS):
        infra.append("command_metadata.depths must equal [1,5,10,15]")
    if int(command.get("decode_position", -1)) != DECODE_POSITION:
        infra.append("command_metadata.decode_position must be 1000")
    if int(random_seed.get("seed", -1)) != int(drift_metrics.get("bootstrap_seed", -2)):
        infra.append("random_seed.seed must match drift_metrics.bootstrap_seed")
    if quant_config.get("quantization_format") != "nvfp4_e2m1_weight_sim":
        infra.append("quantization_config.quantization_format mismatch")
    if int(quant_config.get("block_size", -1)) != 16:
        infra.append("quantization_config.block_size must be 16 for NVFP4-compatible simulation")
    if quant_config.get("native_kernel_claim") is not False:
        infra.append("quantization_config.native_kernel_claim must be false for simulator packet")
    if list(quant_config.get("depths", [])) != list(DEPTHS):
        infra.append("quantization_config.depths must equal [1,5,10,15]")
    if int(quant_config.get("decode_position", -1)) != DECODE_POSITION:
        infra.append("quantization_config.decode_position must be 1000")
    if drift_metrics.get("thresholds") != THRESHOLDS:
        infra.append("drift_metrics.thresholds mismatch checker thresholds")
    if int(drift_metrics.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("drift_metrics.bootstrap_samples must be 1000")
    if int(bootstrap_ci.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("bootstrap_ci.bootstrap_samples must be 1000")
    if list(drift_metrics.get("depths", [])) != list(DEPTHS):
        infra.append("drift_metrics.depths must equal [1,5,10,15]")
    if int(drift_metrics.get("decode_position", -1)) != DECODE_POSITION:
        infra.append("drift_metrics.decode_position must be 1000")

    validate_run_event_order(events, infra)
    validate_artifact_hashes(run_dir, artifact_hashes, infra)
    prompt_ids = validate_prompt_manifest(prompt_manifest, drift_metrics, infra)
    validate_trace_tokens(trace_rows, prompt_ids, infra)
    predicted_by_depth = validate_predicted_bounds(run_dir, predicted_bounds, infra)
    logit_entries = load_logit_entries(run_dir, logits_manifest, infra)
    recomputed_rows: list[dict[str, Any]] = []
    if predicted_by_depth and logit_entries:
        recomputed_rows = validate_and_recompute_rows(
            run_dir,
            raw_rows,
            logit_entries,
            prompt_ids,
            infra,
        )

    computed_metrics: dict[str, Any] | None = None
    if not infra:
        try:
            computed_metrics = compute_metrics(
                recomputed_rows,
                predicted_by_depth,
                bootstrap_samples=THRESHOLDS["bootstrap_samples"],
                seed=int(drift_metrics["bootstrap_seed"]),
            )
            computed_metrics["prompt_sha256"] = drift_metrics.get("prompt_sha256")
            compare_nested(drift_metrics, computed_metrics, infra, "drift_metrics")
            expected_bootstrap = {
                "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
                "metric_name": computed_metrics["metric_name"],
                "bootstrap_samples": computed_metrics["bootstrap_samples"],
                "bootstrap_seed": computed_metrics["bootstrap_seed"],
                "depth_metrics": [
                    {
                        "depth": row["depth"],
                        "mean_l2_drift": row["mean_l2_drift"],
                        "bootstrap_ci95": row["bootstrap_ci95"],
                    }
                    for row in computed_metrics["depth_metrics"]
                ],
            }
            compare_nested(bootstrap_ci, expected_bootstrap, infra, "bootstrap_ci")
        except Exception as exc:
            infra.append(f"cannot recompute drift metrics: {exc!r}")

    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "artifact_complete": False,
            "reasons": infra,
        }
    else:
        assert computed_metrics is not None
        decision, reasons = decision_from_metrics(computed_metrics)
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "artifact_complete": True,
            "reasons": reasons,
            "depth_metrics": computed_metrics["depth_metrics"],
            "functional_form": computed_metrics["functional_form"],
            "thresholds": THRESHOLDS,
        }
    write_json(run_dir / "checker_result.json", result)
    write_json(
        run_dir / "artifact_check.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_artifact_check",
            "decision": result["decision"],
            "run_dir": str(run_dir),
            "required_files": REQUIRED_FILES,
            "artifact_complete": result.get("artifact_complete", False),
            "reasons": result["reasons"],
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
