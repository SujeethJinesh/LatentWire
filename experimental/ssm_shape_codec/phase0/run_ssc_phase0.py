#!/usr/bin/env python3
"""Run SSM Shape-Conditioned Codec Phase 0."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.shared import run_phase0_branch as shared

SCHEMA_VERSION = "ssc_phase0_v1"
MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
RESULTS_DIR = ROOT / "experimental/ssm_shape_codec/phase0/results"
POSITIONS = (100, 10000)
CALIBRATION_INDICES = tuple(range(12, 18))
TEST_INDICES = tuple(range(18, 24))
ALL_INDICES = CALIBRATION_INDICES + TEST_INDICES
SEED = 20260508
THRESHOLDS = {
    "mean_relative_nmse_reduction_min": 0.10,
    "bootstrap_ci95_low_gt": 0.05,
    "bootstrap_samples": 1000,
    "bootstrap_seed": SEED,
    "position_non_worse_required": True,
    "codebook_size": 16,
    "normalization_epsilon": 1e-12,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_prompts() -> tuple[dict[str, Any], list[str]]:
    prompts: list[dict[str, Any]] = []
    reasons: list[str] = []
    for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if int(item["index"]) in ALL_INDICES:
            prompts.append(item)
    observed = tuple(int(item["index"]) for item in prompts)
    if observed != ALL_INDICES:
        reasons.append(f"held-out prompt indices must be {ALL_INDICES}; got {observed}")
    for item in prompts:
        if item.get("source_dataset") != shared.EXPECTED_PROMPT_SOURCE_DATASET:
            reasons.append(f"prompt {item.get('index')} source_dataset mismatch")
        if item.get("source_commit") != shared.EXPECTED_PROMPT_SOURCE_COMMIT:
            reasons.append(f"prompt {item.get('index')} source_commit mismatch")
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
        "source": "AIME-2025",
        "selection": "heldout_indices_12_23",
        "prompt_file": str(PROMPT_FILE.relative_to(ROOT)),
        "prompt_file_sha256": shared.file_sha256(PROMPT_FILE),
        "prompt_count": len(prompts),
        "prompt_sha256": shared.prompt_payload_sha256(prompts),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
        "calibration_indices": list(CALIBRATION_INDICES),
        "test_indices": list(TEST_INDICES),
        "prompts": prompts,
    }
    return manifest, reasons


def npz_key(prompt_index: int, position: int) -> str:
    return f"p{prompt_index:03d}_pos{position:05d}"


def quantile_codebook(values: np.ndarray, *, codebook_size: int) -> np.ndarray:
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("cannot fit codebook from empty/nonfinite values")
    quantiles = np.linspace(0.0, 1.0, codebook_size, dtype=np.float64)
    return np.quantile(flat, quantiles).astype(np.float32)


def reconstruct_nearest(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1).astype(np.float32)
    nearest = np.argmin(np.abs(flat[:, None] - codebook[None, :]), axis=1)
    return codebook[nearest].reshape(values.shape)


def normalized_mse(values: np.ndarray, reconstruction: np.ndarray) -> float:
    eps = float(THRESHOLDS["normalization_epsilon"])
    denom = float(np.mean(np.square(values.astype(np.float32)))) + eps
    return float(np.mean(np.square(values.astype(np.float32) - reconstruction.astype(np.float32))) / denom)


def _mean(rows: list[float]) -> float:
    return float(mean(rows)) if rows else 0.0


def compute_codec_metrics(run_dir: Path, state_manifest: dict[str, Any]) -> dict[str, Any]:
    codebook_size = int(THRESHOLDS["codebook_size"])
    per_row: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []
    mamba_layers = [row for row in state_manifest["layers"] if bool(row.get("is_mamba"))]

    for layer in mamba_layers:
        layer_index = int(layer["layer_index"])
        artifact = run_dir / str(layer["artifact"])
        with np.load(artifact) as data:
            calibration_by_position = {
                position: np.concatenate(
                    [data[npz_key(prompt_index, position)].reshape(-1) for prompt_index in CALIBRATION_INDICES]
                )
                for position in POSITIONS
            }
            pooled_calibration = np.concatenate([calibration_by_position[position] for position in POSITIONS])
            baseline_codebook = quantile_codebook(pooled_calibration, codebook_size=codebook_size)
            method_codebooks = {
                position: quantile_codebook(calibration_by_position[position], codebook_size=codebook_size)
                for position in POSITIONS
            }
            layer_rows: list[dict[str, Any]] = []
            for prompt_index in TEST_INDICES:
                for position in POSITIONS:
                    values = data[npz_key(prompt_index, position)].astype(np.float32)
                    baseline_recon = reconstruct_nearest(values, baseline_codebook)
                    method_recon = reconstruct_nearest(values, method_codebooks[position])
                    baseline_nmse = normalized_mse(values, baseline_recon)
                    method_nmse = normalized_mse(values, method_recon)
                    rel = float((baseline_nmse - method_nmse) / max(baseline_nmse, 1e-12))
                    row = {
                        "prompt_index": prompt_index,
                        "layer_index": layer_index,
                        "position": position,
                        "baseline_nmse": baseline_nmse,
                        "method_nmse": method_nmse,
                        "relative_nmse_reduction": rel,
                    }
                    per_row.append(row)
                    layer_rows.append(row)
            layer_summaries.append(
                {
                    "layer_index": layer_index,
                    "mean_relative_nmse_reduction": _mean(
                        [float(row["relative_nmse_reduction"]) for row in layer_rows]
                    ),
                    "mean_baseline_nmse": _mean([float(row["baseline_nmse"]) for row in layer_rows]),
                    "mean_method_nmse": _mean([float(row["method_nmse"]) for row in layer_rows]),
                }
            )

    by_prompt: dict[int, list[float]] = defaultdict(list)
    by_position: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in per_row:
        by_prompt[int(row["prompt_index"])].append(float(row["relative_nmse_reduction"]))
        by_position[int(row["position"])].append(
            {
                "baseline_nmse": float(row["baseline_nmse"]),
                "method_nmse": float(row["method_nmse"]),
                "relative_nmse_reduction": float(row["relative_nmse_reduction"]),
            }
        )
    prompt_values = [_mean(by_prompt[index]) for index in TEST_INDICES]
    rng = random.Random(SEED)
    boot = []
    for _ in range(int(THRESHOLDS["bootstrap_samples"])):
        sample = [prompt_values[rng.randrange(len(prompt_values))] for _ in prompt_values]
        boot.append(_mean(sample))
    boot.sort()
    ci_low = boot[int(0.025 * (len(boot) - 1))]
    ci_high = boot[int(0.975 * (len(boot) - 1))]
    position_summaries = {}
    for position, rows in by_position.items():
        position_summaries[str(position)] = {
            "mean_baseline_nmse": _mean([row["baseline_nmse"] for row in rows]),
            "mean_method_nmse": _mean([row["method_nmse"] for row in rows]),
            "mean_relative_nmse_reduction": _mean([row["relative_nmse_reduction"] for row in rows]),
            "method_non_worse": _mean([row["method_nmse"] for row in rows])
            <= _mean([row["baseline_nmse"] for row in rows]),
        }
    mean_reduction = _mean(prompt_values)
    return {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "metric_name": "shape_conditioned_ssm_state_codec_nmse_reduction",
        "model_id": MODEL_ID,
        "positions": list(POSITIONS),
        "calibration_indices": list(CALIBRATION_INDICES),
        "test_indices": list(TEST_INDICES),
        "mamba_layer_count": len(mamba_layers),
        "row_count": len(per_row),
        "mean_relative_nmse_reduction": mean_reduction,
        "bootstrap_ci95": {"ci95_low": float(ci_low), "ci95_high": float(ci_high)},
        "prompt_mean_relative_nmse_reduction": {
            str(index): prompt_values[offset] for offset, index in enumerate(TEST_INDICES)
        },
        "position_summaries": position_summaries,
        "layer_summaries": layer_summaries,
        "per_row": per_row,
        "thresholds": THRESHOLDS,
        "decision_inputs": {
            "mean_reduction_ge_0_10": mean_reduction >= THRESHOLDS["mean_relative_nmse_reduction_min"],
            "ci_low_gt_0_05": ci_low > THRESHOLDS["bootstrap_ci95_low_gt"],
            "method_non_worse_each_position": all(
                row["method_non_worse"] for row in position_summaries.values()
            ),
        },
    }


def run(args: argparse.Namespace, argv: list[str] | None) -> int:
    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(json.dumps({"created_at_utc": utc_now(), "event": "run_started"}) + "\n")

    random.seed(SEED)
    prompt_manifest, prompt_reasons = load_prompts()
    environment = shared.build_environment(schema_version=SCHEMA_VERSION)
    model_provenance = shared.resolve_model_snapshot(MODEL_ID, schema_version=SCHEMA_VERSION)
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": utc_now(),
        "argv": sys.argv if argv is None else ["run_ssc_phase0.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "ssm_shape_codec",
        "run_dir": str(run_dir),
        "positions": list(POSITIONS),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "local_files_only": True,
            "manual_cache_decode": True,
            "max_new_tokens": max(POSITIONS),
        },
    }
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": SEED,
        "use": "bootstrap/analysis seed preregistered for SSM Shape-Conditioned Codec Phase 0",
    }
    for name, payload in [
        ("prompt_manifest.json", prompt_manifest),
        ("environment.json", environment),
        ("model_provenance.json", model_provenance),
        ("command_metadata.json", command_metadata),
        ("random_seed.json", random_seed),
    ]:
        write_json(run_dir / name, payload)
    if prompt_reasons:
        write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
        write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        return 1
    if not model_provenance.get("snapshot_path"):
        write_json(run_dir / "infra_error.json", {"reasons": [model_provenance.get("error")]})
        write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        return 1
    model, tokenizer, device = shared.load_model_and_tokenizer(
        model_provenance, dtype_name=args.dtype, device_name=args.device
    )
    state_manifest = shared.capture_ssm_states(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompt_manifest["prompts"],
        positions=POSITIONS,
        max_new_tokens=max(POSITIONS),
        batch_size=args.batch_size,
        states_dir=run_dir / "ssm_states",
        run_events_path=run_events_path,
    )
    write_json(run_dir / "raw_state_manifest.json", state_manifest)
    metrics = compute_codec_metrics(run_dir, state_manifest)
    metrics.update(
        {
            "created_at_utc": utc_now(),
            "prompt_sha256": prompt_manifest["prompt_sha256"],
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        }
    )
    write_json(run_dir / "codec_metrics.json", metrics)
    run_events_path.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--run-id", default=f"ssc_phase0_{timestamp}")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)
    return run(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
