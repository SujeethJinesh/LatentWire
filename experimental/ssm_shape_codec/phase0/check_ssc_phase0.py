#!/usr/bin/env python3
"""Check SSM Shape-Conditioned Codec Phase 0 packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.shared import check_phase0_gate as shared_check
from experimental.shared import run_phase0_branch as shared_run
from experimental.ssm_shape_codec.phase0 import run_ssc_phase0 as runner

PASS = "PASS_SSC_PHASE0_SHAPE_CODEC_GAIN"
KILL = "KILL_SSC_PHASE0_NO_CODEC_GAIN"
FAIL_INFRA = "FAIL_INFRA_SSC_PHASE0"
REQUIRED_FILES = [
    "environment.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "raw_state_manifest.json",
    "codec_metrics.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b


def validate_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict):
            infra.append("artifact_hashes contains non-object row")
            continue
        rel = str(row.get("path"))
        seen.add(rel)
        path = run_dir / rel
        if rel in {"artifact_hashes.json", "checker_result.json", "artifact_check.json"}:
            infra.append(f"artifact_hashes must not include checker/self artifact {rel}")
            continue
        if not path.is_file():
            infra.append(f"hashed artifact missing: {rel}")
            continue
        if row.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if row.get("sha256") != shared_check.file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")
    excluded = {"artifact_hashes.json", "checker_result.json", "artifact_check.json"}
    disk_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and str(path.relative_to(run_dir)) not in excluded
    }
    for rel in sorted(disk_files.difference(seen)):
        infra.append(f"artifact_hashes missing on-disk artifact {rel}")


def validate_raw_state_artifacts(run_dir: Path, state_manifest: dict[str, Any], infra: list[str]) -> None:
    import numpy as np

    expected_keys = {
        runner.npz_key(prompt_index, position)
        for prompt_index in runner.ALL_INDICES
        for position in runner.POSITIONS
    }
    layers = state_manifest.get("layers")
    if not isinstance(layers, list):
        infra.append("raw_state_manifest.layers must be a list")
        return
    for layer in layers:
        if not isinstance(layer, dict):
            infra.append("raw_state_manifest.layers contains non-object row")
            continue
        layer_index = int(layer.get("layer_index", -1))
        expected_mamba = layer_index in shared_check.GRANITE_TINY_MAMBA_LAYER_INDICES
        if bool(layer.get("is_mamba")) != expected_mamba:
            infra.append(f"raw_state_manifest layer {layer_index} is_mamba mismatch")
            continue
        if not expected_mamba:
            if layer.get("artifact"):
                infra.append(f"non-Mamba layer {layer_index} must not have artifact")
            continue
        artifact_rel = str(layer.get("artifact", ""))
        if not artifact_rel:
            infra.append(f"Mamba layer {layer_index} missing raw state artifact")
            continue
        artifact = run_dir / artifact_rel
        if not artifact.is_file():
            infra.append(f"Mamba layer {layer_index} artifact missing: {artifact_rel}")
            continue
        if layer.get("artifact_sha256") != shared_check.file_sha256(artifact):
            infra.append(f"Mamba layer {layer_index} artifact sha256 mismatch")
        if layer.get("artifact_bytes") != artifact.stat().st_size:
            infra.append(f"Mamba layer {layer_index} artifact byte size mismatch")
        if int(layer.get("record_count", -1)) != len(expected_keys):
            infra.append(f"Mamba layer {layer_index} record_count mismatch")
        if tuple(layer.get("state_shape_without_batch", [])) != shared_check.GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH:
            infra.append(f"Mamba layer {layer_index} state_shape_without_batch mismatch")
        try:
            with np.load(artifact) as data:
                actual_keys = set(data.files)
                if actual_keys != expected_keys:
                    extra = sorted(actual_keys.difference(expected_keys))[:5]
                    missing = sorted(expected_keys.difference(actual_keys))[:5]
                    infra.append(
                        f"Mamba layer {layer_index} artifact keys must be exactly held-out indices 12-23 "
                        f"x positions 100/10000; extra={extra}, missing={missing}"
                    )
                shapes = {tuple(data[key].shape) for key in data.files}
                if shapes != {shared_check.GRANITE_TINY_SSM_STATE_SHAPE_WITHOUT_BATCH}:
                    infra.append(f"Mamba layer {layer_index} raw state shape mismatch")
                for key in data.files:
                    arr = data[key]
                    if not np.isfinite(arr).all():
                        infra.append(f"Mamba layer {layer_index} artifact key {key} contains nonfinite values")
                        break
        except Exception as exc:
            infra.append(f"cannot read Mamba layer {layer_index} raw state artifact: {exc!r}")


def compare_metrics(observed: Any, computed: Any, infra: list[str], path: str = "codec_metrics") -> None:
    if isinstance(computed, dict):
        if not isinstance(observed, dict):
            infra.append(f"{path} type mismatch")
            return
        for key, value in computed.items():
            if key == "created_at_utc":
                continue
            if key not in observed:
                infra.append(f"{path}.{key} missing")
                continue
            compare_metrics(observed[key], value, infra, f"{path}.{key}")
        return
    if isinstance(computed, list):
        if not isinstance(observed, list) or len(observed) != len(computed):
            infra.append(f"{path} list length mismatch")
            return
        for index, value in enumerate(computed):
            compare_metrics(observed[index], value, infra, f"{path}[{index}]")
        return
    if not close(observed, computed):
        infra.append(f"{path} mismatch: observed={observed!r} computed={computed!r}")


def decision_from_metrics(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    inputs = metrics["decision_inputs"]
    mean_reduction = float(metrics["mean_relative_nmse_reduction"])
    ci_low = float(metrics["bootstrap_ci95"]["ci95_low"])
    non_worse = bool(inputs["method_non_worse_each_position"])
    if (
        mean_reduction >= runner.THRESHOLDS["mean_relative_nmse_reduction_min"]
        and ci_low > runner.THRESHOLDS["bootstrap_ci95_low_gt"]
        and non_worse
    ):
        return PASS, [
            f"shape-codec pass: mean relative NMSE reduction {mean_reduction:.8f}, "
            f"CI low {ci_low:.8f}, method non-worse at both positions"
        ]
    return KILL, [
        f"shape-codec kill: mean relative NMSE reduction {mean_reduction:.8f}, "
        f"CI low {ci_low:.8f}, method_non_worse_each_position={non_worse}"
    ]


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
        state_manifest = load_json(run_dir / "raw_state_manifest.json")
        metrics = load_json(run_dir / "codec_metrics.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
    except Exception as exc:
        result = {"decision": FAIL_INFRA, "artifact_complete": False, "reasons": [*infra, repr(exc)]}
        if run_dir.is_dir():
            write_json(run_dir / "checker_result.json", result)
        return result

    if environment.get("schema_version") != f"{runner.SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if model.get("model_id") != runner.MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if model.get("local_files_only") is not True:
        infra.append("model provenance must record local_files_only true")
    if command.get("branch") != "ssm_shape_codec":
        infra.append("command_metadata.branch must be ssm_shape_codec")
    if list(command.get("positions", [])) != list(runner.POSITIONS):
        infra.append("command_metadata.positions mismatch")
    if int(random_seed.get("seed", -1)) != runner.SEED:
        infra.append("random_seed.seed mismatch")
    if prompt_manifest.get("selection") != "heldout_indices_12_23":
        infra.append("prompt selection must be heldout_indices_12_23")
    if tuple(prompt_manifest.get("calibration_indices", [])) != runner.CALIBRATION_INDICES:
        infra.append("prompt calibration_indices mismatch")
    if tuple(prompt_manifest.get("test_indices", [])) != runner.TEST_INDICES:
        infra.append("prompt test_indices mismatch")
    prompts = prompt_manifest.get("prompts", [])
    if tuple(int(row.get("index", -1)) for row in prompts) != runner.ALL_INDICES:
        infra.append("prompt manifest must contain exactly indices 12-23")
    if prompt_manifest.get("prompt_sha256") != shared_run.prompt_payload_sha256(prompts):
        infra.append("prompt payload sha256 mismatch")
    if list(state_manifest.get("positions", [])) != list(runner.POSITIONS):
        infra.append("raw_state_manifest.positions mismatch")
    if int(state_manifest.get("trace_count", 0)) != len(runner.ALL_INDICES):
        infra.append("raw_state_manifest.trace_count mismatch")
    if int(state_manifest.get("layer_count", 0)) != len(shared_check.GRANITE_TINY_LAYER_TYPES):
        infra.append("raw_state_manifest.layer_count must be 40")
    if int(state_manifest.get("mamba_layer_count", 0)) != len(shared_check.GRANITE_TINY_MAMBA_LAYER_INDICES):
        infra.append("raw_state_manifest.mamba_layer_count must be 36")
    if state_manifest.get("layer_type_source") != shared_check.REAL_SSML_LAYER_TYPE_SOURCE:
        infra.append("raw_state_manifest.layer_type_source must come from real model config")
    if state_manifest.get("cache_state_source") not in shared_check.REAL_SSML_CACHE_STATE_SOURCES:
        infra.append("raw_state_manifest.cache_state_source must identify real cache path")
    expected_records = len(runner.ALL_INDICES) * len(shared_check.GRANITE_TINY_MAMBA_LAYER_INDICES) * len(runner.POSITIONS)
    if int(state_manifest.get("capture_record_count", -1)) != expected_records:
        infra.append("raw_state_manifest.capture_record_count mismatch")
    if metrics.get("thresholds") != runner.THRESHOLDS:
        infra.append("codec_metrics.thresholds mismatch")
    if metrics.get("model_id") != runner.MODEL_ID:
        infra.append("codec_metrics.model_id mismatch")
    validate_hashes(run_dir, artifact_hashes, infra)
    validate_raw_state_artifacts(run_dir, state_manifest, infra)

    computed = None
    if not infra:
        try:
            computed = runner.compute_codec_metrics(run_dir, state_manifest)
            compare_metrics(metrics, computed, infra)
        except Exception as exc:
            infra.append(f"cannot recompute codec metrics: {exc!r}")
    if infra:
        result = {"decision": FAIL_INFRA, "artifact_complete": False, "run_dir": str(run_dir), "reasons": infra}
    else:
        assert computed is not None
        decision, reasons = decision_from_metrics(computed)
        result = {
            "decision": decision,
            "artifact_complete": True,
            "run_dir": str(run_dir),
            "reasons": reasons,
            "mean_relative_nmse_reduction": computed["mean_relative_nmse_reduction"],
            "bootstrap_ci95": computed["bootstrap_ci95"],
            "position_summaries": computed["position_summaries"],
            "thresholds": runner.THRESHOLDS,
        }
    write_json(run_dir / "checker_result.json", result)
    write_json(
        run_dir / "artifact_check.json",
        {
            "schema_version": f"{runner.SCHEMA_VERSION}_artifact_check",
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
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
