#!/usr/bin/env python3
"""Check Stage 1 E1 cross-model KL and FFT diagnostic packets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_stage1_e1_cross_model_kl_fft.md"

SCHEMA_VERSION = "om_stage1_e1_cross_model_kl_fft_v1"
TRACE_COUNT = 12
MAX_NEW_TOKENS = 20000
KL_REGIMES = ["bf16_reference", "static_1pct", "decdec_reactive_top1_proxy", "m11_alpha_0_5"]
QUANTIZED_REGIMES = [regime for regime in KL_REGIMES if regime != "bf16_reference"]
MODEL_KEYS = ["nemotron3_nano", "deepseek_r1_distill_qwen_1_5b", "falcon_h1_0_5b"]

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "nemotron3_nano": {
        "label": "Nemotron-3-Nano",
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "snapshot": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
        "source_run_dir": "experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z",
    },
    "deepseek_r1_distill_qwen_1_5b": {
        "label": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "snapshot": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "source_run_dir": "experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z",
    },
    "falcon_h1_0_5b": {
        "label": "Falcon-H1-0.5B",
        "model_id": "tiiuae/Falcon-H1-0.5B-Instruct",
        "snapshot": "8f2587ca06bff78d8fa1adfccbe8c24d5f86b368",
        "source_run_dir": "experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z",
    },
}

PASS_DECISION = "PASS_E1_CROSS_MODEL_KL_FFT"
AMBIGUOUS_DECISION = "AMBIGUOUS_E1_PARTIAL_REPLICATION"
KILL_DECISION = "KILL_E1_MECHANISM_NOT_CROSS_MODEL"
FAIL_INFRA = "FAIL_INFRA_E1"

REQUIRED_MODEL_FILES = [
    "model_provenance.json",
    "prompt_manifest.json",
    "bf16_traces.jsonl.gz",
    "bf16_trace_manifest.json",
    "activation_means.npz",
    "activation_summary_manifest.json",
    "protected_sets.json",
    "kl_positions.json",
    "kl_rows.jsonl.gz",
    "kl_summary.json",
    "growth_model_fits.json",
    "spectral_summary.json",
]

REQUIRED_ROOT_FILES = [
    "environment.json",
    "environment.txt",
    "command_metadata.json",
    "random_seed.json",
    "source_artifacts.json",
    "stage1_e1_summary.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]


def dense_grid_positions() -> list[int]:
    values = set(range(1, 513))
    values.update(range(520, 2001, 10))
    values.update(range(2050, 10001, 50))
    values.update(range(10100, 20001, 100))
    return sorted(values)


def spectral_positions() -> list[int]:
    return list(range(100, MAX_NEW_TOKENS + 1, 100))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_stage1_e1_")]
    if not candidates:
        raise FileNotFoundError(f"no Stage 1 E1 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_artifact_hashes(run_dir: Path, infra: list[str]) -> None:
    path = run_dir / "artifact_hashes.json"
    if not path.is_file():
        infra.append("missing artifact_hashes.json")
        return
    payload = load_json(path)
    entries = payload.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for artifact in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = str(artifact.relative_to(run_dir))
        if rel in {"artifact_hashes.json", "checker_result.json", "artifact_check.json"}:
            continue
        row = by_path.get(rel)
        if row is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if row.get("bytes") != artifact.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if row.get("sha256") != file_sha256(artifact):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


def validate_kl_rows(model_dir: Path, expected_positions: set[int], infra: list[str]) -> None:
    seen: dict[tuple[int, str], set[int]] = {
        (trace_index, regime): set() for trace_index in range(TRACE_COUNT) for regime in KL_REGIMES
    }
    with gzip.open(model_dir / "kl_rows.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            trace_index = int(row.get("prompt_index", -1))
            regime = str(row.get("regime"))
            position = int(row.get("decode_position", -1))
            kl = float(row.get("kl_bf16_q", float("nan")))
            if trace_index not in range(TRACE_COUNT):
                infra.append(f"{model_dir.name} kl row {line_no}: invalid prompt_index {trace_index}")
                continue
            if regime not in KL_REGIMES:
                infra.append(f"{model_dir.name} kl row {line_no}: invalid regime {regime}")
                continue
            if position not in expected_positions:
                infra.append(f"{model_dir.name} kl row {line_no}: unexpected position {position}")
                continue
            if not math.isfinite(kl) or kl < -1e-8:
                infra.append(f"{model_dir.name} kl row {line_no}: invalid KL {kl}")
                continue
            seen[(trace_index, regime)].add(position)
    for key, positions in seen.items():
        if positions != expected_positions:
            infra.append(f"{model_dir.name} kl rows incomplete for {key}: got {len(positions)} positions")


def model_passes(model_dir: Path) -> tuple[bool, dict[str, Any]]:
    fits = load_json(model_dir / "growth_model_fits.json")
    spectral = load_json(model_dir / "spectral_summary.json")
    fit_classes = {
        regime: fits.get("regime_fits", {}).get(regime, {}).get("best_fit")
        for regime in QUANTIZED_REGIMES
    }
    sqrt_all = all(value == "sublinear_sqrt" for value in fit_classes.values())
    entropy = spectral.get("median_normalized_spectral_entropy")
    autocorr = spectral.get("median_autocorrelation_length_tokens")
    entropy_ok = entropy is not None and float(entropy) >= 0.75
    autocorr_ok = autocorr is not None and 50.0 <= float(autocorr) <= 200.0
    return sqrt_all and entropy_ok and autocorr_ok, {
        "fit_classes": fit_classes,
        "median_normalized_spectral_entropy": entropy,
        "median_autocorrelation_length_tokens": autocorr,
        "sqrt_sublinear_all_regimes": sqrt_all,
        "entropy_ok": entropy_ok,
        "autocorr_ok": autocorr_ok,
    }


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    missing_root = [rel for rel in REQUIRED_ROOT_FILES if not (run_dir / rel).is_file()]
    if missing_root:
        infra.append(f"missing root files: {missing_root}")

    model_results: dict[str, Any] = {}
    expected_kl_positions = set(dense_grid_positions())
    expected_spectral_positions = spectral_positions()
    for model_key in MODEL_KEYS:
        model_dir = run_dir / model_key
        if not model_dir.is_dir():
            infra.append(f"missing model directory {model_key}")
            continue
        missing = [rel for rel in REQUIRED_MODEL_FILES if not (model_dir / rel).is_file()]
        if missing:
            infra.append(f"{model_key}: missing files {missing}")
            continue
        provenance = load_json(model_dir / "model_provenance.json")
        expected = MODEL_CONFIGS[model_key]
        if provenance.get("model_id") != expected["model_id"]:
            infra.append(f"{model_key}: model_id mismatch")
        if provenance.get("hf_snapshot_commit") != expected["snapshot"]:
            infra.append(f"{model_key}: snapshot mismatch")
        kl_positions_payload = load_json(model_dir / "kl_positions.json")
        if [int(value) for value in kl_positions_payload.get("positions", [])] != sorted(expected_kl_positions):
            infra.append(f"{model_key}: KL positions mismatch")
        activation_manifest = load_json(model_dir / "activation_summary_manifest.json")
        if [int(value) for value in activation_manifest.get("positions", [])] != expected_spectral_positions:
            infra.append(f"{model_key}: spectral positions mismatch")
        validate_kl_rows(model_dir, expected_kl_positions, infra)
        if not infra:
            passed, summary = model_passes(model_dir)
            model_results[model_key] = {"model_pass": passed, **summary}

    if not infra:
        validate_artifact_hashes(run_dir, infra)

    if infra:
        decision = FAIL_INFRA
        reasons = infra
    else:
        pass_count = sum(1 for row in model_results.values() if row["model_pass"])
        if pass_count == len(MODEL_KEYS):
            decision = PASS_DECISION
        elif pass_count >= 1:
            decision = AMBIGUOUS_DECISION
        else:
            decision = KILL_DECISION
        reasons = [f"{pass_count}/{len(MODEL_KEYS)} models satisfy the E1 model-level criterion"]

    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": not infra,
        "reasons": reasons,
        "run_dir": str(run_dir),
        "model_results": model_results,
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(
        run_dir / "artifact_check.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_artifact_check",
            "decision": decision,
            "artifact_complete": not infra,
            "reasons": reasons,
            "run_dir": str(run_dir),
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir or latest_run_dir()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
