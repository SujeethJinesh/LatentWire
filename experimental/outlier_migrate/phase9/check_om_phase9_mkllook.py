#!/usr/bin/env python3
"""Check Phase 9 M-KLLOOK offline KL-lookahead oracle packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_mkllook.md"

SCHEMA_VERSION = "om_phase9_mkllook_v1"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260528

MODEL_SPECS = {
    "granite": {
        "model_id": "ibm-granite/granite-4.0-h-small",
        "snapshot": "b8c0982bab7fde4eb48110f5a069527c008fab39",
    },
    "nemotron": {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "snapshot": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
    },
}

REGIMES = ["bf16", "static_1pct", "m11b_top10", "mkllook_top10", "random_top10"]
RECOVERY_REGIMES = ["m11b_top10", "mkllook_top10", "random_top10"]

CEILING_HIGH = "CEILING_HIGH_PROXY_BOTTLENECK"
CEILING_CONFIRMED_LOW = "CEILING_CONFIRMED_LOW"
AMBIGUOUS = "AMBIGUOUS_ORACLE"
FAIL_INFRA = "FAIL_INFRA_MKLLOOK"

THRESHOLDS = {
    "oracle_minus_m11b_high_ge": 0.10,
    "oracle_minus_random_high_ge": 0.15,
    "oracle_minus_m11b_low_le": 0.05,
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
}

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "oracle_sampling_config.json",
    "candidate_deltas.json",
    "protected_sets.json",
    "quantization_config.json",
    "excluded_tensors.json",
    "per_trace_metrics.json",
    "metrics.json",
    "bootstrap_ci.json",
    "control_metrics.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
OPTIONAL_FILES = [
    "activation_magnitudes.jsonl.gz",
    "activation_magnitude_manifest.json",
    "bf16_traces.jsonl.gz",
    "bf16_trace_manifest.json",
    "source_artifacts.json",
]


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


def is_close(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(left)), abs(float(right)))


def validate_files(run_dir: Path, infra: list[str]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for rel in REQUIRED_FILES:
        path = run_dir / rel
        if not path.is_file():
            infra.append(f"missing required file: {rel}")
            continue
        if path.suffix == ".json":
            try:
                loaded[rel] = load_json(path)
            except Exception as exc:
                infra.append(f"invalid JSON in {rel}: {exc}")
    for rel in OPTIONAL_FILES:
        path = run_dir / rel
        if path.is_file() and path.suffix == ".json":
            try:
                loaded[rel] = load_json(path)
            except Exception as exc:
                infra.append(f"invalid JSON in optional {rel}: {exc}")
    return loaded


def validate_basic_metadata(run_dir: Path, loaded: dict[str, Any], infra: list[str]) -> None:
    if not PREREG_PATH.is_file():
        infra.append("preregistration file missing")
    metrics = loaded.get("metrics.json", {})
    model_key = metrics.get("model_key")
    if model_key not in MODEL_SPECS:
        infra.append("metrics.model_key must be granite or nemotron")
        return
    spec = MODEL_SPECS[model_key]
    provenance = loaded.get("model_provenance.json", {})
    if provenance.get("model_id") != spec["model_id"]:
        infra.append("model_provenance.model_id mismatch")
    if provenance.get("hf_snapshot_commit") != spec["snapshot"]:
        infra.append("model snapshot mismatch")
    if int(metrics.get("trace_count", -1)) != TRACE_COUNT:
        infra.append("metrics.trace_count mismatch")
    if int(metrics.get("scoring_position", -1)) != SCORING_POSITION:
        infra.append("metrics.scoring_position mismatch")
    if int(metrics.get("scoring_window_tokens", -1)) != SCORING_WINDOW_TOKENS:
        infra.append("metrics.scoring_window_tokens mismatch")
    prompt_manifest = loaded.get("prompt_manifest.json", {})
    if int(prompt_manifest.get("prompt_count", -1)) != TRACE_COUNT:
        infra.append("prompt_manifest.prompt_count mismatch")
    sampling = loaded.get("oracle_sampling_config.json", {})
    if int(sampling.get("candidate_count", 0)) <= 0:
        infra.append("oracle_sampling_config.candidate_count must be positive")
    if int(sampling.get("sampled_layer_count", 0)) <= 0:
        infra.append("oracle_sampling_config.sampled_layer_count must be positive")
    if sampling.get("frozen_before_scoring") is not True:
        infra.append("oracle_sampling_config must mark frozen_before_scoring=true")


def validate_score_artifacts(loaded: dict[str, Any], infra: list[str]) -> None:
    metrics = loaded.get("metrics.json", {})
    results = metrics.get("results_by_regime", {})
    if set(results) != set(RECOVERY_REGIMES):
        infra.append("metrics.results_by_regime must contain exactly M11b, M-KLLOOK, and random top-10")
    for regime in RECOVERY_REGIMES:
        summary = results.get(regime, {})
        if summary.get("median_recovery") is None:
            infra.append(f"{regime}.median_recovery missing")
        ci = summary.get("bootstrap_ci95", {})
        if ci.get("ci95_low") is None or ci.get("ci95_high") is None:
            infra.append(f"{regime}.bootstrap_ci95 missing bounds")
    bootstrap = loaded.get("bootstrap_ci.json", {})
    if int(bootstrap.get("bootstrap_samples", -1)) != BOOTSTRAP_SAMPLES:
        infra.append("bootstrap sample count mismatch")
    if int(bootstrap.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED:
        infra.append("bootstrap seed mismatch")
    deltas = loaded.get("candidate_deltas.json", {})
    rows = deltas.get("candidates", [])
    if not isinstance(rows, list) or not rows:
        infra.append("candidate_deltas.candidates must be non-empty")
    required_candidate_keys = {"layer_index", "channel_index", "delta_kl", "selected_by_oracle"}
    for idx, row in enumerate(rows[:10]):
        if not required_candidate_keys.issubset(row):
            infra.append(f"candidate_deltas.candidates[{idx}] missing required keys")


def validate_protected_sets(loaded: dict[str, Any], infra: list[str]) -> None:
    protected = loaded.get("protected_sets.json", {})
    regimes = protected.get("regimes", {})
    expected = {"static_1pct", "m11b_top10", "mkllook_top10", "random_top10"}
    if set(regimes) != expected:
        infra.append("protected_sets.regimes mismatch")
        return
    oracle_layers = regimes.get("mkllook_top10", {}).get("layers", {})
    random_layers = regimes.get("random_top10", {}).get("layers", {})
    if not oracle_layers:
        infra.append("mkllook_top10 has no layer entries")
    if set(oracle_layers) != set(random_layers):
        infra.append("oracle and random protected layer sets differ")
    for layer_key, layer in oracle_layers.items():
        if int(layer.get("protected_count", 0)) <= 0:
            infra.append(f"mkllook layer {layer_key} protected_count must be positive")
        if len(layer.get("protected_channels", [])) != int(layer.get("protected_count", -1)):
            infra.append(f"mkllook layer {layer_key} protected channel count mismatch")
        if layer.get("source") != "sampled_forward_kl_lookahead":
            infra.append(f"mkllook layer {layer_key} source mismatch")


def decision_from_metrics(metrics: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    results = metrics.get("results_by_regime", {})
    oracle = results.get("mkllook_top10", {})
    m11b = results.get("m11b_top10", {})
    random = results.get("random_top10", {})
    if oracle.get("median_recovery") is None or m11b.get("median_recovery") is None:
        return FAIL_INFRA, ["missing oracle or M11b median recovery"], {}
    oracle_med = float(oracle["median_recovery"])
    m11b_med = float(m11b["median_recovery"])
    random_med = None if random.get("median_recovery") is None else float(random["median_recovery"])
    oracle_minus_m11b = oracle_med - m11b_med
    oracle_minus_random = None if random_med is None else oracle_med - random_med
    details = {
        "oracle_median": oracle_med,
        "m11b_top10_median": m11b_med,
        "random_top10_median": random_med,
        "oracle_minus_m11b": oracle_minus_m11b,
        "oracle_minus_random": oracle_minus_random,
    }
    if (
        oracle_minus_m11b >= THRESHOLDS["oracle_minus_m11b_high_ge"]
        and oracle_minus_random is not None
        and oracle_minus_random >= THRESHOLDS["oracle_minus_random_high_ge"]
    ):
        return CEILING_HIGH, ["oracle beats M11b and random by preregistered margins"], details
    if oracle_minus_m11b <= THRESHOLDS["oracle_minus_m11b_low_le"] or (
        oracle_minus_random is not None and oracle_minus_random <= 0.0
    ):
        return CEILING_CONFIRMED_LOW, ["oracle does not materially beat M11b or random control"], details
    return AMBIGUOUS, ["oracle gain is positive but below high-ceiling threshold"], details


def validate_artifact_hashes(run_dir: Path, loaded: dict[str, Any], infra: list[str]) -> None:
    hashes = loaded.get("artifact_hashes.json", {})
    entries = hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in [*REQUIRED_FILES, *[rel for rel in OPTIONAL_FILES if (run_dir / rel).is_file()]]:
        if rel == "artifact_hashes.json":
            continue
        item = by_path.get(rel)
        if item is None:
            infra.append(f"artifact hash missing for {rel}")
            continue
        if item.get("bytes") != (run_dir / rel).stat().st_size:
            infra.append(f"artifact hash byte mismatch for {rel}")
        actual = file_sha256(run_dir / rel)
        if actual != item.get("sha256"):
            infra.append(f"artifact hash mismatch for {rel}")


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    loaded = validate_files(run_dir, infra)
    if (run_dir / "infra_error.json").is_file():
        infra.append("infra_error.json present")
    if not infra:
        validate_basic_metadata(run_dir, loaded, infra)
        validate_score_artifacts(loaded, infra)
        validate_protected_sets(loaded, infra)
        validate_artifact_hashes(run_dir, loaded, infra)
    if infra:
        decision = FAIL_INFRA
        reasons = infra
        details: dict[str, Any] = {}
    else:
        decision, reasons, details = decision_from_metrics(loaded["metrics.json"])
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "reasons": reasons,
        "details": details,
        "artifact_complete": not infra,
        "thresholds": THRESHOLDS,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "checker_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != FAIL_INFRA else 1


if __name__ == "__main__":
    raise SystemExit(main())
