#!/usr/bin/env python3
"""Check Phase 9 M-LAMBDA layerwise budget-waterfilling packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_mlambda.md"

SCHEMA_VERSION = "om_phase9_mlambda_v1"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
UPDATE_CADENCE = 100
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

REGIMES = ["bf16", "static_1pct", "m11b_top10", "mlambda_top10", "random_lambda_top10"]
RECOVERY_REGIMES = ["m11b_top10", "mlambda_top10", "random_lambda_top10"]

PASS = "PASS_LAMBDA"
KEEP_FOR_STACK = "KEEP_FOR_STACK_LAMBDA"
DROP = "DROP_LAMBDA"
FAIL_INFRA = "FAIL_INFRA_LAMBDA"

THRESHOLDS = {
    "pass_minus_m11b_ge": 0.05,
    "keep_minus_m11b_ge": 0.02,
    "drop_random_control_if_le": 0.0,
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
    "lambda_budget_allocation.json",
    "protected_sets.json",
    "protected_trajectories.json",
    "quantization_config.json",
    "excluded_tensors.json",
    "score_cache/bf16.json",
    "score_cache/static_1pct.json",
    "score_cache/m11b_top10.json",
    "score_cache/mlambda_top10.json",
    "score_cache/random_lambda_top10.json",
    "source_artifacts.json",
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


def bootstrap_median(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(BOOTSTRAP_SEED)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(median(sample)))
    boot.sort()
    return {"ci95_low": boot[int(0.025 * (len(boot) - 1))], "ci95_high": boot[int(0.975 * (len(boot) - 1))]}


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


def validate_metadata(loaded: dict[str, Any], infra: list[str]) -> None:
    metrics = loaded.get("metrics.json", {})
    model_key = metrics.get("model_key")
    if model_key not in MODEL_SPECS:
        infra.append("metrics.model_key must be granite or nemotron")
        return
    spec = MODEL_SPECS[model_key]
    model = loaded.get("model_provenance.json", {})
    if model.get("model_id") != spec["model_id"]:
        infra.append("model_id mismatch")
    if model.get("hf_snapshot_commit") != spec["snapshot"]:
        infra.append("model snapshot mismatch")
    if int(metrics.get("trace_count", -1)) != TRACE_COUNT:
        infra.append("trace_count mismatch")
    if int(metrics.get("scoring_position", -1)) != SCORING_POSITION:
        infra.append("scoring_position mismatch")
    if int(metrics.get("scoring_window_tokens", -1)) != SCORING_WINDOW_TOKENS:
        infra.append("scoring_window_tokens mismatch")
    allocation = loaded.get("lambda_budget_allocation.json", {})
    if allocation.get("total_budget_matches_m11b_top10") is not True:
        infra.append("lambda budget must match M11b top-10 total budget")
    if int(allocation.get("total_allocated_channels", 0)) <= 0:
        infra.append("lambda allocation must protect at least one channel")
    if not isinstance(allocation.get("layer_allocations"), dict) or not allocation.get("layer_allocations"):
        infra.append("lambda allocation must include layer_allocations")


def validate_scores(loaded: dict[str, Any], infra: list[str]) -> None:
    results = loaded.get("metrics.json", {}).get("results_by_regime", {})
    if set(results) != set(RECOVERY_REGIMES):
        infra.append("metrics.results_by_regime mismatch")
    for regime in RECOVERY_REGIMES:
        summary = results.get(regime, {})
        if summary.get("median_recovery") is None:
            infra.append(f"{regime}.median_recovery missing")
        ci = summary.get("bootstrap_ci95", {})
        if ci.get("ci95_low") is None or ci.get("ci95_high") is None:
            infra.append(f"{regime}.bootstrap_ci95 missing")
    bootstrap = loaded.get("bootstrap_ci.json", {})
    if int(bootstrap.get("bootstrap_samples", -1)) != BOOTSTRAP_SAMPLES:
        infra.append("bootstrap sample count mismatch")
    sources = loaded.get("source_artifacts.json", {})
    if not isinstance(sources.get("artifacts"), list) or not sources.get("artifacts"):
        infra.append("source_artifacts.artifacts must be non-empty")


def validate_protected_sets(loaded: dict[str, Any], infra: list[str]) -> None:
    protected = loaded.get("protected_sets.json", {})
    regimes = protected.get("regimes", {})
    expected = {"static_1pct", "m11b_top10", "mlambda_top10", "random_lambda_top10"}
    if set(regimes) != expected:
        infra.append("protected_sets.regimes mismatch")
        return
    m11b_layers = regimes["m11b_top10"]["layers"]
    mlambda_layers = regimes["mlambda_top10"]["layers"]
    random_layers = regimes["random_lambda_top10"]["layers"]
    if set(m11b_layers) != set(mlambda_layers) or set(m11b_layers) != set(random_layers):
        infra.append("layer set mismatch across M11b/M-LAMBDA/random")
    m11b_total = sum(int(layer["protected_count"]) for layer in m11b_layers.values())
    lambda_total = sum(int(layer["protected_count"]) for layer in mlambda_layers.values())
    random_total = sum(int(layer["protected_count"]) for layer in random_layers.values())
    if m11b_total != lambda_total or m11b_total != random_total:
        infra.append("protected total budget mismatch")


def validate_hashes(run_dir: Path, loaded: dict[str, Any], infra: list[str]) -> None:
    entries = loaded.get("artifact_hashes.json", {}).get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in [*REQUIRED_FILES, *[rel for rel in OPTIONAL_FILES if (run_dir / rel).is_file()]]:
        if rel == "artifact_hashes.json":
            continue
        path = run_dir / rel
        item = by_path.get(rel)
        if item is None:
            infra.append(f"artifact hash missing for {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact hash byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact hash mismatch for {rel}")


def decision_from_metrics(metrics: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    results = metrics.get("results_by_regime", {})
    lam = results.get("mlambda_top10", {})
    m11b = results.get("m11b_top10", {})
    random_control = results.get("random_lambda_top10", {})
    if lam.get("median_recovery") is None or m11b.get("median_recovery") is None:
        return FAIL_INFRA, ["missing M-LAMBDA or M11b median"], {}
    lam_med = float(lam["median_recovery"])
    m11b_med = float(m11b["median_recovery"])
    random_med = None if random_control.get("median_recovery") is None else float(random_control["median_recovery"])
    details = {
        "mlambda_median": lam_med,
        "m11b_top10_median": m11b_med,
        "random_lambda_median": random_med,
        "mlambda_minus_m11b": lam_med - m11b_med,
        "mlambda_minus_random": None if random_med is None else lam_med - random_med,
    }
    if lam_med - m11b_med >= THRESHOLDS["pass_minus_m11b_ge"] and lam.get("bootstrap_ci95", {}).get("ci95_low", -1.0) > 0.0:
        return PASS, ["M-LAMBDA beats M11b by at least 0.05 with positive CI lower bound"], details
    if random_med is not None and lam_med - random_med <= THRESHOLDS["drop_random_control_if_le"]:
        return DROP, ["M-LAMBDA does not beat random layer-waterfilled control"], details
    if lam_med - m11b_med >= THRESHOLDS["keep_minus_m11b_ge"]:
        return KEEP_FOR_STACK, ["M-LAMBDA has positive marginal direction but not pass threshold"], details
    return DROP, ["M-LAMBDA does not improve enough to retain by default"], details


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    loaded = validate_files(run_dir, infra)
    if (run_dir / "infra_error.json").is_file():
        infra.append("infra_error.json present")
    if not infra:
        validate_metadata(loaded, infra)
        validate_scores(loaded, infra)
        validate_protected_sets(loaded, infra)
        validate_hashes(run_dir, loaded, infra)
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
