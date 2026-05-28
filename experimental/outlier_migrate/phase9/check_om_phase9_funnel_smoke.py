#!/usr/bin/env python3
"""Check Phase 9 positive-method funnel smoke packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_funnel_smoke.md"

SCHEMA_VERSION = "om_phase9_funnel_smoke_v1"
TRACE_COUNT = 3
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512

MODEL_SPECS = {
    "deepseek": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "snapshot": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "smoke_indices": [5, 11, 8],
        "rescue_floor": 0.335,
    },
    "falcon": {
        "model_id": "tiiuae/Falcon-H1-0.5B-Instruct",
        "snapshot": "8f2587ca06bff78d8fa1adfccbe8c24d5f86b368",
        "smoke_indices": [7, 1, 11],
        "rescue_floor": 0.20,
    },
}

REGIMES = [
    "bf16",
    "static_1pct",
    "m11b_top10",
    "mlambda_top10_smoke",
    "hyst_top10_smoke",
    "random_lambda_top10",
    "random_hyst_top10",
]
METHOD_REGIMES = ["mlambda_top10_smoke", "hyst_top10_smoke"]
RECOVERY_REGIMES = ["m11b_top10", *METHOD_REGIMES, "random_lambda_top10", "random_hyst_top10"]

PASS = "PASS_SMOKE_SURVIVOR"
KILL = "KILL_SMOKE_NO_SURVIVOR"
FAIL_INFRA = "FAIL_INFRA_FUNNEL_SMOKE"

THRESHOLDS = {
    "beats_m11b_by_ge": 0.10,
    "wins_trace_count_ge": 2,
    "catastrophic_negative_lt": -1.0,
    "deepseek_rescue_gt": 0.335,
    "falcon_rescue_ge": 0.20,
}

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "smoke_trace_selection.json",
    "protected_sets.json",
    "protected_trajectories.json",
    "quantization_config.json",
    "excluded_tensors.json",
    "score_cache/bf16.json",
    "score_cache/static_1pct.json",
    "score_cache/m11b_top10.json",
    "score_cache/mlambda_top10_smoke.json",
    "score_cache/hyst_top10_smoke.json",
    "score_cache/random_lambda_top10.json",
    "score_cache/random_hyst_top10.json",
    "source_artifacts.json",
    "per_trace_metrics.json",
    "metrics.json",
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
        infra.append("metrics.model_key must be deepseek or falcon")
        return
    spec = MODEL_SPECS[model_key]
    model = loaded.get("model_provenance.json", {})
    if model.get("model_id") != spec["model_id"]:
        infra.append("model_id mismatch")
    if model.get("hf_snapshot_commit") != spec["snapshot"]:
        infra.append("model snapshot mismatch")
    prompt_manifest = loaded.get("prompt_manifest.json", {})
    indices = [int(row.get("index", -1)) for row in prompt_manifest.get("prompts", [])]
    if indices != spec["smoke_indices"]:
        infra.append(f"smoke indices mismatch: expected {spec['smoke_indices']} got {indices}")
    if int(metrics.get("trace_count", -1)) != TRACE_COUNT:
        infra.append("trace_count mismatch")
    if int(metrics.get("scoring_position", -1)) != SCORING_POSITION:
        infra.append("scoring_position mismatch")
    selection = loaded.get("smoke_trace_selection.json", {})
    if selection.get("selection_source") != "docs/smoke_trace_selection.md":
        infra.append("smoke_trace_selection source mismatch")
    if selection.get("smoke_indices") != spec["smoke_indices"]:
        infra.append("smoke_trace_selection indices mismatch")


def validate_scores(loaded: dict[str, Any], infra: list[str]) -> None:
    results = loaded.get("metrics.json", {}).get("results_by_regime", {})
    if set(results) != set(RECOVERY_REGIMES):
        infra.append("metrics.results_by_regime mismatch")
    for regime in RECOVERY_REGIMES:
        summary = results.get(regime, {})
        if summary.get("median_recovery") is None and int(summary.get("included_trace_count", 0)) > 0:
            infra.append(f"{regime}.median_recovery missing")
    per_trace = loaded.get("per_trace_metrics.json", {}).get("traces", [])
    if len(per_trace) != TRACE_COUNT:
        infra.append("per_trace_metrics trace count mismatch")
    sources = loaded.get("source_artifacts.json", {})
    if not isinstance(sources.get("artifacts"), list) or not sources.get("artifacts"):
        infra.append("source_artifacts.artifacts must be non-empty")


def validate_protected_sets(loaded: dict[str, Any], infra: list[str]) -> None:
    protected = loaded.get("protected_sets.json", {})
    regimes = protected.get("regimes", {})
    expected = {"static_1pct", "m11b_top10", "mlambda_top10_smoke", "hyst_top10_smoke", "random_lambda_top10", "random_hyst_top10"}
    if set(regimes) != expected:
        infra.append("protected_sets.regimes mismatch")
        return
    base_layers = set(regimes["m11b_top10"]["layers"])
    for regime in expected:
        if set(regimes[regime]["layers"]) != base_layers:
            infra.append(f"{regime} layer keys mismatch")
    m11b_total = sum(int(layer["protected_count"]) for layer in regimes["m11b_top10"]["layers"].values())
    for regime in ["mlambda_top10_smoke", "hyst_top10_smoke", "random_lambda_top10", "random_hyst_top10"]:
        total = sum(int(layer["protected_count"]) for layer in regimes[regime]["layers"].values())
        if total != m11b_total:
            infra.append(f"{regime} total budget mismatch")


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
    model_key = str(metrics["model_key"])
    results = metrics["results_by_regime"]
    m11b_values = results["m11b_top10"].get("per_trace_recovery_included", [])
    survivors: dict[str, Any] = {}
    reasons: list[str] = []
    for regime in METHOD_REGIMES:
        summary = results[regime]
        values = summary.get("per_trace_recovery_included", [])
        if not values or not m11b_values:
            continue
        med = float(summary["median_recovery"])
        m11b_med = float(results["m11b_top10"]["median_recovery"])
        wins = sum(1 for left, right in zip(values, m11b_values) if float(left) > float(right))
        catastrophic = sum(1 for value in values if float(value) < THRESHOLDS["catastrophic_negative_lt"])
        gates: list[str] = []
        if med - m11b_med >= THRESHOLDS["beats_m11b_by_ge"]:
            gates.append("beats_m11b_by_ge_0.10")
        if wins >= THRESHOLDS["wins_trace_count_ge"]:
            gates.append("wins_2_of_3_smoke_traces")
        if model_key == "falcon" and med >= THRESHOLDS["falcon_rescue_ge"]:
            gates.append("falcon_rescue_ge_0.20")
        if model_key == "deepseek" and med > THRESHOLDS["deepseek_rescue_gt"] and med > m11b_med:
            gates.append("deepseek_rescue_gt_0.335_and_above_m11b")
        if gates and catastrophic == 0:
            survivors[regime] = {"median_recovery": med, "m11b_median": m11b_med, "wins": wins, "gates": gates}
            reasons.append(f"{regime} survives smoke via {', '.join(gates)}")
    if survivors:
        return PASS, reasons, {"survivors": survivors}
    return KILL, ["no LAMBDA/HYST smoke survivor"], {"survivors": {}}


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
