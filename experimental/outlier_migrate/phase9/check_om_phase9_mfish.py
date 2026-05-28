#!/usr/bin/env python3
"""Check Phase 9 M-FISH packets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_mfish.md"

SCHEMA_VERSION = "om_phase9_mfish_v1"
MODEL_KEY = "deepseek"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_SNAPSHOT = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
UPDATE_CADENCE = 100
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260528
ALPHA = 0.3

REGIMES = [
    "bf16",
    "static_1pct",
    "m11b_top5",
    "m11b_top10",
    "static_top10",
    "mfish_top5",
    "mfish_top10",
    "random_fisher_top10",
]
RECOVERY_REGIMES = [regime for regime in REGIMES if regime not in {"bf16", "static_1pct"}]

PASS_ARCHITECTURE_FILL = "PASS_MFISH_ARCHITECTURE_FILL"
PASS_STRONG = "PASS_MFISH_STRONG"
AMBIGUOUS = "AMBIGUOUS_MFISH"
KILL = "KILL_MFISH"
FAIL_INFRA = "FAIL_INFRA_MFISH"

THRESHOLDS = {
    "architecture_fill_margin_ge": 0.05,
    "architecture_fill_median_gt": 0.30,
    "architecture_fill_ci_low_gt": 0.0,
    "strong_median_gt": 0.55,
    "strong_ci_low_gt": 0.20,
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
    "fisher_weights.json",
    "protected_sets.json",
    "protected_trajectories.json",
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap_median(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(BOOTSTRAP_SEED)
    medians = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(float(median(sample)))
    medians.sort()
    return [medians[int(0.025 * (len(medians) - 1))], medians[int(0.975 * (len(medians) - 1))]]


def decision_from_summaries(summaries: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    mfish = summaries["mfish_top10"]
    m11b = summaries["m11b_top10"]
    random_control = summaries["random_fisher_top10"]
    median_mfish = float(mfish["median_recovery"])
    median_m11b = float(m11b["median_recovery"])
    median_random = float(random_control["median_recovery"])
    ci_low = float(mfish["bootstrap_ci95"][0])
    margin = median_mfish - median_m11b
    if median_mfish > THRESHOLDS["strong_median_gt"] and ci_low > THRESHOLDS["strong_ci_low_gt"]:
        return PASS_STRONG, ["M-FISH top-10 exceeds strong median and CI thresholds"]
    if (
        margin >= THRESHOLDS["architecture_fill_margin_ge"]
        and median_mfish > THRESHOLDS["architecture_fill_median_gt"]
        and ci_low > THRESHOLDS["architecture_fill_ci_low_gt"]
        and median_mfish > median_random
    ):
        return PASS_ARCHITECTURE_FILL, ["M-FISH improves over M11b and random-Fisher on DeepSeek"]
    if median_mfish <= median_m11b or median_mfish <= median_random:
        return KILL, ["M-FISH top-10 does not beat M11b top-10 and random-Fisher"]
    return AMBIGUOUS, ["M-FISH is positive but does not clear PASS thresholds"]


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    for relative in REQUIRED_FILES:
        if not (run_dir / relative).is_file():
            infra.append(f"missing {relative}")
    if infra:
        payload = {"schema_version": f"{SCHEMA_VERSION}_checker", "decision": FAIL_INFRA, "infractions": infra}
        write_json(run_dir / "checker_result.json", payload)
        return payload

    metrics = load_json(run_dir / "metrics.json")
    if metrics.get("model_id") != MODEL_ID:
        infra.append("metrics.model_id mismatch")
    if metrics.get("trace_count") != TRACE_COUNT:
        infra.append("trace_count mismatch")
    if metrics.get("scoring_position") != SCORING_POSITION:
        infra.append("scoring_position mismatch")
    if metrics.get("alpha") != ALPHA:
        infra.append("alpha mismatch")
    if set(metrics.get("completed_regimes", [])) != set(REGIMES):
        infra.append("completed_regimes mismatch")

    provenance = load_json(run_dir / "model_provenance.json")
    if provenance.get("model_id") != MODEL_ID or provenance.get("hf_snapshot_commit") != MODEL_SNAPSHOT:
        infra.append("model provenance mismatch")

    fisher = load_json(run_dir / "fisher_weights.json")
    if fisher.get("sampled_position_count", 0) <= 0:
        infra.append("fisher sampled_position_count must be positive")

    if infra:
        payload = {"schema_version": f"{SCHEMA_VERSION}_checker", "decision": FAIL_INFRA, "infractions": infra}
        write_json(run_dir / "checker_result.json", payload)
        return payload

    decision, reasons = decision_from_summaries(metrics["results_by_regime"])
    payload = {
        "schema_version": f"{SCHEMA_VERSION}_checker",
        "decision": decision,
        "reasons": reasons,
        "metrics": {
            "mfish_top10": metrics["results_by_regime"]["mfish_top10"],
            "m11b_top10": metrics["results_by_regime"]["m11b_top10"],
            "random_fisher_top10": metrics["results_by_regime"]["random_fisher_top10"],
        },
    }
    write_json(run_dir / "checker_result.json", payload)
    return payload


def latest_run() -> Path:
    candidates = sorted(path for path in RESULTS_DIR.iterdir() if path.name.startswith("om_phase9_mfish_"))
    if not candidates:
        raise FileNotFoundError(f"no M-FISH result dirs under {RESULTS_DIR}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or latest_run()
    print(json.dumps(evaluate(run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
