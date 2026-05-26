#!/usr/bin/env python3
"""Check Stage 1 E5 Qwen3-8B scale-up packets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_stage1_e5_qwen3_scaleup.md"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_11.jsonl"

SCHEMA_VERSION = "om_stage1_e5_qwen3_scaleup_v1"
MODEL_ID = "RedHatAI/Qwen3-8B-quantized.w4a16"
TRACE_COUNT = 12
LEAVING_POSITIONS = (100, 20000)
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
UPDATE_CADENCE = 100
ALPHA = 0.3
TOP_FRACTION = 0.01
BUDGET_FRACTION = 0.10
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260526

PASS = "PASS_E5_QWEN3_SCALEUP"
AMBIGUOUS = "AMBIGUOUS_E5_QWEN3_SCALEUP"
KILL = "KILL_E5_QWEN3_SCALEUP"
SKIPPED_INFRA = "SKIPPED_INFRA_E5"
FAIL_INFRA = "FAIL_INFRA_E5"

REGIMES = ["bf16", "static_1pct", "m11b_top10"]
RECOVERY_REGIME = "m11b_top10"
THRESHOLDS = {
    "pass_leaving_rate_min": 0.40,
    "pass_median_recovery_gt": 0.30,
    "pass_ci_low_gt": 0.10,
    "kill_recovery_absent_median_le": 0.0,
}
REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "protected_sets.json",
    "quantization_config.json",
    "excluded_tensors.json",
    "per_trace_metrics.json",
    "metrics.json",
    "bootstrap_ci.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.glob("om_stage1_e5_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no E5 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def bootstrap_median(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(float(median(sample)))
    draws.sort()
    return {"ci95_low": draws[int(0.025 * (len(draws) - 1))], "ci95_high": draws[int(0.975 * (len(draws) - 1))]}


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def compute_leaving_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = {}
    for row in rows:
        prompt_index = int(row["prompt_index"])
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        by_trace_layer.setdefault(prompt_index, {}).setdefault(layer_index, {})[position] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    trace_values: list[dict[str, Any]] = []
    for prompt_index, by_layer in sorted(by_trace_layer.items()):
        layer_values: list[float] = []
        for layer_index, by_position in sorted(by_layer.items()):
            base = by_position[LEAVING_POSITIONS[0]]
            final = by_position[LEAVING_POSITIONS[1]]
            top_k = max(1, math.ceil(len(base) * TOP_FRACTION))
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            selected = [channel for channel, rank in enumerate(base_ranks) if rank < top_k]
            left = sum(1 for channel in selected if final_ranks[channel] >= top_k)
            layer_values.append(left / len(selected))
        trace_values.append({"prompt_index": prompt_index, "left_set_fraction": float(mean(layer_values)), "layer_count": len(layer_values)})
    values = [float(row["left_set_fraction"]) for row in trace_values]
    return {"aggregate": {"left_set_fraction": float(mean(values)), "trace_count": len(values)}, "trace_metrics": trace_values}


def summarize_recovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in rows if not bool(row.get("no_recoverable_static_gap"))]
    values = [float(row["recoveries"][RECOVERY_REGIME]) for row in included]
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": bootstrap_median(values),
        "included_trace_count": len(values),
        "total_trace_count": len(rows),
        "no_recoverable_static_gap_count": len(rows) - len(values),
    }


def classify(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    if metrics.get("status") == SKIPPED_INFRA:
        return SKIPPED_INFRA, [str(metrics.get("reason", "infrastructure skip"))]
    leaving = float(metrics["leaving_rate"]["aggregate"]["left_set_fraction"])
    recovery = metrics["results_by_regime"][RECOVERY_REGIME]
    median_value = recovery.get("median_recovery")
    ci_low = recovery.get("bootstrap_ci95", {}).get("ci95_low")
    ci_high = recovery.get("bootstrap_ci95", {}).get("ci95_high")
    if median_value is None or ci_low is None:
        return AMBIGUOUS, ["no positive static-gap traces for recovery"]
    median_f = float(median_value)
    ci_low_f = float(ci_low)
    if leaving >= THRESHOLDS["pass_leaving_rate_min"] and median_f > THRESHOLDS["pass_median_recovery_gt"] and ci_low_f > THRESHOLDS["pass_ci_low_gt"]:
        return PASS, ["leaving rate and M11b top-10 recovery both satisfy pass criteria"]
    if leaving < THRESHOLDS["pass_leaving_rate_min"] or median_f <= THRESHOLDS["kill_recovery_absent_median_le"] or (ci_high is not None and float(ci_high) <= THRESHOLDS["pass_ci_low_gt"]):
        return KILL, ["drift or M11b top-10 recovery is absent under preregistered thresholds"]
    return AMBIGUOUS, ["mixed drift/recovery signal or wide confidence interval"]


def validate_packet(run_dir: Path) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    if (run_dir / "infra_error.json").is_file():
        payload = load_json(run_dir / "infra_error.json")
        return [], {"metrics.json": {"status": payload.get("decision", SKIPPED_INFRA), "reason": payload.get("reason")}}, []
    infra = [f"missing required file: {rel}" for rel in REQUIRED_FILES if not (run_dir / rel).is_file()]
    loaded: dict[str, Any] = {}
    for rel in ["metrics.json", "per_trace_metrics.json", "model_provenance.json", "random_seed.json", "decoding_config.json", "quantization_config.json"]:
        if (run_dir / rel).is_file():
            try:
                loaded[rel] = load_json(run_dir / rel)
            except Exception as exc:
                infra.append(f"bad JSON {rel}: {exc!r}")
    metrics = loaded.get("metrics.json", {})
    rows = loaded.get("per_trace_metrics.json", {}).get("traces", [])
    if metrics.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
        infra.append("metrics schema_version mismatch")
    if metrics.get("model_id") != MODEL_ID:
        infra.append("metrics.model_id mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch")
    if len(rows) != TRACE_COUNT:
        infra.append(f"per_trace_metrics must contain exactly {TRACE_COUNT} traces")
    if loaded.get("model_provenance.json", {}).get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if loaded.get("random_seed.json", {}).get("seed") != BOOTSTRAP_SEED:
        infra.append("random seed mismatch")
    if loaded.get("decoding_config.json", {}).get("scoring_position") != SCORING_POSITION:
        infra.append("scoring position mismatch")
    if loaded.get("quantization_config.json", {}).get("weight_bits") != 4:
        infra.append("quantization_config.weight_bits mismatch")
    for row in rows if isinstance(rows, list) else []:
        perplexities = row.get("perplexities", {})
        recoveries = row.get("recoveries", {})
        if set(perplexities) != set(REGIMES):
            infra.append(f"trace {row.get('prompt_index')}: perplexity regimes mismatch")
            continue
        static_gap = float(perplexities["static_1pct"]) - float(perplexities["bf16"])
        if abs(float(row.get("static_gap", float("nan"))) - static_gap) > 1e-7:
            infra.append(f"trace {row.get('prompt_index')}: static_gap mismatch")
        if static_gap > 0 and RECOVERY_REGIME in recoveries:
            expected = 1.0 - (float(perplexities[RECOVERY_REGIME]) - float(perplexities["bf16"])) / static_gap
            if abs(float(recoveries[RECOVERY_REGIME]) - expected) > 1e-6:
                infra.append(f"trace {row.get('prompt_index')}: recovery mismatch")
    return infra, loaded, rows


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra, loaded, rows = validate_packet(run_dir)
    if infra:
        decision, reasons = FAIL_INFRA, infra
    else:
        metrics = loaded["metrics.json"]
        if rows:
            recomputed = summarize_recovery(rows)
            reported = metrics["results_by_regime"][RECOVERY_REGIME]["median_recovery"]
            if reported is not None and abs(float(reported) - float(recomputed["median_recovery"])) > 1e-9:
                decision, reasons = FAIL_INFRA, ["M11b recovery summary does not recompute"]
            else:
                decision, reasons = classify(metrics)
        else:
            decision, reasons = classify(metrics)
    result = {"schema_version": f"{SCHEMA_VERSION}_checker_result", "decision": decision, "artifact_complete": decision not in {FAIL_INFRA, SKIPPED_INFRA}, "reasons": reasons, "run_dir": str(run_dir), "thresholds": THRESHOLDS}
    write_json(run_dir / "checker_result.json", result)
    write_json(run_dir / "artifact_check.json", {k: result[k] for k in ["schema_version", "decision", "artifact_complete", "reasons", "run_dir"]})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.run_dir or latest_run_dir())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
