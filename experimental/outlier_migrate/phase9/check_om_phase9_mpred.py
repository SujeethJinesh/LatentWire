#!/usr/bin/env python3
"""Check Phase 9 M-PRED predictive-channel packets."""

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
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_mpred.md"

SCHEMA_VERSION = "om_phase9_mpred_v1"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
UPDATE_CADENCE = 100
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260527

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

REGIMES = [
    "bf16",
    "static_1pct",
    "m11b_top5",
    "m11b_top10",
    "static_top10",
    "mpred_top5_alpha_0_5",
    "mpred_top5_alpha_0_8",
    "mpred_top5_alpha_0_95",
    "mpred_top5_alpha_0_99",
    "mpred_top10_alpha_0_95",
    "random_walk_top5",
    "mpred_random_alpha_top5",
]
RECOVERY_REGIMES = [regime for regime in REGIMES if regime not in {"bf16", "static_1pct"}]
MPRED_REGIMES = [
    "mpred_top5_alpha_0_5",
    "mpred_top5_alpha_0_8",
    "mpred_top5_alpha_0_95",
    "mpred_top5_alpha_0_99",
    "mpred_top10_alpha_0_95",
]

PASS = "PASS_MPRED_BEATS_M11B"
PASS_TIGHTENS_GRANITE = "PASS_TIGHTENS_GRANITE_MPRED"
AMBIGUOUS = "AMBIGUOUS_MPRED"
KILL = "KILL_MPRED"
FAIL_INFRA = "FAIL_INFRA_MPRED"

THRESHOLDS = {
    "pass_mpred_minus_m11b_ge": 0.05,
    "tighten_granite_ci_width_reduction_ge": 0.25,
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
OPTIONAL_FILES = [
    "activation_magnitudes.jsonl.gz",
    "activation_magnitude_manifest.json",
    "bf16_traces.jsonl.gz",
    "bf16_trace_manifest.json",
]


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


def is_close(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(left)), abs(float(right)))


def bootstrap_median(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(BOOTSTRAP_SEED)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(median(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def summarize_recovery(rows: list[dict[str, Any]], regime: str) -> dict[str, Any]:
    included = [row for row in rows if not bool(row.get("no_recoverable_static_gap"))]
    values = [float(row["recoveries"][regime]) for row in included]
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": bootstrap_median(values),
        "included_trace_count": len(values),
        "total_trace_count": len(rows),
        "per_trace_recovery_included": values,
        "no_recoverable_static_gap_count": len(rows) - len(values),
        "no_recoverable_static_gap_fraction": (len(rows) - len(values)) / len(rows) if rows else 0.0,
    }


def best_regime(summaries: dict[str, dict[str, Any]], regimes: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    candidates = [(regime, summaries[regime]) for regime in regimes if summaries[regime].get("median_recovery") is not None]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: float(item[1]["median_recovery"]))


def ci_width(summary: dict[str, Any]) -> float | None:
    ci = summary.get("bootstrap_ci95", {})
    if ci.get("ci95_low") is None or ci.get("ci95_high") is None:
        return None
    return float(ci["ci95_high"]) - float(ci["ci95_low"])


def ci_non_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ci = left.get("bootstrap_ci95", {})
    right_ci = right.get("bootstrap_ci95", {})
    if left_ci.get("ci95_low") is None or right_ci.get("ci95_high") is None:
        return False
    return float(left_ci["ci95_low"]) > float(right_ci["ci95_high"])


def decision_from_summaries(
    summaries: dict[str, dict[str, Any]], model_key: str
) -> tuple[str, list[str], dict[str, Any]]:
    best_mpred_name, best_mpred = best_regime(summaries, MPRED_REGIMES)
    best_m11b_name, best_m11b = best_regime(summaries, ["m11b_top5", "m11b_top10"])
    random_alpha = summaries.get("mpred_random_alpha_top5", {})
    if best_mpred is None or best_m11b is None:
        return FAIL_INFRA, ["missing M-PRED or M11b recovery summaries"], {}

    mpred_med = float(best_mpred["median_recovery"])
    m11b_med = float(best_m11b["median_recovery"])
    random_alpha_med = random_alpha.get("median_recovery")
    details = {
        "best_mpred_regime": best_mpred_name,
        "best_mpred_median": mpred_med,
        "best_m11b_regime": best_m11b_name,
        "best_m11b_median": m11b_med,
        "best_mpred_minus_best_m11b": mpred_med - m11b_med,
        "random_alpha_median": random_alpha_med,
    }

    if mpred_med - m11b_med >= THRESHOLDS["pass_mpred_minus_m11b_ge"] and ci_non_overlap(best_mpred, best_m11b):
        return PASS, [f"{best_mpred_name} beats {best_m11b_name} by >=0.05 with non-overlapping CI"], details

    if model_key == "granite":
        mpred_width = ci_width(best_mpred)
        m11b_top5_width = ci_width(summaries["m11b_top5"])
        if (
            mpred_width is not None
            and m11b_top5_width is not None
            and mpred_med >= 0.0
            and mpred_width <= (1.0 - THRESHOLDS["tighten_granite_ci_width_reduction_ge"]) * m11b_top5_width
        ):
            details["best_mpred_ci_width"] = mpred_width
            details["m11b_top5_ci_width"] = m11b_top5_width
            return PASS_TIGHTENS_GRANITE, ["M-PRED tightens Granite CI by at least 25% with non-negative median"], details

    if random_alpha_med is not None and mpred_med <= float(random_alpha_med):
        return KILL, [f"{best_mpred_name} does not beat random-alpha control"], details
    if mpred_med < m11b_med:
        return KILL, [f"{best_mpred_name} loses to {best_m11b_name}"], details
    return AMBIGUOUS, ["M-PRED direction is positive but does not satisfy PASS criteria"], details


def validate_rows(rows: list[dict[str, Any]], infra: list[str]) -> None:
    expected_start = SCORING_POSITION - SCORING_WINDOW_TOKENS + 1
    if len(rows) != TRACE_COUNT:
        infra.append(f"per_trace_metrics must contain exactly {TRACE_COUNT} traces")
    for row in rows:
        prompt_index = int(row.get("prompt_index", -1))
        if int(row.get("scored_tokens", -1)) != SCORING_WINDOW_TOKENS:
            infra.append(f"trace {prompt_index}: scored token mismatch")
        if int(row.get("score_start", -1)) != expected_start:
            infra.append(f"trace {prompt_index}: score_start mismatch")
        if int(row.get("score_end", -1)) != SCORING_POSITION:
            infra.append(f"trace {prompt_index}: score_end mismatch")
        perplexities = row.get("perplexities", {})
        recoveries = row.get("recoveries", {})
        if set(perplexities) != set(REGIMES):
            infra.append(f"trace {prompt_index}: perplexity regimes mismatch")
            continue
        if set(recoveries) != set(RECOVERY_REGIMES):
            infra.append(f"trace {prompt_index}: recovery regimes mismatch")
            continue
        static_gap = float(perplexities["static_1pct"]) - float(perplexities["bf16"])
        if not is_close(float(row.get("static_gap", float("nan"))), static_gap):
            infra.append(f"trace {prompt_index}: static_gap mismatch")
        no_gap = static_gap <= 0.0
        if bool(row.get("no_recoverable_static_gap")) != no_gap:
            infra.append(f"trace {prompt_index}: no-gap flag mismatch")
        for regime in RECOVERY_REGIMES:
            if no_gap:
                if recoveries.get(regime) is not None:
                    infra.append(f"trace {prompt_index}: no-gap recovery for {regime} must be null")
                continue
            expected = 1.0 - (float(perplexities[regime]) - float(perplexities["bf16"])) / static_gap
            if not is_close(float(recoveries[regime]), expected, tol=1e-7):
                infra.append(f"trace {prompt_index}: recovery formula mismatch for {regime}")


def infer_model_key(model_id: str) -> str | None:
    for key, spec in MODEL_SPECS.items():
        if spec["model_id"] == model_id:
            return key
    return None


def validate_packet(run_dir: Path) -> tuple[list[str], dict[str, Any], list[dict[str, Any]], str | None]:
    infra: list[str] = []
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required file: {rel}")
    loaded: dict[str, Any] = {}
    for rel in [
        "metrics.json",
        "bootstrap_ci.json",
        "control_metrics.json",
        "per_trace_metrics.json",
        "model_provenance.json",
        "random_seed.json",
        "protected_sets.json",
        "excluded_tensors.json",
    ]:
        path = run_dir / rel
        if path.is_file():
            try:
                loaded[rel] = load_json(path)
            except Exception as exc:
                infra.append(f"bad JSON {rel}: {exc!r}")
    rows = loaded.get("per_trace_metrics.json", {}).get("traces", [])
    if not isinstance(rows, list):
        infra.append("per_trace_metrics.traces must be a list")
        rows = []
    validate_rows(rows, infra)

    model = loaded.get("model_provenance.json", {})
    model_key = infer_model_key(str(model.get("model_id")))
    if model_key is None:
        infra.append("model_provenance.model_id is not a preregistered M-PRED model")
    elif model.get("hf_snapshot_commit") != MODEL_SPECS[model_key]["snapshot"]:
        infra.append("model snapshot commit mismatch")
    if loaded.get("random_seed.json", {}).get("seed") != BOOTSTRAP_SEED:
        infra.append("random_seed.seed mismatch")
    metrics = loaded.get("metrics.json", {})
    if metrics.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
        infra.append("metrics schema_version mismatch")
    if PREREG_PATH.is_file() and metrics.get("preregistration_sha256") != file_sha256(PREREG_PATH):
        infra.append("metrics.preregistration_sha256 mismatch")
    protected = loaded.get("protected_sets.json", {}).get("regimes", {})
    excluded = loaded.get("excluded_tensors.json", {}).get("by_regime", {})
    if set(protected) != set(REGIMES) - {"bf16"}:
        infra.append("protected_sets.regimes mismatch")
    if set(excluded) != set(REGIMES) - {"bf16"}:
        infra.append("excluded_tensors.by_regime mismatch")
    return infra, loaded, rows, model_key


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra, _loaded, rows, model_key = validate_packet(run_dir)
    if infra or model_key is None:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "artifact_complete": False,
            "reasons": infra,
        }
    else:
        summaries = {regime: summarize_recovery(rows, regime) for regime in RECOVERY_REGIMES}
        decision, reasons, details = decision_from_summaries(summaries, model_key)
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "artifact_complete": decision != FAIL_INFRA,
            "model_key": model_key,
            "reasons": reasons,
            "decision_details": details,
            "results_by_regime": summaries,
            "thresholds": THRESHOLDS,
        }
    write_json(
        run_dir / "artifact_check.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_artifact_check",
            "decision": result["decision"],
            "run_dir": str(run_dir),
            "artifact_complete": result.get("artifact_complete", False),
            "reasons": result["reasons"],
        },
    )
    write_json(run_dir / "checker_result.json", result)
    return result


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_phase9_mpred_")]
    if not candidates:
        raise FileNotFoundError(f"no M-PRED result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
