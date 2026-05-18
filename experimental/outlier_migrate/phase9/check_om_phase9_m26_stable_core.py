#!/usr/bin/env python3
"""Check Phase 9 M26 stable-core protection packets."""

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
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_m26_stable_core.md"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"

SCHEMA_VERSION = "om_phase9_m26_v1"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260530
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"

MODEL_ID = "ibm-granite/granite-4.0-h-small"
MODEL_SNAPSHOT = "b8c0982bab7fde4eb48110f5a069527c008fab39"

PASS_DECISION = "PASS_M26_STABLE_CORE"
KILL_NO_IMPROVEMENT = "KILL_M26_NO_IMPROVEMENT"
KILL_RANDOM_CONTROL_BEATS = "KILL_M26_RANDOM_CONTROL_BEATS"
AMBIGUOUS = "AMBIGUOUS_M26"
FAIL_INFRA = "FAIL_INFRA_M26"

REGIMES = [
    "bf16",
    "static_1pct",
    "m26_core",
    "core_current_top1_union",
    "random_matched_core",
]
RECOVERY_REGIMES = [
    "m26_core",
    "core_current_top1_union",
    "random_matched_core",
]

THRESHOLDS = {
    "pass_median_recovery_min": 0.30,
    "pass_ci95_low_gt": 0.10,
    "pass_beats_static_1pct_by_ge": 0.15,
    "pass_beats_random_by_ge": 0.20,
    "kill_no_improvement_within_static_abs_le": 0.05,
    "kill_random_beats_by_gt": 0.10,
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
    "stable_core.json",
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
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"] + OPTIONAL_FILES


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


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def expected_source_file(index: int) -> str:
    return "aime2025-I.jsonl" if index < 15 else "aime2025-II.jsonl"


def expected_prompt_id(index: int) -> str:
    if index < 15:
        return f"opencompass_AIME2025_I_{index}"
    return f"opencompass_AIME2025_II_{index - 15}"


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    payload = "".join(str(row["prompt"]) for row in ordered).encode("utf-8")
    return bytes_sha256(payload)


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


def decision_from_summaries(summaries: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    core = summaries.get("m26_core", {})
    random_control = summaries.get("random_matched_core", {})
    if core.get("median_recovery") is None:
        return FAIL_INFRA, ["no traces had a positive recoverable static top-1% gap"]

    core_med = float(core["median_recovery"])
    core_low = core.get("bootstrap_ci95", {}).get("ci95_low")
    random_med = (
        float(random_control["median_recovery"])
        if random_control.get("median_recovery") is not None
        else float("nan")
    )

    if random_control.get("median_recovery") is not None and random_med - core_med > THRESHOLDS["kill_random_beats_by_gt"]:
        return KILL_RANDOM_CONTROL_BEATS, ["random matched-size control beats M26 by more than 0.10 median recovery"]

    if abs(core_med - 0.0) <= THRESHOLDS["kill_no_improvement_within_static_abs_le"]:
        return KILL_NO_IMPROVEMENT, ["M26 core-only is within 0.05 median recovery of static-1% baseline"]

    if (
        core_med >= THRESHOLDS["pass_median_recovery_min"]
        and core_low is not None
        and float(core_low) > THRESHOLDS["pass_ci95_low_gt"]
        and core_med - 0.0 >= THRESHOLDS["pass_beats_static_1pct_by_ge"]
        and random_control.get("median_recovery") is not None
        and core_med - random_med >= THRESHOLDS["pass_beats_random_by_ge"]
    ):
        return PASS_DECISION, ["M26 core-only satisfies recovery, CI, static, and random-control thresholds"]

    return AMBIGUOUS, ["M26 is neither pass nor a specific no-improvement/random-control kill"]


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_phase9_m26_")]
    if not candidates:
        raise FileNotFoundError(f"no M26 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in HASHED_FILES:
        path = run_dir / rel
        if not path.is_file():
            continue
        item = by_path.get(rel)
        if item is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    missing = [rel for rel in REQUIRED_FILES if not (run_dir / rel).is_file()]
    if missing:
        infra.append(f"missing required files: {missing}")

    per_trace = {"traces": []}
    metrics: dict[str, Any] = {}
    artifact_hashes: dict[str, Any] = {}
    if not missing:
        per_trace = load_json(run_dir / "per_trace_metrics.json")
        metrics = load_json(run_dir / "metrics.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
        validate_artifact_hashes(run_dir, artifact_hashes, infra)

    rows = per_trace.get("traces", []) if isinstance(per_trace, dict) else []
    if len(rows) != TRACE_COUNT:
        infra.append(f"per_trace_metrics must contain {TRACE_COUNT} traces")
    for row in rows:
        for regime in REGIMES:
            if regime not in row.get("perplexities", {}):
                infra.append(f"trace {row.get('prompt_index')}: missing perplexity for {regime}")
        for regime in RECOVERY_REGIMES:
            if regime not in row.get("recoveries", {}):
                infra.append(f"trace {row.get('prompt_index')}: missing recovery for {regime}")
        if int(row.get("scored_tokens", -1)) != SCORING_WINDOW_TOKENS:
            infra.append(f"trace {row.get('prompt_index')}: scored_tokens mismatch")

    summaries = {regime: summarize_recovery(rows, regime) for regime in RECOVERY_REGIMES}
    decision, reasons = decision_from_summaries(summaries) if not infra else (FAIL_INFRA, infra)

    result = {
        "decision": decision,
        "reasons": reasons,
        "artifact_complete": not infra,
        "run_dir": str(run_dir),
        "results_by_regime": summaries,
        "thresholds": THRESHOLDS,
    }
    write_json(run_dir / "checker_result.json", result)
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": decision,
        "artifact_complete": not infra,
        "reasons": reasons,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "artifact_check.json", artifact_check)
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
