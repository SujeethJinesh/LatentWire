#!/usr/bin/env python3
"""Check Phase 9 ParoQuant baseline packets."""

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
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_paroquant_baseline.md"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"

SCHEMA_VERSION = "om_paroquant_baseline_v1"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260601
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"

MODEL_ID = "ibm-granite/granite-4.0-h-small"
MODEL_SNAPSHOT = "b8c0982bab7fde4eb48110f5a069527c008fab39"

PASS_DECISION = "PASS_PAROQUANT_BASELINE_REPORTED"
FAIL_INFRA = "FAIL_INFRA_PAROQUANT_BASELINE"

REGIMES = ["bf16", "static_1pct", "paroquant_w4a16"]
RECOVERY_REGIMES = ["paroquant_w4a16"]
INTERPRETATION_BANDS = {
    "substantial_recovery_gt": 0.50,
    "meaningful_incomplete_low": 0.30,
    "meaningful_incomplete_high": 0.50,
    "low_recovery_lt": 0.30,
}

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "quantization_config.json",
    "paroquant_config.json",
    "paroquant_limitations.json",
    "source_artifacts.json",
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
    "bf16_traces.jsonl.gz",
    "bf16_trace_manifest.json",
    "activation_magnitudes.jsonl.gz",
    "activation_magnitude_manifest.json",
    "protected_sets.json",
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
        for chunk in iter(lambda: handle.read(1024 * 1024, ), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_close(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(left)), abs(float(right)))


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_paroquant_")]
    if not candidates:
        raise FileNotFoundError(f"no ParoQuant result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


def validate_prompt_manifest(prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra: list[str]) -> None:
    if file_sha256(DEFAULT_PROMPT_FILE) != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("canonical prompt file hash drifted")
    if prompt_manifest.get("selection") != "deterministic_indices_0_11_vacation_revision":
        infra.append("prompt_manifest.selection must be deterministic_indices_0_11_vacation_revision")
    if prompt_manifest.get("prompt_file_sha256") != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("prompt file SHA mismatch")
    prompts = prompt_manifest.get("prompts", [])
    if not isinstance(prompts, list):
        infra.append("prompt_manifest.prompts must be a list")
        prompts = []
    if len(prompts) != TRACE_COUNT or prompt_manifest.get("prompt_count") != TRACE_COUNT:
        infra.append(f"prompt manifest must contain exactly {TRACE_COUNT} prompts")
    indices = []
    for row in prompts:
        if not isinstance(row, dict):
            infra.append("prompt row is not an object")
            continue
        index = int(row.get("index", -1))
        indices.append(index)
        if row.get("prompt_id") != expected_prompt_id(index):
            infra.append(f"prompt {index}: prompt_id mismatch")
        if row.get("source_dataset") != EXPECTED_PROMPT_SOURCE_DATASET:
            infra.append(f"prompt {index}: source_dataset mismatch")
        if row.get("source_commit") != EXPECTED_PROMPT_SOURCE_COMMIT:
            infra.append(f"prompt {index}: source_commit mismatch")
        if row.get("source_file") != expected_source_file(index):
            infra.append(f"prompt {index}: source_file mismatch")
    if indices != list(range(TRACE_COUNT)):
        infra.append("prompt indices must be exactly 0-11 in order")
    if prompts and prompt_manifest.get("prompt_sha256") != prompt_payload_sha256(prompts):
        infra.append("prompt payload SHA does not match prompt rows")
    if metrics.get("prompt_sha256") != prompt_manifest.get("prompt_sha256"):
        infra.append("metrics.prompt_sha256 must match prompt_manifest")


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


def validate_trace_rows(rows: list[dict[str, Any]], infra: list[str]) -> None:
    expected_score_start = SCORING_POSITION - SCORING_WINDOW_TOKENS + 1
    for row in rows:
        prompt_index = int(row.get("prompt_index", -1))
        if int(row.get("scored_tokens", -1)) != SCORING_WINDOW_TOKENS:
            infra.append(f"trace {prompt_index}: scored_tokens mismatch")
        if int(row.get("score_start", -1)) != expected_score_start:
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
            infra.append(f"trace {prompt_index}: no_recoverable_static_gap mismatch")
        for regime in RECOVERY_REGIMES:
            if no_gap:
                if recoveries[regime] is not None:
                    infra.append(f"trace {prompt_index}: no-gap recovery for {regime} must be null")
                continue
            expected = 1.0 - (float(perplexities[regime]) - float(perplexities["bf16"])) / static_gap
            if not is_close(float(recoveries[regime]), expected, tol=1e-7):
                infra.append(f"trace {prompt_index}: recovery formula mismatch for {regime}")


def interpretation_band(median_recovery: float | None) -> str:
    if median_recovery is None:
        return "no_positive_static_gap"
    if median_recovery > INTERPRETATION_BANDS["substantial_recovery_gt"]:
        return "substantial_recovery_gt_0_50"
    if median_recovery >= INTERPRETATION_BANDS["meaningful_incomplete_low"]:
        return "meaningful_but_incomplete_0_30_to_0_50"
    return "low_recovery_lt_0_30"


def validate_packet(run_dir: Path) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
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
        "prompt_manifest.json",
        "model_provenance.json",
        "command_metadata.json",
        "random_seed.json",
        "decoding_config.json",
        "quantization_config.json",
        "paroquant_config.json",
        "paroquant_limitations.json",
        "source_artifacts.json",
        "excluded_tensors.json",
        "artifact_hashes.json",
    ]:
        path = run_dir / rel
        if path.is_file():
            try:
                loaded[rel] = load_json(path)
            except Exception as exc:
                infra.append(f"cannot parse {rel}: {exc}")

    metrics = loaded.get("metrics.json", {})
    rows = loaded.get("per_trace_metrics.json", {}).get("traces", [])
    if not isinstance(rows, list):
        rows = []
        infra.append("per_trace_metrics.traces must be a list")

    validate_prompt_manifest(loaded.get("prompt_manifest.json", {}), metrics, infra)
    validate_trace_rows(rows, infra)
    validate_artifact_hashes(run_dir, loaded.get("artifact_hashes.json", {}), infra)

    if loaded.get("model_provenance.json", {}).get("hf_snapshot_commit") != MODEL_SNAPSHOT:
        infra.append("model snapshot commit mismatch")
    if loaded.get("model_provenance.json", {}).get("model_id") != MODEL_ID:
        infra.append("model id mismatch")
    if loaded.get("random_seed.json", {}).get("seed") != BOOTSTRAP_SEED:
        infra.append("bootstrap seed mismatch")
    if loaded.get("decoding_config.json", {}).get("scoring_position") != SCORING_POSITION:
        infra.append("scoring position mismatch")
    if loaded.get("decoding_config.json", {}).get("scoring_window_tokens") != SCORING_WINDOW_TOKENS:
        infra.append("scoring window mismatch")
    if metrics.get("trace_count") != TRACE_COUNT:
        infra.append("metrics.trace_count mismatch")
    if metrics.get("model_snapshot_commit") != MODEL_SNAPSHOT:
        infra.append("metrics.model_snapshot_commit mismatch")
    if metrics.get("implementation_mode") not in {
        "upstream_paroquant",
        "local_algorithmic_reproduction",
        "algorithmic_reproduction_not_full_upstream",
    }:
        infra.append("metrics.implementation_mode is invalid")

    summaries = metrics.get("results_by_regime", {})
    expected_summary = summarize_recovery(rows, "paroquant_w4a16") if rows else {}
    actual_summary = summaries.get("paroquant_w4a16", {})
    for key in ["median_recovery", "mean_recovery"]:
        if expected_summary.get(key) is not None and not is_close(expected_summary[key], actual_summary.get(key)):
            infra.append(f"metrics results_by_regime.paroquant_w4a16.{key} mismatch")
    if actual_summary.get("bootstrap_ci95") != expected_summary.get("bootstrap_ci95"):
        infra.append("bootstrap CI mismatch for paroquant_w4a16")
    if loaded.get("bootstrap_ci.json", {}).get("results_by_regime", {}).get("paroquant_w4a16") != actual_summary:
        infra.append("bootstrap_ci.json does not mirror metrics ParoQuant summary")

    return infra, metrics, rows


def evaluate(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    infra, metrics, rows = validate_packet(run_dir)
    artifact_complete = not infra
    median_recovery = (
        metrics.get("results_by_regime", {}).get("paroquant_w4a16", {}).get("median_recovery")
        if isinstance(metrics.get("results_by_regime"), dict)
        else None
    )
    decision = PASS_DECISION if artifact_complete else FAIL_INFRA
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": artifact_complete,
        "infra_reasons": infra,
        "run_dir": str(run_dir),
        "interpretation_band": interpretation_band(median_recovery),
        "median_recovery": median_recovery,
        "ci95": metrics.get("results_by_regime", {}).get("paroquant_w4a16", {}).get("bootstrap_ci95"),
        "included_trace_count": metrics.get("included_trace_count"),
        "total_trace_count": len(rows),
        "implementation_mode": metrics.get("implementation_mode"),
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(
        run_dir / "artifact_check.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_artifact_check",
            "artifact_complete": artifact_complete,
            "decision": decision,
            "checked_files": REQUIRED_FILES,
            "infra_reasons": infra,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or latest_run_dir()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == PASS_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
