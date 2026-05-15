#!/usr/bin/env python3
"""Check Phase 9 M11 EMA-smoothed drift protection packets."""

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
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_m11_ema_drift.md"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"

SCHEMA_VERSION = "om_phase9_m11_v1"
TRACE_COUNT = 12
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 512
UPDATE_CADENCE = 100
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260525
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"

MODEL_ID = "ibm-granite/granite-4.0-h-small"
MODEL_SNAPSHOT = "b8c0982bab7fde4eb48110f5a069527c008fab39"
ALPHAS = (0.1, 0.3, 0.5)

PASS_DECISION = "PASS_M11_EMA_DRIFT"
KILL_NO_IMPROVEMENT = "KILL_M11_NO_IMPROVEMENT"
KILL_RANDOM_CONTROL = "KILL_M11_RANDOM_CONTROL_BEATS"
KILL_AMBIGUOUS = "KILL_M11_AMBIGUOUS"
FAIL_INFRA = "FAIL_INFRA_M11"

THRESHOLDS = {
    "pass_median_recovery_min": 0.30,
    "pass_ci95_low_gt": 0.10,
    "pass_beats_static_1pct_by_ge": 0.15,
    "pass_beats_random_walk_by_ge": 0.20,
    "kill_within_static_1pct_abs_le": 0.05,
    "random_control_beats_by_gt": 0.10,
    "ema_threshold": 0.5,
    "absolute_cap_fraction": 0.03,
    "bootstrap_seed": BOOTSTRAP_SEED,
}

REGIMES = [
    "bf16",
    "static_1pct",
    "m11_alpha_0_1",
    "m11_alpha_0_3",
    "m11_alpha_0_5",
    "random_walk_protection",
]

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
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


def is_close(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(left)), abs(float(right)))


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_phase9_m11_")]
    if not candidates:
        raise FileNotFoundError(f"no M11 result dirs found under {RESULTS_DIR}")
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


def regime_for_alpha(alpha: float) -> str:
    return "m11_alpha_" + str(alpha).replace(".", "_")


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


def decision_from_summaries(
    alpha_summaries: dict[str, dict[str, Any]], random_walk: dict[str, Any]
) -> tuple[str, list[str], str | None]:
    if not alpha_summaries:
        return FAIL_INFRA, ["no M11 alpha summaries were available"], None
    random_med = (
        float(random_walk["median_recovery"]) if random_walk.get("median_recovery") is not None else float("nan")
    )
    best_alpha = max(
        alpha_summaries,
        key=lambda key: float("-inf")
        if alpha_summaries[key].get("median_recovery") is None
        else float(alpha_summaries[key]["median_recovery"]),
    )
    best = alpha_summaries[best_alpha]
    if best.get("median_recovery") is None:
        return FAIL_INFRA, ["no traces had a positive recoverable static-1% gap"], None
    for alpha, summary in alpha_summaries.items():
        if summary.get("median_recovery") is None:
            continue
        med = float(summary["median_recovery"])
        low = float(summary["bootstrap_ci95"]["ci95_low"])
        if (
            med >= THRESHOLDS["pass_median_recovery_min"]
            and low > THRESHOLDS["pass_ci95_low_gt"]
            and med >= THRESHOLDS["pass_beats_static_1pct_by_ge"]
            and med - random_med >= THRESHOLDS["pass_beats_random_walk_by_ge"]
        ):
            return PASS_DECISION, [f"{alpha} satisfies all M11 pass criteria"], alpha
    if all(
        summary.get("median_recovery") is not None
        and abs(float(summary["median_recovery"])) <= THRESHOLDS["kill_within_static_1pct_abs_le"]
        for summary in alpha_summaries.values()
    ):
        return KILL_NO_IMPROVEMENT, ["all M11 alphas are within 0.05 median recovery of static-1%"], best_alpha
    if all(
        summary.get("median_recovery") is not None
        and random_med - float(summary["median_recovery"]) > THRESHOLDS["random_control_beats_by_gt"]
        for summary in alpha_summaries.values()
    ):
        return KILL_RANDOM_CONTROL, ["random-walk control beats every M11 alpha by >0.10 median recovery"], best_alpha
    return KILL_AMBIGUOUS, ["M11 is neither pass nor a specific no-improvement/random-control kill"], best_alpha


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
        expected_recoveries = {regime_for_alpha(alpha) for alpha in ALPHAS} | {"random_walk_protection"}
        if set(recoveries) != expected_recoveries:
            infra.append(f"trace {prompt_index}: recovery regimes mismatch")
            continue
        static_gap = float(perplexities["static_1pct"]) - float(perplexities["bf16"])
        if not is_close(float(row.get("static_gap", float("nan"))), static_gap):
            infra.append(f"trace {prompt_index}: static_gap mismatch")
        no_gap = static_gap <= 0.0
        if bool(row.get("no_recoverable_static_gap")) != no_gap:
            infra.append(f"trace {prompt_index}: no_recoverable_static_gap mismatch")
        for regime in expected_recoveries:
            if no_gap:
                if recoveries[regime] is not None:
                    infra.append(f"trace {prompt_index}: no-gap recovery for {regime} must be null")
                continue
            expected = 1.0 - (float(perplexities[regime]) - float(perplexities["bf16"])) / static_gap
            if not is_close(float(recoveries[regime]), expected, tol=1e-7):
                infra.append(f"trace {prompt_index}: recovery formula mismatch for {regime}")


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
        "protected_trajectories.json",
        "quantization_config.json",
        "excluded_tensors.json",
        "artifact_hashes.json",
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
    if len(rows) != TRACE_COUNT:
        infra.append(f"per_trace_metrics must contain exactly {TRACE_COUNT} traces")
    validate_trace_rows(rows, infra)
    metrics = loaded.get("metrics.json", {})
    if metrics.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
        infra.append("metrics schema_version mismatch")
    if metrics.get("model_id") != MODEL_ID:
        infra.append("metrics.model_id mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch")
    if metrics.get("alphas") != list(ALPHAS):
        infra.append("metrics.alphas mismatch")
    if metrics.get("scoring_position") != SCORING_POSITION:
        infra.append("metrics.scoring_position mismatch")
    if metrics.get("scoring_window_tokens") != SCORING_WINDOW_TOKENS:
        infra.append("metrics.scoring_window_tokens mismatch")
    if metrics.get("update_cadence") != UPDATE_CADENCE:
        infra.append("metrics.update_cadence mismatch")
    validate_prompt_manifest(loaded.get("prompt_manifest.json", {}), metrics, infra)
    model = loaded.get("model_provenance.json", {})
    if model.get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if model.get("hf_snapshot_commit") != MODEL_SNAPSHOT:
        infra.append("model snapshot commit mismatch")
    if loaded.get("random_seed.json", {}).get("seed") != BOOTSTRAP_SEED:
        infra.append("random_seed.seed mismatch")
    decoding = loaded.get("decoding_config.json", {})
    if decoding.get("do_sample") is not False or decoding.get("num_beams") != 1:
        infra.append("decoding_config must be deterministic greedy")
    trajectories = loaded.get("protected_trajectories.json", {})
    if trajectories.get("schema_version") != f"{SCHEMA_VERSION}_protected_trajectories":
        infra.append("protected_trajectories schema mismatch")
    if trajectories.get("alphas") != list(ALPHAS):
        infra.append("protected_trajectories alpha mismatch")
    qcfg = loaded.get("quantization_config.json", {})
    if qcfg.get("weight_bits") != 4 or qcfg.get("scheme") != "symmetric_per_output_channel_int4":
        infra.append("quantization_config must specify symmetric per-output-channel INT4")
    if qcfg.get("activation_dtype") != "float16":
        infra.append("quantization_config.activation_dtype must be float16")
    excluded = loaded.get("excluded_tensors.json", {}).get("by_regime", {})
    if set(excluded) != set(REGIMES) - {"bf16"}:
        infra.append("excluded_tensors.by_regime mismatch")
    if PREREG_PATH.is_file() and metrics.get("preregistration_sha256") != file_sha256(PREREG_PATH):
        infra.append("metrics.preregistration_sha256 mismatch")
    if "artifact_hashes.json" in loaded:
        validate_artifact_hashes(run_dir, loaded["artifact_hashes.json"], infra)
    return infra, loaded, rows


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra, _loaded, rows = validate_packet(run_dir)
    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "artifact_complete": False,
            "reasons": infra,
        }
    else:
        alpha_summaries = {regime_for_alpha(alpha): summarize_recovery(rows, regime_for_alpha(alpha)) for alpha in ALPHAS}
        random_walk = summarize_recovery(rows, "random_walk_protection")
        decision, reasons, best_alpha = decision_from_summaries(alpha_summaries, random_walk)
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "artifact_complete": decision != FAIL_INFRA,
            "reasons": reasons,
            "best_alpha": best_alpha,
            "primary_results_by_alpha": alpha_summaries,
            "control_results": {
                "static_1pct": {"median_recovery": 0.0},
                "random_walk_protection": random_walk,
            },
            "thresholds": THRESHOLDS,
        }
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": result["decision"],
        "run_dir": str(run_dir),
        "artifact_complete": result.get("artifact_complete", False),
        "reasons": result["reasons"],
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(run_dir / "artifact_check.json", artifact_check)
    return result


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
