#!/usr/bin/env python3
"""Check OutlierMigrate Phase 3 intervention result packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase3/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase3/preregister_om_phase3_intervention.md"

MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
SCHEMA_VERSION = "om_phase3_v1"
TRACE_COUNT = 24
PRIMARY_GRID = (100, 1000, 5000, 10000)
SPARSE_GRID = (100, 5000, 10000)
DENSE_GRID = (100, 500, 1000, 2000, 5000, 7500, 10000)
SCORING_POSITION = 10000
SCORING_WINDOW_TOKENS = 64
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260509
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"
EXPECTED_PROMPT_PAYLOAD_SHA256 = "sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e"
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"

PASS_DECISION = "PASS_OM_PHASE3_MIGRATION_AWARE_RECOVERS"
KILL_DECISION = "KILL_OM_PHASE3_INTERVENTION_FAILS"
AMBIGUOUS_DECISION = "KILL_OM_PHASE3_AMBIGUOUS"
FAIL_INFRA = "FAIL_INFRA_OM_PHASE3"

THRESHOLDS = {
    "pass_median_recovery_min": 0.50,
    "pass_ci95_low_gt": 0.30,
    "kill_median_recovery_lt": 0.20,
    "kill_ci95_high_lt": 0.30,
    "max_no_recoverable_static_gap_fraction": 0.25,
    "control_stop_outperforms_union_by_gt": 0.10,
}

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "activation_magnitude_manifest.json",
    "activation_magnitudes.jsonl.gz",
    "bf16_traces.jsonl.gz",
    "protected_sets.json",
    "quantization_config.json",
    "excluded_tensors.json",
    "per_trace_metrics.json",
    "metrics.json",
    "bootstrap_ci.json",
    "control_metrics.json",
    "grid_sensitivity_metrics.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]


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


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_close(left: float, right: float, *, tol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tol * max(1.0, abs(float(left)), abs(float(right)))


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no Phase 3 result dirs found under {RESULTS_DIR}")
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


def bootstrap_median(values: list[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(median(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def summarize_recovery(rows: list[dict[str, Any]], regime: str) -> dict[str, Any]:
    values = [float(row["recoveries"][regime]) for row in rows]
    return {
        "median_recovery": float(median(values)),
        "bootstrap_ci95": bootstrap_median(values),
        "trace_count": len(values),
        "per_trace_recovery": values,
    }


def decision_from_metrics(primary: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    med = float(primary["median_recovery"])
    ci_low = float(primary["bootstrap_ci95"]["ci95_low"])
    ci_high = float(primary["bootstrap_ci95"]["ci95_high"])
    no_gap_count = sum(1 for row in rows if bool(row.get("no_recoverable_static_gap")))
    no_gap_fraction = no_gap_count / len(rows)
    if med >= THRESHOLDS["pass_median_recovery_min"] and ci_low > THRESHOLDS["pass_ci95_low_gt"]:
        return PASS_DECISION, [
            f"median recovery {med:.8f} >= 0.50 and CI95 lower {ci_low:.8f} > 0.30"
        ]
    if med < THRESHOLDS["kill_median_recovery_lt"]:
        return KILL_DECISION, [f"median recovery {med:.8f} < 0.20"]
    if ci_high < THRESHOLDS["kill_ci95_high_lt"]:
        return KILL_DECISION, [f"CI95 upper {ci_high:.8f} < 0.30"]
    if no_gap_fraction > THRESHOLDS["max_no_recoverable_static_gap_fraction"]:
        return KILL_DECISION, [
            f"{no_gap_count}/{len(rows)} traces have no recoverable static gap (>25%)"
        ]
    non_improved = sum(
        1
        for row in rows
        if float(row["perplexities"]["union_primary"]) >= float(row["perplexities"]["static_1pct"])
    )
    if non_improved >= 18:
        return KILL_DECISION, [
            f"union does not reduce perplexity relative to static-1% on {non_improved}/24 traces"
        ]
    return AMBIGUOUS_DECISION, [
        f"median recovery {med:.8f}, CI95=[{ci_low:.8f}, {ci_high:.8f}] is neither PASS nor KILL"
    ]


def validate_prompt_manifest(prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra: list[str]) -> None:
    if file_sha256(DEFAULT_PROMPT_FILE) != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("canonical prompt file hash drifted")
    if prompt_manifest.get("source") != "AIME-2025":
        infra.append("prompt_manifest.source must be AIME-2025")
    if prompt_manifest.get("selection") != "deterministic_indices_0_23":
        infra.append("prompt_manifest.selection must be deterministic_indices_0_23")
    if prompt_manifest.get("prompt_file_sha256") != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("prompt file SHA mismatch")
    prompts = prompt_manifest.get("prompts", [])
    if not isinstance(prompts, list):
        infra.append("prompt_manifest.prompts must be a list")
        prompts = []
    if len(prompts) != TRACE_COUNT or prompt_manifest.get("prompt_count") != TRACE_COUNT:
        infra.append("prompt manifest must contain exactly 24 prompts")
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
        infra.append("prompt indices must be exactly 0-23 in order")
    if prompts:
        expected = prompt_payload_sha256(prompts)
        if prompt_manifest.get("prompt_sha256") != expected:
            infra.append("prompt payload SHA does not match prompt rows")
        if prompt_manifest.get("prompt_sha256") != EXPECTED_PROMPT_PAYLOAD_SHA256:
            infra.append("prompt payload SHA does not match frozen Phase 1 set")
    if metrics.get("prompt_sha256") != prompt_manifest.get("prompt_sha256"):
        infra.append("metrics.prompt_sha256 must match prompt_manifest")


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in HASHED_FILES:
        item = by_path.get(rel)
        path = run_dir / rel
        if item is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if not path.is_file():
            infra.append(f"hashed artifact missing on disk: {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


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
        "grid_sensitivity_metrics.json",
        "per_trace_metrics.json",
        "prompt_manifest.json",
        "model_provenance.json",
        "command_metadata.json",
        "random_seed.json",
        "decoding_config.json",
        "protected_sets.json",
        "quantization_config.json",
        "excluded_tensors.json",
        "activation_magnitude_manifest.json",
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
        infra.append("per_trace_metrics must contain exactly 24 traces")
    metrics = loaded.get("metrics.json", {})
    if metrics.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
        infra.append("metrics schema_version mismatch")
    if metrics.get("model_id") != MODEL_ID:
        infra.append(f"metrics.model_id must be {MODEL_ID}")
    if metrics.get("primary_grid") != list(PRIMARY_GRID):
        infra.append("metrics.primary_grid mismatch")
    if metrics.get("scoring_position") != SCORING_POSITION:
        infra.append("metrics.scoring_position mismatch")
    if metrics.get("scoring_window_tokens") != SCORING_WINDOW_TOKENS:
        infra.append("metrics.scoring_window_tokens mismatch")
    thresholds = metrics.get("thresholds", {})
    for key, expected in THRESHOLDS.items():
        if key not in thresholds or not is_close(float(thresholds[key]), expected):
            infra.append(f"metrics.thresholds.{key} mismatch")
    validate_prompt_manifest(loaded.get("prompt_manifest.json", {}), metrics, infra)
    if loaded.get("model_provenance.json", {}).get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if not loaded.get("model_provenance.json", {}).get("hf_snapshot_commit"):
        infra.append("model_provenance missing hf_snapshot_commit")
    if loaded.get("random_seed.json", {}).get("seed") != BOOTSTRAP_SEED:
        infra.append("random_seed.seed mismatch")
    qcfg = loaded.get("quantization_config.json", {})
    if qcfg.get("weight_bits") != 4 or qcfg.get("scheme") != "symmetric_per_output_channel_int4":
        infra.append("quantization_config must specify symmetric per-output-channel INT4")
    if qcfg.get("activation_dtype") != "float16":
        infra.append("quantization_config.activation_dtype must be float16")
    forbidden = qcfg.get("forbidden_methods", [])
    if "AWQ-style activation-aware scaling" not in forbidden:
        infra.append("quantization_config must forbid AWQ-style activation-aware scaling")
    protected = loaded.get("protected_sets.json", {})
    required_regimes = {
        "static_1pct",
        "union_primary",
        "static_2pct",
        "magnitude_average",
        "grid_sparse",
        "grid_dense",
    }
    if set(protected.get("regimes", {}).keys()) != required_regimes:
        infra.append("protected_sets.regimes must contain exactly the preregistered regimes")
    grid = loaded.get("grid_sensitivity_metrics.json", {})
    if grid.get("grids", {}).get("sparse", {}).get("positions") != list(SPARSE_GRID):
        infra.append("grid_sensitivity sparse grid mismatch")
    if grid.get("grids", {}).get("dense", {}).get("positions") != list(DENSE_GRID):
        infra.append("grid_sensitivity dense grid mismatch")
    controls = loaded.get("control_metrics.json", {})
    if set(controls.get("controls", {}).keys()) != {"static_2pct", "magnitude_average"}:
        infra.append("control_metrics must contain static_2pct and magnitude_average")
    if PREREG_PATH.is_file():
        prereg_sha = file_sha256(PREREG_PATH)
        if metrics.get("preregistration_sha256") != prereg_sha:
            infra.append("metrics.preregistration_sha256 does not match current preregistration")
    else:
        infra.append("missing Phase 3 preregistration file")
    if "artifact_hashes.json" in loaded:
        validate_artifact_hashes(run_dir, loaded["artifact_hashes.json"], infra)
    return infra, loaded, rows


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra, loaded, rows = validate_packet(run_dir)
    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "artifact_complete": False,
            "reasons": infra,
        }
    else:
        primary = summarize_recovery(rows, "union_primary")
        static_2 = summarize_recovery(rows, "static_2pct")
        average = summarize_recovery(rows, "magnitude_average")
        sparse = summarize_recovery(rows, "grid_sparse")
        dense = summarize_recovery(rows, "grid_dense")
        metrics = loaded["metrics.json"]
        for key in ["median_recovery", "bootstrap_ci95"]:
            if key == "median_recovery":
                if not is_close(metrics["primary_result"][key], primary[key]):
                    infra.append("metrics.primary_result.median_recovery mismatch")
            else:
                for ci_key in ["ci95_low", "ci95_high"]:
                    if not is_close(metrics["primary_result"][key][ci_key], primary[key][ci_key]):
                        infra.append(f"metrics.primary_result.bootstrap_ci95.{ci_key} mismatch")
        if infra:
            result = {
                "decision": FAIL_INFRA,
                "run_dir": str(run_dir),
                "artifact_complete": False,
                "reasons": infra,
            }
        else:
            decision, reasons = decision_from_metrics(primary, rows)
            union_med = float(primary["median_recovery"])
            control_stop = []
            for name, summary in [("static_2pct", static_2), ("magnitude_average", average)]:
                delta = float(summary["median_recovery"]) - union_med
                if delta > THRESHOLDS["control_stop_outperforms_union_by_gt"]:
                    control_stop.append(
                        {
                            "control": name,
                            "control_median_recovery": float(summary["median_recovery"]),
                            "union_median_recovery": union_med,
                            "delta": delta,
                        }
                    )
            result = {
                "decision": decision,
                "run_dir": str(run_dir),
                "artifact_complete": True,
                "reasons": reasons,
                "primary_result": primary,
                "control_results": {
                    "static_2pct": static_2,
                    "magnitude_average": average,
                    "union_outperforms_both_controls": (
                        union_med > float(static_2["median_recovery"])
                        and union_med > float(average["median_recovery"])
                    ),
                    "control_stop_condition": bool(control_stop),
                    "control_stop_details": control_stop,
                },
                "grid_sensitivity": {
                    "sparse": sparse,
                    "primary": primary,
                    "dense": dense,
                },
                "thresholds": THRESHOLDS,
            }
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": result["decision"],
        "run_dir": str(run_dir),
        "artifact_complete": result.get("artifact_complete", False),
        "reasons": result["reasons"],
        "control_stop_condition": result.get("control_results", {}).get("control_stop_condition", False),
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
    return 0 if result["decision"] == PASS_DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
