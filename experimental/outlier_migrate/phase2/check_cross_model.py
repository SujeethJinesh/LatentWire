#!/usr/bin/env python3
"""Check OutlierMigrate Phase 2 partial cross-model result packets.

This checker is intentionally scoped to the May 8 authorized work window:
Nemotron-3-only validation is admissible evidence, but it is not the full
cross-model gate because Qwen3.6 and Kimi Linear remain deferred.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase2/results"

from experimental.outlier_migrate.phase1 import check_om_phase1 as phase1


SCHEMA_VERSION = "om_phase2_partial_nemotron3_v1"
MODEL_KEY = "nemotron3_nano"
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
POSITIONS = phase1.POSITIONS
TRACE_COUNT = phase1.TRACE_COUNT
THRESHOLDS = phase1.THRESHOLDS

PASS_PARTIAL_NEMOTRON3 = "PARTIAL_PASS_OM_PHASE2_NEMOTRON3_ONLY_QWEN36_KIMI_DEFERRED"
KILL_PARTIAL_NEMOTRON3 = "PARTIAL_KILL_OM_PHASE2_NEMOTRON3_ONLY_QWEN36_KIMI_DEFERRED"
FAIL_INFRA = "FAIL_INFRA_OM_PHASE2_NEMOTRON3_PARTIAL"
PREREG_AMBIGUOUS = "PREREG_AMBIGUOUS_OM_PHASE2_NEMOTRON3_CI_OVERLAP_QWEN36_KIMI_DEFERRED"

PHASE1_REFERENCE_RUN = "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z"
PHASE1_REFERENCE_DECISION = phase1.PASS_DYNAMIC
DEFERRED_MODELS = [
    "Qwen/Qwen3.6-35B-A3B",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
]

REQUIRED_FILES = [
    "environment.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "validation_scope.json",
    "phase1_reference.json",
    "activation_magnitude_manifest.json",
    "activation_magnitudes.jsonl.gz",
    "metrics.json",
    "bootstrap_ci.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]

load_json = phase1.load_json
write_json = phase1.write_json
file_sha256 = phase1.file_sha256
iter_activation_rows = phase1.iter_activation_rows
compute_metrics = phase1.compute_metrics
is_close = phase1.is_close


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no OutlierMigrate Phase 2 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


def validate_validation_scope(scope: dict[str, Any], infra: list[str]) -> None:
    if scope.get("queue_entry") != "cross_model_validation_outlier_migrate":
        infra.append("validation_scope.queue_entry mismatch")
    if scope.get("scope") != "partial_nemotron3_only_authorized_window":
        infra.append("validation_scope.scope must mark the authorized-window partial scope")
    if scope.get("model_key") != MODEL_KEY or scope.get("model_id") != MODEL_ID:
        infra.append("validation_scope model key/id mismatch")
    if scope.get("not_full_phase2_gate") is not True:
        infra.append("validation_scope.not_full_phase2_gate must be true")
    if scope.get("full_cross_validation_complete") is not False:
        infra.append("validation_scope.full_cross_validation_complete must be false")
    if scope.get("deferred_models") != DEFERRED_MODELS:
        infra.append("validation_scope.deferred_models mismatch")
    forbidden = scope.get("forbidden_during_window", [])
    if "download_qwen36_or_kimi_weights" not in forbidden:
        infra.append("validation_scope must record Qwen3.6/Kimi download prohibition")


def validate_phase1_reference(reference: dict[str, Any], infra: list[str]) -> None:
    if reference.get("phase1_run_dir") != PHASE1_REFERENCE_RUN:
        infra.append("phase1_reference.phase1_run_dir mismatch")
    if reference.get("phase1_decision") != PHASE1_REFERENCE_DECISION:
        infra.append("phase1_reference.phase1_decision mismatch")
    artifact_check = ROOT / PHASE1_REFERENCE_RUN / "artifact_check.json"
    metrics = ROOT / PHASE1_REFERENCE_RUN / "metrics.json"
    if not artifact_check.is_file():
        infra.append("referenced Phase 1 artifact_check.json missing")
    elif reference.get("phase1_artifact_check_sha256") != file_sha256(artifact_check):
        infra.append("phase1_reference artifact_check sha mismatch")
    if not metrics.is_file():
        infra.append("referenced Phase 1 metrics.json missing")
    elif reference.get("phase1_metrics_sha256") != file_sha256(metrics):
        infra.append("phase1_reference metrics sha mismatch")


def decision_from_metrics(point: float, ci_low: float, _ci_high: float) -> tuple[str, list[str]]:
    threshold = float(THRESHOLDS["migration_fraction_threshold"])
    if point >= threshold and ci_low > threshold:
        return PASS_PARTIAL_NEMOTRON3, [
            f"partial Nemotron-3 transfer: point={point:.8f} >= {threshold:.2f} "
            f"and ci_low={ci_low:.8f} > {threshold:.2f}; Qwen3.6/Kimi validation deferred"
        ]
    if point < threshold:
        return KILL_PARTIAL_NEMOTRON3, [
            f"partial Nemotron-3 no-transfer: point={point:.8f} < {threshold:.2f}; "
            "does not complete the full cross-model gate"
        ]
    return PREREG_AMBIGUOUS, [
        f"partial Nemotron-3 CI overlap: point={point:.8f} >= {threshold:.2f} "
        f"but ci_low={ci_low:.8f} <= {threshold:.2f}; human review required"
    ]


def infra_result(run_dir: Path, reasons: list[str]) -> dict[str, Any]:
    result = {
        "decision": FAIL_INFRA,
        "run_dir": str(run_dir),
        "reasons": reasons,
        "artifact_complete": False,
        "full_cross_validation_complete": False,
    }
    if run_dir.is_dir():
        write_json(run_dir / "checker_result.json", result)
        write_json(
            run_dir / "artifact_check.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_artifact_check",
                "decision": FAIL_INFRA,
                "run_dir": str(run_dir),
                "required_files": REQUIRED_FILES,
                "artifact_complete": False,
                "full_cross_validation_complete": False,
                "reasons": reasons,
            },
        )
    return result


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required artifact: {rel}")
    try:
        environment = load_json(run_dir / "environment.json")
        model = load_json(run_dir / "model_provenance.json")
        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        command = load_json(run_dir / "command_metadata.json")
        random_seed = load_json(run_dir / "random_seed.json")
        validation_scope = load_json(run_dir / "validation_scope.json")
        phase1_reference = load_json(run_dir / "phase1_reference.json")
        activation_manifest = load_json(run_dir / "activation_magnitude_manifest.json")
        metrics = load_json(run_dir / "metrics.json")
        bootstrap_ci = load_json(run_dir / "bootstrap_ci.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
    except Exception as exc:
        return infra_result(run_dir, [*infra, f"bad JSON artifacts: {exc!r}"])

    if environment.get("schema_version") != f"{SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if command.get("branch") != "outlier_migrate_phase2_partial_nemotron3":
        infra.append("command_metadata.branch must be outlier_migrate_phase2_partial_nemotron3")
    if model.get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if not model.get("hf_snapshot_commit") or not model.get("snapshot_path"):
        infra.append("model provenance must record local HF snapshot commit and path")
    if metrics.get("model_id") != MODEL_ID or metrics.get("model_key") != MODEL_KEY:
        infra.append("metrics model key/id mismatch")
    if metrics.get("validation_scope") != "partial_nemotron3_only_authorized_window":
        infra.append("metrics.validation_scope mismatch")
    if metrics.get("full_cross_validation_complete") is not False:
        infra.append("metrics.full_cross_validation_complete must be false")
    if metrics.get("phase1_decision") != PHASE1_REFERENCE_DECISION:
        infra.append("metrics.phase1_decision mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch")
    if int(metrics.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("metrics.bootstrap_samples must be 1000")
    if int(bootstrap_ci.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("bootstrap_ci.bootstrap_samples must be 1000")
    if int(random_seed.get("seed", -1)) != int(metrics.get("bootstrap_seed", -2)):
        infra.append("random_seed.seed must match metrics.bootstrap_seed")
    if not math.isclose(float(metrics.get("phase1_migration_fraction", -1.0)), 0.843165650406504):
        infra.append("metrics.phase1_migration_fraction must record the human-verified Phase 1 value")

    validate_validation_scope(validation_scope, infra)
    validate_phase1_reference(phase1_reference, infra)
    prompt_indices = phase1.validate_prompt_manifest(prompt_manifest, metrics, infra)
    validate_artifact_hashes(run_dir, artifact_hashes, infra)

    rows: list[dict[str, Any]] = []
    try:
        rows = list(iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz"))
    except Exception as exc:
        infra.append(f"cannot read activation_magnitudes.jsonl.gz: {exc!r}")
    if rows:
        prompt_ids_by_index = {
            int(row["index"]): str(row["prompt_id"])
            for row in prompt_manifest.get("prompts", [])
            if isinstance(row, dict) and "index" in row and "prompt_id" in row
        }
        phase1.validate_activation_rows(
            rows,
            prompt_indices=prompt_indices,
            prompt_ids_by_index=prompt_ids_by_index,
            activation_manifest=activation_manifest,
            metrics=metrics,
            infra=infra,
        )
        phase1.validate_activation_artifact_reference(run_dir, metrics, infra)

    computed: dict[str, Any] | None = None
    if rows and not infra:
        computed = compute_metrics(
            rows,
            bootstrap_samples=THRESHOLDS["bootstrap_samples"],
            seed=int(metrics["bootstrap_seed"]),
        )
        if not is_close(float(metrics["migration_fraction"]), computed["migration_fraction"]):
            infra.append("metrics.migration_fraction does not match recomputation")
        for key in ["ci95_low", "ci95_high"]:
            if not is_close(float(metrics["bootstrap_ci95"][key]), float(computed["bootstrap_ci95"][key])):
                infra.append(f"metrics.bootstrap_ci95.{key} does not match recomputation")
            if not is_close(float(bootstrap_ci["bootstrap_ci95"][key]), float(computed["bootstrap_ci95"][key])):
                infra.append(f"bootstrap_ci.bootstrap_ci95.{key} does not match recomputation")
        if not is_close(float(bootstrap_ci["migration_fraction"]), computed["migration_fraction"]):
            infra.append("bootstrap_ci.migration_fraction does not match recomputation")
        if metrics.get("top_channels_by_layer") != computed.get("top_channels_by_layer"):
            infra.append("metrics.top_channels_by_layer does not match recomputation")
        if metrics.get("top_channel_selection") != computed.get("top_channel_selection"):
            infra.append("metrics.top_channel_selection mismatch")

    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "reasons": infra,
            "artifact_complete": False,
            "full_cross_validation_complete": False,
        }
    else:
        assert computed is not None
        point = float(computed["migration_fraction"])
        ci_low = float(computed["bootstrap_ci95"]["ci95_low"])
        ci_high = float(computed["bootstrap_ci95"]["ci95_high"])
        decision, reasons = decision_from_metrics(point, ci_low, ci_high)
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "reasons": reasons,
            "artifact_complete": True,
            "full_cross_validation_complete": False,
            "validation_scope": "partial_nemotron3_only_authorized_window",
            "deferred_models": DEFERRED_MODELS,
            "migration_fraction": point,
            "bootstrap_ci95": computed["bootstrap_ci95"],
            "trace_metrics": computed["trace_metrics"],
            "layer_metrics": computed["layer_metrics"],
            "thresholds": THRESHOLDS,
        }
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": result["decision"],
        "run_dir": str(run_dir),
        "required_files": REQUIRED_FILES,
        "artifact_complete": result.get("artifact_complete", False),
        "full_cross_validation_complete": False,
        "validation_scope": "partial_nemotron3_only_authorized_window",
        "deferred_models": DEFERRED_MODELS,
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
    return 0 if result["decision"] == PASS_PARTIAL_NEMOTRON3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
