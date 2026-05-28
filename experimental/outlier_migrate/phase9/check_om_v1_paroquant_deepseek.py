#!/usr/bin/env python3
"""Check V1 ParoQuant-on-DeepSeek rotation-smoke packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase9 import check_om_paroquant_baseline as base

RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_v1_paroquant_deepseek_falcon.md"
DEFAULT_PROMPT_FILE = base.DEFAULT_PROMPT_FILE

SCHEMA_VERSION = "om_v1_paroquant_deepseek_v1"
TRACE_COUNT = 12
SCORING_POSITION = base.SCORING_POSITION
SCORING_WINDOW_TOKENS = base.SCORING_WINDOW_TOKENS
BOOTSTRAP_SAMPLES = base.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = 20260624
EXPECTED_PROMPT_FILE_SHA256 = base.EXPECTED_PROMPT_FILE_SHA256
EXPECTED_PROMPT_SOURCE_DATASET = base.EXPECTED_PROMPT_SOURCE_DATASET
EXPECTED_PROMPT_SOURCE_COMMIT = base.EXPECTED_PROMPT_SOURCE_COMMIT

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_SNAPSHOT = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
STATIC_TOP10_REFERENCE_MEDIAN = 0.376752614594403

PASS_ROTATION_DOMINATES = "PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES"
KILL_WEAK = "KILL_V1_PAROQUANT_DEEPSEEK_WEAK"
AMBIGUOUS = "AMBIGUOUS_V1_PAROQUANT_DEEPSEEK"
FAIL_INFRA = "FAIL_INFRA_V1_PAROQUANT_DEEPSEEK"

REGIMES = base.REGIMES
RECOVERY_REGIMES = base.RECOVERY_REGIMES
INTERPRETATION_BANDS = base.INTERPRETATION_BANDS
REQUIRED_FILES = base.REQUIRED_FILES
OPTIONAL_FILES = base.OPTIONAL_FILES
HASHED_FILES = base.HASHED_FILES


def _patch_base() -> None:
    base.RESULTS_DIR = RESULTS_DIR
    base.PREREG_PATH = PREREG_PATH
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.TRACE_COUNT = TRACE_COUNT
    base.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    base.MODEL_ID = MODEL_ID
    base.MODEL_SNAPSHOT = MODEL_SNAPSHOT
    base.PASS_DECISION = PASS_ROTATION_DOMINATES
    base.FAIL_INFRA = FAIL_INFRA


def decision_from_median(median_recovery: float | None) -> tuple[str, list[str]]:
    if median_recovery is None:
        return FAIL_INFRA, ["no traces had a positive recoverable static top-1% gap"]
    median_value = float(median_recovery)
    if median_value > STATIC_TOP10_REFERENCE_MEDIAN:
        return PASS_ROTATION_DOMINATES, ["ParoQuant beats the DeepSeek static-top10 reference median"]
    if median_value < 0.30:
        return KILL_WEAK, ["ParoQuant median recovery is below 0.30 on DeepSeek"]
    return AMBIGUOUS, ["ParoQuant lies below static-top10 but above the weak baseline threshold"]


def evaluate(run_dir: Path) -> dict:
    _patch_base()
    run_dir = run_dir.resolve()
    infra, metrics, rows = base.validate_packet(run_dir)
    artifact_complete = not infra
    median_recovery = metrics.get("results_by_regime", {}).get("paroquant_w4a16", {}).get("median_recovery")
    if artifact_complete:
        decision, reasons = decision_from_median(median_recovery)
    else:
        decision, reasons = FAIL_INFRA, infra
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": artifact_complete,
        "reasons": reasons,
        "infra_reasons": infra,
        "run_dir": str(run_dir),
        "median_recovery": median_recovery,
        "ci95": metrics.get("results_by_regime", {}).get("paroquant_w4a16", {}).get("bootstrap_ci95"),
        "static_top10_reference_median": STATIC_TOP10_REFERENCE_MEDIAN,
        "paroquant_minus_static_top10_median": (
            None if median_recovery is None else float(median_recovery) - STATIC_TOP10_REFERENCE_MEDIAN
        ),
        "included_trace_count": metrics.get("included_trace_count"),
        "total_trace_count": len(rows),
        "implementation_mode": metrics.get("implementation_mode"),
        "interpretation_band": base.interpretation_band(median_recovery),
    }
    base.write_json(run_dir / "checker_result.json", result)
    base.write_json(
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


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_v1_paroquant_deepseek_")]
    if not candidates:
        raise FileNotFoundError(f"no V1 ParoQuant DeepSeek result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or latest_run_dir()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != FAIL_INFRA else 1


_patch_base()


if __name__ == "__main__":
    raise SystemExit(main())
