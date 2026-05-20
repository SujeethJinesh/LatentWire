#!/usr/bin/env python3
"""Check Phase 9 M11b Nemotron replication packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase9 import check_om_phase9_m11b_budget_scaling as base

RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_phase9_m11b_nemotron_replication.md"
DEFAULT_PROMPT_FILE = base.DEFAULT_PROMPT_FILE

SCHEMA_VERSION = "om_phase9_m11b_nemotron_v1"
TRACE_COUNT = 12
SCORING_POSITION = base.SCORING_POSITION
SCORING_WINDOW_TOKENS = base.SCORING_WINDOW_TOKENS
UPDATE_CADENCE = base.UPDATE_CADENCE
BOOTSTRAP_SAMPLES = base.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = 20260604
EXPECTED_PROMPT_FILE_SHA256 = base.EXPECTED_PROMPT_FILE_SHA256
EXPECTED_PROMPT_SOURCE_DATASET = base.EXPECTED_PROMPT_SOURCE_DATASET
EXPECTED_PROMPT_SOURCE_COMMIT = base.EXPECTED_PROMPT_SOURCE_COMMIT

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_SNAPSHOT = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
ALPHA = base.ALPHA
BUDGET_FRACTIONS = base.BUDGET_FRACTIONS

PASS_DECISION = "PASS_M11B_NEMOTRON_REPLICATES"
KILL_BUDGET_INSUFFICIENT = "KILL_M11B_NEMOTRON_BUDGET_INSUFFICIENT"
AMBIGUOUS = "AMBIGUOUS_M11B_NEMOTRON"
FAIL_INFRA = "FAIL_INFRA_M11B_NEMOTRON"

THRESHOLDS = {
    "pass_high_budget_median_recovery_min": 0.30,
    "pass_beats_static_top10_by_ge": 0.15,
    "kill_high_budget_within_top1_abs_le": 0.05,
    "alpha": ALPHA,
    "bootstrap_seed": BOOTSTRAP_SEED,
}

REGIMES = base.REGIMES
RECOVERY_REGIMES = base.RECOVERY_REGIMES
HIGH_BUDGET_REGIMES = base.HIGH_BUDGET_REGIMES
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
    base.PASS_DECISION = PASS_DECISION
    base.KILL_BUDGET_INSUFFICIENT = KILL_BUDGET_INSUFFICIENT
    base.AMBIGUOUS = AMBIGUOUS
    base.FAIL_INFRA = FAIL_INFRA
    base.THRESHOLDS = THRESHOLDS


def expected_source_file(index: int) -> str:
    return base.expected_source_file(index)


def expected_prompt_id(index: int) -> str:
    return base.expected_prompt_id(index)


def prompt_payload_sha256(prompts: list[dict]) -> str:
    return base.prompt_payload_sha256(prompts)


def bootstrap_median(values: list[float]) -> dict[str, float | None]:
    _patch_base()
    return base.bootstrap_median(values)


def evaluate(run_dir: Path) -> dict:
    _patch_base()
    return base.evaluate(run_dir)


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_phase9_m11b_nemotron_")]
    if not candidates:
        raise FileNotFoundError(f"no M11b Nemotron result dirs found under {RESULTS_DIR}")
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
