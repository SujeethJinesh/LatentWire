#!/usr/bin/env python3
"""Run V2 M11b DeepSeek baseline vetting."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase9 import check_om_v2_m11b_deepseek as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as base


def _patch_base() -> None:
    base.__doc__ = __doc__
    base.checker = checker
    base.DEFAULT_MODEL_ID = checker.MODEL_ID
    base.DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
    base.DEFAULT_RESULTS_DIR = checker.RESULTS_DIR
    base.SCHEMA_VERSION = checker.SCHEMA_VERSION
    base.UPDATE_POSITIONS = tuple(range(checker.UPDATE_CADENCE, checker.SCORING_POSITION + 1, checker.UPDATE_CADENCE))


def main(argv: list[str] | None = None) -> int:
    _patch_base()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
