#!/usr/bin/env python3
"""Run V1 ParoQuant-on-Nemotron baseline vetting."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase9 import check_om_v1_paroquant_nemotron as checker
from experimental.outlier_migrate.phase9 import run_om_paroquant_baseline as base


def _patch_base() -> None:
    base.__doc__ = __doc__
    base.checker = checker
    base.DEFAULT_MODEL_ID = checker.MODEL_ID
    base.DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
    base.DEFAULT_RESULTS_DIR = checker.RESULTS_DIR
    base.SCHEMA_VERSION = checker.SCHEMA_VERSION


def main(argv: list[str] | None = None) -> int:
    _patch_base()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
