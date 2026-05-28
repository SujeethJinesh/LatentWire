#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_driftrot_residual_subset.py --run-id om_driftrot_residual_granite_tail4_diag_20260528T2052Z --candidate-id residual_tight_top8x32_tail4 --prompt-indices 4 --split-name diagnostic --scale-clip-min 0.5 --scale-clip-max 2.0
