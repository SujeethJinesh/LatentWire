#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_phase9_m10_position_binned_scales.py --run-id om_phase9_m10_granite_small_vac12_20260515T085800Z --trace-count 12 --batch-size 1 --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z
