#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_phase9_m11_ema_drift.py --run-id om_phase9_m11_granite_small_vac12_20260516T010728Z --batch-size 1 --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z
