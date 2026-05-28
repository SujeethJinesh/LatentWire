#!/usr/bin/env bash
set -euo pipefail
experimental/outlier_migrate/phase9/run_om_phase9_m18_joint_kv_activation.py --run-id om_phase9_m18_granite_small_vac12_20260516T193500Z --reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z --batch-size 1
