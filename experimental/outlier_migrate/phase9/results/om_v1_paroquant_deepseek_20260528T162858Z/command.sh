#!/usr/bin/env bash
set -euo pipefail
run_om_paroquant_baseline.py --run-id om_v1_paroquant_deepseek_20260528T162858Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z --batch-size 1 --dtype bfloat16
