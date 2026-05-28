#!/usr/bin/env bash
set -euo pipefail
run_om_paroquant_baseline.py --run-id om_v1_paroquant_falcon_20260528T154653Z --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z --batch-size 1 --dtype bfloat16
