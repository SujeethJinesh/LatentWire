#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LatentWire
source .venv_gpu/bin/activate

python experimental/outlier_migrate/phase9/run_om_v1_paroquant_deepseek.py \
  --run-id "om_v1_paroquant_deepseek_$(date -u +%Y%m%dT%H%M%SZ)" \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z \
  --batch-size 1 \
  --dtype bfloat16
