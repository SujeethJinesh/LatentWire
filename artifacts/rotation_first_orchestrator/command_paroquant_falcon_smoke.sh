#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LatentWire
source .venv_gpu/bin/activate

python experimental/outlier_migrate/phase9/run_om_v1_paroquant_falcon.py \
  --run-id "om_v1_paroquant_falcon_$(date -u +%Y%m%dT%H%M%SZ)" \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z \
  --batch-size 1 \
  --dtype bfloat16
