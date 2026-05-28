#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LatentWire
source .venv_gpu/bin/activate

python experimental/outlier_migrate/phase9/run_om_phase9_funnel_smoke.py \
  --model-key falcon \
  --methods hyst \
  --hyst-exit-margin-pct-points 5 \
  --run-id "om_phase9_funnel_smoke_falcon_hyst_m5_$(date -u +%Y%m%dT%H%M%SZ)" \
  --batch-size 1 \
  --dtype bfloat16
