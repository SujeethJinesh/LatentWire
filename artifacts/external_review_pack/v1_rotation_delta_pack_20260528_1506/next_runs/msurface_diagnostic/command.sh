#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LatentWire
source .venv_gpu/bin/activate

python experimental/outlier_migrate/phase9/run_om_phase9_msurface_sanity.py \
  --run-id "om_phase9_msurface_granite_sanity_$(date -u +%Y%m%dT%H%M%SZ)" \
  --prompt-indices 0,1 \
  --positions 100,20000 \
  --dtype bfloat16
