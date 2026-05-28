#!/usr/bin/env bash
set -euo pipefail

# Guarded template only. This runs a GPU diagnostic if and only if the
# orchestrator explicitly gates it.
: "${ORCHESTRATOR_GATE:?Set ORCHESTRATOR_GATE=1 only after the orchestrator gates M-SURFACE diagnostic work.}"

cd /workspace/LatentWire
source .venv_gpu/bin/activate

python experimental/outlier_migrate/phase9/run_om_phase9_msurface_sanity.py \
  --run-id "om_phase9_msurface_driftrot_granite_sanity_$(date -u +%Y%m%dT%H%M%SZ)" \
  --prompt-indices 0,1 \
  --positions 100,20000 \
  --dtype bfloat16

