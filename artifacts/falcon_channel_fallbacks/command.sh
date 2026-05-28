#!/usr/bin/env bash
set -euo pipefail

# Guarded template. Defaults to HYST because LAMBDA is lower priority and
# should stay deferred unless the orchestrator explicitly selects it.
: "${ORCHESTRATOR_GATE:?Set ORCHESTRATOR_GATE=1 only after ParoQuant Falcon smoke is weak or the orchestrator gates channel fallbacks.}"

METHOD="${1:-hyst}"

cd /workspace/LatentWire
source .venv_gpu/bin/activate

case "$METHOD" in
  hyst)
    python experimental/outlier_migrate/phase9/run_om_phase9_funnel_smoke.py \
      --model-key falcon \
      --methods hyst \
      --hyst-exit-margin-pct-points 5 \
      --run-id "om_phase9_funnel_smoke_falcon_hyst_m5_$(date -u +%Y%m%dT%H%M%SZ)" \
      --batch-size 1 \
      --dtype bfloat16
    ;;
  lambda)
    python experimental/outlier_migrate/phase9/run_om_phase9_funnel_smoke.py \
      --model-key falcon \
      --methods lambda \
      --run-id "om_phase9_funnel_smoke_falcon_lambda_$(date -u +%Y%m%dT%H%M%SZ)" \
      --batch-size 1 \
      --dtype bfloat16
    ;;
  *)
    echo "usage: $0 [hyst|lambda]" >&2
    exit 2
    ;;
esac

