#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
source .venv_gpu/bin/activate

BASE_RUN="experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z"

python experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py \
  --run-id om_driftrot_granite_clip_loose_20260528T1700Z \
  --reuse-trace-run-dir "$BASE_RUN" \
  --reuse-score-run-dir "$BASE_RUN" \
  --reuse-protected-run-dir "$BASE_RUN" \
  --m11b-reference-run-dir "$BASE_RUN" \
  --scale-clip-min 0.125 \
  --scale-clip-max 8.0 \
  --batch-size 1 \
  --dtype bfloat16

python experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py \
  --run-id om_driftrot_granite_clip_tight_20260528T1700Z \
  --reuse-trace-run-dir "$BASE_RUN" \
  --reuse-score-run-dir "$BASE_RUN" \
  --reuse-protected-run-dir "$BASE_RUN" \
  --m11b-reference-run-dir "$BASE_RUN" \
  --scale-clip-min 0.5 \
  --scale-clip-max 2.0 \
  --batch-size 1 \
  --dtype bfloat16
