#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
source .venv_gpu/bin/activate
python experimental/outlier_migrate/phase9/run_om_v1_paroquant_nemotron.py \
  --run-id om_v1_paroquant_nemotron_20260528T0318Z \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --batch-size 1 \
  --dtype bfloat16


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


#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LatentWire
source .venv_gpu/bin/activate

python experimental/outlier_migrate/phase9/run_om_phase9_msurface_sanity.py \
  --run-id "om_phase9_msurface_granite_sanity_$(date -u +%Y%m%dT%H%M%SZ)" \
  --prompt-indices 0,1 \
  --positions 100,20000 \
  --dtype bfloat16
