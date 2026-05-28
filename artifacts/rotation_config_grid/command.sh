#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
source .venv_gpu/bin/activate

cat <<'EOF'
Guarded template only; do not run as C3.

Granite:
python experimental/outlier_migrate/phase9/run_om_paroquant_baseline.py \
  --run-id om_driftrot_grid_granite_<CONFIG_ID>_$(date -u +%Y%m%dT%H%M%SZ) \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z \
  --group-size <GROUP_SIZE> --num-rotations <NUM_ROTATIONS> \
  --scale-clip-min <CLIP_MIN> --scale-clip-max <CLIP_MAX> --batch-size 1

Nemotron:
python experimental/outlier_migrate/phase9/run_om_v1_paroquant_nemotron.py \
  --run-id om_driftrot_grid_nemotron_<CONFIG_ID>_$(date -u +%Y%m%dT%H%M%SZ) \
  --reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --reuse-score-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --reuse-protected-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --m11b-reference-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z \
  --group-size <GROUP_SIZE> --num-rotations <NUM_ROTATIONS> \
  --scale-clip-min <CLIP_MIN> --scale-clip-max <CLIP_MAX> --batch-size 1 --dtype bfloat16
EOF

