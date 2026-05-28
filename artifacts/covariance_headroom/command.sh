#!/usr/bin/env bash
set -euo pipefail
cd /workspace/LatentWire
source .venv_gpu/bin/activate
python - <<'PY'
print("C1 used cached activation magnitudes and activation_means.npz only.")
print("See artifacts/covariance_headroom/metrics.json for the computed proxy summary.")
print("No GPU job was run and no covariance cache was found.")
PY

