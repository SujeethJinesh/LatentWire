#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

# Optional: enable MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "✅ macOS setup complete. Activate with: source .venv/bin/activate"
