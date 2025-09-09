#!/usr/bin/env bash
set -euo pipefail

PYBIN="$(command -v python3.11 || command -v python3)"
echo "Using Python interpreter: ${PYBIN}"
"${PYBIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

# Optional: enable MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "✅ macOS setup complete. Python: $(python -V). Activate with: source .venv/bin/activate"
