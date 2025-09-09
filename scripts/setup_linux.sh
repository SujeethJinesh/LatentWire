#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

echo "✅ Linux setup complete. Activate with: source .venv/bin/activate"
