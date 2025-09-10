#!/usr/bin/env bash
set -euo pipefail

# Load available modules
module load python3
module load cudatoolkit/12.5

# Create venv
python -m venv .venv
source .venv/bin/activate

# Upgrade pip first
python -m pip install --upgrade pip wheel

# Install CPU-safe packages on login node
python -m pip install numpy transformers datasets

echo "✅ Basic setup complete. Now get compute node for GPU packages"