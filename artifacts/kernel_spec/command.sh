#!/usr/bin/env bash
set -euo pipefail

cd /workspace/LatentWire
if [ -f .venv_gpu/bin/activate ]; then
  source .venv_gpu/bin/activate
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

python artifacts/kernel_spec/pytorch_reference.py --self-test
