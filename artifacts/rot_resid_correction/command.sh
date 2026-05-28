#!/usr/bin/env bash
set -euo pipefail

echo "C5 rotated residual correction has no runnable GPU command yet."
echo "Required first: emit DeltaW column norms and linear-input activation EMA during ParoQuant smoke."
python -m py_compile artifacts/rot_resid_correction/pytorch_reference.py
