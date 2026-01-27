#!/usr/bin/env bash
set -euo pipefail
export PATH=/workspace/conda/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/tmp
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
cd /workspace/LatentWire
PY=/workspace/conda/envs/rosetta/bin/python
proportions=(0.75 0.50 0.25 0.10)
orders=(front back)
for p in "${proportions[@]}"; do
  for o in "${orders[@]}"; do
    tag="step3_int8_p${p}_${o}_$(date +%Y%m%d_%H%M%S)"
    echo "RUN_TAG=${tag}"
    "$PY" quantization/scripts/run_step1_kv_ptq.py \
      --mode gpu \
      --kv-quant-scheme int8 \
      --kv-cache-proportion "$p" \
      --kv-cache-order-mode "$o" \
      --run-tag "$tag"
  done
done
