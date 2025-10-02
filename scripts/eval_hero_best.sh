#!/usr/bin/env bash
set -euo pipefail

# eval_hero_best.sh
# Evaluate the best checkpoint from hero_resume training
#
# Usage:
#   bash scripts/eval_hero_best.sh

RUN_TAG="${RUN_TAG:-hero_resume}"
EVAL_TAG="${EVAL_TAG:-eval_best_$(date +%Y%m%d_%H%M%S)}"

# Checkpoint to evaluate
CKPT="${CKPT:-runs/hero_resume/ckpt_stageb_best}"

# Model configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"

# Evaluation settings
SAMPLES="${SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-88}"

# Latent configuration (must match training)
LATENT_LEN="${LATENT_LEN:-64}"

# Device configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1,2,3}"
DEFAULT_LLAMA_DEVICE_MAP='{"model.embed_tokens":0,"model.rotary_emb":0,"model.layers.0":0,"model.layers.1":0,"model.layers.2":0,"model.layers.3":0,"model.layers.4":0,"model.layers.5":0,"model.layers.6":0,"model.layers.7":0,"model.layers.8":1,"model.layers.9":1,"model.layers.10":1,"model.layers.11":1,"model.layers.12":1,"model.layers.13":1,"model.layers.14":1,"model.layers.15":1,"model.layers.16":2,"model.layers.17":2,"model.layers.18":2,"model.layers.19":2,"model.layers.20":2,"model.layers.21":2,"model.layers.22":2,"model.layers.23":2,"model.layers.24":3,"model.layers.25":3,"model.layers.26":3,"model.layers.27":3,"model.layers.28":3,"model.layers.29":3,"model.layers.30":3,"model.layers.31":3,"model.norm":3,"lm_head":3}'
LLAMA_DEVICE_MAP="${LLAMA_DEVICE_MAP:-$DEFAULT_LLAMA_DEVICE_MAP}"
GPU_MEM_GIB="${GPU_MEM_GIB:-70}"

# Output directory
EVAL_DIR="runs/${RUN_TAG}/${EVAL_TAG}"
mkdir -p "$EVAL_DIR"
LOG="${EVAL_DIR}/eval.log"

echo "=== Evaluating Hero Best Checkpoint ===" | tee "$LOG"
echo "Checkpoint: $CKPT" | tee -a "$LOG"
echo "Dataset: $DATASET" | tee -a "$LOG"
echo "Samples: $SAMPLES" | tee -a "$LOG"
echo "Output: $EVAL_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# CUDA preflight
echo "=== CUDA preflight ===" | tee -a "$LOG"
python3 -c "import os, torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES'))" 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Run evaluation
echo "=== Running Evaluation ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py \
  --models llama \
  --ckpt "$CKPT" \
  --llama_id "$LLAMA_ID" \
  --llama_device_map "$LLAMA_DEVICE_MAP" \
  --llama_devices "$LLAMA_DEVICES" \
  --gpu_mem_gib "$GPU_MEM_GIB" \
  --dataset "$DATASET" \
  --samples "$SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --chunk_size "$CHUNK_SIZE" \
  --fresh_eval \
  --use_chat_template yes \
  --latent_anchor_mode chat \
  --append_bos_after_prefix yes \
  --first_token_top_p 1.0 \
  --first_token_temperature 0.0 \
  --prefix_gain 1.1 \
  --token_budget_mode content_only \
  --token_budget_k "$LATENT_LEN" \
  --out_dir "$EVAL_DIR" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "✓ Evaluation complete. Logs: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Results saved to: $EVAL_DIR" | tee -a "$LOG"
