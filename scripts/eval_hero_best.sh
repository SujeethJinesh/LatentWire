#!/usr/bin/env bash
set -euo pipefail

# eval_hero_best.sh
# Evaluate the peak checkpoint from hero_resume training
#
# This script evaluates the _best checkpoint (saved when first_acc peaks ≥10%)
# which contains properly saved LoRA/Prefix weights after the 2025-10-02 bug fix.
#
# Usage:
#   bash scripts/eval_hero_best.sh

RUN_TAG="${RUN_TAG:-hero_resume}"
SAMPLES="${SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-88}"
DATASET="${DATASET:-squad}"

# Peak checkpoint path
CKPT_DIR="runs/${RUN_TAG}/ckpt_stageb_best"

# Evaluation log directory
EVAL_DIR="runs/${RUN_TAG}/eval_best_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"
LOG="${EVAL_DIR}/eval.log"

# Model configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LATENT_LEN="${LATENT_LEN:-64}"

# Environment setup
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== Evaluating Peak Checkpoint ===" | tee "$LOG"
echo "Checkpoint: $CKPT_DIR" | tee -a "$LOG"
echo "Samples: $SAMPLES" | tee -a "$LOG"
echo "Output: $EVAL_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Check if checkpoint exists
if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: Checkpoint directory not found: $CKPT_DIR" | tee -a "$LOG"
  echo "Make sure training has completed and created a peak checkpoint." | tee -a "$LOG"
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py \
  --models llama \
  --ckpt "$CKPT_DIR" \
  --out_dir "$EVAL_DIR" \
  --llama_id "$LLAMA_ID" \
  --samples "$SAMPLES" \
  --dataset "$DATASET" \
  --fresh_eval \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --chunk_size "$CHUNK_SIZE" \
  --latent_anchor_mode chat \
  --append_bos_after_prefix yes \
  --use_chat_template yes \
  --first_token_top_p 1.0 \
  --first_token_temperature 0.0 \
  --prefix_gain 1.1 \
  --token_budget_mode content_only \
  --token_budget_k "$LATENT_LEN" \
  2>&1 | tee -a "$LOG"

echo -e "\n✓ Evaluation complete.\n" | tee -a "$LOG"
if [ -f "${EVAL_DIR}/metrics.json" ]; then
  echo "Results saved to: $EVAL_DIR" | tee -a "$LOG"
  echo "Key files:" | tee -a "$LOG"
  echo "  - Metrics: ${EVAL_DIR}/metrics.json" | tee -a "$LOG"
  echo "  - Predictions: ${EVAL_DIR}/predictions.jsonl" | tee -a "$LOG"
  echo "  - Log: ${LOG}" | tee -a "$LOG"
else
  echo "[WARN] Metrics files not created - eval may have failed" | tee -a "$LOG"
  echo "Check log: ${LOG}" | tee -a "$LOG"
fi
