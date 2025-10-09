#!/usr/bin/env bash
set -euo pipefail

# milestone0_baseline.sh
# Milestone 0: Baseline Verification (LoRA-only)
#
# Purpose: Reproduce a 2-epoch smoke with only the encoder and LoRA trained.
# Confirms encoder gradients flow and documents baseline metrics (EM/F1, first-token top-k, latency, compression).
#
# This script uses the --baseline_verification flag which:
#   - Disables deep prefix
#   - Disables latent adapters
#   - Sets all KD weights to 0
#   - Disables gist head
#   - Keeps encoder trainable
#   - Keeps LoRA enabled (small adapters to help frozen LLM listen to latent)
#
# Usage:
#   bash scripts/milestone0_baseline.sh

RUN_TAG="${RUN_TAG:-milestone0_baseline_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/${RUN_TAG}"
CKPT_DIR="${RUN_DIR}/ckpt"
LOG="${RUN_DIR}/baseline.log"
DIAGNOSTIC_LOG="${RUN_DIR}/diagnostics.jsonl"

mkdir -p "$CKPT_DIR"
: > "$DIAGNOSTIC_LOG"

# Model configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"

# Baseline smoke test: 2 epochs, minimal samples
EPOCHS=2
TRAIN_SAMPLES="${TRAIN_SAMPLES:-200}"  # ~8 steps per epoch with batch size 24
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# Latent configuration
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# LoRA configuration (small adapters on early layers)
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_FIRSTN="${LORA_FIRSTN:-12}"

# Environment setup
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Mac-specific MPS fallback
if [[ "$(uname)" == "Darwin" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Common arguments
COMMON_ARGS=(
  --models llama,qwen
  --llama_id "$LLAMA_ID"
  --qwen_id "$QWEN_ID"
  --dataset "$DATASET"
  --use_chat_template
  --encoder_type byte
  --encoder_use_chat_template
  --latent_len "$LATENT_LEN"
  --d_z "$D_Z"
  --sequential_models
)

# LoRA arguments (keep enabled for baseline)
if [[ $USE_LORA -eq 1 ]]; then
  LORA_ARGS=(
    --use_lora
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_firstN "$LORA_FIRSTN"
    --lora_target_modules "attn_firstN:${LORA_FIRSTN}"
  )
else
  LORA_ARGS=()
fi

echo "================================================================================"
echo "MILESTONE 0: Baseline Verification (LoRA-only)"
echo "================================================================================"
echo "RUN_TAG: $RUN_TAG"
echo "Epochs: $EPOCHS"
echo "Training samples: $TRAIN_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Latent length: $LATENT_LEN"
echo "Latent dim: $D_Z"
echo "LoRA enabled: $USE_LORA (r=$LORA_R, alpha=$LORA_ALPHA, firstN=$LORA_FIRSTN)"
echo "================================================================================"
echo "" | tee -a "$LOG"

# CUDA preflight check
echo "=== CUDA preflight ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import os, torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("PYTORCH_CUDA_ALLOC_CONF:", os.getenv("PYTORCH_CUDA_ALLOC_CONF"))
PY

# Training with baseline verification mode
echo -e "\n=== Baseline Training (2 epochs) ===\n" | tee -a "$LOG"
python -u latentwire/train.py \
  "${COMMON_ARGS[@]}" \
  --samples "$TRAIN_SAMPLES" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --save_dir "$CKPT_DIR" \
  --auto_resume \
  --save_training_stats \
  --baseline_verification \
  --train_append_bos_after_prefix yes \
  --warm_anchor_mode chat \
  --first_token_ce_weight 0.5 \
  --first_token_ce_schedule none \
  --K 4 \
  --k_ce_weight 0.5 \
  --adapter_hidden_mult 2 \
  --adapter_dropout 0.0 \
  --max_answer_tokens 24 \
  --lr 1e-4 \
  --max_grad_norm 1.0 \
  --grad_diag_interval 5 \
  --grad_diag_components "encoder,adapter,tf,first,kce" \
  --diagnostic_log "$DIAGNOSTIC_LOG" \
  "${LORA_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

# Evaluation
echo -e "\n=== Baseline Evaluation ===\n" | tee -a "$LOG"
python -u latentwire/eval.py \
  --models llama,qwen \
  --ckpt "$CKPT_DIR" \
  --llama_id "$LLAMA_ID" \
  --qwen_id "$QWEN_ID" \
  --samples "$EVAL_SAMPLES" \
  --dataset "$DATASET" \
  --fresh_eval \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --latent_anchor_mode chat \
  --append_bos_after_prefix yes \
  --use_chat_template yes \
  --first_token_top_p 1.0 \
  --first_token_temperature 0.0 \
  --sequential_eval \
  --token_budget_mode content_only \
  --token_budget_k "$LATENT_LEN" \
  2>&1 | tee -a "$LOG"

echo -e "\n================================================================================"
echo "Milestone 0 baseline complete!"
echo "================================================================================"
echo "Run directory: $RUN_DIR"
echo "Training log:  $LOG"
echo "Diagnostics:   $DIAGNOSTIC_LOG"
echo "Checkpoint:    $CKPT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review metrics in $LOG (search for 'EM', 'F1', 'FirstTok@1')"
echo "  2. Check $DIAGNOSTIC_LOG for gradient flow confirmation"
echo "  3. Verify FirstTok@1 ≥ 0.15 before proceeding to Milestone 1"
echo "================================================================================"
