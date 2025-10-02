#!/usr/bin/env bash
set -euo pipefail

# resume_hero_stageb.sh
# Resume/Continue Stage B training with schedule fixes for dropout retention
#
# This script can be run multiple times - it will automatically resume from the latest
# checkpoint in runs/hero_resume/ckpt_stageb and continue training until 8 epochs total.
#
# Based on v1 analysis showing peak performance (19.4% first_acc) at keep_prob=0.6-0.85,
# with regression when annealing to 1.0. This is NOT an architecture limit—schedule issue.
#
# Key changes from original run:
# OOM fixes:
# - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (fix fragmentation)
# - KD_TEACHER_CHUNK=1 (reduce memory per chunk in latent mode)
# Performance improvements:
# - TEXT_TEACHER_CHUNK=4 (speed up warm-up 4× by batching text teacher)
# Quality improvements:
# - FIRST_TOKEN_CE_WEIGHT_STAGEB: 9.0 → 11.0 (increase acceptance pressure)
# - KD_WEIGHT_STAGEB: 1.0 → 0.5 (reduce competing gradients, free memory)
# Schedule fixes (based on v1 hero_resume analysis):
# - LATENT_KEEP_END: 1.0 → 0.85 (freeze dropout at sweet spot)
# - EPOCHS_STAGEB: 6 → 8 (consolidate with frozen dropout)
# - Peak checkpointing: train.py now saves "_best" checkpoint with LoRA/Prefix weights
#
# IMPORTANT: Peak checkpoint bug was fixed in train.py on 2025-10-02 to save LoRA/Prefix
# weights. Previous _best checkpoint is invalid. Run this script to capture new peak.
#
# Usage:
#   bash scripts/resume_hero_stageb.sh
#
# The script will:
# - Resume from latest checkpoint in runs/hero_resume/ckpt_stageb
# - Continue training until 8 epochs complete
# - Save new peak checkpoints with complete LoRA/Prefix weights to ckpt_stageb_best

RUN_TAG="${RUN_TAG:-hero_resume}"
BASE_RUN_TAG="$RUN_TAG"

# OOM fixes - critical for avoiding fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Hero run configuration
TRAIN_SAMPLES_STAGEB=${TRAIN_SAMPLES_STAGEB:-16000}
EPOCHS_STAGEB=${EPOCHS_STAGEB:-8}  # EXTENDED from 6 to 8 for consolidation with frozen dropout
SAMPLES="${SAMPLES:-1000}"

# Stage B hyperparameters - Quality improvements
WARMUP_TEXT_LATENT_EPOCHS_STAGEB="${WARMUP_TEXT_LATENT_EPOCHS_STAGEB:-2.0}"
WARMUP_TAIL_PROB_STAGEB="${WARMUP_TAIL_PROB_STAGEB:-0.02}"
FIRST_TOKEN_CE_WEIGHT_STAGEB="${FIRST_TOKEN_CE_WEIGHT_STAGEB:-11.0}"
LATENT_PRIVATE_LEN="${LATENT_PRIVATE_LEN:-24}"
KD_WEIGHT_STAGEB="${KD_WEIGHT_STAGEB:-0.5}"
WARMUP_TEXT_TEACHER_WEIGHT_STAGEB="${WARMUP_TEXT_TEACHER_WEIGHT_STAGEB:-2.0}"
WARMUP_TEXT_LATENT_WEIGHT_STAGEB="${WARMUP_TEXT_LATENT_WEIGHT_STAGEB:-0.2}"
WARMUP_TEXT_LATENT_WEIGHT_END_STAGEB="${WARMUP_TEXT_LATENT_WEIGHT_END_STAGEB:-1.0}"

# SCHEDULE FIX: Freeze dropout at 0.85 (sweet spot from v1 analysis)
LATENT_KEEP_START="${LATENT_KEEP_START:-0.5}"
LATENT_KEEP_END="${LATENT_KEEP_END:-0.85}"  # CHANGED from 1.0 to 0.85
LATENT_KEEP_POWER="${LATENT_KEEP_POWER:-2.0}"

# Model configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-88}"

# Architecture parameters
LATENT_LEN="${LATENT_LEN:-64}"
D_Z="${D_Z:-256}"
BATCH_SIZE_STAGEB="${BATCH_SIZE_STAGEB:-36}"
GRAD_ACCUM_STAGEB="${GRAD_ACCUM_STAGEB:-12}"
DEEP_PREFIX_LEN="${DEEP_PREFIX_LEN:-100}"
DEEP_PREFIX_DROPOUT="${DEEP_PREFIX_DROPOUT:-0.05}"
REFINER_LAYERS="${REFINER_LAYERS:-2}"
REFINER_HEADS="${REFINER_HEADS:-4}"

# LoRA parameters
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_FIRSTN="${LORA_FIRSTN:-16}"

# Gist head (disabled by default)
USE_GIST_HEAD="${USE_GIST_HEAD:-0}"
GIST_TARGET_LEN="${GIST_TARGET_LEN:-48}"
GIST_HIDDEN="${GIST_HIDDEN:-512}"
GIST_LAYERS="${GIST_LAYERS:-2}"
GIST_DROPOUT="${GIST_DROPOUT:-0.1}"
GIST_WEIGHT="${GIST_WEIGHT:-0.02}"
GIST_MASK_PROB="${GIST_MASK_PROB:-0.15}"

# Environment setup
export LW_APPLY_CHAT_TEMPLATE=1
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"
export TEXT_TEACHER_CHUNK="${TEXT_TEACHER_CHUNK:-4}"  # INCREASED from 1 to 4 for speed (warm-up)
export KD_TEACHER_CHUNK="${KD_TEACHER_CHUNK:-1}"  # Keep at 1 for OOM safety (latent mode)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1,2,3}"
DEFAULT_LLAMA_DEVICE_MAP='{"model.embed_tokens":0,"model.rotary_emb":0,"model.layers.0":0,"model.layers.1":0,"model.layers.2":0,"model.layers.3":0,"model.layers.4":0,"model.layers.5":0,"model.layers.6":0,"model.layers.7":0,"model.layers.8":1,"model.layers.9":1,"model.layers.10":1,"model.layers.11":1,"model.layers.12":1,"model.layers.13":1,"model.layers.14":1,"model.layers.15":1,"model.layers.16":2,"model.layers.17":2,"model.layers.18":2,"model.layers.19":2,"model.layers.20":2,"model.layers.21":2,"model.layers.22":2,"model.layers.23":2,"model.layers.24":3,"model.layers.25":3,"model.layers.26":3,"model.layers.27":3,"model.layers.28":3,"model.layers.29":3,"model.layers.30":3,"model.layers.31":3,"model.norm":3,"lm_head":3}'
LLAMA_DEVICE_MAP="${LLAMA_DEVICE_MAP:-$DEFAULT_LLAMA_DEVICE_MAP}"
GPU_MEM_GIB="${GPU_MEM_GIB:-70}"

SAVE_EVERY_STAGEB="${SAVE_EVERY_STAGEB:-0}"

# Checkpoint paths
ORIGINAL_CHECKPOINT="${CHECKPOINT_BASE:-runs/hero/ckpt_stageb}"
SAVE_DIR="${SAVE_DIR:-runs/${RUN_TAG}/ckpt_stageb}"
LOG="runs/${RUN_TAG}/pipeline_$(date +%Y%m%d_%H%M%S).log"
DIAGNOSTIC_LOG="runs/${RUN_TAG}/diagnostics.jsonl"

mkdir -p "runs/${RUN_TAG}"

# Resume logic: Copy original checkpoint to save_dir if save_dir is empty
# This allows --auto_resume to find it on first run, and find the latest on subsequent runs
if [ ! -f "${SAVE_DIR}/state.pt" ]; then
  echo "No checkpoint in ${SAVE_DIR}, copying from ${ORIGINAL_CHECKPOINT}..."
  mkdir -p "${SAVE_DIR}"
  cp -r "${ORIGINAL_CHECKPOINT}"/* "${SAVE_DIR}/" 2>/dev/null || true
  echo "Copied original checkpoint to ${SAVE_DIR}"
fi
echo "Will use --auto_resume to find latest checkpoint in ${SAVE_DIR}"

COMMON_ARGS=(
  --models llama
  --llama_id "$LLAMA_ID"
  --dataset "$DATASET"
  --use_chat_template
  --encoder_type stq
  --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2
  --encoder_use_chat_template
  --llama_device_map "$LLAMA_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES"
  --gpu_mem_gib "$GPU_MEM_GIB"
  --latent_len "$LATENT_LEN"
  --d_z "$D_Z"
  --latent_refiner_layers "$REFINER_LAYERS"
  --latent_refiner_heads "$REFINER_HEADS"
)

if [[ $USE_GIST_HEAD -eq 1 ]]; then
  GIST_ARGS=(
    --use_gist_head
    --gist_target_len "$GIST_TARGET_LEN"
    --gist_hidden "$GIST_HIDDEN"
    --gist_layers "$GIST_LAYERS"
    --gist_dropout "$GIST_DROPOUT"
    --gist_weight "$GIST_WEIGHT"
    --gist_mask_prob "$GIST_MASK_PROB"
  )
  GRAD_COMPONENTS_LATENT="tf,first,kce,kd,align,latent_align,latent_prefix_align,gist"
else
  GIST_ARGS=()
  GRAD_COMPONENTS_LATENT="tf,first,kce,kd,align,latent_align,latent_prefix_align"
fi

if [[ $USE_LORA -eq 1 ]]; then
  LORA_ARGS=(
    --use_lora
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_firstN "$LORA_FIRSTN"
  )
else
  LORA_ARGS=()
fi

WARMUP_FLAG=(--kd_skip_text)

echo "=== Resume Hero Stage B Training (Schedule Fix) ===" | tee "$LOG"
echo "Run tag: $RUN_TAG" | tee -a "$LOG"
echo "Checkpoint directory: $SAVE_DIR (using --auto_resume)" | tee -a "$LOG"
echo "Epochs: $EPOCHS_STAGEB (extended for consolidation)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "SCHEDULE FIXES (based on v1 analysis):" | tee -a "$LOG"
echo "  - LATENT_KEEP_END: 1.0 → 0.85 (freeze at sweet spot)" | tee -a "$LOG"
echo "  - EPOCHS extended to 8 (consolidate with frozen dropout)" | tee -a "$LOG"
echo "  - Peak checkpointing: train.py saves '_best' automatically" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Analysis showed peak at keep_prob=0.6-0.85:" | tee -a "$LOG"
echo "  - Peak: 19.4% first_acc at keep_prob=0.613" | tee -a "$LOG"
echo "  - 26 steps achieved ≥10% (all at keep_prob 0.55-0.82)" | tee -a "$LOG"
echo "  - Final eval (keep_prob=1.0) only 4.4% - model never learned full latents" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Retained from v1:" | tee -a "$LOG"
echo "  - FIRST_TOKEN_CE_WEIGHT=11.0" | tee -a "$LOG"
echo "  - KD_WEIGHT=0.5" | tee -a "$LOG"
echo "  - OOM fixes: expandable_segments, KD_TEACHER_CHUNK=1" | tee -a "$LOG"
echo "  - Performance: TEXT_TEACHER_CHUNK=4" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# CUDA preflight check
python3 -c "import os, torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES')); print('PYTORCH_CUDA_ALLOC_CONF:', os.getenv('PYTORCH_CUDA_ALLOC_CONF'))" 2>&1 | tee -a "$LOG"

echo -e "\n=== Stage B: Training with frozen dropout (keep_prob→0.85) ===\n" | tee -a "$LOG"

steps_per_epoch_stageb=$(( (TRAIN_SAMPLES_STAGEB + BATCH_SIZE_STAGEB - 1) / BATCH_SIZE_STAGEB ))
save_every_stageb=$SAVE_EVERY_STAGEB
if [[ $save_every_stageb -le 0 ]]; then
  save_every_stageb=$steps_per_epoch_stageb
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  "${COMMON_ARGS[@]}" \
  --samples "$TRAIN_SAMPLES_STAGEB" --epochs "$EPOCHS_STAGEB" \
  --batch_size "$BATCH_SIZE_STAGEB" --grad_accum_steps "$GRAD_ACCUM_STAGEB" \
  --save_dir "$SAVE_DIR" --auto_resume --save_training_stats \
  --use_prefix --prefix_tokens "$DEEP_PREFIX_LEN" --prefix_projection --peft_prefix_all_layers yes \
  --train_append_bos_after_prefix yes \
  --warm_anchor_mode chat \
  --latent_private_len "$LATENT_PRIVATE_LEN" \
  --use_deep_prefix --deep_prefix_len "$DEEP_PREFIX_LEN" --deep_prefix_dropout "$DEEP_PREFIX_DROPOUT" \
  --first_token_ce_weight "$FIRST_TOKEN_CE_WEIGHT_STAGEB" --first_token_ce_schedule none \
  --K 8 --k_ce_weight 0.5 --kd_first_k_weight "$KD_WEIGHT_STAGEB" --kd_tau 2.0 --state_kd_weight 0.1 --state_kd_layers 0,1,2,3,4 \
  --latent_align_weight 1.0 --latent_prefix_align_weight 0.5 \
  --latent_keep_start "$LATENT_KEEP_START" --latent_keep_end "$LATENT_KEEP_END" --latent_keep_power "$LATENT_KEEP_POWER" \
  --warmup_text_latent_epochs "$WARMUP_TEXT_LATENT_EPOCHS_STAGEB" \
  --warmup_align_tokens 8 --warmup_align_weight 1.5 \
  --warmup_text_teacher_weight "$WARMUP_TEXT_TEACHER_WEIGHT_STAGEB" \
  --warmup_text_latent_weight "$WARMUP_TEXT_LATENT_WEIGHT_STAGEB" --warmup_text_latent_weight_end "$WARMUP_TEXT_LATENT_WEIGHT_END_STAGEB" \
  --warmup_tail_prob "$WARMUP_TAIL_PROB_STAGEB" \
  --adapter_hidden_mult 4 --adapter_dropout 0.1 \
  --max_answer_tokens 24 --lr 5e-5 --max_grad_norm 1.0 \
  --grad_diag_interval 25 --grad_diag_components "$GRAD_COMPONENTS_LATENT" \
  --diagnostic_log "$DIAGNOSTIC_LOG" \
  --save_every "$save_every_stageb" \
  "${GIST_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  "${WARMUP_FLAG[@]}" \
  2>&1 | tee -a "$LOG"

# --- Evaluation ---
echo -e "\n=== Evaluation (Llama only) ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py \
  --models llama \
  --ckpt "$SAVE_DIR" \
  --llama_id "$LLAMA_ID" \
  --samples "$SAMPLES" --dataset "$DATASET" \
  --fresh_eval --max_new_tokens "$MAX_NEW_TOKENS" \
  --chunk_size "$CHUNK_SIZE" \
  --latent_anchor_mode chat --append_bos_after_prefix yes \
  --use_chat_template yes \
  --first_token_top_p 1.0 --first_token_temperature 0.0 \
  --prefix_gain 1.1 \
  --token_budget_mode content_only --token_budget_k "$LATENT_LEN" \
  2>&1 | tee -a "$LOG"

echo -e "\n✓ Hero Stage B (schedule fix) complete. Logs: $LOG\n"
echo "Note: Evaluate using the _best checkpoint for peak performance:"
echo "  --ckpt ${SAVE_DIR}_best"
