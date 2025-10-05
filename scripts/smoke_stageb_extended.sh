#!/usr/bin/env bash
set -euo pipefail

# smoke_stageb_extended.sh
# Extended Stage B smoke test to validate if more steps solve F1=0
#
# This runs ONLY Stage B with 8 epochs (320 steps, 2× original smoke)
# to test if the first_acc regression (8.33%→4.17%) can be overcome.
#
# If this achieves F1 > 0, hero is safe. If F1 still 0, we need architectural fixes.

RUN_TAG="${RUN_TAG:-smoke_stageb_ext}"
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"

# Extended Stage B: 8 epochs instead of 4
TRAIN_SAMPLES_STAGEB=960  # Same 40 steps/epoch
EPOCHS_STAGEB=8           # 2× original (320 steps total)
WARMUP_TEXT_LATENT_EPOCHS_STAGEB=0.5  # 20 steps warm-up
SAMPLES=200

# Use Stage A checkpoint from previous smoke
RESUME_FROM="${RESUME_FROM:-runs/smoke/ckpt/stageA}"

RUN_DIR="runs/${RUN_TAG}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$CKPT_DIR"
LOG="${RUN_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
DIAGNOSTIC_LOG="${RUN_DIR}/diagnostics.jsonl"
: > "$DIAGNOSTIC_LOG"

export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

LLAMA_DEVICE_MAP='{"model.embed_tokens":0,"model.rotary_emb":0,"model.layers.0":0,"model.layers.1":0,"model.layers.2":0,"model.layers.3":0,"model.layers.4":0,"model.layers.5":0,"model.layers.6":0,"model.layers.7":0,"model.layers.8":1,"model.layers.9":1,"model.layers.10":1,"model.layers.11":1,"model.layers.12":1,"model.layers.13":1,"model.layers.14":1,"model.layers.15":1,"model.layers.16":2,"model.layers.17":2,"model.layers.18":2,"model.layers.19":2,"model.layers.20":2,"model.layers.21":2,"model.layers.22":2,"model.layers.23":2,"model.layers.24":3,"model.layers.25":3,"model.layers.26":3,"model.layers.27":3,"model.layers.28":3,"model.layers.29":3,"model.layers.30":3,"model.layers.31":3,"model.norm":3,"lm_head":3}'

echo "=== Extended Stage B Smoke Test ===" | tee "$LOG"
echo "Resume from: $RESUME_FROM" | tee -a "$LOG"
echo "Epochs: $EPOCHS_STAGEB (2× original)" | tee -a "$LOG"
echo "Expected steps: 320 (40/epoch × 8 epochs)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Stage B Extended ---
echo "=== Stage B: Extended LoRA Training (8 epochs) ===" | tee -a "$LOG"

python -u latentwire/train.py \
  --models llama \
  --llama_id "$LLAMA_ID" \
  --dataset "$DATASET" \
  --use_chat_template \
  --encoder_type stq \
  --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder_use_chat_template \
  --llama_device_map "$LLAMA_DEVICE_MAP" \
  --llama_devices "0,1,2,3" \
  --gpu_mem_gib 70 \
  --latent_len 64 \
  --d_z 256 \
  --latent_refiner_layers 2 \
  --latent_refiner_heads 4 \
  --samples "$TRAIN_SAMPLES_STAGEB" \
  --epochs "$EPOCHS_STAGEB" \
  --batch_size 24 \
  --grad_accum_steps 12 \
  --resume_from "$RESUME_FROM" \
  --save_dir "$CKPT_DIR" \
  --auto_resume \
  --no_load_optimizer \
  --reset_epoch \
  --save_training_stats \
  --train_append_bos_after_prefix yes \
  --warm_anchor_mode chat \
  --latent_private_len 24 \
  --use_deep_prefix \
  --deep_prefix_len 100 \
  --deep_prefix_dropout 0.05 \
  --first_token_ce_weight 9.0 \
  --first_token_ce_schedule none \
  --K 8 \
  --k_ce_weight 0.5 \
  --kd_first_k_weight 1.0 \
  --kd_tau 2.0 \
  --state_kd_weight 0.1 \
  --state_kd_layers 0,1,2,3,4 \
  --latent_align_weight 1.0 \
  --latent_prefix_align_weight 0.5 \
  --latent_keep_start 0.5 \
  --latent_keep_end 0.85 \
  --latent_keep_power 2.0 \
  --warmup_text_latent_epochs "$WARMUP_TEXT_LATENT_EPOCHS_STAGEB" \
  --warmup_align_tokens 8 \
  --warmup_align_weight 1.5 \
  --warmup_text_teacher_weight 2.0 \
  --warmup_text_latent_weight 0.2 \
  --warmup_text_latent_weight_end 1.0 \
  --warmup_tail_prob 0.0 \
  --adapter_hidden_mult 4 \
  --adapter_dropout 0.1 \
  --max_answer_tokens 24 \
  --lr 5e-5 \
  --max_grad_norm 1.0 \
  --grad_diag_interval 25 \
  --grad_diag_components "tf,first,kce,kd,align,latent_align,latent_prefix_align" \
  --diagnostic_log "$DIAGNOSTIC_LOG" \
  --save_every 0 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_firstN 16 \
  --kd_skip_text \
  2>&1 | tee -a "$LOG"

# --- Evaluation ---
echo "" | tee -a "$LOG"
echo "=== Evaluation ===" | tee -a "$LOG"

python -u latentwire/eval.py \
  --models llama \
  --ckpt "$CKPT_DIR" \
  --llama_id "$LLAMA_ID" \
  --samples "$SAMPLES" \
  --dataset "$DATASET" \
  --fresh_eval \
  --max_new_tokens 16 \
  --chunk_size 88 \
  --latent_anchor_mode chat \
  --append_bos_after_prefix yes \
  --use_chat_template yes \
  --first_token_top_p 1.0 \
  --first_token_temperature 0.0 \
  --prefix_gain 1.1 \
  --token_budget_mode content_only \
  --token_budget_k 64 \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "✓ Extended Stage B smoke complete" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "  Diagnostics: $DIAGNOSTIC_LOG" | tee -a "$LOG"
