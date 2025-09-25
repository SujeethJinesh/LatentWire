#!/usr/bin/env bash
set -euo pipefail

# run_llama_single.sh
# End-to-end pipeline for the single-model (Llama-only) LatentWire workflow.
#
# Usage examples:
#   bash scripts/run_llama_single.sh            # quick smoke run (default)
#   bash scripts/run_llama_single.sh --hero     # longer "hero" configuration
#
# The pipeline performs:
#   Stage A – latent encoder warm-up on Llama.
#   Stage B – adapter/prefix training with mixed text↔latent warm-up.
#   Stage C – evaluation of text vs latent prompting on Llama.
#
# Results land in runs/${RUN_TAG}/ with checkpoints under runs/${RUN_TAG}/ckpt/.

: "${RUN_TAG:=llama_single_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/${RUN_TAG}"
CKPT_DIR="${RUN_DIR}/ckpt"
CKPT_STAGEA="${CKPT_DIR}/stageA"
CKPT_STAGEB="${CKPT_DIR}/stageB"
mkdir -p "$CKPT_STAGEA" "$CKPT_STAGEB"
LOG="${RUN_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-96}"

hero=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hero)
      hero=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $hero -eq 1 ]]; then
  TRAIN_SAMPLES_STAGEA=${TRAIN_SAMPLES_STAGEA:-8000}
  TRAIN_SAMPLES_STAGEB=${TRAIN_SAMPLES_STAGEB:-16000}
  EPOCHS_STAGEA=${EPOCHS_STAGEA:-6}
  EPOCHS_STAGEB=${EPOCHS_STAGEB:-10}
  SAMPLES="${SAMPLES:-1000}"
else
  TRAIN_SAMPLES_STAGEA=${TRAIN_SAMPLES_STAGEA:-640}
  TRAIN_SAMPLES_STAGEB=${TRAIN_SAMPLES_STAGEB:-1280}
  EPOCHS_STAGEA=${EPOCHS_STAGEA:-4}
  EPOCHS_STAGEB=${EPOCHS_STAGEB:-6}
  SAMPLES="${SAMPLES:-200}"
fi

LATENT_LEN="${LATENT_LEN:-64}"
D_Z="${D_Z:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export LW_APPLY_CHAT_TEMPLATE=1
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1,2,3}"
LLAMA_DEVICE_MAP="${LLAMA_DEVICE_MAP:-auto}"
GPU_MEM_GIB="${GPU_MEM_GIB:-78}"

COMMON_ARGS=(
  --models llama
  --llama_id "$LLAMA_ID"
  --dataset "$DATASET"
  --latent_len "$LATENT_LEN"
  --d_z "$D_Z"
  --use_chat_template
  --encoder_type stq
  --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2
  --encoder_use_chat_template
  --llama_device_map "$LLAMA_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES"
  --gpu_mem_gib "$GPU_MEM_GIB"
)

echo -e "\n=== CUDA preflight ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import os, torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
PY

# --- Stage A ---
echo -e "\n=== Stage A: Llama latent fit ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  "${COMMON_ARGS[@]}" \
  --samples "$TRAIN_SAMPLES_STAGEA" --epochs "$EPOCHS_STAGEA" \
  --batch_size "$BATCH_SIZE" --grad_accum_steps 16 \
  --save_dir "$CKPT_STAGEA" --auto_resume --save_training_stats \
  --train_append_bos_after_prefix yes \
  --warm_anchor_mode chat \
  --latent_private_len 16 \
  --first_token_ce_weight 2.0 --first_token_ce_schedule cosine --first_token_ce_peak 6.0 --first_token_ce_warmup_frac 0.3 \
  --K 4 --k_ce_weight 0.5 --kd_first_k_weight 0.5 --kd_tau 1.0 --state_kd_weight 0.1 --state_kd_layers 0,1,2 \
  --latent_align_weight 0.5 \
  --latent_keep_start 0.7 --latent_keep_end 1.0 --latent_keep_power 2.0 \
  --adapter_hidden_mult 4 --adapter_dropout 0.1 \
  --max_answer_tokens 24 --lr 5e-5 --max_grad_norm 1.0 \
  2>&1 | tee -a "$LOG"

# --- Stage B ---
echo -e "\n=== Stage B: Llama prefix training + warm-up ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  "${COMMON_ARGS[@]}" \
  --samples "$TRAIN_SAMPLES_STAGEB" --epochs "$EPOCHS_STAGEB" \
  --batch_size "$BATCH_SIZE" --grad_accum_steps 16 \
  --resume_from "$CKPT_STAGEA" \
  --save_dir "$CKPT_STAGEB" --auto_resume --no_load_optimizer --reset_epoch --save_training_stats \
  --use_prefix --prefix_tokens 24 --prefix_projection --peft_prefix_all_layers yes \
  --train_append_bos_after_prefix yes \
  --warm_anchor_mode chat \
  --latent_private_len 16 \
  --first_token_ce_weight 5.0 --first_token_ce_schedule cosine --first_token_ce_peak 10.0 --first_token_ce_warmup_frac 0.25 \
  --K 4 --k_ce_weight 0.5 --kd_first_k_weight 1.5 --kd_tau 0.7 --state_kd_weight 0.1 --state_kd_layers 0,1,2 \
  --latent_align_weight 1.0 \
  --latent_keep_start 0.5 --latent_keep_end 1.0 --latent_keep_power 2.0 \
  --warmup_text_latent_epochs 1.0 \
  --warmup_align_tokens 8 --warmup_align_weight 1.5 \
  --warmup_text_teacher_weight 2.0 \
  --warmup_text_latent_weight 0.0 --warmup_text_latent_weight_end 1.0 \
  --warmup_tail_prob 0.0 \
  --adapter_hidden_mult 4 --adapter_dropout 0.1 \
  --max_answer_tokens 24 --lr 5e-5 --max_grad_norm 1.0 \
  2>&1 | tee -a "$LOG"

# --- Stage C ---
echo -e "\n=== Stage C: Evaluation (Llama only) ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py \
  --models llama \
  --ckpt "$CKPT_STAGEB" \
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

echo -e "\n✓ Llama-only pipeline complete. Logs: $LOG\n"
