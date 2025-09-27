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
BASE_RUN_TAG="$RUN_TAG"

SAVE_EVERY_STAGEA="${SAVE_EVERY_STAGEA:-0}"
SAVE_EVERY_STAGEB="${SAVE_EVERY_STAGEB:-0}"

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
  if [[ "$RUN_TAG" == llama_single_* ]]; then
    RUN_TAG="hero"
  fi
  BASE_RUN_TAG="$RUN_TAG"
else
#   TRAIN_SAMPLES_STAGEA=${TRAIN_SAMPLES_STAGEA:-640}
#   TRAIN_SAMPLES_STAGEB=${TRAIN_SAMPLES_STAGEB:-2560}
#   EPOCHS_STAGEA=${EPOCHS_STAGEA:-4}
#   EPOCHS_STAGEB=${EPOCHS_STAGEB:-8}
  TRAIN_SAMPLES_STAGEA=${TRAIN_SAMPLES_STAGEA:-320}
  TRAIN_SAMPLES_STAGEB=${TRAIN_SAMPLES_STAGEB:-2560}
  EPOCHS_STAGEA=${EPOCHS_STAGEA:-2}
  EPOCHS_STAGEB=${EPOCHS_STAGEB:-2}
  SAMPLES="${SAMPLES:-200}"
fi

KD_WEIGHT_STAGEA_DEFAULT="0.5"
KD_WEIGHT_STAGEB_DEFAULT="0.5"

KD_WEIGHT_STAGEA="${KD_WEIGHT_STAGEA:-$KD_WEIGHT_STAGEA_DEFAULT}"
KD_WEIGHT_STAGEB="${KD_WEIGHT_STAGEB:-$KD_WEIGHT_STAGEB_DEFAULT}"

LATENT_LEN="${LATENT_LEN:-64}"
D_Z="${D_Z:-256}"
BATCH_SIZE_STAGEA="${BATCH_SIZE_STAGEA:-24}"
BATCH_SIZE_STAGEB="${BATCH_SIZE_STAGEB:-32}"
DEEP_PREFIX_LEN="${DEEP_PREFIX_LEN:-24}"
DEEP_PREFIX_DROPOUT="${DEEP_PREFIX_DROPOUT:-0.1}"
REFINER_LAYERS="${REFINER_LAYERS:-2}"
REFINER_HEADS="${REFINER_HEADS:-4}"

LATENT_LEN_LIST="${LATENT_LEN_LIST:-$LATENT_LEN}"
D_Z_LIST="${D_Z_LIST:-$D_Z}"
REFINER_LAYERS_LIST="${REFINER_LAYERS_LIST:-$REFINER_LAYERS}"
REFINER_HEADS_LIST="${REFINER_HEADS_LIST:-$REFINER_HEADS}"
USE_GIST_HEAD="${USE_GIST_HEAD:-1}"
GIST_TARGET_LEN="${GIST_TARGET_LEN:-48}"
GIST_HIDDEN="${GIST_HIDDEN:-512}"
GIST_LAYERS="${GIST_LAYERS:-2}"
GIST_DROPOUT="${GIST_DROPOUT:-0.1}"
GIST_WEIGHT="${GIST_WEIGHT:-0.02}"
GIST_MASK_PROB="${GIST_MASK_PROB:-0.15}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_FIRSTN="${LORA_FIRSTN:-8}"

export LW_APPLY_CHAT_TEMPLATE=1
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="false"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1,2,3}"
LLAMA_DEVICE_MAP="${LLAMA_DEVICE_MAP:-auto}"
GPU_MEM_GIB="${GPU_MEM_GIB:-78}"

COMMON_ARGS_BASE=(
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
)

IFS_SAVE="$IFS"
IFS=',' read -r -a LATENT_LEN_GRID <<< "$LATENT_LEN_LIST"
IFS=',' read -r -a D_Z_GRID <<< "$D_Z_LIST"
IFS=',' read -r -a REFINER_LAYERS_GRID <<< "$REFINER_LAYERS_LIST"
IFS=',' read -r -a REFINER_HEADS_GRID <<< "$REFINER_HEADS_LIST"
IFS="$IFS_SAVE"

multi_combo=0
if [[ ${#LATENT_LEN_GRID[@]} -gt 1 || ${#D_Z_GRID[@]} -gt 1 || ${#REFINER_LAYERS_GRID[@]} -gt 1 || ${#REFINER_HEADS_GRID[@]} -gt 1 ]]; then
  multi_combo=1
fi

combo_index=0
for LATENT_LEN_CURRENT in "${LATENT_LEN_GRID[@]}"; do
  for D_Z_CURRENT in "${D_Z_GRID[@]}"; do
    for REFINER_LAYERS_CURRENT in "${REFINER_LAYERS_GRID[@]}"; do
      for REFINER_HEADS_CURRENT in "${REFINER_HEADS_GRID[@]}"; do
        combo_index=$((combo_index + 1))
        combo_suffix="m${LATENT_LEN_CURRENT}_dz${D_Z_CURRENT}_rl${REFINER_LAYERS_CURRENT}_rh${REFINER_HEADS_CURRENT}"
        if [[ $multi_combo -eq 1 ]]; then
          RUN_TAG="${BASE_RUN_TAG}_${combo_suffix}"
        else
          RUN_TAG="$BASE_RUN_TAG"
        fi
        RUN_DIR="runs/${RUN_TAG}"
        CKPT_DIR="${RUN_DIR}/ckpt"
        CKPT_STAGEA="${CKPT_DIR}/stageA"
        CKPT_STAGEB="${CKPT_DIR}/stageB"
        mkdir -p "$CKPT_STAGEA" "$CKPT_STAGEB"
        if [[ $multi_combo -eq 1 ]]; then
          LOG="${RUN_DIR}/pipeline_${combo_suffix}_$(date +%Y%m%d_%H%M%S).log"
        else
          LOG="${RUN_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
        fi
        DIAGNOSTIC_LOG="${RUN_DIR}/diagnostics.jsonl"
        : > "$DIAGNOSTIC_LOG"

        COMMON_ARGS=(
          "${COMMON_ARGS_BASE[@]}"
          --latent_len "$LATENT_LEN_CURRENT"
          --d_z "$D_Z_CURRENT"
          --latent_refiner_layers "$REFINER_LAYERS_CURRENT"
          --latent_refiner_heads "$REFINER_HEADS_CURRENT"
        )

        echo -e "\n>>> Combination $combo_index: $combo_suffix" | tee -a "$LOG"
        echo -e "    RUN_TAG=$RUN_TAG" | tee -a "$LOG"

        echo -e "\n=== CUDA preflight ===" | tee -a "$LOG"
        python - <<'PY' 2>&1 | tee -a "$LOG"
import os, torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
PY

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
LORA_ARGS=(
  --use_lora
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_firstN "$LORA_FIRSTN"
)

        if [[ $USE_GIST_HEAD -eq 1 ]]; then
          WARMUP_FLAG=(--kd_skip_text)
        else
          WARMUP_FLAG=(--kd_skip_text)
        fi

        # --- Stage A ---
        echo -e "\n=== Stage A: Llama latent fit ===\n" | tee -a "$LOG"
        steps_per_epoch_stagea=$(( (TRAIN_SAMPLES_STAGEA + BATCH_SIZE_STAGEA - 1) / BATCH_SIZE_STAGEA ))
        save_every_stagea=$SAVE_EVERY_STAGEA
        if [[ $hero -eq 1 && $save_every_stagea -le 0 ]]; then
          save_every_stagea=$steps_per_epoch_stagea
        fi
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
          "${COMMON_ARGS[@]}" \
          --samples "$TRAIN_SAMPLES_STAGEA" --epochs "$EPOCHS_STAGEA" \
          --batch_size "$BATCH_SIZE_STAGEA" --grad_accum_steps 16 \
          --save_dir "$CKPT_STAGEA" --auto_resume --save_training_stats \
          --train_append_bos_after_prefix yes \
          --warm_anchor_mode chat \
          --latent_private_len 16 \
          --use_deep_prefix --deep_prefix_len "$DEEP_PREFIX_LEN" --deep_prefix_dropout "$DEEP_PREFIX_DROPOUT" \
          --first_token_ce_weight 1.5 --first_token_ce_schedule cosine --first_token_ce_peak 4.0 --first_token_ce_warmup_frac 0.3 \
          --K 8 --k_ce_weight 0.5 --kd_first_k_weight "$KD_WEIGHT_STAGEA" --kd_tau 2.0 --state_kd_weight 0.1 --state_kd_layers 0,1,2,3 \
          --latent_align_weight 0.5 --latent_prefix_align_weight 0.25 \
          --latent_keep_start 0.7 --latent_keep_end 1.0 --latent_keep_power 2.0 \
          --warmup_text_latent_epochs 0.5 \
          --warmup_align_tokens 8 --warmup_align_weight 1.0 \
          --warmup_text_teacher_weight 1.5 \
          --warmup_text_latent_weight 0.0 --warmup_text_latent_weight_end 0.5 \
          --warmup_tail_prob 0.0 \
          --adapter_hidden_mult 4 --adapter_dropout 0.1 \
          --max_answer_tokens 24 --lr 5e-5 --max_grad_norm 1.0 \
          --grad_diag_interval 100 --grad_diag_components "$GRAD_COMPONENTS_LATENT" \
          --diagnostic_log "$DIAGNOSTIC_LOG" \
          --save_every "$save_every_stagea" \
          "${GIST_ARGS[@]}" \
          "${LORA_ARGS[@]}" \
          "${WARMUP_FLAG[@]}" \
          2>&1 | tee -a "$LOG"

        # --- Stage B ---
        echo -e "\n=== Stage B: Llama prefix training + warm-up ===\n" | tee -a "$LOG"
        steps_per_epoch_stageb=$(( (TRAIN_SAMPLES_STAGEB + BATCH_SIZE_STAGEB - 1) / BATCH_SIZE_STAGEB ))
        save_every_stageb=$SAVE_EVERY_STAGEB
        if [[ $hero -eq 1 && $save_every_stageb -le 0 ]]; then
          save_every_stageb=$steps_per_epoch_stageb
        fi
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
          "${COMMON_ARGS[@]}" \
          --samples "$TRAIN_SAMPLES_STAGEB" --epochs "$EPOCHS_STAGEB" \
          --batch_size "$BATCH_SIZE_STAGEB" --grad_accum_steps 16 \
          --resume_from "$CKPT_STAGEA" \
          --save_dir "$CKPT_STAGEB" --auto_resume --no_load_optimizer --reset_epoch --save_training_stats \
          --use_prefix --prefix_tokens 24 --prefix_projection --peft_prefix_all_layers yes \
          --train_append_bos_after_prefix yes \
          --warm_anchor_mode chat \
          --latent_private_len 16 \
          --use_deep_prefix --deep_prefix_len "$DEEP_PREFIX_LEN" --deep_prefix_dropout "$DEEP_PREFIX_DROPOUT" \
          --first_token_ce_weight 6.0 --first_token_ce_schedule cosine --first_token_ce_peak 10.0 --first_token_ce_warmup_frac 0.4 \
          --K 8 --k_ce_weight 0.5 --kd_first_k_weight "$KD_WEIGHT_STAGEB" --kd_tau 2.0 --state_kd_weight 0.1 --state_kd_layers 0,1,2,3,4 \
          --latent_align_weight 1.0 --latent_prefix_align_weight 0.5 \
          --latent_keep_start 0.5 --latent_keep_end 1.0 --latent_keep_power 2.0 \
          --warmup_text_latent_epochs 1.5 \
          --warmup_align_tokens 8 --warmup_align_weight 1.5 \
          --warmup_text_teacher_weight 2.0 \
          --warmup_text_latent_weight 0.2 --warmup_text_latent_weight_end 1.0 \
          --warmup_tail_prob 0.1 \
          --adapter_hidden_mult 4 --adapter_dropout 0.1 \
          --max_answer_tokens 24 --lr 5e-5 --max_grad_norm 1.0 \
          --grad_diag_interval 25 --grad_diag_components "$GRAD_COMPONENTS_LATENT" \
          --diagnostic_log "$DIAGNOSTIC_LOG" \
          --save_every "$save_every_stageb" \
          "${GIST_ARGS[@]}" \
          "${LORA_ARGS[@]}" \
          "${WARMUP_FLAG[@]}" \
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
          --token_budget_mode content_only --token_budget_k "$LATENT_LEN_CURRENT" \
          2>&1 | tee -a "$LOG"

        echo -e "\n✓ Llama-only pipeline complete for $RUN_TAG. Logs: $LOG\n"
      done
    done
  done
done
