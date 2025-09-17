#!/usr/bin/env bash
set -euo pipefail

# Recommended pipeline for 4×H100, targeting ≥4× compression at ~0.8× Text F1 (SQuAD)
DO_TRAIN=1
DO_FINAL_EVAL=1

DATASET="squad"        # start with SQuAD for stability; add hotpot after convergence
SAMPLES=1000           # final eval
SMOKE_SAMPLES=400      # per-epoch eval
MAX_NEW_TOKENS=16
SEQUENTIAL_EVAL=1
FRESH_EVAL=1
LOAD_4BIT=1
CHUNK_SIZE=8
TOKEN_BUDGET_MODE="content_only"
TOKEN_BUDGET_K=32

DETERMINISTIC_EVAL=1
if [[ "$DETERMINISTIC_EVAL" -eq 1 ]]; then
  FIRST_TOKEN_TOP_P=1.0
  FIRST_TOKEN_TEMPERATURE=0.0
else
  FIRST_TOKEN_TOP_P=0.95
  FIRST_TOKEN_TEMPERATURE=0.7
fi

# Anchor & decode
LATENT_ANCHOR_MODE="text"
LATENT_ANCHOR_TEXT="Answer: "
APPEND_BOS_AFTER_PREFIX="no"
CALIBRATION="embed_rms"
PREFIX_GAIN=1.15

# Training knobs (stronger first steps & KD)
EPOCHS=12
BATCH_SIZE=40
GRAD_ACCUM_STEPS=32
TRAIN_SAMPLES=87599
ENCODER_TYPE="byte"           # keep byte for generality; switch to 'simple-st' if GPU-limited
LATENT_LEN=32                 # ≈4× compression if text prompts avg ~128 tokens
LATENT_SHARED_LEN=24
LATENT_PRIVATE_LEN=4
D_Z=256
BYTE_MAX=2048
LR=3e-5
SCALE_L2=0.05
ADAPTER_RMS_L2=0.0
MAX_GRAD_NORM=1.0
WARM_ANCHOR_TEXT="Answer: "
FIRST_TOKEN_CE=2.0            # ↑ from 1.0
TRAIN_APPEND_BOS="no"
MAX_ANSWER_TOKENS=24
ADAPTER_HIDDEN_MULT=2
ADAPTER_COLORIZE=1
ADAPTER_METADATA=1
MANIFOLD_STAT_WEIGHT=0.001
STATE_KD_WEIGHT=0.0
STATE_KD_LAYERS="0,1,2"
K=6                           # ↑ from 4
K_CE_WEIGHT=1.0               # ↑ from 0.5
KD_FIRST_K_WEIGHT=1.5         # ↑ from 1.0
KD_TAU=1.25                   # slightly sharper matching

# Model IDs (8B/7B)
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

# GPUs
CUDA_VISIBLE_DEVICES="0,1,2,3"
LLAMA_DEVICES="0,1"
QWEN_DEVICES="2,3"
GPU_MEM_GIB=78
LLAMA_DEVICE_MAP="auto"
QWEN_DEVICE_MAP="auto"

source .venv/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TRAIN_ARGS_COMMON=(
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES"
  --epochs 1 --batch_size "$BATCH_SIZE"
  --encoder_type "$ENCODER_TYPE"
  --latent_len "$LATENT_LEN" --latent_shared_len "$LATENT_SHARED_LEN" --latent_private_len "$LATENT_PRIVATE_LEN" --d_z "$D_Z"
  --qwen_id "$QWEN_ID" --llama_id "$LLAMA_ID"
  --lr "$LR" --scale_l2 "$SCALE_L2" --adapter_rms_l2 "$ADAPTER_RMS_L2" --max_grad_norm "$MAX_GRAD_NORM"
  --max_bytes "$BYTE_MAX" --max_answer_tokens "$MAX_ANSWER_TOKENS"
  --first_token_ce_weight "$FIRST_TOKEN_CE"
  --train_append_bos_after_prefix "$TRAIN_APPEND_BOS"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --adapter_hidden_mult "$ADAPTER_HIDDEN_MULT"
  --manifold_stat_weight "$MANIFOLD_STAT_WEIGHT"
  --state_kd_weight "$STATE_KD_WEIGHT" --state_kd_layers "$STATE_KD_LAYERS"
  --K "$K" --k_ce_weight "$K_CE_WEIGHT" --kd_first_k_weight "$KD_FIRST_K_WEIGHT" --kd_tau "$KD_TAU"
  --llama_device_map "$LLAMA_DEVICE_MAP" --qwen_device_map "$QWEN_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES" --qwen_devices "$QWEN_DEVICES" --gpu_mem_gib "$GPU_MEM_GIB"
)

EVAL_ARGS_COMMON=(
  --dataset "$DATASET" --max_new_tokens "$MAX_NEW_TOKENS"
  --latent_anchor_mode "$LATENT_ANCHOR_MODE" --latent_anchor_text "$LATENT_ANCHOR_TEXT"
  --append_bos_after_prefix "$APPEND_BOS_AFTER_PREFIX"
  --calibration "$CALIBRATION" --prefix_gain "$PREFIX_GAIN"
  --token_budget_mode "$TOKEN_BUDGET_MODE" --token_budget_k "$TOKEN_BUDGET_K"
  --first_token_top_p "$FIRST_TOKEN_TOP_P" --first_token_temperature "$FIRST_TOKEN_TEMPERATURE"
  --latent_quant_bits 6 --latent_quant_group_size 32 --latent_quant_scale_bits 16
  --min_new_tokens 3 --eos_ban_steps 6 --chunk_size "$CHUNK_SIZE"
  --sequential_eval
  --llama_device_map "$LLAMA_DEVICE_MAP" --qwen_device_map "$QWEN_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES" --qwen_devices "$QWEN_DEVICES" --gpu_mem_gib "$GPU_MEM_GIB"
)

RUN="8B_interlingua_reco"
RUN_DIR="runs/${RUN}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RUN_DIR}/pipeline_${ts}.log"

print_header() {
  echo ""; echo "========================================="; echo "$1"; echo "========================================="; echo ""
}

run_eval() {
  local ckpt_path="$1"; local out_dir="$2"; local n_samples="$3"
  mkdir -p "$out_dir"
  EVAL_ARGS=( --ckpt "$ckpt_path" --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" --samples "$n_samples" --out_dir "$out_dir" "${EVAL_ARGS_COMMON[@]}" )
  if [[ $FRESH_EVAL -eq 1 ]]; then EVAL_ARGS+=(--fresh_eval); fi
  if [[ $LOAD_4BIT -eq 1 ]]; then EVAL_ARGS+=(--load_4bit); fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py "${EVAL_ARGS[@]}"
}

{
  print_header "Starting pipeline at $(date)"
  if [[ $DO_TRAIN -eq 1 ]]; then
    print_header "TRAIN + PER-EPOCH EVAL"
    for epoch in $(seq 1 $EPOCHS); do
      print_header "EPOCH $epoch/$EPOCHS"
      TRAIN_ARGS=( "${TRAIN_ARGS_COMMON[@]}" --save_dir "$CKPT_DIR" --save_every 1000000 --save_training_stats --debug --auto_resume )
      TRAIN_ARGS+=(--grad_ckpt)
      if [[ -n "${WARM_ANCHOR_TEXT}" ]]; then TRAIN_ARGS+=(--warm_anchor_text "$WARM_ANCHOR_TEXT"); fi
      if [[ $LOAD_4BIT -eq 1 ]]; then TRAIN_ARGS+=(--load_4bit); fi
      if [[ "$ADAPTER_COLORIZE" == "1" ]]; then TRAIN_ARGS+=(--adapter_colorize); fi
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py "${TRAIN_ARGS[@]}"
      epoch_ckpt="${RUN_DIR}/epoch${epoch}"; rm -rf "$epoch_ckpt"; cp -r "$CKPT_DIR" "$epoch_ckpt"
      run_eval "$epoch_ckpt" "${RUN_DIR}/eval_epoch${epoch}" "$SMOKE_SAMPLES"
    done
  fi
  if [[ $DO_FINAL_EVAL -eq 1 ]]; then
    print_header "FINAL FULL EVAL ON LAST CKPT"
    run_eval "${RUN_DIR}/epoch${EPOCHS}" "${RUN_DIR}/eval_final" "$SAMPLES"
  fi
} 2>&1 | tee "$LOG_FILE"

echo ""; echo "All output saved to: $LOG_FILE"
