#!/usr/bin/env bash
set -euo pipefail

# Phase 1 best-lock runs (128 tokens, answer teacher)
# Purpose: retain the 75% peak by saving best checkpoint via early_stop_patience>0
# Configs:
#   - lr=8e-5 (hits 75% peak)
#   - lr=1e-4 (baseline) to see if best-lock prevents late drift

if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75
fi

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

detect_nproc() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        echo "$NUM_GPUS"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            echo "$count"
            return
        fi
    fi
    python - <<'PY' 2>/dev/null || echo 1
import torch
print(torch.cuda.device_count() or 1)
PY
}

NPROC=$(detect_nproc)
TRAIN_STEPS=${TRAIN_STEPS:-2000}
EARLY_STOP=${EARLY_STOP:-2}

declare -a CONFIGS=(
    "label=phase1_128tok_lr8e5_best LR=8e-5"
    "label=phase1_128tok_lr1e4_best LR=1e-4"
)

for entry in "${CONFIGS[@]}"; do
    eval "$entry"
    echo "=========================================="
    echo "Launching Phase1 config: $label"
    echo "  Soft tokens: 128"
    echo "  LR:          $LR"
    echo "  Train steps: $TRAIN_STEPS"
    echo "  Early stop:  $EARLY_STOP (best checkpoint saved/reloaded)"
    echo "  GPUs:        $NPROC"
    echo "=========================================="

    RUN_ID="${label}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="paper_writing/runs/${RUN_ID}"
    mkdir -p "$OUTPUT_DIR"
    SUMMARY_LOG="$OUTPUT_DIR/summary.log"

    RANDOM_PORT=$((29500 + RANDOM % 1000))

    {
    torchrun --standalone --nproc_per_node="$NPROC" --master_port "$RANDOM_PORT" paper_writing/cross_attention.py \
        --source_model mistralai/Mistral-7B-Instruct-v0.3 \
        --target_model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --per_device_batch 2 \
        --eval_every 250 \
        --eval_samples 200 \
        --eval_batch_size 36 \
        --max_new_tokens 256 \
        --eval_prompt_mode soft_plus_text \
        --bridge dit \
        --soft_tokens 128 \
        --dit_dim 512 \
        --dit_depth 6 \
        --dit_heads 8 \
        --dit_steps_train 4 \
        --dit_steps_eval 8 \
        --dit_dropout 0.1 \
        --dit_pool mean \
        --dit_loss_weight 0.1 \
        --info_nce_weight 0.05 \
        --dit_teacher answer \
        --train_steps "$TRAIN_STEPS" \
        --warmup_steps 200 \
        --early_stop_patience "$EARLY_STOP" \
        --lr "$LR" \
        --weight_decay 0.01 \
        --seed 1234 \
        --bf16 \
        --no_compile \
        --prompt_alignment_weight 0.001 \
        --token_alignment_weight 0.0 \
        --save_path "$OUTPUT_DIR/checkpoint.pt" \
        --log_dir "$OUTPUT_DIR"
    } 2>&1 | tee "$OUTPUT_DIR/train.log"

    echo "End time: $(date)" | tee "$SUMMARY_LOG"
    echo "Logs: $OUTPUT_DIR/train.log" | tee -a "$SUMMARY_LOG"

    preserved="paper_writing/preserved_data/${RUN_ID}"
    mkdir -p "paper_writing/preserved_data"
    cp -R "$OUTPUT_DIR" "$preserved"
    echo "Saved run to: $OUTPUT_DIR"
    echo "Copied artifacts to: $preserved"
done
