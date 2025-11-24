#!/usr/bin/env bash
set -euo pipefail

# Phase 1 Llama->Llama layer grid (soft-only)
# - 3x3 learned combos over source_layer {0,16,31} and target_layer {0,16,31}
# - Diagonal combos run passthrough (no training, train_steps=0)
# - Off-diagonals train 1500 steps, early_stop_patience=2

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

SOURCE_LAYERS=(0 16 31)
TARGET_LAYERS=(0 16 31)

for S in "${SOURCE_LAYERS[@]}"; do
    for T in "${TARGET_LAYERS[@]}"; do
        PASSTHROUGH=0
        TRAIN_STEPS=1500
        EARLY_STOP=2
        LABEL="layer_s${S}_t${T}"
        if [[ "$S" -eq "$T" ]]; then
            PASSTHROUGH=1
            TRAIN_STEPS=0
            EARLY_STOP=0
            LABEL="layer_s${S}_t${T}_passthrough"
        fi

        echo "=========================================="
        echo "Launching Llama->Llama layer grid:"
        echo "  Label:        $LABEL"
        echo "  Source layer: $S"
        echo "  Target layer: $T"
        echo "  Passthrough:  $PASSTHROUGH"
        echo "  Train steps:  $TRAIN_STEPS"
        echo "  Early stop:   $EARLY_STOP"
        echo "  GPUs:         $NPROC"
        echo "=========================================="

    RUN_ID="${LABEL}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="paper_writing/runs/${RUN_ID}"
        mkdir -p "$OUTPUT_DIR"
        SUMMARY_LOG="$OUTPUT_DIR/summary.log"

        RANDOM_PORT=$((29500 + RANDOM % 1000))

        EXTRA_ARGS="--source_layer ${S} --target_layer ${T} --eval_prompt_mode soft_only --soft_tokens 128 --prompt_alignment_weight 0.001 --token_alignment_weight 0.0"
        if [[ "$PASSTHROUGH" -eq 1 ]]; then
            EXTRA_ARGS="$EXTRA_ARGS --passthrough_soft"
        fi

        {
        torchrun --standalone --nproc_per_node="$NPROC" --master_port "$RANDOM_PORT" paper_writing/cross_attention.py \
            --source_model meta-llama/Meta-Llama-3.1-8B-Instruct \
            --target_model meta-llama/Meta-Llama-3.1-8B-Instruct \
            --per_device_batch 2 \
            --eval_every 250 \
            --eval_samples 100 \
            --eval_batch_size 36 \
            --max_new_tokens 256 \
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
            --lr 1e-4 \
            --weight_decay 0.01 \
            --seed 1234 \
            --bf16 \
            --no_compile \
            --prompt_alignment_weight 0.001 \
            --token_alignment_weight 0.0 \
            --skip_source_baseline \
            --save_path "$OUTPUT_DIR/checkpoint.pt" \
            --log_dir "$OUTPUT_DIR" \
            $EXTRA_ARGS
        } 2>&1 | tee "$OUTPUT_DIR/train.log"

        echo "End time: $(date)" | tee "$SUMMARY_LOG"
        echo "Logs: $OUTPUT_DIR/train.log" | tee -a "$SUMMARY_LOG"

        preserved="paper_writing/preserved_data/${RUN_ID}"
        mkdir -p "paper_writing/preserved_data"
        cp -R "$OUTPUT_DIR" "$preserved"
        echo "Saved run to: $OUTPUT_DIR"
        echo "Copied artifacts to: $preserved"
    done
done
