#!/usr/bin/env bash
# run_reverse_direction.sh
#
# Reverse direction experiment: Mistral → Llama
# Tests bidirectional communication capability
#
# Usage: PYTHONPATH=.. bash run_reverse_direction.sh

set -e

OUTPUT_BASE="${OUTPUT_DIR:-runs/reverse_direction_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="$OUTPUT_BASE/reverse_direction.log"

echo "=============================================="
echo "REVERSE DIRECTION: Mistral → Llama"
echo "=============================================="
echo "Output: $OUTPUT_BASE"
echo "Log: $LOG_FILE"
echo ""

{
    echo "Starting reverse direction experiments at $(date)"
    echo ""

    # Seeds for reproducibility
    SEEDS=(42 123 456)

    echo "=============================================="
    echo "SST-2 Reverse: Mistral → Llama (3 seeds)"
    echo "=============================================="

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Reverse Bridge: sst2 (seed=$SEED)"

        python train_telepathy_sst2.py \
            --source_model "mistralai/Mistral-7B-Instruct-v0.3" \
            --target_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --source_layer 16 \
            --soft_tokens 8 \
            --steps 2000 \
            --batch_size 16 \
            --grad_accum 2 \
            --lr 2e-4 \
            --eval_every 400 \
            --save_every 500 \
            --diversity_weight 0.1 \
            --output_dir "$OUTPUT_BASE/reverse_sst2_seed${SEED}" \
            --save_path "$OUTPUT_BASE/reverse_sst2_seed${SEED}/bridge_reverse_sst2_seed${SEED}.pt" \
            --seed "$SEED" \
            --bf16

        echo ">>> Done: reverse sst2 seed=$SEED"
    done

    echo ""
    echo "=============================================="
    echo "REVERSE DIRECTION COMPLETE"
    echo "=============================================="
    echo ""
    echo "Results saved to: $OUTPUT_BASE"
    echo "Finished at $(date)"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "All experiments complete!"
echo "Log file: $LOG_FILE"
