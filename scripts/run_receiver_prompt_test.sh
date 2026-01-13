#!/usr/bin/env bash
set -e

# =============================================================================
# Receiver-Side Prompt Engineering - Quick Local Test
# =============================================================================
# This script tests the receiver prompt engineering feature locally.
# For full ablation on HPC, use: sbatch telepathy/submit_receiver_prompt_ablation.slurm
#
# Usage:
#   bash scripts/run_receiver_prompt_test.sh [CHECKPOINT_DIR] [PRESET]
#
# Examples:
#   bash scripts/run_receiver_prompt_test.sh runs/my_run/best cot
#   bash scripts/run_receiver_prompt_test.sh runs/my_run/epoch10 none
# =============================================================================

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Parse arguments
CKPT_DIR="${1:-}"
PRESET="${2:-cot}"

if [ -z "$CKPT_DIR" ]; then
    echo "Usage: $0 CHECKPOINT_DIR [PRESET]"
    echo ""
    echo "PRESET options:"
    echo "  none       - Baseline: just 'Answer: '"
    echo "  cot        - 'Let's think step by step. Answer: '"
    echo "  cot_concise - 'Think carefully. Answer: '"
    echo "  direct     - 'Answer directly: Answer: '"
    echo "  careful    - 'Be precise. Answer: '"
    echo "  json       - 'Respond in JSON format. Answer: '"
    echo ""
    echo "Example: $0 runs/my_run/best cot"
    exit 1
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

OUTPUT_DIR="runs/receiver_prompt_test"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/test_${PRESET}_${TIMESTAMP}.log"

echo "=============================================================="
echo "RECEIVER PROMPT TEST"
echo "=============================================================="
echo "Checkpoint: $CKPT_DIR"
echo "Preset: $PRESET"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "=============================================================="

{
    python latentwire/eval.py \
        --ckpt "$CKPT_DIR" \
        --dataset squad \
        --samples 50 \
        --max_new_tokens 12 \
        --receiver_prompt_preset "$PRESET" \
        --models llama \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        --out_dir "$OUTPUT_DIR" \
        --debug \
        --debug_print_first 5

    echo ""
    echo "Test complete! Check $OUTPUT_DIR for results."
} 2>&1 | tee "$LOG_FILE"
