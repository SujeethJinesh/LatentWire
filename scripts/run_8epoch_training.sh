#!/usr/bin/env bash
set -e

# =============================================================================
# 8-Epoch Training with Per-Epoch Evaluation
# =============================================================================
# This script runs 8 epochs of training with full evaluation after each epoch
# to track model progress and performance over time.
#
# Features:
# - Trains for 8 epochs total
# - Evaluates on validation set after each epoch
# - Saves checkpoints and metrics for each epoch
# - Generates plots showing training progression
# - Suitable for paper results and performance tracking
# =============================================================================

# Configuration
EXPERIMENT_NAME="8epoch_pereval"
OUTPUT_DIR="${OUTPUT_DIR:-runs/$EXPERIMENT_NAME}"
EPOCHS=8
SAMPLES=10000  # Adjust based on your dataset size
EVAL_SAMPLES=500  # Validation samples per epoch

# Model configuration
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"
DATASET="squad"

# Hyperparameters
BATCH_SIZE=64
LATENT_LEN=32
D_Z=256
LR=3e-4
FIRST_TOKEN_CE_WEIGHT=0.5
K_TOKENS=4

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/epoch_evals"
mkdir -p "$OUTPUT_DIR/figures"

# Create timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"

echo "=============================================================="
echo "Starting 8-Epoch Training with Per-Epoch Evaluation"
echo "=============================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Models: Llama 3.1 8B + Qwen 2.5 7B"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Samples: $SAMPLES train / $EVAL_SAMPLES eval per epoch"
echo "=============================================================="

# Function to run evaluation for a specific epoch
run_epoch_eval() {
    local EPOCH=$1
    local CHECKPOINT_DIR="$OUTPUT_DIR/epoch$EPOCH"
    local EVAL_OUTPUT="$OUTPUT_DIR/epoch_evals/epoch${EPOCH}_eval.json"

    echo ""
    echo "Running evaluation for epoch $EPOCH..."

    python latentwire/eval.py \
        --ckpt "$CHECKPOINT_DIR" \
        --samples "$EVAL_SAMPLES" \
        --max_new_tokens 12 \
        --dataset "$DATASET" \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        --output_file "$EVAL_OUTPUT" \
        --compute_statistical_tests

    # Extract key metrics for tracking
    if [ -f "$EVAL_OUTPUT" ]; then
        echo "Epoch $EPOCH evaluation complete. Key metrics:"
        python -c "
import json
with open('$EVAL_OUTPUT', 'r') as f:
    data = json.load(f)
    if 'aggregate_stats' in data:
        stats = data['aggregate_stats']
        print(f'  Latent F1: {stats.get(\"latent_f1_mean\", 0):.4f}')
        print(f'  Text F1: {stats.get(\"text_f1_mean\", 0):.4f}')
        print(f'  First Token Acc: {stats.get(\"first_tok_acc\", 0):.4f}')
        print(f'  NLL/token: {stats.get(\"latent_nll_mean\", 0):.4f}')
"
    fi
}

# Main training loop with per-epoch checkpointing
{
    echo "Starting training at $(date)"
    echo ""

    # Run training with explicit epoch-by-epoch checkpointing
    for EPOCH in $(seq 1 $EPOCHS); do
        echo "=============================================================="
        echo "Training Epoch $EPOCH/$EPOCHS"
        echo "=============================================================="

        # Set checkpoint directory for this epoch
        EPOCH_CHECKPOINT="$OUTPUT_DIR/epoch$EPOCH"

        # Training command for single epoch
        python latentwire/train.py \
            --llama_id "$LLAMA_ID" \
            --qwen_id "$QWEN_ID" \
            --samples "$SAMPLES" \
            --epochs 1 \
            --initial_epoch $((EPOCH - 1)) \
            --batch_size "$BATCH_SIZE" \
            --latent_len "$LATENT_LEN" \
            --d_z "$D_Z" \
            --lr "$LR" \
            --encoder_type byte \
            --dataset "$DATASET" \
            --sequential_models \
            --warm_anchor_text "Answer: " \
            --first_token_ce_weight "$FIRST_TOKEN_CE_WEIGHT" \
            --k_token_ce_from_prefix "$K_TOKENS" \
            --output_dir "$EPOCH_CHECKPOINT" \
            $(if [ $EPOCH -gt 1 ]; then echo "--resume_from $OUTPUT_DIR/epoch$((EPOCH - 1))"; fi) \
            --save_every_n_steps 500 \
            --log_every 50

        # Run evaluation for this epoch
        run_epoch_eval "$EPOCH"

        # Save training metrics snapshot
        if [ -f "$EPOCH_CHECKPOINT/diagnostics.jsonl" ]; then
            tail -1 "$EPOCH_CHECKPOINT/diagnostics.jsonl" > "$OUTPUT_DIR/epoch_evals/epoch${EPOCH}_train_metrics.json"
        fi

        echo "Epoch $EPOCH complete at $(date)"
        echo ""
    done

    echo "=============================================================="
    echo "All epochs complete! Generating summary plots..."
    echo "=============================================================="

    # Generate training progression plots
    python -c "
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
epoch_dir = output_dir / 'epoch_evals'
fig_dir = output_dir / 'figures'

# Collect metrics across epochs
epochs = []
latent_f1 = []
text_f1 = []
first_tok_acc = []
latent_nll = []

for epoch in range(1, $EPOCHS + 1):
    eval_file = epoch_dir / f'epoch{epoch}_eval.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            data = json.load(f)
            if 'aggregate_stats' in data:
                stats = data['aggregate_stats']
                epochs.append(epoch)
                latent_f1.append(stats.get('latent_f1_mean', 0))
                text_f1.append(stats.get('text_f1_mean', 0))
                first_tok_acc.append(stats.get('first_tok_acc', 0))
                latent_nll.append(stats.get('latent_nll_mean', 0))

if epochs:
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # F1 Score progression
    axes[0, 0].plot(epochs, latent_f1, 'b-o', label='Latent')
    axes[0, 0].plot(epochs, text_f1, 'g--s', label='Text Baseline')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score Progression')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # First Token Accuracy
    axes[0, 1].plot(epochs, first_tok_acc, 'r-o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('First Token Accuracy')
    axes[0, 1].set_title('First Token Accuracy Progression')
    axes[0, 1].grid(True, alpha=0.3)

    # NLL/token
    axes[1, 0].plot(epochs, latent_nll, 'm-o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('NLL per Token')
    axes[1, 0].set_title('Negative Log Likelihood Progression')
    axes[1, 0].grid(True, alpha=0.3)

    # Relative improvement
    if latent_f1[0] > 0:
        relative_improvement = [(f / latent_f1[0] - 1) * 100 for f in latent_f1]
        axes[1, 1].plot(epochs, relative_improvement, 'c-o')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Relative Improvement (%)')
        axes[1, 1].set_title('Relative F1 Improvement from Epoch 1')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('8-Epoch Training Progression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'training_progression.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'training_progression.pdf', bbox_inches='tight')

    print(f'Training progression plots saved to {fig_dir}')

    # Save summary statistics
    summary = {
        'experiment_name': '$EXPERIMENT_NAME',
        'total_epochs': $EPOCHS,
        'final_metrics': {
            'latent_f1': latent_f1[-1] if latent_f1 else 0,
            'text_f1': text_f1[-1] if text_f1 else 0,
            'first_tok_acc': first_tok_acc[-1] if first_tok_acc else 0,
            'latent_nll': latent_nll[-1] if latent_nll else 0,
        },
        'improvement': {
            'f1_absolute': (latent_f1[-1] - latent_f1[0]) if len(latent_f1) > 1 else 0,
            'f1_relative_pct': ((latent_f1[-1] / latent_f1[0] - 1) * 100) if len(latent_f1) > 1 and latent_f1[0] > 0 else 0,
        },
        'epoch_metrics': {
            'epochs': epochs,
            'latent_f1': latent_f1,
            'text_f1': text_f1,
            'first_tok_acc': first_tok_acc,
            'latent_nll': latent_nll,
        }
    }

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Training summary saved to {output_dir}/training_summary.json')
"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "8-Epoch Training Complete!"
echo "=============================================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - Checkpoints: $OUTPUT_DIR/epoch{1-$EPOCHS}/"
echo "  - Evaluations: $OUTPUT_DIR/epoch_evals/"
echo "  - Plots: $OUTPUT_DIR/figures/"
echo "  - Summary: $OUTPUT_DIR/training_summary.json"
echo "  - Full log: $LOG_FILE"
echo "=============================================================="