#!/usr/bin/env bash
# ============================================================================
# Feature Sweep: Test 4 Supervision Mechanisms Individually
# ============================================================================
#
# Tests each of the 4 new supervision features individually (not combined)
# to determine which mechanisms help fix mode collapse before combining them.
#
# Features tested:
# 1. Baseline (current setup)
# 2. Contrastive diversity loss
# 3. K-token cross-entropy
# 4. Reconstruction loss
# 5. Knowledge distillation
#
# Run with: bash scripts/run_feature_sweep.sh
# ============================================================================

set -e

# Configuration
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Common parameters
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET="squad"
SAMPLES=10000
EVAL_SAMPLES=100
EPOCHS=3
BATCH_SIZE=48
LR=5e-4
MAX_LENGTH=512
TARGET_SEQ=256
SOURCE_LENGTH=300
SHOW_SAMPLES=15

# LoRA settings
USE_LORA="--use_lora"
LORA_R=16
LORA_ALPHA=32
LORA_LAYERS=8

# Output directory
OUTPUT_BASE="runs/feature_sweep"
mkdir -p "$OUTPUT_BASE"

# Start time
START_TIME=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$OUTPUT_BASE/sweep_summary_${START_TIME}.txt"

echo "Feature Sweep: Testing 4 Supervision Mechanisms"
echo "Started: $(date)"
echo "================================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Training samples: $SAMPLES"
echo "Eval samples: $EVAL_SAMPLES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Sequence: $TARGET_SEQ (compression: $(python3 -c "print(f'{$SOURCE_LENGTH/$TARGET_SEQ:.2f}')") x)"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA, layers=$LORA_LAYERS"
echo "================================================"
echo ""

# Save header to summary file
{
  echo "Feature Sweep Summary"
  echo "Started: $(date)"
  echo "================================================"
  echo "Model: $MODEL"
  echo "Dataset: $DATASET"
  echo "Training samples: $SAMPLES"
  echo "Eval samples: $EVAL_SAMPLES"
  echo "Epochs: $EPOCHS"
  echo "Batch size: $BATCH_SIZE"
  echo "Learning rate: $LR"
  echo "================================================"
  echo ""
} > "$SUMMARY_FILE"

# Function to run a single experiment
run_experiment() {
    local config_name="$1"
    local output_dir="$OUTPUT_BASE/$config_name"
    shift
    local extra_args="$@"

    echo "================================================================"
    echo "Running: $config_name"
    echo "Output: $output_dir"
    echo "Args: $extra_args"
    echo "================================================================"

    mkdir -p "$output_dir"
    LOG_FILE="$output_dir/train_${START_TIME}.log"
    DIAGNOSTIC_LOG="$output_dir/diagnostics.jsonl"

    # Run training with tee for logging
    {
        python3 train_sequence_compression_enhanced.py \
            --model_id "$MODEL" \
            --dataset "$DATASET" \
            --samples "$SAMPLES" \
            --eval_samples "$EVAL_SAMPLES" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --max_length "$MAX_LENGTH" \
            --target_sequence_length "$TARGET_SEQ" \
            --source_length "$SOURCE_LENGTH" \
            --show_samples "$SHOW_SAMPLES" \
            $USE_LORA \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --lora_layers "$LORA_LAYERS" \
            --save_dir "$output_dir" \
            --diagnostic_log "$DIAGNOSTIC_LOG" \
            $extra_args
    } 2>&1 | tee "$LOG_FILE"

    # Extract final metrics from diagnostics
    if [ -f "$DIAGNOSTIC_LOG" ]; then
        FINAL_METRICS=$(python3 -c "
import json
import sys

try:
    with open('$DIAGNOSTIC_LOG') as f:
        lines = [json.loads(line) for line in f]

    # Get final eval metrics
    eval_lines = [l for l in lines if l.get('type') == 'full_eval']
    if eval_lines:
        last_eval = eval_lines[-1]
        f1 = last_eval.get('f1', 0.0) * 100
        em = last_eval.get('em', 0.0) * 100
        diversity = last_eval.get('diversity', 0.0) * 100
        print(f'F1={f1:.2f}%, EM={em:.2f}%, Diversity={diversity:.0f}%')
    else:
        print('No eval metrics found')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
")
    else
        FINAL_METRICS="No diagnostics file"
    fi

    echo ""
    echo "Configuration: $config_name"
    echo "  Metrics: $FINAL_METRICS"
    echo "  Log: $LOG_FILE"
    echo ""

    # Append to summary
    {
        echo ""
        echo "Configuration: $config_name"
        if [ -n "$extra_args" ]; then
            echo "  Args: $extra_args"
        else
            echo "  Args: (baseline - no extra features)"
        fi
        echo "  $FINAL_METRICS"
        echo "  Log: $LOG_FILE"
    } >> "$SUMMARY_FILE"
}

# ============================================================================
# BASELINE (No additional features)
# ============================================================================

echo ""
echo "###################################################################"
echo "# BASELINE (1 config)"
echo "###################################################################"
echo ""

run_experiment "baseline" ""

# ============================================================================
# FEATURE 1: Contrastive Diversity Loss
# ============================================================================

echo ""
echo "###################################################################"
echo "# FEATURE 1: Contrastive Diversity Loss (6 configs)"
echo "###################################################################"
echo ""

for weight in 0.1 0.3 0.5; do
    for temp in 0.07 0.1; do
        config_name="contrastive_w${weight}_t${temp}"
        run_experiment "$config_name" \
            "--use_contrastive" \
            "--contrastive_weight $weight" \
            "--contrastive_temp $temp"
    done
done

# ============================================================================
# FEATURE 2: K-token Cross-Entropy
# ============================================================================

echo ""
echo "###################################################################"
echo "# FEATURE 2: K-token Cross-Entropy (3 configs)"
echo "###################################################################"
echo ""

for K in 2 4 8; do
    config_name="k_token_k${K}"
    run_experiment "$config_name" \
        "--use_k_token_ce" \
        "--k_token_k $K"
done

# ============================================================================
# FEATURE 3: Reconstruction Loss
# ============================================================================

echo ""
echo "###################################################################"
echo "# FEATURE 3: Reconstruction Loss (6 configs)"
echo "###################################################################"
echo ""

for weight in 0.01 0.05 0.1 0.2; do
    for layers in 2 4; do
        config_name="reconstruction_w${weight}_l${layers}"
        run_experiment "$config_name" \
            "--use_reconstruction" \
            "--reconstruction_weight $weight" \
            "--reconstruction_layers $layers"
    done
done

# Reduced to 8 configs (4 weights × 2 layer options)

# ============================================================================
# FEATURE 4: Knowledge Distillation
# ============================================================================

echo ""
echo "###################################################################"
echo "# FEATURE 4: Knowledge Distillation (6 configs)"
echo "###################################################################"
echo ""

for weight in 0.1 0.3 0.5; do
    for tau in 1.0 2.0; do
        config_name="kd_w${weight}_tau${tau}"
        run_experiment "$config_name" \
            "--use_kd" \
            "--kd_weight $weight" \
            "--kd_tau $tau" \
            "--kd_k 4"
    done
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================"
echo "SWEEP COMPLETE"
echo "================================================================"
echo "Ended: $(date)"
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "To view summary:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "To analyze diagnostics:"
echo "  python3 scripts/analyze_feature_sweep.py $OUTPUT_BASE"
echo ""

# Print summary to console
cat "$SUMMARY_FILE"
