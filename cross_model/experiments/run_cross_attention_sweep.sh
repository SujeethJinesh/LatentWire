#!/usr/bin/env bash
set -e

# Hyperparameter sweep for overnight run on 4x H100
# Expected runtime: 8-12 hours total
# Each experiment: ~2-3 hours for 3000 steps

echo "Starting cross-attention hyperparameter sweep..."
echo "Timestamp: $(date)"
echo "This sweep will test multiple configurations overnight"
echo ""

# Base configuration (constant across all runs)
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TRANSLATOR_TYPE="bottleneck_gated"
PER_DEVICE_BATCH=10  # Effective batch = 40 with 4 GPUs
TRAIN_STEPS=3000  # ~16 epochs, good balance
EVAL_EVERY=300
EVAL_SAMPLES=200
MAX_NEW_TOKENS=256
SEED=1234

# Create main output directory
MAIN_OUTPUT_DIR="runs/cross_attention_sweep_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$MAIN_OUTPUT_DIR"

# Log file for the entire sweep
SWEEP_LOG="$MAIN_OUTPUT_DIR/sweep_summary.log"

echo "=== CROSS-ATTENTION HYPERPARAMETER SWEEP ===" | tee "$SWEEP_LOG"
echo "Main output directory: $MAIN_OUTPUT_DIR" | tee -a "$SWEEP_LOG"
echo "Start time: $(date)" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local lr=$2
    local bottleneck=$3
    local tokens=$4
    local depth=$5
    local wd=$6
    local warmup=$7

    echo "----------------------------------------" | tee -a "$SWEEP_LOG"
    echo "Starting experiment: $exp_name" | tee -a "$SWEEP_LOG"
    echo "  LR=$lr, Bottleneck=$bottleneck, Tokens=$tokens, Depth=$depth, WD=$wd" | tee -a "$SWEEP_LOG"
    echo "  Time: $(date)" | tee -a "$SWEEP_LOG"

    OUTPUT_DIR="$MAIN_OUTPUT_DIR/$exp_name"
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/training.log"

    # Run the training
    {
        torchrun --nproc_per_node=4 cross_model/experiments/cross_attention.py \
            --source_model "$SOURCE_MODEL" \
            --target_model "$TARGET_MODEL" \
            --translator_type "$TRANSLATOR_TYPE" \
            --bottleneck_dim "$bottleneck" \
            --soft_tokens "$tokens" \
            --depth "$depth" \
            --heads 16 \
            --lr "$lr" \
            --weight_decay "$wd" \
            --train_steps "$TRAIN_STEPS" \
            --warmup_steps "$warmup" \
            --per_device_batch "$PER_DEVICE_BATCH" \
            --eval_every "$EVAL_EVERY" \
            --eval_samples "$EVAL_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --seed "$SEED" \
            --bf16 \
            --save_path "$OUTPUT_DIR/translator_checkpoint.pt"
    } 2>&1 | tee "$LOG_FILE"

    # Extract final metrics
    echo "Completed: $exp_name at $(date)" | tee -a "$SWEEP_LOG"
    tail -n 5 "$LOG_FILE" | grep "Final" | tee -a "$SWEEP_LOG"
    echo "" | tee -a "$SWEEP_LOG"
}

# ============================================
# SWEEP CONFIGURATION
# ============================================

# Experiment 1: Baseline (best settings from previous runs)
run_experiment \
    "1_baseline_lr1e4_b1024_t48_d6" \
    1e-4 \
    1024 \
    48 \
    6 \
    0.01 \
    600  # 20% warmup

# Experiment 2: Lower LR for stability
run_experiment \
    "2_lower_lr_5e5_b1024_t48_d6" \
    5e-5 \
    1024 \
    48 \
    6 \
    0.01 \
    600

# Experiment 3: More soft tokens
run_experiment \
    "3_more_tokens_lr1e4_b1024_t64_d6" \
    1e-4 \
    1024 \
    64 \
    6 \
    0.01 \
    600

# Experiment 4: Smaller bottleneck (faster, might be enough)
run_experiment \
    "4_smaller_bottleneck_lr1e4_b768_t48_d6" \
    1e-4 \
    768 \
    48 \
    6 \
    0.01 \
    600

# Experiment 5: Shallower but wider
run_experiment \
    "5_shallower_lr1e4_b1536_t48_d4" \
    1e-4 \
    1536 \
    48 \
    4 \
    0.01 \
    600

# Experiment 6: Higher weight decay (BLIP-2 uses 0.05)
run_experiment \
    "6_higher_wd_lr1e4_b1024_t48_d6" \
    1e-4 \
    1024 \
    48 \
    6 \
    0.05 \
    600

# ============================================
# ANALYSIS SECTION
# ============================================

echo "============================================" | tee -a "$SWEEP_LOG"
echo "SWEEP COMPLETE!" | tee -a "$SWEEP_LOG"
echo "End time: $(date)" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

# Collect all final results
echo "FINAL RESULTS SUMMARY:" | tee -a "$SWEEP_LOG"
echo "----------------------------------------" | tee -a "$SWEEP_LOG"

for exp_dir in "$MAIN_OUTPUT_DIR"/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        log_file="$exp_dir/training.log"
        if [ -f "$log_file" ]; then
            echo "$exp_name:" | tee -a "$SWEEP_LOG"
            grep -E "Final.*Target-alone.*Bridged" "$log_file" | tail -1 | tee -a "$SWEEP_LOG"
            echo "" | tee -a "$SWEEP_LOG"
        fi
    fi
done

echo "----------------------------------------" | tee -a "$SWEEP_LOG"
echo "All results saved to: $MAIN_OUTPUT_DIR" | tee -a "$SWEEP_LOG"
echo "Summary log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"

# Create a simple CSV for easy analysis
echo "Creating results CSV..." | tee -a "$SWEEP_LOG"
CSV_FILE="$MAIN_OUTPUT_DIR/results.csv"
echo "experiment,lr,bottleneck,tokens,depth,wd,target_acc,bridged_acc" > "$CSV_FILE"

for exp_dir in "$MAIN_OUTPUT_DIR"/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        log_file="$exp_dir/training.log"
        if [ -f "$log_file" ]; then
            # Extract hyperparameters from experiment name
            # Format: N_description_lrX_bX_tX_dX
            final_line=$(grep -E "Final.*Target-alone.*Bridged" "$log_file" | tail -1)
            target_acc=$(echo "$final_line" | grep -oE "Target-alone acc: [0-9.]+" | grep -oE "[0-9.]+$")
            bridged_acc=$(echo "$final_line" | grep -oE "Bridged acc: [0-9.]+" | grep -oE "[0-9.]+$")

            # Parse experiment name for parameters (simplified)
            echo "$exp_name,,,,,,$target_acc,$bridged_acc" >> "$CSV_FILE"
        fi
    fi
done

echo "CSV results saved to: $CSV_FILE" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"
echo "To monitor progress: tail -f $SWEEP_LOG" | tee -a "$SWEEP_LOG"