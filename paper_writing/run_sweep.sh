#!/usr/bin/env bash
set -e

# Focused sweep for overnight run - 4 key experiments
# Each ~2-3 hours, total 8-12 hours on 4x H100
# Tests the most critical hyperparameters based on previous results

echo "Starting FOCUSED cross-attention sweep (4 experiments)"
echo "Expected runtime: 8-12 hours total"
echo "Timestamp: $(date)"
echo ""

# Base configuration
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PER_DEVICE_BATCH=10
EVAL_EVERY=250
EVAL_SAMPLES=500  # Reduced from 1000 to avoid OOM; still 2.5× better than original 200
MAX_NEW_TOKENS=256

# Create output directory
SWEEP_DIR="runs/focused_sweep_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$SWEEP_DIR"
SUMMARY_LOG="$SWEEP_DIR/summary.log"

echo "=== FOCUSED HYPERPARAMETER SWEEP ===" | tee "$SUMMARY_LOG"
echo "Output: $SWEEP_DIR" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

run_config() {
    local name=$1
    local desc=$2
    shift 2  # Remove first two args, rest are for the script

    echo "========================================" | tee -a "$SUMMARY_LOG"
    echo "Experiment: $name" | tee -a "$SUMMARY_LOG"
    echo "Description: $desc" | tee -a "$SUMMARY_LOG"
    echo "Start: $(date)" | tee -a "$SUMMARY_LOG"
    echo "----------------------------------------" | tee -a "$SUMMARY_LOG"

    OUTPUT_DIR="$SWEEP_DIR/$name"
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/train.log"

    # Run with all remaining arguments
    {
        torchrun --nproc_per_node=4 cross_model/experiments/cross_attention.py \
            --source_model "$SOURCE_MODEL" \
            --target_model "$TARGET_MODEL" \
            --per_device_batch "$PER_DEVICE_BATCH" \
            --eval_every "$EVAL_EVERY" \
            --eval_samples "$EVAL_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --bf16 \
            --save_path "$OUTPUT_DIR/checkpoint.pt" \
            "$@"
    } 2>&1 | tee "$LOG_FILE"

    # Extract results
    echo "End: $(date)" | tee -a "$SUMMARY_LOG"
    grep "Final.*acc:" "$LOG_FILE" | tail -1 | tee -a "$SUMMARY_LOG"
    echo "" | tee -a "$SUMMARY_LOG"
}

# ============================================
# EXPERIMENT 1: Conservative baseline
# Based on what peaked at 18.5% but more stable
# ============================================
run_config \
    "1_conservative" \
    "Lower LR, longer warmup, proven architecture" \
    --lr 5e-5 \
    --bottleneck_dim 1024 \
    --soft_tokens 48 \
    --depth 6 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3500 \
    --warmup_steps 1000 \
    --seed 1234

# ============================================
# EXPERIMENT 2: Aggressive (higher LR, less warmup)
# Test if we can train faster
# ============================================
run_config \
    "2_aggressive" \
    "Higher LR, standard warmup, same architecture" \
    --lr 2e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens 48 \
    --depth 6 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 2500 \
    --warmup_steps 500 \
    --seed 1234

# ============================================
# EXPERIMENT 3: More capacity
# More tokens and deeper network
# ============================================
run_config \
    "3_high_capacity" \
    "64 tokens, 8 layers deep, might capture more" \
    --lr 1e-4 \
    --bottleneck_dim 1024 \
    --soft_tokens 64 \
    --depth 8 \
    --heads 16 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 750 \
    --seed 1234

# ============================================
# EXPERIMENT 4: Efficient small model
# Fewer parameters but might be sufficient
# ============================================
run_config \
    "4_efficient" \
    "Smaller bottleneck, fewer tokens, shallower" \
    --lr 1e-4 \
    --bottleneck_dim 768 \
    --soft_tokens 32 \
    --depth 4 \
    --heads 12 \
    --weight_decay 0.01 \
    --train_steps 3000 \
    --warmup_steps 600 \
    --seed 1234

# ============================================
# Final analysis
# ============================================

echo "========================================" | tee -a "$SUMMARY_LOG"
echo "SWEEP COMPLETE at $(date)" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
echo "RESULTS SUMMARY:" | tee -a "$SUMMARY_LOG"
echo "----------------------------------------" | tee -a "$SUMMARY_LOG"

# Create comparison table
{
    echo "Experiment,Target-Alone,Bridged,Gap"
    for dir in "$SWEEP_DIR"/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            log="$dir/train.log"
            if [ -f "$log" ]; then
                final=$(grep "Final.*acc:" "$log" | tail -1)
                target=$(echo "$final" | grep -oE "Target-alone acc: [0-9.]+" | grep -oE "[0-9.]+")
                bridged=$(echo "$final" | grep -oE "Bridged acc: [0-9.]+" | grep -oE "[0-9.]+")
                if [ -n "$target" ] && [ -n "$bridged" ]; then
                    gap=$(echo "$target - $bridged" | bc -l | xargs printf "%.3f")
                    echo "$name,$target,$bridged,$gap"
                fi
            fi
        fi
    done
} | column -t -s',' | tee -a "$SUMMARY_LOG"

echo "" | tee -a "$SUMMARY_LOG"
echo "All logs saved to: $SWEEP_DIR" | tee -a "$SUMMARY_LOG"

# Create a quick Python analysis script
cat > "$SWEEP_DIR/analyze.py" << 'EOF'
import json
import glob
import os

sweep_dir = os.path.dirname(os.path.abspath(__file__))
results = []

for exp_dir in glob.glob(os.path.join(sweep_dir, "*/")):
    if os.path.isdir(exp_dir):
        name = os.path.basename(exp_dir.rstrip("/"))
        log_file = os.path.join(exp_dir, "train.log")

        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()

            # Find all eval lines
            evals = []
            for line in lines:
                if "Eval" in line and "Target-alone" in line:
                    parts = line.split("|")
                    step = int(parts[0].split()[-1])
                    target = float(parts[1].split(":")[-1])
                    bridged = float(parts[2].split(":")[-1])
                    evals.append({"step": step, "target": target, "bridged": bridged})

            if evals:
                results.append({
                    "name": name,
                    "evals": evals,
                    "final_target": evals[-1]["target"],
                    "final_bridged": evals[-1]["bridged"],
                    "max_bridged": max(e["bridged"] for e in evals),
                    "max_bridged_step": max(evals, key=lambda e: e["bridged"])["step"]
                })

# Sort by final bridged accuracy
results.sort(key=lambda x: x["final_bridged"], reverse=True)

print("\n=== Best Final Bridged Accuracy ===")
for r in results[:3]:
    print(f"{r['name']}: {r['final_bridged']:.1%} (target: {r['final_target']:.1%})")

print("\n=== Best Peak Bridged Accuracy ===")
results.sort(key=lambda x: x["max_bridged"], reverse=True)
for r in results[:3]:
    print(f"{r['name']}: {r['max_bridged']:.1%} at step {r['max_bridged_step']}")

# Save as JSON
with open(os.path.join(sweep_dir, "analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to: {os.path.join(sweep_dir, 'analysis.json')}")
EOF

echo "Analysis script created: $SWEEP_DIR/analyze.py" | tee -a "$SUMMARY_LOG"
echo "Run 'python $SWEEP_DIR/analyze.py' after completion for detailed analysis" | tee -a "$SUMMARY_LOG"