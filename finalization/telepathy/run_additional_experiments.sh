#!/usr/bin/env bash
# run_additional_experiments.sh
# Runs all additional experiments for reviewer response and paper strengthening
#
# Usage: bash run_additional_experiments.sh
#
# Experiments:
# 1. Full fine-tuning baseline (vs LoRA, vs Bridge)
# 2. Memory analysis (peak memory comparison)
# 3. Same-model bridge (Llama→Llama, Mistral→Mistral ablation)
# 4. Architecture ablations (internal dim, heads, depth, layer)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Configuration
OUTPUT_BASE="runs/additional_experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

export PYTHONPATH=..

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/experiments_${TIMESTAMP}.log"

echo "Starting additional experiments at $(date)" | tee "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 1: Full Fine-Tuning Baselines
# ============================================================
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "EXPERIMENT 1: Full Fine-Tuning Baselines" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# SST-2 with different layer counts
for LAYERS in 2 4 8; do
    echo "[$(date +%H:%M:%S)] Full FT SST-2 with $LAYERS layers..." | tee -a "$LOG_FILE"
    python train_full_finetune_baseline.py \
        --dataset sst2 \
        --finetune_layers $LAYERS \
        --epochs 3 \
        --batch_size 2 \
        --output_dir "$OUTPUT_DIR/full_finetune" \
        2>&1 | tee -a "$LOG_FILE"
done

# AG News
echo "[$(date +%H:%M:%S)] Full FT AG News with 4 layers..." | tee -a "$LOG_FILE"
python train_full_finetune_baseline.py \
    --dataset agnews \
    --finetune_layers 4 \
    --epochs 3 \
    --batch_size 2 \
    --output_dir "$OUTPUT_DIR/full_finetune" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 2: Memory Analysis
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "EXPERIMENT 2: Memory Analysis" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[$(date +%H:%M:%S)] Running memory benchmark..." | tee -a "$LOG_FILE"
python benchmark_memory.py \
    --output_dir "$OUTPUT_DIR/memory" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 3: Same-Model Bridge
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "EXPERIMENT 3: Same-Model Bridge" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Llama → Llama
echo "[$(date +%H:%M:%S)] Same-model bridge: Llama → Llama on SST-2..." | tee -a "$LOG_FILE"
python train_same_model_bridge.py \
    --model llama \
    --dataset sst2 \
    --steps 2000 \
    --output_dir "$OUTPUT_DIR/same_model" \
    2>&1 | tee -a "$LOG_FILE"

# Mistral → Mistral
echo "[$(date +%H:%M:%S)] Same-model bridge: Mistral → Mistral on SST-2..." | tee -a "$LOG_FILE"
python train_same_model_bridge.py \
    --model mistral \
    --dataset sst2 \
    --steps 2000 \
    --output_dir "$OUTPUT_DIR/same_model" \
    2>&1 | tee -a "$LOG_FILE"

# AG News same-model
echo "[$(date +%H:%M:%S)] Same-model bridge: Llama → Llama on AG News..." | tee -a "$LOG_FILE"
python train_same_model_bridge.py \
    --model llama \
    --dataset agnews \
    --steps 2000 \
    --output_dir "$OUTPUT_DIR/same_model" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# EXPERIMENT 4: Ablation Studies
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "EXPERIMENT 4: Ablation Studies" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run all ablations on SST-2
echo "[$(date +%H:%M:%S)] Running all ablations on SST-2..." | tee -a "$LOG_FILE"
python run_ablations.py \
    --ablation all \
    --dataset sst2 \
    --steps 1000 \
    --output_dir "$OUTPUT_DIR/ablations" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Summary
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List all result files
echo "Result files:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -name "*.json" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "To analyze results:" | tee -a "$LOG_FILE"
echo "  cat $OUTPUT_DIR/*/*.json | jq '.'" | tee -a "$LOG_FILE"
