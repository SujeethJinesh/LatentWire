#!/usr/bin/env bash
# run_high_priority_fixes.sh
# HIGH PRIORITY: Re-run PIQA and generate proper baseline JSON files
#
# Issues addressed:
# 1. PIQA failed due to trust_remote_code - now fixed in eval_reasoning_benchmarks.py
# 2. Zero-shot baselines lack explicit JSON files
#
# Usage: bash run_high_priority_fixes.sh

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Configuration
OUTPUT_BASE="runs/high_priority_fixes"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

export PYTHONPATH=..

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/high_priority_fixes_${TIMESTAMP}.log"

echo "Starting high priority fixes at $(date)" | tee "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# TASK 1: Re-run PIQA benchmark (trust_remote_code fix)
# ============================================================
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "TASK 1: Re-running PIQA benchmark" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[$(date +%H:%M:%S)] Running PIQA with trust_remote_code=True..." | tee -a "$LOG_FILE"
python eval_reasoning_benchmarks.py \
    --benchmark piqa \
    --steps 2000 \
    --soft_tokens 16 \
    --eval_samples 200 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# TASK 2: Generate Zero-Shot Baseline JSON Files
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "TASK 2: Generating zero-shot baseline JSON files" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[$(date +%H:%M:%S)] Running zero-shot baselines for all datasets..." | tee -a "$LOG_FILE"
python eval_zeroshot_baselines.py \
    --output_dir "$OUTPUT_DIR/zeroshot_baselines" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Summary
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "HIGH PRIORITY FIXES COMPLETE" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List all result files
echo "Result files:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -name "*.json" | tee -a "$LOG_FILE"
