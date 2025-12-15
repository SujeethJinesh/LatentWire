#!/usr/bin/env bash
# run_zeroshot_reasoning.sh
# Evaluates Llama and Mistral zero-shot on reasoning benchmarks
# This provides proper baselines for the reasoning results table
#
# Benchmarks:
# - BoolQ (Yes/No QA)
# - PIQA (Physical intuition)
# - CommonsenseQA (5-way reasoning)
#
# Usage: bash run_zeroshot_reasoning.sh

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Configuration
OUTPUT_BASE="runs/zeroshot_reasoning"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

export PYTHONPATH=..

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/zeroshot_reasoning_${TIMESTAMP}.log"

echo "Starting zero-shot reasoning baseline evaluation at $(date)" | tee "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run zero-shot evaluation on all reasoning benchmarks
echo "Evaluating Llama and Mistral zero-shot on reasoning benchmarks..." | tee -a "$LOG_FILE"
python eval_zeroshot_reasoning.py \
    --output_dir "$OUTPUT_DIR" \
    --benchmarks boolq piqa commonsenseqa \
    --max_samples 500 \
    --gpu 0 \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ZERO-SHOT REASONING BASELINES COMPLETE" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List result files
echo "Result files:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -name "*.json" | tee -a "$LOG_FILE"
