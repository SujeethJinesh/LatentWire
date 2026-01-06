#!/usr/bin/env bash
# run_reasoning_benchmarks.sh
# Evaluates Telepathy bridge on reasoning benchmarks
#
# Official HuggingFace datasets used:
# - BoolQ (google/boolq) - Yes/No QA
# - PIQA (ybisk/piqa) - Physical intuition
# - WinoGrande (allenai/winogrande) - Commonsense
# - ARC-Challenge (allenai/ai2_arc) - Science QA
# - CommonsenseQA (tau/commonsense_qa) - Commonsense reasoning
#
# Usage: bash run_reasoning_benchmarks.sh

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Configuration
OUTPUT_BASE="runs/reasoning_benchmarks"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

export PYTHONPATH=..

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/reasoning_${TIMESTAMP}.log"

echo "Starting reasoning benchmark evaluation at $(date)" | tee "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# Binary Benchmarks (easiest - like SST-2)
# ============================================================
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "BINARY BENCHMARKS (2-way classification)" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# BoolQ - Yes/No reading comprehension
echo "[$(date +%H:%M:%S)] Running BoolQ (Yes/No QA)..." | tee -a "$LOG_FILE"
python eval_reasoning_benchmarks.py \
    --benchmark boolq \
    --steps 2000 \
    --soft_tokens 16 \
    --eval_samples 200 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# PIQA - Physical intuition
echo "[$(date +%H:%M:%S)] Running PIQA (Physical intuition)..." | tee -a "$LOG_FILE"
python eval_reasoning_benchmarks.py \
    --benchmark piqa \
    --steps 2000 \
    --soft_tokens 16 \
    --eval_samples 200 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# WinoGrande - Commonsense coreference
echo "[$(date +%H:%M:%S)] Running WinoGrande (Commonsense)..." | tee -a "$LOG_FILE"
python eval_reasoning_benchmarks.py \
    --benchmark winogrande \
    --steps 2000 \
    --soft_tokens 16 \
    --eval_samples 200 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Multi-way Benchmarks (harder)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "MULTI-WAY BENCHMARKS (4-5 way classification)" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ARC-Challenge - Science QA (4-way)
echo "[$(date +%H:%M:%S)] Running ARC-Challenge (Science QA)..." | tee -a "$LOG_FILE"
python eval_reasoning_benchmarks.py \
    --benchmark arc_challenge \
    --steps 2000 \
    --soft_tokens 16 \
    --eval_samples 200 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# CommonsenseQA - Commonsense reasoning (5-way)
echo "[$(date +%H:%M:%S)] Running CommonsenseQA (5-way)..." | tee -a "$LOG_FILE"
python eval_reasoning_benchmarks.py \
    --benchmark commonsenseqa \
    --steps 2000 \
    --soft_tokens 16 \
    --eval_samples 200 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Summary
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "=" | head -c 60 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ALL REASONING BENCHMARKS COMPLETE" | tee -a "$LOG_FILE"
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
echo "Expected results interpretation:" | tee -a "$LOG_FILE"
echo "  - Binary (BoolQ, PIQA, WinoGrande): >50% = above random" | tee -a "$LOG_FILE"
echo "  - ARC-Challenge (4-way): >25% = above random" | tee -a "$LOG_FILE"
echo "  - CommonsenseQA (5-way): >20% = above random" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "If results are significantly above random, the bridge can" | tee -a "$LOG_FILE"
echo "transfer reasoning-relevant information across models!" | tee -a "$LOG_FILE"
