#!/bin/bash
# run_arxiv_experiments.sh
#
# SINGLE UNIFIED SCRIPT for all ArXiv paper experiments.
# Consolidates: run_unified_comparison.sh, run_latency_benchmark.sh, run_overnight.sh
#
# What this runs:
#   1. Bridge training + evaluation (SST-2, AG News, TREC)
#   2. All baselines: Prompt-Tuning, Text-Relay, Zero-shot, Few-shot
#   3. Latency benchmarks (single + batched)
#   4. Memory analysis
#   5. Statistical significance tests
#
# Usage:
#   bash run_arxiv_experiments.sh                    # Full run
#   DATASETS="sst2" bash run_arxiv_experiments.sh   # Single dataset
#   SKIP_LATENCY=1 bash run_arxiv_experiments.sh    # Skip latency benchmarks
#
# Output: Single consolidated JSON + detailed logs

set -e

# ==============================================================================
# CONFIGURATION (override via environment variables)
# ==============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-runs/arxiv_experiments}"
DATASETS="${DATASETS:-sst2 agnews trec}"
SOFT_TOKENS="${SOFT_TOKENS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
FEWSHOT_SHOTS="${FEWSHOT_SHOTS:-5}"
SEED="${SEED:-42}"
SKIP_LATENCY="${SKIP_LATENCY:-}"
SKIP_MEMORY="${SKIP_MEMORY:-}"
SKIP_TEXT_RELAY="${SKIP_TEXT_RELAY:-}"

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FULL_OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"
mkdir -p "$FULL_OUTPUT_DIR"

# Master log file
MASTER_LOG="$FULL_OUTPUT_DIR/master_${TIMESTAMP}.log"

# ==============================================================================
# LOGGING FUNCTIONS
# ==============================================================================
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

section() {
    echo "" | tee -a "$MASTER_LOG"
    echo "==============================================================================" | tee -a "$MASTER_LOG"
    echo "$1" | tee -a "$MASTER_LOG"
    echo "==============================================================================" | tee -a "$MASTER_LOG"
}

# ==============================================================================
# HEADER
# ==============================================================================
section "ARXIV EXPERIMENTS - LATENTWIRE PAPER"
log "Started: $(date)"
log "Output directory: $FULL_OUTPUT_DIR"
log ""
log "Configuration:"
log "  Datasets: $DATASETS"
log "  Soft tokens: $SOFT_TOKENS"
log "  Train steps: $TRAIN_STEPS"
log "  Eval samples: $EVAL_SAMPLES"
log "  Few-shot examples: $FEWSHOT_SHOTS"
log "  Seed: $SEED"
log "  Skip latency: ${SKIP_LATENCY:-no}"
log "  Skip memory: ${SKIP_MEMORY:-no}"
log "  Skip text-relay: ${SKIP_TEXT_RELAY:-no}"
log ""

# ==============================================================================
# PHASE 1: MAIN COMPARISON (Bridge + all baselines)
# ==============================================================================
section "[1/4] MAIN COMPARISON: Bridge + Baselines"
log "Training bridge and evaluating all baselines on: $DATASETS"

COMPARISON_DIR="$FULL_OUTPUT_DIR/comparison"
mkdir -p "$COMPARISON_DIR"

EXTRA_ARGS=""
if [ -n "$SKIP_TEXT_RELAY" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_text_relay"
fi

{
    python telepathy/run_unified_comparison.py \
        --datasets $DATASETS \
        --output_dir "$COMPARISON_DIR" \
        --soft_tokens $SOFT_TOKENS \
        --train_steps $TRAIN_STEPS \
        --eval_samples $EVAL_SAMPLES \
        --fewshot_shots $FEWSHOT_SHOTS \
        --seed $SEED \
        $EXTRA_ARGS
} 2>&1 | tee -a "$MASTER_LOG" | tee "$COMPARISON_DIR/comparison.log"

log "Phase 1 complete: $(date)"

# ==============================================================================
# PHASE 2: LATENCY BENCHMARKS
# ==============================================================================
if [ -z "$SKIP_LATENCY" ]; then
    section "[2/4] LATENCY BENCHMARKS"

    LATENCY_DIR="$FULL_OUTPUT_DIR/latency"
    mkdir -p "$LATENCY_DIR"

    # Find checkpoint from phase 1
    CHECKPOINT=$(find "$COMPARISON_DIR" -name "bridge.pt" -o -name "best_bridge.pt" 2>/dev/null | head -1)

    if [ -n "$CHECKPOINT" ]; then
        log "Using checkpoint: $CHECKPOINT"
        CKPT_ARG="--checkpoint $CHECKPOINT"
    else
        log "No checkpoint found, running latency benchmark without trained bridge"
        CKPT_ARG=""
    fi

    # Single-sample latency
    log "Running single-sample latency benchmark..."
    {
        python telepathy/benchmark_latency.py \
            $CKPT_ARG \
            --num_trials 50 \
            --output_dir "$LATENCY_DIR"
    } 2>&1 | tee -a "$MASTER_LOG" | tee "$LATENCY_DIR/single_latency.log"

    # Batched throughput
    log "Running batched throughput benchmark..."
    {
        python telepathy/benchmark_batched_latency.py \
            --batch_sizes 1 2 4 8 16 \
            --num_samples 32 \
            --output_dir "$LATENCY_DIR"
    } 2>&1 | tee -a "$MASTER_LOG" | tee "$LATENCY_DIR/batched_latency.log"

    log "Phase 2 complete: $(date)"
else
    log "[2/4] LATENCY BENCHMARKS - SKIPPED"
fi

# ==============================================================================
# PHASE 3: MEMORY ANALYSIS
# ==============================================================================
if [ -z "$SKIP_MEMORY" ]; then
    section "[3/4] MEMORY ANALYSIS"

    MEMORY_DIR="$FULL_OUTPUT_DIR/memory"
    mkdir -p "$MEMORY_DIR"

    log "Running memory analysis..."
    {
        python telepathy/benchmark_memory.py \
            --output_dir "$MEMORY_DIR"
    } 2>&1 | tee -a "$MASTER_LOG" | tee "$MEMORY_DIR/memory.log"

    log "Phase 3 complete: $(date)"
else
    log "[3/4] MEMORY ANALYSIS - SKIPPED"
fi

# ==============================================================================
# PHASE 4: STATISTICAL SIGNIFICANCE
# ==============================================================================
section "[4/4] STATISTICAL SIGNIFICANCE"

log "Computing statistical significance tests..."
{
    python telepathy/compute_significance.py \
        --base_dir "$FULL_OUTPUT_DIR" \
        --output "$FULL_OUTPUT_DIR/significance_tests.json"
} 2>&1 | tee -a "$MASTER_LOG" | tee "$FULL_OUTPUT_DIR/significance.log" || {
    log "Warning: Significance tests may need multi-seed data"
}

log "Phase 4 complete: $(date)"

# ==============================================================================
# SUMMARY
# ==============================================================================
section "EXPERIMENT COMPLETE"
log "Finished: $(date)"
log ""
log "Output files:"
log "  Master log: $MASTER_LOG"
log "  Comparison results: $COMPARISON_DIR/unified_results_*.json"
if [ -z "$SKIP_LATENCY" ]; then
    log "  Latency benchmark: $LATENCY_DIR/latency_benchmark.json"
    log "  Batched throughput: $LATENCY_DIR/batched_latency.json"
fi
if [ -z "$SKIP_MEMORY" ]; then
    log "  Memory analysis: $MEMORY_DIR/memory_analysis.json"
fi
log "  Significance tests: $FULL_OUTPUT_DIR/significance_tests.json"
log ""

# Print summary from comparison results
RESULTS_FILE=$(find "$COMPARISON_DIR" -name "unified_results_*.json" | head -1)
if [ -n "$RESULTS_FILE" ]; then
    log "Quick Summary (from $RESULTS_FILE):"
    python -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
for ds, results in data.get('results', {}).items():
    bridge = results.get('bridge', {}).get('accuracy', 'N/A')
    pt = results.get('prompt_tuning', {}).get('accuracy', 'N/A')
    llama = results.get('llama_zeroshot', {}).get('accuracy', 'N/A')
    mistral = results.get('mistral_zeroshot', {}).get('accuracy', 'N/A')
    print(f'  {ds}: Bridge={bridge:.1f}%, Prompt-Tuning={pt:.1f}%, Llama={llama:.1f}%, Mistral={mistral:.1f}%')
" 2>/dev/null || log "  (Run 'cat $RESULTS_FILE | python -m json.tool' to view)"
fi

log ""
log "To analyze results:"
log "  cat $RESULTS_FILE | python -m json.tool"
