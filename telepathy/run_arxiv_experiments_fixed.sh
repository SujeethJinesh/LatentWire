#!/usr/bin/env bash
# telepathy/run_arxiv_experiments_fixed.sh
#
# COMPREHENSIVE ARXIV EXPERIMENTS WITH FIXED CODE
# ================================================
#
# This script reruns ALL experiments (SST-2, AG News, TREC) with the fixed code.
#
# Experiments included:
# 1. Unified comparison (Bridge, Prompt-Tuning, Zero-shot, Few-shot)
# 2. Latency benchmarks (single-sample and batched)
# 3. Per-experiment logging for analysis
#
# Usage (from LatentWire root):
#   PYTHONPATH=. bash telepathy/run_arxiv_experiments_fixed.sh
#
# Expected runtime: ~3-5 hours on single H100
# ================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Set PYTHONPATH to include parent directory (LatentWire root)
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="${OUTPUT_BASE:-runs/arxiv_experiments_fixed_${TIMESTAMP}}"
SEED=42
DATASETS="sst2 agnews trec"
SOFT_TOKENS=8
TRAIN_STEPS=2000
EVAL_SAMPLES=200
FEWSHOT_SHOTS=5

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE/comparison"
mkdir -p "$OUTPUT_BASE/latency"
mkdir -p "$OUTPUT_BASE/logs"

# Main log file
LOG_FILE="$OUTPUT_BASE/experiment_suite_${TIMESTAMP}.log"

echo "=============================================="
echo "ARXIV EXPERIMENTS SUITE (FIXED CODE)"
echo "=============================================="
echo "Timestamp:      $TIMESTAMP"
echo "Output:         $OUTPUT_BASE"
echo "Log:            $LOG_FILE"
echo "Seed:           $SEED"
echo "Datasets:       $DATASETS"
echo "Soft tokens:    $SOFT_TOKENS"
echo "Train steps:    $TRAIN_STEPS"
echo "Eval samples:   $EVAL_SAMPLES"
echo "=============================================="
echo ""

# Start master logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting experiments at $(date)"
echo ""

# =============================================================================
# SECTION 1: UNIFIED COMPARISON (Main Results)
# =============================================================================
# This runs all baselines in ONE script for fair comparison:
# - Bridge (Llama → Mistral via soft tokens)
# - Prompt-Tuning (Mistral only, proves sender is needed)
# - Text-Relay (Llama summarizes → text → Mistral)
# - Zero-shot baselines (Llama, Mistral direct)
# - Few-shot prompting (5-shot)

echo ""
echo "=============================================="
echo "SECTION 1: UNIFIED COMPARISON"
echo "=============================================="
echo "Running all baselines for fair comparison..."
echo ""

COMPARISON_DIR="$OUTPUT_BASE/comparison"
COMPARISON_LOG="$OUTPUT_BASE/logs/unified_comparison_${TIMESTAMP}.log"

{
    echo "Command: python run_unified_comparison.py"
    echo "Start time: $(date)"
    echo ""

    python run_unified_comparison.py \
        --datasets $DATASETS \
        --output_dir "$COMPARISON_DIR" \
        --soft_tokens $SOFT_TOKENS \
        --train_steps $TRAIN_STEPS \
        --eval_samples $EVAL_SAMPLES \
        --fewshot_shots $FEWSHOT_SHOTS \
        --seed $SEED

    echo ""
    echo "End time: $(date)"
} 2>&1 | tee "$COMPARISON_LOG"

echo "Unified comparison complete!"
echo "Results saved to: $COMPARISON_DIR"
echo ""

# =============================================================================
# SECTION 2: LATENCY BENCHMARKS
# =============================================================================
# Measures inference speed for Bridge vs Text-Relay vs Direct

echo ""
echo "=============================================="
echo "SECTION 2: LATENCY BENCHMARKS"
echo "=============================================="
echo ""

LATENCY_DIR="$OUTPUT_BASE/latency"

# Check if bridge checkpoints exist from unified comparison
BRIDGE_CHECKPOINT=""
if [ -f "$COMPARISON_DIR/bridge_sst2.pt" ]; then
    BRIDGE_CHECKPOINT="$COMPARISON_DIR/bridge_sst2.pt"
    echo "Using bridge checkpoint from unified comparison: $BRIDGE_CHECKPOINT"
else
    echo "Warning: No bridge checkpoint found. Latency will run without checkpoint."
fi

# 2A: Single-sample latency benchmark
echo ""
echo "--- 2A: Single-Sample Latency ---"
echo ""

SINGLE_LATENCY_LOG="$OUTPUT_BASE/logs/latency_single_${TIMESTAMP}.log"

{
    echo "Command: python benchmark_latency.py"
    echo "Start time: $(date)"
    echo ""

    if [ -n "$BRIDGE_CHECKPOINT" ]; then
        python benchmark_latency.py \
            --checkpoint "$BRIDGE_CHECKPOINT" \
            --num_trials 50 \
            --output_dir "$LATENCY_DIR" \
            --gpu 0
    else
        python benchmark_latency.py \
            --num_trials 50 \
            --output_dir "$LATENCY_DIR" \
            --gpu 0
    fi

    echo ""
    echo "End time: $(date)"
} 2>&1 | tee "$SINGLE_LATENCY_LOG"

echo "Single-sample latency complete!"
echo ""

# 2B: Batched latency benchmark
echo ""
echo "--- 2B: Batched Latency & Throughput ---"
echo ""

# Check if batched benchmark script exists
if [ -f "benchmark_batched_latency.py" ]; then
    BATCHED_LATENCY_LOG="$OUTPUT_BASE/logs/latency_batched_${TIMESTAMP}.log"

    {
        echo "Command: python benchmark_batched_latency.py"
        echo "Start time: $(date)"
        echo ""

        python benchmark_batched_latency.py \
            --batch_sizes 1 2 4 8 16 32 \
            --num_samples 64 \
            --output_dir "$LATENCY_DIR"

        echo ""
        echo "End time: $(date)"
    } 2>&1 | tee "$BATCHED_LATENCY_LOG"

    echo "Batched latency complete!"
else
    echo "Warning: benchmark_batched_latency.py not found. Skipping batched benchmark."
fi

echo ""
echo "All latency benchmarks complete!"
echo "Results saved to: $LATENCY_DIR"
echo ""

# =============================================================================
# SECTION 3: GENERATE COMPREHENSIVE SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "SECTION 3: GENERATING SUMMARY"
echo "=============================================="
echo ""

SUMMARY_FILE="$OUTPUT_BASE/SUMMARY.txt"

{
    echo "=========================================="
    echo "ARXIV EXPERIMENTS SUMMARY"
    echo "=========================================="
    echo ""
    echo "Timestamp: $TIMESTAMP"
    echo "Output directory: $OUTPUT_BASE"
    echo "Seed: $SEED"
    echo "Datasets: $DATASETS"
    echo ""
    echo "=========================================="
    echo "EXPERIMENT STRUCTURE"
    echo "=========================================="
    echo ""
    echo "1. Unified Comparison:"
    echo "   - Location: $COMPARISON_DIR"
    echo "   - Contains: All baseline comparisons (Bridge, Prompt-Tuning, etc.)"
    echo "   - Log: $COMPARISON_LOG"
    echo ""
    echo "2. Latency Benchmarks:"
    echo "   - Location: $LATENCY_DIR"
    echo "   - Contains: Single-sample and batched latency measurements"
    echo "   - Logs: $OUTPUT_BASE/logs/latency_*.log"
    echo ""
    echo "=========================================="
    echo "RESULTS FILES"
    echo "=========================================="
    echo ""
    ls -lh "$COMPARISON_DIR"/*.json 2>/dev/null || echo "No JSON files found in comparison directory"
    echo ""
    ls -lh "$LATENCY_DIR"/*.json 2>/dev/null || echo "No JSON files found in latency directory"
    echo ""
    echo "=========================================="
    echo "KEY METRICS TO EXTRACT"
    echo "=========================================="
    echo ""
    echo "From unified_results_*.json:"
    echo "  - Bridge accuracy per dataset"
    echo "  - Prompt-Tuning accuracy (proves sender helps)"
    echo "  - Zero-shot baselines"
    echo "  - Few-shot baselines"
    echo "  - Per-method latencies"
    echo ""
    echo "From latency_benchmark.json:"
    echo "  - Bridge vs Text-Relay speedup"
    echo "  - Breakdown: encode, bridge, decode times"
    echo "  - Qualitative examples of summaries"
    echo ""
    echo "From batched_latency.json:"
    echo "  - Throughput at various batch sizes"
    echo "  - Scaling characteristics"
    echo "  - Bridge vs Direct comparison"
    echo ""
    echo "=========================================="
    echo "NEXT STEPS"
    echo "=========================================="
    echo ""
    echo "1. Analyze results:"
    echo "   cat $COMPARISON_DIR/unified_results_*.json | jq '.comparison_table'"
    echo ""
    echo "2. Extract accuracy table:"
    echo "   python -c \"import json; d=json.load(open('$COMPARISON_DIR/unified_results_*.json')); print(d['comparison_table'])\""
    echo ""
    echo "3. Check latency gains:"
    echo "   cat $LATENCY_DIR/latency_benchmark.json | jq '{bridge: .bridge.total_ms, relay: .text_relay.total_ms}'"
    echo ""
    echo "4. Review logs for any errors or warnings:"
    echo "   grep -i 'error\|warning\|failed' $LOG_FILE"
    echo ""
    echo "=========================================="
    echo "EXPERIMENT CHECKLIST"
    echo "=========================================="
    echo ""

    # Check if key files exist
    if [ -f "$COMPARISON_DIR"/unified_results_*.json ]; then
        echo "[✓] Unified comparison results found"
    else
        echo "[✗] Unified comparison results MISSING"
    fi

    if [ -f "$LATENCY_DIR/latency_benchmark.json" ]; then
        echo "[✓] Single-sample latency results found"
    else
        echo "[✗] Single-sample latency results MISSING"
    fi

    if [ -f "$LATENCY_DIR/batched_latency.json" ]; then
        echo "[✓] Batched latency results found"
    else
        echo "[✗] Batched latency results MISSING (may be skipped)"
    fi

    echo ""

    # Count bridge checkpoints
    CKPT_COUNT=$(ls -1 "$COMPARISON_DIR"/bridge_*.pt 2>/dev/null | wc -l)
    echo "Bridge checkpoints saved: $CKPT_COUNT"

    echo ""
    echo "=========================================="

} | tee "$SUMMARY_FILE"

echo ""
echo "Summary written to: $SUMMARY_FILE"
echo ""

# =============================================================================
# FINAL OUTPUT
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENTS COMPLETE!"
echo "=============================================="
echo ""
echo "Total runtime: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s"
echo ""
echo "Output structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── comparison/           # Main results (Bridge vs baselines)"
echo "  │   ├── unified_results_*.json"
echo "  │   └── bridge_*.pt       # Trained checkpoints"
echo "  ├── latency/              # Speed benchmarks"
echo "  │   ├── latency_benchmark.json"
echo "  │   └── batched_latency.json"
echo "  ├── logs/                 # Per-experiment logs"
echo "  │   ├── unified_comparison_*.log"
echo "  │   ├── latency_single_*.log"
echo "  │   └── latency_batched_*.log"
echo "  ├── SUMMARY.txt           # Human-readable summary"
echo "  └── experiment_suite_*.log # Master log"
echo ""
echo "Master log: $LOG_FILE"
echo "Summary:    $SUMMARY_FILE"
echo ""
echo "To view results:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "To analyze JSON results:"
echo "  python -c \"import json; print(json.dumps(json.load(open('$COMPARISON_DIR/unified_results_'+'*.json')), indent=2))\""
echo ""
echo "To commit results:"
echo "  git add $OUTPUT_BASE"
echo "  git commit -m 'feat: arxiv experiments with fixed code'"
echo "  git push"
echo ""
echo "=============================================="
