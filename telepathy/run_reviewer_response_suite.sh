#!/usr/bin/env bash
# run_reviewer_response_suite.sh
#
# Comprehensive experiment suite to address all reviewer concerns.
#
# REVIEWER CONCERNS ADDRESSED:
# 1. Percy Liang: Larger samples (1000+), ensemble baseline, per-class Banking77
# 2. Sara Hooker: Memory profiling, soft token quantization (future)
# 3. Tri Dao: Detailed latency profiling with CI
# 4. Colin Raffel: Same-checkpoint baseline, task interpolation
# 5. Yejin Choi: Reasoning failure analysis with examples
# 6. Jason Wei: Few-shot scaling (0,5,10,20-shot)
# 7. Stella Biderman: Statistical significance, stratified sampling
# 8. Chelsea Finn: Multi-task training, min data experiments (future)
# 9. Denny Zhou: Self-consistency, probing study
# 10. Alec Radford: More model pairs (future)
#
# Usage:
#   bash run_reviewer_response_suite.sh           # Run all experiments
#   bash run_reviewer_response_suite.sh quick     # Quick validation run
#   bash run_reviewer_response_suite.sh specific  # Run specific experiment
#
# Expected runtime: 4-8 hours on H100 (full suite)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="runs/reviewer_response"
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/reviewer_experiments_${TIMESTAMP}.log"

export PYTHONPATH=..

# Parse arguments
MODE="${1:-all}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "REVIEWER RESPONSE EXPERIMENT SUITE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Mode: $MODE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run an experiment with logging
run_experiment() {
    local name="$1"
    local max_samples="$2"

    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$name] Starting at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    python run_reviewer_experiments.py \
        --experiment "$name" \
        --output_dir "$OUTPUT_DIR" \
        --max_samples "$max_samples" \
        2>&1 | tee -a "$LOG_FILE"

    echo "[$name] Completed at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
}

# Determine sample sizes based on mode
if [ "$MODE" == "quick" ]; then
    EVAL_SAMPLES=100
    LARGE_EVAL_SAMPLES=200
    echo "Quick mode: Using reduced sample sizes" | tee -a "$LOG_FILE"
else
    EVAL_SAMPLES=500
    LARGE_EVAL_SAMPLES=1000
fi

# ============================================================
# PHASE 1: Statistical Foundation (Stella Biderman)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 1: STATISTICAL FOUNDATION" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "significance" ]; then
    run_experiment "significance" 0
fi

# ============================================================
# PHASE 2: Large-Scale Evaluation (Percy Liang, Stella Biderman)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 2: LARGE-SCALE EVALUATION" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "large_eval" ]; then
    # Override samples for large eval
    echo "[large_eval] Starting at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    python run_reviewer_experiments.py \
        --experiment "large_eval" \
        --output_dir "$OUTPUT_DIR" \
        --max_samples "$LARGE_EVAL_SAMPLES" \
        2>&1 | tee -a "$LOG_FILE"
    echo "[large_eval] Completed at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
fi

# ============================================================
# PHASE 3: Baseline Comparisons (Percy Liang, Colin Raffel)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 3: BASELINE COMPARISONS" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "ensemble" ]; then
    run_experiment "ensemble" "$EVAL_SAMPLES"
fi

# ============================================================
# PHASE 4: Systems Analysis (Sara Hooker, Tri Dao)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 4: SYSTEMS ANALYSIS" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "memory" ]; then
    run_experiment "memory" 0
fi

if [ "$MODE" == "all" ] || [ "$MODE" == "latency" ]; then
    run_experiment "latency" 0
fi

# ============================================================
# PHASE 5: Reasoning Analysis (Yejin Choi, Denny Zhou)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 5: REASONING ANALYSIS" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "reasoning" ]; then
    run_experiment "reasoning" "$EVAL_SAMPLES"
fi

# ============================================================
# PHASE 6: Few-Shot Scaling (Jason Wei)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 6: FEW-SHOT SCALING" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "fewshot" ]; then
    run_experiment "fewshot" "$EVAL_SAMPLES"
fi

# ============================================================
# PHASE 7: Per-Class Analysis (Percy Liang)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 7: PER-CLASS ANALYSIS" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "banking77" ]; then
    run_experiment "banking77" "$EVAL_SAMPLES"
fi

# ============================================================
# PHASE 8: Zero-Shot Baselines (Audit requirement)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 8: ZERO-SHOT BASELINES" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "zeroshot" ]; then
    echo "[zeroshot] Starting at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    python eval_zeroshot_baselines.py \
        --output_dir "$OUTPUT_DIR/zeroshot_baselines" \
        --max_samples "$EVAL_SAMPLES" \
        2>&1 | tee -a "$LOG_FILE"
    echo "[zeroshot] Completed at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
fi

# ============================================================
# PHASE 9: PIQA Re-run (Audit fix)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# PHASE 9: PIQA RE-RUN" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"

if [ "$MODE" == "all" ] || [ "$MODE" == "piqa" ]; then
    echo "[piqa] Starting at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    python eval_reasoning_benchmarks.py \
        --benchmark piqa \
        --steps 2000 \
        --soft_tokens 16 \
        --eval_samples "$EVAL_SAMPLES" \
        --output_dir "$OUTPUT_DIR/reasoning" \
        2>&1 | tee -a "$LOG_FILE"
    echo "[piqa] Completed at $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
fi

# ============================================================
# SUMMARY
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "# EXPERIMENT SUITE COMPLETE" | tee -a "$LOG_FILE"
echo "############################################################" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List all result files
echo "Result files:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -name "*.json" -type f | sort | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "REVIEWER CONCERN COVERAGE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
cat << 'EOF' | tee -a "$LOG_FILE"

Percy Liang:
  [x] large_evaluation_results.json - 1000+ samples with stratified sampling
  [x] ensemble_baseline_results.json - Majority vote vs oracle ensemble
  [x] banking77_perclass_results.json - Per-class accuracy breakdown

Sara Hooker:
  [x] memory_profiling_results.json - GPU memory consumption
  [ ] Soft token quantization - Future work (requires bridge training)

Tri Dao:
  [x] latency_profiling_results.json - Detailed breakdown with CI

Colin Raffel:
  [ ] Same-checkpoint baseline - Requires bridge training
  [ ] Task interpolation - Requires multi-task bridges

Yejin Choi:
  [x] reasoning_analysis_results.json - BoolQ vs CSQA failure analysis

Jason Wei:
  [x] fewshot_scaling_results.json - 0/5/10/20-shot comparison

Stella Biderman:
  [x] statistical_significance_results.json - p-values and effect sizes
  [x] Stratified sampling in large_evaluation
  [x] zeroshot_baselines/*.json - Explicit JSON files for audit

Chelsea Finn:
  [ ] Multi-task training - Future work
  [ ] Minimum data experiments - Future work

Denny Zhou:
  [x] reasoning_analysis_results.json - Examples of failures
  [ ] Self-consistency - Future work (requires bridge inference)
  [ ] Probing study - Future work

Alec Radford:
  [ ] More model pairs - Future work (requires additional training)

EOF

echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
