#!/usr/bin/env bash
# =============================================================================
# COMPREHENSIVE EXPERIMENT RUNNER FOR LATENTWIRE PAPER REVISION
# =============================================================================
# This script orchestrates ALL experiments needed for the paper revision:
# - Phase 1: Statistical rigor with multiple seeds and bootstrap CI
# - Phase 2: Linear probe baseline comparisons
# - Phase 3: Fair baseline comparisons (LLMLingua, token-budget, etc.)
# - Phase 4: Efficiency measurements (latency, memory, throughput)
#
# Resilient to preemption with automatic checkpoint resumption
# Uses only 1 checkpoint for efficiency (shared across evaluations)
#
# Usage:
#   bash finalization/run_all_experiments.sh
#
# Environment Variables:
#   SKIP_TRAINING: Set to "yes" to skip training (assumes checkpoint exists)
#   CHECKPOINT_PATH: Path to pre-trained checkpoint (if SKIP_TRAINING=yes)
#   SEEDS: Space-separated list of random seeds (default: "42 123 456")
#   DATASETS: Datasets to evaluate (default: all)
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment configuration
EXP_NAME="${EXP_NAME:-latentwire_paper_revision}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-runs/$EXP_NAME}"
CHECKPOINT_DIR="$BASE_OUTPUT_DIR/checkpoint"
RESULTS_DIR="$BASE_OUTPUT_DIR/results"
LOG_DIR="$BASE_OUTPUT_DIR/logs"

# Training configuration
SKIP_TRAINING="${SKIP_TRAINING:-no}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TRAINING_DATASET="squad"
TRAINING_SAMPLES=87599
TRAINING_EPOCHS=24
LATENT_LEN=32
D_Z=256

# Evaluation configuration
SEEDS="${SEEDS:-42 123 456}"
DATASETS="${DATASETS:-sst2 agnews trec xsum}"
BOOTSTRAP_SAMPLES=10000
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"  # Empty means use full test sets

# Model configuration
SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# Hardware configuration
NUM_GPUS="${NUM_GPUS:-1}"  # Use 1 GPU for efficiency
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

# Paths
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Critical environment variables for optimal performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$LOG_DIR"
mkdir -p "$RESULTS_DIR/phase1_statistical"
mkdir -p "$RESULTS_DIR/phase2_linear_probe"
mkdir -p "$RESULTS_DIR/phase3_baselines"
mkdir -p "$RESULTS_DIR/phase4_efficiency"

# Set up logging
MAIN_LOG="$LOG_DIR/main_experiment_${TIMESTAMP}.log"
exec > >(tee -a "$MAIN_LOG")
exec 2>&1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_message() {
    local level=$1
    shift
    local message="$@"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message"
}

run_with_retry() {
    local max_attempts=3
    local attempt=1
    local cmd="$@"

    while [ $attempt -le $max_attempts ]; do
        log_message "INFO" "Attempt $attempt of $max_attempts: $cmd"
        if eval "$cmd"; then
            return 0
        else
            log_message "WARN" "Command failed, attempt $attempt"
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                sleep 10
            fi
        fi
    done

    log_message "ERROR" "Command failed after $max_attempts attempts"
    return 1
}

find_checkpoint() {
    # Find the latest epoch checkpoint in the directory
    local ckpt_dir=$1
    local latest_ckpt=""
    local max_epoch=-1

    for ckpt in "$ckpt_dir"/epoch*; do
        if [ -d "$ckpt" ]; then
            epoch_num=$(basename "$ckpt" | sed 's/epoch//')
            if [[ "$epoch_num" =~ ^[0-9]+$ ]] && [ "$epoch_num" -gt "$max_epoch" ]; then
                max_epoch=$epoch_num
                latest_ckpt=$ckpt
            fi
        fi
    done

    echo "$latest_ckpt"
}

# =============================================================================
# PHASE 0: TRAINING (OR CHECKPOINT VERIFICATION)
# =============================================================================

phase0_training() {
    log_message "INFO" "========================================="
    log_message "INFO" "PHASE 0: TRAINING / CHECKPOINT SETUP"
    log_message "INFO" "========================================="

    if [ "$SKIP_TRAINING" = "yes" ]; then
        if [ -z "$CHECKPOINT_PATH" ]; then
            # Try to find existing checkpoint
            CHECKPOINT_PATH=$(find_checkpoint "$CHECKPOINT_DIR")
            if [ -z "$CHECKPOINT_PATH" ]; then
                log_message "ERROR" "SKIP_TRAINING=yes but no checkpoint found in $CHECKPOINT_DIR"
                exit 1
            fi
        fi

        if [ ! -d "$CHECKPOINT_PATH" ]; then
            log_message "ERROR" "Checkpoint not found: $CHECKPOINT_PATH"
            exit 1
        fi

        log_message "INFO" "Using existing checkpoint: $CHECKPOINT_PATH"
    else
        log_message "INFO" "Training new checkpoint..."

        # Build training command
        local train_cmd="python latentwire/train.py"
        train_cmd="$train_cmd --llama_id '$SOURCE_MODEL'"
        train_cmd="$train_cmd --qwen_id '$TARGET_MODEL'"
        train_cmd="$train_cmd --dataset $TRAINING_DATASET"
        train_cmd="$train_cmd --samples $TRAINING_SAMPLES"
        train_cmd="$train_cmd --epochs $TRAINING_EPOCHS"
        train_cmd="$train_cmd --batch_size $BATCH_SIZE"
        train_cmd="$train_cmd --latent_len $LATENT_LEN"
        train_cmd="$train_cmd --d_z $D_Z"
        train_cmd="$train_cmd --output_dir $CHECKPOINT_DIR"
        train_cmd="$train_cmd --encoder_type byte"
        train_cmd="$train_cmd --sequential_models"
        train_cmd="$train_cmd --warm_anchor_text 'Answer: '"
        train_cmd="$train_cmd --first_token_ce_weight 0.5"
        train_cmd="$train_cmd --mixed_precision bf16"
        train_cmd="$train_cmd --seed 42"

        # Check for resumption
        local existing_ckpt=$(find_checkpoint "$CHECKPOINT_DIR")
        if [ -n "$existing_ckpt" ]; then
            log_message "INFO" "Found existing checkpoint, resuming from: $existing_ckpt"
            train_cmd="$train_cmd --resume_from $existing_ckpt"
        fi

        # Run training
        local train_log="$LOG_DIR/training_${TIMESTAMP}.log"
        log_message "INFO" "Training log: $train_log"

        if run_with_retry "$train_cmd 2>&1 | tee '$train_log'"; then
            CHECKPOINT_PATH=$(find_checkpoint "$CHECKPOINT_DIR")
            log_message "INFO" "Training complete. Checkpoint: $CHECKPOINT_PATH"
        else
            log_message "ERROR" "Training failed"
            exit 1
        fi
    fi

    # Verify checkpoint exists and is valid
    if [ ! -f "$CHECKPOINT_PATH/encoder.pt" ]; then
        log_message "ERROR" "Invalid checkpoint - missing encoder.pt"
        exit 1
    fi

    export CHECKPOINT_PATH
    log_message "INFO" "Checkpoint ready: $CHECKPOINT_PATH"
}

# =============================================================================
# PHASE 1: STATISTICAL RIGOR
# =============================================================================

phase1_statistical_rigor() {
    log_message "INFO" "========================================="
    log_message "INFO" "PHASE 1: STATISTICAL RIGOR"
    log_message "INFO" "========================================="

    local phase_dir="$RESULTS_DIR/phase1_statistical"

    for dataset in $DATASETS; do
        log_message "INFO" "Evaluating dataset: $dataset"

        # Determine dataset-specific parameters
        case $dataset in
            sst2)
                eval_samples="${MAX_EVAL_SAMPLES:-872}"
                eval_script="latentwire/eval_sst2.py"
                ;;
            agnews)
                eval_samples="${MAX_EVAL_SAMPLES:-7600}"
                eval_script="latentwire/eval_agnews.py"
                ;;
            trec)
                eval_samples="${MAX_EVAL_SAMPLES:-500}"
                eval_script="telepathy/eval_telepathy_trec.py"
                ;;
            xsum)
                eval_samples="${MAX_EVAL_SAMPLES:-11334}"
                eval_script="latentwire/eval.py"
                ;;
            *)
                log_message "WARN" "Unknown dataset: $dataset, using default eval"
                eval_samples="${MAX_EVAL_SAMPLES:-1000}"
                eval_script="latentwire/eval.py"
                ;;
        esac

        for seed in $SEEDS; do
            log_message "INFO" "Running seed $seed for $dataset"

            local output_file="$phase_dir/${dataset}_seed${seed}_results.json"
            local eval_log="$LOG_DIR/phase1_${dataset}_seed${seed}.log"

            # Build evaluation command
            local eval_cmd="python $eval_script"
            eval_cmd="$eval_cmd --ckpt $CHECKPOINT_PATH"
            eval_cmd="$eval_cmd --dataset $dataset"
            eval_cmd="$eval_cmd --samples $eval_samples"
            eval_cmd="$eval_cmd --batch_size $EVAL_BATCH_SIZE"
            eval_cmd="$eval_cmd --seed $seed"
            eval_cmd="$eval_cmd --output_file $output_file"
            eval_cmd="$eval_cmd --include_baselines"  # Text-relay, token-budget
            eval_cmd="$eval_cmd --sequential_eval"
            eval_cmd="$eval_cmd --fresh_eval"

            # Run evaluation
            if run_with_retry "$eval_cmd 2>&1 | tee '$eval_log'"; then
                log_message "INFO" "Completed $dataset with seed $seed"
            else
                log_message "ERROR" "Failed $dataset with seed $seed"
            fi
        done
    done

    # Run statistical analysis
    log_message "INFO" "Running statistical analysis..."
    local stats_cmd="python scripts/statistical_testing.py"
    stats_cmd="$stats_cmd --results_dir $phase_dir"
    stats_cmd="$stats_cmd --bootstrap_samples $BOOTSTRAP_SAMPLES"
    stats_cmd="$stats_cmd --output_file $phase_dir/statistical_summary.json"

    if run_with_retry "$stats_cmd 2>&1 | tee '$LOG_DIR/statistical_analysis.log'"; then
        log_message "INFO" "Statistical analysis complete"
    else
        log_message "ERROR" "Statistical analysis failed"
    fi
}

# =============================================================================
# PHASE 2: LINEAR PROBE BASELINE
# =============================================================================

phase2_linear_probe() {
    log_message "INFO" "========================================="
    log_message "INFO" "PHASE 2: LINEAR PROBE BASELINE"
    log_message "INFO" "========================================="

    local phase_dir="$RESULTS_DIR/phase2_linear_probe"

    for dataset in $DATASETS; do
        if [ "$dataset" = "xsum" ]; then
            log_message "INFO" "Skipping linear probe for generation task: $dataset"
            continue
        fi

        log_message "INFO" "Running linear probe for $dataset"

        local probe_cmd="python latentwire/linear_probe_baseline.py"
        probe_cmd="$probe_cmd --source_model $SOURCE_MODEL"
        probe_cmd="$probe_cmd --dataset $dataset"
        probe_cmd="$probe_cmd --layer 16"  # Middle layer as per documentation
        probe_cmd="$probe_cmd --cv_folds 5"
        probe_cmd="$probe_cmd --output_dir $phase_dir"
        probe_cmd="$probe_cmd --batch_size $EVAL_BATCH_SIZE"

        local probe_log="$LOG_DIR/phase2_linear_probe_${dataset}.log"

        if run_with_retry "$probe_cmd 2>&1 | tee '$probe_log'"; then
            log_message "INFO" "Linear probe complete for $dataset"
        else
            log_message "ERROR" "Linear probe failed for $dataset"
        fi
    done

    # Compare with LatentWire results
    log_message "INFO" "Generating comparison report..."
    local compare_cmd="python scripts/compare_linear_probe.py"
    compare_cmd="$compare_cmd --latentwire_results $RESULTS_DIR/phase1_statistical"
    compare_cmd="$compare_cmd --probe_results $phase_dir"
    compare_cmd="$compare_cmd --output_file $phase_dir/comparison_report.json"

    run_with_retry "$compare_cmd 2>&1 | tee '$LOG_DIR/linear_probe_comparison.log'"
}

# =============================================================================
# PHASE 3: FAIR BASELINE COMPARISONS
# =============================================================================

phase3_baselines() {
    log_message "INFO" "========================================="
    log_message "INFO" "PHASE 3: FAIR BASELINE COMPARISONS"
    log_message "INFO" "========================================="

    local phase_dir="$RESULTS_DIR/phase3_baselines"

    # Run LLMLingua-2 baseline
    log_message "INFO" "Running LLMLingua-2 baseline..."
    local llmlingua_cmd="bash scripts/run_llmlingua_baseline.sh"
    llmlingua_cmd="$llmlingua_cmd --output_dir $phase_dir/llmlingua"
    llmlingua_cmd="$llmlingua_cmd --compression_ratio 8"  # Match LatentWire
    llmlingua_cmd="$llmlingua_cmd --datasets '$DATASETS'"

    if run_with_retry "$llmlingua_cmd 2>&1 | tee '$LOG_DIR/phase3_llmlingua.log'"; then
        log_message "INFO" "LLMLingua-2 baseline complete"
    else
        log_message "ERROR" "LLMLingua-2 baseline failed"
    fi

    # Run direct prompting baselines (zero-shot and few-shot)
    for dataset in $DATASETS; do
        log_message "INFO" "Running direct prompting baselines for $dataset"

        # Zero-shot
        local zeroshot_cmd="python latentwire/eval.py"
        zeroshot_cmd="$zeroshot_cmd --mode zeroshot"
        zeroshot_cmd="$zeroshot_cmd --dataset $dataset"
        zeroshot_cmd="$zeroshot_cmd --model $TARGET_MODEL"
        zeroshot_cmd="$zeroshot_cmd --samples ${MAX_EVAL_SAMPLES:-1000}"
        zeroshot_cmd="$zeroshot_cmd --output_file $phase_dir/zeroshot_${dataset}.json"

        run_with_retry "$zeroshot_cmd 2>&1 | tee '$LOG_DIR/phase3_zeroshot_${dataset}.log'"

        # Few-shot (3-shot)
        local fewshot_cmd="python latentwire/eval.py"
        fewshot_cmd="$fewshot_cmd --mode fewshot"
        fewshot_cmd="$fewshot_cmd --num_shots 3"
        fewshot_cmd="$fewshot_cmd --dataset $dataset"
        fewshot_cmd="$fewshot_cmd --model $TARGET_MODEL"
        fewshot_cmd="$fewshot_cmd --samples ${MAX_EVAL_SAMPLES:-1000}"
        fewshot_cmd="$fewshot_cmd --output_file $phase_dir/fewshot_${dataset}.json"

        run_with_retry "$fewshot_cmd 2>&1 | tee '$LOG_DIR/phase3_fewshot_${dataset}.log'"
    done

    # Generate baseline comparison table
    log_message "INFO" "Generating baseline comparison table..."
    local compare_cmd="python scripts/generate_baseline_table.py"
    compare_cmd="$compare_cmd --latentwire_results $RESULTS_DIR/phase1_statistical"
    compare_cmd="$compare_cmd --baseline_results $phase_dir"
    compare_cmd="$compare_cmd --output_file $phase_dir/baseline_comparison.json"
    compare_cmd="$compare_cmd --latex_output $phase_dir/baseline_table.tex"

    run_with_retry "$compare_cmd 2>&1 | tee '$LOG_DIR/baseline_comparison.log'"
}

# =============================================================================
# PHASE 4: EFFICIENCY MEASUREMENTS
# =============================================================================

phase4_efficiency() {
    log_message "INFO" "========================================="
    log_message "INFO" "PHASE 4: EFFICIENCY MEASUREMENTS"
    log_message "INFO" "========================================="

    local phase_dir="$RESULTS_DIR/phase4_efficiency"

    for dataset in $DATASETS; do
        log_message "INFO" "Measuring efficiency for $dataset"

        # Run efficiency benchmark
        local bench_cmd="python scripts/benchmark_efficiency.py"
        bench_cmd="$bench_cmd --checkpoint $CHECKPOINT_PATH"
        bench_cmd="$bench_cmd --dataset $dataset"
        bench_cmd="$bench_cmd --samples 100"  # Smaller sample for timing
        bench_cmd="$bench_cmd --warmup_runs 3"
        bench_cmd="$bench_cmd --benchmark_runs 10"
        bench_cmd="$bench_cmd --measure_memory"
        bench_cmd="$bench_cmd --measure_latency"
        bench_cmd="$bench_cmd --measure_throughput"
        bench_cmd="$bench_cmd --output_file $phase_dir/efficiency_${dataset}.json"

        local bench_log="$LOG_DIR/phase4_efficiency_${dataset}.log"

        if run_with_retry "$bench_cmd 2>&1 | tee '$bench_log'"; then
            log_message "INFO" "Efficiency measurement complete for $dataset"
        else
            log_message "ERROR" "Efficiency measurement failed for $dataset"
        fi
    done

    # Test different compression levels (quantization)
    log_message "INFO" "Testing quantization levels..."
    for quant in fp16 int8 int4; do
        log_message "INFO" "Testing $quant quantization"

        local quant_cmd="python scripts/test_quantization.py"
        quant_cmd="$quant_cmd --checkpoint $CHECKPOINT_PATH"
        quant_cmd="$quant_cmd --quantization $quant"
        quant_cmd="$quant_cmd --dataset squad"  # Use one dataset for testing
        quant_cmd="$quant_cmd --samples 100"
        quant_cmd="$quant_cmd --output_file $phase_dir/quantization_${quant}.json"

        run_with_retry "$quant_cmd 2>&1 | tee '$LOG_DIR/phase4_quant_${quant}.log'"
    done

    # Generate efficiency report
    log_message "INFO" "Generating efficiency report..."
    local report_cmd="python scripts/generate_efficiency_report.py"
    report_cmd="$report_cmd --efficiency_dir $phase_dir"
    report_cmd="$report_cmd --output_file $phase_dir/efficiency_summary.json"
    report_cmd="$report_cmd --generate_plots"

    run_with_retry "$report_cmd 2>&1 | tee '$LOG_DIR/efficiency_report.log'"
}

# =============================================================================
# FINAL AGGREGATION AND REPORT GENERATION
# =============================================================================

generate_final_report() {
    log_message "INFO" "========================================="
    log_message "INFO" "GENERATING FINAL REPORT"
    log_message "INFO" "========================================="

    # Aggregate all results
    local aggregate_cmd="python finalization/aggregate_results.py"
    aggregate_cmd="$aggregate_cmd --results_dir $RESULTS_DIR"
    aggregate_cmd="$aggregate_cmd --output_file $BASE_OUTPUT_DIR/final_results.json"
    aggregate_cmd="$aggregate_cmd --generate_latex_tables"
    aggregate_cmd="$aggregate_cmd --generate_plots"

    if run_with_retry "$aggregate_cmd 2>&1 | tee '$LOG_DIR/aggregation.log'"; then
        log_message "INFO" "Results aggregation complete"
    else
        log_message "ERROR" "Results aggregation failed"
    fi

    # Generate paper tables and figures
    local paper_cmd="python finalization/generate_paper.py"
    paper_cmd="$paper_cmd --results_file $BASE_OUTPUT_DIR/final_results.json"
    paper_cmd="$paper_cmd --output_dir $BASE_OUTPUT_DIR/paper_assets"

    if run_with_retry "$paper_cmd 2>&1 | tee '$LOG_DIR/paper_generation.log'"; then
        log_message "INFO" "Paper assets generated"
    else
        log_message "ERROR" "Paper generation failed"
    fi

    # Create summary report
    cat > "$BASE_OUTPUT_DIR/experiment_summary.md" << EOF
# LatentWire Experiment Results

## Experiment Configuration
- **Date**: $(date)
- **Checkpoint**: $CHECKPOINT_PATH
- **Seeds**: $SEEDS
- **Datasets**: $DATASETS
- **Output Directory**: $BASE_OUTPUT_DIR

## Phase Completion Status
1. **Statistical Rigor**: $([ -f "$RESULTS_DIR/phase1_statistical/statistical_summary.json" ] && echo "✓ Complete" || echo "✗ Failed")
2. **Linear Probe**: $([ -f "$RESULTS_DIR/phase2_linear_probe/comparison_report.json" ] && echo "✓ Complete" || echo "✗ Failed")
3. **Baselines**: $([ -f "$RESULTS_DIR/phase3_baselines/baseline_comparison.json" ] && echo "✓ Complete" || echo "✗ Failed")
4. **Efficiency**: $([ -f "$RESULTS_DIR/phase4_efficiency/efficiency_summary.json" ] && echo "✓ Complete" || echo "✗ Failed")

## Key Results
$([ -f "$BASE_OUTPUT_DIR/final_results.json" ] && python -c "
import json
with open('$BASE_OUTPUT_DIR/final_results.json') as f:
    data = json.load(f)
    if 'summary' in data:
        for key, value in data['summary'].items():
            print(f'- **{key}**: {value}')
" || echo "Results not yet available")

## Logs
- Main log: $MAIN_LOG
- Training log: $LOG_DIR/training_*.log
- Evaluation logs: $LOG_DIR/phase*.log

## Next Steps
1. Review statistical significance in: $RESULTS_DIR/phase1_statistical/statistical_summary.json
2. Check baseline comparisons in: $RESULTS_DIR/phase3_baselines/baseline_comparison.json
3. Review efficiency metrics in: $RESULTS_DIR/phase4_efficiency/efficiency_summary.json
4. Paper assets available in: $BASE_OUTPUT_DIR/paper_assets/
EOF

    log_message "INFO" "Experiment summary written to: $BASE_OUTPUT_DIR/experiment_summary.md"
}

# =============================================================================
# CLEANUP AND ERROR HANDLING
# =============================================================================

cleanup() {
    log_message "INFO" "Cleanup handler triggered"

    # Save current state
    cat > "$BASE_OUTPUT_DIR/state.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "checkpoint": "$CHECKPOINT_PATH",
    "completed_phases": [
        $([ -d "$RESULTS_DIR/phase1_statistical" ] && echo '"phase1",' || echo "")
        $([ -d "$RESULTS_DIR/phase2_linear_probe" ] && echo '"phase2",' || echo "")
        $([ -d "$RESULTS_DIR/phase3_baselines" ] && echo '"phase3",' || echo "")
        $([ -d "$RESULTS_DIR/phase4_efficiency" ] && echo '"phase4"' || echo "")
    ]
}
EOF

    # Push logs to git if available
    if command -v git &> /dev/null && [ -d .git ]; then
        git add -f "$LOG_DIR"/*.log "$BASE_OUTPUT_DIR"/*.json "$BASE_OUTPUT_DIR"/*.md 2>/dev/null || true
        git commit -m "experiment: $EXP_NAME results ($(hostname))" || true
        git push || true
    fi

    log_message "INFO" "Cleanup complete"
}

# Set up signal handlers
trap cleanup EXIT SIGINT SIGTERM

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_message "INFO" "========================================="
    log_message "INFO" "LATENTWIRE COMPREHENSIVE EXPERIMENT RUNNER"
    log_message "INFO" "========================================="
    log_message "INFO" "Experiment: $EXP_NAME"
    log_message "INFO" "Output directory: $BASE_OUTPUT_DIR"
    log_message "INFO" "Datasets: $DATASETS"
    log_message "INFO" "Seeds: $SEEDS"

    # Phase 0: Training or checkpoint setup
    phase0_training

    # Phase 1: Statistical rigor
    phase1_statistical_rigor

    # Phase 2: Linear probe baseline
    phase2_linear_probe

    # Phase 3: Fair baseline comparisons
    phase3_baselines

    # Phase 4: Efficiency measurements
    phase4_efficiency

    # Generate final report
    generate_final_report

    log_message "INFO" "========================================="
    log_message "INFO" "ALL EXPERIMENTS COMPLETE"
    log_message "INFO" "========================================="
    log_message "INFO" "Results directory: $RESULTS_DIR"
    log_message "INFO" "Summary: $BASE_OUTPUT_DIR/experiment_summary.md"
    log_message "INFO" "Paper assets: $BASE_OUTPUT_DIR/paper_assets/"

    # Print quick summary
    echo ""
    echo "Quick Summary:"
    echo "--------------"
    [ -f "$RESULTS_DIR/phase1_statistical/statistical_summary.json" ] && echo "✓ Statistical analysis complete" || echo "✗ Statistical analysis failed"
    [ -f "$RESULTS_DIR/phase2_linear_probe/comparison_report.json" ] && echo "✓ Linear probe comparison complete" || echo "✗ Linear probe failed"
    [ -f "$RESULTS_DIR/phase3_baselines/baseline_comparison.json" ] && echo "✓ Baseline comparisons complete" || echo "✗ Baselines failed"
    [ -f "$RESULTS_DIR/phase4_efficiency/efficiency_summary.json" ] && echo "✓ Efficiency measurements complete" || echo "✗ Efficiency failed"
    [ -f "$BASE_OUTPUT_DIR/final_results.json" ] && echo "✓ Final aggregation complete" || echo "✗ Aggregation failed"

    echo ""
    echo "To view results:"
    echo "  cat $BASE_OUTPUT_DIR/experiment_summary.md"
    echo ""
    echo "To monitor logs:"
    echo "  tail -f $MAIN_LOG"
}

# Run main function
main "$@"