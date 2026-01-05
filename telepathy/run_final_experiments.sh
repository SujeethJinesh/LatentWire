#!/usr/bin/env bash
# =============================================================================
# FINAL INTEGRATED EXPERIMENT SUITE FOR LATENTWIRE
# =============================================================================
# This script orchestrates ALL experiments required for the paper submission.
# It includes validation, comprehensive experiments, and proper error handling.
#
# Features:
# - Memory-safe configurations for HPC cluster
# - 3 seeds (42, 123, 456) for statistical rigor
# - 3 model families with different sizes
# - All required experiments from revision plan
# - Comprehensive logging with tee
# - Validation checks before main experiments
#
# Usage:
#   # Local testing (small scale):
#   bash telepathy/run_final_experiments.sh --test
#
#   # HPC submission:
#   sbatch telepathy/submit_final_experiments.slurm
#
# =============================================================================

set -e  # Exit on error

# Configuration
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-runs/final_experiments}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="final_suite_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"

# Parse command line arguments
TEST_MODE=false
SKIP_VALIDATION=false
PHASE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="${HF_HOME:-/projects/m000066/sujinesh/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/projects/m000066/sujinesh/.cache/huggingface}"
export CUDA_LAUNCH_BLOCKING=0  # Better performance

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/results"
mkdir -p "$OUTPUT_DIR/figures"

# Set up logging
MAIN_LOG="$OUTPUT_DIR/logs/main_${TIMESTAMP}.log"
VALIDATION_LOG="$OUTPUT_DIR/logs/validation_${TIMESTAMP}.log"
EXPERIMENT_LOG="$OUTPUT_DIR/logs/experiments_${TIMESTAMP}.log"

echo "=============================================================="
echo "FINAL INTEGRATED EXPERIMENT SUITE"
echo "=============================================================="
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo "Test mode: $TEST_MODE"
echo "Skip validation: $SKIP_VALIDATION"
echo "Phase: ${PHASE:-all}"
echo "=============================================================="

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Function to run command with logging
run_with_log() {
    local cmd="$1"
    local log_file="$2"
    local description="$3"

    log_message "Starting: $description"
    echo "Command: $cmd" >> "$log_file"
    echo "Started at: $(date)" >> "$log_file"
    echo "---" >> "$log_file"

    if eval "$cmd" 2>&1 | tee -a "$log_file"; then
        log_message "✓ Completed: $description"
        return 0
    else
        log_message "✗ Failed: $description"
        return 1
    fi
}

# =============================================================================
# PHASE 0: VALIDATION CHECKS
# =============================================================================
run_validation() {
    log_message "=== PHASE 0: VALIDATION CHECKS ==="

    if [ "$SKIP_VALIDATION" = true ]; then
        log_message "Skipping validation (--skip-validation flag)"
        return 0
    fi

    # GPU check
    log_message "Checking GPU availability..."
    if ! nvidia-smi &> /dev/null; then
        log_message "WARNING: No GPUs detected. Some experiments may fail."
    else
        nvidia-smi | tee -a "$VALIDATION_LOG"
    fi

    # Python environment check
    log_message "Python environment:"
    python --version | tee -a "$VALIDATION_LOG"

    # Import checks
    log_message "Checking imports..."
    python -c "
import torch
import transformers
import numpy as np
from pathlib import Path
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
" 2>&1 | tee -a "$VALIDATION_LOG"

    # Run validation script if it exists
    if [ -f "telepathy/validate_setup.py" ]; then
        log_message "Running validation script..."
        run_with_log \
            "python telepathy/validate_setup.py --output_dir $OUTPUT_DIR/validation" \
            "$VALIDATION_LOG" \
            "Validation tests"
    fi

    log_message "Validation completed"
    return 0
}

# =============================================================================
# PHASE 1: STATISTICAL RIGOR - FULL TEST SETS
# =============================================================================
run_phase1_statistical() {
    log_message "=== PHASE 1: STATISTICAL RIGOR ==="

    local seeds=(42 123 456)
    local tasks=("sst2" "agnews" "trec")

    if [ "$TEST_MODE" = true ]; then
        seeds=(42)
        tasks=("sst2")
    fi

    for seed in "${seeds[@]}"; do
        for task in "${tasks[@]}"; do
            local exp_name="phase1_${task}_seed${seed}"
            local exp_dir="$OUTPUT_DIR/results/$exp_name"
            mkdir -p "$exp_dir"

            run_with_log \
                "python telepathy/unified_cross_model_experiments.py \
                    --task $task \
                    --seed $seed \
                    --compression_factor 8 \
                    --test_samples -1 \
                    --train_samples 1000 \
                    --batch_size 4 \
                    --epochs 10 \
                    --learning_rate 1e-4 \
                    --sender_model meta-llama/Llama-3.2-1B-Instruct \
                    --receiver_model Qwen/Qwen2.5-1.5B-Instruct \
                    --output_dir $exp_dir \
                    --save_predictions \
                    --compute_bootstrap_ci \
                    --n_bootstrap 1000" \
                "$OUTPUT_DIR/logs/${exp_name}.log" \
                "Statistical experiment: $task (seed $seed)"
        done
    done

    # Aggregate results across seeds
    log_message "Aggregating multi-seed results..."
    python -c "
import json
from pathlib import Path
from scripts.statistical_testing import aggregate_multi_seed_results

output_dir = Path('$OUTPUT_DIR/results')
tasks = ['sst2', 'agnews', 'trec'] if not $TEST_MODE else ['sst2']
seeds = [42, 123, 456] if not $TEST_MODE else [42]

for task in tasks:
    results = []
    for seed in seeds:
        result_file = output_dir / f'phase1_{task}_seed{seed}' / 'results.json'
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))

    if results:
        aggregated = aggregate_multi_seed_results(results)
        with open(output_dir / f'phase1_{task}_aggregated.json', 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f'Aggregated results for {task}: {aggregated}')
"
}

# =============================================================================
# PHASE 2: LINEAR PROBE BASELINE
# =============================================================================
run_phase2_linear_probe() {
    log_message "=== PHASE 2: LINEAR PROBE BASELINE ==="

    local models=("meta-llama/Llama-3.2-1B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3")
    local layers=(8 16 24 31)
    local tasks=("sst2" "agnews")

    if [ "$TEST_MODE" = true ]; then
        models=("meta-llama/Llama-3.2-1B-Instruct")
        layers=(16)
        tasks=("sst2")
    fi

    for model in "${models[@]}"; do
        for layer in "${layers[@]}"; do
            for task in "${tasks[@]}"; do
                local model_name=$(echo "$model" | sed 's/.*\///; s/-/_/g')
                local exp_name="phase2_probe_${model_name}_L${layer}_${task}"
                local exp_dir="$OUTPUT_DIR/results/$exp_name"
                mkdir -p "$exp_dir"

                run_with_log \
                    "python telepathy/linear_probe_baseline.py \
                        --model_id $model \
                        --layer_index $layer \
                        --task $task \
                        --train_samples 1000 \
                        --test_samples -1 \
                        --batch_size 32 \
                        --epochs 10 \
                        --learning_rate 1e-3 \
                        --output_dir $exp_dir \
                        --save_predictions" \
                    "$OUTPUT_DIR/logs/${exp_name}.log" \
                    "Linear probe: $model_name layer $layer on $task"
            done
        done
    done
}

# =============================================================================
# PHASE 3: FAIR BASELINE COMPARISONS
# =============================================================================
run_phase3_baselines() {
    log_message "=== PHASE 3: FAIR BASELINE COMPARISONS ==="

    local baselines=("random" "majority" "telepathy" "llmlingua" "direct_prompting")
    local tasks=("sst2" "agnews" "trec")

    if [ "$TEST_MODE" = true ]; then
        baselines=("random" "telepathy")
        tasks=("sst2")
    fi

    for baseline in "${baselines[@]}"; do
        for task in "${tasks[@]}"; do
            local exp_name="phase3_${baseline}_${task}"
            local exp_dir="$OUTPUT_DIR/results/$exp_name"
            mkdir -p "$exp_dir"

            case $baseline in
                "telepathy")
                    run_with_log \
                        "python telepathy/unified_cross_model_experiments.py \
                            --task $task \
                            --compression_factor 8 \
                            --test_samples 500 \
                            --train_samples 1000 \
                            --batch_size 4 \
                            --output_dir $exp_dir" \
                        "$OUTPUT_DIR/logs/${exp_name}.log" \
                        "Telepathy baseline: $task"
                    ;;
                "llmlingua")
                    run_with_log \
                        "python telepathy/llmlingua_baseline.py \
                            --task $task \
                            --compression_rate 0.125 \
                            --test_samples 500 \
                            --output_dir $exp_dir" \
                        "$OUTPUT_DIR/logs/${exp_name}.log" \
                        "LLMLingua baseline: $task"
                    ;;
                "direct_prompting")
                    run_with_log \
                        "python telepathy/direct_prompting_baseline.py \
                            --task $task \
                            --model meta-llama/Llama-3.1-1B-Instruct \
                            --test_samples 500 \
                            --output_dir $exp_dir" \
                        "$OUTPUT_DIR/logs/${exp_name}.log" \
                        "Direct prompting baseline: $task"
                    ;;
                *)
                    run_with_log \
                        "python telepathy/simple_baselines.py \
                            --baseline $baseline \
                            --task $task \
                            --test_samples 500 \
                            --output_dir $exp_dir" \
                        "$OUTPUT_DIR/logs/${exp_name}.log" \
                        "$baseline baseline: $task"
                    ;;
            esac
        done
    done
}

# =============================================================================
# PHASE 4: LATENCY MEASUREMENTS
# =============================================================================
run_phase4_latency() {
    log_message "=== PHASE 4: LATENCY MEASUREMENTS ==="

    local compression_factors=(4 8 16 32)
    local batch_sizes=(1 4 8 16)

    if [ "$TEST_MODE" = true ]; then
        compression_factors=(8)
        batch_sizes=(1 4)
    fi

    for cf in "${compression_factors[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            local exp_name="phase4_latency_cf${cf}_bs${bs}"
            local exp_dir="$OUTPUT_DIR/results/$exp_name"
            mkdir -p "$exp_dir"

            run_with_log \
                "python telepathy/measure_latency.py \
                    --compression_factor $cf \
                    --batch_size $bs \
                    --num_iterations 100 \
                    --warmup_iterations 10 \
                    --sender_model meta-llama/Llama-3.2-1B-Instruct \
                    --receiver_model Qwen/Qwen2.5-1.5B-Instruct \
                    --output_dir $exp_dir" \
                "$OUTPUT_DIR/logs/${exp_name}.log" \
                "Latency measurement: CF=$cf, BS=$bs"
        done
    done
}

# =============================================================================
# PHASE 5: GENERATION TASK (XSUM)
# =============================================================================
run_phase5_generation() {
    log_message "=== PHASE 5: GENERATION TASK (XSUM) ==="

    local exp_name="phase5_xsum_generation"
    local exp_dir="$OUTPUT_DIR/results/$exp_name"
    mkdir -p "$exp_dir"

    local samples=500
    if [ "$TEST_MODE" = true ]; then
        samples=10
    fi

    run_with_log \
        "python telepathy/xsum_generation_eval.py \
            --compression_factor 8 \
            --test_samples $samples \
            --train_samples 1000 \
            --batch_size 2 \
            --max_length 128 \
            --sender_model meta-llama/Llama-3.1-1B-Instruct \
            --receiver_model Qwen/Qwen2.5-1.5B-Instruct \
            --output_dir $exp_dir \
            --compute_rouge \
            --save_generations" \
        "$OUTPUT_DIR/logs/${exp_name}.log" \
        "XSUM generation evaluation"
}

# =============================================================================
# PHASE 6: MODEL SIZE ABLATIONS
# =============================================================================
run_phase6_ablations() {
    log_message "=== PHASE 6: MODEL SIZE ABLATIONS ==="

    local model_configs=(
        "meta-llama/Llama-3.1-1B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,small"
        "meta-llama/Llama-3.1-3B-Instruct,Qwen/Qwen2.5-3B-Instruct,medium"
        "meta-llama/Meta-Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,large"
    )

    if [ "$TEST_MODE" = true ]; then
        model_configs=("meta-llama/Llama-3.1-1B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,small")
    fi

    for config in "${model_configs[@]}"; do
        IFS=',' read -r sender receiver size <<< "$config"
        local exp_name="phase6_ablation_${size}"
        local exp_dir="$OUTPUT_DIR/results/$exp_name"
        mkdir -p "$exp_dir"

        # Set batch size based on model size
        local batch_size=4
        if [ "$size" = "medium" ]; then
            batch_size=2
        elif [ "$size" = "large" ]; then
            batch_size=1
        fi

        run_with_log \
            "python telepathy/unified_cross_model_experiments.py \
                --task sst2 \
                --compression_factor 8 \
                --test_samples 500 \
                --train_samples 1000 \
                --batch_size $batch_size \
                --epochs 5 \
                --sender_model $sender \
                --receiver_model $receiver \
                --output_dir $exp_dir \
                --save_checkpoints" \
            "$OUTPUT_DIR/logs/${exp_name}.log" \
            "Model size ablation: $size"
    done
}

# =============================================================================
# FINAL ANALYSIS AND REPORTING
# =============================================================================
generate_final_report() {
    log_message "=== GENERATING FINAL REPORT ==="

    local report_file="$OUTPUT_DIR/FINAL_REPORT.md"

    python -c "
import json
import numpy as np
from pathlib import Path
from datetime import datetime

output_dir = Path('$OUTPUT_DIR')
results_dir = output_dir / 'results'
report_file = output_dir / 'FINAL_REPORT.md'

# Collect all results
all_results = {}
for result_file in results_dir.glob('*/results.json'):
    exp_name = result_file.parent.name
    with open(result_file) as f:
        all_results[exp_name] = json.load(f)

# Generate report
with open(report_file, 'w') as f:
    f.write('# FINAL EXPERIMENT REPORT\\n')
    f.write(f'Generated: {datetime.now().isoformat()}\\n\\n')

    # Phase 1: Statistical results
    f.write('## Phase 1: Statistical Rigor\\n\\n')
    f.write('| Task | Telepathy Acc | 95% CI | Baseline | p-value |\\n')
    f.write('|------|---------------|--------|----------|---------|\\n')

    for task in ['sst2', 'agnews', 'trec']:
        agg_file = results_dir / f'phase1_{task}_aggregated.json'
        if agg_file.exists():
            with open(agg_file) as af:
                agg = json.load(af)
                acc = agg.get('mean_accuracy', 0)
                ci = agg.get('ci_95', [0, 0])
                f.write(f'| {task.upper()} | {acc:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | - | - |\\n')

    f.write('\\n')

    # Phase 2: Linear probe results
    f.write('## Phase 2: Linear Probe Baseline\\n\\n')
    f.write('| Model | Layer | Task | Accuracy |\\n')
    f.write('|-------|-------|------|----------|\\n')

    for exp_name, results in all_results.items():
        if 'phase2_probe' in exp_name:
            parts = exp_name.split('_')
            model = parts[2]
            layer = parts[3].replace('L', '')
            task = parts[4] if len(parts) > 4 else 'unknown'
            acc = results.get('accuracy', 0)
            f.write(f'| {model} | {layer} | {task} | {acc:.3f} |\\n')

    f.write('\\n')

    # Summary statistics
    f.write('## Summary Statistics\\n\\n')
    f.write(f'- Total experiments: {len(all_results)}\\n')
    f.write(f'- Successful completions: {sum(1 for r in all_results.values() if r.get(\"status\") == \"completed\")}\\n')
    f.write(f'- Average runtime: {np.mean([r.get(\"runtime_seconds\", 0) for r in all_results.values()]):.1f}s\\n')

    print(f'Report saved to {report_file}')

# Also create a JSON summary
summary = {
    'timestamp': '$TIMESTAMP',
    'experiment_name': '$EXPERIMENT_NAME',
    'output_dir': '$OUTPUT_DIR',
    'num_experiments': len(all_results),
    'phases_completed': [],
    'best_results': {}
}

# Find best results per task
for task in ['sst2', 'agnews', 'trec']:
    task_results = {k: v for k, v in all_results.items() if task in k}
    if task_results:
        best_exp = max(task_results.items(), key=lambda x: x[1].get('accuracy', 0))
        summary['best_results'][task] = {
            'experiment': best_exp[0],
            'accuracy': best_exp[1].get('accuracy', 0)
        }

with open(output_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to {output_dir / \"summary.json\"}')"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

log_message "Starting Final Experiment Suite"

# Phase 0: Validation
if ! run_validation; then
    log_message "ERROR: Validation failed. Exiting."
    exit 1
fi

# Execute phases based on command line argument
if [ -z "$PHASE" ] || [ "$PHASE" = "all" ]; then
    # Run all phases
    run_phase1_statistical
    run_phase2_linear_probe
    run_phase3_baselines
    run_phase4_latency
    run_phase5_generation
    run_phase6_ablations
else
    # Run specific phase
    case $PHASE in
        1) run_phase1_statistical ;;
        2) run_phase2_linear_probe ;;
        3) run_phase3_baselines ;;
        4) run_phase4_latency ;;
        5) run_phase5_generation ;;
        6) run_phase6_ablations ;;
        *)
            log_message "ERROR: Invalid phase $PHASE"
            exit 1
            ;;
    esac
fi

# Generate final report
generate_final_report

# Archive results
log_message "Archiving results..."
tar -czf "$OUTPUT_DIR.tar.gz" "$OUTPUT_DIR" \
    --exclude="*.pth" \
    --exclude="*.safetensors" \
    --exclude="*.bin"

log_message "=============================================================="
log_message "EXPERIMENT SUITE COMPLETED SUCCESSFULLY"
log_message "Results directory: $OUTPUT_DIR"
log_message "Archive: $OUTPUT_DIR.tar.gz"
log_message "Final report: $OUTPUT_DIR/FINAL_REPORT.md"
log_message "=============================================================="

# Print summary statistics
python -c "
import json
from pathlib import Path

summary_file = Path('$OUTPUT_DIR') / 'summary.json'
if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)

    print('\\nEXPERIMENT SUMMARY:')
    print(f'Total experiments: {summary[\"num_experiments\"]}')
    print('\\nBest results per task:')
    for task, results in summary.get('best_results', {}).items():
        print(f'  {task.upper()}: {results[\"accuracy\"]:.3f} ({results[\"experiment\"]})')
"

exit 0