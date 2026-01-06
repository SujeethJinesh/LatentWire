#!/usr/bin/env bash
# =============================================================================
# LATENTWIRE MASTER EXECUTION SCRIPT - RUN_ALL.sh
# =============================================================================
# This is the ULTIMATE master script that combines ALL shell scripts into ONE
# Handles both local and HPC execution with intelligent environment detection
# Includes all experiment phases, monitoring, SLURM submission logic inline
#
# Usage:
#   bash RUN_ALL.sh [COMMAND] [OPTIONS]
#
# Commands:
#   train        - Run training only
#   eval         - Run evaluation only
#   experiment   - Run full experiment pipeline
#   monitor      - Monitor running experiments
#   test         - Run test suite
#   slurm        - Submit to SLURM cluster
#   quick        - Quick start with minimal samples
#   finalize     - Run paper finalization experiments
#   compile      - Compile paper LaTeX
#   help         - Show this help message
#
# Environment Variables:
#   EXECUTION_MODE   - "local" or "hpc" (auto-detected if not set)
#   NUM_GPUS        - Number of GPUs to use (default: 4 on HPC, 1 locally)
#   SKIP_PHASES     - Space-separated list of phases to skip
#   CHECKPOINT_PATH - Path to existing checkpoint (skips training)
# =============================================================================

set -euo pipefail

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Script metadata
SCRIPT_VERSION="2.0.0"
SCRIPT_NAME="RUN_ALL.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Detect execution environment
detect_environment() {
    if [[ -d "/projects/m000066" ]]; then
        echo "hpc"
    elif [[ -d "/home/sjinesh" ]] && command -v sbatch &>/dev/null; then
        echo "hpc"
    else
        echo "local"
    fi
}

EXECUTION_MODE="${EXECUTION_MODE:-$(detect_environment)}"

# Environment-specific configuration
if [[ "$EXECUTION_MODE" == "hpc" ]]; then
    # HPC Marlowe configuration
    WORK_DIR="/projects/m000066/sujinesh/LatentWire"
    DEFAULT_NUM_GPUS=4
    DEFAULT_PARTITION="preempt"
    DEFAULT_ACCOUNT="marlowe-m000066"
    DEFAULT_TIME="12:00:00"
    DEFAULT_MEM="256GB"
    USE_SLURM=true
else
    # Local development configuration
    WORK_DIR="${SCRIPT_DIR}"
    DEFAULT_NUM_GPUS=1
    USE_SLURM=false
fi

# Common configuration
OUTPUT_BASE="${OUTPUT_BASE:-runs}"
LOG_BASE="${LOG_BASE:-${OUTPUT_BASE}/logs}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${OUTPUT_BASE}/checkpoints}"
RESULTS_BASE="${RESULTS_BASE:-${OUTPUT_BASE}/results}"

# Model configuration
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Training configuration
TRAINING_DATASET="${TRAINING_DATASET:-squad}"
TRAINING_SAMPLES="${TRAINING_SAMPLES:-87599}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# Evaluation configuration
EVAL_DATASETS="${EVAL_DATASETS:-sst2 agnews trec squad}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
SEEDS="${SEEDS:-42 123 456}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

# Hardware configuration
NUM_GPUS="${NUM_GPUS:-$DEFAULT_NUM_GPUS}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local color=""

    case "$level" in
        INFO) color="$GREEN" ;;
        WARN) color="$YELLOW" ;;
        ERROR) color="$RED" ;;
        DEBUG) color="$CYAN" ;;
        PHASE) color="$MAGENTA" ;;
        *) color="$NC" ;;
    esac

    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${color}[$level]${NC} $message"
}

print_banner() {
    local title="$1"
    echo ""
    echo "=============================================================="
    echo -e "${BOLD}$title${NC}"
    echo "=============================================================="
}

print_section() {
    local title="$1"
    echo ""
    echo -e "${CYAN}>>> $title${NC}"
    echo "--------------------------------------------------------------"
}

confirm() {
    local prompt="$1"
    read -p "$prompt (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

create_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log "DEBUG" "Created directory: $dir"
    done
}

run_with_logging() {
    local log_file="$1"
    shift
    local cmd="$*"

    log "INFO" "Running: $cmd"
    log "INFO" "Log file: $log_file"

    {
        eval "$cmd"
    } 2>&1 | tee -a "$log_file"

    return ${PIPESTATUS[0]}
}

run_with_retry() {
    local max_attempts="${MAX_ATTEMPTS:-3}"
    local attempt=1
    local cmd="$*"

    while [ $attempt -le $max_attempts ]; do
        log "INFO" "Attempt $attempt of $max_attempts"
        if eval "$cmd"; then
            return 0
        else
            log "WARN" "Command failed on attempt $attempt"
            attempt=$((attempt + 1))
            [ $attempt -le $max_attempts ] && sleep 10
        fi
    done

    log "ERROR" "Command failed after $max_attempts attempts"
    return 1
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        log "ERROR" "Required command not found: $cmd"
        return 1
    fi
}

check_python_module() {
    local module="$1"
    if ! python -c "import $module" &>/dev/null 2>&1; then
        log "ERROR" "Required Python module not found: $module"
        return 1
    fi
}

get_gpu_info() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    elif [[ -f /proc/driver/nvidia/version ]]; then
        cat /proc/driver/nvidia/version || true
    else
        echo "No NVIDIA GPU detected"
    fi
}

find_latest_checkpoint() {
    local search_dir="${1:-$CHECKPOINT_BASE}"
    local latest=""

    if [[ -d "$search_dir" ]]; then
        latest=$(find "$search_dir" -name "epoch*" -type d | sort -V | tail -1)
    fi

    echo "$latest"
}

# =============================================================================
# ENVIRONMENT SETUP FUNCTIONS
# =============================================================================

setup_python_environment() {
    log "PHASE" "Setting up Python environment"

    export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=true

    # PyTorch optimizations
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    export CUDA_LAUNCH_BLOCKING=0

    # MPS fallback for Mac
    if [[ "$(uname)" == "Darwin" ]]; then
        export PYTORCH_ENABLE_MPS_FALLBACK=1
    fi

    # Thread optimizations
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

    log "INFO" "Python environment configured"
    log "DEBUG" "PYTHONPATH: $PYTHONPATH"
}

validate_environment() {
    log "PHASE" "Validating environment"

    local errors=0

    # Check Python
    if ! check_command "python"; then
        ((errors++))
    else
        local python_version=$(python --version 2>&1)
        log "INFO" "Python: $python_version"
    fi

    # Check critical Python packages
    for module in torch transformers datasets latentwire; do
        if ! check_python_module "$module"; then
            ((errors++))
        else
            log "DEBUG" "Found Python module: $module"
        fi
    done

    # Check GPU availability
    log "INFO" "GPU Information:"
    get_gpu_info | while IFS= read -r line; do
        log "INFO" "  $line"
    done

    # Check Git
    if check_command "git"; then
        local git_status=$(git status --short 2>/dev/null | head -5 || echo "Not a git repository")
        log "DEBUG" "Git status: $git_status"
    fi

    if [[ $errors -gt 0 ]]; then
        log "ERROR" "Environment validation failed with $errors errors"
        return 1
    fi

    log "INFO" "Environment validation successful"
}

# =============================================================================
# SLURM FUNCTIONS
# =============================================================================

generate_slurm_script() {
    local job_name="$1"
    local command="$2"
    local output_file="$3"

    cat > "$output_file" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --gpus=${NUM_GPUS}
#SBATCH --account=${DEFAULT_ACCOUNT}
#SBATCH --partition=${DEFAULT_PARTITION}
#SBATCH --time=${DEFAULT_TIME}
#SBATCH --mem=${DEFAULT_MEM}
#SBATCH --output=${WORK_DIR}/runs/${job_name}_%j.log
#SBATCH --error=${WORK_DIR}/runs/${job_name}_%j.err

# =============================================================================
# SLURM Job: ${job_name}
# Generated: $(date)
# =============================================================================

cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "GPUs: \$CUDA_VISIBLE_DEVICES"
echo "Start time: \$(date)"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Pull latest code
git pull || true

# Create directories
mkdir -p runs logs results

# Run command
${command}

# Push results
git add -A
git commit -m "results: ${job_name} (SLURM job \$SLURM_JOB_ID)" || true
git push || true

echo "=============================================================="
echo "Job completed at \$(date)"
echo "=============================================================="
EOF

    log "INFO" "Generated SLURM script: $output_file"
}

submit_slurm_job() {
    local script_path="$1"

    if [[ ! -f "$script_path" ]]; then
        log "ERROR" "SLURM script not found: $script_path"
        return 1
    fi

    local output=$(sbatch "$script_path" 2>&1)
    local job_id=$(echo "$output" | grep -oE '[0-9]+' | head -1)

    if [[ -n "$job_id" ]]; then
        log "INFO" "SLURM job submitted: $job_id"
        echo "$job_id"
        return 0
    else
        log "ERROR" "Failed to submit SLURM job: $output"
        return 1
    fi
}

monitor_slurm_job() {
    local job_id="$1"
    local log_file="${WORK_DIR}/runs/*_${job_id}.log"

    log "INFO" "Monitoring SLURM job: $job_id"

    # Wait for log file
    local waited=0
    while [[ ! -f $log_file ]] && [[ $waited -lt 30 ]]; do
        sleep 1
        ((waited++))
    done

    if ls $log_file 1>/dev/null 2>&1; then
        log "INFO" "Tailing log file..."
        tail -f $log_file
    else
        log "WARN" "Log file not found, job may be queued"
        log "INFO" "Check status: squeue -j $job_id"
    fi
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

run_training() {
    print_section "Training Phase"

    local output_dir="${1:-$CHECKPOINT_BASE/train_${TIMESTAMP}}"
    local log_file="$LOG_BASE/training_${TIMESTAMP}.log"

    create_directories "$output_dir" "$LOG_BASE"

    # Check for existing checkpoint
    local resume_flag=""
    if [[ -n "${CHECKPOINT_PATH:-}" ]] && [[ -d "$CHECKPOINT_PATH" ]]; then
        resume_flag="--resume_from $CHECKPOINT_PATH"
        log "INFO" "Resuming from checkpoint: $CHECKPOINT_PATH"
    fi

    local cmd="python latentwire/train.py \
        --llama_id '$SOURCE_MODEL' \
        --qwen_id '$TARGET_MODEL' \
        --dataset $TRAINING_DATASET \
        --samples $TRAINING_SAMPLES \
        --epochs $TRAINING_EPOCHS \
        --batch_size $BATCH_SIZE \
        --latent_len $LATENT_LEN \
        --d_z $D_Z \
        --output_dir $output_dir \
        --encoder_type byte \
        --sequential_models \
        --warm_anchor_text 'Answer: ' \
        --first_token_ce_weight 0.5 \
        --mixed_precision bf16 \
        --seed 42 \
        $resume_flag"

    if [[ $NUM_GPUS -gt 1 ]]; then
        cmd="torchrun --nproc_per_node=$NUM_GPUS $cmd --distributed"
    fi

    run_with_logging "$log_file" "$cmd"

    # Export checkpoint path
    export CHECKPOINT_PATH="$output_dir/epoch$(($TRAINING_EPOCHS - 1))"
    log "INFO" "Training complete. Checkpoint: $CHECKPOINT_PATH"
}

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

run_evaluation() {
    print_section "Evaluation Phase"

    local checkpoint="${1:-$CHECKPOINT_PATH}"
    local dataset="${2:-squad}"
    local output_dir="${3:-$RESULTS_BASE/eval_${TIMESTAMP}}"
    local log_file="$LOG_BASE/eval_${dataset}_${TIMESTAMP}.log"

    if [[ ! -d "$checkpoint" ]]; then
        log "ERROR" "Checkpoint not found: $checkpoint"
        return 1
    fi

    create_directories "$output_dir" "$LOG_BASE"

    # Determine evaluation script based on dataset
    local eval_script="latentwire/eval.py"
    case "$dataset" in
        sst2) eval_script="latentwire/eval_sst2.py" ;;
        agnews) eval_script="latentwire/eval_agnews.py" ;;
        trec) eval_script="telepathy/eval_telepathy_trec.py" ;;
        gsm8k) eval_script="latentwire/gsm8k_eval.py" ;;
    esac

    local cmd="python $eval_script \
        --ckpt $checkpoint \
        --dataset $dataset \
        --samples $EVAL_SAMPLES \
        --batch_size 16 \
        --output_file $output_dir/${dataset}_results.json \
        --include_baselines \
        --sequential_eval \
        --fresh_eval"

    run_with_logging "$log_file" "$cmd"

    log "INFO" "Evaluation complete for $dataset"
    log "INFO" "Results: $output_dir/${dataset}_results.json"
}

run_multi_seed_evaluation() {
    print_section "Multi-seed Evaluation"

    local checkpoint="${1:-$CHECKPOINT_PATH}"
    local output_dir="${2:-$RESULTS_BASE/multiseed_${TIMESTAMP}}"

    create_directories "$output_dir"

    for dataset in $EVAL_DATASETS; do
        for seed in $SEEDS; do
            log "INFO" "Evaluating $dataset with seed $seed"

            local seed_output="$output_dir/${dataset}_seed${seed}"
            run_evaluation "$checkpoint" "$dataset" "$seed_output"
        done
    done

    # Run statistical analysis
    log "INFO" "Running statistical analysis"
    python scripts/statistical_testing.py \
        --results_dir "$output_dir" \
        --bootstrap_samples $BOOTSTRAP_SAMPLES \
        --output_file "$output_dir/statistical_summary.json"
}

# =============================================================================
# BASELINE COMPARISON FUNCTIONS
# =============================================================================

run_linear_probe_baseline() {
    print_section "Linear Probe Baseline"

    local output_dir="${1:-$RESULTS_BASE/linear_probe_${TIMESTAMP}}"
    local log_file="$LOG_BASE/linear_probe_${TIMESTAMP}.log"

    create_directories "$output_dir"

    for dataset in $EVAL_DATASETS; do
        if [[ "$dataset" == "xsum" ]]; then
            log "INFO" "Skipping linear probe for generation task: $dataset"
            continue
        fi

        local cmd="python latentwire/linear_probe_baseline.py \
            --source_model $SOURCE_MODEL \
            --dataset $dataset \
            --layer 16 \
            --cv_folds 5 \
            --output_dir $output_dir \
            --batch_size 16"

        run_with_logging "$log_file" "$cmd"
    done
}

run_llmlingua_baseline() {
    print_section "LLMLingua Baseline"

    local output_dir="${1:-$RESULTS_BASE/llmlingua_${TIMESTAMP}}"
    local log_file="$LOG_BASE/llmlingua_${TIMESTAMP}.log"

    create_directories "$output_dir"

    local cmd="bash scripts/run_llmlingua_baseline.sh \
        --output_dir $output_dir \
        --compression_ratio 8 \
        --datasets '$EVAL_DATASETS'"

    run_with_logging "$log_file" "$cmd"
}

# =============================================================================
# EXPERIMENT ORCHESTRATION
# =============================================================================

run_full_experiment() {
    print_banner "FULL EXPERIMENT PIPELINE"

    local exp_name="${1:-full_experiment_${TIMESTAMP}}"
    local base_dir="$OUTPUT_BASE/$exp_name"

    create_directories "$base_dir" "$base_dir/checkpoint" "$base_dir/results" "$base_dir/logs"

    # Override paths for this experiment
    LOG_BASE="$base_dir/logs"
    CHECKPOINT_BASE="$base_dir/checkpoint"
    RESULTS_BASE="$base_dir/results"

    # Phase 0: Training
    if [[ -z "${SKIP_PHASES:-}" ]] || [[ ! "$SKIP_PHASES" =~ "train" ]]; then
        log "PHASE" "Phase 0: Training"
        run_training "$CHECKPOINT_BASE"
    else
        CHECKPOINT_PATH=$(find_latest_checkpoint "$CHECKPOINT_BASE")
        log "INFO" "Skipping training, using checkpoint: $CHECKPOINT_PATH"
    fi

    # Phase 1: Statistical evaluation
    if [[ -z "${SKIP_PHASES:-}" ]] || [[ ! "$SKIP_PHASES" =~ "stat" ]]; then
        log "PHASE" "Phase 1: Statistical Evaluation"
        run_multi_seed_evaluation "$CHECKPOINT_PATH" "$RESULTS_BASE/phase1_statistical"
    fi

    # Phase 2: Linear probe baseline
    if [[ -z "${SKIP_PHASES:-}" ]] || [[ ! "$SKIP_PHASES" =~ "probe" ]]; then
        log "PHASE" "Phase 2: Linear Probe Baseline"
        run_linear_probe_baseline "$RESULTS_BASE/phase2_linear_probe"
    fi

    # Phase 3: LLMLingua baseline
    if [[ -z "${SKIP_PHASES:-}" ]] || [[ ! "$SKIP_PHASES" =~ "lingua" ]]; then
        log "PHASE" "Phase 3: LLMLingua Baseline"
        run_llmlingua_baseline "$RESULTS_BASE/phase3_llmlingua"
    fi

    # Phase 4: Efficiency measurements
    if [[ -z "${SKIP_PHASES:-}" ]] || [[ ! "$SKIP_PHASES" =~ "efficiency" ]]; then
        log "PHASE" "Phase 4: Efficiency Measurements"
        run_efficiency_measurements "$CHECKPOINT_PATH" "$RESULTS_BASE/phase4_efficiency"
    fi

    # Generate final report
    generate_experiment_report "$base_dir"
}

run_efficiency_measurements() {
    print_section "Efficiency Measurements"

    local checkpoint="${1:-$CHECKPOINT_PATH}"
    local output_dir="${2:-$RESULTS_BASE/efficiency_${TIMESTAMP}}"
    local log_file="$LOG_BASE/efficiency_${TIMESTAMP}.log"

    create_directories "$output_dir"

    local cmd="python scripts/benchmark_efficiency.py \
        --checkpoint $checkpoint \
        --dataset squad \
        --samples 100 \
        --warmup_runs 3 \
        --benchmark_runs 10 \
        --measure_memory \
        --measure_latency \
        --measure_throughput \
        --output_file $output_dir/efficiency_results.json"

    run_with_logging "$log_file" "$cmd"
}

generate_experiment_report() {
    local exp_dir="$1"

    print_section "Generating Experiment Report"

    cat > "$exp_dir/experiment_report.md" << EOF
# Experiment Report

## Configuration
- **Date**: $(date)
- **Experiment**: $(basename "$exp_dir")
- **Models**: $SOURCE_MODEL → $TARGET_MODEL
- **Datasets**: $EVAL_DATASETS
- **Seeds**: $SEEDS

## Results

### Phase 1: Statistical Evaluation
$(ls -la "$exp_dir/results/phase1_statistical" 2>/dev/null || echo "Not completed")

### Phase 2: Linear Probe Baseline
$(ls -la "$exp_dir/results/phase2_linear_probe" 2>/dev/null || echo "Not completed")

### Phase 3: LLMLingua Baseline
$(ls -la "$exp_dir/results/phase3_llmlingua" 2>/dev/null || echo "Not completed")

### Phase 4: Efficiency Measurements
$(ls -la "$exp_dir/results/phase4_efficiency" 2>/dev/null || echo "Not completed")

## Logs
$(ls -la "$exp_dir/logs" 2>/dev/null || echo "No logs")

EOF

    log "INFO" "Report generated: $exp_dir/experiment_report.md"
}

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

monitor_experiments() {
    print_banner "EXPERIMENT MONITOR"

    local run_dir="${1:-$OUTPUT_BASE}"
    local refresh_interval="${2:-10}"

    while true; do
        clear
        print_banner "EXPERIMENT MONITOR"
        echo "Time: $(date)"
        echo "Directory: $run_dir"
        echo ""

        # Check running processes
        echo "Running Processes:"
        ps aux | grep -E "python|bash" | grep -v grep | head -5 || echo "  None"
        echo ""

        # Check latest logs
        echo "Latest Activity:"
        find "$run_dir" -name "*.log" -type f -mmin -5 2>/dev/null | \
            xargs -I {} sh -c 'echo "  $(basename {}): $(tail -1 {})"' | head -5
        echo ""

        # Check GPU usage
        if command -v nvidia-smi &>/dev/null; then
            echo "GPU Usage:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits | sed 's/^/  /'
        fi

        echo ""
        echo "Press Ctrl+C to exit"

        sleep "$refresh_interval"
    done
}

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

run_test_suite() {
    print_banner "TEST SUITE"

    local log_file="$LOG_BASE/tests_${TIMESTAMP}.log"
    create_directories "$LOG_BASE"

    log "INFO" "Running test suite"

    # Unit tests
    log "INFO" "Running unit tests"
    run_with_logging "$log_file" "python -m pytest tests/ -v --tb=short"

    # Integration tests
    log "INFO" "Running integration tests"
    run_with_logging "$log_file" "bash tests/test_everything.sh"

    # Memory tests
    log "INFO" "Running memory tests"
    run_with_logging "$log_file" "bash tests/test_memory_calculations.sh"

    log "INFO" "Test suite complete"
}

# =============================================================================
# QUICK START FUNCTION
# =============================================================================

run_quick_start() {
    print_banner "QUICK START"

    log "INFO" "Running quick experiment with minimal samples"

    # Override with minimal settings
    TRAINING_SAMPLES=1000
    TRAINING_EPOCHS=2
    EVAL_SAMPLES=100
    EVAL_DATASETS="squad"
    SEEDS="42"

    run_full_experiment "quick_${TIMESTAMP}"
}

# =============================================================================
# PAPER COMPILATION
# =============================================================================

compile_paper() {
    print_banner "PAPER COMPILATION"

    local paper_dir="${1:-paper}"
    local log_file="$LOG_BASE/paper_compilation_${TIMESTAMP}.log"

    create_directories "$LOG_BASE"

    if [[ ! -d "$paper_dir" ]]; then
        log "ERROR" "Paper directory not found: $paper_dir"
        return 1
    fi

    cd "$paper_dir"

    log "INFO" "Compiling LaTeX"
    run_with_logging "$log_file" "pdflatex -interaction=nonstopmode main.tex"
    run_with_logging "$log_file" "bibtex main"
    run_with_logging "$log_file" "pdflatex -interaction=nonstopmode main.tex"
    run_with_logging "$log_file" "pdflatex -interaction=nonstopmode main.tex"

    cd - >/dev/null

    log "INFO" "Paper compiled: $paper_dir/main.pdf"
}

# =============================================================================
# HELP FUNCTION
# =============================================================================

show_help() {
    cat << EOF
$SCRIPT_NAME v$SCRIPT_VERSION - Master Execution Script

USAGE:
    bash $SCRIPT_NAME [COMMAND] [OPTIONS]

COMMANDS:
    train       Run training only
    eval        Run evaluation only
    experiment  Run full experiment pipeline
    monitor     Monitor running experiments
    test        Run test suite
    slurm       Submit to SLURM cluster
    quick       Quick start with minimal samples
    finalize    Run paper finalization experiments
    compile     Compile paper LaTeX
    help        Show this help message

OPTIONS:
    --checkpoint PATH    Use existing checkpoint
    --dataset DATASET    Dataset to use
    --output DIR        Output directory
    --gpus N            Number of GPUs
    --skip-phases LIST  Space-separated phases to skip

ENVIRONMENT VARIABLES:
    EXECUTION_MODE      "local" or "hpc" (auto-detected)
    NUM_GPUS           Number of GPUs (default: 4 on HPC, 1 local)
    SKIP_PHASES        Phases to skip (train, stat, probe, lingua, efficiency)
    CHECKPOINT_PATH    Path to existing checkpoint

EXAMPLES:
    # Run full experiment
    bash $SCRIPT_NAME experiment

    # Train only
    bash $SCRIPT_NAME train

    # Evaluate with existing checkpoint
    CHECKPOINT_PATH=runs/checkpoint/epoch23 bash $SCRIPT_NAME eval

    # Quick test
    bash $SCRIPT_NAME quick

    # Submit to SLURM
    bash $SCRIPT_NAME slurm experiment

    # Monitor experiments
    bash $SCRIPT_NAME monitor

EOF
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    local command="${1:-help}"
    shift || true

    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint)
                CHECKPOINT_PATH="$2"
                shift 2
                ;;
            --dataset)
                EVAL_DATASETS="$2"
                shift 2
                ;;
            --output)
                OUTPUT_BASE="$2"
                shift 2
                ;;
            --gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --skip-phases)
                SKIP_PHASES="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    # Setup environment
    setup_python_environment

    # Execute command
    case "$command" in
        train)
            validate_environment
            run_training
            ;;

        eval|evaluate)
            validate_environment
            if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
                CHECKPOINT_PATH=$(find_latest_checkpoint)
            fi
            for dataset in $EVAL_DATASETS; do
                run_evaluation "$CHECKPOINT_PATH" "$dataset"
            done
            ;;

        experiment)
            validate_environment
            run_full_experiment
            ;;

        monitor)
            monitor_experiments "$@"
            ;;

        test)
            validate_environment
            run_test_suite
            ;;

        slurm)
            if [[ "$EXECUTION_MODE" != "hpc" ]]; then
                log "ERROR" "SLURM submission only available on HPC"
                exit 1
            fi
            local subcmd="${1:-experiment}"
            local slurm_script="/tmp/slurm_${TIMESTAMP}.sh"
            generate_slurm_script "latentwire_${subcmd}" \
                "bash $SCRIPT_DIR/$SCRIPT_NAME $subcmd" \
                "$slurm_script"

            if confirm "Submit SLURM job?"; then
                job_id=$(submit_slurm_job "$slurm_script")
                if confirm "Monitor job $job_id?"; then
                    monitor_slurm_job "$job_id"
                fi
            fi
            ;;

        quick)
            validate_environment
            run_quick_start
            ;;

        finalize)
            validate_environment
            log "INFO" "Running finalization experiments"
            bash "$SCRIPT_DIR/scripts/run_finalization.sh"
            ;;

        compile)
            compile_paper "$@"
            ;;

        help|--help|-h)
            show_help
            ;;

        *)
            log "ERROR" "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

cleanup() {
    log "INFO" "Cleanup triggered"

    # Save state
    if [[ -n "${OUTPUT_BASE:-}" ]]; then
        cat > "$OUTPUT_BASE/last_run.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "command": "$0 $@",
    "checkpoint": "${CHECKPOINT_PATH:-}",
    "pid": $$,
    "exit_code": $?
}
EOF
    fi

    # Push to git if available
    if [[ "$EXECUTION_MODE" == "hpc" ]] && command -v git &>/dev/null; then
        git add -A 2>/dev/null || true
        git commit -m "checkpoint: interrupted run $(date)" 2>/dev/null || true
        git push 2>/dev/null || true
    fi
}

trap cleanup EXIT SIGINT SIGTERM

# =============================================================================
# ENTRY POINT
# =============================================================================

# Check if sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Change to script directory
    cd "$SCRIPT_DIR"

    # Run main function
    main "$@"
fi