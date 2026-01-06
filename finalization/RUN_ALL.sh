#!/bin/bash
# =============================================================================
# LATENTWIRE COMPREHENSIVE EXPERIMENT RUNNER - ULTIMATE CONSOLIDATED SCRIPT
# =============================================================================
# This is the ONLY script needed for ALL LatentWire operations.
# It consolidates ALL functionality from all other scripts.
#
# Features:
#   - Environment auto-detection (HPC vs Local)
#   - SLURM job submission and monitoring
#   - All experiment phases (training, evaluation, baselines, efficiency)
#   - Testing and validation
#   - Paper compilation
#   - Result aggregation and analysis
#   - DDP training support
#   - Memory profiling and benchmarking
#
# Usage:
#   bash RUN_ALL.sh [COMMAND] [OPTIONS]
#
# Commands:
#   train         Run training only
#   eval          Run evaluation only
#   experiment    Run full experiment pipeline
#   test          Run test suite
#   monitor       Monitor running experiments
#   slurm         Submit to SLURM cluster
#   quick         Quick start with minimal samples
#   finalize      Run paper finalization experiments
#   compile       Compile paper LaTeX
#   ddp           Run DDP training
#   benchmark     Run efficiency benchmarks
#   help          Show help message
#
# Options:
#   --local       Force local execution
#   --hpc         Force HPC execution
#   --phase N     Run specific phase (1-4)
#   --dataset D   Run specific dataset
#   --skip-train  Skip training, use existing checkpoint
#   --checkpoint  Path to checkpoint
#   --gpus N      Number of GPUs
#   --debug       Enable debug mode
#   --dry-run     Show what would be executed
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Script metadata
SCRIPT_VERSION="3.0.0"
SCRIPT_NAME="RUN_ALL.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default configuration
COMMAND="help"
EXECUTION_MODE="auto"
DEBUG_MODE="no"
DRY_RUN="no"
SKIP_TRAINING="no"
CHECKPOINT_PATH=""
SPECIFIC_PHASE=""
SPECIFIC_DATASET=""
NUM_GPUS=""
INTERACTIVE="yes"

# Experiment configuration
EXP_NAME="latentwire_unified"
BASE_OUTPUT_DIR="runs/${EXP_NAME}_${TIMESTAMP}"

# Model configuration
SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Training hyperparameters
TRAINING_DATASET="squad"
TRAINING_SAMPLES=87599
TRAINING_EPOCHS=24
LATENT_LEN=32
D_Z=256
BATCH_SIZE=8
EVAL_BATCH_SIZE=16

# Evaluation configuration
SEEDS="42 123 456"
DATASETS="sst2 agnews trec squad"
BOOTSTRAP_SAMPLES=10000

# HPC configuration
SLURM_ACCOUNT="marlowe-m000066"
SLURM_PARTITION="preempt"
SLURM_TIME="12:00:00"
SLURM_MEMORY="256GB"
SLURM_GPUS=4

# Environment paths
if [[ -d "/projects/m000066" ]]; then
    WORK_DIR="/projects/m000066/sujinesh/LatentWire"
    LOG_BASE="/projects/m000066/sujinesh/LatentWire/runs"
    IS_HPC="yes"
else
    WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    LOG_BASE="$WORK_DIR/runs"
    IS_HPC="no"
fi

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }
print_debug() { [[ "$DEBUG_MODE" == "yes" ]] && echo -e "${CYAN}[D]${NC} $1"; }

print_header() {
    echo ""
    echo -e "${CYAN}=============================================================="
    echo -e "$1"
    echo -e "==============================================================${NC}"
    echo ""
}

print_subheader() {
    echo ""
    echo -e "${MAGENTA}----------------------------------------------------------"
    echo -e "$1"
    echo -e "----------------------------------------------------------${NC}"
    echo ""
}

show_help() {
    cat << EOF
${BOLD}LatentWire Unified Execution Script v${SCRIPT_VERSION}${NC}

${BOLD}USAGE:${NC}
    bash RUN_ALL.sh [COMMAND] [OPTIONS]

${BOLD}COMMANDS:${NC}
    ${GREEN}train${NC}         Run training only
    ${GREEN}eval${NC}          Run evaluation only
    ${GREEN}experiment${NC}    Run full experiment pipeline
    ${GREEN}test${NC}          Run comprehensive test suite
    ${GREEN}monitor${NC}       Monitor running experiments
    ${GREEN}slurm${NC}         Submit job to SLURM cluster
    ${GREEN}quick${NC}         Quick start with minimal samples
    ${GREEN}finalize${NC}      Run paper finalization experiments
    ${GREEN}compile${NC}       Compile paper LaTeX to PDF
    ${GREEN}ddp${NC}           Run distributed data parallel training
    ${GREEN}benchmark${NC}     Run efficiency benchmarks
    ${GREEN}validate${NC}      Validate setup and dependencies
    ${GREEN}clean${NC}         Clean up temporary files
    ${GREEN}help${NC}          Show this help message

${BOLD}OPTIONS:${NC}
    ${BLUE}--local${NC}           Force local execution
    ${BLUE}--hpc${NC}             Force HPC/SLURM execution
    ${BLUE}--phase N${NC}         Run specific phase (1-4)
    ${BLUE}--dataset NAME${NC}    Run specific dataset
    ${BLUE}--skip-train${NC}      Skip training, use existing checkpoint
    ${BLUE}--checkpoint PATH${NC} Specify checkpoint path
    ${BLUE}--gpus N${NC}          Number of GPUs to use
    ${BLUE}--batch-size N${NC}    Batch size for training/eval
    ${BLUE}--debug${NC}           Enable debug output
    ${BLUE}--dry-run${NC}         Show what would be executed
    ${BLUE}--no-interactive${NC}  Skip confirmation prompts

${BOLD}EXPERIMENT PHASES:${NC}
    ${YELLOW}Phase 1${NC}: Statistical rigor (multiple seeds, bootstrap CI)
    ${YELLOW}Phase 2${NC}: Linear probe baseline comparisons
    ${YELLOW}Phase 3${NC}: Fair baseline comparisons (LLMLingua, token-budget)
    ${YELLOW}Phase 4${NC}: Efficiency measurements (latency, memory, throughput)

${BOLD}DATASETS:${NC}
    ${CYAN}sst2${NC}     Sentiment analysis (Stanford Sentiment Treebank)
    ${CYAN}agnews${NC}   News classification (AG News)
    ${CYAN}trec${NC}     Question classification (TREC)
    ${CYAN}squad${NC}    Question answering (SQuAD)
    ${CYAN}xsum${NC}     Summarization (XSum) - optional

${BOLD}EXAMPLES:${NC}
    # Run full experiments on HPC
    bash RUN_ALL.sh experiment --hpc

    # Quick local test
    bash RUN_ALL.sh quick --local

    # Run specific phase on specific dataset
    bash RUN_ALL.sh experiment --phase 1 --dataset sst2

    # Submit SLURM job for finalization
    bash RUN_ALL.sh slurm finalize

    # Monitor running jobs
    bash RUN_ALL.sh monitor

    # Run DDP training with 4 GPUs
    bash RUN_ALL.sh ddp --gpus 4

${BOLD}ENVIRONMENT:${NC}
    Current mode: $([ "$IS_HPC" == "yes" ] && echo "HPC" || echo "Local")
    Work dir: $WORK_DIR
    Python: $(command -v python &>/dev/null && python --version 2>&1 | head -1 || echo "Not found")
    CUDA: $(command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=count --format=csv,noheader | wc -l || echo "0") GPU(s)

EOF
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

# Store original arguments for SLURM passthrough
ORIGINAL_ARGS="$@"

# Parse command first
if [[ $# -gt 0 ]]; then
    case "$1" in
        train|eval|experiment|test|monitor|slurm|quick|finalize|compile|ddp|benchmark|validate|clean|help)
            COMMAND="$1"
            shift
            ;;
        *)
            if [[ "$1" != "--"* ]]; then
                print_error "Unknown command: $1"
                show_help
                exit 1
            fi
            ;;
    esac
fi

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            EXECUTION_MODE="local"
            shift
            ;;
        --hpc)
            EXECUTION_MODE="hpc"
            shift
            ;;
        --phase)
            SPECIFIC_PHASE="$2"
            shift 2
            ;;
        --dataset)
            SPECIFIC_DATASET="$2"
            shift 2
            ;;
        --skip-train|--skip-training)
            SKIP_TRAINING="yes"
            shift
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            SKIP_TRAINING="yes"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="yes"
            set -x
            shift
            ;;
        --dry-run)
            DRY_RUN="yes"
            shift
            ;;
        --no-interactive)
            INTERACTIVE="no"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# ENVIRONMENT DETECTION AND VALIDATION
# =============================================================================

detect_environment() {
    print_debug "Detecting execution environment..."

    if [[ "$EXECUTION_MODE" == "auto" ]]; then
        if [[ "$IS_HPC" == "yes" ]]; then
            EXECUTION_MODE="hpc"
        else
            EXECUTION_MODE="local"
        fi
    fi

    print_debug "Execution mode: $EXECUTION_MODE"

    # Set GPU count based on environment
    if [[ -z "$NUM_GPUS" ]]; then
        if [[ "$EXECUTION_MODE" == "hpc" ]]; then
            NUM_GPUS=$SLURM_GPUS
        else
            # Try to detect available GPUs
            if command -v nvidia-smi &>/dev/null; then
                NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
            else
                NUM_GPUS=0
            fi
        fi
    fi

    print_debug "GPUs to use: $NUM_GPUS"
}

validate_environment() {
    print_header "ENVIRONMENT VALIDATION"

    # Check Python
    if ! command -v python &>/dev/null; then
        print_error "Python not found!"
        exit 1
    fi
    print_status "Python: $(python --version 2>&1)"

    # Check PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        print_error "PyTorch not installed!"
        exit 1
    fi
    print_status "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

    # Check CUDA if GPUs requested
    if [[ "$NUM_GPUS" -gt 0 ]]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            print_warning "CUDA not available, will use CPU"
            NUM_GPUS=0
        else
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
            print_status "CUDA: $CUDA_VERSION"
        fi
    fi

    # Check required packages
    for pkg in transformers datasets accelerate; do
        if python -c "import $pkg" 2>/dev/null; then
            print_status "$pkg: installed"
        else
            print_warning "$pkg not installed"
        fi
    done

    # Check SLURM if HPC mode
    if [[ "$EXECUTION_MODE" == "hpc" ]]; then
        if ! command -v sbatch &>/dev/null; then
            print_error "SLURM not available!"
            exit 1
        fi
        print_status "SLURM: available"
    fi

    # Check Git
    if command -v git &>/dev/null; then
        print_status "Git: available"
    else
        print_warning "Git not available"
    fi
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

run_training() {
    print_header "TRAINING"

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

    local output_dir="${1:-$BASE_OUTPUT_DIR/checkpoint}"
    local log_file="$BASE_OUTPUT_DIR/logs/training_${TIMESTAMP}.log"

    mkdir -p "$(dirname "$log_file")"

    if [[ "$DRY_RUN" == "yes" ]]; then
        print_info "DRY RUN: Would execute training"
        return 0
    fi

    local cmd="python latentwire/train.py"
    cmd="$cmd --llama_id '$SOURCE_MODEL'"
    cmd="$cmd --qwen_id '$TARGET_MODEL'"
    cmd="$cmd --dataset $TRAINING_DATASET"
    cmd="$cmd --samples $TRAINING_SAMPLES"
    cmd="$cmd --epochs $TRAINING_EPOCHS"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --latent_len $LATENT_LEN"
    cmd="$cmd --d_z $D_Z"
    cmd="$cmd --output_dir '$output_dir'"
    cmd="$cmd --sequential_models"
    cmd="$cmd --warm_anchor_text 'Answer: '"
    cmd="$cmd --first_token_ce_weight 0.5"

    if [[ "$NUM_GPUS" -gt 1 ]]; then
        cmd="$cmd --distributed"
    fi

    print_info "Starting training..."
    print_debug "Command: $cmd"

    { eval "$cmd"; } 2>&1 | tee "$log_file"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_status "Training completed successfully"
        CHECKPOINT_PATH="$output_dir/epoch$((TRAINING_EPOCHS-1))"
        print_info "Checkpoint: $CHECKPOINT_PATH"
    else
        print_error "Training failed"
        exit 1
    fi
}

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

run_evaluation() {
    print_header "EVALUATION"

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

    if [[ -z "$CHECKPOINT_PATH" ]]; then
        # Find latest checkpoint
        CHECKPOINT_PATH=$(find "$WORK_DIR/runs" -type d -name "epoch*" 2>/dev/null | sort -V | tail -1)
        if [[ -z "$CHECKPOINT_PATH" ]]; then
            print_error "No checkpoint found!"
            exit 1
        fi
    fi

    print_info "Using checkpoint: $CHECKPOINT_PATH"

    local datasets_to_eval="${SPECIFIC_DATASET:-$DATASETS}"

    for dataset in $datasets_to_eval; do
        print_subheader "Evaluating $dataset"

        local eval_script="latentwire/eval.py"
        case $dataset in
            sst2) eval_script="latentwire/eval_sst2.py" ;;
            agnews) eval_script="latentwire/eval_agnews.py" ;;
            trec) eval_script="telepathy/eval_telepathy_trec.py" ;;
        esac

        for seed in $SEEDS; do
            print_info "Seed $seed..."

            local output_file="$BASE_OUTPUT_DIR/results/${dataset}_seed${seed}.json"
            local log_file="$BASE_OUTPUT_DIR/logs/eval_${dataset}_seed${seed}.log"

            mkdir -p "$(dirname "$output_file")" "$(dirname "$log_file")"

            if [[ "$DRY_RUN" == "yes" ]]; then
                print_info "DRY RUN: Would evaluate $dataset with seed $seed"
                continue
            fi

            python "$eval_script" \
                --ckpt "$CHECKPOINT_PATH" \
                --dataset "$dataset" \
                --seed "$seed" \
                --batch_size "$EVAL_BATCH_SIZE" \
                --output_file "$output_file" \
                --sequential_eval \
                --fresh_eval \
                2>&1 | tee "$log_file"
        done
    done

    print_status "Evaluation completed"
}

# =============================================================================
# EXPERIMENT PHASES
# =============================================================================

run_phase1_statistical() {
    print_header "PHASE 1: STATISTICAL RIGOR"

    run_evaluation

    # Statistical analysis
    print_info "Running statistical analysis..."

    python scripts/statistical_testing.py \
        --results_dir "$BASE_OUTPUT_DIR/results" \
        --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
        --output_file "$BASE_OUTPUT_DIR/results/statistical_summary.json"

    print_status "Phase 1 completed"
}

run_phase2_linear_probe() {
    print_header "PHASE 2: LINEAR PROBE BASELINE"

    local datasets_to_probe="${SPECIFIC_DATASET:-$DATASETS}"

    for dataset in $datasets_to_probe; do
        if [[ "$dataset" == "xsum" ]]; then
            print_info "Skipping $dataset (generation task)"
            continue
        fi

        print_info "Running linear probe for $dataset..."

        python latentwire/linear_probe_baseline.py \
            --source_model "$SOURCE_MODEL" \
            --dataset "$dataset" \
            --layer 16 \
            --cv_folds 5 \
            --output_dir "$BASE_OUTPUT_DIR/results/phase2_linear_probe" \
            --batch_size "$EVAL_BATCH_SIZE"
    done

    print_status "Phase 2 completed"
}

run_phase3_baselines() {
    print_header "PHASE 3: FAIR BASELINE COMPARISONS"

    # LLMLingua baseline
    print_info "Running LLMLingua-2 baseline..."

    bash scripts/run_llmlingua_baseline.sh \
        OUTPUT_DIR="$BASE_OUTPUT_DIR/results/phase3_llmlingua" \
        DATASET="${SPECIFIC_DATASET:-squad}" \
        SAMPLES=200

    print_status "Phase 3 completed"
}

run_phase4_efficiency() {
    print_header "PHASE 4: EFFICIENCY MEASUREMENTS"

    local datasets_to_bench="${SPECIFIC_DATASET:-squad}"

    for dataset in $datasets_to_bench; do
        print_info "Measuring efficiency for $dataset..."

        python scripts/benchmark_efficiency.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --dataset "$dataset" \
            --samples 100 \
            --warmup_runs 3 \
            --benchmark_runs 10 \
            --output_file "$BASE_OUTPUT_DIR/results/phase4_efficiency_${dataset}.json"
    done

    print_status "Phase 4 completed"
}

# =============================================================================
# SLURM FUNCTIONS
# =============================================================================

create_slurm_script() {
    local slurm_file="$WORK_DIR/finalization/submit_${EXP_NAME}_${TIMESTAMP}.slurm"

    cat > "$slurm_file" << EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --nodes=1
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --mem=${SLURM_MEMORY}
#SBATCH --output=${LOG_BASE}/${EXP_NAME}_%j.log
#SBATCH --error=${LOG_BASE}/${EXP_NAME}_%j.err

cd $WORK_DIR

echo "=============================================================="
echo "SLURM Job: \$SLURM_JOB_ID on \$SLURMD_NODENAME"
echo "Start: \$(date)"
echo "=============================================================="

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

git pull

# Re-run this script with original arguments in local mode
bash finalization/RUN_ALL.sh $ORIGINAL_ARGS --local --no-interactive

git add -A
git commit -m "results: ${EXP_NAME} (job \$SLURM_JOB_ID)" || true
git push || true

echo "=============================================================="
echo "Completed: \$(date)"
echo "=============================================================="
EOF

    echo "$slurm_file"
}

submit_slurm_job() {
    print_header "SLURM JOB SUBMISSION"

    if [[ "$EXECUTION_MODE" != "hpc" ]]; then
        print_error "SLURM submission requires HPC environment"
        exit 1
    fi

    local slurm_script=$(create_slurm_script)
    print_info "Created script: $slurm_script"

    if [[ "$DRY_RUN" == "yes" ]]; then
        print_info "DRY RUN: Would submit $slurm_script"
        return 0
    fi

    local job_output=$(sbatch "$slurm_script" 2>&1)

    if [[ $? -eq 0 ]]; then
        local job_id=$(echo "$job_output" | grep -oE '[0-9]+' | head -1)
        print_status "Job submitted: $job_id"

        echo ""
        echo "Monitor with:"
        echo "  squeue -j $job_id"
        echo "  tail -f ${LOG_BASE}/${EXP_NAME}_${job_id}.log"

        if [[ "$INTERACTIVE" == "yes" ]]; then
            echo ""
            read -p "Start monitoring? (Y/n): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                sleep 5
                tail -f "${LOG_BASE}/${EXP_NAME}_${job_id}.log" 2>/dev/null
            fi
        fi
    else
        print_error "Submission failed: $job_output"
        exit 1
    fi
}

# =============================================================================
# MONITORING FUNCTIONS
# =============================================================================

monitor_experiments() {
    print_header "EXPERIMENT MONITORING"

    if [[ "$EXECUTION_MODE" != "hpc" ]]; then
        # Local monitoring - just show recent logs
        print_info "Recent experiment logs:"
        ls -lt "$LOG_BASE"/*.log 2>/dev/null | head -10
    else
        # HPC monitoring
        print_subheader "Your Active Jobs"
        squeue -u $USER

        local latest_job=$(squeue -u $USER -h -o "%i" | head -1)
        if [[ -n "$latest_job" ]]; then
            print_info "Latest job: $latest_job"

            local log_file="${LOG_BASE}/*_${latest_job}.log"
            if ls $log_file 1>/dev/null 2>&1; then
                print_subheader "Recent Output"
                tail -20 $log_file

                echo ""
                read -p "Continue monitoring? (Y/n): " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    tail -f $log_file
                fi
            fi
        else
            print_info "No active jobs"

            print_subheader "Recent Jobs"
            sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End --starttime=$(date -d '1 day ago' +%Y-%m-%d) | head -10
        fi
    fi
}

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

run_tests() {
    print_header "RUNNING TEST SUITE"

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

    local test_log="$BASE_OUTPUT_DIR/logs/tests_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$test_log")"

    {
        # Import tests
        print_subheader "Import Tests"
        python -c "
import torch
print(f'PyTorch: {torch.__version__}')
import transformers
print(f'Transformers: {transformers.__version__}')
from latentwire.train import train_latentwire
print('✓ Training imports')
from latentwire.eval import evaluate_checkpoint
print('✓ Evaluation imports')
"

        # Data tests
        print_subheader "Data Loading Tests"
        python -c "
from latentwire.data import get_dataset
for dataset in ['squad', 'sst2', 'agnews']:
    data = get_dataset(dataset, split='train', max_samples=10)
    print(f'✓ {dataset}: {len(data)} samples')
"

        # Model tests
        print_subheader "Model Tests"
        python -c "
from latentwire.models import Encoder, Adapter
import torch

encoder = Encoder(d_z=256, vocab_size=256)
adapter = Adapter(d_z=256, d_model=4096)
dummy = torch.randint(0, 256, (2, 64))
latent = encoder(dummy)
adapted = adapter(latent)
print(f'✓ Encoder: {latent.shape}')
print(f'✓ Adapter: {adapted.shape}')
"

        # Quick training test
        if [[ "$SKIP_TRAINING" != "yes" ]]; then
            print_subheader "Quick Training Test"
            python latentwire/train.py \
                --llama_id "$SOURCE_MODEL" \
                --qwen_id "$TARGET_MODEL" \
                --dataset squad \
                --samples 10 \
                --epochs 1 \
                --batch_size 2 \
                --latent_len 8 \
                --d_z 64 \
                --output_dir "$BASE_OUTPUT_DIR/test_checkpoint" \
                --sequential_models
        fi

        print_status "All tests passed!"

    } 2>&1 | tee "$test_log"

    print_info "Test log: $test_log"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

compile_paper() {
    print_header "COMPILING PAPER"

    local paper_dir="$WORK_DIR/paper"
    [[ ! -d "$paper_dir" ]] && paper_dir="$WORK_DIR/docs/paper"

    if [[ ! -d "$paper_dir" ]]; then
        print_error "Paper directory not found"
        exit 1
    fi

    cd "$paper_dir"

    local main_tex=""
    for tex in main.tex paper.tex latentwire.tex; do
        [[ -f "$tex" ]] && main_tex="$tex" && break
    done

    if [[ -z "$main_tex" ]]; then
        print_error "No main LaTeX file found"
        exit 1
    fi

    print_info "Compiling $main_tex..."

    pdflatex -interaction=nonstopmode "$main_tex"
    bibtex "${main_tex%.tex}" 2>/dev/null || true
    pdflatex -interaction=nonstopmode "$main_tex"
    pdflatex -interaction=nonstopmode "$main_tex"

    local pdf="${main_tex%.tex}.pdf"
    if [[ -f "$pdf" ]]; then
        print_status "Paper compiled: $pdf"
        cp "$pdf" "$WORK_DIR/runs/latentwire_paper_${TIMESTAMP}.pdf"
    else
        print_error "Compilation failed"
        exit 1
    fi
}

run_quick_start() {
    print_header "QUICK START"

    # Override with minimal settings
    TRAINING_SAMPLES=100
    TRAINING_EPOCHS=2
    SEEDS="42"
    DATASETS="squad"
    BATCH_SIZE=4

    print_info "Running with reduced settings for quick testing"

    # Run training
    if [[ "$SKIP_TRAINING" != "yes" ]]; then
        run_training
    fi

    # Run evaluation
    run_evaluation

    print_status "Quick start completed!"
}

run_ddp_training() {
    print_header "DDP TRAINING"

    if [[ "$NUM_GPUS" -lt 2 ]]; then
        print_error "DDP requires at least 2 GPUs"
        exit 1
    fi

    print_info "Launching DDP training with $NUM_GPUS GPUs..."

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        latentwire/train.py \
        --llama_id "$SOURCE_MODEL" \
        --qwen_id "$TARGET_MODEL" \
        --dataset "$TRAINING_DATASET" \
        --samples "$TRAINING_SAMPLES" \
        --epochs "$TRAINING_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --output_dir "$BASE_OUTPUT_DIR/ddp_checkpoint" \
        --distributed
}

run_benchmarks() {
    print_header "EFFICIENCY BENCHMARKS"

    if [[ -z "$CHECKPOINT_PATH" ]]; then
        print_error "Checkpoint required for benchmarking"
        exit 1
    fi

    print_info "Running benchmarks..."

    python scripts/benchmark_efficiency.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --dataset squad \
        --samples 100 \
        --warmup_runs 3 \
        --benchmark_runs 10 \
        --measure_memory \
        --measure_latency \
        --measure_throughput \
        --output_file "$BASE_OUTPUT_DIR/benchmark_results.json"

    print_status "Benchmarks completed"
}

clean_workspace() {
    print_header "CLEANING WORKSPACE"

    print_warning "This will remove temporary files and old logs"

    if [[ "$INTERACTIVE" == "yes" ]]; then
        read -p "Continue? (y/N): " -n 1 -r
        echo ""
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 0
    fi

    # Clean old logs (older than 7 days)
    find "$LOG_BASE" -name "*.log" -mtime +7 -delete 2>/dev/null

    # Clean Python cache
    find "$WORK_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find "$WORK_DIR" -name "*.pyc" -delete 2>/dev/null

    # Clean temporary files
    rm -f "$WORK_DIR"/*.tmp 2>/dev/null

    print_status "Workspace cleaned"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Show header
    clear
    print_header "LATENTWIRE UNIFIED EXECUTION v${SCRIPT_VERSION}"

    # Detect environment
    detect_environment

    # Show configuration
    print_info "Command: $COMMAND"
    print_info "Mode: $EXECUTION_MODE"
    print_info "GPUs: $NUM_GPUS"
    [[ -n "$SPECIFIC_PHASE" ]] && print_info "Phase: $SPECIFIC_PHASE"
    [[ -n "$SPECIFIC_DATASET" ]] && print_info "Dataset: $SPECIFIC_DATASET"
    [[ "$SKIP_TRAINING" == "yes" ]] && print_info "Skip training: Yes"
    [[ "$DEBUG_MODE" == "yes" ]] && print_info "Debug: Enabled"
    [[ "$DRY_RUN" == "yes" ]] && print_info "Dry run: Yes"

    # Execute command
    case "$COMMAND" in
        help)
            show_help
            ;;
        validate)
            validate_environment
            ;;
        test)
            validate_environment
            run_tests
            ;;
        train)
            validate_environment
            run_training
            ;;
        eval)
            validate_environment
            run_evaluation
            ;;
        experiment)
            validate_environment

            # Handle phases
            if [[ -n "$SPECIFIC_PHASE" ]]; then
                case "$SPECIFIC_PHASE" in
                    1) run_phase1_statistical ;;
                    2) run_phase2_linear_probe ;;
                    3) run_phase3_baselines ;;
                    4) run_phase4_efficiency ;;
                    *) print_error "Invalid phase: $SPECIFIC_PHASE"; exit 1 ;;
                esac
            else
                # Run all phases
                if [[ "$SKIP_TRAINING" != "yes" ]]; then
                    run_training
                fi
                run_phase1_statistical
                run_phase2_linear_probe
                run_phase3_baselines
                run_phase4_efficiency
            fi
            ;;
        quick)
            validate_environment
            run_quick_start
            ;;
        finalize)
            validate_environment

            # Full finalization pipeline
            print_header "PAPER FINALIZATION"

            if [[ "$SKIP_TRAINING" != "yes" ]]; then
                run_training
            fi

            run_phase1_statistical
            run_phase2_linear_probe
            run_phase3_baselines
            run_phase4_efficiency

            # Aggregate results
            print_info "Aggregating results..."
            python finalization/aggregate_results.py \
                --results_dir "$BASE_OUTPUT_DIR/results" \
                --output_file "$BASE_OUTPUT_DIR/final_results.json" \
                --generate_latex_tables \
                --generate_plots

            print_status "Finalization completed!"
            ;;
        monitor)
            monitor_experiments
            ;;
        slurm)
            if [[ "$EXECUTION_MODE" != "hpc" ]]; then
                print_error "SLURM submission requires HPC environment"
                exit 1
            fi
            submit_slurm_job
            ;;
        compile)
            compile_paper
            ;;
        ddp)
            validate_environment
            run_ddp_training
            ;;
        benchmark)
            validate_environment
            run_benchmarks
            ;;
        clean)
            clean_workspace
            ;;
        *)
            print_error "Invalid command: $COMMAND"
            show_help
            exit 1
            ;;
    esac

    # Show completion message
    echo ""
    print_status "Command '$COMMAND' completed successfully!"
    [[ -d "$BASE_OUTPUT_DIR" ]] && print_info "Results: $BASE_OUTPUT_DIR"
}

# Handle interrupts gracefully
trap 'echo ""; print_warning "Interrupted by user"; exit 130' INT TERM

# Run main function
main