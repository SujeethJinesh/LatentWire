#!/bin/bash
# =============================================================================
# LATENTWIRE EXPERIMENT RUNNER - FIXED FOR CONSOLIDATED LATENTWIRE.py
# =============================================================================
# This version uses the consolidated LATENTWIRE.py directly instead of
# separate latentwire/*.py files
# =============================================================================

set -e
set -o pipefail

SCRIPT_VERSION="5.1.0-FIXED"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
COMMAND="${1:-help}"
OUTPUT_DIR="runs/exp_${TIMESTAMP}"

# Model configuration
SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Training defaults
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-5000}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# Evaluation defaults
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
SEEDS="${SEEDS:-42 123 456}"

# SLURM configuration
SLURM_ACCOUNT="${SLURM_ACCOUNT:-marlowe-m000066}"
SLURM_PARTITION="${SLURM_PARTITION:-preempt}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_GPUS="${SLURM_GPUS:-4}"

# Detect environment
if [[ -d "/projects/m000066" ]]; then
    WORK_DIR="/projects/m000066/sujinesh/LatentWire"
    IS_HPC="yes"
else
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    IS_HPC="no"
fi

# Find LATENTWIRE.py
LATENTWIRE="$WORK_DIR/LATENTWIRE.py"
if [[ ! -f "$LATENTWIRE" ]]; then
    LATENTWIRE="$WORK_DIR/finalization/LATENTWIRE.py"
fi

if [[ ! -f "$LATENTWIRE" ]]; then
    echo -e "${RED}[✗]${NC} LATENTWIRE.py not found!"
    exit 1
fi

# Utility functions
print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

print_header() {
    echo ""
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
    echo ""
}

# =============================================================================
# TRAINING - Uses consolidated LATENTWIRE.py
# =============================================================================

run_training() {
    print_header "TRAINING"
    
    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
    
    local checkpoint_dir="$OUTPUT_DIR/checkpoint"
    local log_file="$OUTPUT_DIR/training_${TIMESTAMP}.log"
    
    mkdir -p "$OUTPUT_DIR"
    
    print_info "Configuration:"
    print_info "  Dataset: $DATASET"
    print_info "  Samples: $SAMPLES"
    print_info "  Epochs: $EPOCHS"
    print_info "  Latent: ${LATENT_LEN}x${D_Z}"
    print_info "  Output: $checkpoint_dir"
    
    {
        python3 "$LATENTWIRE" train \
            --experiment_name "train_${TIMESTAMP}" \
            --dataset "$DATASET" \
            --samples "$SAMPLES" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --latent_len "$LATENT_LEN" \
            --d_z "$D_Z"
    } 2>&1 | tee "$log_file"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_status "Training completed"
        print_info "Log: $log_file"
    else
        print_error "Training failed"
        exit 1
    fi
}

# =============================================================================
# EVALUATION - Uses consolidated LATENTWIRE.py
# =============================================================================

run_evaluation() {
    print_header "EVALUATION"
    
    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
    
    local checkpoint="${2:-}"
    if [[ -z "$checkpoint" ]]; then
        checkpoint=$(find "$OUTPUT_DIR" -type d -name "checkpoint*" 2>/dev/null | sort -V | tail -1)
        if [[ -z "$checkpoint" ]]; then
            print_error "No checkpoint found. Specify with: eval <checkpoint_path>"
            exit 1
        fi
    fi
    
    print_info "Checkpoint: $checkpoint"
    print_info "Seeds: $SEEDS"
    
    local results_dir="$OUTPUT_DIR/results"
    mkdir -p "$results_dir"
    
    for seed in $SEEDS; do
        print_info "Evaluating with seed $seed..."
        local log_file="$OUTPUT_DIR/eval_seed${seed}_${TIMESTAMP}.log"
        
        {
            python3 "$LATENTWIRE" eval \
                --checkpoint "$checkpoint" \
                --dataset "$DATASET" \
                --samples "$EVAL_SAMPLES" \
                --output "$results_dir/eval_seed${seed}.json"
        } 2>&1 | tee "$log_file"
    done
    
    print_status "Evaluation completed"
    print_info "Results: $results_dir"
}

# =============================================================================
# QUICK TEST
# =============================================================================

run_quick_test() {
    print_header "QUICK TEST"
    
    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
    
    print_info "Testing Python syntax..."
    python3 -m py_compile "$LATENTWIRE"
    print_status "Syntax OK"
    
    print_info "Testing imports..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from LATENTWIRE import ExperimentConfig, ByteEncoder, SharedAdapter, Trainer
print('✓ All imports successful')
"
    
    print_info "Testing configuration..."
    python3 "$LATENTWIRE" test --verbose
    
    print_status "Quick test passed"
}

# =============================================================================
# EXPERIMENT - Full pipeline
# =============================================================================

run_experiment() {
    print_header "FULL EXPERIMENT PIPELINE"
    
    print_info "Phase 1: Training"
    run_training
    
    print_info "Phase 2: Evaluation"
    run_evaluation
    
    print_status "Experiment completed"
    print_info "All results in: $OUTPUT_DIR"
}

# =============================================================================
# SLURM SUBMISSION
# =============================================================================

submit_slurm() {
    print_header "SLURM SUBMISSION"
    
    if [[ "$IS_HPC" != "yes" ]]; then
        print_error "SLURM submission requires HPC environment"
        exit 1
    fi
    
    local cmd="${2:-experiment}"
    local slurm_file="$OUTPUT_DIR/submit_${TIMESTAMP}.slurm"
    
    mkdir -p "$OUTPUT_DIR"
    
    cat > "$slurm_file" << EOF
#!/bin/bash
#SBATCH --job-name=latentwire_${cmd}
#SBATCH --nodes=1
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --mem=256GB
#SBATCH --output=$WORK_DIR/runs/slurm_%j.log
#SBATCH --error=$WORK_DIR/runs/slurm_%j.err

cd $WORK_DIR
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

echo "Job ID: \$SLURM_JOB_ID"
echo "Start: \$(date)"

bash RUN.sh $cmd

echo "Completed: \$(date)"
EOF
    
    print_info "Submitting job: $cmd"
    sbatch "$slurm_file"
    
    print_status "Job submitted"
    echo "Monitor with: squeue -u \$USER"
}

# =============================================================================
# HELP
# =============================================================================

show_help() {
    cat << EOF
LatentWire Experiment Runner v${SCRIPT_VERSION}

USAGE:
    bash RUN.sh [COMMAND]

COMMANDS:
    train       Run training only
    eval        Run evaluation only (requires checkpoint)
    quick_test  Quick sanity check
    experiment  Run full pipeline (train + eval)
    slurm       Submit job to SLURM cluster
    help        Show this help

ENVIRONMENT VARIABLES:
    DATASET      Dataset name (default: squad)
    SAMPLES      Training samples (default: 5000)
    EPOCHS       Training epochs (default: 6)
    BATCH_SIZE   Batch size (default: 8)
    EVAL_SAMPLES Evaluation samples (default: 500)
    SEEDS        Evaluation seeds (default: "42 123 456")

EXAMPLES:
    # Quick test
    bash RUN.sh quick_test
    
    # Training with custom config
    SAMPLES=10000 EPOCHS=8 bash RUN.sh train
    
    # Full experiment
    bash RUN.sh experiment

CONFIGURATION:
    Using: $LATENTWIRE
    Mode: $([ "$IS_HPC" == "yes" ] && echo "HPC" || echo "Local")
    
EOF
}

# =============================================================================
# MAIN
# =============================================================================

case "$COMMAND" in
    train)
        run_training
        ;;
    eval)
        run_evaluation "$@"
        ;;
    quick_test|test)
        run_quick_test
        ;;
    experiment)
        run_experiment
        ;;
    slurm)
        submit_slurm "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
