#!/bin/bash
# =============================================================================
# LATENTWIRE EXPERIMENT RUNNER - SIMPLIFIED FOR PAPER REVIEW
# =============================================================================
# Essential commands only: train, eval, test, experiment, slurm
# Under 500 lines, focused on what's needed for paper review
# =============================================================================

set -e
set -o pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Script metadata
SCRIPT_VERSION="4.0.0"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
COMMAND="${1:-help}"
OUTPUT_DIR="runs/exp_${TIMESTAMP}"

# Model configuration
SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Training defaults
DATASET="squad"
SAMPLES=87599
EPOCHS=8
BATCH_SIZE=8
LATENT_LEN=32
D_Z=256

# Evaluation defaults
EVAL_SAMPLES=200
SEEDS="42 123 456"

# SLURM configuration
SLURM_ACCOUNT="marlowe-m000066"
SLURM_PARTITION="preempt"
SLURM_TIME="12:00:00"
SLURM_GPUS=4

# Environment detection
if [[ -d "/projects/m000066" ]]; then
    WORK_DIR="/projects/m000066/sujinesh/LatentWire"
    IS_HPC="yes"
else
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    IS_HPC="no"
fi

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
# TRAINING
# =============================================================================

run_training() {
    print_header "TRAINING"

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
    export PYTORCH_ENABLE_MPS_FALLBACK=1

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
        python3 latentwire/train.py \
            --llama_id "$SOURCE_MODEL" \
            --qwen_id "$TARGET_MODEL" \
            --dataset "$DATASET" \
            --samples "$SAMPLES" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --latent_len "$LATENT_LEN" \
            --d_z "$D_Z" \
            --output_dir "$checkpoint_dir" \
            --sequential_models \
            --warm_anchor_text "Answer: " \
            --first_token_ce_weight 0.5
    } 2>&1 | tee "$log_file"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_status "Training completed"
        print_info "Checkpoint: $checkpoint_dir/epoch$((EPOCHS-1))"
        print_info "Log: $log_file"
    else
        print_error "Training failed"
        exit 1
    fi
}

# =============================================================================
# EVALUATION
# =============================================================================

run_evaluation() {
    print_header "EVALUATION"

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
    export PYTORCH_ENABLE_MPS_FALLBACK=1

    # Find checkpoint
    local checkpoint="${2:-}"
    if [[ -z "$checkpoint" ]]; then
        checkpoint=$(find "$OUTPUT_DIR" -type d -name "epoch*" 2>/dev/null | sort -V | tail -1)
        if [[ -z "$checkpoint" ]]; then
            print_error "No checkpoint found. Specify with: eval <checkpoint_path>"
            exit 1
        fi
    fi

    print_info "Checkpoint: $checkpoint"
    print_info "Seeds: $SEEDS"

    local results_dir="$OUTPUT_DIR/results"
    mkdir -p "$results_dir"

    # Run evaluation for each seed
    for seed in $SEEDS; do
        print_info "Evaluating with seed $seed..."

        local log_file="$OUTPUT_DIR/eval_seed${seed}_${TIMESTAMP}.log"

        {
            python3 latentwire/eval.py \
                --ckpt "$checkpoint" \
                --dataset "$DATASET" \
                --samples "$EVAL_SAMPLES" \
                --seed "$seed" \
                --sequential_eval \
                --fresh_eval \
                --calibration embed_rms \
                --latent_anchor_mode text \
                --latent_anchor_text "Answer: " \
                --append_bos_after_prefix yes \
                --max_new_tokens 12
        } 2>&1 | tee "$log_file"
    done

    # Run statistical analysis
    print_info "Running statistical analysis..."

    python3 scripts/statistical_testing.py \
        --results_dir "$results_dir" \
        --bootstrap_samples 10000 \
        --output_file "$results_dir/statistical_summary.json"

    print_status "Evaluation completed"
    print_info "Results: $results_dir"
}

# =============================================================================
# TESTING
# =============================================================================

run_tests() {
    print_header "TESTING"

    cd "$WORK_DIR"
    export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
    export PYTORCH_ENABLE_MPS_FALLBACK=1

    local test_dir="$OUTPUT_DIR/tests"
    mkdir -p "$test_dir"

    print_info "Running test suite..."

    # Import tests
    print_info "Testing imports..."
    python3 -c "
import torch
import transformers
from latentwire.train import main
from latentwire.eval import evaluate_checkpoint
print('✓ All imports successful')
" || { print_error "Import test failed"; exit 1; }

    # Data loading test
    print_info "Testing data loading..."
    python3 -c "
from latentwire.data import get_dataset
data = get_dataset('squad', split='train', max_samples=10)
print(f'✓ Loaded {len(data)} samples')
" || { print_error "Data test failed"; exit 1; }

    # Model test
    print_info "Testing models..."
    python3 -c "
from latentwire.models import Encoder, Adapter
import torch
encoder = Encoder(d_z=256, vocab_size=256)
adapter = Adapter(d_z=256, d_model=4096)
dummy = torch.randint(0, 256, (2, 64))
latent = encoder(dummy)
adapted = adapter(latent)
print(f'✓ Encoder output: {latent.shape}')
print(f'✓ Adapter output: {adapted.shape}')
" || { print_error "Model test failed"; exit 1; }

    # Quick training test
    print_info "Running quick training test..."
    python3 latentwire/train.py \
        --llama_id "$SOURCE_MODEL" \
        --qwen_id "$TARGET_MODEL" \
        --dataset squad \
        --samples 10 \
        --epochs 1 \
        --batch_size 2 \
        --latent_len 8 \
        --d_z 64 \
        --output_dir "$test_dir/quick_checkpoint" \
        --sequential_models \
        > "$test_dir/quick_train.log" 2>&1

    if [[ $? -eq 0 ]]; then
        print_status "All tests passed"
    else
        print_error "Quick training test failed"
        cat "$test_dir/quick_train.log"
        exit 1
    fi
}

# =============================================================================
# FULL EXPERIMENT PIPELINE
# =============================================================================

run_experiment() {
    print_header "FULL EXPERIMENT PIPELINE"

    # Phase 1: Training
    print_info "Phase 1: Training"
    run_training

    # Phase 2: Multi-seed evaluation
    print_info "Phase 2: Evaluation with statistical rigor"
    checkpoint="$OUTPUT_DIR/checkpoint/epoch$((EPOCHS-1))"
    run_evaluation "" "$checkpoint"

    # Phase 3: Linear probe baseline
    print_info "Phase 3: Linear probe baseline"
    python3 latentwire/linear_probe_baseline.py \
        --source_model "$SOURCE_MODEL" \
        --dataset "$DATASET" \
        --layer 16 \
        --cv_folds 5 \
        --output_dir "$OUTPUT_DIR/linear_probe" \
        --batch_size 16

    # Phase 4: Efficiency benchmarks
    print_info "Phase 4: Efficiency measurements"
    python3 scripts/benchmark_efficiency.py \
        --checkpoint "$checkpoint" \
        --dataset "$DATASET" \
        --samples 100 \
        --warmup_runs 3 \
        --benchmark_runs 10 \
        --output_file "$OUTPUT_DIR/efficiency_results.json"

    # Aggregate all results
    print_info "Aggregating results..."
    python3 scripts/analyze_all_results.py \
        --results_dir "$OUTPUT_DIR" \
        --output_file "$OUTPUT_DIR/final_summary.json"

    print_status "Experiment pipeline completed"
    print_info "All results in: $OUTPUT_DIR"
}

# =============================================================================
# SLURM SUBMISSION
# =============================================================================

create_slurm_script() {
    local cmd="$1"
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

echo "=============================================================="
echo "Job ID: \$SLURM_JOB_ID on \$SLURMD_NODENAME"
echo "Start: \$(date)"
echo "=============================================================="

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

# Pull latest code
git pull

# Run the command
bash finalization/RUN.sh $cmd

# Push results
git add -A
git commit -m "results: $cmd (job \$SLURM_JOB_ID)" || true
git push || true

echo "=============================================================="
echo "Completed: \$(date)"
echo "=============================================================="
EOF

    echo "$slurm_file"
}

submit_slurm() {
    print_header "SLURM SUBMISSION"

    if [[ "$IS_HPC" != "yes" ]]; then
        print_error "SLURM submission requires HPC environment"
        exit 1
    fi

    local cmd="${2:-experiment}"
    local slurm_script=$(create_slurm_script "$cmd")

    print_info "Submitting job for command: $cmd"
    print_info "Script: $slurm_script"

    sbatch "$slurm_script"

    print_status "Job submitted"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f $WORK_DIR/runs/slurm_*.log"
}

# =============================================================================
# HELP
# =============================================================================

show_help() {
    cat << EOF
LatentWire Experiment Runner v${SCRIPT_VERSION}

USAGE:
    bash RUN.sh [COMMAND] [OPTIONS]

COMMANDS:
    train       Run training only
    eval        Run evaluation only (requires checkpoint)
    test        Run test suite
    experiment  Run full experiment pipeline (train + eval + analysis)
    slurm       Submit job to SLURM cluster
    help        Show this help

OPTIONS:
    After command, you can specify:
    - For eval: checkpoint path
    - For slurm: command to run (e.g., "slurm train")

EXAMPLES:
    # Run training
    bash RUN.sh train

    # Run evaluation on existing checkpoint
    bash RUN.sh eval runs/exp_20240115/checkpoint/epoch23

    # Run full experiment pipeline
    bash RUN.sh experiment

    # Submit experiment to SLURM
    bash RUN.sh slurm experiment

    # Run tests
    bash RUN.sh test

ENVIRONMENT:
    Mode: $([ "$IS_HPC" == "yes" ] && echo "HPC" || echo "Local")
    Work dir: $WORK_DIR
    Output dir: $OUTPUT_DIR

CONFIGURATION:
    Models: Llama-3.1-8B + Qwen2.5-7B
    Dataset: $DATASET
    Training: ${SAMPLES} samples, ${EPOCHS} epochs
    Latent: ${LATENT_LEN} tokens x ${D_Z} dimensions
    Evaluation: ${EVAL_SAMPLES} samples, seeds: $SEEDS
EOF
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    case "$COMMAND" in
        train)
            run_training
            ;;
        eval)
            run_evaluation "$@"
            ;;
        test)
            run_tests
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
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"