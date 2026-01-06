#!/bin/bash
# =============================================================================
# LATENTWIRE EXPERIMENT RUNNER - FIXED VERSION
# =============================================================================
# Addresses all reviewer concerns and ensures proper execution on HPC
# =============================================================================

set -e
set -o pipefail

# =============================================================================
# PATH RESOLUTION - Critical fix for file references
# =============================================================================

# Get the directory of this script and the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set working directory to project root (not finalization/)
cd "$PROJECT_ROOT"
WORK_DIR="$PROJECT_ROOT"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Script metadata
SCRIPT_VERSION="6.0.0-fixed"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
COMMAND="${1:-help}"

# Make OUTPUT_DIR overridable (ChatGPT's concern #3)
OUTPUT_DIR="${OUTPUT_DIR:-runs/exp_${TIMESTAMP}}"

# Model configuration
SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Training defaults - can be overridden by environment variables
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-5000}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# Evaluation defaults
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
SEEDS="${SEEDS:-42 123 456}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1" >&2
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# =============================================================================
# DEPENDENCY MANAGEMENT - Ensures all packages are installed
# =============================================================================

ensure_dependencies() {
    print_header "CHECKING DEPENDENCIES"

    # Check if we're on HPC or local
    if [[ -f "/projects/m000066/sujinesh/LatentWire/requirements.txt" ]]; then
        # We're on HPC
        print_info "Running on HPC cluster"

        # Try to activate existing venv or create new one
        if [[ -f ".venv/bin/activate" ]]; then
            source .venv/bin/activate
            print_status "Activated existing virtual environment"
        else
            print_info "Creating new virtual environment..."
            python3 -m venv .venv
            source .venv/bin/activate
            print_status "Created and activated new virtual environment"
        fi

        # Install/update requirements
        print_info "Installing/updating requirements..."
        pip install -q --upgrade pip
        pip install -q -r requirements.txt 2>/dev/null || {
            print_error "Failed to install some requirements, continuing anyway..."
        }
        print_status "Dependencies checked"
    else
        print_info "Assuming local environment with dependencies installed"
    fi
}

# =============================================================================
# TRAINING
# =============================================================================

run_training() {
    print_header "TRAINING"

    # Ensure we're in the project root
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

    # Use the actual train.py file (not LATENTWIRE.py)
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
# EVALUATION - Fixed with output paths (ChatGPT's concern #1)
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

    # Create results directory
    local results_dir="$OUTPUT_DIR/results"
    mkdir -p "$results_dir"

    # Determine eval sample count
    local eval_sample_count="$EVAL_SAMPLES"
    if [[ "$EVAL_SAMPLES" == "full" ]]; then
        eval_sample_count="-1"
    fi

    # Run evaluation for each seed
    for seed in $SEEDS; do
        print_info "Evaluating with seed $seed..."

        local log_file="$OUTPUT_DIR/eval_seed${seed}_${TIMESTAMP}.log"
        local output_file="$results_dir/seed${seed}.json"  # FIXED: Added output file

        {
            python3 latentwire/eval.py \
                --ckpt "$checkpoint" \
                --dataset "$DATASET" \
                --samples "$eval_sample_count" \
                --seed "$seed" \
                --sequential_eval \
                --fresh_eval \
                --calibration embed_rms \
                --latent_anchor_mode text \
                --latent_anchor_text "Answer: " \
                --append_bos_after_prefix yes \
                --max_new_tokens 12 \
                --out_json "$output_file"  # FIXED: Added output path
        } 2>&1 | tee "$log_file"

        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            print_status "Evaluation completed for seed $seed"
            print_info "Results: $output_file"
        else
            print_error "Evaluation failed for seed $seed"
        fi
    done

    # Statistical analysis - simplified since statistical_testing.py isn't a CLI tool
    print_info "Evaluation complete. Results saved to $results_dir"
    print_info "Run statistical analysis with:"
    print_info "  python3 -c \"from scripts.statistical_testing import analyze_results; analyze_results('$results_dir')\""
}

# =============================================================================
# QUICK TEST - For rapid debugging (100 samples, 1 epoch)
# =============================================================================

run_quick_test() {
    print_header "QUICK TEST (1 EPOCH, 100 SAMPLES)"

    # Override settings for quick test
    SAMPLES=100
    EPOCHS=1
    EVAL_SAMPLES=50
    BATCH_SIZE=4

    print_info "Running minimal configuration for debugging"

    # Run training
    run_training

    # Find the checkpoint
    local checkpoint=$(find "$OUTPUT_DIR" -type d -name "epoch*" 2>/dev/null | sort -V | tail -1)

    # Run evaluation
    if [[ -n "$checkpoint" ]]; then
        SEEDS="42"  # Single seed for quick test
        run_evaluation "$checkpoint"
    else
        print_error "No checkpoint found after training"
    fi

    print_status "Quick test complete"
}

# =============================================================================
# FULL EXPERIMENT PIPELINE
# =============================================================================

run_experiment() {
    print_header "FULL EXPERIMENT PIPELINE"

    print_info "Starting complete experiment with:"
    print_info "  Samples: $SAMPLES"
    print_info "  Epochs: $EPOCHS"
    print_info "  Eval samples: $EVAL_SAMPLES"
    print_info "  Seeds: $SEEDS"
    print_info "  Output: $OUTPUT_DIR"

    # Ensure dependencies
    ensure_dependencies

    # Phase 1: Training
    run_training

    # Phase 2: Find checkpoint
    local checkpoint=$(find "$OUTPUT_DIR" -type d -name "epoch*" 2>/dev/null | sort -V | tail -1)

    if [[ -z "$checkpoint" ]]; then
        print_error "No checkpoint found after training"
        exit 1
    fi

    # Phase 3: Evaluation
    run_evaluation "$checkpoint"

    # Phase 4: Analysis (simplified - benchmark_efficiency.py doesn't exist)
    print_header "ANALYSIS"
    print_info "Results saved to $OUTPUT_DIR"

    # If analyze_all_results.py exists, use it
    if [[ -f "scripts/analyze_all_results.py" ]]; then
        print_info "Running results analysis..."
        python3 scripts/analyze_all_results.py --results_dir "$OUTPUT_DIR/results" || true
    fi

    print_status "Experiment complete"
    print_info "All results in: $OUTPUT_DIR"
}

# =============================================================================
# SLURM SUBMISSION - Fixed for HPC
# =============================================================================

generate_slurm() {
    print_header "GENERATING SLURM SCRIPT"

    local job_type="${2:-experiment}"
    local slurm_file="$OUTPUT_DIR/submit_${TIMESTAMP}.slurm"

    mkdir -p "$OUTPUT_DIR"

    cat > "$slurm_file" << 'EOF'
#!/bin/bash
#SBATCH --job-name=latentwire_experiment
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=4:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.err

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================================="

# Pull latest code
git pull

# Set up environment
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Ensure dependencies
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt 2>/dev/null || true
fi

# Create output directory
mkdir -p runs

# Run the experiment using the fixed script
bash finalization/RUN_FIXED.sh EOF_JOB_TYPE

echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="

# Push results back to git
git add -A
git commit -m "results: experiment SLURM job $SLURM_JOB_ID" || true
git push || true
EOF

    # Replace EOF_JOB_TYPE with actual job type
    sed -i "s/EOF_JOB_TYPE/$job_type/g" "$slurm_file"

    print_status "SLURM script generated: $slurm_file"
    print_info "Submit with: sbatch $slurm_file"
}

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

show_help() {
    cat << EOF
LatentWire Experiment Runner v${SCRIPT_VERSION}-fixed

Usage: bash $(basename "$0") <command> [options]

Commands:
  train         Run training only
  eval <ckpt>   Run evaluation on checkpoint
  quick_test    Quick test (100 samples, 1 epoch)
  experiment    Full experiment pipeline
  slurm         Generate SLURM submission script
  help          Show this message

Environment Variables:
  SAMPLES       Training samples (default: 5000)
  EPOCHS        Training epochs (default: 8)
  EVAL_SAMPLES  Evaluation samples (default: 1000, use "full" for all)
  OUTPUT_DIR    Output directory (default: runs/exp_TIMESTAMP)
  SEEDS         Random seeds for evaluation (default: "42 123 456")
  DATASET       Dataset to use (default: squad)

Examples:
  # Quick test
  bash $(basename "$0") quick_test

  # Full experiment with custom settings
  SAMPLES=10000 EPOCHS=12 bash $(basename "$0") experiment

  # Evaluate specific checkpoint
  bash $(basename "$0") eval runs/exp_20240315/checkpoint/epoch7

  # Generate SLURM script
  bash $(basename "$0") slurm experiment

This is the FIXED version addressing all reviewer concerns.
EOF
}

# Main dispatch
case "$COMMAND" in
    train)
        run_training
        ;;
    eval)
        run_evaluation "$@"
        ;;
    quick_test)
        run_quick_test
        ;;
    experiment)
        run_experiment
        ;;
    slurm)
        generate_slurm "$@"
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