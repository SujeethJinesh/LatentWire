#!/bin/bash
# =============================================================================
# LATENTWIRE EXPERIMENT RUNNER - CONSOLIDATED VERSION
# =============================================================================
# Includes all fixes and comprehensive logging
# =============================================================================

set -e
set -o pipefail

# =============================================================================
# PATH RESOLUTION - Fix for file references
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set working directory to project root (not finalization/)
cd "$PROJECT_ROOT"
WORK_DIR="$PROJECT_ROOT"

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION="8.0.0"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Command and output directory
COMMAND="${1:-help}"

# Make OUTPUT_DIR overridable via environment
OUTPUT_DIR="${OUTPUT_DIR:-runs/exp_${TIMESTAMP}}"

# Create output directory immediately for logs
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# MASTER LOG FILE - Captures everything
# =============================================================================

MASTER_LOG="$OUTPUT_DIR/master_${TIMESTAMP}.log"
echo "Starting LatentWire Experiment - $(date)" | tee "$MASTER_LOG"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$MASTER_LOG"
echo "Command: $COMMAND $*" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

# Redirect all output to both terminal and master log
exec > >(tee -a "$MASTER_LOG")
exec 2>&1

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

SOURCE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Training defaults - can be overridden by environment variables
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-5000}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Reduced for 40GB memory
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
# LOG SUMMARY FUNCTION
# =============================================================================

create_log_summary() {
    local summary_file="$OUTPUT_DIR/LOG_INDEX.md"

    cat > "$summary_file" << EOF
# LatentWire Experiment Logs
**Generated:** $(date)
**Experiment ID:** ${TIMESTAMP}
**Command:** $COMMAND $*

## Log Files Generated

### Master Log
- \`master_${TIMESTAMP}.log\` - Complete experiment output

### Phase-Specific Logs
EOF

    # List all log files
    for logfile in "$OUTPUT_DIR"/*.log; do
        if [[ -f "$logfile" ]]; then
            basename=$(basename "$logfile")
            size=$(du -h "$logfile" | cut -f1)
            lines=$(wc -l < "$logfile")
            echo "- \`$basename\` - $size, $lines lines" >> "$summary_file"
        fi
    done

    echo "" >> "$summary_file"
    echo "## How to View Logs" >> "$summary_file"
    echo "\`\`\`bash" >> "$summary_file"
    echo "# View complete experiment" >> "$summary_file"
    echo "cat $OUTPUT_DIR/master_${TIMESTAMP}.log" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "# Check for errors" >> "$summary_file"
    echo "grep -i error $OUTPUT_DIR/*.log" >> "$summary_file"
    echo "\`\`\`" >> "$summary_file"

    print_status "Log summary created: $summary_file"
}

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

ensure_dependencies() {
    print_header "CHECKING DEPENDENCIES"

    local dep_log="$OUTPUT_DIR/dependencies_${TIMESTAMP}.log"

    {
        echo "Dependency check started: $(date)"

        # Check Python version
        python3 --version

        # Check if we're on HPC or local
        if [[ -f "/projects/m000066/sujinesh/LatentWire/requirements.txt" ]]; then
            echo "Environment: HPC cluster"

            # Try to activate existing venv or create new one
            if [[ -f ".venv/bin/activate" ]]; then
                source .venv/bin/activate
                echo "Activated existing virtual environment"
            else
                echo "Creating new virtual environment..."
                python3 -m venv .venv
                source .venv/bin/activate
                echo "Created and activated new virtual environment"
            fi

            # Install/update requirements
            echo "Installing/updating requirements..."
            pip install -q --upgrade pip
            pip install -q -r requirements.txt 2>/dev/null || {
                echo "Warning: Some packages failed to install, continuing anyway..."
            }

            echo "Final package list:"
            pip list
        else
            echo "Environment: Local development"

            # Check for and activate local virtual environment
            if [[ -f ".venv/bin/activate" ]]; then
                source .venv/bin/activate
                echo "Activated local virtual environment"
            elif [[ -f "venv/bin/activate" ]]; then
                source venv/bin/activate
                echo "Activated local virtual environment"
            else
                echo "Warning: No virtual environment found. PyTorch may not be available."
                echo "To create one: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
            fi

            pip list
        fi

        echo "Dependency check completed: $(date)"
    } 2>&1 | tee "$dep_log"

    print_status "Dependencies logged to: $dep_log"
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
    local train_log="$OUTPUT_DIR/training_${TIMESTAMP}.log"

    mkdir -p "$checkpoint_dir"

    print_info "Configuration:"
    print_info "  Dataset: $DATASET"
    print_info "  Samples: $SAMPLES"
    print_info "  Epochs: $EPOCHS"
    print_info "  Batch Size: $BATCH_SIZE"
    print_info "  Latent: ${LATENT_LEN}x${D_Z}"
    print_info "  Models: $SOURCE_MODEL -> $TARGET_MODEL"
    print_info "  Output: $checkpoint_dir"

    # Save configuration to JSON
    cat > "$OUTPUT_DIR/config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "dataset": "$DATASET",
    "samples": $SAMPLES,
    "epochs": $EPOCHS,
    "batch_size": $BATCH_SIZE,
    "latent_len": $LATENT_LEN,
    "d_z": $D_Z,
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL"
}
EOF

    {
        echo "Training started: $(date)"

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

        echo "Training completed: $(date)"
    } 2>&1 | tee "$train_log"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_status "Training completed successfully"
        print_info "Checkpoint: $checkpoint_dir/epoch$((EPOCHS-1))"
        print_info "Training log: $train_log"

        # Copy diagnostics if generated
        if [[ -f "$checkpoint_dir/diagnostics.jsonl" ]]; then
            cp "$checkpoint_dir/diagnostics.jsonl" "$OUTPUT_DIR/training_diagnostics.jsonl"
            print_info "Diagnostics saved: $OUTPUT_DIR/training_diagnostics.jsonl"
        fi
    else
        print_error "Training failed - check $train_log for details"
        exit 1
    fi
}

# =============================================================================
# EVALUATION - Fixed with output paths
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

    # Create evaluation summary log
    local eval_summary="$OUTPUT_DIR/evaluation_summary_${TIMESTAMP}.log"

    {
        echo "Evaluation started: $(date)"
        echo "Checkpoint: $checkpoint"
        echo "Seeds: $SEEDS"
        echo "Samples: $EVAL_SAMPLES"
    } | tee "$eval_summary"

    # Determine eval sample count
    local eval_sample_count="$EVAL_SAMPLES"
    if [[ "$EVAL_SAMPLES" == "full" ]]; then
        eval_sample_count="-1"
    fi

    # Run evaluation for each seed
    for seed in $SEEDS; do
        print_info "Evaluating with seed $seed..."

        local eval_log="$OUTPUT_DIR/eval_seed${seed}_${TIMESTAMP}.log"
        local output_file="$results_dir/seed${seed}.json"

        {
            echo "Seed $seed evaluation started: $(date)"

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
                --out_json "$output_file"

            echo "Seed $seed evaluation completed: $(date)"
        } 2>&1 | tee "$eval_log"

        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            print_status "Evaluation completed for seed $seed"
            echo "Seed $seed: SUCCESS - Results in $output_file" | tee -a "$eval_summary"

            # Extract key metrics from log
            if grep -q "F1:" "$eval_log"; then
                grep "F1:" "$eval_log" | tail -1 | tee -a "$eval_summary"
            fi
        else
            print_error "Evaluation failed for seed $seed"
            echo "Seed $seed: FAILED - Check $eval_log" | tee -a "$eval_summary"
        fi
    done

    {
        echo "Evaluation completed: $(date)"
        echo "Results directory: $results_dir"
    } | tee -a "$eval_summary"

    # Statistical analysis
    print_info "Running statistical analysis..."

    local stats_log="$OUTPUT_DIR/statistical_analysis_${TIMESTAMP}.log"
    {
        echo "Statistical analysis started: $(date)"

        if [[ -f "scripts/analyze_all_results.py" ]]; then
            python3 scripts/analyze_all_results.py --results_dir "$results_dir" || true
        else
            echo "Note: analyze_all_results.py not found, skipping"
        fi

        echo "Statistical analysis completed: $(date)"
    } 2>&1 | tee "$stats_log"

    print_status "All evaluation logs saved to $OUTPUT_DIR"
}

# =============================================================================
# QUICK TEST - For rapid debugging
# =============================================================================

run_quick_test() {
    print_header "QUICK TEST (1 EPOCH, 100 SAMPLES)"

    # Override settings for quick test
    SAMPLES=100
    EPOCHS=1
    EVAL_SAMPLES=50
    BATCH_SIZE=2

    print_info "Running minimal configuration for debugging"

    # Run training
    run_training

    # Find checkpoint
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
    print_info "  Batch Size: $BATCH_SIZE"
    print_info "  Eval samples: $EVAL_SAMPLES"
    print_info "  Seeds: $SEEDS"
    print_info "  Output: $OUTPUT_DIR"

    # Phase 0: Dependencies
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

    # Phase 4: Linear probe baseline
    if [[ -f "latentwire/linear_probe_baseline.py" ]]; then
        print_header "LINEAR PROBE BASELINE"
        local probe_log="$OUTPUT_DIR/linear_probe_${TIMESTAMP}.log"
        {
            python3 latentwire/linear_probe_baseline.py \
                --checkpoint "$checkpoint" \
                --dataset "$DATASET" \
                --samples 1000
        } 2>&1 | tee "$probe_log" || true
    fi

    # Phase 5: Analysis
    print_header "ANALYSIS"

    if [[ -f "scripts/analyze_all_results.py" ]]; then
        print_info "Running results analysis..."
        python3 scripts/analyze_all_results.py --results_dir "$OUTPUT_DIR/results" || true
    fi

    # Create final log index
    create_log_summary

    print_status "Experiment complete"
    print_info "All results in: $OUTPUT_DIR"
}

# =============================================================================
# SLURM SUBMISSION
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
#SBATCH --mem=40GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.err

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export OUTPUT_DIR="runs/job_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================================="

# Pull latest code
git pull

# Set up environment
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export BATCH_SIZE=4  # For 40GB memory

# Ensure dependencies
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt 2>/dev/null || true
fi

# Run the experiment
bash finalization/RUN.sh EOF_JOB_TYPE

echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="

# List all log files
ls -lh "$OUTPUT_DIR"/*.log

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
LatentWire Experiment Runner v${SCRIPT_VERSION}

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
  BATCH_SIZE    Batch size (default: 4 for 40GB)
  EVAL_SAMPLES  Evaluation samples (default: 1000, use "full" for all)
  OUTPUT_DIR    Output directory (default: runs/exp_TIMESTAMP)
  SEEDS         Random seeds for evaluation (default: "42 123 456")
  DATASET       Dataset to use (default: squad)

Logging:
  All outputs are captured to timestamped log files:
  - master_*.log: Complete experiment output
  - training_*.log: Training phase
  - eval_seed*_*.log: Per-seed evaluation
  - dependencies_*.log: Package installation
  - config.json: Experiment configuration
  - LOG_INDEX.md: Index of all log files

Examples:
  # Quick test
  bash $(basename "$0") quick_test

  # Full experiment with custom settings
  SAMPLES=10000 EPOCHS=12 bash $(basename "$0") experiment

  # Evaluate specific checkpoint
  bash $(basename "$0") eval runs/exp_20240315/checkpoint/epoch7

  # Generate SLURM script
  bash $(basename "$0") slurm experiment

EOF
}

# Main dispatch
case "$COMMAND" in
    train)
        run_training
        create_log_summary
        ;;
    eval)
        run_evaluation "$@"
        create_log_summary
        ;;
    quick_test)
        run_quick_test
        create_log_summary
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

# Final log notice
echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "All logs saved to: $OUTPUT_DIR" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "Completed: $(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"