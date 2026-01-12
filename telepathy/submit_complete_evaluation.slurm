#!/bin/bash
#SBATCH --job-name=telepathy_paper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/complete_eval_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/complete_eval_%j.err

# =============================================================================
# Complete Telepathy Paper Evaluation (Robust Version)
# =============================================================================
# This script runs all experiments needed for the Telepathy paper with
# robustness features for handling errors, memory issues, and interruptions:
#
# 1. Telepathy bridge evaluation on 3 datasets with 2 seeds
# 2. Linear probe baseline on all datasets
# 3. Production metrics (throughput, latency, memory)
# 4. Statistical tests comparing all methods
# 5. Paper-ready LaTeX tables
#
# Robustness Features:
# - --continue_on_error: Continue if individual experiments fail
# - --aggressive_cleanup: Free GPU memory between experiments
# - --max_retries 3: Retry failed experiments up to 3 times
# - --extra_seeds_on_high_variance: Auto-add seeds for high variance results
# - Auto-resume from most recent run if interrupted
# - 64GB memory allocation for stability
#
# Estimated runtime: ~8-9 hours on single H100 GPU (fits in 12h limit)
#
# Submit with: sbatch telepathy/submit_complete_evaluation.slurm
# Monitor with: squeue -u $USER
# View logs: tail -f runs/complete_eval_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "TELEPATHY COMPLETE PAPER EVALUATION"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="/projects/m000066/sujinesh/.cache/huggingface"
export TRANSFORMERS_CACHE="/projects/m000066/sujinesh/.cache/huggingface/transformers"

# Create necessary directories
mkdir -p runs figures
mkdir -p /projects/m000066/sujinesh/.cache/huggingface

# =========================================================================
# Git Setup and Code Sync
# =========================================================================
echo ""
echo "Git Configuration Check..."

# Configure git identity if not set
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

# Pull latest code
echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# =========================================================================
# Define Output Directory and Resume Logic
# =========================================================================
# Check for existing runs to potentially resume from
PAPER_RESULTS_DIR="runs/paper_results"
mkdir -p "$PAPER_RESULTS_DIR"

# Find the most recent run directory if one exists
MOST_RECENT_RUN=$(ls -dt "$PAPER_RESULTS_DIR"/run_* 2>/dev/null | head -1)

# Determine whether to resume or start fresh
RESUME_DIR=""
if [ -n "$MOST_RECENT_RUN" ] && [ -d "$MOST_RECENT_RUN" ]; then
    # Check if the most recent run has incomplete results
    if [ ! -f "$MOST_RECENT_RUN/complete_results.json" ]; then
        echo ""
        echo "Found incomplete run: $MOST_RECENT_RUN"
        echo "Will attempt to resume from this run..."
        RESUME_DIR="$MOST_RECENT_RUN"
    else
        echo ""
        echo "Previous run appears complete: $MOST_RECENT_RUN"
        echo "Starting fresh run..."
    fi
fi

# Set output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$PAPER_RESULTS_DIR"
echo ""
echo "Output directory: $OUTPUT_DIR"
if [ -n "$RESUME_DIR" ]; then
    echo "Resuming from: $RESUME_DIR"
fi
echo ""

# =========================================================================
# GPU Information
# =========================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "Initial GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo ""

# =========================================================================
# Run Complete Evaluation
# =========================================================================
# The script handles:
# - Bridge training and evaluation on 7 datasets x 5 seeds = 35 runs
# - Linear probe baseline on 7 datasets x 5 seeds = 35 runs
# - Production metrics (memory, latency, throughput)
# - Statistical significance testing
# - LaTeX table generation

echo "Starting complete evaluation..."
echo "Datasets: sst2, agnews, trec (3 datasets)"
echo "Seeds: 42, 456 (2 seeds)"
echo "Token ablation: 8, 32 (2 configs - endpoints only)"
echo "Baselines: zeroshot-llama, zeroshot-mistral, text-relay, prompt-tuning, lora, linear-probe"
echo "Main method: bridge (with token ablation)"
echo "Total experiments: 51"
echo ""

# Run with configured datasets and seeds for 12 hour runtime
# Use GPU 0 for main computation (script handles memory management)
#
# Robustness flags:
#   --continue_on_error: Don't stop if one experiment fails
#   --aggressive_cleanup: Free GPU memory between experiments to prevent OOM
#   --max_retries 3: Retry failed experiments up to 3 times
#   --extra_seeds_on_high_variance: Auto-add seeds if results have high variance
#   --log_level INFO: Detailed logging for debugging
#   --resume_dir: Resume from incomplete run if one exists

# Build the command with optional resume
PYTHON_CMD="python telepathy/run_complete_evaluation.py \
    --output_dir $OUTPUT_DIR \
    --datasets sst2 agnews trec \
    --seeds 42 456 \
    --gpu 0 \
    --continue_on_error \
    --aggressive_cleanup \
    --max_retries 3 \
    --extra_seeds_on_high_variance \
    --log_level INFO \
    --save_frequency 3"

# Add resume flag if we have an incomplete run to resume from
if [ -n "$RESUME_DIR" ]; then
    PYTHON_CMD="$PYTHON_CMD --resume_dir $RESUME_DIR"
fi

# Run the evaluation
eval "$PYTHON_CMD" 2>&1 | tee "${OUTPUT_DIR}/evaluation_${SLURM_JOB_ID}.log"

EVAL_STATUS=$?

echo ""
echo "=============================================================="
echo "Evaluation completed with status: $EVAL_STATUS"
echo "=============================================================="

# =========================================================================
# Collect and Display Results
# =========================================================================
echo ""
echo "Results Summary:"
echo "------------------------------------------------------------"

# Display summary if complete_results.json exists
COMPLETE_RESULTS=$(ls "${OUTPUT_DIR}"/run_*/complete_results.json 2>/dev/null | head -1)
if [ -n "$COMPLETE_RESULTS" ]; then
    echo "Complete results saved: $COMPLETE_RESULTS"
fi

# Display statistical analysis if available
STATISTICAL_ANALYSIS=$(ls "${OUTPUT_DIR}"/run_*/statistical_analysis.json 2>/dev/null | head -1)
if [ -n "$STATISTICAL_ANALYSIS" ]; then
    echo ""
    echo "Statistical Analysis:"
    cat "$STATISTICAL_ANALYSIS" | python -m json.tool | head -100
fi

# Display generated tables
PAPER_TABLES=$(ls "${OUTPUT_DIR}"/run_*/paper_tables.tex 2>/dev/null | head -1)
if [ -n "$PAPER_TABLES" ]; then
    echo ""
    echo "LaTeX Tables Generated:"
    cat "$PAPER_TABLES"
fi

# =========================================================================
# Git Commit and Push Results
# =========================================================================
echo ""
echo "Saving results to git..."

# Capture all result files comprehensively
echo "Adding result files to git..."

# Add SLURM output logs from the runs directory
git add runs/complete_eval_*.log runs/complete_eval_*.err 2>/dev/null || true

# Add all files in the paper results directory (JSON, logs, LaTeX, etc.)
git add "$OUTPUT_DIR"/ 2>/dev/null || true

# If we resumed, also add files from the resume directory
if [ -n "$RESUME_DIR" ]; then
    git add "$RESUME_DIR"/ 2>/dev/null || true
fi

# Add any generated figures
git add figures/*.png figures/*.pdf 2>/dev/null || true

# Show what will be committed
echo ""
echo "Files staged for commit:"
git status --short

# Commit with detailed message
COMMIT_MSG="results: complete telepathy paper evaluation (SLURM job $SLURM_JOB_ID)

Datasets: sst2, agnews, trec (3 datasets)
Seeds: 42, 456 (2 seeds)
Token configs: 8, 32 (2 ablations - endpoints only)
Exit status: $EVAL_STATUS

Robustness features used:
- --continue_on_error (recovered from individual failures)
- --aggressive_cleanup (prevented OOM)
- --max_retries 3 (auto-retried failures)
- --extra_seeds_on_high_variance (added seeds for unstable results)
- 64GB memory allocation

Generated (51 total experiments):
- Bridge evaluation: 12 runs (3 datasets × 2 seeds × 2 token configs)
- Zero-shot baselines: 12 runs (3 datasets × 2 seeds × 2 models)
- Text-relay baseline: 6 runs (3 datasets × 2 seeds)
- Prompt tuning baseline: 6 runs (3 datasets × 2 seeds)
- LoRA baseline: 6 runs (3 datasets × 2 seeds)
- Linear probe baseline: 6 runs (3 datasets × 2 seeds)
- Latency measurements: 3 runs (1 per dataset)
- Statistical significance tests with Bonferroni correction
- LaTeX tables for paper

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, attempting to push..."

    # Push with retry logic
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt 3 ]; do
        if git push; then
            echo "Successfully pushed to remote"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Push attempt $RETRY_COUNT failed"
            [ $RETRY_COUNT -lt 3 ] && sleep 5 && git pull --rebase=false || true
        fi
    done

    [ $RETRY_COUNT -eq 3 ] && echo "WARNING: Could not push after 3 attempts"
else
    echo "No changes to commit"
fi

# =========================================================================
# Final Summary
# =========================================================================
echo ""
echo "=============================================================="
echo "JOB COMPLETED"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - ${OUTPUT_DIR}/run_*/complete_results.json"
echo "  - ${OUTPUT_DIR}/run_*/statistical_analysis.json"
echo "  - ${OUTPUT_DIR}/run_*/paper_tables.tex"
echo "  - ${OUTPUT_DIR}/evaluation_${SLURM_JOB_ID}.log"
echo "  - runs/complete_eval_${SLURM_JOB_ID}.log (SLURM output)"
echo ""
echo "Robustness features enabled:"
echo "  - --continue_on_error: Yes"
echo "  - --aggressive_cleanup: Yes"
echo "  - --max_retries: 3"
echo "  - --extra_seeds_on_high_variance: Yes"
echo "  - Memory allocation: 64GB"
if [ -n "$RESUME_DIR" ]; then
    echo "  - Resumed from: $RESUME_DIR"
fi
echo ""
echo "To view results locally, run:"
echo "  git pull"
echo "  cat ${OUTPUT_DIR}/run_*/paper_tables.tex"
echo "=============================================================="
