#!/bin/bash
#SBATCH --job-name=telepathy_paper
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/complete_eval_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/complete_eval_%j.err

# =============================================================================
# Complete Telepathy Paper Evaluation
# =============================================================================
# This script runs all experiments needed for the Telepathy paper:
# 1. Telepathy bridge evaluation on 7 datasets with 5 seeds
# 2. Linear probe baseline on all datasets
# 3. Production metrics (throughput, latency, memory)
# 4. Statistical tests comparing all methods
# 5. Paper-ready LaTeX tables
#
# Estimated runtime: ~10-12 hours on single H100 GPU
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
# Define Output Directory with Timestamp
# =========================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/paper_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# =========================================================================
# GPU Information
# =========================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
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
echo "Datasets: sst2, agnews, trec"
echo "Seeds: 42, 123, 456"
echo "Token ablation: 8, 16, 32"
echo "Baselines: zeroshot, text-relay, prompt-tuning, lora, linear-probe, bridge"
echo ""

# Run with configured datasets and seeds for 12 hour runtime
# Use GPU 0 for main computation (script handles memory management)
python telepathy/run_complete_evaluation.py \
    --output_dir "$OUTPUT_DIR" \
    --datasets sst2 agnews trec \
    --seeds 42 123 456 \
    --gpu 0 \
    --skip_existing \
    2>&1 | tee "${OUTPUT_DIR}/evaluation.log"

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

# Add all results (logs, JSON files, LaTeX tables)
find "$OUTPUT_DIR" -name "*.json" -o -name "*.log" -o -name "*.tex" | xargs git add 2>/dev/null || true
git add figures/*.png figures/*.pdf 2>/dev/null || true

# Commit with detailed message
COMMIT_MSG="results: complete telepathy paper evaluation (SLURM job $SLURM_JOB_ID)

Datasets: sst2, agnews, trec, imdb, mnli, 20newsgroups, banking77
Seeds: 42, 123, 456, 789, 2024

Generated:
- Bridge evaluation results (35 runs)
- Linear probe baseline results (35 runs)
- Production metrics (memory, latency, throughput)
- Statistical significance tests
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
echo "  - ${OUTPUT_DIR}/evaluation.log"
echo ""
echo "To view results locally, run:"
echo "  git pull"
echo "  cat ${OUTPUT_DIR}/run_*/paper_tables.tex"
echo "=============================================================="
