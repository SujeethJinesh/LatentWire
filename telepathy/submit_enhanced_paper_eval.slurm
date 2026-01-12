#!/bin/bash
#SBATCH --job-name=telepathy_enhanced
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/enhanced_eval_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/enhanced_eval_%j.err

# =============================================================================
# Enhanced Telepathy Paper Evaluation (Robust Version)
# =============================================================================
# This script runs the enhanced evaluation with:
#
# 1. 5 seeds (42, 123, 456, 789, 1011) for robust statistical analysis
# 2. 4 token configs (8, 16, 24, 32) for comprehensive ablation
# 3. 3 classification datasets: SST-2, AG News, TREC
# 4. 3 reasoning benchmarks: BoolQ, ARC-Easy, GSM8K
# 5. Improved text-relay baseline with task-specific prompting
# 6. Enhanced resume capability with atomic checkpointing
# 7. Comprehensive memory management for 12-hour H100 execution
#
# Robustness Features:
# - Git pull at start, push at end
# - Auto-resume from incomplete runs
# - Comprehensive error handling and logging
# - 128GB memory allocation for stability
#
# Estimated runtime: ~10-12 hours on single H100 GPU
#
# Submit with: sbatch telepathy/submit_enhanced_paper_eval.slurm
# Monitor with: squeue -u $USER
# View logs: tail -f runs/enhanced_eval_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "TELEPATHY ENHANCED PAPER EVALUATION"
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
ENHANCED_RESULTS_DIR="runs/enhanced_paper_results"
mkdir -p "$ENHANCED_RESULTS_DIR"

# Find the most recent run directory if one exists
MOST_RECENT_RUN=$(ls -dt "$ENHANCED_RESULTS_DIR"/run_* 2>/dev/null | head -1)

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
OUTPUT_DIR="$ENHANCED_RESULTS_DIR"
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
# Run Enhanced Evaluation
# =========================================================================
# The script handles:
# - Bridge training and evaluation on 6 datasets x 5 seeds x 4 token configs
# - All baselines (zero-shot, text-relay, linear-probe, etc.)
# - Production metrics (memory, latency, throughput)
# - Statistical significance testing
# - LaTeX table generation

echo "Starting enhanced evaluation..."
echo "Datasets: sst2, agnews, trec, boolq, arc_easy, gsm8k (6 datasets)"
echo "Seeds: 42, 123, 456, 789, 1011 (5 seeds)"
echo "Token ablation: 8, 16, 24, 32 (4 configs)"
echo ""

# Build the command with optional resume
PYTHON_CMD="python telepathy/run_enhanced_paper_evaluation.py \
    --output_dir $OUTPUT_DIR \
    --gpu 0 \
    --log_level INFO"

# Add resume flag if we have an incomplete run to resume from
if [ -n "$RESUME_DIR" ]; then
    PYTHON_CMD="$PYTHON_CMD --resume_dir $RESUME_DIR"
fi

# Run the evaluation with tee to capture logs
eval "$PYTHON_CMD" 2>&1 | tee "${OUTPUT_DIR}/enhanced_evaluation_${SLURM_JOB_ID}.log"

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
if [ -z "$COMPLETE_RESULTS" ]; then
    # Also check for enhanced_evaluation_results.json
    COMPLETE_RESULTS=$(ls "${OUTPUT_DIR}"/run_*/enhanced_evaluation_results.json 2>/dev/null | head -1)
fi
if [ -z "$COMPLETE_RESULTS" ]; then
    # Check directly in output dir
    COMPLETE_RESULTS=$(ls "${OUTPUT_DIR}"/enhanced_evaluation_results.json 2>/dev/null | head -1)
fi

if [ -n "$COMPLETE_RESULTS" ]; then
    echo "Complete results saved: $COMPLETE_RESULTS"

    # Display summary
    python -c "
import json
import sys
try:
    with open('$COMPLETE_RESULTS') as f:
        results = json.load(f)

    if 'metadata' in results:
        print(f\"Total time: {results['metadata'].get('total_time_hours', 'N/A')} hours\")
        print(f\"Seeds: {results['metadata'].get('seeds', 'N/A')}\")
        print(f\"Token configs: {results['metadata'].get('token_configs', 'N/A')}\")

    # Classification results
    if 'classification' in results and 'aggregated' in results['classification']:
        print()
        print('Classification Results (Bridge):')
        bridge_results = results['classification']['aggregated'].get('bridge', {})
        for dataset in bridge_results.keys():
            print(f'  {dataset}:')
            for tokens, stats in bridge_results[dataset].items():
                if isinstance(stats, dict):
                    print(f\"    {tokens} tokens: {stats.get('mean', 'N/A'):.3f} +/- {stats.get('std', 'N/A'):.3f}\")

    # Reasoning results
    if 'reasoning' in results and 'aggregated' in results['reasoning']:
        print()
        print('Reasoning Benchmarks (Llama):')
        llama_results = results['reasoning']['aggregated'].get('llama', {})
        for benchmark, stats in llama_results.items():
            if isinstance(stats, dict):
                print(f\"  {benchmark}: {stats.get('mean', 'N/A'):.3f} +/- {stats.get('std', 'N/A'):.3f}\")
except Exception as e:
    print(f'Could not parse results: {e}')
" 2>/dev/null || echo "Could not display results summary"
fi

# Display statistical analysis if available
STATISTICAL_ANALYSIS=$(ls "${OUTPUT_DIR}"/run_*/statistical_analysis.json 2>/dev/null | head -1)
if [ -n "$STATISTICAL_ANALYSIS" ]; then
    echo ""
    echo "Statistical Analysis:"
    cat "$STATISTICAL_ANALYSIS" | python -m json.tool 2>/dev/null | head -100
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
git add runs/enhanced_eval_*.log runs/enhanced_eval_*.err 2>/dev/null || true

# Add all files in the enhanced results directory (JSON, logs, LaTeX, etc.)
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
COMMIT_MSG="results: enhanced telepathy paper evaluation (SLURM job $SLURM_JOB_ID)

Datasets: sst2, agnews, trec, boolq, arc_easy, gsm8k (6 datasets)
Seeds: 42, 123, 456, 789, 1011 (5 seeds)
Token configs: 8, 16, 24, 32 (4 ablations)
Exit status: $EVAL_STATUS

Features:
- 5 seeds for robust statistical analysis
- 4 token configs for comprehensive ablation
- 6 datasets (3 classification + 3 reasoning)
- Enhanced resume capability with atomic checkpointing
- Comprehensive memory management (128GB allocation)

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
echo "Total runtime: $SECONDS seconds"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - ${OUTPUT_DIR}/run_*/complete_results.json"
echo "  - ${OUTPUT_DIR}/run_*/statistical_analysis.json"
echo "  - ${OUTPUT_DIR}/run_*/paper_tables.tex"
echo "  - ${OUTPUT_DIR}/enhanced_evaluation_${SLURM_JOB_ID}.log"
echo "  - runs/enhanced_eval_${SLURM_JOB_ID}.log (SLURM output)"
echo ""
echo "Configuration:"
echo "  - Datasets: 6 (sst2, agnews, trec, boolq, arc_easy, gsm8k)"
echo "  - Seeds: 5 (42, 123, 456, 789, 1011)"
echo "  - Token configs: 4 (8, 16, 24, 32)"
echo "  - Memory allocation: 128GB"
if [ -n "$RESUME_DIR" ]; then
    echo "  - Resumed from: $RESUME_DIR"
fi
echo ""
echo "To view results locally, run:"
echo "  git pull"
echo "  cat ${OUTPUT_DIR}/run_*/paper_tables.tex"
echo "=============================================================="
