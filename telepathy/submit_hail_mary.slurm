#!/bin/bash
#SBATCH --job-name=telepathy_hail_mary
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/hail_mary_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/hail_mary_%j.err

# =============================================================================
# Telepathy Hail Mary Experiments
# =============================================================================
# This script runs the HIGH PRIORITY experiments identified in REVIEW.md:
#
# 1. Simple Reasoning Datasets (~3h)
#    - ARC-Easy: Elementary science questions
#    - WinoGrande: Common sense reasoning
#    - HellaSwag: Situational reasoning
#
# 2. Layer-31 Linear Probe (~1h)
#    - Fair comparison with bridge (both use layer 31)
#    - Addresses layer mismatch issue in original evaluation
#
# 3. Classification with Bridge (~3h)
#    - SST-2: Sentiment analysis
#    - AG News: Topic classification
#
# 4. Statistical Significance Testing (~2h)
#    - Paired t-tests with Bonferroni correction
#    - Effect sizes (Cohen's d)
#
# Key improvements over standard evaluation:
# - Uses layer 31 for linear probe (same as bridge)
# - Includes simple reasoning datasets that run quickly
# - 3 seeds for statistical analysis
# - Fast mode (500 training steps)
#
# Estimated runtime: ~9-10 hours on single H100 GPU
#
# Submit with: sbatch telepathy/submit_hail_mary.slurm
# Monitor with: squeue -u $USER
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "TELEPATHY HAIL MARY EXPERIMENTS"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo ""
echo "Focus: Simple reasoning + Layer-31 probe + Statistical tests"
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
# Git Setup
# =========================================================================
echo ""
echo "Git Configuration Check..."

git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# =========================================================================
# Define Output Directory
# =========================================================================
HAIL_MARY_DIR="runs/hail_mary_results"
mkdir -p "$HAIL_MARY_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$HAIL_MARY_DIR"

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
# Run Hail Mary Evaluation
# =========================================================================
echo ""
echo "Starting hail mary evaluation..."
echo ""
echo "Configuration:"
echo "  - Datasets: arc_easy, winogrande, hellaswag, sst2, agnews"
echo "  - Linear probe layer: 31 (matching bridge)"
echo "  - Seeds: 42, 123, 456"
echo "  - Token configs: 8, 24"
echo "  - Mode: fast (500 training steps)"
echo ""

python telepathy/run_enhanced_paper_evaluation.py \
    --output_dir "$OUTPUT_DIR" \
    --gpu 0 \
    --hail_mary_mode \
    --fast_mode \
    --run_diagnostics \
    --log_level INFO \
    2>&1 | tee "${OUTPUT_DIR}/hail_mary_${SLURM_JOB_ID}.log"

EVAL_STATUS=$?

echo ""
echo "=============================================================="
echo "Evaluation completed with status: $EVAL_STATUS"
echo "=============================================================="

# =========================================================================
# Results Summary
# =========================================================================
echo ""
echo "Results Summary:"
echo "------------------------------------------------------------"

# Find results
RESULTS_FILE=$(ls "${OUTPUT_DIR}"/run_*/enhanced_evaluation_results.json 2>/dev/null | head -1)
if [ -z "$RESULTS_FILE" ]; then
    RESULTS_FILE=$(ls "${OUTPUT_DIR}"/enhanced_evaluation_results.json 2>/dev/null | head -1)
fi

if [ -n "$RESULTS_FILE" ]; then
    echo "Results saved: $RESULTS_FILE"

    python -c "
import json
try:
    with open('$RESULTS_FILE') as f:
        results = json.load(f)

    print()
    print('=== HAIL MARY RESULTS ===')
    print()

    # Get all methods and datasets
    methods = ['bridge_8', 'bridge_24', 'zeroshot_llama', 'zeroshot_mistral', 'linear_probe']

    # Simple reasoning datasets
    for dataset in ['arc_easy', 'winogrande', 'hellaswag']:
        print(f'\\n{dataset.upper()}:')
        for method in methods:
            if method in results.get('classification', {}).get('aggregated', {}):
                stats = results['classification']['aggregated'][method].get(dataset, {})
                if stats and 'mean' in stats:
                    print(f'  {method}: {stats[\"mean\"]:.1f}% +/- {stats[\"std\"]:.1f}%')

    # Classification datasets
    for dataset in ['sst2', 'agnews']:
        print(f'\\n{dataset.upper()}:')
        for method in methods:
            if method in results.get('classification', {}).get('aggregated', {}):
                stats = results['classification']['aggregated'][method].get(dataset, {})
                if stats and 'mean' in stats:
                    print(f'  {method}: {stats[\"mean\"]:.1f}% +/- {stats[\"std\"]:.1f}%')

except Exception as e:
    print(f'Could not parse results: {e}')
" 2>/dev/null || echo "Could not display results summary"
fi

# Display statistical analysis
STATS_FILE=$(ls "${OUTPUT_DIR}"/run_*/statistical_analysis.json 2>/dev/null | head -1)
if [ -n "$STATS_FILE" ]; then
    echo ""
    echo "=== STATISTICAL SIGNIFICANCE ==="
    python -c "
import json
try:
    with open('$STATS_FILE') as f:
        stats = json.load(f)
    for ds, ds_stats in stats.get('per_dataset', {}).items():
        print(f'\\n{ds}:')
        for comp, comp_stats in ds_stats.get('comparisons', {}).items():
            p_val = comp_stats.get('p_value', 'N/A')
            sig = '*' if isinstance(p_val, float) and p_val < 0.05 else ''
            if isinstance(p_val, float):
                print(f'  {comp}: p={p_val:.4f}{sig}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null || echo "Could not display statistical analysis"
fi

# =========================================================================
# Git Commit and Push
# =========================================================================
echo ""
echo "Saving results to git..."

git add runs/hail_mary_*.log runs/hail_mary_*.err 2>/dev/null || true
git add "$OUTPUT_DIR"/ 2>/dev/null || true
git add figures/*.png figures/*.pdf 2>/dev/null || true

echo ""
echo "Files staged for commit:"
git status --short

COMMIT_MSG="results: hail mary experiments (SLURM job $SLURM_JOB_ID)

Datasets: arc_easy, winogrande, hellaswag, sst2, agnews
Simple reasoning + layer-31 probe + statistical tests
Seeds: 42, 123, 456
Exit status: $EVAL_STATUS

Key improvements:
- Layer 31 linear probe (fair comparison with bridge)
- Simple reasoning datasets (ARC-Easy, WinoGrande, HellaSwag)
- Statistical significance testing with Bonferroni correction
- Fast mode (500 train steps)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, attempting to push..."

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
echo "HAIL MARY EXPERIMENTS COMPLETED"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo ""
echo "Key results to check:"
echo "  - Simple reasoning: ARC-Easy, WinoGrande, HellaSwag"
echo "  - Layer-31 probe vs bridge comparison"
echo "  - Statistical significance tests"
echo ""
echo "Files:"
echo "  - ${OUTPUT_DIR}/run_*/enhanced_evaluation_results.json"
echo "  - ${OUTPUT_DIR}/run_*/statistical_analysis.json"
echo "  - ${OUTPUT_DIR}/hail_mary_${SLURM_JOB_ID}.log"
echo ""
echo "To view results locally:"
echo "  git pull"
echo "  python -c \"import json; print(json.load(open('${OUTPUT_DIR}/run_*/enhanced_evaluation_results.json')))\""
echo "=============================================================="
