#!/bin/bash
#SBATCH --job-name=reasoning_analysis
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/reasoning_analysis_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/reasoning_analysis_%j.err

# =============================================================================
# Telepathy Reasoning Failure Analysis
# =============================================================================
# This script runs comprehensive diagnostics to understand WHY the Telepathy
# bridge fails on reasoning tasks (GSM8K) while succeeding on classification.
#
# Tests performed:
# A. First-token accuracy: Does bridge preserve question semantics?
# B. Soft token diversity: Are tokens collapsed or diverse?
# C. Layer comparison: Layer 16 vs Layer 31 (if both checkpoints provided)
# D. Entity preservation: Can bridge transmit specific numbers?
# E. Chain-of-thought: Does reasoning chain survive compression?
#
# Expected runtime: 2-3 hours
#
# Submit with: sbatch telepathy/submit_reasoning_analysis.slurm
# Monitor with: squeue -u $USER
# View log: tail -f runs/reasoning_analysis_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "Telepathy Reasoning Failure Analysis"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
OUTPUT_DIR="runs/reasoning_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Pull latest code
echo ""
echo "Pulling latest code..."
git pull

# =============================================================================
# Configuration
# =============================================================================

# IMPORTANT: Set these paths to your actual checkpoint files
# You can provide one or both checkpoints

# Primary checkpoint (required) - typically Layer 31 for reasoning
CHECKPOINT="${CHECKPOINT:-runs/gsm8k_bridge/bridge.pt}"

# Optional: Layer 16 checkpoint for comparison (Test C)
# If not set, Test C will be skipped
CHECKPOINT_LAYER16="${CHECKPOINT_LAYER16:-}"

# Other parameters
SOURCE_LAYER="${SOURCE_LAYER:-31}"  # 31 for answer-oriented, 16 for concepts
SOFT_TOKENS="${SOFT_TOKENS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

echo ""
echo "Configuration:"
echo "  Primary checkpoint: $CHECKPOINT"
if [ -n "$CHECKPOINT_LAYER16" ]; then
    echo "  Layer 16 checkpoint: $CHECKPOINT_LAYER16"
else
    echo "  Layer 16 checkpoint: Not provided (Test C will be skipped)"
fi
echo "  Source layer: $SOURCE_LAYER"
echo "  Soft tokens: $SOFT_TOKENS"
echo "  Samples per test: $NUM_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# =============================================================================
# Run Analysis
# =============================================================================

echo "=============================================================="
echo "Starting diagnostic tests..."
echo "=============================================================="
echo ""

# Build command
CMD="python telepathy/analyze_reasoning_failure.py \
    --checkpoint $CHECKPOINT \
    --source_layer $SOURCE_LAYER \
    --soft_tokens $SOFT_TOKENS \
    --num_samples $NUM_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --bf16"

# Add Layer 16 checkpoint if provided
if [ -n "$CHECKPOINT_LAYER16" ]; then
    CMD="$CMD --checkpoint_layer16 $CHECKPOINT_LAYER16"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the analysis
$CMD

EXIT_CODE=$?

echo ""
echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Analysis completed successfully!"
else
    echo "Analysis failed with exit code: $EXIT_CODE"
fi
echo "=============================================================="

# =============================================================================
# Push Results
# =============================================================================

echo ""
echo "Pushing results to git..."

git add runs/reasoning_analysis_*
git add telepathy/analyze_reasoning_failure.py
git add telepathy/submit_reasoning_analysis.slurm

git commit -m "results: reasoning failure analysis (SLURM job $SLURM_JOB_ID)

Diagnostic tests on Telepathy bridge reasoning failure:
- First-token accuracy
- Soft token diversity
- Entity preservation (numbers)
- Chain-of-thought preservation
$([ -n \"$CHECKPOINT_LAYER16\" ] && echo \"- Layer 16 vs 31 comparison\")

Checkpoint: $CHECKPOINT
Source layer: $SOURCE_LAYER
Samples: $NUM_SAMPLES

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" || true

git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================================="
