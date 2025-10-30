#!/usr/bin/env bash
"""
Run unified cross-model alignment experiments.
Combines Procrustes and learned adapter experiments optimized for 2 GPUs.
"""

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/unified_experiments}"
SCRIPT_NAME="unified_cross_model_experiments.py"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For debugging CUDA issues
export CUDA_LAUNCH_BLOCKING=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/unified_experiments_${TIMESTAMP}.log"

echo "=================================================================================="
echo "UNIFIED CROSS-MODEL ALIGNMENT EXPERIMENTS (Enhanced with 2024 Research)"
echo "=================================================================================="
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Key Enhancements Based on 2024 Literature:"
echo "  ✓ InfoNCE contrastive loss (τ=0.07) - ESSENTIAL for alignment"
echo "  ✓ CKA similarity metric (superior to SVCCA)"
echo "  ✓ Multi-layer alignment [8, 16, 24]"
echo "  ✓ 10,000 training samples (10x increase)"
echo "  ✓ Batch size 16 (4x increase for contrastive)"
echo "  ✓ NO MAX_LENGTH truncation - using full sequences"
echo "  ✓ 10 epochs with cosine annealing"
echo ""
echo "GPU Allocation:"
echo "  - CPU: Procrustes alignment (no GPU needed)"
echo "  - GPU 0: Linear adapter"
echo "  - GPU 1: Affine adapter"
echo "  - GPU 0 (sequential): LoRA adapter"
echo ""
echo "Expected Results (from literature):"
echo "  - CKA scores: 0.6-0.7 (vs 0.3 baseline)"
echo "  - Generation loss: <2.0 (vs 3.4 baseline)"
echo "  - No mode collapse (contrastive prevents)"
echo ""
echo "Starting experiments..."
echo "=================================================================================="
echo ""

# Run the unified experiment script with comprehensive logging
{
    python experimental/learning/$SCRIPT_NAME
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================================================="
echo "EXPERIMENTS COMPLETE"
echo "=================================================================================="
echo "Results saved to:"
echo "  - Procrustes: $OUTPUT_DIR/procrustes_results_*.json"
echo "  - Linear adapter: runs/learned_adapters/linear_*.json"
echo "  - Affine adapter: runs/learned_adapters/affine_*.json"
echo "  - LoRA adapter: runs/learned_adapters/lora_*.json"
echo "  - Full log: $LOG_FILE"
echo "=================================================================================="