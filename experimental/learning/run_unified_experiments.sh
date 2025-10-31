#!/usr/bin/env bash
# Run unified cross-model alignment experiments.
# Combines Procrustes and learned adapter experiments optimized for 2 GPUs.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration - always use absolute paths
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs/unified_experiments}"
SCRIPT_NAME="unified_cross_model_experiments.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Set up environment
export PYTHONPATH=.

# Auto-detect platform (Mac vs HPC)
if [[ "$(uname)" == "Darwin" ]]; then
    echo "==> Detected Mac platform - configuring for MPS/CPU"
    PLATFORM="mac"

    # Mac-specific settings
    export PYTORCH_ENABLE_MPS_FALLBACK=1  # Allow fallback for unsupported MPS ops
    export DISABLE_FLASH_ATTENTION=1  # Flash Attention not available on Mac
    export USE_MPS=1  # Signal to Python script to use MPS

    # No CUDA on Mac
    unset CUDA_VISIBLE_DEVICES

    # Adjust batch size and samples for Mac memory constraints
    export MAC_BATCH_SIZE=4  # Smaller batch for Mac
    export MAC_SAMPLES=1000  # Fewer samples for testing
    export MAC_EPOCHS=2  # Fewer epochs for testing
else
    echo "==> Detected HPC platform - configuring for H100 GPUs"
    PLATFORM="hpc"

    # H100 GPU optimizations
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}  # Use all 4 H100 GPUs
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management on H100
    export NCCL_P2P_DISABLE=1  # Avoid P2P issues on some HPC systems

    # Enable TF32 for H100 (3x speedup on matrix operations)
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

    # Signal HPC mode
    export USE_CUDA=1

    # For debugging CUDA issues (disable in production for performance)
    # export CUDA_LAUNCH_BLOCKING=1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/unified_experiments_${TIMESTAMP}.log"

echo "=================================================================================="
echo "UNIFIED CROSS-MODEL ALIGNMENT EXPERIMENTS (Enhanced with 2025 Research)"
echo "=================================================================================="
echo "Timestamp: $TIMESTAMP"
echo "Platform: $PLATFORM"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Key Enhancements Based on 2025 Literature:"
echo "  ✓ InfoNCE contrastive loss (τ=0.07) - ESSENTIAL for alignment"
echo "  ✓ CKA similarity metric (superior to SVCCA)"
echo "  ✓ Multi-layer alignment [8, 16, 24]"
echo ""

if [[ "$PLATFORM" == "mac" ]]; then
    echo "Mac Configuration:"
    echo "  ✓ Using MPS/CPU instead of CUDA"
    echo "  ✓ Batch size: 4 (Mac memory constraints)"
    echo "  ✓ Samples: 1000 (for faster testing)"
    echo "  ✓ Epochs: 2 (for faster testing)"
    echo "  ✓ Flash Attention disabled (not supported on MPS)"
    echo ""
    echo "Device Allocation:"
    echo "  - MPS device: All experiments (if available)"
    echo "  - CPU fallback: For unsupported operations"
else
    echo "HPC Configuration:"
    echo "  ✓ 10,000 training samples (10x increase)"
    echo "  ✓ Batch size 16 (4x increase for contrastive)"
    echo "  ✓ NO MAX_LENGTH truncation - using full sequences"
    echo "  ✓ 10 epochs with cosine annealing"
    echo ""
    echo "GPU Allocation (4 GPUs available):"
    echo "  - GPU 0 & 1: Procrustes alignment (models distributed)"
    echo "  - GPU 0: Linear adapter (parallel)"
    echo "  - GPU 1: Affine adapter (parallel)"
    echo "  - GPU 2: LoRA adapter (parallel)"
    echo "  - GPU 3: Available for overflow/additional experiments"
fi

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
    python "$SCRIPT_PATH"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================================================="
echo "EXPERIMENTS COMPLETE"
echo "=================================================================================="
echo "Results saved to:"
echo "  - Procrustes: $OUTPUT_DIR/procrustes_results_*.json"
echo "  - Linear adapter: $SCRIPT_DIR/runs/learned_adapters/linear_*.json"
echo "  - Affine adapter: $SCRIPT_DIR/runs/learned_adapters/affine_*.json"
echo "  - LoRA adapter: $SCRIPT_DIR/runs/learned_adapters/lora_*.json"
echo "  - Full log: $LOG_FILE"
echo "=================================================================================="