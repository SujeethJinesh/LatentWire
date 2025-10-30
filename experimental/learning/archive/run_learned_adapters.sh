#!/usr/bin/env bash
set -e

# Learned Adapter Ablation - Parallel execution on GPUs 1, 2, 3
# GPU 0 is reserved for Procrustes experiment

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/learned_adapters}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "================================================================================"
echo "LEARNED ADAPTER ABLATION - Parallel Execution"
echo "================================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "GPU Allocation:"
echo "  GPU 1: Linear Adapter  (16.8M params)"
echo "  GPU 2: Affine Adapter  (16.8M params)"
echo "  GPU 3: LoRA-8 Adapter  (65k params)"
echo ""
echo "Starting all 3 adapters in parallel..."
echo "================================================================================"
echo ""

# Launch all 3 in parallel with proper GPU visibility
CUDA_VISIBLE_DEVICES=1 python3 learned_adapter_ablation.py linear &
PID_LINEAR=$!
echo "Linear adapter started on GPU 1 (PID: $PID_LINEAR)"

CUDA_VISIBLE_DEVICES=2 python3 learned_adapter_ablation.py affine &
PID_AFFINE=$!
echo "Affine adapter started on GPU 2 (PID: $PID_AFFINE)"

CUDA_VISIBLE_DEVICES=3 python3 learned_adapter_ablation.py lora &
PID_LORA=$!
echo "LoRA adapter started on GPU 3 (PID: $PID_LORA)"

echo ""
echo "All processes launched. Waiting for completion..."
echo "You can monitor progress in:"
echo "  - $OUTPUT_DIR/linear_gpu1_*.log"
echo "  - $OUTPUT_DIR/affine_gpu2_*.log"
echo "  - $OUTPUT_DIR/lora_gpu3_*.log"
echo ""

# Wait for all to complete
wait $PID_LINEAR
EXIT_LINEAR=$?
echo "Linear adapter completed (exit code: $EXIT_LINEAR)"

wait $PID_AFFINE
EXIT_AFFINE=$?
echo "Affine adapter completed (exit code: $EXIT_AFFINE)"

wait $PID_LORA
EXIT_LORA=$?
echo "LoRA adapter completed (exit code: $EXIT_LORA)"

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================================"
echo "Exit codes: Linear=$EXIT_LINEAR, Affine=$EXIT_AFFINE, LoRA=$EXIT_LORA"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo "  - *_results_*.json (comprehensive metrics + generations)"
echo "  - *_gpu*_*.log (detailed training logs)"
echo "================================================================================"

# Exit with error if any failed
if [ $EXIT_LINEAR -ne 0 ] || [ $EXIT_AFFINE -ne 0 ] || [ $EXIT_LORA -ne 0 ]; then
    echo "ERROR: One or more experiments failed!"
    exit 1
fi

exit 0
