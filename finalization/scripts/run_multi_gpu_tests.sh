#!/usr/bin/env bash
# Test script for multi-GPU setup verification
# Tests CUDA_VISIBLE_DEVICES and batch size scaling

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/gpu_tests}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/multi_gpu_test_${TIMESTAMP}.log"

echo "=============================================================="
echo "Multi-GPU Setup Test Suite"
echo "=============================================================="
echo "Timestamp: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Function to run test with specific CUDA_VISIBLE_DEVICES
run_gpu_test() {
    local test_name=$1
    local cuda_devices=$2
    local extra_args=${3:-""}

    echo "--------------------------------------------------------------"
    echo "Test: $test_name"
    echo "CUDA_VISIBLE_DEVICES: $cuda_devices"
    echo "--------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=$cuda_devices python "$PROJECT_ROOT/scripts/test_multi_gpu_setup.py" \
        --output-json "$OUTPUT_DIR/${test_name}_results.json" \
        $extra_args 2>&1 | tee -a "$LOG_FILE"

    echo ""
}

# Main test execution
{
    echo "Starting multi-GPU tests at $(date)"
    echo ""

    # Test 1: Default (all GPUs)
    echo "=============================================================="
    echo "Test 1: Default Configuration (All Available GPUs)"
    echo "=============================================================="
    python "$PROJECT_ROOT/scripts/test_multi_gpu_setup.py" \
        --output-json "$OUTPUT_DIR/default_results.json" 2>&1

    echo ""

    # Test 2: Single GPU configurations
    echo "=============================================================="
    echo "Test 2: Single GPU Configurations"
    echo "=============================================================="

    run_gpu_test "single_gpu0" "0"
    run_gpu_test "single_gpu1" "1"

    # Test 3: Multi-GPU configurations
    echo "=============================================================="
    echo "Test 3: Multi-GPU Configurations"
    echo "=============================================================="

    run_gpu_test "dual_gpu_01" "0,1"
    run_gpu_test "dual_gpu_23" "2,3"
    run_gpu_test "triple_gpu" "0,1,2"
    run_gpu_test "quad_gpu" "0,1,2,3"

    # Test 4: Non-contiguous GPU configurations
    echo "=============================================================="
    echo "Test 4: Non-contiguous GPU Configurations"
    echo "=============================================================="

    run_gpu_test "skip_gpu1" "0,2"
    run_gpu_test "skip_gpu2" "0,3"
    run_gpu_test "reverse_order" "3,2,1,0"

    # Test 5: Subprocess tests
    echo "=============================================================="
    echo "Test 5: Subprocess GPU Tests"
    echo "=============================================================="
    python "$PROJECT_ROOT/scripts/test_multi_gpu_setup.py" \
        --subprocess-tests \
        --output-json "$OUTPUT_DIR/subprocess_results.json" 2>&1

    echo ""

    # Test 6: Quick training test with different GPU configs
    if [ -f "$PROJECT_ROOT/latentwire/train.py" ]; then
        echo "=============================================================="
        echo "Test 6: Quick Training Test with GPU Configurations"
        echo "=============================================================="

        # Single GPU training test
        echo "Testing single GPU training (10 samples)..."
        CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/latentwire/train.py" \
            --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --samples 10 \
            --epochs 1 \
            --batch_size 2 \
            --latent_len 8 \
            --d_z 64 \
            --encoder_type byte \
            --dataset squad \
            --output_dir "$OUTPUT_DIR/single_gpu_train" \
            --no_save_checkpoint 2>&1 | head -50

        echo ""

        # Dual GPU training test (if available)
        if nvidia-smi -L 2>/dev/null | grep -c "GPU" | grep -q "[2-9]"; then
            echo "Testing dual GPU training (10 samples)..."
            CUDA_VISIBLE_DEVICES=0,1 python "$PROJECT_ROOT/latentwire/train.py" \
                --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
                --samples 10 \
                --epochs 1 \
                --batch_size 4 \
                --latent_len 8 \
                --d_z 64 \
                --encoder_type byte \
                --dataset squad \
                --output_dir "$OUTPUT_DIR/dual_gpu_train" \
                --no_save_checkpoint 2>&1 | head -50
        fi
    fi

    # Test 7: DDP test (if multiple GPUs available)
    if command -v torchrun &> /dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        if [ "$GPU_COUNT" -gt "1" ]; then
            echo "=============================================================="
            echo "Test 7: DistributedDataParallel (DDP) Test"
            echo "=============================================================="
            echo "Running DDP test with $GPU_COUNT GPUs..."

            torchrun --nproc_per_node=$GPU_COUNT \
                --master_port=29500 \
                "$PROJECT_ROOT/scripts/test_multi_gpu_setup.py" \
                --test-ddp-worker 2>&1 | tee -a "$LOG_FILE"
        else
            echo "Skipping DDP test (requires 2+ GPUs, found $GPU_COUNT)"
        fi
    else
        echo "Skipping DDP test (torchrun not available)"
    fi

    echo ""
    echo "=============================================================="
    echo "Test Summary"
    echo "=============================================================="

    # Generate summary
    python -c "
import json
import os
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results_files = list(output_dir.glob('*_results.json'))

print(f'Found {len(results_files)} test result files')
print()

# Aggregate results
all_configs = {}
for result_file in results_files:
    with open(result_file) as f:
        data = json.load(f)
        test_name = result_file.stem

        if 'tests' in data and 'cuda_detection' in data['tests']:
            cuda_info = data['tests']['cuda_detection']
            all_configs[test_name] = {
                'device_count': cuda_info.get('device_count', 0),
                'cuda_visible_devices': cuda_info.get('cuda_visible_devices', 'not set')
            }

# Display summary table
if all_configs:
    print('Configuration Summary:')
    print('-' * 60)
    print(f'{'Test Name':<30} {'CUDA_VISIBLE_DEVICES':<20} {'GPUs':<10}')
    print('-' * 60)
    for test_name, config in sorted(all_configs.items()):
        cvd = config['cuda_visible_devices']
        if cvd == 'not set':
            cvd = 'all'
        print(f'{test_name:<30} {cvd:<20} {config['device_count']:<10}')
else:
    print('No test results found')
"

    echo ""
    echo "=============================================================="
    echo "Complete! Results saved to:"
    echo "  - $OUTPUT_DIR/"
    echo "  - $LOG_FILE"
    echo "=============================================================="

} 2>&1 | tee "$LOG_FILE"

# Final summary
echo ""
echo "Testing complete at $(date)"
echo "Check the log file for detailed results: $LOG_FILE"

# Check if any tests failed
if grep -q "Error\|Failed\|Traceback" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "⚠️  Warning: Some tests may have encountered errors."
    echo "Please review the log file for details."
    exit 1
else
    echo "✅ All tests completed successfully!"
    exit 0
fi