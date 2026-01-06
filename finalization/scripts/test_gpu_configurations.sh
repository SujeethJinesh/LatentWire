#!/usr/bin/env bash
# =============================================================================
# GPU Detection and Batch Size Configuration Test Suite
# =============================================================================
# This script tests GPU detection and batch size adjustments with different
# CUDA_VISIBLE_DEVICES configurations to ensure scripts properly adapt.
#
# Usage:
#   bash scripts/test_gpu_configurations.sh
#
# Tests:
#   1. No GPUs (CPU-only)
#   2. Single GPU
#   3. Two GPUs
#   4. Four GPUs
#   5. Specific GPU subset
# =============================================================================

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/gpu_detection_tests}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs_${TIMESTAMP}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directories
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo "GPU Detection and Configuration Test Suite"
echo "=============================================================="
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $LOG_DIR"
echo ""

# Function to run test with specific GPU configuration
run_gpu_test() {
    local test_name="$1"
    local cuda_devices="$2"
    local description="$3"

    echo "--------------------------------------------------------------"
    echo "Test: $test_name"
    echo "Description: $description"
    echo "CUDA_VISIBLE_DEVICES: ${cuda_devices:-'(not set)'}"
    echo "--------------------------------------------------------------"

    # Set up log file
    LOG_FILE="$LOG_DIR/${test_name}_test.log"

    # Run the test
    {
        echo "Running GPU detection test: $test_name"
        echo "CUDA_VISIBLE_DEVICES=$cuda_devices"
        echo ""

        if [ -z "$cuda_devices" ]; then
            # Unset for CPU-only test
            unset CUDA_VISIBLE_DEVICES
        else
            export CUDA_VISIBLE_DEVICES="$cuda_devices"
        fi

        python test_gpu_detection.py

        # Save the JSON output with test name
        if [ -f "gpu_detection_test_results.json" ]; then
            mv gpu_detection_test_results.json "$LOG_DIR/${test_name}_results.json"
        fi

    } 2>&1 | tee "$LOG_FILE"

    echo ""
}

# =============================================================================
# Run Tests with Different GPU Configurations
# =============================================================================

echo "Starting GPU configuration tests..."
echo ""

# Test 1: CPU-only (no GPUs)
run_gpu_test "cpu_only" "" "CPU-only mode, no GPUs visible"

# Test 2: Single GPU (GPU 0)
run_gpu_test "single_gpu" "0" "Single GPU mode (GPU 0)"

# Test 3: Two GPUs
run_gpu_test "two_gpus" "0,1" "Two GPU mode (GPUs 0,1)"

# Test 4: Four GPUs (full configuration)
run_gpu_test "four_gpus" "0,1,2,3" "Four GPU mode (all GPUs)"

# Test 5: Specific GPU subset (GPUs 2,3)
run_gpu_test "gpu_subset" "2,3" "Specific GPU subset (GPUs 2,3)"

# Test 6: Invalid GPU ID (should fall back to CPU)
run_gpu_test "invalid_gpu" "99" "Invalid GPU ID (should use CPU)"

# =============================================================================
# Test Batch Size Scaling
# =============================================================================

echo "=============================================================="
echo "Testing Batch Size Scaling Logic"
echo "=============================================================="

{
    python -c "
import os
import json

# Test configurations
configs = [
    {'gpus': 0, 'base_bs': 8, 'expected_strategy': 'reduced'},
    {'gpus': 1, 'base_bs': 8, 'expected_strategy': 'base'},
    {'gpus': 2, 'base_bs': 8, 'expected_strategy': 'scaled'},
    {'gpus': 4, 'base_bs': 8, 'expected_strategy': 'scaled'},
]

print('Batch Size Scaling Tests:')
print('-' * 40)

for config in configs:
    n_gpus = config['gpus']
    base_bs = config['base_bs']

    # Calculate expected batch sizes
    if n_gpus == 0:
        # CPU: reduce batch size
        expected_bs = max(1, base_bs // 4)
        actual_strategy = 'reduced'
    elif n_gpus == 1:
        # Single GPU: use base
        expected_bs = base_bs
        actual_strategy = 'base'
    else:
        # Multi-GPU: scale up
        expected_bs = base_bs * n_gpus
        actual_strategy = 'scaled'

    print(f'GPUs: {n_gpus}, Base BS: {base_bs} -> Expected BS: {expected_bs} ({actual_strategy})')

print()
print('Batch size scaling formula:')
print('  CPU (0 GPUs): base_bs // 4')
print('  1 GPU: base_bs')
print('  N GPUs: base_bs * N')
"
} 2>&1 | tee "$LOG_DIR/batch_size_scaling_test.log"

echo ""

# =============================================================================
# Verify Script Compatibility
# =============================================================================

echo "=============================================================="
echo "Checking Script GPU Compatibility"
echo "=============================================================="

# List of scripts to check for GPU handling
SCRIPTS_TO_CHECK=(
    "latentwire/eval_agnews.py"
    "latentwire/eval_sst2.py"
    "latentwire/gsm8k_eval.py"
    "telepathy/eval_telepathy_trec.py"
)

{
    echo "Checking scripts for GPU handling..."
    echo ""

    for script in "${SCRIPTS_TO_CHECK[@]}"; do
        if [ -f "$script" ]; then
            echo "Checking: $script"

            # Check for GPU-related patterns
            has_cuda_check=$(grep -c "torch.cuda.is_available\|cuda.device_count" "$script" || echo "0")
            has_batch_adjust=$(grep -c "batch_size.*gpu\|batch_size.*device" "$script" || echo "0")
            has_device_setting=$(grep -c "device.*=.*cuda\|to(device)" "$script" || echo "0")

            if [ "$has_cuda_check" -gt 0 ]; then
                echo "  ✓ Has CUDA availability check"
            else
                echo "  ✗ Missing CUDA availability check"
            fi

            if [ "$has_batch_adjust" -gt 0 ]; then
                echo "  ✓ Has batch size adjustment logic"
            else
                echo "  ✗ Missing batch size adjustment"
            fi

            if [ "$has_device_setting" -gt 0 ]; then
                echo "  ✓ Has device configuration"
            else
                echo "  ✗ Missing device configuration"
            fi

            echo ""
        else
            echo "Script not found: $script"
            echo ""
        fi
    done
} 2>&1 | tee "$LOG_DIR/script_compatibility_check.log"

# =============================================================================
# Generate Summary Report
# =============================================================================

echo "=============================================================="
echo "Generating Summary Report"
echo "=============================================================="

SUMMARY_FILE="$LOG_DIR/summary_report.txt"

{
    echo "GPU Detection and Configuration Test Summary"
    echo "============================================="
    echo "Timestamp: $TIMESTAMP"
    echo ""

    echo "Test Results:"
    echo "-------------"

    # Check each test result
    for test_file in "$LOG_DIR"/*_results.json; do
        if [ -f "$test_file" ]; then
            test_name=$(basename "$test_file" _results.json)

            # Extract key information using Python
            python -c "
import json
with open('$test_file', 'r') as f:
    data = json.load(f)
    gpu_info = data.get('gpu_info', {})
    print(f'$test_name:')
    print(f'  CUDA Available: {gpu_info.get(\"cuda_available\", False)}')
    print(f'  Device Count: {gpu_info.get(\"device_count\", 0)}')
    print(f'  Visible Devices: {gpu_info.get(\"visible_devices\", \"Not set\")}')

    # Show recommended batch size from first test
    if data.get('batch_size_tests'):
        first_test = data['batch_size_tests'][0]
        print(f'  Recommended Batch Size: {first_test.get(\"calculated_batch_size\", \"N/A\")}')
    print()
" || echo "  Error reading $test_file"
        fi
    done

    echo "Configuration Recommendations:"
    echo "------------------------------"
    echo "1. Always check torch.cuda.is_available() before using CUDA"
    echo "2. Adjust batch sizes based on torch.cuda.device_count()"
    echo "3. Use gradient accumulation when batch size < desired effective batch size"
    echo "4. Set per_device_batch_size = total_batch_size // n_gpus for DataParallel"
    echo "5. Handle CPU fallback gracefully when no GPUs available"
    echo ""

    echo "Log files saved in: $LOG_DIR"

} | tee "$SUMMARY_FILE"

echo ""
echo "=============================================================="
echo "Test Suite Complete!"
echo "=============================================================="
echo "Results saved to: $LOG_DIR"
echo "Summary report: $SUMMARY_FILE"
echo ""

# Show key findings
echo "Key Findings:"
echo "-------------"

# Count result files
n_tests=$(ls -1 "$LOG_DIR"/*_results.json 2>/dev/null | wc -l)
echo "- Completed $n_tests GPU configuration tests"

# Check if any tests showed GPUs
if grep -q '"cuda_available": true' "$LOG_DIR"/*_results.json 2>/dev/null; then
    echo "- GPUs detected in some configurations"

    # Get max GPU count
    max_gpus=$(grep -h '"device_count":' "$LOG_DIR"/*_results.json 2>/dev/null |
               sed 's/.*"device_count": *\([0-9]*\).*/\1/' |
               sort -n | tail -1)
    echo "- Maximum GPUs detected: $max_gpus"
else
    echo "- No GPUs detected (CPU-only environment)"
fi

echo ""
echo "To test with specific scripts, run:"
echo "  python test_gpu_detection.py script1.py script2.py ..."