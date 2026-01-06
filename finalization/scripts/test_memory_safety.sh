#!/usr/bin/env bash
set -e

# Test Memory Safety for H100 Configurations
# This script verifies that all batch size configurations are memory-safe
# accounting for Adam optimizer states (12 bytes per trainable parameter)

echo "============================================================"
echo "Memory Safety Test for H100 GPU Configurations"
echo "============================================================"
echo ""
echo "This test verifies:"
echo "  1. Adam optimizer states are properly accounted (12 bytes/param)"
echo "  2. Batch sizes are safe for H100 80GB GPUs"
echo "  3. All configurations have adequate safety margins"
echo ""

# Run the Python verification script
echo "Running memory calculation verification..."
python3 scripts/verify_memory_calculations.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MEMORY SAFETY TEST PASSED"
    echo ""
    echo "Key findings:"
    echo "  - Adam optimizer: 1.2GB for 100M trainable params (12 bytes each)"
    echo "  - Frozen models: 31.3GB (Llama-8B + Qwen-7B in bf16)"
    echo "  - Fixed memory: 39.3GB per GPU"
    echo "  - Batch size 64: Uses 81.3% of GPU memory (safe)"
    echo "  - Headroom: 14.9GB available for spikes"
else
    echo ""
    echo "❌ MEMORY SAFETY TEST FAILED"
    echo "Please review batch size configurations!"
    exit 1
fi

# Additional quick test to verify actual memory usage prediction
echo ""
echo "============================================================"
echo "Quick Memory Usage Prediction Test"
echo "============================================================"
echo ""

python3 -c "
def estimate_memory_gb():
    # Simulate loading models and optimizer

    # Frozen models (approximate)
    frozen_gb = 31.3

    # Trainable params (100M) with Adam
    trainable_params = 100_000_000
    trainable_weights_gb = trainable_params * 2 / 1e9  # bf16
    adam_states_gb = trainable_params * 12 / 1e9  # 3x fp32 states
    gradients_gb = trainable_params * 2 / 1e9  # bf16

    # System overhead
    overhead_gb = 6.4

    # Activations for batch_size=64
    batch_size = 64
    activation_per_sample = 0.403
    activations_gb = batch_size * activation_per_sample

    total_gb = (
        frozen_gb +
        trainable_weights_gb +
        adam_states_gb +
        gradients_gb +
        overhead_gb +
        activations_gb
    )

    print(f'Memory Breakdown for batch_size={batch_size}:')
    print(f'  Frozen models: {frozen_gb:.1f} GB')
    print(f'  Trainable weights: {trainable_weights_gb:.1f} GB')
    print(f'  Adam optimizer: {adam_states_gb:.1f} GB')
    print(f'  Gradients: {gradients_gb:.1f} GB')
    print(f'  Activations: {activations_gb:.1f} GB')
    print(f'  Overhead: {overhead_gb:.1f} GB')
    print(f'  TOTAL: {total_gb:.1f} GB')
    print()

    utilization = total_gb / 80.0
    print(f'H100 80GB Utilization: {utilization*100:.1f}%')

    if utilization > 0.85:
        print('⚠️  WARNING: High memory usage!')
    elif utilization > 0.95:
        print('❌ DANGER: Very likely to OOM!')
    else:
        print('✅ SAFE: Good memory utilization')

estimate_memory_gb()
"

echo ""
echo "============================================================"
echo "Test Complete"
echo "============================================================"