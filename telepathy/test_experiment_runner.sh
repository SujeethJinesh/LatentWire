#!/usr/bin/env bash
# =============================================================================
# TEST SCRIPT FOR EXPERIMENT RUNNER
# =============================================================================
# Quick test to verify the experiment runner works before launching full runs
#
# Usage:
#   bash telepathy/test_experiment_runner.sh
# =============================================================================

set -e

echo "=============================================================="
echo "TESTING EXPERIMENT RUNNER"
echo "=============================================================="
echo "This will run a minimal test to verify all components work"
echo "=============================================================="

# Test configuration (minimal for quick validation)
export OUTPUT_DIR="runs/test_experiments"
export TRAIN_SAMPLES=10
export EVAL_SAMPLES=10
export BATCH_SIZE=2
export EPOCHS=1
export DATASETS=("sst2")
export SEEDS=(42)
export EXPERIMENTS=("bridge")

# Clean previous test
rm -rf "$OUTPUT_DIR"

# Run the main script with test config
echo "Running test with minimal configuration..."
bash telepathy/run_experiments.sh

# Check results
if [ -f "${OUTPUT_DIR}/experiment_state.json" ]; then
    echo ""
    echo "✓ State file created successfully"

    python3 -c "
import json
with open('${OUTPUT_DIR}/experiment_state.json', 'r') as f:
    state = json.load(f)
print(f\"✓ Completed: {len(state['completed'])} experiments\")
print(f\"✓ Failed: {len(state['failed'])} experiments\")

if state['completed']:
    print('✓ Test PASSED - experiment runner works correctly')
    exit(0)
else:
    print('✗ Test FAILED - no experiments completed')
    exit(1)
"
else
    echo "✗ State file not found - test FAILED"
    exit 1
fi

echo ""
echo "=============================================================="
echo "TEST COMPLETE"
echo "=============================================================="
echo "The experiment runner is working correctly!"
echo "You can now run the full experiments with:"
echo "  srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=40G --time=04:00:00 --pty bash telepathy/run_experiments.sh"
echo "Or submit as batch job:"
echo "  sbatch telepathy/submit_experiments.slurm"
echo "=============================================================="