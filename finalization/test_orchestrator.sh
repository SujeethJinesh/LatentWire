#!/usr/bin/env bash
# =============================================================================
# Test Script for Experiment Orchestrator
# =============================================================================
# Quick test to verify the orchestrator works correctly.
# Runs a minimal experiment with small parameters.
#
# Usage:
#   bash finalization/test_orchestrator.sh
# =============================================================================

set -e

# Critical environment variable
export PYTHONUNBUFFERED=1  # Immediate output flushing

echo "=============================================================="
echo "TESTING EXPERIMENT ORCHESTRATOR"
echo "=============================================================="
echo ""

# Set test parameters (very small for quick testing)
export EXP_NAME="test_orchestrator_$(date +%Y%m%d_%H%M%S)"
export DATASET="squad"
export SAMPLES="100"  # Very small
export EPOCHS="1"     # Single epoch
export LATENT_LEN="16"  # Smaller latent
export D_Z="128"      # Smaller dimension
export MAX_RETRIES="2"  # Fewer retries for testing

# Override work directory for local testing if not on HPC
if [[ ! -d "/projects/m000066" ]]; then
    export WORK_DIR="$(pwd)"
    echo "Local testing mode - using current directory as WORK_DIR"
fi

echo "Test Configuration:"
echo "  Experiment name: $EXP_NAME"
echo "  Dataset: $DATASET"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Work directory: ${WORK_DIR:-$(pwd)}"
echo ""

# Test 1: Check if orchestrator script exists and is executable
echo "Test 1: Checking orchestrator script..."
if [[ -f "finalization/run_experiment.sh" ]]; then
    echo "✓ Orchestrator script exists"
else
    echo "✗ Orchestrator script not found!"
    exit 1
fi

# Test 2: Dry run to check syntax and initial setup
echo ""
echo "Test 2: Dry run for syntax checking..."
if bash -n finalization/run_experiment.sh; then
    echo "✓ Script syntax is valid"
else
    echo "✗ Script has syntax errors!"
    exit 1
fi

# Test 3: Test GPU detection (non-blocking)
echo ""
echo "Test 3: Testing GPU detection..."
python -c "
import torch
if torch.cuda.is_available():
    count = torch.cuda.device_count()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory // (1024**3)}GB)')
    print(f'✓ Detected {count} GPU(s)')
else:
    print('✓ No GPUs detected (CPU mode)')
"

# Test 4: Test checkpoint directory creation and state management
echo ""
echo "Test 4: Testing state management..."
TEST_CHECKPOINT_DIR="runs/${EXP_NAME}"
mkdir -p "$TEST_CHECKPOINT_DIR"

# Create a mock state file
cat > "$TEST_CHECKPOINT_DIR/.orchestrator_state" << EOF
{
    "experiment": "$EXP_NAME",
    "status": "testing",
    "epoch": 0,
    "checkpoint": "",
    "timestamp": "$(date -Iseconds)",
    "slurm_job_id": "",
    "retry_count": 0
}
EOF

if [[ -f "$TEST_CHECKPOINT_DIR/.orchestrator_state" ]]; then
    echo "✓ State file created successfully"
    cat "$TEST_CHECKPOINT_DIR/.orchestrator_state" | python -m json.tool > /dev/null 2>&1 && \
        echo "✓ State file is valid JSON" || echo "✗ State file is not valid JSON"
else
    echo "✗ Failed to create state file"
fi

# Test 5: Test signal handling
echo ""
echo "Test 5: Testing signal handling..."
bash -c '
    trap "echo Signal received && exit 0" TERM INT
    echo "Waiting for signal (5 seconds)..."
    sleep 5 &
    PID=$!
    sleep 1
    kill -TERM $PID 2>/dev/null || true
    wait $PID 2>/dev/null
' && echo "✓ Signal handling works" || echo "✗ Signal handling failed"

# Test 6: Quick training test (optional - requires GPUs)
echo ""
echo "Test 6: Quick training test"
echo "Do you want to run a quick training test? (y/N): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Running minimal training test..."
    echo "(This will take a few minutes)"

    # Run the orchestrator with test parameters
    bash finalization/run_experiment.sh

    # Check if checkpoint was created
    if ls "runs/${EXP_NAME}/epoch"* 1> /dev/null 2>&1; then
        echo "✓ Training created checkpoint successfully"
    else
        echo "✗ No checkpoint created"
    fi
else
    echo "Skipping training test"
fi

# Clean up test files (optional)
echo ""
echo "Test files created in: runs/${EXP_NAME}"
echo "Do you want to clean up test files? (y/N): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    rm -rf "runs/${EXP_NAME}"
    echo "✓ Test files cleaned up"
else
    echo "Test files preserved in: runs/${EXP_NAME}"
fi

echo ""
echo "=============================================================="
echo "ORCHESTRATOR TESTS COMPLETE"
echo "=============================================================="
echo ""
echo "To run a full experiment on HPC:"
echo "  1. Push code to git: git add -A && git commit -m 'Add orchestrator' && git push"
echo "  2. On HPC: cd /projects/m000066/sujinesh/LatentWire && git pull"
echo "  3. Submit job: sbatch finalization/submit_experiment.slurm"
echo "  4. Monitor: squeue -u \$USER"
echo ""
echo "To run interactively on HPC:"
echo "  srun --gpus=4 --account=marlowe-m000066 --partition=preempt --time=01:00:00 bash finalization/run_experiment.sh"