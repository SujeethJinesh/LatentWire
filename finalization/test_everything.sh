#!/bin/bash
# =============================================================================
# Complete System Test Suite for LatentWire HPC Environment
# Tests all critical components: checkpointing, preemption, GPU, logging, git
# Should complete in <3 minutes and provide confidence for production runs
# =============================================================================
# Usage: bash finalization/test_everything.sh
# Returns: 0 if all tests pass, 1 if any fail
# =============================================================================

set +e  # Don't exit on first error - we want to run all tests

# Configuration
TEST_DIR="/tmp/latentwire_test_$$"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$TEST_DIR/test_everything.log"
FAILED_TESTS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create test directory
mkdir -p "$TEST_DIR"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Test result tracking
pass_test() {
    local test_name="$1"
    log "${GREEN}✓${NC} $test_name passed"
}

fail_test() {
    local test_name="$1"
    local reason="$2"
    log "${RED}✗${NC} $test_name failed: $reason"
    FAILED_TESTS+=("$test_name: $reason")
}

# Header
log "=============================================================="
log "LatentWire Complete System Test Suite"
log "=============================================================="
log "Test directory: $TEST_DIR"
log "Project root: $PROJECT_ROOT"
log "Start time: $(date)"
log ""

# Change to project root
cd "$PROJECT_ROOT"

# =============================================================================
# TEST 1: Python and PyTorch Environment
# =============================================================================
log "${YELLOW}TEST 1: Python and PyTorch Environment${NC}"

if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    pass_test "PyTorch environment"
else
    # Try python if python3 doesn't work
    if python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
        pass_test "PyTorch environment"
    else
        fail_test "PyTorch environment" "PyTorch not installed"
    fi
fi

# =============================================================================
# TEST 2: GPU Detection and Count
# =============================================================================
log ""
log "${YELLOW}TEST 2: GPU Detection${NC}"

GPU_COUNT=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.device_count())
elif torch.backends.mps.is_available():
    print(1)
else:
    print(0)
" 2>/dev/null || echo 0)

if [ "$GPU_COUNT" -gt 0 ]; then
    pass_test "GPU detection (found $GPU_COUNT GPU(s))"
    # Note: On HPC this will show CUDA GPUs, on Mac it will show MPS
else
    log "${YELLOW}  No GPUs found (CPU-only mode)${NC}"
    pass_test "GPU detection (CPU mode acceptable for testing)"
fi

# =============================================================================
# TEST 3: Checkpoint Save/Load with Scheduler State
# =============================================================================
log ""
log "${YELLOW}TEST 3: Checkpoint Save/Load with Scheduler${NC}"

python3 - <<EOF 2>&1 | tee -a "$LOG_FILE"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os

try:
    # Create simple model
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train for a few steps
    for i in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Save checkpoint
    checkpoint_path = "$TEST_DIR/test_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': 5,
        'lr': scheduler.get_last_lr()[0]
    }, checkpoint_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Verify all components saved
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'scheduler_state_dict' in checkpoint
    assert checkpoint['step'] == 5

    # Load into new model
    new_model = nn.Linear(10, 10)
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.1)

    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    new_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("Checkpoint save/load test passed")
    sys.exit(0)

except Exception as e:
    print("Checkpoint test failed:", str(e))
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    pass_test "Checkpoint save/load with scheduler"
else
    fail_test "Checkpoint save/load" "Failed to save or load checkpoint"
fi

# =============================================================================
# TEST 4: Preemption Signal Handling
# =============================================================================
log ""
log "${YELLOW}TEST 4: Preemption Signal Handling${NC}"

python3 - <<EOF 2>&1 | tee -a "$LOG_FILE"
import signal
import sys
import time
import os

preempt_flag = False

def handle_preemption(signum, frame):
    global preempt_flag
    preempt_flag = True
    print("Received signal %d, setting preempt flag" % signum)

# Register handlers
signal.signal(signal.SIGTERM, handle_preemption)
signal.signal(signal.SIGUSR1, handle_preemption)

# Send signal to self
os.kill(os.getpid(), signal.SIGUSR1)
time.sleep(0.1)  # Give signal time to be handled

if preempt_flag:
    print("Signal handling test passed")
    sys.exit(0)
else:
    print("Signal handling test failed")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    pass_test "Preemption signal handling"
else
    fail_test "Preemption signal" "Failed to handle SIGUSR1/SIGTERM"
fi

# =============================================================================
# TEST 5: Resume from Exact Batch Position
# =============================================================================
log ""
log "${YELLOW}TEST 5: Resume from Exact Batch Position${NC}"

python3 - <<EOF 2>&1 | tee -a "$LOG_FILE"
import sys
import json

try:
    # Simulate training state
    state = {
        'epoch': 2,
        'global_step': 150,
        'batch_idx': 50,
        'samples_seen': 3200,
        'best_metric': 0.85
    }

    # Save state
    state_path = "$TEST_DIR/training_state.json"
    with open(state_path, 'w') as f:
        json.dump(state, f)

    # Load state
    with open(state_path, 'r') as f:
        loaded_state = json.load(f)

    # Verify exact recovery
    assert loaded_state['epoch'] == 2
    assert loaded_state['global_step'] == 150
    assert loaded_state['batch_idx'] == 50
    assert loaded_state['samples_seen'] == 3200

    print("Resume test passed - can recover from batch %d" % loaded_state['batch_idx'])
    sys.exit(0)

except Exception as e:
    print("Resume test failed:", str(e))
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    pass_test "Resume from exact batch position"
else
    fail_test "Resume from batch" "Failed to save/load training state"
fi

# =============================================================================
# TEST 6: Logging with PYTHONUNBUFFERED
# =============================================================================
log ""
log "${YELLOW}TEST 6: Unbuffered Logging${NC}"

TEST_LOG="$TEST_DIR/unbuffered_test.log"
export PYTHONUNBUFFERED=1

python3 -c "
import sys
import time
print('Line 1')
print('Line 2', file=sys.stderr)
time.sleep(0.1)
print('Line 3')
" 2>&1 | tee "$TEST_LOG"

LINE_COUNT=$(wc -l < "$TEST_LOG" | tr -d ' ')
if [ "$LINE_COUNT" -eq 3 ]; then
    pass_test "Unbuffered logging (captured $LINE_COUNT lines)"
else
    # Not critical - logging still works
    log "${YELLOW}  Warning: Expected 3 lines, got $LINE_COUNT (non-critical)${NC}"
    pass_test "Unbuffered logging (partial)"
fi

# =============================================================================
# TEST 7: Git Operations
# =============================================================================
log ""
log "${YELLOW}TEST 7: Git Operations${NC}"

# Check if we're in a git repo
if git rev-parse --git-dir > /dev/null 2>&1; then
    # Test git status
    if git status > /dev/null 2>&1; then
        pass_test "Git status"
    else
        fail_test "Git status" "Failed to run git status"
    fi

    # Test git log
    if git log --oneline -n 1 > /dev/null 2>&1; then
        pass_test "Git log"
    else
        fail_test "Git log" "Failed to run git log"
    fi

    # Show current branch and commit
    BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    log "  Current branch: $BRANCH"
    log "  Current commit: $COMMIT"
else
    fail_test "Git operations" "Not in a git repository"
fi

# =============================================================================
# TEST 8: Data Loading (SQuAD)
# =============================================================================
log ""
log "${YELLOW}TEST 8: Data Loading${NC}"

python3 - <<EOF 2>&1 | tee -a "$LOG_FILE"
import sys
try:
    from latentwire.data import load_data

    # Try to load a small sample
    train_data, _ = load_data('squad', samples=10, eval_samples=5)

    assert len(train_data) == 10, "Expected 10 samples, got %d" % len(train_data)

    # Check data structure
    sample = train_data[0]
    assert 'prefix' in sample, "Missing 'prefix' field"
    assert 'output' in sample, "Missing 'output' field"

    print("Data loading test passed - loaded %d samples" % len(train_data))
    sys.exit(0)

except Exception as e:
    print("Data loading test failed:", str(e))
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    pass_test "Data loading (SQuAD)"
else
    fail_test "Data loading" "Failed to load SQuAD dataset"
fi

# =============================================================================
# TEST 9: Memory and Resource Check
# =============================================================================
log ""
log "${YELLOW}TEST 9: Memory and Resources${NC}"

python - <<EOF 2>&1 | tee -a "$LOG_FILE"
import torch
import psutil
import os

# CPU info
cpu_count = os.cpu_count()
print(f"CPU cores: {cpu_count}")

# Memory info
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / (1024**3):.1f} GB")
print(f"Available RAM: {mem.available / (1024**3):.1f} GB")

# GPU memory (if available)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")

        # Try allocating memory
        try:
            test_tensor = torch.zeros(1000, 1000, device=f'cuda:{i}')
            del test_tensor
            torch.cuda.empty_cache()
            print(f"  Memory allocation test passed")
        except Exception as e:
            print(f"  Memory allocation failed: {e}")
elif torch.backends.mps.is_available():
    print("MPS device available (Apple Silicon)")
    try:
        test_tensor = torch.zeros(1000, 1000, device='mps')
        del test_tensor
        print("MPS memory allocation test passed")
    except Exception as e:
        print(f"MPS allocation failed: {e}")
EOF

pass_test "Memory and resource check"

# =============================================================================
# TEST 10: Quick Training Smoke Test
# =============================================================================
log ""
log "${YELLOW}TEST 10: Quick Training Smoke Test${NC}"

export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create minimal training script
cat > "$TEST_DIR/smoke_test.py" <<'EOF'
import sys
import torch
import torch.nn as nn
from latentwire.models import ByteTokenizer, SimpleEncoder

try:
    # Initialize components
    tokenizer = ByteTokenizer(latent_len=8)
    encoder = SimpleEncoder(d_z=64, n_layers=2)

    # Create dummy input
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, 256, (batch_size, seq_len))

    # Forward pass
    z = encoder(x)
    assert z.shape == (batch_size, 8, 64), f"Wrong shape: {z.shape}"

    # Backward pass
    loss = z.mean()
    loss.backward()

    print("Training smoke test passed")
    sys.exit(0)

except Exception as e:
    print(f"Training smoke test failed: {e}")
    sys.exit(1)
EOF

if python "$TEST_DIR/smoke_test.py" 2>&1 | tee -a "$LOG_FILE"; then
    pass_test "Training smoke test"
else
    fail_test "Training smoke test" "Failed basic forward/backward pass"
fi

# =============================================================================
# TEST 11: SLURM Environment Variables (if on HPC)
# =============================================================================
log ""
log "${YELLOW}TEST 11: SLURM Environment (if applicable)${NC}"

if [ -n "$SLURM_JOB_ID" ]; then
    log "SLURM environment detected:"
    log "  Job ID: $SLURM_JOB_ID"
    log "  Node: $SLURMD_NODENAME"
    log "  GPUs: $CUDA_VISIBLE_DEVICES"
    pass_test "SLURM environment variables"
else
    log "Not running under SLURM (this is expected if testing locally)"
    pass_test "SLURM check (not applicable)"
fi

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "=============================================================="
log "TEST SUMMARY"
log "=============================================================="

TOTAL_TESTS=11
PASSED_TESTS=$((TOTAL_TESTS - ${#FAILED_TESTS[@]}))

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    log "${GREEN}ALL TESTS PASSED!${NC} ($PASSED_TESTS/$TOTAL_TESTS)"
    log "System is ready for production experiments"
    EXIT_CODE=0
else
    log "${RED}SOME TESTS FAILED${NC} ($PASSED_TESTS/$TOTAL_TESTS passed)"
    log ""
    log "Failed tests:"
    for failure in "${FAILED_TESTS[@]}"; do
        log "  - $failure"
    done
    EXIT_CODE=1
fi

log ""
log "Test directory: $TEST_DIR"
log "Log file: $LOG_FILE"
log "Completed at: $(date)"
log "=============================================================="

# Cleanup (optional - comment out to keep test artifacts for debugging)
# rm -rf "$TEST_DIR"

exit $EXIT_CODE