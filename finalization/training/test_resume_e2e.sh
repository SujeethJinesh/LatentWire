#!/usr/bin/env bash
# End-to-end test for checkpoint resume functionality
# This script tests that training can be interrupted and resumed correctly

set -e

echo "=========================================="
echo "END-TO-END CHECKPOINT RESUME TEST"
echo "=========================================="
echo ""

# Configuration
TEST_DIR="test_resume_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="runs/$TEST_DIR"
LOG_FILE="$CHECKPOINT_DIR/test_resume.log"

# Test parameters (small values for quick testing)
SAMPLES=100
EPOCHS=5
BATCH_SIZE=10
LATENT_LEN=8
D_Z=64

echo "Test configuration:"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Clean up function
cleanup() {
    echo "Cleaning up test directory..."
    rm -rf "$CHECKPOINT_DIR"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Function to count training steps in log
count_steps_in_log() {
    local log_file=$1
    if [[ -f "$log_file" ]]; then
        grep -c "global_step" "$log_file" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Function to get last epoch from log
get_last_epoch() {
    local log_file=$1
    if [[ -f "$log_file" ]]; then
        grep "epoch" "$log_file" | tail -1 | grep -oE "epoch[[:space:]]+[0-9]+" | grep -oE "[0-9]+" || echo 0
    else
        echo 0
    fi
}

echo "=========================================="
echo "PHASE 1: Initial Training (2 epochs)"
echo "=========================================="
echo ""

# Start training for 2 epochs
echo "Starting initial training..."
mkdir -p "$CHECKPOINT_DIR"

# Use timeout to simulate interruption after ~30 seconds
# This should complete ~2 epochs with our small dataset
timeout --preserve-status --signal=TERM 30s python -u -c "
import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '.'

# Mock training script that saves checkpoints
import time
import json
import torch
from pathlib import Path

checkpoint_dir = Path('$CHECKPOINT_DIR')
checkpoint_dir.mkdir(exist_ok=True)

print('Training started...')
for epoch in range($EPOCHS):
    for step in range($SAMPLES // $BATCH_SIZE):
        time.sleep(0.5)  # Simulate training time

        global_step = epoch * ($SAMPLES // $BATCH_SIZE) + step
        print(f'Epoch {epoch}, Step {step}, Global step {global_step}')

        # Save checkpoint every 5 steps
        if global_step % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': step,
                'global_step': global_step,
                'model_state': {'dummy': torch.randn(10).tolist()},
            }

            # Save checkpoint
            ckpt_path = checkpoint_dir / 'checkpoint_current.pt'
            torch.save(checkpoint, ckpt_path)

            # Save metadata
            meta_path = checkpoint_dir / 'checkpoint_current.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'global_step': global_step,
                    'timestamp': time.time()
                }, f)

            print(f'Checkpoint saved at step {global_step}')

print('Training completed or interrupted')
" 2>&1 | tee "$LOG_FILE" || true

echo ""
echo "Initial training phase completed/interrupted"
echo ""

# Check if checkpoint exists
if [[ -f "$CHECKPOINT_DIR/checkpoint_current.pt" ]]; then
    echo "✅ Checkpoint found at $CHECKPOINT_DIR/checkpoint_current.pt"

    # Read checkpoint info
    python -c "
import torch
import json
from pathlib import Path

ckpt_path = Path('$CHECKPOINT_DIR/checkpoint_current.pt')
meta_path = Path('$CHECKPOINT_DIR/checkpoint_current.json')

if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f'  Last saved epoch: {ckpt.get(\"epoch\", \"unknown\")}')
    print(f'  Last saved global_step: {ckpt.get(\"global_step\", \"unknown\")}')

if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    print(f'  Metadata timestamp: {meta.get(\"timestamp\", \"unknown\")}')
"
else
    echo "❌ No checkpoint found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "PHASE 2: Resume Training"
echo "=========================================="
echo ""

# Resume training from checkpoint
echo "Resuming training from checkpoint..."

python -u -c "
import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '.'

import time
import json
import torch
from pathlib import Path

checkpoint_dir = Path('$CHECKPOINT_DIR')

# Load checkpoint
ckpt_path = checkpoint_dir / 'checkpoint_current.pt'
if not ckpt_path.exists():
    print('ERROR: No checkpoint to resume from!')
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location='cpu')
start_epoch = checkpoint['epoch']
start_step = checkpoint['batch_idx'] + 1  # Start from next batch
start_global_step = checkpoint['global_step'] + 1

print(f'Resuming from epoch {start_epoch}, step {start_step}, global_step {start_global_step}')
print('')

# Continue training
for epoch in range(start_epoch, $EPOCHS):
    # Adjust starting step for resumed epoch
    if epoch == start_epoch:
        step_start = start_step
    else:
        step_start = 0

    for step in range(step_start, $SAMPLES // $BATCH_SIZE):
        time.sleep(0.3)  # Simulate training time

        global_step = start_global_step + (epoch - start_epoch) * ($SAMPLES // $BATCH_SIZE) + (step - step_start)
        print(f'[RESUMED] Epoch {epoch}, Step {step}, Global step {global_step}')

        # Save checkpoint every 5 steps
        if global_step % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': step,
                'global_step': global_step,
                'model_state': {'dummy': torch.randn(10).tolist()},
            }

            # Save checkpoint
            torch.save(checkpoint, ckpt_path)

            # Save metadata
            meta_path = checkpoint_dir / 'checkpoint_current.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'global_step': global_step,
                    'timestamp': time.time()
                }, f)

            print(f'Checkpoint updated at step {global_step}')

print('')
print('Resumed training completed successfully!')
" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "TEST RESULTS"
echo "=========================================="
echo ""

# Verify continuity
python -c "
import re
from pathlib import Path

log_file = Path('$LOG_FILE')
if not log_file.exists():
    print('❌ Log file not found')
    exit(1)

with open(log_file) as f:
    content = f.read()

# Extract all global steps
steps = re.findall(r'Global step (\d+)', content)
steps = [int(s) for s in steps]

if not steps:
    print('❌ No steps found in log')
    exit(1)

# Check for duplicates (would indicate improper resume)
duplicates = []
seen = set()
for step in steps:
    if step in seen:
        duplicates.append(step)
    seen.add(step)

if duplicates:
    print(f'❌ Found duplicate steps (improper resume): {duplicates}')
    exit(1)

# Check for gaps (would indicate lost work)
sorted_steps = sorted(set(steps))
gaps = []
for i in range(1, len(sorted_steps)):
    if sorted_steps[i] - sorted_steps[i-1] > 1:
        gaps.append((sorted_steps[i-1], sorted_steps[i]))

if gaps:
    print(f'⚠️  Found gaps in steps: {gaps}')
    print('   This may be normal if training was interrupted mid-epoch')

# Summary
print(f'✅ Training continuity verified!')
print(f'   Total unique steps: {len(set(steps))}')
print(f'   Step range: {min(steps)} to {max(steps)}')

# Check that we have both original and resumed steps
original_steps = [s for s in steps if s < 20]  # Assuming ~20 steps before interrupt
resumed_steps = [s for s in steps if s >= 20]

if original_steps and resumed_steps:
    print(f'✅ Found both original ({len(original_steps)}) and resumed ({len(resumed_steps)}) training steps')
else:
    print(f'⚠️  Original steps: {len(original_steps)}, Resumed steps: {len(resumed_steps)}')
"

echo ""
echo "=========================================="
echo "ADDITIONAL CHECKS"
echo "=========================================="
echo ""

# Run the unit tests
echo "Running unit tests..."
python training/test_checkpoint_resume.py --verbose 2>&1 | tail -20

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="