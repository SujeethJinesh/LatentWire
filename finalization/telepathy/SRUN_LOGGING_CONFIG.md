# SRUN Logging Configuration Guide

## Complete Logging Setup for Interactive SRUN Commands

When using `srun` for interactive GPU jobs, proper logging configuration is critical to ensure all output is captured, especially during preemption or errors.

## Standard SRUN Command with Complete Logging

### Basic Interactive Command
```bash
srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=40G --time=04:00:00 --pty bash
```

### With Full Logging (Recommended)
```bash
# Create log directory
mkdir -p /projects/m000066/sujinesh/LatentWire/runs/srun_logs

# Run with output redirection to both terminal and file
srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=40G --time=04:00:00 \
    --output=/projects/m000066/sujinesh/LatentWire/runs/srun_logs/srun_%j_%t.log \
    --error=/projects/m000066/sujinesh/LatentWire/runs/srun_logs/srun_%j_%t.err \
    --pty bash -c '
        # Set up environment for unbuffered output
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=.

        # Create session log
        LOG_FILE="/projects/m000066/sujinesh/LatentWire/runs/srun_session_$(date +%Y%m%d_%H%M%S).log"

        # Start logging session
        exec > >(tee -a "$LOG_FILE")
        exec 2>&1

        echo "SRUN Session Started: $(date)"
        echo "Job ID: $SLURM_JOB_ID"
        echo "Node: $SLURMD_NODENAME"
        echo "GPUs: $CUDA_VISIBLE_DEVICES"
        echo "Log file: $LOG_FILE"
        echo "======================================"

        # Start interactive bash
        bash
    '
```

## Python Script Execution with Complete Logging

### Method 1: Direct Script Execution with Logging
```bash
srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=40G --time=04:00:00 \
    --output=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.log \
    --error=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.err \
    bash -c '
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=.
        cd /projects/m000066/sujinesh/LatentWire

        # Run with stdbuf to disable buffering
        stdbuf -o0 -e0 python latentwire/train.py \
            --samples 1000 \
            --epochs 1 \
            --output_dir runs/test_run
    '
```

### Method 2: Wrapper Script with Tee
```bash
# Create a wrapper script
cat > /projects/m000066/sujinesh/LatentWire/run_with_logging.sh << 'EOF'
#!/bin/bash
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

LOG_DIR="/projects/m000066/sujinesh/LatentWire/runs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/experiment_${TIMESTAMP}.log"

echo "Starting experiment at $(date)" | tee "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"

# Run command with tee to capture everything
{
    python latentwire/train.py "$@"
} 2>&1 | tee -a "$LOG_FILE"

echo "Experiment complete at $(date)" | tee -a "$LOG_FILE"
EOF

chmod +x /projects/m000066/sujinesh/LatentWire/run_with_logging.sh

# Execute with srun
srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=40G --time=04:00:00 \
    /projects/m000066/sujinesh/LatentWire/run_with_logging.sh \
    --samples 1000 --epochs 1 --output_dir runs/test_run
```

## Environment Variables for Complete Logging

### Essential Variables
```bash
export PYTHONUNBUFFERED=1           # Disable Python output buffering
export PYTHONPATH=.                 # Set Python path
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For compatibility
export CUDA_LAUNCH_BLOCKING=1       # For debugging CUDA errors (optional)
```

### Python-specific Buffering Control
```python
# In Python scripts, add at the beginning:
import sys
sys.stdout = sys.__stdout__  # Reset stdout to unbuffered
sys.stderr = sys.__stderr__  # Reset stderr to unbuffered

# Or use explicit flushing:
print("Important message", flush=True)

# For logging module:
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log')
    ],
    force=True  # Override any existing configuration
)
```

## Handling TQDM Progress Bars

TQDM progress bars can cause issues with logging. Configure properly:

```python
from tqdm import tqdm
import sys

# Force tqdm to use ASCII and write to stderr
tqdm_kwargs = {
    'file': sys.stderr,
    'ascii': True,
    'ncols': 80,
    'leave': True
}

# Use in loops
for item in tqdm(items, **tqdm_kwargs):
    process(item)
```

## Verifying Logging is Working

### Test Script
Create `test_logging.py`:
```python
#!/usr/bin/env python3
import sys
import time
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

print("Test 1: Regular print", flush=True)
logging.info("Test 2: Logging info")
sys.stdout.write("Test 3: Direct stdout write\n")
sys.stdout.flush()

print("\nTest 4: Progress bar:", flush=True)
for i in tqdm(range(5), desc="Processing", ascii=True):
    time.sleep(1)
    if i == 2:
        print(f"  Mid-progress message at i={i}", flush=True)

print("\nTest 5: Error simulation:", flush=True)
sys.stderr.write("This is an error message\n")
sys.stderr.flush()

print("\nAll tests complete!", flush=True)
```

### Run Test
```bash
srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=4G --time=00:05:00 \
    --output=test_%j.log --error=test_%j.err \
    bash -c 'export PYTHONUNBUFFERED=1; python test_logging.py'

# Check logs
cat test_*.log test_*.err
```

## Monitoring and Recovery

### Real-time Monitoring
```bash
# From another terminal
tail -f /projects/m000066/sujinesh/LatentWire/runs/experiment_*.log

# Or use watch for status
watch -n 1 'tail -20 /projects/m000066/sujinesh/LatentWire/runs/experiment_*.log'
```

### Handling Preemption
SLURM automatically saves output/error to specified files even on preemption:
- Output is flushed periodically (every few KB)
- Final flush happens on SIGTERM (preemption signal)
- Use `PYTHONUNBUFFERED=1` to ensure immediate Python output

### Log Recovery After Crash
```bash
# Find all logs from failed jobs
find /projects/m000066/sujinesh/LatentWire/runs -name "*.log" -mtime -1 | \
    xargs grep -l "ERROR\|CRITICAL\|Traceback"

# Get last N lines before crash
tail -n 100 /projects/m000066/sujinesh/LatentWire/runs/experiment_JOBID.log
```

## Best Practices Checklist

✅ **Always set `PYTHONUNBUFFERED=1`**
✅ **Use `--output` and `--error` flags with srun**
✅ **Create log directory before running**
✅ **Use `tee` for dual output (terminal + file)**
✅ **Add `flush=True` to critical Python print statements**
✅ **Use `stdbuf -o0 -e0` for complete unbuffering**
✅ **Configure logging module properly in Python**
✅ **Handle tqdm progress bars with proper file parameter**
✅ **Test logging setup with small test script first**
✅ **Monitor logs from separate terminal during execution**

## Common Issues and Solutions

### Issue: No output until job completes
**Solution**: Set `PYTHONUNBUFFERED=1` and use `flush=True`

### Issue: Progress bars garble log output
**Solution**: Configure tqdm to use stderr and ASCII mode

### Issue: Logs lost on preemption
**Solution**: Use `--output` flag with srun, not just shell redirection

### Issue: Python logging not appearing
**Solution**: Use `force=True` in `basicConfig` and explicit handlers

### Issue: Mixed stdout/stderr order
**Solution**: Redirect stderr to stdout with `2>&1`

## Example Full Command for Production

```bash
# Complete production command with all logging safeguards
srun --account=marlowe-m000066 --partition=preempt \
    --gpus=4 --mem=256G --time=12:00:00 \
    --job-name=experiment \
    --output=/projects/m000066/sujinesh/LatentWire/runs/exp_%j_%t.log \
    --error=/projects/m000066/sujinesh/LatentWire/runs/exp_%j_%t.err \
    bash -c '
        set -e
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=.
        export PYTORCH_ENABLE_MPS_FALLBACK=1

        cd /projects/m000066/sujinesh/LatentWire

        # Additional session log with tee
        SESSION_LOG="runs/session_$(date +%Y%m%d_%H%M%S).log"

        {
            echo "===== SRUN Session Start ====="
            echo "Time: $(date)"
            echo "Job ID: $SLURM_JOB_ID"
            echo "Node: $SLURMD_NODENAME"
            echo "GPUs: $CUDA_VISIBLE_DEVICES"
            echo "============================="

            stdbuf -o0 -e0 python latentwire/train.py \
                --samples 87599 \
                --epochs 24 \
                --output_dir runs/production_run

            echo "===== SRUN Session End ====="
            echo "Time: $(date)"
        } 2>&1 | tee "$SESSION_LOG"
    '
```

This configuration ensures complete logging capture even during preemption or errors.