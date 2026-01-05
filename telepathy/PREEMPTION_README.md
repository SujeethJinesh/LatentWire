# Preemption-Safe Training for LatentWire

This directory contains infrastructure for handling SLURM preemption gracefully, ensuring training can be interrupted and resumed without losing progress.

## Overview

The preemption system provides:
- **Signal handling** for SIGTERM (preemption warning)
- **Immediate checkpoint saving** within grace period
- **Automatic resumption** from exact training state
- **Periodic checkpointing** at configurable intervals
- **Atomic writes** to prevent corruption
- **Mid-batch recovery** with exact RNG state preservation

## Components

### 1. `preemptible_training.py`
Main training wrapper that orchestrates preemption-safe training:
- Monitors for preemption signals
- Manages checkpoint saving/loading
- Handles training subprocess
- Coordinates with SLURM for requeuing

### 2. `training_signals.py`
Signal handling utilities for integration with existing training scripts:
- Signal handler installation
- Checkpoint trigger management
- Exit coordination

### 3. `submit_preemptible.slurm`
SLURM submission script with preemption support:
- Configures signal timing (`--signal=TERM@120`)
- Enables automatic requeuing (`--requeue`)
- Manages git synchronization

### 4. `test_preemption.py`
Test script for local validation of preemption handling

## Usage

### Basic Usage

Submit a preemption-safe training job:

```bash
# On HPC
cd /projects/m000066/sujinesh/LatentWire
git pull
sbatch telepathy/submit_preemptible.slurm

# Monitor job
squeue -u $USER
tail -f runs/preemptible_exp/state.pt

# Job will automatically requeue if preempted
```

### Custom Configuration

Modify checkpoint interval and grace period:

```bash
python telepathy/preemptible_training.py \
    --checkpoint_interval 600 \   # Save every 10 minutes
    --grace_period 180 \          # 3 minutes to save on preemption
    --auto_resume \
    --save_dir runs/my_experiment \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 10000 \
    --epochs 10
```

### Integration with Existing Scripts

To add preemption support to existing training scripts:

```python
# In your training script
from telepathy.training_signals import (
    install_signal_handlers,
    should_save_checkpoint,
    mark_checkpoint_saved,
    should_exit_training
)

# At start of training
install_signal_handlers()

# In training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # ... training code ...

        # Check for checkpoint save
        if should_save_checkpoint(interval=300):  # 5 minutes
            save_checkpoint(...)
            mark_checkpoint_saved()

            # Exit if preempted
            if should_exit_training():
                sys.exit(99)  # Special code for preemption
```

## SLURM Configuration

Key SLURM directives for preemption support:

```bash
#SBATCH --signal=TERM@120    # Send SIGTERM 120 seconds before job ends
#SBATCH --requeue           # Allow automatic requeuing
#SBATCH --open-mode=append  # Append to logs on resume
```

## Signal Handling

The system responds to three signals:

| Signal | Trigger | Action |
|--------|---------|--------|
| SIGTERM | SLURM preemption warning | Save checkpoint, exit for requeue |
| SIGUSR1 | Manual save request | Save checkpoint, continue training |
| SIGINT | Ctrl+C or user interrupt | Save checkpoint, exit cleanly |

## Checkpoint Structure

Checkpoints include:
- Model weights (`encoder.pt`, `adapter_*.pt`)
- Optimizer state (`optimizer.pt`)
- Training state (`state.pt`):
  - Current epoch and global step
  - Batch index for mid-epoch resume
  - RNG states (CPU, CUDA)
  - Best loss tracked
  - Timestamp and metadata

## Testing Locally

Test the preemption system without SLURM:

```bash
# Run test script
python telepathy/test_preemption.py --duration 60 --interval 10

# In another terminal, find PID and send signals
ps aux | grep test_preemption
kill -TERM <pid>  # Test preemption
kill -USR1 <pid>  # Test manual save
```

## Recovery Scenarios

### Scenario 1: Clean Preemption
1. Job receives SIGTERM with 120s warning
2. Checkpoint saved within grace period
3. Job exits with code 99
4. SLURM requeues job
5. Job resumes from checkpoint

### Scenario 2: Mid-Batch Interruption
1. Training interrupted during batch 150/200
2. State saved with `batch_idx=150`
3. On resume, training skips to batch 150
4. RNG states restored for determinism

### Scenario 3: Emergency Save
1. Unexpected error occurs
2. Emergency checkpoint attempted
3. Saved to `emergency_state.pt`
4. Can manually resume from emergency checkpoint

## Best Practices

1. **Set appropriate checkpoint intervals**
   - Balance between overhead and safety
   - Typically 5-10 minutes for long runs

2. **Use sufficient grace period**
   - Allow time for model serialization
   - 120 seconds usually sufficient for large models

3. **Monitor checkpoint sizes**
   - Ensure disk space available
   - Use atomic writes to prevent corruption

4. **Test before production**
   - Validate signal handling locally
   - Verify checkpoint loading works

5. **Git synchronization**
   - Always pull before starting
   - Commit checkpoints periodically
   - Push results on completion

## Troubleshooting

### Job not requeuing
- Check `--requeue` directive is set
- Verify exit code is appropriate (99 for preemption)
- Check SLURM logs for errors

### Checkpoint corruption
- Atomic writes prevent most corruption
- Check for `.tmp` files indicating incomplete writes
- Use backup checkpoints if available

### Signal not received
- Verify `--signal=TERM@<seconds>` is set
- Check grace period is sufficient
- Monitor process with `strace -p <pid>` if needed

### Resume fails
- Check checkpoint compatibility
- Verify all required files present
- Review state.pt for corruption

## Performance Considerations

- Checkpoint saving adds ~5-30s overhead depending on model size
- Atomic writes use temporary disk space
- Frequent checkpoints may impact training throughput
- Balance safety vs performance based on job priority

## Future Enhancements

Potential improvements:
- Differential checkpointing (save only changes)
- Compression for checkpoint files
- Multi-node checkpoint coordination
- Checkpoint validation and integrity checks
- Automatic checkpoint cleanup policies