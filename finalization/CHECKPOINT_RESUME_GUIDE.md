# Checkpoint Resume Guide

## Executive Summary

All training scripts in the finalization directory now support automatic checkpoint resumption. This is **critical** for re-running failed experiments on preemptible compute resources.

## Key Features

✅ **Automatic Checkpoint Discovery** - Finds latest checkpoint automatically
✅ **Atomic Save Operations** - Prevents corruption during preemption
✅ **State Preservation** - Complete training state saved and restored
✅ **Preemption Handling** - Graceful shutdown with checkpoint save
✅ **Unified Interface** - Consistent --resume flag across all scripts

## Quick Start

### Basic Usage

```bash
# Fresh training (will save checkpoints automatically)
python latentwire/train.py --epochs 10 --save_dir runs/exp1 --save_every 100

# Resume from latest checkpoint in save_dir
python latentwire/train.py --auto_resume --save_dir runs/exp1

# Resume from specific checkpoint
python latentwire/train.py --resume_from runs/exp1/epoch5
```

### Using the Wrapper Script

```bash
# Start new training with automatic checkpointing
./finalization/run_with_resume.sh train --dataset squad --epochs 20

# Resume training from interruption
./finalization/run_with_resume.sh train --resume --save-dir runs/experiment_123

# Run main experiment with resume capability
./finalization/run_with_resume.sh main --resume --compression-type telepathy
```

## Checkpoint Structure

```
checkpoint_dir/
├── state.pt              # Main state (model, optimizer, scheduler)
├── metadata.json         # Checkpoint metadata (step, epoch, timestamp)
├── config.json          # Training configuration
├── encoder.pt           # Model components (if saved separately)
├── adapter_llama.pt
├── adapter_qwen.pt
└── training_stats.json  # Training metrics and statistics
```

## Implementation Details

### 1. CheckpointManager Class

The core checkpoint management system providing:
- Automatic checkpoint discovery
- Atomic file operations
- Old checkpoint cleanup
- Preemption signal handling

```python
from checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    save_dir="./checkpoints",
    save_interval=100,        # Save every 100 steps
    keep_last_n=3,           # Keep only 3 recent checkpoints
    enable_preemption_handling=True
)

# Save checkpoint
state = {'model': model.state_dict(), 'step': 100}
manager.save_checkpoint(state, step=100, epoch=1)

# Load latest checkpoint
loaded_state = manager.load_checkpoint()
```

### 2. Integration with train.py

The main training script has built-in resume support:

```python
# Command line arguments
--auto_resume              # Automatically find and resume from latest
--resume_from PATH         # Resume from specific checkpoint
--save_dir DIR            # Directory for checkpoints
--save_every N            # Save checkpoint every N steps
```

### 3. MAIN_EXPERIMENT.py Integration

```python
# The ExperimentCheckpointer provides high-level interface
from checkpoint_manager import ExperimentCheckpointer

checkpointer = ExperimentCheckpointer(config)

# Save training state
checkpointer.save_training_state(
    model, optimizer, scheduler,
    epoch=5, step=500,
    metrics={'loss': 0.25},
    best_metric=0.25
)

# Resume training
epoch, step, best_metric, metrics = checkpointer.resume_training(
    model, optimizer, scheduler
)
```

## SLURM Integration

For HPC environments with preemptible jobs:

```bash
#!/bin/bash
#SBATCH --signal=TERM@60    # Send SIGTERM 60s before timeout
#SBATCH --requeue          # Auto-requeue on preemption

python latentwire/train.py \
    --auto_resume \
    --save_dir ./checkpoints \
    --save_every 100 \
    [other training args]
```

## Common Scenarios

### 1. Job Preempted During Training

```bash
# Original job
python latentwire/train.py --epochs 50 --save_dir runs/exp1

# Job gets preempted at epoch 23
# Simply rerun with --auto_resume
python latentwire/train.py --auto_resume --save_dir runs/exp1 --epochs 50
# Training continues from epoch 23
```

### 2. Continuing After Error

```bash
# Training failed due to OOM at step 1500
# Fix batch size and resume
python latentwire/train.py \
    --resume_from runs/exp1/step_1500 \
    --batch_size 32  # Reduced from 64
```

### 3. Finding Latest Checkpoint

```bash
# Manually find latest checkpoint
python -c "
from checkpoint_manager import CheckpointManager
m = CheckpointManager(save_dir='runs/exp1')
print(m.find_latest_checkpoint())
"
```

## Testing Checkpoint System

```bash
# Run comprehensive tests
python finalization/test_resume_functionality.py

# Quick test of checkpoint operations
python finalization/checkpoint_manager.py
```

## Troubleshooting

### Issue: "No checkpoint found to resume"
**Solution**: Verify save_dir contains checkpoint directories (epoch*, step_*)

### Issue: "State file not found"
**Solution**: Checkpoint directory exists but state.pt is missing. May indicate interrupted save.

### Issue: Model weights not restored
**Solution**: Ensure model architecture matches checkpoint. Check for model.module (DDP) vs model.

### Issue: Training restarts from epoch 0
**Solution**: Use --auto_resume or --resume_from, not just --resume

## Best Practices

1. **Always use --save_every** for long training runs
   ```bash
   --save_every 500  # Save every 500 steps
   ```

2. **Set appropriate keep_last_n** to manage disk space
   ```python
   keep_last_n=3  # Keep only 3 recent checkpoints
   ```

3. **Use atomic saves** (handled automatically by CheckpointManager)

4. **Test resume before long runs**
   ```bash
   # Quick test with few samples
   python latentwire/train.py --samples 100 --epochs 2 --save_every 50
   python latentwire/train.py --auto_resume
   ```

5. **Monitor checkpoint saves in logs**
   ```
   💾 Saved checkpoint: runs/exp1/step_500
   ```

## API Reference

### CheckpointManager

```python
manager = CheckpointManager(
    save_dir: str,              # Directory for checkpoints
    save_interval: int,         # Steps between saves
    keep_last_n: int,          # Number of checkpoints to keep
    enable_preemption_handling: bool,  # Setup signal handlers
    verbose: bool              # Print detailed logs
)

# Find latest checkpoint
path = manager.find_latest_checkpoint(checkpoint_dir)

# Save checkpoint
path = manager.save_checkpoint(
    state: dict,               # State dictionary
    step: int,                 # Current step
    epoch: int,                # Current epoch
    is_best: bool,            # Mark as best checkpoint
    extra_artifacts: dict      # Additional files to save
)

# Load checkpoint
state = manager.load_checkpoint(
    checkpoint_path: str,      # Path to checkpoint
    map_location: str         # Device mapping
)

# Check if should save
should_save = manager.should_save(step: int, force: bool)
```

### Command Line Arguments

```bash
# train.py arguments
--auto_resume              # Resume from latest in save_dir
--resume_from PATH         # Resume from specific checkpoint
--save_dir DIR            # Directory for checkpoints
--save_every N            # Save every N steps
--no_load_optimizer       # Skip optimizer state on resume
--no_load_lr_scheduler    # Skip scheduler state on resume
--reset_epoch            # Reset epoch/step counters on resume

# MAIN_EXPERIMENT.py arguments
--resume                  # Resume from latest checkpoint
--resume-from PATH        # Resume from specific checkpoint
--save-interval N         # Save every N steps
--no-checkpoints         # Disable checkpoint saving
```

## Verification Checklist

Before running long experiments, verify:

- [ ] Checkpoint saving works: Run short test and check save_dir
- [ ] Resume works: Interrupt and restart with --auto_resume
- [ ] Preemption handling: Send SIGTERM and verify save
- [ ] State preservation: Check model weights, optimizer lr, metrics
- [ ] Disk space: Monitor checkpoint directory size
- [ ] Logs capture: Verify tee logging to file

## Summary

The checkpoint resume system ensures training can recover from:
- SLURM preemption
- Out-of-memory errors
- Network interruptions
- Manual stops (Ctrl+C)
- System crashes

All experiments should use `--save_every` and `--auto_resume` flags to ensure robustness on preemptible compute resources.