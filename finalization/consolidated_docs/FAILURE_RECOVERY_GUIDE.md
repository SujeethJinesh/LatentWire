# Failure Recovery Guide for RUN_ALL.sh

## Overview

RUN_ALL.sh now includes a robust failure recovery mechanism that allows you to resume experiments from where they failed, saving significant compute time by not re-running completed phases.

## How It Works

1. **State Tracking**: The script automatically creates a `.experiment_state` file that tracks:
   - Which phases have completed successfully
   - Which phase failed (if any)
   - Checkpoint paths to avoid re-training
   - Timestamp and metadata

2. **Automatic Recovery**: When a phase fails:
   - The current state is saved
   - Clear recovery instructions are displayed
   - You can resume from the exact point of failure

## Usage

### Starting a New Experiment

```bash
# Normal execution - state tracking happens automatically
bash RUN_ALL.sh experiment

# Run specific phases
bash RUN_ALL.sh experiment --phase 1
```

### Resuming After Failure

When an experiment fails, you'll see instructions like:

```
FAILURE RECOVERY INSTRUCTIONS
========================================
The experiment has failed but state has been saved.

To resume from where it failed, run:
  bash RUN_ALL.sh experiment --resume runs/latentwire_unified_20240115_143022/.experiment_state

This will:
  1. Skip already completed phases
  2. Re-attempt the failed phase
  3. Continue with remaining phases
```

Simply run the displayed command to resume:

```bash
bash RUN_ALL.sh experiment --resume runs/latentwire_unified_20240115_143022/.experiment_state
```

### Checking State

You can inspect the current state at any time:

```bash
# View state file
cat runs/latentwire_unified_*/.experiment_state

# Check which phases completed
grep "=completed" runs/latentwire_unified_*/.experiment_state

# Check for failures
grep "=failed" runs/latentwire_unified_*/.experiment_state
```

## State File Format

The state file uses an INI-like format:

```ini
[metadata]
version=3.1.0
start_time=1736201422
base_output_dir=runs/latentwire_unified_20240115_143022

[phases]
training=completed
phase1_statistical=completed
phase2_linear_probe=in_progress
phase3_baselines=pending
phase4_efficiency=pending
aggregation=pending

[checkpoints]
checkpoint_path=runs/latentwire_unified_20240115_143022/checkpoint/epoch23
final_epoch=23

[failures]
failure_count=1
last_failure=phase2_linear_probe
last_failure_time=Mon Jan 15 14:45:22 PST 2024
```

## Advanced Usage

### Manually Editing State

If needed, you can manually edit the state file to force re-running a phase:

```bash
# Edit state file
vi runs/latentwire_unified_*/.experiment_state

# Change a completed phase to pending to re-run it
# Change: phase1_statistical=completed
# To:     phase1_statistical=pending
```

### Skipping Failed Phases

To skip a consistently failing phase and continue:

```bash
# Edit state file and mark failed phase as completed
vi runs/latentwire_unified_*/.experiment_state

# Change: phase3_baselines=failed
# To:     phase3_baselines=completed
```

### Starting Fresh from Checkpoint

If training succeeded but you want to re-run evaluation:

```bash
# Use existing checkpoint without training
bash RUN_ALL.sh experiment --checkpoint runs/latentwire_unified_*/checkpoint/epoch23
```

## Phase Dependencies

The phases run in this order:

1. **training**: Train the LatentWire model
2. **phase1_statistical**: Multiple seeds, bootstrap confidence intervals
3. **phase2_linear_probe**: Linear probe baseline comparisons
4. **phase3_baselines**: LLMLingua and token-budget baselines
5. **phase4_efficiency**: Latency, memory, throughput measurements
6. **aggregation**: Combine all results and generate tables

If training fails, no other phases will run. If any evaluation phase fails, subsequent phases can still run (they're independent).

## Common Scenarios

### Scenario 1: OOM During Training

```bash
# Reduce batch size and resume
export BATCH_SIZE=4
bash RUN_ALL.sh experiment --resume runs/latentwire_unified_*/.experiment_state
```

### Scenario 2: Phase 3 Timeout on HPC

```bash
# Resume will skip completed phases 1-2 and retry phase 3
bash RUN_ALL.sh experiment --resume runs/latentwire_unified_*/.experiment_state
```

### Scenario 3: Want to Re-run with Different Seeds

```bash
# Edit state to mark phase1 as pending
vi runs/latentwire_unified_*/.experiment_state

# Change seeds and resume
export SEEDS="42 123 456 789 1337"
bash RUN_ALL.sh experiment --resume runs/latentwire_unified_*/.experiment_state
```

## Testing the Recovery Mechanism

A test script is provided to verify the recovery system works:

```bash
bash test_recovery.sh
```

This will:
- Simulate failures at different phases
- Verify state files are created correctly
- Test the resume functionality
- Validate state management functions

## Troubleshooting

### State File Not Created

The state file is created when the output directory is initialized. If it's missing:

```bash
# Manually create state file
mkdir -p runs/test_recovery
touch runs/test_recovery/.experiment_state
```

### Resume Not Working

Check that:
1. The state file path is correct
2. The state file is readable
3. The base_output_dir in the state file exists

### Phases Running Out of Order

The script enforces phase dependencies. If phases seem to run incorrectly:

```bash
# Check state file for inconsistencies
cat runs/latentwire_unified_*/.experiment_state

# Reset specific phase
vi runs/latentwire_unified_*/.experiment_state
```

## Best Practices

1. **Always use resume after failures** - Don't restart from scratch
2. **Check logs before resuming** - Understand why it failed
3. **Save state files** - They're small and valuable for debugging
4. **Use --dry-run to test** - Verify what will run before executing

## Implementation Details

The recovery system uses:
- Bash trap ERR for error handling
- State file with INI format for persistence
- Phase tracking with pending/in_progress/completed/failed states
- Checkpoint path saving to avoid re-training
- Failure counting and error message logging

This ensures robust recovery even from unexpected failures like OOM, timeouts, or cluster preemption.