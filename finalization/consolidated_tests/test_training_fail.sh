#!/bin/bash
# This will create the output directory and state file, then fail
BASE_OUTPUT_DIR="runs/test_recovery_training_fail"
mkdir -p "$BASE_OUTPUT_DIR"

# Run with dry-run first to create state, then force a failure
bash RUN_ALL.sh quick --dry-run --no-interactive

# Now simulate a failure by exiting with error
echo "SIMULATED FAILURE IN TRAINING"
exit 1
