#!/bin/bash
# Source the main script to get all functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../RUN_ALL.sh"

# Override functions to simulate quick completion
run_training() {
    print_header "TRAINING (SIMULATED)"
    mark_phase_started "training"
    sleep 0.5
    CHECKPOINT_PATH="fake/checkpoint/path"
    save_state "checkpoint_path" "$CHECKPOINT_PATH" "checkpoints"
    mark_phase_completed "training"
    return 0
}

run_phase1_statistical() {
    print_header "PHASE 1 (SIMULATED)"
    mark_phase_started "phase1_statistical"
    sleep 0.5
    mark_phase_completed "phase1_statistical"
    return 0
}

# Set up minimal config
BASE_OUTPUT_DIR="runs/test_tracking_$(date +%s)"
mkdir -p "$BASE_OUTPUT_DIR"
init_state_file

# Run phases
run_training
run_phase1_statistical

# Show final state
print_state_summary
