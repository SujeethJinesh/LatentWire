#!/bin/bash
source $(dirname "$0")/../RUN_ALL.sh

# Override run_training to simulate failure
run_training() {
    print_header "TRAINING (SIMULATED FAILURE)"

    CURRENT_PHASE="training"
    mark_phase_started "training"

    print_error "Simulating training failure..."
    exit 42  # Specific exit code to verify
}

# Run with our override
main
