#!/usr/bin/env bash
#
# run_with_resume.sh - Unified training launcher with automatic checkpoint resume
#
# This script provides a consistent interface for running experiments with
# automatic checkpoint discovery and resumption. Critical for re-running
# failed experiments on preemptible compute resources.
#
# Usage:
#   ./run_with_resume.sh [experiment_type] [options]
#
# Examples:
#   ./run_with_resume.sh train --dataset squad --epochs 10
#   ./run_with_resume.sh train --resume --checkpoint-dir runs/experiment_123
#   ./run_with_resume.sh main --compression-type telepathy
#
# Features:
# - Automatic checkpoint discovery
# - Preemption-safe training
# - Consistent logging
# - GPU utilization monitoring

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    else
        print_warning "No NVIDIA GPU detected or nvidia-smi not available"
    fi
}

find_latest_checkpoint() {
    local search_dir="${1:-runs}"

    if [ ! -d "$search_dir" ]; then
        return 1
    fi

    # Look for checkpoint directories
    local latest=""

    # Check for step_* directories
    for dir in "$search_dir"/step_* 2>/dev/null; do
        if [ -d "$dir" ] && [ -f "$dir/state.pt" ]; then
            latest="$dir"
        fi
    done

    # Check for epoch* directories
    for dir in "$search_dir"/epoch* 2>/dev/null; do
        if [ -d "$dir" ] && [ -f "$dir/state.pt" ]; then
            latest="$dir"
        fi
    done

    if [ -n "$latest" ]; then
        echo "$latest"
        return 0
    fi

    return 1
}

# ============================================================================
# Training Functions
# ============================================================================

run_train() {
    local RESUME_FLAG=""
    local CHECKPOINT_DIR=""
    local SAVE_DIR="runs/experiment_${TIMESTAMP}"
    local EPOCHS=10
    local DATASET="squad"
    local SAMPLES=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --resume)
                RESUME_FLAG="--auto_resume"
                shift
                ;;
            --resume-from)
                RESUME_FLAG="--resume_from $2"
                shift 2
                ;;
            --checkpoint-dir|--save-dir)
                SAVE_DIR="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --samples)
                SAMPLES="--samples $2"
                shift 2
                ;;
            *)
                # Pass through other arguments
                EXTRA_ARGS="${EXTRA_ARGS} $1"
                shift
                ;;
        esac
    done

    # Auto-discover checkpoint if --resume is set without specific path
    if [ "$RESUME_FLAG" == "--auto_resume" ] && [ -d "$SAVE_DIR" ]; then
        if CKPT=$(find_latest_checkpoint "$SAVE_DIR"); then
            print_info "Found checkpoint: $CKPT"
            RESUME_FLAG="--resume_from $CKPT"
        else
            print_warning "No checkpoint found in $SAVE_DIR, starting fresh"
            RESUME_FLAG=""
        fi
    fi

    # Setup environment
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    export PYTORCH_ENABLE_MPS_FALLBACK=1

    # Create output directory
    mkdir -p "$SAVE_DIR/logs"
    LOG_FILE="$SAVE_DIR/logs/train_${TIMESTAMP}.log"

    print_header "STARTING TRAINING"
    print_info "Dataset: $DATASET"
    print_info "Epochs: $EPOCHS"
    print_info "Save directory: $SAVE_DIR"
    print_info "Log file: $LOG_FILE"

    if [ -n "$RESUME_FLAG" ]; then
        print_info "Resuming from checkpoint"
    fi

    check_gpu

    # Run training with tee for logging
    {
        echo "Command: python latentwire/train.py \\"
        echo "  --dataset $DATASET \\"
        echo "  --epochs $EPOCHS \\"
        echo "  --save_dir $SAVE_DIR \\"
        echo "  --save_every 100 \\"
        echo "  $SAMPLES \\"
        echo "  $RESUME_FLAG \\"
        echo "  $EXTRA_ARGS"
        echo ""
        echo "Starting at: $(date)"
        echo "============================================================"
        echo ""

        python "${PROJECT_ROOT}/latentwire/train.py" \
            --dataset "$DATASET" \
            --epochs "$EPOCHS" \
            --save_dir "$SAVE_DIR" \
            --save_every 100 \
            $SAMPLES \
            $RESUME_FLAG \
            $EXTRA_ARGS

        echo ""
        echo "============================================================"
        echo "Completed at: $(date)"

    } 2>&1 | tee "$LOG_FILE"

    # Check if training was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_info "Training completed successfully"

        # Look for final checkpoint
        if FINAL_CKPT=$(find_latest_checkpoint "$SAVE_DIR"); then
            print_info "Final checkpoint: $FINAL_CKPT"
        fi
    else
        print_error "Training failed"
        print_info "Check log file: $LOG_FILE"
        exit 1
    fi
}

run_main_experiment() {
    local RESUME_FLAG=""
    local OUTPUT_DIR="runs/main_experiment_${TIMESTAMP}"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --resume)
                RESUME_FLAG="--resume"
                shift
                ;;
            --resume-from)
                RESUME_FLAG="--resume-from $2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            *)
                EXTRA_ARGS="${EXTRA_ARGS} $1"
                shift
                ;;
        esac
    done

    # Setup environment
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/main_experiment_${TIMESTAMP}.log"

    print_header "STARTING MAIN EXPERIMENT"
    print_info "Output directory: $OUTPUT_DIR"
    print_info "Log file: $LOG_FILE"

    # Run experiment
    {
        echo "Command: python MAIN_EXPERIMENT.py \\"
        echo "  --output-dir $OUTPUT_DIR \\"
        echo "  $RESUME_FLAG \\"
        echo "  $EXTRA_ARGS"
        echo ""
        echo "Starting at: $(date)"
        echo "============================================================"
        echo ""

        cd "$SCRIPT_DIR"
        python MAIN_EXPERIMENT.py \
            --output-dir "$OUTPUT_DIR" \
            $RESUME_FLAG \
            $EXTRA_ARGS

        echo ""
        echo "============================================================"
        echo "Completed at: $(date)"

    } 2>&1 | tee "$LOG_FILE"
}

# ============================================================================
# Test Functions
# ============================================================================

test_checkpoint_system() {
    print_header "TESTING CHECKPOINT SYSTEM"

    # Test checkpoint manager
    print_info "Testing CheckpointManager..."
    python -c "from checkpoint_manager import CheckpointManager; print('✅ CheckpointManager imports successfully')"

    # Test integration with train.py
    print_info "Testing train.py checkpoint functions..."
    python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from latentwire.train import find_latest_checkpoint, load_checkpoint
print('✅ train.py checkpoint functions available')
"

    # Run quick checkpoint save/load test
    print_info "Running checkpoint save/load test..."
    python "${SCRIPT_DIR}/checkpoint_manager.py"

    print_info "All checkpoint tests passed!"
}

# ============================================================================
# Main Entry Point
# ============================================================================

show_usage() {
    cat << EOF
Usage: $0 [command] [options]

Commands:
    train               Run training with checkpoint support
    main               Run MAIN_EXPERIMENT.py with checkpoint support
    test               Test checkpoint system
    help               Show this help message

Common Options:
    --resume           Resume from latest checkpoint
    --resume-from PATH Resume from specific checkpoint
    --save-dir DIR     Directory for checkpoints
    --epochs N         Number of training epochs
    --dataset NAME     Dataset to use (squad, hotpotqa, etc.)

Examples:
    # Fresh training
    $0 train --dataset squad --epochs 10

    # Resume from latest checkpoint
    $0 train --resume --save-dir runs/experiment_123

    # Resume from specific checkpoint
    $0 train --resume-from runs/experiment_123/step_500

    # Run main experiment with resume
    $0 main --resume --compression-type telepathy

EOF
}

# Parse command
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    train)
        run_train "$@"
        ;;
    main)
        run_main_experiment "$@"
        ;;
    test)
        test_checkpoint_system
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac