#!/usr/bin/env bash
# =============================================================================
# EXPERIMENT MONITORING SCRIPT
# =============================================================================
# Use this script to monitor the progress of running experiments
#
# Usage:
#   bash finalization/monitor_experiments.sh [RUN_DIR]
#
# If RUN_DIR is not specified, uses the latest run
# =============================================================================

set -euo pipefail

# Configuration
RUN_DIR="${1:-runs/paper_revision_latest}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-10}"  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo -e "${RED}Error: Run directory not found: $RUN_DIR${NC}"
    echo "Available runs:"
    ls -dt runs/*/ 2>/dev/null | head -10
    exit 1
fi

# Function to check phase completion
check_phase() {
    local phase_num=$1
    local phase_name=$2
    local check_file=$3

    if [ -f "$check_file" ]; then
        echo -e "${GREEN}✓${NC} Phase $phase_num: $phase_name"
        return 0
    else
        echo -e "${YELLOW}⧗${NC} Phase $phase_num: $phase_name"
        return 1
    fi
}

# Function to get latest log lines
show_latest_activity() {
    local log_dir="$RUN_DIR/logs"

    if [ -d "$log_dir" ]; then
        local latest_log=$(ls -t "$log_dir"/*.log 2>/dev/null | head -1)

        if [ -n "$latest_log" ]; then
            echo -e "\n${BLUE}Latest Activity ($(basename $latest_log)):${NC}"
            tail -5 "$latest_log" | sed 's/^/  /'
        fi
    fi
}

# Function to show progress statistics
show_statistics() {
    local results_dir="$RUN_DIR/results"

    if [ -d "$results_dir/phase1_statistical" ]; then
        local completed_evals=$(find "$results_dir/phase1_statistical" -name "*.json" 2>/dev/null | wc -l)
        local expected_evals=$((4 * 3))  # 4 datasets * 3 seeds
        echo "  Statistical Evaluations: $completed_evals / $expected_evals"
    fi

    if [ -f "$RUN_DIR/checkpoint/training_state.json" ]; then
        local epoch=$(python -c "import json; print(json.load(open('$RUN_DIR/checkpoint/training_state.json'))['epoch'])" 2>/dev/null || echo "?")
        echo "  Training Epoch: $epoch / 24"
    fi
}

# Function to estimate time remaining
estimate_time() {
    local start_file="$RUN_DIR/logs/main_experiment_*.log"

    if ls $start_file 1> /dev/null 2>&1; then
        local start_time=$(head -1 $(ls -t $start_file | head -1) | grep -oE '\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]' | head -1 | tr -d '[]')

        if [ -n "$start_time" ]; then
            local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$start_time" +%s 2>/dev/null || echo "0")
            local current_epoch=$(date +%s)
            local elapsed=$((current_epoch - start_epoch))

            if [ $elapsed -gt 0 ]; then
                local hours=$((elapsed / 3600))
                local minutes=$(((elapsed % 3600) / 60))
                echo -e "\n${BLUE}Elapsed Time:${NC} ${hours}h ${minutes}m"
            fi
        fi
    fi
}

# Function to check for errors
check_errors() {
    local error_count=0
    local log_dir="$RUN_DIR/logs"

    if [ -d "$log_dir" ]; then
        error_count=$(grep -i "ERROR\|FAILED\|Exception\|Traceback" "$log_dir"/*.log 2>/dev/null | wc -l)

        if [ $error_count -gt 0 ]; then
            echo -e "\n${RED}⚠ Errors Detected:${NC} $error_count"
            echo "Recent errors:"
            grep -i "ERROR\|FAILED" "$log_dir"/*.log 2>/dev/null | tail -3 | sed 's/^/  /'
        fi
    fi
}

# Main monitoring loop
clear_screen() {
    printf "\033c"
}

monitor_once() {
    clear_screen
    echo "=============================================================="
    echo "LATENTWIRE EXPERIMENT MONITOR"
    echo "=============================================================="
    echo "Run Directory: $RUN_DIR"
    echo "Time: $(date)"
    echo ""

    # Check if experiment is running
    if pgrep -f "run_all_experiments.sh" > /dev/null 2>&1; then
        echo -e "${GREEN}● Experiment Running${NC}"
    else
        echo -e "${YELLOW}○ Experiment Not Running${NC}"
    fi

    echo ""
    echo "Phase Completion Status:"
    echo "------------------------"

    # Check each phase
    check_phase 0 "Training/Checkpoint" "$RUN_DIR/checkpoint/encoder.pt"
    check_phase 1 "Statistical Rigor" "$RUN_DIR/results/phase1_statistical/statistical_summary.json"
    check_phase 2 "Linear Probe" "$RUN_DIR/results/phase2_linear_probe/comparison_report.json"
    check_phase 3 "Baselines" "$RUN_DIR/results/phase3_baselines/baseline_comparison.json"
    check_phase 4 "Efficiency" "$RUN_DIR/results/phase4_efficiency/efficiency_summary.json"

    echo ""
    echo "Progress Details:"
    echo "----------------"
    show_statistics

    estimate_time
    show_latest_activity
    check_errors

    # Show final results if available
    if [ -f "$RUN_DIR/final_results.json" ]; then
        echo -e "\n${GREEN}✓ EXPERIMENTS COMPLETE${NC}"
        echo "Final results available at: $RUN_DIR/final_results.json"
        echo "Paper assets at: $RUN_DIR/paper_assets/"
    fi

    echo ""
    echo "=============================================================="
    echo "Commands:"
    echo "  View logs:     tail -f $RUN_DIR/logs/main_experiment_*.log"
    echo "  Check SLURM:   squeue -u \$USER"
    echo "  Stop monitor:  Ctrl+C"
}

# Handle continuous monitoring or single run
if [ "${MONITOR_MODE:-continuous}" = "once" ]; then
    monitor_once
else
    echo "Starting continuous monitoring (refresh every ${REFRESH_INTERVAL}s)..."
    echo "Press Ctrl+C to stop"

    while true; do
        monitor_once
        sleep $REFRESH_INTERVAL
    done
fi