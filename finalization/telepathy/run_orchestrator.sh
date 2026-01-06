#!/bin/bash
# Quick-start script for the preemptible experiment orchestrator
# This script provides common usage patterns for running experiments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="${OUTPUT_BASE:-runs/orchestrated}"

# Set up environment
export PYTHONPATH="$PROJECT_DIR"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Parse command line arguments
COMMAND="${1:-help}"
shift || true

# Helper functions
print_header() {
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
}

print_usage() {
    cat << EOF
Preemptible Orchestrator Quick-Start Script
===========================================

Usage: bash telepathy/run_orchestrator.sh [COMMAND] [OPTIONS]

Commands:
  test        Run test suite to validate setup
  single      Run a single experiment type
  all         Run all experiments
  resume      Resume from previous run
  monitor     Monitor running experiments
  report      Generate report from results
  help        Show this help message

Examples:
  # Test the setup
  bash telepathy/run_orchestrator.sh test

  # Run SST-2 experiments only
  bash telepathy/run_orchestrator.sh single sst2

  # Run all experiments
  bash telepathy/run_orchestrator.sh all

  # Resume from checkpoint
  bash telepathy/run_orchestrator.sh resume runs/orchestrated

  # Monitor running experiments
  bash telepathy/run_orchestrator.sh monitor

  # Generate report
  bash telepathy/run_orchestrator.sh report runs/orchestrated

For HPC submission:
  sbatch telepathy/submit_preemptible_orchestrator.slurm
EOF
}

# Command implementations
run_test() {
    print_header "Running Orchestrator Test Suite"
    python telepathy/test_orchestrator.py
}

run_single() {
    local EXPERIMENT="${1:-sst2}"
    local OUTPUT_DIR="${OUTPUT_BASE}/single_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"

    print_header "Running Single Experiment: ${EXPERIMENT}"
    echo "Output directory: ${OUTPUT_DIR}"

    mkdir -p "${OUTPUT_DIR}/logs"
    LOG_FILE="${OUTPUT_DIR}/logs/orchestrator.log"

    {
        python telepathy/run_preemptible_experiments.py \
            --experiment "${EXPERIMENT}" \
            --output_dir "${OUTPUT_DIR}" \
            "$@"
    } 2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Experiment complete! Results saved to:"
    echo "  ${OUTPUT_DIR}"
}

run_all() {
    local OUTPUT_DIR="${OUTPUT_BASE}/all_$(date +%Y%m%d_%H%M%S)"

    print_header "Running All Experiments"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "WARNING: This will run many experiments and may take a long time!"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    mkdir -p "${OUTPUT_DIR}/logs"
    LOG_FILE="${OUTPUT_DIR}/logs/orchestrator.log"

    {
        python telepathy/run_preemptible_experiments.py \
            --experiment all \
            --output_dir "${OUTPUT_DIR}" \
            "$@"
    } 2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "All experiments complete! Results saved to:"
    echo "  ${OUTPUT_DIR}"
}

run_resume() {
    local CHECKPOINT_DIR="${1:-${OUTPUT_BASE}}"

    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "Error: Checkpoint directory not found: ${CHECKPOINT_DIR}"
        exit 1
    fi

    print_header "Resuming from Checkpoint"
    echo "Checkpoint directory: ${CHECKPOINT_DIR}"

    LOG_FILE="${CHECKPOINT_DIR}/logs/orchestrator_resume_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/run_preemptible_experiments.py \
            --resume \
            --output_dir "${CHECKPOINT_DIR}" \
            "$@"
    } 2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Resume complete! Results saved to:"
    echo "  ${CHECKPOINT_DIR}"
}

monitor_experiments() {
    local LOG_DIR="${1:-${OUTPUT_BASE}/*/logs}"

    print_header "Monitoring Experiments"

    # Find latest log files
    LATEST_LOGS=$(find ${LOG_DIR} -name "orchestrator*.log" -type f 2>/dev/null | \
                  xargs ls -t 2>/dev/null | head -5)

    if [ -z "$LATEST_LOGS" ]; then
        echo "No orchestrator logs found in ${LOG_DIR}"
        exit 1
    fi

    echo "Latest logs:"
    echo "$LATEST_LOGS" | nl

    echo ""
    echo "Tailing latest log..."
    echo "Press Ctrl+C to stop"
    echo ""

    LATEST=$(echo "$LATEST_LOGS" | head -1)
    tail -f "$LATEST"
}

generate_report() {
    local RESULTS_DIR="${1:-${OUTPUT_BASE}}"

    print_header "Generating Report"
    echo "Results directory: ${RESULTS_DIR}"

    python -c "
import sys
import json
from pathlib import Path
from datetime import datetime

results_dir = Path('${RESULTS_DIR}')

# Find all result files
result_files = list(results_dir.rglob('*_results.json'))

if not result_files:
    print('No result files found!')
    sys.exit(1)

print(f'Found {len(result_files)} result files')

# Generate markdown report
report = []
report.append('# Experiment Results Report')
report.append(f'Generated: {datetime.now().isoformat()}')
report.append('')

for result_file in sorted(result_files):
    with open(result_file) as f:
        data = json.load(f)

    exp_name = result_file.stem.replace('_results', '')
    report.append(f'## {exp_name}')
    report.append('')

    if 'metrics' in data:
        for metric, value in data['metrics'].items():
            report.append(f'- {metric}: {value}')
    report.append('')

# Save report
report_path = results_dir / 'consolidated_report.md'
with open(report_path, 'w') as f:
    f.write('\n'.join(report))

print(f'Report saved to: {report_path}')
"

    echo ""
    echo "Report generation complete!"
}

# Main script logic
cd "$PROJECT_DIR"

case "$COMMAND" in
    test)
        run_test
        ;;
    single)
        run_single "$@"
        ;;
    all)
        run_all "$@"
        ;;
    resume)
        run_resume "$@"
        ;;
    monitor)
        monitor_experiments "$@"
        ;;
    report)
        generate_report "$@"
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        print_usage
        exit 1
        ;;
esac