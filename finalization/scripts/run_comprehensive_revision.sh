#!/usr/bin/env bash
# Run comprehensive revision experiments for LatentWire paper
#
# This script orchestrates all experiments needed for the paper revision
# in response to reviewer feedback.
#
# Usage:
#   bash scripts/run_comprehensive_revision.sh           # Run all phases
#   bash scripts/run_comprehensive_revision.sh --phase 1 # Run specific phase
#   bash scripts/run_comprehensive_revision.sh --help    # Show help

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/comprehensive_revision}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_DIR}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/comprehensive_revision_${TIMESTAMP}.log"

# Setup environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Create output directory
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo "COMPREHENSIVE REVISION EXPERIMENTS"
echo "=============================================================="
echo "Output directory: $LOG_DIR"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "=============================================================="
echo ""

# Run the comprehensive orchestrator with logging
{
    python telepathy/run_comprehensive_revision.py \
        --output_dir "$LOG_DIR" \
        "$@"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "EXPERIMENTS COMPLETE"
echo "=============================================================="
echo "End time: $(date)"
echo "Results saved to: $LOG_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Generate summary report
if [ -f "$LOG_DIR/all_results.json" ]; then
    echo "Generating summary report..."
    python -c "
import json
import sys

with open('$LOG_DIR/all_results.json', 'r') as f:
    results = json.load(f)

print('\\n' + '='*70)
print('SUMMARY REPORT')
print('='*70)

# Check each phase
for phase_num in range(1, 7):
    phase_key = str(phase_num)
    if phase_key in results.get('phases', {}):
        phase_data = results['phases'][phase_key]
        if 'error' in phase_data:
            print(f'Phase {phase_num}: FAILED - {phase_data[\"error\"][:50]}...')
        else:
            print(f'Phase {phase_num}: COMPLETED')

            # Print key metrics if available
            if phase_num == 1 and 'aggregated_results' in phase_data:
                for dataset in ['sst2', 'agnews', 'trec']:
                    if dataset in phase_data:
                        bridge_acc = phase_data[dataset].get('bridge', {}).get('accuracy_mean', 'N/A')
                        if bridge_acc != 'N/A':
                            print(f'  {dataset} Bridge: {bridge_acc:.1f}%')
    else:
        print(f'Phase {phase_num}: NOT RUN')

print('='*70)
"
fi

echo "=============================================================="