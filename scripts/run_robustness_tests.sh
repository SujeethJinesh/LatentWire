#!/usr/bin/env bash
set -e

# =============================================================================
# Quick Robustness Test Runner for Latent Telepathy
# =============================================================================
# This script runs a subset of robustness tests locally for quick validation
# For full tests, use the SLURM script on HPC
# =============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/robustness_local}"
CHECKPOINT="${1:-runs/telepathy_latest/checkpoint.pt}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/robustness_quick_${TIMESTAMP}.log"

echo "=============================================================="
echo "Latent Telepathy Robustness Tests (Quick Mode)"
echo "=============================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please provide a valid checkpoint path as first argument"
    exit 1
fi

# Run quick tests with tee for logging
{
    echo "Starting quick robustness tests..."
    echo "Time: $(date)"
    echo ""

    # Run in fast mode for local testing
    python telepathy/robustness_test_suite.py \
        --checkpoint "$CHECKPOINT" \
        --device cuda \
        --fast

    echo ""
    echo "Generating test category summary..."
    python -c "
import json
from pathlib import Path

report_path = Path('runs/robustness_report.json')
if report_path.exists():
    with open(report_path) as f:
        report = json.load(f)

    # Create a markdown summary
    summary = []
    summary.append('# Robustness Test Results\\n')
    summary.append(f\"**Overall Pass Rate:** {report['overall']['pass_rate']*100:.1f}%\\n\")
    summary.append(f\"**Total Tests:** {report['overall']['total_tests']}\\n\")
    summary.append(f\"**Average Score:** {report['overall']['avg_score']:.3f}\\n\")
    summary.append('\\n## Category Breakdown\\n')

    for cat, stats in report['categories'].items():
        emoji = '✅' if stats['pass_rate'] >= 0.8 else '⚠️' if stats['pass_rate'] >= 0.6 else '❌'
        summary.append(f\"- {emoji} **{cat}:** {stats['pass_rate']*100:.1f}% ({stats['passed']}/{stats['total']})\\n\")

    summary.append('\\n## Key Findings\\n')

    # Identify weak points
    failed_tests = [t for t in report['tests'] if not t['passed']]
    if failed_tests:
        summary.append('### Failed Tests\\n')
        for test in failed_tests[:5]:  # Show top 5 failures
            summary.append(f\"- {test['name']}: Score {test['score']:.3f} (Degradation: {test['degradation']:.1f}%)\\n\")

    # Save summary
    summary_path = Path('$OUTPUT_DIR/robustness_summary.md')
    with open(summary_path, 'w') as f:
        f.writelines(summary)

    print(''.join(summary))
    print(f'\\nSummary saved to: {summary_path}')

    # Copy full report
    import shutil
    shutil.copy(report_path, '$OUTPUT_DIR/robustness_report.json')
else:
    print('ERROR: No report generated!')
"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "Robustness tests complete!"
echo "=============================================================="
echo "Results saved to:"
echo "  - $OUTPUT_DIR/robustness_report.json (full report)"
echo "  - $OUTPUT_DIR/robustness_summary.md (summary)"
echo "  - $LOG_FILE (execution log)"
echo ""
echo "For full testing, run on HPC with:"
echo "  sbatch telepathy/submit_robustness_tests.slurm"