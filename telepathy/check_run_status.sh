#!/bin/bash
# Quick script to check status of evaluation runs

echo "==================================================================="
echo "Checking Telepathy Evaluation Run Status"
echo "==================================================================="

RUNS_DIR="/projects/m000066/sujinesh/LatentWire/runs"

# Find all paper_results directories
echo "Found run directories:"
for dir in $RUNS_DIR/paper_results_*/run_*; do
    if [ -d "$dir" ]; then
        RUN_NAME=$(basename $(dirname $dir))/$(basename $dir)

        # Check for complete_results.json
        if [ -f "$dir/complete_results.json" ]; then
            # Count experiments in complete_results.json
            NUM_EXPERIMENTS=$(python3 -c "import json; d=json.load(open('$dir/complete_results.json')); print(sum(len(v) for k,v in d.items() if isinstance(v, list) and k != 'failed_experiments'))" 2>/dev/null || echo "?")
            echo "  ✓ $RUN_NAME - COMPLETE ($NUM_EXPERIMENTS experiments)"
        else
            # Count individual result files
            NUM_RESULTS=$(find $dir -name "*.json" | wc -l)
            echo "  ⚠ $RUN_NAME - INCOMPLETE ($NUM_RESULTS result files)"
        fi

        # Check for Linear Probe specifically
        if [ -d "$dir/linear_probe" ]; then
            LP_COUNT=$(ls $dir/linear_probe/*.json 2>/dev/null | wc -l)
            if [ $LP_COUNT -gt 0 ]; then
                echo "      └─ Linear Probe: $LP_COUNT results"
            fi
        fi
    fi
done

echo ""
echo "Most recent complete run:"
LATEST_COMPLETE=$(find $RUNS_DIR -name "complete_results.json" -type f -exec ls -t {} \; | head -1)
if [ -n "$LATEST_COMPLETE" ]; then
    LATEST_DIR=$(dirname $LATEST_COMPLETE)
    echo "  $LATEST_DIR"

    # Show key metrics if available
    if [ -f "$LATEST_DIR/paper_tables_comprehensive.tex" ]; then
        echo ""
        echo "  Key results from paper_tables.tex:"
        grep "Bridge (32 tokens)" "$LATEST_DIR/paper_tables_comprehensive.tex" | head -1 || true
    fi
else
    echo "  No complete runs found"
fi

echo ""
echo "==================================================================="
echo "To resume most recent incomplete run, use:"
echo "  sbatch telepathy/submit_complete_evaluation.slurm"
echo "(The script will automatically detect and resume)"
echo "==================================================================="