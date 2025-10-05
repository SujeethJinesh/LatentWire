#!/usr/bin/env bash
# monitor_f1_breakthrough.sh
# Monitor eval output in real-time to detect when F1 > 0 first appears
#
# Usage:
#   bash scripts/monitor_f1_breakthrough.sh runs/smoke_stageb_ext/pipeline_*.log

set -euo pipefail

LOG_FILE="${1:-}"

if [[ -z "$LOG_FILE" ]]; then
    echo "Usage: bash scripts/monitor_f1_breakthrough.sh <log_file>"
    exit 1
fi

echo "Monitoring: $LOG_FILE"
echo "Waiting for F1 breakthrough (F1 > 0)..."
echo ""

# Extract all F1 scores from log
extract_f1() {
    grep -E "Llama.*F1:" "$LOG_FILE" 2>/dev/null | while read -r line; do
        # Extract F1 value (handles both "F1: 0.123" and "F1=0.123" formats)
        f1=$(echo "$line" | grep -oE "F1[=:] *[0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
        if [[ -n "$f1" ]]; then
            echo "$f1"
        fi
    done
}

# Monitor in real-time if file is growing
if [[ -f "$LOG_FILE" ]]; then
    # Extract existing F1 scores
    f1_scores=$(extract_f1)

    if [[ -n "$f1_scores" ]]; then
        echo "=== F1 SCORES FOUND ==="
        echo "$f1_scores" | nl -v 1 -w 3 -s ': F1='

        # Check for breakthrough
        max_f1=$(echo "$f1_scores" | sort -rn | head -1)
        if (( $(echo "$max_f1 > 0" | bc -l) )); then
            echo ""
            echo "🎉 BREAKTHROUGH DETECTED! F1 > 0"
            echo "   Peak F1: $max_f1"

            # Find first occurrence of F1 > 0
            first_positive=$(echo "$f1_scores" | awk '$1 > 0 {print NR, $1; exit}')
            if [[ -n "$first_positive" ]]; then
                echo "   First positive at sample: $first_positive"
            fi
        else
            echo ""
            echo "❌ No F1 > 0 detected yet (max: $max_f1)"
        fi
    else
        echo "No F1 scores found in log (training may still be running)"
    fi

    # Show recent training progress
    echo ""
    echo "=== RECENT TRAINING PROGRESS ==="
    grep -E "step.*first_acc" "$LOG_FILE" 2>/dev/null | tail -5 || echo "(no training logs yet)"

    # Show eval metrics if available
    echo ""
    echo "=== EVAL METRICS ==="
    grep -E "EM:|F1:|NLL/token|First-token acc:" "$LOG_FILE" 2>/dev/null | tail -10 || echo "(eval not started)"

else
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi
