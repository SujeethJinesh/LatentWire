#!/bin/bash
# monitor_experiments.sh
#
# Run this in a separate terminal to monitor experiment progress
# Usage: ./monitor_experiments.sh [runs_dir]

RUNS_DIR="${1:-runs}"

echo "============================================================"
echo "EXPERIMENT MONITOR"
echo "============================================================"
echo "Watching: $RUNS_DIR"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "============================================================"
    echo "EXPERIMENT STATUS @ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # Show GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU USAGE:"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=, read -r idx mem_used mem_total util; do
            printf "  GPU %s: %5s/%5s MB (%3s%% util)\n" "$idx" "$mem_used" "$mem_total" "$util"
        done
        echo ""
    fi

    # Find all active experiment directories (modified in last hour)
    echo "ACTIVE EXPERIMENTS:"
    echo "-------------------"

    # Look for recent log files
    find "$RUNS_DIR" -name "*.log" -mmin -60 2>/dev/null | sort | while read -r logfile; do
        if [ -f "$logfile" ]; then
            dir=$(dirname "$logfile")
            name=$(basename "$dir")

            # Get latest checkpoint or final result
            checkpoint=$(grep -E "\[CHECKPOINT\]|\[FINAL\]" "$logfile" 2>/dev/null | tail -1)
            if [ -n "$checkpoint" ]; then
                echo "  $name:"
                echo "    $checkpoint"
            else
                # Fallback: show last progress-like line
                progress=$(grep -E "Training:|it/s|Accuracy|Step" "$logfile" 2>/dev/null | tail -1 | head -c 70)
                if [ -n "$progress" ]; then
                    echo "  $name:"
                    echo "    $progress..."
                fi
            fi
        fi
    done

    echo ""
    echo "-------------------"
    echo "COMPLETED RESULTS:"
    echo "-------------------"

    # Show final results from JSON files
    find "$RUNS_DIR" -name "*_results.json" -mmin -120 2>/dev/null | sort | while read -r jsonfile; do
        if [ -f "$jsonfile" ]; then
            name=$(basename "$(dirname "$jsonfile")")
            if [[ "$jsonfile" == *"banking77"* ]]; then
                acc=$(python3 -c "import json; print(f\"{json.load(open('$jsonfile'))['final_results']['accuracy']:.1f}%\")" 2>/dev/null || echo "N/A")
                echo "  $name: Accuracy $acc"
            elif [[ "$jsonfile" == *"passkey"* ]]; then
                exact=$(python3 -c "import json; print(f\"{json.load(open('$jsonfile'))['final_results']['exact_match']:.1f}%\")" 2>/dev/null || echo "N/A")
                digit=$(python3 -c "import json; print(f\"{json.load(open('$jsonfile'))['final_results']['digit_accuracy']:.1f}%\")" 2>/dev/null || echo "N/A")
                echo "  $name: Exact $exact, Digit $digit"
            elif [[ "$jsonfile" == *"text_relay"* ]]; then
                sst2=$(python3 -c "import json; print(f\"{json.load(open('$jsonfile'))['results']['sst2']['accuracy']:.1f}%\")" 2>/dev/null || echo "N/A")
                agnews=$(python3 -c "import json; print(f\"{json.load(open('$jsonfile'))['results']['agnews']['accuracy']:.1f}%\")" 2>/dev/null || echo "N/A")
                echo "  $name: SST-2 $sst2, AG News $agnews"
            fi
        fi
    done

    echo ""
    echo "============================================================"
    echo "Refreshing in 15 seconds... (Ctrl+C to stop)"

    sleep 15
done
