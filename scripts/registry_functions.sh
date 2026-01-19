#!/bin/bash
# ==============================================================================
# registry_functions.sh - Bash helpers for SLURM integration with experiment registry
#
# Usage:
#   source scripts/registry_functions.sh
#   init_registry
#   should_run_experiment "experiment_name" && echo "Should run" || echo "Should skip"
#   update_registry "experiment_name" "completed" 0 "path/to/result.json"
# ==============================================================================

# Registry file location (can be overridden before sourcing)
REGISTRY_FILE="${REGISTRY_FILE:-runs/experiment_registry.json}"
REGISTRY_SCRIPT="telepathy/experiment_registry.py"

# Tracking for skipped experiments
SKIPPED_EXPERIMENTS=()

# ------------------------------------------------------------------------------
# Initialize the registry (create if not exists)
# ------------------------------------------------------------------------------
init_registry() {
    local registry_dir=$(dirname "$REGISTRY_FILE")
    mkdir -p "$registry_dir"

    if [ ! -f "$REGISTRY_FILE" ]; then
        log_msg "Creating new experiment registry at $REGISTRY_FILE"
        python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" summary > /dev/null 2>&1 || {
            # Fallback: create minimal registry
            echo '{"meta": {"created_at": "'$(date -Iseconds)'", "last_updated": "'$(date -Iseconds)'", "job_id": "'${SLURM_JOB_ID:-local}'"}, "experiments": {}}' > "$REGISTRY_FILE"
        }
    fi
    log_msg "Registry initialized: $REGISTRY_FILE"
}

# ------------------------------------------------------------------------------
# Check if an experiment should run
# Returns: 0 to run, 1 to skip
# Sets: SKIP_REASON global variable with the reason
# ------------------------------------------------------------------------------
should_run_experiment() {
    local name="$1"
    local result

    result=$(python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" should_run "$name" 2>/dev/null)
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        # Python script failed, assume we should run
        SKIP_REASON="registry check failed"
        return 0
    fi

    # Parse result: "1|reason" or "0|reason"
    local should_run=$(echo "$result" | cut -d'|' -f1)
    SKIP_REASON=$(echo "$result" | cut -d'|' -f2-)

    if [ "$should_run" = "1" ]; then
        return 0  # Should run
    else
        SKIPPED_EXPERIMENTS+=("$name")
        return 1  # Should skip
    fi
}

# ------------------------------------------------------------------------------
# Update registry with experiment status
# Args:
#   $1 - name: Experiment name
#   $2 - status: "running", "completed", or "failed"
#   $3 - exit_code: Process exit code (for failed/completed)
#   $4 - result_file: Path to result file (optional, for completed)
#   $5 - error_msg: Error message (optional, for failed)
#   $6 - gpu: GPU index (optional, for running)
# ------------------------------------------------------------------------------
update_registry() {
    local name="$1"
    local status="$2"
    local exit_code="${3:-0}"
    local result_file="${4:-}"
    local error_msg="${5:-}"
    local gpu="${6:-0}"

    case "$status" in
        running)
            python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" start "$name" --gpu "$gpu" 2>/dev/null || {
                log_msg "Warning: Failed to update registry for $name (running)"
            }
            ;;
        completed)
            local metrics_arg=""
            # Try to extract accuracy from result file
            if [ -n "$result_file" ] && [ -f "$result_file" ]; then
                local accuracy=$(python -c "import json; d=json.load(open('$result_file')); print(d.get('final_results', d.get('results', {})).get('accuracy', -1))" 2>/dev/null)
                if [ -n "$accuracy" ] && [ "$accuracy" != "-1" ]; then
                    metrics_arg="--metrics '{\"accuracy\": $accuracy}'"
                fi
            fi
            if [ -n "$result_file" ]; then
                eval python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" complete "$name" --result-file "$result_file" $metrics_arg 2>/dev/null || {
                    log_msg "Warning: Failed to update registry for $name (completed)"
                }
            else
                eval python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" complete "$name" $metrics_arg 2>/dev/null || {
                    log_msg "Warning: Failed to update registry for $name (completed)"
                }
            fi
            ;;
        failed)
            python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" fail "$name" --error "$error_msg" --exit-code "$exit_code" 2>/dev/null || {
                log_msg "Warning: Failed to update registry for $name (failed)"
            }
            ;;
        *)
            log_msg "Warning: Unknown status '$status' for $name"
            ;;
    esac
}

# ------------------------------------------------------------------------------
# Get registry summary (JSON format)
# ------------------------------------------------------------------------------
get_registry_summary() {
    python "$REGISTRY_SCRIPT" --registry "$REGISTRY_FILE" summary 2>/dev/null || {
        echo '{"error": "failed to get summary"}'
    }
}

# ------------------------------------------------------------------------------
# Print human-readable registry summary
# ------------------------------------------------------------------------------
print_registry_summary() {
    echo ""
    echo "=============================================================="
    echo "EXPERIMENT REGISTRY SUMMARY"
    echo "=============================================================="

    local summary=$(get_registry_summary)
    local total=$(echo "$summary" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('total', 0))" 2>/dev/null || echo "0")
    local completed=$(echo "$summary" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('completed', 0))" 2>/dev/null || echo "0")
    local failed=$(echo "$summary" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('failed', 0))" 2>/dev/null || echo "0")
    local pending=$(echo "$summary" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('pending', 0))" 2>/dev/null || echo "0")
    local running=$(echo "$summary" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('running', 0))" 2>/dev/null || echo "0")
    local needs_rerun=$(echo "$summary" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('needs_rerun', 0))" 2>/dev/null || echo "0")

    echo "Total experiments: $total"
    echo "  Completed: $completed"
    echo "  Failed: $failed"
    echo "  Pending: $pending"
    echo "  Running: $running"
    echo "  Needs re-run: $needs_rerun"

    if [ ${#SKIPPED_EXPERIMENTS[@]} -gt 0 ]; then
        echo ""
        echo "Skipped in this job: ${#SKIPPED_EXPERIMENTS[@]}"
        for exp in "${SKIPPED_EXPERIMENTS[@]}"; do
            echo "  - $exp"
        done
    fi
    echo "=============================================================="
}

# ------------------------------------------------------------------------------
# Find result file for an experiment
# Args:
#   $1 - output_dir: Base output directory
#   $2 - experiment_name: Experiment name
# Returns: Path to result file or empty string
# ------------------------------------------------------------------------------
find_result_file() {
    local output_dir="$1"
    local exp_name="$2"

    # Common patterns for result files
    local patterns=(
        "${output_dir}/${exp_name}/*_results.json"
        "${output_dir}/*/${exp_name}/*_results.json"
        "${output_dir}/${exp_name}.json"
        "${output_dir}/*/${exp_name}.json"
    )

    for pattern in "${patterns[@]}"; do
        local found=$(ls $pattern 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            echo "$found"
            return 0
        fi
    done

    # No result file found
    echo ""
    return 1
}

# ------------------------------------------------------------------------------
# Capture error message from log file (last 50 lines with error patterns)
# Args:
#   $1 - log_file: Path to log file
# Returns: Error message string (max 500 chars)
# ------------------------------------------------------------------------------
extract_error_msg() {
    local log_file="$1"

    if [ ! -f "$log_file" ]; then
        echo "Log file not found"
        return
    fi

    # Look for common error patterns
    local error_msg=$(tail -50 "$log_file" 2>/dev/null | grep -E "(Error|Exception|CUDA|OOM|NaN|Failed|Traceback)" | tail -5 | head -c 500)

    if [ -z "$error_msg" ]; then
        # Fallback: just get last few lines
        error_msg=$(tail -3 "$log_file" 2>/dev/null | head -c 500)
    fi

    echo "$error_msg"
}

# ------------------------------------------------------------------------------
# Enhanced run_experiment with registry integration
# This wraps the existing run_experiment function
# Args:
#   $1 - name: Experiment name
#   $2 - gpu: GPU index
#   $3+ - cmd: Command to run
# ------------------------------------------------------------------------------
run_experiment_with_registry() {
    local name="$1"
    local gpu="$2"
    shift 2
    local cmd="$@"

    # Check if we should run
    if ! should_run_experiment "$name"; then
        log_msg "SKIP: $name ($SKIP_REASON)"
        return 0  # Return success for skipped experiments
    fi

    # Mark as running
    update_registry "$name" "running" 0 "" "" "$gpu"

    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    log_msg "START: $name (GPU $gpu)"

    # Run the experiment
    local start_time=$(date +%s)
    local output_capture=$(mktemp)

    if eval "$cmd" 2>&1 | tee "$output_capture"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        log_msg "SUCCESS: $name (${duration}s)"

        # Try to find result file
        local result_file=$(find_result_file "$OUTPUT_DIR" "$name")
        update_registry "$name" "completed" 0 "$result_file"

        rm -f "$output_capture"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        FAILED_EXPERIMENTS+=("$name")

        # Extract error message
        local error_msg=$(extract_error_msg "$output_capture")
        log_msg "FAILED: $name (exit code: $exit_code, ${duration}s)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name failed with exit code $exit_code" >> "${OUTPUT_DIR}/experiment_errors.log"
        echo "$error_msg" >> "${OUTPUT_DIR}/experiment_errors.log"

        update_registry "$name" "failed" "$exit_code" "" "$error_msg" "$gpu"

        rm -f "$output_capture"
        return $exit_code
    fi
}

# ------------------------------------------------------------------------------
# Commit and push registry to git
# ------------------------------------------------------------------------------
commit_registry() {
    local msg="${1:-Update experiment registry}"

    if [ -f "$REGISTRY_FILE" ]; then
        git add "$REGISTRY_FILE" 2>/dev/null || true
        git commit -m "$msg" 2>/dev/null || true
    fi
}

log_msg() {
    # Use existing log_msg if defined, otherwise define a basic one
    if ! type -t log_msg_original > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    fi
}
