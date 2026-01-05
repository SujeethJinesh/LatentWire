#!/usr/bin/env bash
# =============================================================================
# Run training with comprehensive logging that survives preemption
# =============================================================================
# This script wraps any training command with the comprehensive logging system
# ensuring all output is captured and persisted even if the job is preempted.
#
# Usage:
#   bash telepathy/run_with_comprehensive_logging.sh [experiment_name] [command...]
#
# Example:
#   bash telepathy/run_with_comprehensive_logging.sh baseline_run python latentwire/train.py --samples 1000
# =============================================================================

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <command...>"
    echo "Example: $0 baseline_run python latentwire/train.py --samples 1000"
    exit 1
fi

# Parse arguments
EXPERIMENT_NAME="$1"
shift  # Remove experiment name from arguments
COMMAND="$@"  # Remaining arguments are the command to run

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/${EXPERIMENT_NAME}_${TIMESTAMP}"
LOG_DIR="$OUTPUT_DIR/logs"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

# Create directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# Log file paths
MAIN_LOG="$LOG_DIR/main_${TIMESTAMP}.log"
METRICS_LOG="$LOG_DIR/metrics_${TIMESTAMP}.jsonl"
ERROR_LOG="$LOG_DIR/error_${TIMESTAMP}.log"

echo "=============================================================="
echo "Comprehensive Logging Setup"
echo "=============================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Main log: $MAIN_LOG"
echo "Metrics log: $METRICS_LOG"
echo "Command: $COMMAND"
echo "Start time: $(date)"
echo "=============================================================="

# Create Python wrapper that adds logging to any script
cat << 'EOF' > "$OUTPUT_DIR/logging_wrapper.py"
#!/usr/bin/env python3
"""
Dynamic wrapper that adds comprehensive logging to any Python script.
"""
import sys
import os
import io
import json
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# Get paths from environment
OUTPUT_DIR = os.environ['OUTPUT_DIR']
METRICS_LOG = os.environ['METRICS_LOG']
CHECKPOINT_DIR = os.environ['CHECKPOINT_DIR']

# Add project to path
sys.path.insert(0, os.environ.get('PROJECT_ROOT', '.'))

# Import logging utilities
from telepathy.logging_utils import (
    TeeLogger,
    StructuredLogger,
    CheckpointLogger,
    GitLogBackup
)

class LoggingInterceptor:
    """Intercepts and logs all output from the wrapped script."""

    def __init__(self):
        self.metrics_logger = StructuredLogger(METRICS_LOG)
        self.checkpoint_logger = CheckpointLogger(CHECKPOINT_DIR)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.git_backup = None
        self.start_time = time.time()

        # Set up git backup if in git repo
        if Path('.git').exists():
            self.git_backup = GitLogBackup('.', backup_interval=300)
            self.git_backup.start()

        # Register signal handlers
        signal.signal(signal.SIGUSR1, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle preemption signals."""
        print(f"\n[LOGGING] Received signal {signum} - saving state...")

        # Save checkpoint info
        self.checkpoint_logger.save_state({
            'signal': signum,
            'runtime': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        })

        # Log event
        self.metrics_logger.log({
            'event': 'signal_received',
            'signal': signum,
            'runtime': time.time() - self.start_time
        })

        # Final git backup
        if self.git_backup:
            self.git_backup.stop()

        # Let the signal propagate
        sys.exit(0)

    def __enter__(self):
        # Log start event
        self.metrics_logger.log({
            'event': 'script_start',
            'command': ' '.join(sys.argv),
            'pid': os.getpid(),
            'job_id': os.getenv('SLURM_JOB_ID', 'local')
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log completion
        self.metrics_logger.log({
            'event': 'script_complete',
            'runtime': time.time() - self.start_time,
            'success': exc_type is None,
            'error': str(exc_val) if exc_val else None
        })

        # Stop git backup
        if self.git_backup:
            self.git_backup.stop()

        # Close loggers
        self.metrics_logger.close()
        self.checkpoint_logger.close()

        return False

# Intercept the import of the target script
with LoggingInterceptor():
    # Get the script to run from arguments
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        sys.argv = sys.argv[1:]  # Shift arguments for the target script

        # Execute the script
        with open(script_path) as f:
            code = compile(f.read(), script_path, 'exec')
            exec(code, {'__name__': '__main__', '__file__': script_path})
    else:
        print("Error: No script specified to run")
        sys.exit(1)
EOF

# Export environment variables for the wrapper
export OUTPUT_DIR="$OUTPUT_DIR"
export METRICS_LOG="$METRICS_LOG"
export CHECKPOINT_DIR="$CHECKPOINT_DIR"
export PROJECT_ROOT="$(pwd)"
export PYTHONUNBUFFERED=1

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "=============================================================="
    echo "Cleaning up and saving logs..."
    echo "=============================================================="

    # Compress large logs
    for log in "$LOG_DIR"/*.log; do
        if [ -f "$log" ]; then
            SIZE=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log" 2>/dev/null || echo 0)
            if [ "$SIZE" -gt 104857600 ]; then  # > 100MB
                echo "Compressing large log: $log"
                gzip -9 "$log"
            fi
        fi
    done

    # Git commit if in repo
    if [ -d .git ]; then
        echo "Saving logs to git..."
        git add "$OUTPUT_DIR/**/*.log" "$OUTPUT_DIR/**/*.jsonl" "$OUTPUT_DIR/**/*.json" 2>/dev/null || true
        git commit -m "logs: $EXPERIMENT_NAME completed/interrupted

Timestamp: $(date -Iseconds)
Command: $COMMAND" || true

        # Try to push
        git push || echo "Warning: Could not push to remote"
    fi

    echo "End time: $(date)"
    echo "Output saved to: $OUTPUT_DIR"
}

# Register cleanup function
trap cleanup EXIT INT TERM

# =============================================================================
# Run the command with comprehensive logging
# =============================================================================

echo ""
echo "Starting command execution with logging..."
echo ""

# Check if the command is a Python script
if [[ "$COMMAND" == python* ]] || [[ "$COMMAND" == python3* ]]; then
    # Extract the Python command and script
    PYTHON_CMD=$(echo "$COMMAND" | awk '{print $1}')
    SCRIPT_AND_ARGS=$(echo "$COMMAND" | cut -d' ' -f2-)

    # Use the logging wrapper for Python scripts
    {
        $PYTHON_CMD "$OUTPUT_DIR/logging_wrapper.py" $SCRIPT_AND_ARGS
    } 2>&1 | tee "$MAIN_LOG"

    EXIT_CODE=${PIPESTATUS[0]}
else
    # For non-Python commands, just use tee
    {
        eval "$COMMAND"
    } 2>&1 | tee "$MAIN_LOG"

    EXIT_CODE=${PIPESTATUS[0]}
fi

# =============================================================================
# Post-processing
# =============================================================================

echo ""
echo "=============================================================="
echo "Execution Complete"
echo "=============================================================="
echo "Exit code: $EXIT_CODE"
echo "Logs saved to: $OUTPUT_DIR"

# Create summary
cat << EOF > "$OUTPUT_DIR/summary.json"
{
    "experiment_name": "$EXPERIMENT_NAME",
    "command": "$COMMAND",
    "start_time": "$TIMESTAMP",
    "end_time": "$(date -Iseconds)",
    "exit_code": $EXIT_CODE,
    "output_dir": "$OUTPUT_DIR",
    "main_log": "$MAIN_LOG",
    "metrics_log": "$METRICS_LOG"
}
EOF

# List generated files
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.* "$LOG_DIR"/*.* 2>/dev/null | head -20

exit $EXIT_CODE