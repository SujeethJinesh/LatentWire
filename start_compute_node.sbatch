#!/bin/bash
#SBATCH --job-name=code-server
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=code-server-%j.log
#SBATCH --error=code-server-%j.err
#SBATCH --signal=B:TERM@30    # Send SIGTERM 30 seconds before time limit

# Cleanup function
cleanup() {
    echo "Cleaning up code-server files..."
    rm -f ${HOME}/code-server-${SLURM_JOB_ID}.log
    rm -f ${HOME}/code-server-${SLURM_JOB_ID}.err
    echo "Cleanup complete at $(date)"
}

# Trap signals for cleanup
trap cleanup EXIT TERM INT

# Clean up old logs on start (older than 3 days)
find ~ -maxdepth 1 -name "code-server-*.log" -mtime +3 -delete 2>/dev/null
find ~ -maxdepth 1 -name "code-server-*.err" -mtime +3 -delete 2>/dev/null

# Load the module
module load code-server/4.93.1

# Generate a random port
PORT=$(shuf -i 8000-9999 -n 1)
HOSTNAME=$(hostname)

# Print connection instructions with copy-paste ready commands
echo "========================================="
echo "CODE-SERVER SETUP COMPLETE!"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $HOSTNAME"
echo "Port: $PORT"
echo "Started at: $(date)"
echo ""
echo "STEP 1: Copy and run this EXACT command on your LOCAL machine:"
echo ""
echo "ssh -L $PORT:$HOSTNAME:$PORT marlowe"
echo ""
echo "STEP 2: Open your browser and go to:"
echo ""
echo "http://localhost:$PORT"
echo ""
echo "========================================="
echo "To stop code-server, run: scancel $SLURM_JOB_ID"
echo "To check status, run: squeue -j $SLURM_JOB_ID"
echo "========================================="

# Keep code-server running with auto-restart if it crashes
while true; do
    echo "Starting code-server at $(date)..."
    
    # Start code-server with connection token disabled and no timeout
    code-server \
        --bind-addr 0.0.0.0:$PORT \
        --auth none \
        --disable-workspace-trust \
        --disable-telemetry \
        --disable-update-check \
        2>&1
    
    # If code-server exits, wait and restart
    EXIT_CODE=$?
    echo "Code-server exited with code $EXIT_CODE at $(date)"
    
    # Check if we're close to the time limit (within 5 minutes)
    if [ -n "$SLURM_JOB_END_TIME" ]; then
        TIME_LEFT=$((SLURM_JOB_END_TIME - $(date +%s)))
        if [ $TIME_LEFT -lt 300 ]; then
            echo "Less than 5 minutes left, exiting gracefully..."
            break
        fi
    fi
    
    echo "Restarting in 5 seconds..."
    sleep 5
done

echo "Job ending at $(date)"
# Cleanup happens automatically when job ends