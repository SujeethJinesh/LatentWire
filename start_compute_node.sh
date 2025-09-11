#!/bin/bash
# Submit the job and capture the job ID
JOB_ID=$(sbatch --parsable << 'EOF'
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

# Trap signals for cleanup when job ends
trap cleanup EXIT TERM INT

# Optional: Clean up old logs from previous runs (older than 3 days)
find ~ -maxdepth 1 -name "code-server-*.log" -mtime +3 -delete 2>/dev/null
find ~ -maxdepth 1 -name "code-server-*.err" -mtime +3 -delete 2>/dev/null

module load code-server/4.93.1
PORT=$(shuf -i 8000-9999 -n 1)
HOSTNAME=$(hostname)

echo "========================================="
echo "COPY AND RUN THIS ON YOUR LOCAL MACHINE:"
echo ""
echo "ssh -L $PORT:$HOSTNAME:$PORT marlowe"
echo ""
echo "THEN OPEN IN YOUR BROWSER:"
echo ""
echo "http://localhost:$PORT"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Logs will auto-delete when job ends"
echo "========================================="

# Start code-server
code-server --bind-addr 0.0.0.0:$PORT --auth none

# Cleanup happens automatically when job ends or is cancelled
EOF
)

echo "Job $JOB_ID submitted! Waiting for it to start..."
sleep 5

# Show the connection instructions from the log
echo ""
echo "Connection instructions:"
echo "------------------------"
tail -n 20 code-server-${JOB_ID}.log | grep -A 10 "COPY AND RUN"

echo ""
echo "To see full logs: tail -f code-server-${JOB_ID}.log"
echo "To stop: scancel $JOB_ID"
echo "Logs will auto-delete when job ends"
