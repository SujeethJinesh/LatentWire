#!/bin/bash
#SBATCH --job-name=code-server
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/code-server-%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/code-server-%j.err
#SBATCH --signal=B:TERM@30

WORK_DIR="/projects/m000066/sujinesh/LatentWire"
LOGIN_HOST="${SLURM_SUBMIT_HOST:-marlowe}"
PORT="$(shuf -i 8000-9999 -n 1)"
INFO_FILE="${WORK_DIR}/code-server-${SLURM_JOB_ID}-info.txt"
TUNNEL_PID_FILE="/tmp/revtun-${SLURM_JOB_ID}.pid"

cd "$WORK_DIR" || exit 1

cleanup() {
  echo "Cleanup @ $(date)"
  # Kill reverse tunnel
  if [ -f "$TUNNEL_PID_FILE" ]; then
    TPID="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
    if [ -n "$TPID" ] && ps -p "$TPID" >/dev/null 2>&1; then kill "$TPID" || true; fi
    rm -f "$TUNNEL_PID_FILE"
  fi
  # Remove this job's artifacts
  rm -f "${WORK_DIR}/code-server-${SLURM_JOB_ID}.log" \
        "${WORK_DIR}/code-server-${SLURM_JOB_ID}.err" \
        "${WORK_DIR}/code-server-${SLURM_JOB_ID}-info.txt" 2>/dev/null || true
  # Remove ALL other code-server info/logs so only current job remains
  rm -f ${WORK_DIR}/code-server-*.log ${WORK_DIR}/code-server-*.err ${WORK_DIR}/code-server-*-info.txt 2>/dev/null || true
  # Temp + caches
  rm -rf "/tmp/code-server-${SLURM_JOB_ID}" "/tmp/code-server-ext-${SLURM_JOB_ID}" 2>/dev/null || true
  rm -rf ~/.local/share/code-server/logs/* ~/.cache/code-server/* 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# Clean any previous info/log files at start as well
rm -f ${WORK_DIR}/code-server-*.log ${WORK_DIR}/code-server-*.err ${WORK_DIR}/code-server-*-info.txt 2>/dev/null || true

module load code-server/4.93.1

# Start reverse tunnel compute -> login
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -N -R ${PORT}:127.0.0.1:${PORT} "${LOGIN_HOST}" &
echo $! > "$TUNNEL_PID_FILE"

# Emit a single, canonical info file (easy to `cat`)
{
  echo "========================================="
  echo "CODE-SERVER CONNECTION INFO"
  echo "Created:      $(date)"
  echo "Job ID:       ${SLURM_JOB_ID}"
  echo "Compute node: $(hostname)"
  echo "Port:         ${PORT}"
  echo "Login host:   ${LOGIN_HOST}"
  echo ""
  echo "SSH TUNNEL (run on YOUR LAPTOP):"
  echo "ssh -L ${PORT}:localhost:${PORT} ${LOGIN_HOST}"
  echo ""
  echo "BROWSER URL:"
  echo "http://localhost:${PORT}"
  echo ""
  echo "To cancel job:"
  echo "scancel ${SLURM_JOB_ID}"
  echo "========================================="
} | tee "$INFO_FILE"

# Run code-server bound to localhost; restart if it exits and time remains
while true; do
  echo "Starting code-server @ $(date)"
  code-server \
    --bind-addr 127.0.0.1:${PORT} \
    --auth none \
    --disable-workspace-trust \
    --disable-telemetry \
    --disable-update-check \
    --user-data-dir "/tmp/code-server-${SLURM_JOB_ID}" \
    --extensions-dir "/tmp/code-server-ext-${SLURM_JOB_ID}" \
    "$WORK_DIR"
  ec=$?
  echo "code-server exited with $ec @ $(date)"

  if [ -n "$SLURM_JOB_END_TIME" ]; then
    now=$(date +%s); left=$((SLURM_JOB_END_TIME - now))
    [ "$left" -lt 300 ] && echo "Under 5m left; exiting." && break
  fi
  echo "Restarting in 5s..."; sleep 5
done
