#!/usr/bin/env bash
set -euo pipefail

# One-command interactive job that:
# - allocates 1 node / 4 GPUs on preempt under your account for 12h
# - starts a reverse tunnel to the REAL login host (compute can resolve it)
# - loads the module system on the compute node, then loads code-server/4.93.1
# - starts code-server in the background, writes a connection info file
# - drops you into a bash shell on the compute node

ACCOUNT="${ACCOUNT:-marlowe-m000066}"
PARTITION="${PARTITION:-preempt}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-40G}"
GPUS="${GPUS:-4}"
WORK_DIR="${WORK_DIR:-/projects/m000066/sujinesh/LatentWire}"
# For your laptop instructions you want 'marlowe'
LAPTOP_ALIAS="${LAPTOP_ALIAS:-marlowe}"

# Compute node must tunnel back to the *real* login hostname
LOGIN_HOST_REAL="$(hostname -f || hostname)"

PORT="$(shuf -i 8000-9999 -n 1)"

exec srun \
  -N 1 \
  -p "$PARTITION" \
  --gpus="$GPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --account="$ACCOUNT" \
  --job-name=code-server \
  --pty env PORT="$PORT" WORK_DIR="$WORK_DIR" LAPTOP_ALIAS="$LAPTOP_ALIAS" LOGIN_HOST_REAL="$LOGIN_HOST_REAL" bash -lc '
set -euo pipefail

cd "$WORK_DIR" || exit 1

INFO_FILE="$WORK_DIR/code-server-${SLURM_JOB_ID}-info.txt"
LOG_FILE="$WORK_DIR/code-server-${SLURM_JOB_ID}.log"
ERR_FILE="$WORK_DIR/code-server-${SLURM_JOB_ID}.err"
TUNNEL_PID_FILE="/tmp/revtun-${SLURM_JOB_ID}.pid"

# Clean old info/logs (best effort)
rm -f "$WORK_DIR"/code-server-*.log "$WORK_DIR"/code-server-*.err "$WORK_DIR"/code-server-*-info.txt 2>/dev/null || true

# --- Start reverse tunnel: compute -> real login host (resolvable from compute) ---
( ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
     -N -R "${PORT}:127.0.0.1:${PORT}" "$LOGIN_HOST_REAL" &
  echo $! > "$TUNNEL_PID_FILE"
) || true

# Write user-facing info (uses your laptop alias "marlowe" for convenience)
{
  echo "========================================="
  echo "CODE-SERVER CONNECTION INFO"
  echo "Created:      $(date)"
  echo "Job ID:       ${SLURM_JOB_ID}"
  echo "Compute node: $(hostname)"
  echo "Login host:   ${LAPTOP_ALIAS}   (tunnel target: ${LOGIN_HOST_REAL})"
  echo "Port:         ${PORT}"
  echo ""
  echo "SSH TUNNEL (run on YOUR LAPTOP):"
  echo "  ssh -L ${PORT}:localhost:${PORT} ${LAPTOP_ALIAS}"
  echo ""
  echo "BROWSER URL:"
  echo "  http://localhost:${PORT}"
  echo ""
  echo "CANCEL JOB FROM LOGIN:"
  echo "  scancel ${SLURM_JOB_ID}"
  echo "========================================="
} | tee "$INFO_FILE"

# --- Ensure the module system is initialized on the compute node ---
# Sites vary; try common init paths quietly
if ! command -v module >/dev/null 2>&1; then
  [ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -r /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi

# Load code-server module if available
CS_BIN=""
if command -v code-server >/dev/null 2>&1; then
  CS_BIN="$(command -v code-server)"
else
  # Try the known module name; ignore errors if not present
  ( module load code-server/4.93.1 2>/dev/null || true )
  ( module load code-server 2>/dev/null || true )
  command -v code-server >/dev/null 2>&1 && CS_BIN="$(command -v code-server)" || true
fi

# Start code-server in background if we found it
if [ -n "$CS_BIN" ]; then
  echo "Starting code-server ($CS_BIN) on $(hostname):${PORT} @ $(date)" | tee -a "$LOG_FILE"
  (
    set +e
    while true; do
      "$CS_BIN" \
        --bind-addr "127.0.0.1:${PORT}" \
        --auth none \
        --disable-workspace-trust \
        --disable-telemetry \
        --disable-update-check \
        --user-data-dir "/tmp/code-server-${SLURM_JOB_ID}" \
        --extensions-dir "/tmp/code-server-ext-${SLURM_JOB_ID}" \
        "$WORK_DIR" >>"$LOG_FILE" 2>>"$ERR_FILE"
      EC=$?
      echo "code-server exited with $EC @ $(date)" | tee -a "$LOG_FILE"
      if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
        NOW=$(date +%s); LEFT=$((SLURM_JOB_END_TIME - NOW))
        [ "$LEFT" -lt 300 ] && echo "Under 5 minutes left; not restarting." | tee -a "$LOG_FILE" && break
      fi
      echo "Restarting in 5s..." | tee -a "$LOG_FILE"
      sleep 5
    done
  ) &
else
  echo "[warn] code-server not found on compute node PATH after module init." | tee -a "$ERR_FILE"
  echo "[hint] Try:  module load code-server/4.93.1    # then rerun" | tee -a "$ERR_FILE"
fi

# Clean up the tunnel on exit
trap "
  if [ -f \"$TUNNEL_PID_FILE\" ]; then
    TPID=\$(cat \"$TUNNEL_PID_FILE\" 2>/dev/null || true)
    if [ -n \"\$TPID\" ] && ps -p \"\$TPID\" >/dev/null 2>&1; then kill \"\$TPID\" || true; fi
    rm -f \"$TUNNEL_PID_FILE\"
  fi
" EXIT INT TERM

echo
echo \"--- You are now on \$(hostname). Info file: \$INFO_FILE ---\"
echo \"(From your laptop: ssh -L ${PORT}:localhost:${PORT} ${LAPTOP_ALIAS} ; then open http://localhost:${PORT})\"
echo
exec bash -l
'
