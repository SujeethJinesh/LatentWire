#!/usr/bin/env bash
set -euo pipefail

# Minimal code-server starter for the CURRENT node.
# - Starts reverse SSH tunnel back to your login host
# - Starts code-server in background
# - Writes a single info file with the exact ssh -L command & URL
# - Leaves your shell free

# ----- Config (change if you like) -----
WORK_DIR="${WORK_DIR:-/projects/m000066/sujinesh/LatentWire}"
# What your LAPTOP will ssh to (your usual alias):
LAPTOP_ALIAS="${LAPTOP_ALIAS:-marlowe}"
# Code-server module to load if needed:
CS_MODULE="${CS_MODULE:-code-server/4.93.1}"
# Port to use (picked randomly by default)
PORT="${PORT:-$(shuf -i 8000-9999 -n 1)}"
# ---------------------------------------

cd "$WORK_DIR" || { echo "Missing WORK_DIR: $WORK_DIR"; exit 1; }

JOBID="${SLURM_JOB_ID:-$$}"
INFO_FILE="$WORK_DIR/code-server-${JOBID}-info.txt"
LOG_FILE="$WORK_DIR/code-server-${JOBID}.log"
ERR_FILE="$WORK_DIR/code-server-${JOBID}.err"
TUNNEL_PID_FILE="/tmp/revtun-${JOBID}.pid"
CS_PID_FILE="/tmp/codeserver-${JOBID}.pid"

# Use the real hostname of the login node for the *tunnel target* so compute can resolve it.
LOGIN_HOST_REAL="${SLURM_SUBMIT_HOST:-$(hostname -f)}"

# Best-effort clean old files
rm -f "$WORK_DIR"/code-server-*.log "$WORK_DIR"/code-server-*.err "$WORK_DIR"/code-server-*-info.txt 2>/dev/null || true

# Initialize environment modules if present (quietly)
if ! command -v module >/dev/null 2>&1; then
  [ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -r /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi

# Load code-server module if code-server not already in PATH
if ! command -v code-server >/dev/null 2>&1; then
  module load "$CS_MODULE" 2>/dev/null || true
fi

if ! command -v code-server >/dev/null 2>&1; then
  echo "[ERROR] code-server not found in PATH and module '$CS_MODULE' didn't load." | tee -a "$ERR_FILE"
  echo "        Try 'module avail' and adjust CS_MODULE in this script." | tee -a "$ERR_FILE"
  exit 1
fi

# Start reverse tunnel (compute -> login-real), backgrounded
# -f puts ssh in background after auth; ExitOnForwardFailure ensures we fail if the port can't be bound
ssh -f -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -N -R "${PORT}:127.0.0.1:${PORT}" "${LOGIN_HOST_REAL}"
# Save the last background ssh PID (the one we just launched)
# shellcheck disable=SC2009
TPID="$(ps -o pid= -C ssh --sort=-pid | head -n1 | tr -d ' ')"
echo "${TPID:-}" > "$TUNNEL_PID_FILE"

# Write connection info (uses LAPTOP_ALIAS for the command you run on your laptop)
{
  echo "========================================="
  echo "CODE-SERVER CONNECTION INFO"
  echo "Created:      $(date)"
  echo "Job ID:       ${JOBID}"
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
  echo "Stop code-server:"
  echo "  kill \$(cat $CS_PID_FILE)  # then optionally: kill \$(cat $TUNNEL_PID_FILE)"
  echo "========================================="
} | tee "$INFO_FILE"

# Start code-server in background and save PID
echo "Starting code-server on $(hostname):${PORT} @ $(date)" | tee -a "$LOG_FILE"
nohup code-server \
  --bind-addr "127.0.0.1:${PORT}" \
  --auth none \
  --disable-workspace-trust \
  --disable-telemetry \
  --disable-update-check \
  --user-data-dir "/tmp/code-server-${JOBID}" \
  --extensions-dir "/tmp/code-server-ext-${JOBID}" \
  "$WORK_DIR" >>"$LOG_FILE" 2>>"$ERR_FILE" &
echo $! > "$CS_PID_FILE"

echo
echo "Done. Info file: $INFO_FILE"
echo "Tail logs:   tail -f $LOG_FILE"
echo "Stop server: kill \$(cat $CS_PID_FILE)"
echo "Stop tunnel: kill \$(cat $TUNNEL_PID_FILE)"
echo
