#!/usr/bin/env bash
set -euo pipefail

# Start code-server + reverse tunnel on THIS compute node, idempotently.
# If already running, print connection info and exit.

WORK_DIR="${WORK_DIR:-/projects/m000066/sujinesh/LatentWire}"
LAPTOP_ALIAS="${LAPTOP_ALIAS:-marlowe}"           # what you ssh to from your laptop
CS_MODULE="${CS_MODULE:-code-server/4.93.1}"
PORT="${PORT:-$(shuf -i 8000-9999 -n 1)}"

cd "$WORK_DIR" || { echo "[ERROR] WORK_DIR not found: $WORK_DIR"; exit 1; }

JOBID="${SLURM_JOB_ID:-$$}"
INFO_FILE="$WORK_DIR/code-server-${JOBID}-info.txt"
LOG_FILE="$WORK_DIR/code-server-${JOBID}.log"
ERR_FILE="$WORK_DIR/code-server-${JOBID}.err"
TUNNEL_TAG_FILE="/tmp/revtun-${JOBID}-${PORT}.tag"
CS_PID_FILE="/tmp/codeserver-${JOBID}-${PORT}.pid"

# The compute node must tunnel back to a resolvable login host:
LOGIN_HOST_REAL="${SLURM_SUBMIT_HOST:-$(hostname -f)}"

# --- Detect already-running code-server on this node+port ---
if pgrep -f "code-server .*--bind-addr 127\.0\.0\.1:${PORT}" >/dev/null 2>&1; then
  if [ -f "$INFO_FILE" ]; then
    echo "[info] code-server already running on port $PORT"
    cat "$INFO_FILE"
  else
    echo "[info] code-server appears to be running on $PORT, but no info file."
    echo "Try: ss -ltn | grep :$PORT  &&  pgrep -af code-server"
  fi
  exit 0
fi

# --- Init modules quietly and load code-server if needed ---
if ! command -v module >/dev/null 2>&1; then
  [ -r /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -r /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi
command -v code-server >/dev/null 2>&1 || module load "$CS_MODULE" 2>/dev/null || true
if ! command -v code-server >/dev/null 2>&1; then
  echo "[ERROR] code-server not found after module init (tried '$CS_MODULE')."
  echo "        Adjust CS_MODULE or add code-server to PATH, then retry."
  exit 1
fi

# --- Start reverse tunnel (compute -> login), only if not already matching this port ---
if [ ! -f "$TUNNEL_TAG_FILE" ] || ! pgrep -f "ssh .* -R ${PORT}:127\.0\.0\.1:${PORT} .*${LOGIN_HOST_REAL}" >/dev/null 2>&1; then
  # background (-f) after auth; fail if forward can't be set
  ssh -f -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -N -R "${PORT}:127.0.0.1:${PORT}" "${LOGIN_HOST_REAL}"
  echo "${LOGIN_HOST_REAL}:${PORT}" > "$TUNNEL_TAG_FILE"
fi

# --- Write (or refresh) the connection info file ---
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
  echo "  kill \$(cat $CS_PID_FILE)  # then optionally: pkill -f \"ssh .* -R ${PORT}:\""
  echo "========================================="
} | tee "$INFO_FILE"

# --- Start code-server in background ---
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
echo
