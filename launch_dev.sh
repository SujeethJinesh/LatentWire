#!/usr/bin/env bash
set -euo pipefail

# One-command interactive allocation with code-server:
# - Reserves 1 node, 4 GPUs for 12h on preempt partition under your account
# - Starts reverse SSH tunnel to 'marlowe'
# - Launches code-server bound to 127.0.0.1:$PORT on the compute node (background)
# - Writes a single info file with the exact ssh -L command and URL
# - Drops you into a bash shell on the compute node immediately

# ---- Config (edit if needed) ----
ACCOUNT="marlowe-m000066"
PARTITION="preempt"
TIME="12:00:00"
MEM="40G"
GPUS=4
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
LOGIN_HOST="marlowe"
CODE_SERVER_MODULE="code-server/4.93.1"

# pick a stable random-ish port each run
PORT="$(shuf -i 8000-9999 -n 1)"

# ---- Run one interactive job with srun ----
exec srun \
  -N 1 \
  -p "$PARTITION" \
  --gpus="$GPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --account="$ACCOUNT" \
  --job-name=code-server \
  --pty bash -lc "
    set -euo pipefail

    cd '$WORK_DIR' || exit 1

    # Load code-server if your site uses modules (ignore failures)
    (module load '$CODE_SERVER_MODULE' 2>/dev/null || true) || true

    # Info file path uses the Slurm job ID from this interactive job
    INFO_FILE=\"$WORK_DIR/code-server-\${SLURM_JOB_ID}-info.txt\"
    TUNNEL_PID_FILE=\"/tmp/revtun-\${SLURM_JOB_ID}.pid\"

    # Clean previous info/logs best-effort
    rm -f \"$WORK_DIR\"/code-server-*.log \"$WORK_DIR\"/code-server-*.err \"$WORK_DIR\"/code-server-*-info.txt 2>/dev/null || true

    # Start reverse tunnel compute -> login (background)
    ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
        -N -R ${PORT}:127.0.0.1:${PORT} '${LOGIN_HOST}' & echo \$! > \"\$TUNNEL_PID_FILE\"

    # Write connection info
    {
      echo \"=========================================\"
      echo \"CODE-SERVER CONNECTION INFO\"
      echo \"Created:      \$(date)\"
      echo \"Job ID:       \${SLURM_JOB_ID}\"
      echo \"Compute node: \$(hostname)\"
      echo \"Login host:   ${LOGIN_HOST}\"
      echo \"Port:         ${PORT}\"
      echo \"\"
      echo \"SSH TUNNEL (run on YOUR LAPTOP):\"
      echo \"  ssh -L ${PORT}:localhost:${PORT} ${LOGIN_HOST}\"
      echo \"\"
      echo \"BROWSER URL:\"
      echo \"  http://localhost:${PORT}\"
      echo \"\"
      echo \"CANCEL JOB FROM LOGIN:\"
      echo \"  scancel \${SLURM_JOB_ID}\"
      echo \"=========================================\"
    } | tee \"\$INFO_FILE\"

    # Start code-server in background, keep logs in the project dir
    LOG_FILE=\"$WORK_DIR/code-server-\${SLURM_JOB_ID}.log\"
    ERR_FILE=\"$WORK_DIR/code-server-\${SLURM_JOB_ID}.err\"

    echo \"Starting code-server on \$(hostname):${PORT} @ \$(date)\" | tee -a \"\$LOG_FILE\"
    (
      set +e
      while true; do
        code-server \\
          --bind-addr 127.0.0.1:${PORT} \\
          --auth none \\
          --disable-workspace-trust \\
          --disable-telemetry \\
          --disable-update-check \\
          --user-data-dir \"/tmp/code-server-\${SLURM_JOB_ID}\" \\
          --extensions-dir \"/tmp/code-server-ext-\${SLURM_JOB_ID}\" \\
          '$WORK_DIR' >>\"\$LOG_FILE\" 2>>\"\$ERR_FILE\"
        EC=\$?
        echo \"code-server exited with \$EC @ \$(date)\" | tee -a \"\$LOG_FILE\"
        # graceful exit if time nearly up
        if [ -n \"\${SLURM_JOB_END_TIME:-}\" ]; then
          NOW=\$(date +%s); LEFT=\$((SLURM_JOB_END_TIME - NOW))
          [ \"\$LEFT\" -lt 300 ] && echo \"Under 5 minutes left; not restarting.\" | tee -a \"\$LOG_FILE\" && break
        fi
        echo \"Restarting in 5s...\" | tee -a \"\$LOG_FILE\"
        sleep 5
      done
    ) &

    # Clean up tunnel on exit
    trap '
      if [ -f \"\$TUNNEL_PID_FILE\" ]; then
        TPID=\$(cat \"\$TUNNEL_PID_FILE\" 2>/dev/null || true)
        if [ -n \"\$TPID\" ] && ps -p \"\$TPID\" >/dev/null 2>&1; then kill \"\$TPID\" || true; fi
        rm -f \"\$TUNNEL_PID_FILE\"
      fi
    ' EXIT INT TERM

    # Finally: drop you into a login shell on the compute node
    echo
    echo \"--- You are now on \$(hostname). Info file: \$INFO_FILE ---\"
    echo \"(From your laptop: ssh -L ${PORT}:localhost:${PORT} ${LOGIN_HOST} ; then open http://localhost:${PORT})\"
    echo
    exec bash -l
  "
