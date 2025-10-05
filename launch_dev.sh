#!/usr/bin/env bash
set -euo pipefail

# Config
SBATCH_FILE="${SBATCH_FILE:-start_compute_node.sbatch}"
JOB_NAME="code-server"
POLL_SECS=5
KILL_BUFFER_SECS=30   # kill login-side tmux shortly before job end

# Submit the job
jid_line="$(sbatch "$SBATCH_FILE")"
jid="$(awk '{print $4}' <<<"$jid_line")"
echo "Submitted ${JOB_NAME} job: JID=${jid}"

# Wait for allocation and node
echo -n "Waiting for allocation"
state=""; node=""
while true; do
  read -r state node <<<"$(squeue -j "$jid" -h -o "%T %N")" || true
  [ -n "${state:-}" ] || { echo -n "."; sleep "$POLL_SECS"; continue; }
  if [[ "$state" == "RUNNING" && "$node" != "(null)" && -n "$node" ]]; then
    echo; echo "Allocated on node: $node"
    break
  fi
  echo -n "."
  sleep "$POLL_SECS"
done

# Wait for info file and parse values
info_file="/projects/m000066/sujinesh/LatentWire/code-server-${jid}-info.txt"
echo -n "Waiting for connection info ${info_file}"
until [[ -f "$info_file" ]]; do echo -n "."; sleep "$POLL_SECS"; done
echo
port="$(awk '/^Port:/{print $2}' "$info_file")"
login_host="$(awk '/^Login host:/{print $3}' "$info_file")"   # "marlowe"
compute_node="$(awk '/^Compute node:/{print $3}' "$info_file")"

echo "------------------------------------------------------------"
echo "From YOUR LAPTOP:"
echo "  ssh -L ${port}:localhost:${port} ${login_host}"
echo "Then browse:"
echo "  http://localhost:${port}"
echo "Info file: $info_file"
echo "------------------------------------------------------------"

# Prepare the compute-side init script as a here-doc (avoids fragile quoting)
compute_script=$(cat <<'EOF_CS'
#!/usr/bin/env bash
# Don't 'set -e' here; we want interactive fallback even if tmux/module fails
# Try to make tmux available (ignore errors if no modules)
(module load tmux 2>/dev/null || true) || true

csess="cs_${SLURM_JOB_ID:-alloc}"

if command -v tmux >/dev/null 2>&1; then
  # Create a compute-side tmux with two panes if not already present
  if ! tmux has-session -t "$csess" 2>/dev/null; then
    tmux new-session -d -s "$csess" "bash -l"             # left: dev shell
    tmux split-window -h -t "$csess" "bash -lc 'watch -n1 nvidia-smi; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'"
    tmux select-pane -L -t "$csess"
  fi
  exec tmux attach -t "$csess"
else
  echo "[warn] tmux not available on compute node. Starting single shell."
  echo "[hint] To watch GPUs in background:  watch -n1 nvidia-smi &"
  exec bash -l
fi
EOF_CS
)

# One login-side tmux session; compute panes live inside a single srun step
session="dev_${jid}"
tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true

if command -v tmux >/dev/null 2>&1 && [ -z "${TMUX:-}" ]; then
  # Start tmux; pane 1 runs ONE long-lived Slurm step hosting compute-side tmux
  tmux new-session -d -s "$session" \
    "srun --jobid=${jid} --pty bash -lc 'cat > /tmp/cs_${jid}.sh <<\"CS_EOF\" && chmod +x /tmp/cs_${jid}.sh && /tmp/cs_${jid}.sh
${compute_script}
CS_EOF
'"

  # Pane 2: login-node queue watch -> sticky (Ctrl-C drops to shell)
  tmux split-window -v -t "$session" \
    "bash -lc 'watch -n2 squeue -u sujinesh; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'"

  # Make pane 1 active and show a helpful message
  tmux select-pane -U
  tmux set-option -t "$session" remain-on-exit on
  tmux display-message -t "$session" "Node: ${compute_node} | Tunnel: ssh -L ${port}:localhost:${port} ${login_host} | URL: http://localhost:${port}"

  # Auto-kill login-side tmux shortly before job end
  if command -v scontrol >/dev/null 2>&1; then
    end_iso="$(scontrol show jobid -dd "$jid" | awk -F= '/EndTime=/{print $2}' | awk '{print $1}')"
    if date -d "$end_iso" +"%s" >/dev/null 2>&1; then
      end_secs="$(date -d "$end_iso" +"%s")"
    else
      end_secs="$(( $(date +%s) + 12*3600 ))"
    fi
  else
    end_secs="$(( $(date +%s) + 12*3600 ))"
  fi
  now_secs="$(date +%s)"
  sleep_secs="$(( end_secs - now_secs - KILL_BUFFER_SECS ))"
  if (( sleep_secs > 0 )); then
    (
      sleep "$sleep_secs" || true
      tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true
    ) >/dev/null 2>&1 &
  fi

  tmux attach -t "$session"
else
  echo "(tmux not available on login node or you're already in tmux)"
  echo "Opening a single interactive shell on the compute node..."
  srun --jobid="${jid}" --pty bash -l
fi
