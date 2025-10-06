#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher: ensure code-server job exists, then open ONE interactive step
# to the compute node and create a tmux session "dev" with:
#   left       = dev shell (prints code-server info first)
#   right-top  = watch -n1 nvidia-smi (~80% height, sticky)
#   right-bot  = watch -n2 squeue -u sujinesh (sticky).
# If "dev" exists already, error out.

SBATCH_FILE="${SBATCH_FILE:-start_compute_node.sbatch}"
JOB_NAME="code-server"
POLL_SECS=5
SESSION_NAME="${SESSION_NAME:-dev}"

# 1) Reuse running job or submit a new one
jid="$(squeue -u "$USER" -h -t R -n "$JOB_NAME" -o %A | tail -n1 || true)"
if [[ -z "${jid}" ]]; then
  echo "No running ${JOB_NAME} job found; submitting ${SBATCH_FILE}..."
  out="$(sbatch "$SBATCH_FILE")"
  jid="$(awk '{print $4}' <<<"$out")"
  echo "Submitted ${JOB_NAME} job: JID=${jid}"
fi

# 2) Wait until RUNNING and get node
echo -n "Waiting for allocation"
state=""; node=""
while true; do
  read -r state node <<<"$(squeue -j "$jid" -h -o "%T %N")" || true
  if [[ "$state" == "RUNNING" && -n "${node:-}" && "$node" != "(null)" ]]; then
    echo; echo "Allocated on node: $node"
    break
  fi
  echo -n "."
  sleep "$POLL_SECS"
done

# 3) One interactive step -> build tmux layout on the compute node
INFO_FILE="/projects/m000066/sujinesh/LatentWire/code-server-${jid}-info.txt"
echo "Opening compute session '${SESSION_NAME}' on ${node} (job ${jid})..."
exec srun --jobid="${jid}" -w "${node}" --pty bash -lc "
  # Write a tiny setup script on the compute node and run it
  cat > /tmp/dev_setup_${jid}.sh <<'CS_EOF'
#!/usr/bin/env bash
# No 'set -e' — we want to fall back cleanly if anything fails.

sess=\"${SESSION_NAME}\"
info_file=\"${INFO_FILE}\"

if ! command -v tmux >/dev/null 2>&1; then
  clear
  echo '=== code-server connection info ==='
  if [ -f \"${INFO_FILE}\" ]; then cat \"${INFO_FILE}\"; else echo \"[warn] Info file not found: ${INFO_FILE}\"; fi
  echo
  echo '[warn] tmux is not installed on the compute node — staying in a single shell.'
  echo 'Tip: run `watch -n1 nvidia-smi &` in the background if you want a quick GPU monitor.'
  exec bash -l
fi

# Error out if the session already exists (your preference)
if tmux has-session -t \"\$sess\" 2>/dev/null; then
  echo \"[error] tmux session '\$sess' already exists. Aborting.\"
  exit 1
fi

# Left pane: print tunnel info once, then shell
tmux new-session -d -s \"\$sess\" \
  \"bash -lc 'clear; echo === code-server connection info ===; \
    if [ -f \\\"'\$info_file'\\\" ]; then cat \\\"'\$info_file'\\\"; else echo \\\"[warn] Info file not found: '\$info_file'\\\"; fi; \
    echo; exec bash -l'\"

# Right-top: nvidia-smi (sticky: Ctrl-C -> shell)
tmux split-window -h -t \"\$sess\" \
  \"bash -lc 'watch -n1 nvidia-smi; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'\"

# Right-bottom: squeue (sticky)
tmux split-window -v -t \"\$sess:0.1\" \
  \"bash -lc 'watch -n2 squeue -u sujinesh; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'\"

# Resize right-top to ~80% height
win=\"\$sess:0\"
right_top=\"\$sess:0.1\"
h=\$(tmux display-message -p -t \"\$win\" \"#{window_height}\")
top=\$(( h * 80 / 100 ))
if (( top < 5 )); then top=5; fi
tmux resize-pane -t \"\$right_top\" -y \"\$top\"

# Focus back to left pane
tmux select-pane -L -t \"\$sess\"

exec tmux attach -t \"\$sess\"
CS_EOF

  chmod +x /tmp/dev_setup_${jid}.sh
  exec /tmp/dev_setup_${jid}.sh
"
