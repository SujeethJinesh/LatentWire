#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher: start/reuse code-server job, then
# create & attach a compute-side tmux session "dev" with:
#   left       : dev shell (prints code-server tunnel info first)
#   right-top  : watch -n1 nvidia-smi  (~80% height, sticky)
#   right-bot  : watch -n2 squeue -u sujinesh (sticky)

SBATCH_FILE="${SBATCH_FILE:-start_compute_node.sbatch}"
JOB_NAME="code-server"
POLL_SECS=5
SESSION_NAME="${SESSION_NAME:-dev}"

# 1) Reuse running job or submit a new one
jid="$(squeue -u "$USER" -h -t R -n "$JOB_NAME" -o %A | tail -n1 || true)"
if [[ -z "${jid}" ]]; then
  echo "No running ${JOB_NAME} job found; submitting ${SBATCH_FILE}…"
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

# 3) One interactive step -> build tmux layout on compute node
echo "Opening compute session '${SESSION_NAME}' on ${node} (job ${jid})…"
exec srun --jobid="${jid}" -w "${node}" --pty bash -lc '
  set -euo pipefail
  (module load tmux 2>/dev/null || true) || true

  if ! command -v tmux >/dev/null 2>&1; then
    echo "[error] tmux not found on compute node. Aborting."
    exit 1
  fi

  sess="'"${SESSION_NAME}"'"
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "[error] tmux session '\''"'"${SESSION_NAME}"''\'' already exists. Aborting."
    exit 1
  fi

  # Path to the info file produced by the batch job
  INFO_FILE="/projects/m000066/sujinesh/LatentWire/code-server-${SLURM_JOB_ID}-info.txt"

  # Left pane: print tunnel instructions, then drop to a login shell
  tmux new-session -d -s "$sess" \
    "bash -lc 'clear; echo \"=== code-server connection info ===\"; if [ -f ${INFO_FILE} ]; then cat ${INFO_FILE}; else echo \"[warn] Info file not found: ${INFO_FILE}\"; fi; echo; echo \"(Tip) Create tunnel on your laptop, then open http://localhost:\$(awk \\\"/^Port:/{print \\$2}\\\" ${INFO_FILE} 2>/dev/null)\"; echo; exec bash -l'"

  # Right-top: GPU watch (sticky: Ctrl-C -> shell)
  tmux split-window -h -t "$sess" \
    "bash -lc 'watch -n1 nvidia-smi; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'"

  # Right-bottom: squeue watch (sticky)
  tmux split-window -v -t "$sess:0.1" \
    "bash -lc 'watch -n2 squeue -u sujinesh; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'"

  # Resize: make right-top ~80% of total window height
  win=\"$sess:0\"
  right_top=\"$sess:0.1\"
  h=\$(tmux display -p -t \"\$win\" \"#{window_height}\")
  top=\$(( h * 80 / 100 ))
  if (( top < 5 )); then top=5; fi
  tmux resize-pane -t \"\$right_top\" -y \"\$top\"

  # Focus back to left dev pane
  tmux select-pane -L -t \"$sess\"

  exec tmux attach -t \"$sess\"
'
