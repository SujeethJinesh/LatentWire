#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher: start/reuse code-server job, then
# create & attach a compute-side tmux session "dev" with:
#   left  : dev shell
#   right : split vertically => top=watch nvidia-smi (~80% height), bottom=watch squeue
#
# Notes:
# - Exactly one Slurm step is created (no overlap issues).
# - If tmux isn't available on the compute node, we error out (per your preference).

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

# 3) Open a single interactive step and build the tmux layout on the compute node
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

  # Create the session + panes
  tmux new-session -d -s "$sess" "bash -l"               # left pane = dev shell
  tmux split-window -h -t "$sess" \
    "bash -lc '\''watch -n1 nvidia-smi; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'\''"  # right-top initially
  tmux split-window -v -t "$sess:0.1" \
    "bash -lc '\''watch -n2 squeue -u sujinesh; echo; echo \"[watch exited] Dropping to shell...\"; exec bash -l'\''"  # right-bottom

  # Resize so right-top (nvidia-smi) is ~80% of window height
  # Get total window height and set the right-top pane height accordingly
  win="$sess:0"
  right_top="$sess:0.1"
  h=$(tmux display -p -t "$win" "#{window_height}")
  top=$(( h * 80 / 100 ))
  # Guard against tiny windows
  if (( top < 5 )); then top=5; fi
  tmux resize-pane -t "$right_top" -y "$top"

  # Focus back to left dev pane
  tmux select-pane -L -t "$sess"

  exec tmux attach -t "$sess"
'
