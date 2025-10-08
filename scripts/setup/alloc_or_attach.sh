#!/usr/bin/env bash
set -euo pipefail

# Durable 12h / 4-GPU allocation you can reattach to at any time.
# If the tmux session exists -> attach; else create it with salloc and drop into a compute-node shell.

SESSION="${SESSION:-dev}"                 # tmux session name to (re)attach
ACCOUNT="${ACCOUNT:-marlowe-m000066}"
PARTITION="${PARTITION:-preempt}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-40G}"
GPUS="${GPUS:-4}"
JOB_NAME="${JOB_NAME:-dev-12h}"

# Ensure tmux is available
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux not found on the login node. Install it or 'module load tmux' first." >&2
  exit 1
fi

# If a session already exists, just attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

# Create a tmux session that:
# 1) requests the allocation with salloc
# 2) once granted, starts a compute-node shell with srun --pty bash -l
tmux new-session -d -s "$SESSION" "bash -lc '
  set -e
  echo \"[alloc] Requesting 1 node, ${GPUS} GPU(s), ${MEM}, ${TIME} on ${PARTITION}...\"

  # Disable NVIDIA MPS to prevent Error 805
  export CUDA_MPS_PIPE_DIRECTORY=\"\"
  export CUDA_MPS_LOG_DIRECTORY=\"\"
  export CUDA_VISIBLE_DEVICES_MPS_CLIENT=0
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=0
  export CUDA_DEVICE_MAX_CONNECTIONS=1

  salloc -N 1 -p \"${PARTITION}\" --gpus=\"${GPUS}\" --mem=\"${MEM}\" --time=\"${TIME}\" \
         --account=\"${ACCOUNT}\" --job-name=\"${JOB_NAME}\" \
         srun --pty bash -l
'"

tmux display-message -t "$SESSION" "Allocation starting… (session: $SESSION). Re-attach anytime: tmux attach -t $SESSION"
exec tmux attach -t "$SESSION"
