#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher: start (or reuse) the code-server batch job,
# then open a bash shell on the compute node inside that allocation.

SBATCH_FILE="${SBATCH_FILE:-start_compute_node.sbatch}"
JOB_NAME="code-server"
POLL_SECS=5

# 1) Reuse a running job if it exists; otherwise submit a new one
jid="$(squeue -u "$USER" -h -t R -n "$JOB_NAME" -o %A | tail -n1 || true)"
if [[ -z "${jid}" ]]; then
  echo "No running ${JOB_NAME} job found; submitting ${SBATCH_FILE}…"
  out="$(sbatch "$SBATCH_FILE")"
  jid="$(awk '{print $4}' <<<"$out")"
  echo "Submitted ${JOB_NAME} job: JID=${jid}"
fi

# 2) Wait until the allocation is RUNNING and we have a node name
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

# 3) Open an interactive shell on that node within the job allocation
echo "Opening shell on ${node} (job ${jid})…"
exec srun --jobid="${jid}" -w "${node}" --pty bash -l
