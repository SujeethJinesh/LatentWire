#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher: ensure the code-server job exists,
# then open ONE interactive shell on the compute node.

SBATCH_FILE="${SBATCH_FILE:-start_compute_node.sbatch}"
JOB_NAME="${JOB_NAME:-code-server}"
POLL_SECS="${POLL_SECS:-5}"

# 1) Reuse a running job or submit a new one
jid="$(squeue -u "$USER" -h -t R -n "$JOB_NAME" -o %A | tail -n1 || true)"
if [[ -z "${jid}" ]]; then
  echo "No running ${JOB_NAME} job found; submitting ${SBATCH_FILE}..."
  out="$(sbatch "$SBATCH_FILE")"
  jid="$(awk '{print $4}' <<<"$out")"
  echo "Submitted ${JOB_NAME} job: JID=${jid}"
fi

# 2) Wait for allocation + node
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

# 3) Open a plain bash shell on that node inside the allocation
# Prefer --overlap if supported; else fall back to --oversubscribe.
if srun --help 2>&1 | grep -q -- '--overlap'; then
  OVL="--overlap"
else
  OVL="--oversubscribe"
fi

echo "Opening bash on ${node} (job ${jid})..."
exec srun --jobid="${jid}" -w "${node}" ${OVL} -c1 --cpu-bind=none --pty bash -l
