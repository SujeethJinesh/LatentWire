#!/usr/bin/env bash
set -euo pipefail

# Config (edit if needed)
ACCOUNT="${ACCOUNT:-marlowe-m000066}"
PARTITION="${PARTITION:-preempt}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-40G}"
GPUS="${GPUS:-4}"

echo "Requesting 1 node, ${GPUS} GPU(s), ${MEM}, ${TIME} on ${PARTITION}..."
exec srun \
  -N 1 \
  -p "${PARTITION}" \
  --gpus="${GPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --account="${ACCOUNT}" \
  --job-name=dev-shell \
  --pty bash -l
