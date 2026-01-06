#!/bin/bash
# SLURM-safe wrapper that ensures logs are not lost

# Critical: Disable all buffering
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Ensure output is line-buffered
stdbuf -oL -eL "$@" 2>&1 | tee -a "${SLURM_LOG:-slurm_output.log}"
