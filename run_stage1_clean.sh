#!/bin/bash
# Convenience script to pull, clean, and run Stage 1 training
# Usage: ./run_stage1_clean.sh
set -euo pipefail

echo "=================================="
echo "STAGE 1 CLEAN RUN"
echo "=================================="
echo ""

# Pull latest code
echo "Pulling latest code..."
git pull
echo ""

# Clean previous runs
echo "Cleaning previous runs..."
rm -rf runs
echo ""

# Run Stage 1 with PYTHONPATH set
echo "Starting Stage 1 training..."
echo "=================================="
PYTHONPATH=. ./scripts/run_stage1_h100.sh