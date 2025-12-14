#!/usr/bin/env bash
# telepathy/run_trec_multiseed.sh
#
# FOCUSED SCRIPT: Run TREC Bridge multi-seed experiments
# ========================================================
#
# This is the ONLY remaining critical experiment for the paper.
# TREC shows extreme super-additivity (+41pp vs Llama) and needs
# multi-seed validation.
#
# Usage (from LatentWire root):
#   PYTHONPATH=. bash telepathy/run_trec_multiseed.sh
#
# Expected runtime: ~15 minutes on single H100
# ========================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Set PYTHONPATH to include parent directory (LatentWire root)
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Configuration - Use same output base as previous experiments for consistency
OUTPUT_BASE="${OUTPUT_BASE:-runs/paper_experiments_20251213_190318}"
SEEDS="42 123 456"
STEPS=2000

# Create output directory
mkdir -p "$OUTPUT_BASE/bridge_multiseed"

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_BASE/trec_multiseed_${TIMESTAMP}.log"

echo "=============================================="
echo "TREC BRIDGE MULTI-SEED EXPERIMENTS"
echo "=============================================="
echo "Output: $OUTPUT_BASE"
echo "Log: $LOG_FILE"
echo "Seeds: $SEEDS"
echo "=============================================="
echo ""

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

for SEED in $SEEDS; do
    echo ">>> TREC Bridge: seed=$SEED"

    OUTPUT_DIR="$OUTPUT_BASE/bridge_multiseed/trec_seed${SEED}"
    mkdir -p "$OUTPUT_DIR"

    python train_telepathy_trec.py \
        --soft_tokens 16 \
        --steps $STEPS \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" \
        --save_path "bridge_trec_seed${SEED}.pt" \
        --eval_every 400 \
        --save_every 500

    echo ">>> Done: TREC seed=$SEED"
    echo ""
done

# Compile results
echo ""
echo "=============================================="
echo "COMPILING TREC RESULTS"
echo "=============================================="

python << 'EOF'
import json
import glob
import numpy as np

output_base = "runs/paper_experiments_20251213_190318"
trec_files = glob.glob(f"{output_base}/bridge_multiseed/trec_seed*/trec_results.json")

if not trec_files:
    # Try alternate naming
    trec_files = glob.glob(f"{output_base}/bridge_multiseed/trec_seed*/*_results.json")

if trec_files:
    accs = []
    for f in trec_files:
        with open(f) as fp:
            data = json.load(fp)
        acc = data.get('final_results', {}).get('accuracy', 0)
        accs.append(acc)
        print(f"  {f}: {acc}%")

    if len(accs) >= 2:
        print(f"\nTREC Bridge: {np.mean(accs):.1f}% ± {np.std(accs):.1f}% (n={len(accs)})")
        print(f"Values: {accs}")
else:
    print("No TREC results found yet.")

print("\nUpdate paper table with these results!")
EOF

echo ""
echo "=============================================="
echo "TREC MULTI-SEED COMPLETE"
echo "=============================================="
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. git add & commit results"
echo "  2. Update telepathy.tex with TREC multi-seed numbers"
echo "  3. Recompile paper"
echo "=============================================="
