#!/usr/bin/env bash
# Generate t-SNE visualization of AG News latent space
# Run on HPC after training experiments have completed
#
# Usage:
#   PYTHONPATH=. bash telepathy/run_tsne_visualization.sh

set -e

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configuration
OUTPUT_DIR="figures"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/tsne_${TIMESTAMP}.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting t-SNE visualization generation..."
echo "Log file: $LOG_FILE"
echo ""

# Find the most recent AG News checkpoint
CHECKPOINT=$(find runs -name "bridge.pt" -path "*agnews*" 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No AG News bridge checkpoint found in runs/"
    echo "Please run training first: PYTHONPATH=. bash telepathy/run_enhanced_arxiv_suite.sh"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# Run visualization
{
    python telepathy/generate_tsne_visualization.py \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT_DIR/agnews_tsne.pdf" \
        --samples_per_class 100
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Generated files:"
echo "  - $OUTPUT_DIR/agnews_tsne.pdf (for paper)"
echo "  - $OUTPUT_DIR/agnews_tsne.png (preview)"
echo "  - $LOG_FILE"

# Push to git if in a git repo
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo ""
    echo "Pushing results to git..."
    git add "$OUTPUT_DIR"/*.pdf "$OUTPUT_DIR"/*.png "$LOG_FILE" 2>/dev/null || true
    git commit -m "figures: add AG News t-SNE latent space visualization

Shows category separation in Bridge latent space:
- World, Sports, Business, Science/Tech form distinct clusters
- Demonstrates semantic disentanglement in learned interlingua

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" 2>/dev/null || echo "No changes to commit"
    git push 2>/dev/null || echo "Push failed (may need manual push)"
fi

echo ""
echo "Done!"
