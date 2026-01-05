#!/usr/bin/env bash

# Run comprehensive validation before experiments
# This prevents wasted GPU time from missing dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "LatentWire Pre-flight Validation"
echo "=================================="
echo ""

# Set required environment variables
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Run validation
echo "Running validation checks..."
echo ""

python3 finalization/validate.py "$@"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Validation passed! Ready to run experiments.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. For local development:"
    echo "     bash scripts/run_pipeline.sh"
    echo ""
    echo "  2. For HPC training:"
    echo "     sbatch telepathy/submit_enhanced_arxiv.slurm"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Validation failed. Please fix issues before running.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Missing packages: pip install -r requirements.txt"
    echo "  - No GPU: Run on HPC cluster or GPU-enabled machine"
    echo "  - Git issues: git add -A && git commit -m 'checkpoint'"
    echo "  - Disk space: Clear old runs/ directory"
    echo ""
    exit 1
fi