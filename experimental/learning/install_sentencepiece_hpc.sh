#!/usr/bin/env bash
# Install sentencepiece library on HPC
# Run this ONCE on HPC before running experiments

set -e

echo "========================================"
echo "Installing sentencepiece for HPC"
echo "========================================"
echo ""

# Check if sentencepiece is already installed
if python3 -c "import sentencepiece" 2>/dev/null; then
    echo "✓ sentencepiece is already installed"
    python3 -c "import sentencepiece; print(f'  Version: {sentencepiece.__version__}')"
    exit 0
fi

echo "Installing sentencepiece..."
echo ""

# Install sentencepiece
pip install --user sentencepiece

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""

# Verify installation
if python3 -c "import sentencepiece" 2>/dev/null; then
    echo "✓ sentencepiece successfully installed"
    python3 -c "import sentencepiece; print(f'  Version: {sentencepiece.__version__}')"
else
    echo "✗ Installation failed"
    exit 1
fi

echo ""
echo "You can now run the experiments:"
echo "  bash run_procrustes_ablation.sh"
echo "  bash run_learned_adapters.sh"
