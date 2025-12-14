#!/usr/bin/env bash
# Generate all figures for Telepathy paper
# Usage: bash generate_all.sh

set -e

echo "Generating Telepathy paper figures..."
echo ""

# Generate matplotlib figures
echo "1. Generating data figures (matplotlib)..."
python3 generate_figures.py

# Generate architecture diagram (requires graphviz)
echo ""
echo "2. Generating architecture diagram (graphviz)..."
if command -v dot &> /dev/null; then
    dot -Tpdf architecture.dot -o architecture.pdf
    dot -Tsvg architecture.dot -o architecture.svg
    echo "   Saved: architecture.pdf, architecture.svg"
else
    echo "   WARNING: graphviz not installed. Install with:"
    echo "   brew install graphviz  # macOS"
    echo "   apt install graphviz   # Ubuntu"
fi

echo ""
echo "Done! Figures ready for LaTeX inclusion."
echo ""
echo "Add to LaTeX preamble:"
echo "  \\usepackage{graphicx}"
echo ""
echo "Include figures with:"
echo "  \\includegraphics[width=\\columnwidth]{figures/architecture.pdf}"
