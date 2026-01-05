#!/bin/bash
# Compile LaTeX paper

echo "Compiling paper.tex..."

# Run pdflatex twice for references
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

# Run bibtex if needed
# bibtex paper
# pdflatex -interaction=nonstopmode paper.tex
# pdflatex -interaction=nonstopmode paper.tex

echo "Done! PDF created: paper.pdf"
