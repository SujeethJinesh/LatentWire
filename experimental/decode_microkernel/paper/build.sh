#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TECTONIC_CACHE_DIR="${TECTONIC_CACHE_DIR:-/workspace/tectonic_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
mkdir -p "$TECTONIC_CACHE_DIR" "$XDG_CACHE_HOME"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode decode_microkernel_colm2026.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode decode_microkernel_colm2026.tex
  pdflatex -interaction=nonstopmode decode_microkernel_colm2026.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic decode_microkernel_colm2026.tex
elif [ -x /workspace/bin/tectonic ]; then
  /workspace/bin/tectonic decode_microkernel_colm2026.tex
else
  echo "No LaTeX engine found: install latexmk, pdflatex, or tectonic under /workspace before building." >&2
  exit 2
fi
