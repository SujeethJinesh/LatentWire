#!/bin/bash
# Compile LaTeX paper


# =============================================================================
# LOGGING SETUP
# =============================================================================

# Ensure output directory exists
OUTPUT_DIR="${OUTPUT_DIR:-runs/compile_paper}"
mkdir -p "$OUTPUT_DIR"

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/compile_paper_${TIMESTAMP}.log"

echo "Starting compile_paper at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Wrapper function for logging commands
run_with_logging() {
    echo "Running: $*" | tee -a "$LOG_FILE"
    { "$@"; } 2>&1 | tee -a "$LOG_FILE"
    return ${PIPESTATUS[0]}
}

echo "Compiling paper.tex..."

# Run pdflatex twice for references
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

# Run bibtex if needed
# bibtex paper
# pdflatex -interaction=nonstopmode paper.tex
# pdflatex -interaction=nonstopmode paper.tex

echo "Done! PDF created: paper.pdf"
