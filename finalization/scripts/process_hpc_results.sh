#!/bin/bash
# =============================================================================
# Process HPC Results and Update Paper
# =============================================================================
# This script automates the full pipeline of:
# 1. Pulling latest results from git
# 2. Verifying experiment completion
# 3. Extracting and aggregating results
# 4. Generating LaTeX tables
# 5. Compiling the paper (optional)
#
# Usage:
#   bash finalization/scripts/process_hpc_results.sh
#   bash finalization/scripts/process_hpc_results.sh --skip-compile
#   bash finalization/scripts/process_hpc_results.sh --runs-dir runs/specific_experiment
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
RUNS_DIR="runs"
SKIP_COMPILE=false
BASELINE="mistral_zeroshot"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-compile)
            SKIP_COMPILE=true
            shift
            ;;
        --runs-dir)
            RUNS_DIR="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-compile    Skip LaTeX compilation"
            echo "  --runs-dir DIR    Specify runs directory (default: runs)"
            echo "  --baseline METHOD Baseline method for significance (default: mistral_zeroshot)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  HPC Results Processing Pipeline${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Runs directory: $RUNS_DIR"
echo "Baseline method: $BASELINE"
echo "Skip compile: $SKIP_COMPILE"
echo ""

# =============================================================================
# Step 1: Pull latest results
# =============================================================================
echo -e "${YELLOW}Step 1: Pulling latest results from git...${NC}"
git pull origin main || {
    echo -e "${RED}Warning: git pull failed. Continuing with local files.${NC}"
}
echo -e "${GREEN}Done${NC}"
echo ""

# =============================================================================
# Step 2: Verify experiment completion
# =============================================================================
echo -e "${YELLOW}Step 2: Verifying experiment completion...${NC}"
python finalization/scripts/verify_completion.py --runs_dir "$RUNS_DIR" --output finalization/verification_report.txt || {
    echo -e "${RED}Warning: Some experiments may have issues.${NC}"
    echo "Check finalization/verification_report.txt for details."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}
echo -e "${GREEN}Done${NC}"
echo ""

# =============================================================================
# Step 3: Extract and aggregate results
# =============================================================================
echo -e "${YELLOW}Step 3: Extracting and aggregating results...${NC}"
python finalization/scripts/extract_results.py \
    --runs_dir "$RUNS_DIR" \
    --baseline "$BASELINE" \
    --output finalization/extracted_results.json \
    --latex
echo -e "${GREEN}Done${NC}"
echo ""

# =============================================================================
# Step 4: Generate LaTeX tables
# =============================================================================
echo -e "${YELLOW}Step 4: Generating LaTeX tables...${NC}"
python finalization/scripts/generate_tables.py \
    --runs_dir "$RUNS_DIR" \
    --output_dir telepathy \
    --baseline "$BASELINE"
echo -e "${GREEN}Done${NC}"
echo ""

# =============================================================================
# Step 5: Compile paper (optional)
# =============================================================================
if [ "$SKIP_COMPILE" = false ]; then
    echo -e "${YELLOW}Step 5: Compiling LaTeX paper...${NC}"

    if [ -d "finalization/latex" ]; then
        cd finalization/latex

        # Check if pdflatex is available
        if command -v pdflatex &> /dev/null; then
            pdflatex -interaction=nonstopmode paper_master.tex > /dev/null 2>&1 || true
            bibtex paper_master > /dev/null 2>&1 || true
            pdflatex -interaction=nonstopmode paper_master.tex > /dev/null 2>&1 || true
            pdflatex -interaction=nonstopmode paper_master.tex > /dev/null 2>&1 || true

            # Check for errors
            if grep -qi "error" paper_master.log 2>/dev/null; then
                echo -e "${RED}Warning: LaTeX errors found. Check paper_master.log${NC}"
            else
                echo -e "${GREEN}Paper compiled successfully!${NC}"
            fi

            # Open PDF on macOS
            if [[ "$OSTYPE" == "darwin"* ]] && [ -f paper_master.pdf ]; then
                open paper_master.pdf
            fi
        else
            echo -e "${YELLOW}pdflatex not found. Skipping compilation.${NC}"
        fi

        cd "$PROJECT_ROOT"
    else
        echo -e "${YELLOW}LaTeX directory not found. Skipping compilation.${NC}"
    fi
else
    echo -e "${YELLOW}Step 5: Skipping LaTeX compilation (--skip-compile)${NC}"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Processing Complete!${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Output files:"
echo "  - finalization/verification_report.txt"
echo "  - finalization/extracted_results.json"
echo "  - telepathy/paper_tables.tex"
echo "  - telepathy/statistical_analysis_generated.tex"
if [ "$SKIP_COMPILE" = false ] && [ -f "finalization/latex/paper_master.pdf" ]; then
    echo "  - finalization/latex/paper_master.pdf"
fi
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review the generated tables in telepathy/paper_tables.tex"
echo "2. Check the verification report for any issues"
echo "3. Manually verify key numbers against the paper"
echo "4. Commit changes when satisfied"
echo ""
