#!/bin/bash
# =============================================================================
# HPC Requirements Verification Script
# =============================================================================
# This script verifies all dependencies and prerequisites before running
# experiments on the HPC cluster. Run this BEFORE submitting expensive GPU jobs.
#
# Usage: bash telepathy/verify_hpc_requirements.sh
# =============================================================================

set -e

echo "=============================================================="
echo "HPC Requirements Verification"
echo "Starting at: $(date)"
echo "=============================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_GOOD=true

# Function to check and report status
check_status() {
    local name="$1"
    local command="$2"
    local required="$3"

    echo -n "Checking $name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $required"
    else
        echo -e "${RED}✗${NC} Missing: $required"
        ALL_GOOD=false
    fi
}

# Function to check Python package
check_package() {
    local package="$1"
    local import_name="${2:-$1}"

    echo -n "  - $package: "
    if python -c "import $import_name" 2>/dev/null; then
        version=$(python -c "import $import_name; print(getattr($import_name, '__version__', 'installed'))" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✓${NC} ($version)"
    else
        echo -e "${RED}✗${NC} NOT INSTALLED"
        ALL_GOOD=false
    fi
}

echo ""
echo "1. System Information"
echo "---------------------"
echo "Hostname: $(hostname)"
echo "User: $USER"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo ""

echo "2. Python Environment"
echo "---------------------"
python_version=$(python --version 2>&1 || echo "Python not found")
echo "Python version: $python_version"

# Check Python version >= 3.8
if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo -e "Python 3.8+ check: ${GREEN}✓${NC}"
else
    echo -e "Python 3.8+ check: ${RED}✗${NC} (Found: $python_version)"
    ALL_GOOD=false
fi
echo ""

echo "3. Required Python Packages"
echo "---------------------------"
echo "Checking core packages:"
check_package "torch"
check_package "transformers"
check_package "datasets"
check_package "numpy"
check_package "tqdm"
check_package "scikit-learn" "sklearn"
check_package "scipy"
check_package "pandas"
check_package "matplotlib"

echo ""
echo "Checking additional packages:"
check_package "accelerate"
check_package "sentencepiece"
check_package "tiktoken"
check_package "einops"
check_package "bitsandbytes"
check_package "peft"
check_package "evaluate"
check_package "rouge_score"
check_package "nltk"

# Check PyTorch CUDA availability
echo ""
echo "4. GPU and CUDA Check"
echo "----------------------"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    cuda_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo -e "CUDA available: ${GREEN}✓${NC}"
    echo "  - Number of GPUs: $cuda_count"
    echo "  - CUDA version: $cuda_version"

    # List GPU names
    python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "  Could not get GPU names"
else
    echo -e "CUDA available: ${RED}✗${NC}"
    echo "  No CUDA GPUs detected! This will prevent training."
    ALL_GOOD=false
fi

# Check nvidia-smi
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi check: ✓"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -4 || true
else
    echo -e "nvidia-smi: ${YELLOW}Not available${NC}"
fi

echo ""
echo "5. HuggingFace Model Access"
echo "----------------------------"
# Check if we can import and access models
echo "Testing model access (this may take a moment)..."

# Test Llama access
echo -n "  - Llama-3.1-8B-Instruct: "
if python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token=None)
    print('✓ (accessible)')
except Exception as e:
    if 'token' in str(e).lower() or 'gated' in str(e).lower():
        print('⚠ (requires HF token)')
    else:
        print(f'✗ ({e})')
" 2>/dev/null; then
    true
else
    echo -e "${YELLOW}Check failed${NC}"
fi

# Test Qwen access
echo -n "  - Qwen2.5-7B-Instruct: "
if python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    print('✓ (accessible)')
except Exception as e:
    print(f'✗ ({e})')
" 2>/dev/null; then
    true
else
    echo -e "${YELLOW}Check failed${NC}"
fi

# Check HuggingFace cache directory
echo ""
echo "HuggingFace cache:"
hf_cache="${HF_HOME:-$HOME/.cache/huggingface}"
echo "  - Cache directory: $hf_cache"
if [ -d "$hf_cache" ]; then
    cache_size=$(du -sh "$hf_cache" 2>/dev/null | cut -f1 || echo "unknown")
    echo "  - Cache size: $cache_size"
else
    echo "  - Cache not found (models will be downloaded on first use)"
fi

echo ""
echo "6. Memory and Disk Space"
echo "-------------------------"
# Check available memory
if command -v free &> /dev/null; then
    total_mem=$(free -h | grep "^Mem:" | awk '{print $2}')
    avail_mem=$(free -h | grep "^Mem:" | awk '{print $7}')
    echo "System memory:"
    echo "  - Total: $total_mem"
    echo "  - Available: $avail_mem"
else
    echo "Memory check: command 'free' not available"
fi

# Check disk space in key directories
echo ""
echo "Disk space:"
for dir in "." "$HOME" "/tmp" "/projects/m000066/sujinesh" "/scratch"; do
    if [ -d "$dir" ]; then
        space_info=$(df -h "$dir" 2>/dev/null | tail -1 | awk '{print $4 " available, " $5 " used"}' || echo "unknown")
        echo "  - $dir: $space_info"
    fi
done

echo ""
echo "7. Environment Variables"
echo "------------------------"
echo "PYTHONPATH: ${PYTHONPATH:-not set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "HF_HOME: ${HF_HOME:-not set (using default)}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-not set}"

echo ""
echo "8. Git Repository Status"
echo "------------------------"
if [ -d ".git" ]; then
    echo "Git branch: $(git branch --show-current 2>/dev/null || echo "unknown")"
    echo "Last commit: $(git log -1 --oneline 2>/dev/null || echo "unknown")"

    # Check for uncommitted changes
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        echo -e "Working tree: ${GREEN}clean${NC}"
    else
        echo -e "Working tree: ${YELLOW}has uncommitted changes${NC}"
        echo "  Run 'git status' for details"
    fi
else
    echo -e "Git repository: ${RED}not found${NC}"
    echo "  Make sure you're in the LatentWire directory"
    ALL_GOOD=false
fi

echo ""
echo "9. Project Structure Check"
echo "--------------------------"
required_dirs=("latentwire" "telepathy" "scripts" "experimental")
required_files=("latentwire/train.py" "latentwire/eval.py" "latentwire/models.py")

echo "Checking required directories:"
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "  - $dir: ${GREEN}✓${NC}"
    else
        echo -e "  - $dir: ${RED}✗${NC}"
        ALL_GOOD=false
    fi
done

echo ""
echo "Checking required files:"
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  - $file: ${GREEN}✓${NC}"
    else
        echo -e "  - $file: ${RED}✗${NC}"
        ALL_GOOD=false
    fi
done

echo ""
echo "10. Quick Import Test"
echo "---------------------"
echo "Testing LatentWire imports..."
if python -c "
import sys
sys.path.insert(0, '.')
try:
    from latentwire.models import ByteLatentEncoder, LinearAdapter, LMWrapper
    from latentwire.train import train_epoch
    from latentwire.eval import eval_squad_em_f1
    print('  Core imports: ✓')
except ImportError as e:
    print(f'  Core imports: ✗ ({e})')
    exit(1)
" 2>&1; then
    true
else
    echo -e "  ${RED}Import test failed!${NC}"
    ALL_GOOD=false
fi

echo ""
echo "=============================================================="
echo "VERIFICATION SUMMARY"
echo "=============================================================="
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED!${NC}"
    echo "The environment is ready for running experiments."
    echo ""
    echo "Next steps:"
    echo "1. Submit your SLURM job: sbatch telepathy/your_experiment.slurm"
    echo "2. Monitor with: squeue -u \$USER"
    echo "3. Check logs in: runs/"
else
    echo -e "${RED}✗ SOME CHECKS FAILED${NC}"
    echo "Please fix the issues above before running experiments."
    echo ""
    echo "Common fixes:"
    echo "- Missing packages: pip install -r requirements.txt"
    echo "- No GPUs: Make sure you're on a GPU node (use srun or sbatch)"
    echo "- Model access: Set up HuggingFace token if needed"
    echo "- Wrong directory: cd to /projects/m000066/sujinesh/LatentWire"
fi
echo "=============================================================="

# Exit with appropriate code
if [ "$ALL_GOOD" = true ]; then
    exit 0
else
    exit 1
fi