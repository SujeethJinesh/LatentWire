#!/usr/bin/env bash
set -e

# ==============================================================================
# Production Readiness Validation Script for LatentWire
# ==============================================================================
# This script performs comprehensive validation to ensure the codebase is
# ready for HPC deployment with SLURM
# ==============================================================================

echo "=============================================================="
echo "LatentWire Production Readiness Validation"
echo "Started at: $(date)"
echo "=============================================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-runs/validation}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${OUTPUT_DIR}/validation_report_${TIMESTAMP}.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Redirect output to both console and report file
exec > >(tee -a "$REPORT_FILE")
exec 2>&1

cd "$PROJECT_DIR"

echo ""
echo "Project directory: $PROJECT_DIR"
echo "Report file: $REPORT_FILE"
echo ""

# =============================================================================
# 1. CHECK PYTHON ENVIRONMENT
# =============================================================================
echo "1. PYTHON ENVIRONMENT CHECK"
echo "----------------------------"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"

# Check if in virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: Not in a virtual environment"
fi

# =============================================================================
# 2. CHECK DEPENDENCIES
# =============================================================================
echo ""
echo "2. DEPENDENCY CHECK"
echo "-------------------"

python3 << 'EOF'
import sys
import importlib

required_modules = [
    ('torch', '2.2.0'),
    ('transformers', '4.45.0'),
    ('datasets', '4.0.0'),
    ('accelerate', '1.0.0'),
    ('numpy', '1.21.0'),
    ('scipy', '1.7.0'),
    ('sklearn', '1.0.0'),
    ('rouge_score', '0.1.0'),
    ('statsmodels', '0.13.0'),
    ('pandas', '1.3.0')
]

missing = []
outdated = []

for module_name, min_version in required_modules:
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, '__version__'):
            version = mod.__version__
            # Simple version comparison (not perfect but good enough)
            if version < min_version:
                outdated.append((module_name, version, min_version))
            print(f"✅ {module_name:20s} {version}")
        else:
            print(f"✅ {module_name:20s} (version unknown)")
    except ImportError:
        missing.append(module_name)
        print(f"❌ {module_name:20s} MISSING")

if missing:
    print(f"\n❌ Missing modules: {missing}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

if outdated:
    print(f"\n⚠️  Outdated modules:")
    for name, current, required in outdated:
        print(f"  {name}: {current} < {required}")

print("\n✅ Dependency check passed")
EOF

# =============================================================================
# 3. CHECK LATENTWIRE MODULE IMPORTS
# =============================================================================
echo ""
echo "3. MODULE IMPORT CHECK"
echo "----------------------"

python3 << 'EOF'
import sys
import traceback

modules_to_test = [
    'latentwire.train',
    'latentwire.eval',
    'latentwire.models',
    'latentwire.data',
    'latentwire.core_utils',
    'latentwire.checkpointing',
    'latentwire.data_pipeline',
    'latentwire.feature_registry',
    'latentwire.loss_bundles',
    'latentwire.losses',
    'latentwire.optimized_dataloader'
]

failed = []
for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ {module}")
    except Exception as e:
        failed.append((module, str(e)))
        print(f"❌ {module}: {e}")

if failed:
    print("\n❌ Import failures detected")
    for module, error in failed:
        print(f"  {module}: {error}")
    sys.exit(1)
else:
    print("\n✅ All modules imported successfully")
EOF

# =============================================================================
# 4. CHECK SLURM SCRIPTS
# =============================================================================
echo ""
echo "4. SLURM SCRIPT CHECK"
echo "---------------------"

SLURM_ISSUES=0

for script in telepathy/*.slurm; do
    if [ -f "$script" ]; then
        # Check for required SLURM directives
        if ! grep -q "#SBATCH --account=marlowe-m000066" "$script"; then
            echo "⚠️  $script: Missing correct account directive"
            SLURM_ISSUES=$((SLURM_ISSUES + 1))
        fi

        if ! grep -q "#SBATCH --partition=preempt" "$script"; then
            echo "⚠️  $script: Missing correct partition directive"
            SLURM_ISSUES=$((SLURM_ISSUES + 1))
        fi

        if ! grep -q "WORK_DIR=\"/projects/m000066/sujinesh/LatentWire\"" "$script"; then
            echo "⚠️  $script: Incorrect working directory"
            SLURM_ISSUES=$((SLURM_ISSUES + 1))
        fi

        if [ $SLURM_ISSUES -eq 0 ]; then
            echo "✅ $(basename $script)"
        fi
    fi
done

if [ $SLURM_ISSUES -gt 0 ]; then
    echo "⚠️  Found $SLURM_ISSUES issues in SLURM scripts"
else
    echo "✅ All SLURM scripts configured correctly"
fi

# =============================================================================
# 5. CHECK EXECUTABLE PERMISSIONS
# =============================================================================
echo ""
echo "5. FILE PERMISSION CHECK"
echo "------------------------"

PERMISSION_ISSUES=0

# Check shell scripts
for script in scripts/*.sh; do
    if [ -f "$script" ] && [ ! -x "$script" ]; then
        echo "⚠️  $script is not executable"
        PERMISSION_ISSUES=$((PERMISSION_ISSUES + 1))
    fi
done

# Check SLURM scripts
for script in telepathy/*.slurm; do
    if [ -f "$script" ] && [ ! -x "$script" ]; then
        echo "⚠️  $script is not executable"
        PERMISSION_ISSUES=$((PERMISSION_ISSUES + 1))
    fi
done

if [ $PERMISSION_ISSUES -gt 0 ]; then
    echo "⚠️  Found $PERMISSION_ISSUES permission issues"
    echo "Fix with: chmod +x scripts/*.sh telepathy/*.slurm"
else
    echo "✅ All scripts have correct permissions"
fi

# =============================================================================
# 6. CHECK FOR HARDCODED PATHS
# =============================================================================
echo ""
echo "6. HARDCODED PATH CHECK"
echo "-----------------------"

# Check for problematic hardcoded paths
HARDCODED_ISSUES=0

# Look for home directory paths (should use /projects instead)
if grep -r "/home/sjinesh" latentwire/ scripts/ telepathy/ 2>/dev/null | grep -v ".git" | grep -v "__pycache__"; then
    echo "⚠️  Found hardcoded /home paths (should use /projects)"
    HARDCODED_ISSUES=$((HARDCODED_ISSUES + 1))
fi

# Look for local Mac paths
if grep -r "/Users/" latentwire/ scripts/ telepathy/ 2>/dev/null | grep -v ".git" | grep -v "__pycache__" | grep -v "# Example" | grep -v "# Local"; then
    echo "⚠️  Found hardcoded Mac paths"
    HARDCODED_ISSUES=$((HARDCODED_ISSUES + 1))
fi

if [ $HARDCODED_ISSUES -eq 0 ]; then
    echo "✅ No problematic hardcoded paths found"
fi

# =============================================================================
# 7. DATA LOADING TEST
# =============================================================================
echo ""
echo "7. DATA LOADING TEST"
echo "--------------------"

python3 << 'EOF'
from latentwire.data import load_examples

test_datasets = {
    'squad': 2,
    'hotpotqa': 2,
    'agnews': 2,
    'sst2': 2,
    'gsm8k': 2
}

failed = []
for dataset, num_samples in test_datasets.items():
    try:
        examples = load_examples(dataset, limit=num_samples)
        if len(examples) > 0:
            print(f"✅ {dataset:15s} - Loaded {len(examples)} examples")
        else:
            print(f"⚠️  {dataset:15s} - No examples loaded")
    except Exception as e:
        failed.append((dataset, str(e)))
        print(f"❌ {dataset:15s} - {e}")

if failed:
    print("\n⚠️  Some datasets failed to load")
else:
    print("\n✅ All datasets loaded successfully")
EOF

# =============================================================================
# 8. CHECK FOR COMMON ISSUES
# =============================================================================
echo ""
echo "8. COMMON ISSUES CHECK"
echo "-----------------------"

# Check for missing __init__.py files
MISSING_INIT=0
for dir in latentwire telepathy scripts; do
    if [ -d "$dir" ]; then
        find "$dir" -type d -name "__pycache__" -prune -o -type d -print | while read subdir; do
            if [ "$subdir" != "$dir" ] && [ ! -f "$subdir/__init__.py" ] && ls "$subdir"/*.py 2>/dev/null | grep -q .; then
                echo "⚠️  Missing __init__.py in $subdir"
                MISSING_INIT=$((MISSING_INIT + 1))
            fi
        done
    fi
done

# Check for uncommitted changes
if [ -d .git ]; then
    if ! git diff --quiet; then
        echo "⚠️  Uncommitted changes detected"
        git status --short
    else
        echo "✅ No uncommitted changes"
    fi
fi

# =============================================================================
# 9. GPU/CUDA CHECK (if available)
# =============================================================================
echo ""
echo "9. GPU/CUDA CHECK"
echo "-----------------"

python3 << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
else:
    print("ℹ️  No GPUs detected (expected on local Mac)")
EOF

# =============================================================================
# 10. QUICK SYNTAX CHECK
# =============================================================================
echo ""
echo "10. SYNTAX CHECK"
echo "----------------"

SYNTAX_ERRORS=0
for py_file in latentwire/*.py scripts/*.py telepathy/*.py; do
    if [ -f "$py_file" ]; then
        if ! python3 -m py_compile "$py_file" 2>/dev/null; then
            echo "❌ Syntax error in $py_file"
            SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        fi
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo "✅ No syntax errors found"
else
    echo "❌ Found $SYNTAX_ERRORS files with syntax errors"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "=============================================================="
echo "VALIDATION SUMMARY"
echo "=============================================================="

ISSUES_FOUND=false

if [ $SLURM_ISSUES -gt 0 ]; then
    echo "⚠️  SLURM configuration issues: $SLURM_ISSUES"
    ISSUES_FOUND=true
fi

if [ $PERMISSION_ISSUES -gt 0 ]; then
    echo "⚠️  Permission issues: $PERMISSION_ISSUES"
    ISSUES_FOUND=true
fi

if [ $HARDCODED_ISSUES -gt 0 ]; then
    echo "⚠️  Hardcoded path issues: $HARDCODED_ISSUES"
    ISSUES_FOUND=true
fi

if [ $SYNTAX_ERRORS -gt 0 ]; then
    echo "❌ Syntax errors: $SYNTAX_ERRORS"
    ISSUES_FOUND=true
fi

if [ "$ISSUES_FOUND" = false ]; then
    echo ""
    echo "✅ SYSTEM IS PRODUCTION READY"
    echo ""
    echo "Next steps for HPC deployment:"
    echo "1. Push code to git: git add -A && git commit -m 'Production ready' && git push"
    echo "2. On HPC, pull latest: cd /projects/m000066/sujinesh/LatentWire && git pull"
    echo "3. Submit job: sbatch telepathy/submit_production_readiness.slurm"
    echo "4. Monitor: squeue -u \$USER"
else
    echo ""
    echo "⚠️  ISSUES DETECTED - Review output above"
    echo ""
    echo "Common fixes:"
    echo "- Install dependencies: pip install -r requirements.txt"
    echo "- Fix permissions: chmod +x scripts/*.sh telepathy/*.slurm"
    echo "- Update SLURM scripts with correct account/partition"
fi

echo ""
echo "Full report saved to: $REPORT_FILE"
echo "Completed at: $(date)"
echo "=============================================================="