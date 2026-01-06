# Import Testing Guide for LatentWire

This document explains how to test and fix import issues in the LatentWire project.

## Quick Start

### 1. Check File Structure (No Dependencies Required)
```bash
python3 test_imports.py --check-structure
```
This verifies all expected files exist without requiring PyTorch or other dependencies.

### 2. Set Up Environment
```bash
source setup_env.sh
```
This sets PYTHONPATH and checks for installed dependencies.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Full Import Test
```bash
python3 test_imports.py
```
This tests all imports and requires dependencies to be installed.

## Files Created/Modified

### New Files Created
- `test_imports.py` - Comprehensive import testing script
- `setup_env.sh` - Environment setup script
- `latentwire/metrics.py` - Metrics module for evaluation
- `latentwire/common.py` - Common utilities
- `latentwire/prefix_utils.py` - Prefix handling utilities
- `scripts/run_pipeline.sh` - Main pipeline script
- `requirements.txt` - Updated with all dependencies

### Key Features of test_imports.py

1. **Structure Check Mode** (`--check-structure`):
   - Verifies all expected files exist
   - Doesn't require any dependencies
   - Useful for CI/CD or initial setup

2. **Full Import Test** (default):
   - Tests actual Python imports
   - Identifies missing dependencies
   - Shows which modules fail and why
   - Provides fix recommendations

3. **Comprehensive Coverage**:
   - Core latentwire modules
   - Feature modules
   - CLI modules
   - Telepathy modules
   - Root-level scripts

## Common Issues and Solutions

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch
```

### Issue: "No module named 'transformers'"
**Solution**: Install Transformers
```bash
pip install transformers
```

### Issue: Import fails with "No module named 'latentwire'"
**Solution**: Set PYTHONPATH correctly
```bash
export PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/finalization:$PYTHONPATH
# Or use the setup script:
source setup_env.sh
```

### Issue: Missing __init__.py files
**Solution**: The test script checks for these. If missing, create empty __init__.py files:
```bash
touch latentwire/__init__.py
touch latentwire/features/__init__.py
touch latentwire/cli/__init__.py
touch telepathy/__init__.py
```

## Running on HPC

When running on HPC systems:

1. Load required modules:
```bash
module load python/3.9
module load cuda/12.1  # If using GPU
```

2. Set environment:
```bash
cd /projects/m000066/sujinesh/LatentWire
export PYTHONPATH=.
```

3. Run import test:
```bash
python test_imports.py
```

## Continuous Integration

For CI/CD pipelines, use the structure check:

```yaml
# .github/workflows/test.yml example
- name: Check file structure
  run: python3 test_imports.py --check-structure

- name: Install dependencies
  run: pip install -r requirements.txt

- name: Test imports
  run: |
    export PYTHONPATH=.
    python3 test_imports.py
```

## Summary

The import testing infrastructure ensures:
1. All required files exist
2. Python modules can be imported correctly
3. Dependencies are properly documented
4. PYTHONPATH configuration is correct
5. Clear error messages and fix recommendations

This helps maintain a working codebase and makes it easier to set up the project on new systems.