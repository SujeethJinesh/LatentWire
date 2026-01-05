#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment checker for LatentWire.
Verifies all dependencies and system requirements.
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  [WARNING] Python 3.7+ recommended")
        return False
    print("  [OK] Python version meets requirements")
    return True

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  [OK] Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("  [INFO] CUDA not available (CPU only)")

        # Check MPS availability (Mac)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  [OK] MPS (Mac GPU) available")

        return True
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        print("  [WARNING] PyTorch required for training/evaluation")
        return False

def check_transformers():
    """Check Transformers library."""
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print("  [OK] Transformers installed")
        return True
    except ImportError:
        print("Transformers: NOT INSTALLED")
        print("  [WARNING] Transformers required for LLM loading")
        return False

def check_datasets():
    """Check datasets library."""
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        print("  [OK] Datasets installed")
        return True
    except ImportError:
        print("Datasets: NOT INSTALLED")
        print("  [WARNING] Datasets required for data loading")
        return False

def check_git():
    """Check git installation and repo status."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Git: {version}")

            # Check if in repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("  [OK] In git repository")

                # Get branch
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  [OK] Current branch: {result.stdout.strip()}")
            else:
                print("  [WARNING] Not in git repository")

            return True
    except:
        pass

    print("Git: NOT INSTALLED or not accessible")
    print("  [WARNING] Git required for version control")
    return False

def check_disk_space():
    """Check available disk space."""
    import shutil

    path = Path.cwd()
    stat = shutil.disk_usage(path)

    # Convert to GB
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)

    print(f"Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

    if free_gb < 10:
        print("  [WARNING] Less than 10GB free space")
        return False
    else:
        print("  [OK] Sufficient disk space")
        return True

def check_memory():
    """Check system memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        print(f"Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")

        if total_gb < 16:
            print("  [WARNING] Less than 16GB RAM may cause issues")
            return False
        else:
            print("  [OK] Sufficient memory")
            return True
    except ImportError:
        print("Memory: Cannot check (psutil not installed)")
        return True

def check_environment_variables():
    """Check important environment variables."""
    print("Environment variables:")

    # PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        print(f"  PYTHONPATH: {pythonpath}")
    else:
        print("  PYTHONPATH: Not set (will be set by scripts)")

    # CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devices:
        print(f"  CUDA_VISIBLE_DEVICES: {cuda_devices}")
    else:
        print("  CUDA_VISIBLE_DEVICES: Not set (all GPUs visible)")

    # MPS fallback for Mac
    mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if mps_fallback:
        print(f"  PYTORCH_ENABLE_MPS_FALLBACK: {mps_fallback}")

    return True

def check_project_structure():
    """Check that key project files exist."""
    print("Project structure:")

    required_files = [
        "latentwire/train.py",
        "latentwire/eval.py",
        "latentwire/models.py",
        "latentwire/data.py",
        "telepathy/orchestrator.py",
        "finalization/test_all.py"
    ]

    all_exist = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"  [OK] {filepath}")
        else:
            print(f"  [MISSING] {filepath}")
            all_exist = False

    return all_exist

def main():
    """Run all environment checks."""
    print("="*60)
    print("LATENTWIRE ENVIRONMENT CHECK")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print("="*60)
    print()

    checks = [
        ("Python", check_python),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("Datasets", check_datasets),
        ("Git", check_git),
        ("Disk Space", check_disk_space),
        ("Memory", check_memory),
        ("Environment", check_environment_variables),
        ("Project Files", check_project_structure)
    ]

    results = {}
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  [ERROR] Check failed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All environment checks passed!")
        print("System is ready for LatentWire experiments.")
        return 0
    else:
        print("\n[WARNING] Some environment checks failed.")
        print("The system may still work, but some features may be limited.")
        print("\nFailed checks:")
        for name, result in results.items():
            if not result:
                print(f"  - {name}")
        return 1

if __name__ == "__main__":
    sys.exit(main())