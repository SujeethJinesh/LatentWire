#!/usr/bin/env python3
"""
Dependency checker for LatentWire project.
Verifies all required packages are installed with correct versions.
"""

import sys
import importlib
import subprocess
from typing import List, Tuple, Optional

def check_package(package_name: str, min_version: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a package is installed and meets version requirements."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")

        if min_version and version != "unknown":
            # Simple version comparison (may need enhancement for complex cases)
            installed = tuple(map(int, version.split('.')[:2]))
            required = tuple(map(int, min_version.split('.')[:2]))
            if installed < required:
                return False, f"✗ {package_name}: {version} (needs >={min_version})"

        return True, f"✓ {package_name}: {version}"
    except ImportError:
        return False, f"✗ {package_name}: NOT INSTALLED"

def main():
    """Check all required dependencies."""
    print("=" * 60)
    print("LatentWire Dependency Checker")
    print("=" * 60)

    # Core dependencies with minimum versions
    core_deps = [
        ("torch", "2.2.0"),
        ("transformers", "4.45.2"),  # Exact version for compatibility
        ("datasets", "4.0.0"),
        ("accelerate", "1.10.0"),
        ("numpy", "1.21.0"),
        ("scipy", "1.7.0"),
        ("sklearn", "1.0.0"),
        ("tqdm", None),
        ("sentencepiece", "0.1.99"),
        ("tokenizers", "0.13.0"),
    ]

    # Analysis and visualization
    analysis_deps = [
        ("statsmodels", "0.13.0"),
        ("pandas", "1.3.0"),
        ("matplotlib", "3.3.0"),
        ("seaborn", "0.11.0"),
    ]

    # Optional dependencies
    optional_deps = [
        ("peft", "0.5.0"),
        ("rouge_score", "0.1.2"),
        ("llmlingua", None),  # Optional, no minimum version
    ]

    all_good = True

    print("\nCore Dependencies:")
    print("-" * 40)
    for package, min_ver in core_deps:
        success, msg = check_package(package, min_ver)
        print(msg)
        if not success:
            all_good = False

    print("\nAnalysis & Visualization:")
    print("-" * 40)
    for package, min_ver in analysis_deps:
        success, msg = check_package(package, min_ver)
        print(msg)
        if not success:
            all_good = False

    print("\nOptional Dependencies:")
    print("-" * 40)
    for package, min_ver in optional_deps:
        success, msg = check_package(package, min_ver)
        print(msg)
        # Optional deps don't affect all_good status

    # Check PyTorch GPU availability
    print("\nPyTorch Configuration:")
    print("-" * 40)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        elif torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available: Yes")
        else:
            print("No GPU acceleration available")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

    # Check transformers model support
    print("\nTransformers Configuration:")
    print("-" * 40)
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")

        # Check if key models can be loaded (metadata only)
        test_models = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct"
        ]
        for model_id in test_models:
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                print(f"✓ {model_id}: Config loadable")
            except Exception as e:
                print(f"✗ {model_id}: {str(e)[:50]}...")
    except Exception as e:
        print(f"Error checking transformers: {e}")

    print("\n" + "=" * 60)
    if all_good:
        print("✓ All core dependencies satisfied!")
        print("You can run: pip install -r requirements.txt")
    else:
        print("✗ Some dependencies are missing or outdated")
        print("Please run: pip install -r requirements.txt")
    print("=" * 60)

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())