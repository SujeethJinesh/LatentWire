#!/usr/bin/env python3
"""
Validate that all required dependencies are available for telepathy system.
"""

import sys
from pathlib import Path

def check_imports():
    """Check that all required imports work."""
    print("Checking imports...")

    required_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("rouge_score", "ROUGE Score"),
        ("scipy", "SciPy"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM"),
        ("wandb", "Weights & Biases (optional)"),
    ]

    all_good = True
    for module_name, display_name in required_imports:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            if "wandb" in module_name:
                print(f"  ⚠ {display_name} (optional): {e}")
            else:
                print(f"  ✗ {display_name}: {e}")
                all_good = False

    return all_good


def check_project_modules():
    """Check that project modules can be imported."""
    print("\nChecking project modules...")

    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent))

    project_modules = [
        ("telepathy.train_bridge", "Bridge Training"),
        ("telepathy.train_linear_probe", "Linear Probe"),
        ("telepathy.eval_rouge", "ROUGE Evaluation"),
        ("telepathy.data_utils", "Data Utils"),
        ("scripts.statistical_testing", "Statistical Testing"),
    ]

    all_good = True
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_good = False

    return all_good


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ CUDA available: {device_name} ({memory_gb:.1f} GB)")
            return True
        elif torch.backends.mps.is_available():
            print(f"  ✓ MPS (Apple Silicon) available")
            return True
        else:
            print(f"  ⚠ No GPU available - will use CPU (slower)")
            return True
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False


def check_models():
    """Check that we can load required models."""
    print("\nChecking model access...")

    from transformers import AutoTokenizer

    models_to_check = [
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen2.5-0.5B",
    ]

    all_good = True
    for model_name in models_to_check:
        try:
            # Just try to load tokenizer (lightweight check)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"  ✓ {model_name}")
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")
            all_good = False

    return all_good


def main():
    """Run all validation checks."""
    print("="*60)
    print(" TELEPATHY DEPENDENCY VALIDATION")
    print("="*60)

    checks = [
        ("imports", check_imports),
        ("project", check_project_modules),
        ("gpu", check_gpu),
        ("models", check_models),
    ]

    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()

    print("\n" + "="*60)
    if all(results.values()):
        print(" ✅ ALL CHECKS PASSED")
        print("="*60)
        print("\nSystem is ready for integration testing!")
        print("Run: bash telepathy/run_final_integration_test.sh")
        return 0
    else:
        print(" ❌ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running integration tests.")
        failed = [name for name, passed in results.items() if not passed]
        print(f"Failed checks: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())