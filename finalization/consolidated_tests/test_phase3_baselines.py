#!/usr/bin/env python3
"""
Test Phase 3: Baseline Comparisons

This script verifies that all baseline comparison methods are available and functional:
1. LLMLingua baseline (prompt compression)
2. Token-budget baseline (simple truncation)
3. LinearProbeBaseline (direct classification from embeddings)

Each test runs with minimal data to ensure the code works without requiring
extensive computation.
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all baseline modules can be imported."""
    print("Testing imports...")
    results = {}

    # Test LLMLingua baseline import
    try:
        from latentwire.llmlingua_baseline import LLMLinguaCompressor, evaluate_llmlingua_on_qa
        results['llmlingua_import'] = "[PASS] LLMLingua baseline imported successfully"
    except ImportError as e:
        results['llmlingua_import'] = f"[FAIL] LLMLingua import failed: {e}"
    except Exception as e:
        results['llmlingua_import'] = f"[FAIL] Unexpected error: {e}"

    # Test LinearProbeBaseline import
    try:
        from latentwire.linear_probe_baseline import LinearProbeBaseline
        results['linear_probe_import'] = "[PASS] LinearProbeBaseline imported successfully"
    except ImportError as e:
        results['linear_probe_import'] = f"[FAIL] LinearProbeBaseline import failed: {e}"
    except Exception as e:
        results['linear_probe_import'] = f"[FAIL] Unexpected error: {e}"

    # Test token budget evaluation import
    try:
        # The token budget script is standalone, check if file exists
        token_budget_path = Path(__file__).parent / "scripts" / "baselines" / "evaluate_token_budget.py"
        if token_budget_path.exists():
            results['token_budget_import'] = "[PASS] Token budget evaluation script exists"
        else:
            results['token_budget_import'] = f"[FAIL] Token budget script not found at {token_budget_path}"
    except Exception as e:
        results['token_budget_import'] = f"[FAIL] Unexpected error: {e}"

    return results

def test_llmlingua_baseline():
    """Test LLMLingua baseline functionality with minimal data."""
    print("\nTesting LLMLingua baseline...")

    try:
        from latentwire.llmlingua_baseline import evaluate_llmlingua_on_qa

        # Check if llmlingua is installed
        try:
            import llmlingua
            llmlingua_available = True
        except ImportError:
            return "[WARNING] LLMLingua not installed. Install with: pip install llmlingua"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run with minimal samples
            result = evaluate_llmlingua_on_qa(
                dataset="squad",
                split="validation",
                samples=2,  # Just 2 samples for testing
                target_tokens=32,
                use_llmlingua2=True,
                compressor_model="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                output_dir=tmpdir,
                question_aware=True,
                seed=42
            )

            # Check if results were saved
            results_file = Path(tmpdir) / "llmlingua_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    saved_results = json.load(f)

                compression_ratio = saved_results['summary']['avg_compression_ratio']
                return f"[PASS] LLMLingua baseline works! Avg compression ratio: {compression_ratio:.2f}x"
            else:
                return "[FAIL] LLMLingua baseline ran but didn't save results"

    except ImportError as e:
        return f"[FAIL] Import error: {e}"
    except Exception as e:
        return f"[FAIL] Error running LLMLingua baseline: {e}\n{traceback.format_exc()}"

def test_linear_probe_baseline():
    """Test LinearProbeBaseline functionality."""
    print("\nTesting LinearProbeBaseline...")

    try:
        from latentwire.linear_probe_baseline import LinearProbeBaseline
        import torch
        import numpy as np

        # Create dummy data for testing
        n_samples = 100
        n_features = 768  # Typical embedding size
        n_classes = 3

        # Create synthetic data
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_samples)
        X_val = np.random.randn(20, n_features).astype(np.float32)
        y_val = np.random.randint(0, n_classes, 20)

        # Initialize baseline
        baseline = LinearProbeBaseline(
            model_name="test_model",
            task_name="test_task",
            pooling_method="mean",
            max_iter=10,  # Quick test
            C=1.0,
            random_state=42
        )

        # Fit the probe
        baseline.fit(X_train, y_train, X_val, y_val)

        # Make predictions
        predictions = baseline.predict(X_val)
        accuracy = baseline.score(X_val, y_val)

        return f"[PASS] LinearProbeBaseline works! Test accuracy: {accuracy:.2%}"

    except ImportError as e:
        return f"[FAIL] Import error: {e}"
    except Exception as e:
        return f"[FAIL] Error running LinearProbeBaseline: {e}\n{traceback.format_exc()}"

def test_token_budget_baseline():
    """Test token budget baseline script existence and basic structure."""
    print("\nTesting token budget baseline...")

    try:
        # Check if script exists
        script_path = Path(__file__).parent / "scripts" / "baselines" / "evaluate_token_budget.py"

        if not script_path.exists():
            return f"[FAIL] Token budget script not found at {script_path}"

        # Try to parse the script to check for syntax errors
        with open(script_path) as f:
            script_content = f.read()

        # Compile the script to check for syntax errors
        compile(script_content, str(script_path), 'exec')

        # Check for key functions/components
        required_components = ['AutoTokenizer', 'AutoModelForCausalLM', 'batch_metrics', 'main']
        missing = []
        for component in required_components:
            if component not in script_content:
                missing.append(component)

        if missing:
            return f"[WARNING] Token budget script exists but missing: {', '.join(missing)}"

        return "[PASS] Token budget baseline script is valid and complete"

    except SyntaxError as e:
        return f"[FAIL] Token budget script has syntax error: {e}"
    except Exception as e:
        return f"[FAIL] Error checking token budget baseline: {e}"

def test_data_loading():
    """Test that data loading works for baselines."""
    print("\nTesting data loading...")

    try:
        from latentwire.data import load_examples

        # Load minimal data
        examples = load_examples(dataset="squad", split="validation", samples=5, seed=42)

        if not examples:
            return "[FAIL] No examples loaded"

        # Check structure
        required_keys = {'context', 'question', 'answer', 'source'}
        first_example = examples[0]
        missing_keys = required_keys - set(first_example.keys())

        if missing_keys:
            return f"[FAIL] Examples missing required keys: {missing_keys}"

        return f"[PASS] Data loading works! Loaded {len(examples)} examples"

    except ImportError as e:
        return f"[FAIL] Import error: {e}"
    except Exception as e:
        return f"[FAIL] Error loading data: {e}"

def main():
    """Run all Phase 3 baseline tests."""
    print("="*80)
    print("PHASE 3: BASELINE COMPARISONS TEST")
    print("="*80)
    print()
    print("This test verifies that all baseline comparison methods are available")
    print("and functional for Phase 3 experiments.")
    print()

    # Test imports
    import_results = test_imports()
    print("Import Tests:")
    for key, result in import_results.items():
        print(f"  {result}")

    # Test data loading
    data_result = test_data_loading()
    print(f"\nData Loading Test:\n  {data_result}")

    # Test individual baselines
    llmlingua_result = test_llmlingua_baseline()
    print(f"\nLLMLingua Baseline Test:\n  {llmlingua_result}")

    linear_probe_result = test_linear_probe_baseline()
    print(f"\nLinearProbe Baseline Test:\n  {linear_probe_result}")

    token_budget_result = test_token_budget_baseline()
    print(f"\nToken Budget Baseline Test:\n  {token_budget_result}")

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    all_results = [
        import_results['llmlingua_import'],
        import_results['linear_probe_import'],
        import_results['token_budget_import'],
        data_result,
        llmlingua_result,
        linear_probe_result,
        token_budget_result
    ]

    passed = sum(1 for r in all_results if '[PASS]' in str(r))
    warnings = sum(1 for r in all_results if '[WARNING]' in str(r))
    failed = sum(1 for r in all_results if '[FAIL]' in str(r))

    print(f"Passed: {passed}/{len(all_results)}")
    print(f"Warnings: {warnings}/{len(all_results)}")
    print(f"Failed: {failed}/{len(all_results)}")

    if failed == 0:
        print("\n[SUCCESS] All Phase 3 baseline comparison tests passed!")
        print("\nYou can now run the full baseline experiments:")
        print("  - LLMLingua: bash scripts/run_llmlingua_baseline.sh")
        print("  - Token Budget: bash scripts/baselines/token_budget_baseline.sh")
        print("  - Linear Probe: python latentwire/linear_probe_baseline.py --help")
    elif warnings > 0 and failed == 0:
        print("\n[WARNING] Phase 3 baselines work but with warnings. Check above for details.")
    else:
        print("\n[ERROR] Some Phase 3 baseline tests failed. Please check errors above.")
        print("\nCommon fixes:")
        print("  - Install LLMLingua: pip install llmlingua")
        print("  - Install sklearn: pip install scikit-learn")
        print("  - Ensure transformers is installed: pip install transformers")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())