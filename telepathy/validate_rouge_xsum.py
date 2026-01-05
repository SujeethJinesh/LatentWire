#!/usr/bin/env python3
"""
Validation script to ensure ROUGE metrics are properly integrated in XSUM evaluation.

This script should be run on the HPC where all dependencies are available.

Usage:
    python telepathy/validate_rouge_xsum.py
"""

import sys
import json
import traceback

# Ensure proper path
sys.path.append('.')

def validate_imports():
    """Validate that all necessary imports work."""
    print("1. Validating imports...")

    try:
        from telepathy.rouge_metrics import compute_rouge, RougeResults, save_rouge_results
        print("   [OK] rouge_metrics imports successful")
    except Exception as e:
        print("   [FAIL] rouge_metrics import failed: {}".format(e))
        return False

    try:
        from telepathy.train_xsum_bridge import load_xsum_data
        print("   [OK] train_xsum_bridge imports successful")
    except Exception as e:
        print("   [FAIL] train_xsum_bridge import failed: {}".format(e))
        return False

    try:
        from telepathy.run_unified_comparison import UnifiedBridge
        print("   [OK] UnifiedBridge import successful")
    except Exception as e:
        print("   [FAIL] UnifiedBridge import failed: {}".format(e))
        return False

    return True


def validate_rouge_computation():
    """Validate ROUGE computation with sample data."""
    print("\n2. Validating ROUGE computation...")

    from telepathy.rouge_metrics import compute_rouge

    # Sample data
    predictions = [
        "Scientists discover new species in ocean.",
        "Climate change affects weather patterns.",
        "Technology advances reshape workplace."
    ]

    references = [
        "Researchers find new species in ocean depths.",
        "Global weather patterns disrupted by climate change.",
        "Modern workplace transformed by technology."
    ]

    try:
        # Basic computation
        results = compute_rouge(
            predictions,
            references,
            compute_confidence_intervals=False,
            show_progress=False
        )

        assert results.rouge1_f1_mean > 0, "ROUGE-1 should be positive"
        assert results.rouge2_f1_mean >= 0, "ROUGE-2 should be non-negative"
        assert results.rougeL_f1_mean > 0, "ROUGE-L should be positive"

        print("   [OK] Basic ROUGE computation works")
        print("       ROUGE-1 F1: {:.4f}".format(results.rouge1_f1_mean))
        print("       ROUGE-2 F1: {:.4f}".format(results.rouge2_f1_mean))
        print("       ROUGE-L F1: {:.4f}".format(results.rougeL_f1_mean))

    except Exception as e:
        print("   [FAIL] Basic computation failed: {}".format(e))
        traceback.print_exc()
        return False

    try:
        # With confidence intervals
        results_ci = compute_rouge(
            predictions,
            references,
            compute_confidence_intervals=True,
            n_bootstrap=50,  # Small for speed
            show_progress=False
        )

        assert results_ci.rouge1_f1_ci is not None, "Should have ROUGE-1 CI"
        assert len(results_ci.rouge1_f1_ci) == 2, "CI should be tuple of 2"

        print("   [OK] Confidence interval computation works")
        print("       ROUGE-1 CI: [{:.4f}, {:.4f}]".format(
            results_ci.rouge1_f1_ci[0], results_ci.rouge1_f1_ci[1]
        ))

    except Exception as e:
        print("   [FAIL] CI computation failed: {}".format(e))
        traceback.print_exc()
        return False

    return True


def validate_serialization():
    """Validate JSON serialization."""
    print("\n3. Validating serialization...")

    from telepathy.rouge_metrics import compute_rouge, save_rouge_results
    import tempfile
    import os

    predictions = ["Test summary one.", "Test summary two."]
    references = ["Reference summary one.", "Reference summary two."]

    try:
        results = compute_rouge(
            predictions,
            references,
            compute_confidence_intervals=True,
            n_bootstrap=50,
            show_progress=False
        )

        # Test to_dict
        results_dict = results.to_dict()
        assert 'rouge1' in results_dict
        assert 'f1' in results_dict['rouge1']
        assert 'mean' in results_dict['rouge1']['f1']
        print("   [OK] to_dict() conversion works")

        # Test JSON serialization
        json_str = json.dumps(results_dict, indent=2)
        assert len(json_str) > 0
        print("   [OK] JSON serialization works")

        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_rouge_results(
                results,
                temp_path,
                metadata={'test': True}
            )

            with open(temp_path) as f:
                loaded = json.load(f)

            assert 'rouge_scores' in loaded
            assert 'metadata' in loaded
            print("   [OK] save_rouge_results() works")

        finally:
            os.unlink(temp_path)

    except Exception as e:
        print("   [FAIL] Serialization failed: {}".format(e))
        traceback.print_exc()
        return False

    return True


def validate_eval_script():
    """Validate that eval_xsum_bridge.py has correct imports."""
    print("\n4. Validating eval_xsum_bridge.py structure...")

    try:
        # Read the file
        with open('telepathy/eval_xsum_bridge.py', 'r') as f:
            content = f.read()

        # Check for correct imports
        checks = [
            ('from telepathy.rouge_metrics import compute_rouge', 'Imports compute_rouge from rouge_metrics'),
            ('from telepathy.rouge_metrics import.*RougeResults', 'Imports RougeResults'),
            ('from telepathy.rouge_metrics import.*save_rouge_results', 'Imports save_rouge_results'),
            ('rouge_results = compute_rouge', 'Uses compute_rouge for Bridge evaluation'),
            ('baseline_rouge_results = compute_rouge', 'Uses compute_rouge for baseline'),
            ('compute_confidence_intervals=True', 'Computes confidence intervals'),
            ('results.to_dict()', 'Converts results to dict'),
            ('save_rouge_results', 'Saves detailed ROUGE results')
        ]

        import re
        all_passed = True
        for pattern, description in checks:
            if re.search(pattern, content):
                print("   [OK] {}".format(description))
            else:
                print("   [FAIL] Missing: {}".format(description))
                all_passed = False

        return all_passed

    except Exception as e:
        print("   [FAIL] Could not validate eval script: {}".format(e))
        return False


def main():
    """Run all validation checks."""
    print("="*70)
    print("ROUGE XSUM INTEGRATION VALIDATION")
    print("="*70)

    all_passed = True

    # Run validation steps
    if not validate_imports():
        all_passed = False

    if not validate_rouge_computation():
        all_passed = False

    if not validate_serialization():
        all_passed = False

    if not validate_eval_script():
        all_passed = False

    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED")
        print("ROUGE metrics are properly integrated for XSUM evaluation")
    else:
        print("SOME VALIDATION CHECKS FAILED")
        print("Please review the errors above")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())