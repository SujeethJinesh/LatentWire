#!/usr/bin/env python3
"""
Simple test to verify ROUGE metrics module works correctly without torch dependency.
"""

import sys
import json

# Direct import to avoid telepathy/__init__.py which imports torch
sys.path.insert(0, '.')

# Import only the rouge_metrics module directly
import importlib.util
spec = importlib.util.spec_from_file_location("rouge_metrics", "telepathy/rouge_metrics.py")
rouge_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rouge_metrics)

compute_rouge = rouge_metrics.compute_rouge
save_rouge_results = rouge_metrics.save_rouge_results


def test_simple():
    """Test basic ROUGE functionality."""

    print("="*70)
    print("SIMPLE ROUGE METRICS TEST")
    print("="*70)

    # Test data
    predictions = [
        "The cat sat on the mat.",
        "A quick brown fox jumps.",
        "Machine learning is powerful."
    ]

    references = [
        "The cat was sitting on the mat.",
        "A quick brown fox jumped over the lazy dog.",
        "Machine learning is a powerful technology."
    ]

    print("\nComputing ROUGE scores...")
    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=True,
        n_bootstrap=100,
        show_progress=False
    )

    print("\nResults:")
    print("  ROUGE-1 F1: {:.4f} +/- {:.4f}".format(
        results.rouge1_f1_mean, results.rouge1_f1_std
    ))
    print("  ROUGE-2 F1: {:.4f} +/- {:.4f}".format(
        results.rouge2_f1_mean, results.rouge2_f1_std
    ))
    print("  ROUGE-L F1: {:.4f} +/- {:.4f}".format(
        results.rougeL_f1_mean, results.rougeL_f1_std
    ))

    if results.rouge1_f1_ci:
        print("\nConfidence Intervals (95%):")
        print("  ROUGE-1: [{:.4f}, {:.4f}]".format(
            results.rouge1_f1_ci[0], results.rouge1_f1_ci[1]
        ))
        print("  ROUGE-2: [{:.4f}, {:.4f}]".format(
            results.rouge2_f1_ci[0], results.rouge2_f1_ci[1]
        ))
        print("  ROUGE-L: [{:.4f}, {:.4f}]".format(
            results.rougeL_f1_ci[0], results.rougeL_f1_ci[1]
        ))

    # Test JSON serialization
    results_dict = results.to_dict()
    json_str = json.dumps(results_dict, indent=2)

    print("\nJSON serialization test:")
    print("  Length of JSON: {} characters".format(len(json_str)))
    print("  Has rouge1: {}".format('rouge1' in results_dict))
    print("  Has rouge2: {}".format('rouge2' in results_dict))
    print("  Has rougeL: {}".format('rougeL' in results_dict))
    print("  Has CIs: {}".format('ci_95' in results_dict.get('rouge1', {}).get('f1', {})))

    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70)

    return True


if __name__ == "__main__":
    test_simple()