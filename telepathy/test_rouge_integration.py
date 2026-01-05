#!/usr/bin/env python3
"""
Test script to verify ROUGE metrics integration in XSUM evaluation.

This tests that:
1. ROUGE metrics module is properly imported
2. Confidence intervals are computed
3. Results are saved in correct format
4. All metrics (ROUGE-1, ROUGE-2, ROUGE-L) are included
"""

import sys
import json
from pathlib import Path

sys.path.append('.')
from telepathy.rouge_metrics import compute_rouge, save_rouge_results


def test_rouge_integration():
    """Test ROUGE metrics integration with sample data."""

    print("="*70)
    print("TESTING ROUGE METRICS INTEGRATION FOR XSUM")
    print("="*70)

    # Sample predictions and references
    test_predictions = [
        "Scientists discover new species in deep ocean.",
        "Climate change affects global weather patterns.",
        "Technology advances reshape modern workplace.",
        "Medical breakthrough offers hope for patients.",
        "Economic policy changes impact markets."
    ]

    test_references = [
        "Researchers find previously unknown species in ocean depths.",
        "Global weather patterns disrupted by climate change.",
        "Modern workplace transformed by technological advances.",
        "New medical treatment provides hope for disease patients.",
        "Markets react to new economic policy announcements."
    ]

    # Test 1: Basic ROUGE computation
    print("\n1. Testing basic ROUGE computation...")
    basic_results = compute_rouge(
        test_predictions,
        test_references,
        compute_confidence_intervals=False,
        show_progress=False
    )

    print("   ROUGE-1 F1: {:.4f}".format(basic_results.rouge1_f1_mean))
    print("   ROUGE-2 F1: {:.4f}".format(basic_results.rouge2_f1_mean))
    print("   ROUGE-L F1: {:.4f}".format(basic_results.rougeL_f1_mean))

    assert basic_results.rouge1_f1_mean > 0, "ROUGE-1 should be > 0"
    assert basic_results.rouge2_f1_mean >= 0, "ROUGE-2 should be >= 0"
    assert basic_results.rougeL_f1_mean > 0, "ROUGE-L should be > 0"
    print("   [PASS] Basic metrics computed successfully")

    # Test 2: ROUGE with confidence intervals
    print("\n2. Testing ROUGE with confidence intervals...")
    ci_results = compute_rouge(
        test_predictions,
        test_references,
        compute_confidence_intervals=True,
        n_bootstrap=100,  # Reduced for speed
        show_progress=False
    )

    assert ci_results.rouge1_f1_ci is not None, "Should have ROUGE-1 CI"
    assert ci_results.rouge2_f1_ci is not None, "Should have ROUGE-2 CI"
    assert ci_results.rougeL_f1_ci is not None, "Should have ROUGE-L CI"

    print("   ROUGE-1 F1 CI: [{:.4f}, {:.4f}]".format(ci_results.rouge1_f1_ci[0], ci_results.rouge1_f1_ci[1]))
    print("   ROUGE-2 F1 CI: [{:.4f}, {:.4f}]".format(ci_results.rouge2_f1_ci[0], ci_results.rouge2_f1_ci[1]))
    print("   ROUGE-L F1 CI: [{:.4f}, {:.4f}]".format(ci_results.rougeL_f1_ci[0], ci_results.rougeL_f1_ci[1]))
    print("   [PASS] Confidence intervals computed successfully")

    # Test 3: Results serialization
    print("\n3. Testing results serialization...")
    results_dict = ci_results.to_dict()

    assert 'rouge1' in results_dict, "Should have rouge1 in dict"
    assert 'rouge2' in results_dict, "Should have rouge2 in dict"
    assert 'rougeL' in results_dict, "Should have rougeL in dict"

    assert 'f1' in results_dict['rouge1'], "Should have F1 scores"
    assert 'mean' in results_dict['rouge1']['f1'], "Should have mean"
    assert 'std' in results_dict['rouge1']['f1'], "Should have std"
    assert 'ci_95' in results_dict['rouge1']['f1'], "Should have CI"

    # Try JSON serialization
    json_str = json.dumps(results_dict, indent=2)
    assert len(json_str) > 0, "Should serialize to JSON"
    print("   [PASS] Results serialize to JSON correctly")

    # Test 4: Summary string generation
    print("\n4. Testing summary string generation...")
    summary = ci_results.summary_string()

    assert "ROUGE-1 F1:" in summary, "Should include ROUGE-1"
    assert "ROUGE-2 F1:" in summary, "Should include ROUGE-2"
    assert "ROUGE-L F1:" in summary, "Should include ROUGE-L"
    assert "Confidence Intervals" in summary, "Should include CIs"

    print(summary)
    print("   [PASS] Summary string generated successfully")

    # Test 5: Save/load functionality
    print("\n5. Testing save/load functionality...")
    test_output_dir = Path("runs/test_rouge_integration")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = test_output_dir / "test_rouge_results.json"
    save_rouge_results(
        ci_results,
        str(output_file),
        metadata={
            "test": True,
            "dataset": "test_samples",
            "model": "test_model"
        }
    )

    assert output_file.exists(), "Output file should exist"

    with open(output_file) as f:
        loaded_data = json.load(f)

    assert 'rouge_scores' in loaded_data, "Should have rouge_scores"
    assert 'metadata' in loaded_data, "Should have metadata"
    assert loaded_data['metadata']['test'] is True, "Metadata should be preserved"

    print("   [PASS] Results saved to {}".format(output_file))
    print("   [PASS] Save/load functionality works correctly")

    # Clean up
    output_file.unlink()

    print("\n" + "="*70)
    print("ALL TESTS PASSED - ROUGE INTEGRATION WORKING CORRECTLY")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_rouge_integration()
    sys.exit(0 if success else 1)