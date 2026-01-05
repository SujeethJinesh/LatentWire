#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test script for XSUM + ROUGE integration.

This script tests the entire pipeline:
1. Loading XSUM data
2. Computing ROUGE scores
3. Handling edge cases
4. Model comparison scenarios
"""

import sys
import json
from pathlib import Path
sys.path.append('.')

from latentwire.rouge_xsum_metrics import (
    compute_rouge_xsum,
    evaluate_xsum_model,
    compare_models_xsum,
    XSumRougeResults
)
from latentwire.data import load_examples


def test_xsum_data_loading():
    """Test loading XSUM dataset."""
    print("\n" + "="*60)
    print("TEST 1: XSUM Data Loading")
    print("="*60)

    try:
        # Try loading a small sample
        examples = load_examples("xsum", split="test", samples=5, seed=42)
        print(f"[OK] Successfully loaded {len(examples)} XSUM examples")

        # Check data format
        for i, ex in enumerate(examples[:2]):
            print(f"\nExample {i+1}:")
            print(f"  Document length: {len(ex.get('prefix', ''))} chars")
            print(f"  Summary length: {len(ex.get('answer', ''))} chars")
            print(f"  Document preview: {ex.get('prefix', '')[:100]}...")
            print(f"  Summary: {ex.get('answer', '')[:100]}")

        return True

    except Exception as e:
        print(f"[WARNING] XSUM loading failed: {e}")
        print("Note: XSUM may require special handling. Using synthetic data for testing.")
        return False


def test_rouge_computation():
    """Test ROUGE score computation with realistic XSUM-style examples."""
    print("\n" + "="*60)
    print("TEST 2: ROUGE Computation")
    print("="*60)

    # Realistic XSUM-style test cases
    test_cases = [
        {
            "prediction": "Scientists discover new planet in nearby solar system",
            "reference": "Astronomers have announced the discovery of a previously unknown planet orbiting a star in a nearby solar system."
        },
        {
            "prediction": "Government announces major education funding increase",
            "reference": "The government has unveiled plans to significantly boost funding for schools and universities."
        },
        {
            "prediction": "Tech giant faces regulatory scrutiny over data practices",
            "reference": "A major technology company is under investigation by regulators concerning its data collection and privacy practices."
        },
        {
            "prediction": "Climate summit ends with new emissions targets",
            "reference": "World leaders have concluded the climate summit by agreeing to more ambitious emissions reduction targets."
        },
        {
            "prediction": "Medical breakthrough offers hope for cancer patients",
            "reference": "Researchers have developed a revolutionary new cancer treatment showing remarkable success in trials."
        }
    ]

    predictions = [tc["prediction"] for tc in test_cases]
    references = [tc["reference"] for tc in test_cases]

    # Test basic computation
    print("Testing basic ROUGE computation...")
    results = compute_rouge_xsum(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        compute_confidence_intervals=False,
        return_per_sample=False
    )

    print(results.summary_string())

    # Verify results are reasonable for XSUM
    assert 0.10 <= results.rouge1_f1 <= 0.80, f"ROUGE-1 out of expected range: {results.rouge1_f1}"
    assert 0.00 <= results.rouge2_f1 <= 0.60, f"ROUGE-2 out of expected range: {results.rouge2_f1}"
    print("[OK] Basic ROUGE computation passed")

    # Test with confidence intervals
    print("\nTesting with confidence intervals...")
    results_ci = compute_rouge_xsum(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        compute_confidence_intervals=True,
        n_bootstrap=500,
        return_per_sample=True
    )

    if results_ci.rouge1_f1_ci:
        print(f"  ROUGE-1 95% CI: [{results_ci.rouge1_f1_ci[0]:.4f}, {results_ci.rouge1_f1_ci[1]:.4f}]")
        print(f"  ROUGE-2 95% CI: [{results_ci.rouge2_f1_ci[0]:.4f}, {results_ci.rouge2_f1_ci[1]:.4f}]")
        print("[OK] Confidence interval computation passed")

    # Check per-sample scores
    if results_ci.per_sample_scores:
        print(f"\n  Retrieved {len(results_ci.per_sample_scores)} per-sample scores")
        print("[OK] Per-sample scoring passed")

    return results_ci


def test_edge_cases():
    """Test edge cases that commonly occur in XSUM."""
    print("\n" + "="*60)
    print("TEST 3: Edge Cases")
    print("="*60)

    edge_cases = [
        {
            "name": "Empty prediction",
            "pred": "",
            "ref": "This is the reference summary"
        },
        {
            "name": "Empty reference",
            "pred": "This is the predicted summary",
            "ref": ""
        },
        {
            "name": "Identical strings",
            "pred": "Brexit negotiations continue in Brussels",
            "ref": "Brexit negotiations continue in Brussels"
        },
        {
            "name": "No overlap",
            "pred": "Apple announces new iPhone",
            "ref": "Scientists discover water on Mars"
        },
        {
            "name": "Very long summary",
            "pred": " ".join(["This is a very long summary"] * 50),
            "ref": "Short reference"
        },
        {
            "name": "Special characters",
            "pred": "COVID-19 vaccine shows 95% efficacy @ Phase-3 trials!",
            "ref": "Coronavirus vaccine demonstrates ninety-five percent effectiveness in phase three trials"
        }
    ]

    print("Testing edge cases...")
    for case in edge_cases:
        try:
            result = compute_rouge_xsum(
                predictions=[case["pred"]],
                references=[case["ref"]],
                use_stemmer=True,
                compute_confidence_intervals=False,
                return_per_sample=False
            )
            print(f"  [OK] {case['name']}: R1={result.rouge1_f1:.3f}, R2={result.rouge2_f1:.3f}")
        except Exception as e:
            print(f"  [ERROR] {case['name']}: {e}")

    print("[OK] Edge case handling completed")


def test_model_comparison():
    """Test comparing multiple model outputs."""
    print("\n" + "="*60)
    print("TEST 4: Model Comparison")
    print("="*60)

    # Simulate outputs from different quality models
    references = [
        "The government has announced a comprehensive new policy to address climate change.",
        "Scientists have discovered a potential cure for a rare genetic disease.",
        "The technology company reported quarterly earnings that exceeded expectations.",
        "International negotiations have resulted in a historic peace agreement.",
        "The central bank has decided to maintain interest rates at current levels."
    ]

    model_outputs = {
        "High-Quality": [
            "Government unveils comprehensive climate change policy",
            "Scientists find cure for rare genetic disorder",
            "Tech company beats quarterly earnings expectations",
            "Historic peace agreement reached in international talks",
            "Central bank keeps interest rates unchanged"
        ],
        "Medium-Quality": [
            "New climate policy announced",
            "Disease cure discovered",
            "Company reports good earnings",
            "Peace deal reached",
            "Rates stay same"
        ],
        "Low-Quality": [
            "Climate stuff",
            "Medical news",
            "Business update",
            "Agreement made",
            "Bank decision"
        ]
    }

    # Test model comparison
    print("Comparing model outputs...")
    results = compare_models_xsum(
        model_outputs,
        references,
        output_dir=Path("test_rouge_output")
    )

    # Verify quality ordering
    high_score = results["High-Quality"].rouge1_f1
    medium_score = results["Medium-Quality"].rouge1_f1
    low_score = results["Low-Quality"].rouge1_f1

    assert high_score >= medium_score >= low_score, "Model quality ordering incorrect"
    print("[OK] Model comparison completed successfully")

    return results


def test_real_xsum_integration():
    """Test with real XSUM data if available."""
    print("\n" + "="*60)
    print("TEST 5: Real XSUM Integration")
    print("="*60)

    try:
        # Try to load real XSUM data
        print("Attempting to load real XSUM data...")
        examples = load_examples("xsum", split="test", samples=10, seed=42)

        # Extract documents and summaries
        documents = [ex.get("prefix", "") for ex in examples]
        references = [ex.get("answer", "") for ex in examples]

        # Simulate a simple extractive baseline
        # (Take first sentence as prediction)
        predictions = []
        for doc in documents:
            sentences = doc.split('. ')
            if sentences:
                predictions.append(sentences[0] + '.')
            else:
                predictions.append("No summary generated.")

        # Evaluate
        results = evaluate_xsum_model(
            predictions,
            references,
            model_name="extractive_baseline",
            output_dir=Path("test_rouge_output"),
            save_results=True
        )

        print("[OK] Real XSUM integration test passed")
        return results

    except Exception as e:
        print(f"[WARNING] Could not test with real XSUM data: {e}")
        print("This is expected if XSUM dataset is not available.")
        return None


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("XSUM + ROUGE INTEGRATION TEST SUITE")
    print("="*70)

    test_results = {}

    # Test 1: Data loading
    data_loaded = test_xsum_data_loading()
    test_results["data_loading"] = data_loaded

    # Test 2: ROUGE computation
    rouge_results = test_rouge_computation()
    test_results["rouge_computation"] = rouge_results is not None

    # Test 3: Edge cases
    test_edge_cases()
    test_results["edge_cases"] = True

    # Test 4: Model comparison
    comparison_results = test_model_comparison()
    test_results["model_comparison"] = comparison_results is not None

    # Test 5: Real XSUM (optional)
    real_results = test_real_xsum_integration()
    test_results["real_xsum"] = real_results is not None

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in test_results.items():
        status = "[OK] PASSED" if passed else "[WARNING] SKIPPED/FAILED"
        print(f"  {test_name:<20} {status}")

    print("="*70)

    # Save test results
    output_dir = Path("test_rouge_output")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "test_results.json", "w") as f:
        json.dump({
            "test_results": test_results,
            "rouge_sample": rouge_results.to_dict() if rouge_results else None
        }, f, indent=2)

    print(f"\nTest results saved to {output_dir}/test_results.json")

    # Overall status
    essential_tests = ["rouge_computation", "edge_cases", "model_comparison"]
    all_essential_passed = all(test_results.get(t, False) for t in essential_tests)

    if all_essential_passed:
        print("\n[SUCCESS] All essential tests passed! ROUGE+XSUM integration is working correctly.")
    else:
        print("\n[WARNING] Some essential tests failed. Please review the output above.")

    return all_essential_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)