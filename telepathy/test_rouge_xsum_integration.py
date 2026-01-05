#!/usr/bin/env python
"""
Integration test for ROUGE metrics with XSUM dataset format.

This script tests the rouge_metrics module specifically with XSUM-style data,
ensuring it properly handles real-world summarization scenarios.
"""

import json
import numpy as np
from rouge_metrics import compute_rouge, compute_rouge_batch, save_rouge_results, load_rouge_results
import tempfile
import os


def test_xsum_real_examples():
    """Test with real XSUM-style examples."""
    print("Testing with real XSUM-style examples...")

    # Real-world XSUM examples (abstractive summaries)
    xsum_examples = [
        {
            "prediction": "Scientists discover new planet in nearby solar system",
            "reference": "Astronomers have announced the discovery of a previously unknown planet orbiting a star in a nearby solar system, marking a significant breakthrough in exoplanet research."
        },
        {
            "prediction": "Government announces major education funding increase",
            "reference": "The government has unveiled plans to significantly increase funding for education, with billions of pounds to be invested in schools and universities over the next five years."
        },
        {
            "prediction": "Tech giant faces regulatory scrutiny over data practices",
            "reference": "A major technology company is under investigation by regulators concerning its data collection and privacy practices, with potential fines in the billions."
        },
        {
            "prediction": "Climate summit ends with new emissions targets",
            "reference": "World leaders have concluded a major climate summit by agreeing to more ambitious emissions reduction targets, though critics say the measures don't go far enough."
        },
        {
            "prediction": "Medical breakthrough offers hope for cancer patients",
            "reference": "Researchers have developed a revolutionary new cancer treatment that has shown remarkable success in early trials, potentially offering hope to millions of patients worldwide."
        }
    ]

    predictions = [ex["prediction"] for ex in xsum_examples]
    references = [ex["reference"] for ex in xsum_examples]

    # Compute ROUGE scores with all features
    results = compute_rouge(
        predictions,
        references,
        use_stemmer=True,  # XSUM standard
        compute_confidence_intervals=True,
        n_bootstrap=1000,
        confidence_level=0.95,
        return_per_sample=True,
        show_progress=True
    )

    print("\n" + results.summary_string())

    # Verify results are in expected range for XSUM
    assert 0.15 <= results.rouge1_f1_mean <= 0.45, f"XSUM ROUGE-1 out of typical range: {results.rouge1_f1_mean}"
    assert 0.02 <= results.rouge2_f1_mean <= 0.20, f"XSUM ROUGE-2 out of typical range: {results.rouge2_f1_mean}"
    assert 0.10 <= results.rougeL_f1_mean <= 0.35, f"XSUM ROUGE-L out of typical range: {results.rougeL_f1_mean}"

    # Check per-sample variation
    rouge1_scores = [s['rouge1']['f1'] for s in results.per_sample_scores]
    variation = np.std(rouge1_scores) / np.mean(rouge1_scores)
    print(f"\nCoefficient of variation: {variation:.3f}")
    assert variation > 0.1, "XSUM examples should show variation in quality"

    return results


def test_xsum_edge_cases():
    """Test edge cases common in XSUM dataset."""
    print("\nTesting XSUM edge cases...")

    edge_cases = [
        # Very long summary (XSUM can have lengthy gold summaries)
        {
            "prediction": "Company reports record profits",
            "reference": " ".join(["The multinational corporation has announced record-breaking profits for the fiscal year"] * 10)
        },
        # Summary with numbers and dates
        {
            "prediction": "Stock market rises 2.5% on Friday January 15th 2024",
            "reference": "Markets closed higher on January 15, 2024, with the main index gaining 2.5 percent"
        },
        # Summary with special characters and punctuation
        {
            "prediction": "CEO says: 'We're committed to innovation!'",
            "reference": "The chief executive stated: \"We are dedicated to innovative solutions.\""
        },
        # Very short summary
        {
            "prediction": "Brexit deal reached",
            "reference": "Agreement on Brexit finally achieved"
        },
        # Summary with technical terms
        {
            "prediction": "COVID-19 mRNA vaccine shows 95% efficacy",
            "reference": "The messenger RNA vaccine against coronavirus demonstrates ninety-five percent effectiveness"
        }
    ]

    predictions = [ex["prediction"] for ex in edge_cases]
    references = [ex["reference"] for ex in edge_cases]

    results = compute_rouge(
        predictions,
        references,
        use_stemmer=True,
        compute_confidence_intervals=False,
        return_per_sample=True,
        show_progress=False
    )

    print(f"Edge cases ROUGE-1: {results.rouge1_f1_mean:.4f}")
    print(f"Edge cases ROUGE-2: {results.rouge2_f1_mean:.4f}")
    print(f"Edge cases ROUGE-L: {results.rougeL_f1_mean:.4f}")

    # Check individual edge case handling
    for i, (case, scores) in enumerate(zip(edge_cases, results.per_sample_scores)):
        case_type = list(case.keys())[0] if isinstance(case, dict) else f"Case {i}"
        rouge1_score = scores['rouge1']['f1']
        print(f"  Case {i}: ROUGE-1 F1={rouge1_score:.3f}")

    return results


def test_xsum_model_comparison():
    """Test comparing multiple models on XSUM-style data."""
    print("\nTesting multi-model comparison on XSUM data...")

    # Simulate different model quality levels
    references = [
        "The government has announced a comprehensive new policy to address climate change through renewable energy investments.",
        "Scientists have discovered a potential cure for a rare genetic disease affecting thousands worldwide.",
        "The technology company reported quarterly earnings that exceeded analyst expectations by a significant margin.",
        "International negotiations have resulted in a historic peace agreement between the conflicting nations.",
        "The central bank has decided to maintain interest rates at current levels despite inflation concerns."
    ]

    # High-quality model (good abstractive summaries)
    high_quality = [
        "Government unveils major climate policy focusing on renewables",
        "Breakthrough treatment found for rare genetic disorder",
        "Tech firm beats earnings expectations significantly",
        "Historic peace deal reached in international talks",
        "Central bank holds rates steady amid inflation worries"
    ]

    # Medium-quality model (some key info missing)
    medium_quality = [
        "New climate policy announced",
        "Scientists find disease cure",
        "Company reports good earnings",
        "Peace agreement signed",
        "Interest rates unchanged"
    ]

    # Low-quality model (too brief/inaccurate)
    low_quality = [
        "Climate news",
        "Medical discovery",
        "Business update",
        "Political agreement",
        "Economic decision"
    ]

    # Extractive baseline (copies phrases)
    extractive = [
        "government comprehensive new policy climate change renewable energy",
        "scientists discovered potential cure rare genetic disease",
        "technology company quarterly earnings exceeded analyst expectations",
        "international negotiations historic peace agreement nations",
        "central bank maintain interest rates current levels"
    ]

    # Compute for all models
    model_results = compute_rouge_batch(
        [high_quality, medium_quality, low_quality, extractive],
        references,
        model_names=["HighQuality", "MediumQuality", "LowQuality", "Extractive"],
        compute_confidence_intervals=False,
        show_progress=False
    )

    print("\nModel Comparison Results:")
    print("-" * 50)
    for model_name in ["HighQuality", "MediumQuality", "LowQuality", "Extractive"]:
        r = model_results[model_name]
        print(f"{model_name:15} | R1: {r.rouge1_f1_mean:.3f} | R2: {r.rouge2_f1_mean:.3f} | RL: {r.rougeL_f1_mean:.3f}")

    # Verify quality ordering
    assert model_results["HighQuality"].rouge1_f1_mean > model_results["LowQuality"].rouge1_f1_mean, \
        "High quality should outperform low quality"
    assert model_results["MediumQuality"].rouge1_f1_mean > model_results["LowQuality"].rouge1_f1_mean, \
        "Medium quality should outperform low quality"

    return model_results


def test_save_load_functionality():
    """Test saving and loading ROUGE results."""
    print("\nTesting save/load functionality...")

    # Generate some results
    predictions = ["Test summary one", "Test summary two", "Test summary three"]
    references = ["Reference summary one", "Reference summary two", "Reference summary three"]

    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=True,
        n_bootstrap=100,  # Small for speed
        return_per_sample=True,
        show_progress=False
    )

    # Save with metadata
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    metadata = {
        "dataset": "XSUM",
        "model": "test_model",
        "timestamp": "2024-01-01",
        "num_samples": len(predictions)
    }

    save_rouge_results(results, temp_path, metadata)

    # Load and verify
    loaded_results, loaded_metadata = load_rouge_results(temp_path)

    # Verify core metrics match
    assert abs(loaded_results.rouge1_f1_mean - results.rouge1_f1_mean) < 1e-10, \
        "Loaded ROUGE-1 F1 doesn't match"
    assert abs(loaded_results.rouge2_f1_mean - results.rouge2_f1_mean) < 1e-10, \
        "Loaded ROUGE-2 F1 doesn't match"
    assert abs(loaded_results.rougeL_f1_mean - results.rougeL_f1_mean) < 1e-10, \
        "Loaded ROUGE-L F1 doesn't match"

    # Verify confidence intervals
    if results.rouge1_f1_ci is not None:
        assert loaded_results.rouge1_f1_ci is not None, "CI should be loaded"
        assert abs(loaded_results.rouge1_f1_ci[0] - results.rouge1_f1_ci[0]) < 1e-10, \
            "Loaded CI lower bound doesn't match"

    # Verify metadata
    assert loaded_metadata == metadata, "Loaded metadata doesn't match"

    # Verify per-sample scores
    if results.per_sample_scores is not None:
        assert loaded_results.per_sample_scores is not None, "Per-sample scores should be loaded"
        assert len(loaded_results.per_sample_scores) == len(results.per_sample_scores), \
            "Number of per-sample scores doesn't match"

    print(f"✓ Save/load successful: {temp_path}")

    # Clean up
    os.unlink(temp_path)

    return True


def test_large_batch_performance():
    """Test performance with large batches typical of XSUM evaluation."""
    print("\nTesting large batch performance...")

    import time

    # Generate large batch (simulating full XSUM test set size)
    n_samples = 1000
    predictions = []
    references = []

    np.random.seed(42)
    vocab = ["the", "a", "is", "was", "are", "were", "have", "has", "had",
             "government", "policy", "announced", "climate", "technology",
             "company", "report", "scientists", "discovered", "new", "major"]

    for i in range(n_samples):
        # Generate random summaries of varying lengths
        pred_len = np.random.randint(5, 15)
        ref_len = np.random.randint(10, 25)

        pred_words = np.random.choice(vocab, pred_len)
        ref_words = np.random.choice(vocab, ref_len)

        predictions.append(" ".join(pred_words))
        references.append(" ".join(ref_words))

    # Time the computation
    start_time = time.time()

    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,  # Skip for speed test
        return_per_sample=False,  # Skip for memory efficiency
        show_progress=True
    )

    elapsed_time = time.time() - start_time

    print(f"\nProcessed {n_samples} samples in {elapsed_time:.2f} seconds")
    print(f"Throughput: {n_samples / elapsed_time:.1f} samples/second")

    # Performance requirements
    assert elapsed_time < 30, f"Should process {n_samples} samples in under 30 seconds, took {elapsed_time:.2f}s"
    assert n_samples / elapsed_time > 30, f"Should process >30 samples/sec, got {n_samples/elapsed_time:.1f}"

    print(f"✓ Large batch performance test passed")

    return results


def run_xsum_integration_tests():
    """Run all XSUM integration tests."""
    print("=" * 60)
    print("XSUM ROUGE Integration Tests")
    print("=" * 60)

    tests = [
        ("Real XSUM Examples", test_xsum_real_examples),
        ("XSUM Edge Cases", test_xsum_edge_cases),
        ("Model Comparison", test_xsum_model_comparison),
        ("Save/Load Results", test_save_load_functionality),
        ("Large Batch Performance", test_large_batch_performance)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*40}")
            print(f"Running: {test_name}")
            print(f"{'='*40}")
            test_func()
            passed += 1
            print(f"✓ {test_name} passed")
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"XSUM INTEGRATION TESTS COMPLETE: {passed}/{len(tests)} passed")

    if failed == 0:
        print("✓ All XSUM integration tests passed!")
        print("✓ ROUGE implementation ready for XSUM dataset evaluation")
    else:
        print(f"✗ {failed} tests failed - please review")

    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_xsum_integration_tests()
    exit(0 if success else 1)