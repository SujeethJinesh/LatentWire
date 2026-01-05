#!/usr/bin/env python
"""
Comprehensive validation tests for ROUGE metrics implementation.

This script validates that the rouge_metrics.py module:
1. Uses standard rouge_score library correctly
2. Computes ROUGE-1, ROUGE-2, ROUGE-L properly
3. Handles empty/malformed outputs gracefully
4. Bootstrap CI for ROUGE scores implemented
5. Batch processing efficient
6. Handles Unicode text correctly
7. Works with XSUM dataset format
"""

import numpy as np
import json
import time
from typing import List, Dict, Any, Tuple
from rouge_score import rouge_scorer
from rouge_metrics import compute_rouge, compute_rouge_batch, RougeResults
import warnings

# Suppress tokenization warnings
warnings.filterwarnings("ignore", category=UserWarning)


def test_perfect_match():
    """Test that perfect match gives ROUGE scores of 1.0 (100%)."""
    print("Testing perfect match...")

    predictions = [
        "The cat sat on the mat.",
        "Machine learning is powerful.",
        "Natural language processing enables many applications."
    ]
    references = predictions.copy()  # Perfect match

    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        show_progress=False
    )

    # Check that all F1 scores are 1.0
    assert abs(results.rouge1_f1_mean - 1.0) < 1e-6, f"ROUGE-1 F1 should be 1.0, got {results.rouge1_f1_mean}"
    assert abs(results.rouge2_f1_mean - 1.0) < 1e-6, f"ROUGE-2 F1 should be 1.0, got {results.rouge2_f1_mean}"
    assert abs(results.rougeL_f1_mean - 1.0) < 1e-6, f"ROUGE-L F1 should be 1.0, got {results.rougeL_f1_mean}"

    print(f"✓ Perfect match: ROUGE-1={results.rouge1_f1_mean:.4f}, ROUGE-2={results.rouge2_f1_mean:.4f}, ROUGE-L={results.rougeL_f1_mean:.4f}")


def test_no_overlap():
    """Test that no overlap gives ROUGE scores of 0.0."""
    print("\nTesting no overlap...")

    predictions = [
        "The cat sat on the mat.",
        "Machine learning is powerful.",
        "Natural language processing enables many applications."
    ]
    references = [
        "Dogs run in parks quickly.",
        "Statistics involves complex mathematics.",
        "Computer vision detects objects automatically."
    ]

    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        show_progress=False
    )

    # Check that all F1 scores are 0.0
    assert abs(results.rouge1_f1_mean - 0.0) < 0.01, f"ROUGE-1 F1 should be ~0.0, got {results.rouge1_f1_mean}"
    assert abs(results.rouge2_f1_mean - 0.0) < 1e-6, f"ROUGE-2 F1 should be 0.0, got {results.rouge2_f1_mean}"
    assert abs(results.rougeL_f1_mean - 0.0) < 0.01, f"ROUGE-L F1 should be ~0.0, got {results.rougeL_f1_mean}"

    print(f"✓ No overlap: ROUGE-1={results.rouge1_f1_mean:.4f}, ROUGE-2={results.rouge2_f1_mean:.4f}, ROUGE-L={results.rougeL_f1_mean:.4f}")


def test_empty_strings():
    """Test handling of empty and malformed strings."""
    print("\nTesting empty/malformed strings...")

    predictions = [
        "",  # Empty
        "   ",  # Whitespace only
        "Normal text here.",
        "\n\n\n",  # Newlines only
        "Another normal text."
    ]
    references = [
        "Some reference text.",
        "Another reference.",
        "Normal text here.",
        "More reference text.",
        "Different text entirely."
    ]

    # Should not raise an error
    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        show_progress=False
    )

    # Check that we got valid results
    assert results.rouge1_f1_mean >= 0.0, "ROUGE-1 F1 should be non-negative"
    assert results.rouge2_f1_mean >= 0.0, "ROUGE-2 F1 should be non-negative"
    assert results.rougeL_f1_mean >= 0.0, "ROUGE-L F1 should be non-negative"

    print(f"✓ Empty strings handled: ROUGE-1={results.rouge1_f1_mean:.4f}, ROUGE-2={results.rouge2_f1_mean:.4f}, ROUGE-L={results.rougeL_f1_mean:.4f}")


def test_unicode_text():
    """Test handling of Unicode text including emojis and special characters."""
    print("\nTesting Unicode text...")

    predictions = [
        "Hello 世界! 🌍",
        "Café résumé naïve",
        "Математика это наука",
        "🚀 Rocket to the moon 🌙",
        "日本語のテキスト"
    ]
    references = [
        "Hello 世界! 🌍",  # Perfect match
        "Café resume naive",  # Close match
        "Математика - наука",  # Similar
        "🚀 Rocket moon 🌙",  # Partial match
        "Japanese text"  # Different
    ]

    # Should handle Unicode without errors
    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        show_progress=False
    )

    # Check that we got valid results
    assert 0.0 <= results.rouge1_f1_mean <= 1.0, "ROUGE-1 F1 should be between 0 and 1"
    assert 0.0 <= results.rouge2_f1_mean <= 1.0, "ROUGE-2 F1 should be between 0 and 1"
    assert 0.0 <= results.rougeL_f1_mean <= 1.0, "ROUGE-L F1 should be between 0 and 1"

    print(f"✓ Unicode handled: ROUGE-1={results.rouge1_f1_mean:.4f}, ROUGE-2={results.rouge2_f1_mean:.4f}, ROUGE-L={results.rougeL_f1_mean:.4f}")


def test_bootstrap_confidence_intervals():
    """Test bootstrap confidence interval computation."""
    print("\nTesting bootstrap confidence intervals...")

    # Create data with known variability
    np.random.seed(42)
    n_samples = 100

    # Generate predictions with varying quality
    predictions = []
    references = []

    for i in range(n_samples):
        ref = f"This is reference sentence number {i} with some additional words."
        if i < 30:  # 30% perfect matches
            pred = ref
        elif i < 60:  # 30% partial matches
            pred = f"This is reference sentence number {i}."
        else:  # 40% poor matches
            pred = f"Different text entirely {i}."

        predictions.append(pred)
        references.append(ref)

    # Compute with confidence intervals
    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=True,
        n_bootstrap=500,  # Fewer for speed
        confidence_level=0.95,
        show_progress=False
    )

    # Check that CIs were computed
    assert results.rouge1_f1_ci is not None, "ROUGE-1 CI should be computed"
    assert results.rouge2_f1_ci is not None, "ROUGE-2 CI should be computed"
    assert results.rougeL_f1_ci is not None, "ROUGE-L CI should be computed"

    # Check CI properties
    lower, upper = results.rouge1_f1_ci
    mean = results.rouge1_f1_mean

    assert lower <= mean <= upper, f"Mean {mean} should be within CI [{lower}, {upper}]"
    assert 0.0 <= lower <= 1.0, f"Lower bound {lower} should be in [0, 1]"
    assert 0.0 <= upper <= 1.0, f"Upper bound {upper} should be in [0, 1]"
    assert lower < upper, f"Lower bound {lower} should be less than upper {upper}"

    print(f"✓ Bootstrap CI: ROUGE-1 F1={mean:.4f}, 95% CI=[{lower:.4f}, {upper:.4f}]")


def test_batch_processing_efficiency():
    """Test efficiency of batch processing."""
    print("\nTesting batch processing efficiency...")

    # Generate larger dataset
    n_samples = 500
    predictions = [f"Prediction text number {i} with some words." for i in range(n_samples)]
    references = [f"Reference text number {i} with different words." for i in range(n_samples)]

    # Time single batch processing
    start_time = time.time()
    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        show_progress=False
    )
    batch_time = time.time() - start_time

    # Check throughput
    throughput = n_samples / batch_time
    print(f"✓ Batch efficiency: {n_samples} samples in {batch_time:.2f}s ({throughput:.1f} samples/sec)")

    # Verify memory efficiency (should not explode for large batches)
    assert throughput > 10, f"Processing should be faster than 10 samples/sec, got {throughput:.1f}"


def test_per_sample_scores():
    """Test per-sample score tracking."""
    print("\nTesting per-sample scores...")

    predictions = [
        "The cat sat on mat.",  # Missing 'the'
        "Machine learning is very powerful.",  # Extra word
        "Natural language processing."  # Shortened
    ]
    references = [
        "The cat sat on the mat.",
        "Machine learning is powerful.",
        "Natural language processing enables many applications."
    ]

    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        return_per_sample=True,
        show_progress=False
    )

    # Check per-sample scores
    assert results.per_sample_scores is not None, "Per-sample scores should be returned"
    assert len(results.per_sample_scores) == 3, f"Should have 3 samples, got {len(results.per_sample_scores)}"

    # Verify structure
    for i, sample in enumerate(results.per_sample_scores):
        assert 'index' in sample, f"Sample {i} missing index"
        assert sample['index'] == i, f"Sample index mismatch"
        assert 'rouge1' in sample, f"Sample {i} missing rouge1"
        assert 'f1' in sample['rouge1'], f"Sample {i} missing rouge1 f1"
        assert 0.0 <= sample['rouge1']['f1'] <= 1.0, f"Sample {i} rouge1 f1 out of range"

    print(f"✓ Per-sample scores: {len(results.per_sample_scores)} samples tracked correctly")


def test_xsum_format_compatibility():
    """Test compatibility with XSUM dataset format."""
    print("\nTesting XSUM format compatibility...")

    # Simulate XSUM-style data (abstractive summaries)
    xsum_predictions = [
        "Government announces new climate policy measures.",
        "Tech company releases innovative AI product.",
        "Scientists discover potential cure for disease.",
        "Economic growth exceeds expectations this quarter.",
        "Sports team wins championship after dramatic final."
    ]

    xsum_references = [
        "The government has unveiled a comprehensive new climate policy package aimed at reducing emissions.",
        "A major technology company has launched a groundbreaking artificial intelligence product.",
        "Researchers have identified a promising treatment that could cure a previously incurable disease.",
        "The economy has grown faster than predicted in the latest quarterly figures.",
        "The underdog team claimed victory in a thrilling championship final match."
    ]

    # Process XSUM-style data
    results = compute_rouge(
        xsum_predictions,
        xsum_references,
        use_stemmer=True,  # Standard for XSUM
        compute_confidence_intervals=False,
        show_progress=False
    )

    # XSUM typically has moderate ROUGE scores (abstractive nature)
    assert 0.2 <= results.rouge1_f1_mean <= 0.8, f"XSUM ROUGE-1 typically in [0.2, 0.8], got {results.rouge1_f1_mean}"
    assert 0.0 <= results.rouge2_f1_mean <= 0.5, f"XSUM ROUGE-2 typically in [0.0, 0.5], got {results.rouge2_f1_mean}"
    assert 0.1 <= results.rougeL_f1_mean <= 0.7, f"XSUM ROUGE-L typically in [0.1, 0.7], got {results.rougeL_f1_mean}"

    print(f"✓ XSUM format: ROUGE-1={results.rouge1_f1_mean:.4f}, ROUGE-2={results.rouge2_f1_mean:.4f}, ROUGE-L={results.rougeL_f1_mean:.4f}")


def test_multi_model_batch():
    """Test batch processing for multiple models."""
    print("\nTesting multi-model batch processing...")

    references = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning revolutionizes data analysis.",
        "Climate change requires urgent global action."
    ]

    # Simulate outputs from different models
    model1_predictions = [
        "The quick brown fox jumps over the dog.",  # Missing 'lazy'
        "Machine learning transforms data analysis.",  # Similar meaning
        "Climate change needs immediate global action."  # Synonym substitution
    ]

    model2_predictions = [
        "A fast brown fox leaps over a lazy dog.",  # Different determiners
        "ML revolutionizes data analytics.",  # Abbreviation
        "Global warming requires urgent action."  # Related concept
    ]

    model3_predictions = [
        "Fox jumps over dog.",  # Minimal
        "Data analysis improved by ML.",  # Reordered
        "Action needed for climate."  # Shortened
    ]

    # Batch compute
    results = compute_rouge_batch(
        [model1_predictions, model2_predictions, model3_predictions],
        references,
        model_names=["Model-A", "Model-B", "Model-C"],
        compute_confidence_intervals=False,
        show_progress=False
    )

    # Verify results structure
    assert len(results) == 3, f"Should have 3 models, got {len(results)}"
    assert "Model-A" in results, "Model-A missing from results"
    assert "Model-B" in results, "Model-B missing from results"
    assert "Model-C" in results, "Model-C missing from results"

    # Model-A should perform best (most similar to references)
    assert results["Model-A"].rouge1_f1_mean > results["Model-C"].rouge1_f1_mean, \
        "Model-A should outperform Model-C"

    print("✓ Multi-model batch processing:")
    for model_name, model_results in results.items():
        print(f"  {model_name}: ROUGE-1={model_results.rouge1_f1_mean:.4f}, "
              f"ROUGE-2={model_results.rouge2_f1_mean:.4f}, "
              f"ROUGE-L={model_results.rougeL_f1_mean:.4f}")


def compare_with_official_implementation():
    """Compare our implementation with official rouge_score library."""
    print("\nComparing with official rouge_score implementation...")

    predictions = [
        "The cat sat on the mat.",
        "Machine learning is powerful.",
        "Natural language processing enables applications."
    ]
    references = [
        "The cat was sitting on the mat.",
        "Machine learning is a powerful tool.",
        "Natural language processing enables many applications."
    ]

    # Our implementation
    our_results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        show_progress=False
    )

    # Direct official implementation
    official_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    official_scores = []

    for pred, ref in zip(predictions, references):
        scores = official_scorer.score(ref, pred)
        official_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })

    official_rouge1_mean = np.mean([s['rouge1'] for s in official_scores])
    official_rouge2_mean = np.mean([s['rouge2'] for s in official_scores])
    official_rougeL_mean = np.mean([s['rougeL'] for s in official_scores])

    # Compare (should be identical)
    assert abs(our_results.rouge1_f1_mean - official_rouge1_mean) < 1e-10, \
        f"ROUGE-1 mismatch: ours={our_results.rouge1_f1_mean}, official={official_rouge1_mean}"
    assert abs(our_results.rouge2_f1_mean - official_rouge2_mean) < 1e-10, \
        f"ROUGE-2 mismatch: ours={our_results.rouge2_f1_mean}, official={official_rouge2_mean}"
    assert abs(our_results.rougeL_f1_mean - official_rougeL_mean) < 1e-10, \
        f"ROUGE-L mismatch: ours={our_results.rougeL_f1_mean}, official={official_rougeL_mean}"

    print(f"✓ Implementation matches official rouge_score exactly:")
    print(f"  ROUGE-1: ours={our_results.rouge1_f1_mean:.6f}, official={official_rouge1_mean:.6f}")
    print(f"  ROUGE-2: ours={our_results.rouge2_f1_mean:.6f}, official={official_rouge2_mean:.6f}")
    print(f"  ROUGE-L: ours={our_results.rougeL_f1_mean:.6f}, official={official_rougeL_mean:.6f}")


def test_precision_recall_computation():
    """Test that precision and recall are computed correctly."""
    print("\nTesting precision and recall computation...")

    # Create scenario with known precision/recall characteristics
    predictions = [
        "The cat",  # High precision (all words match), low recall (missing words)
        "The cat sat on the mat and the dog.",  # Low precision (extra words), high recall
        "The cat sat on the mat."  # Perfect match
    ]
    references = [
        "The cat sat on the mat.",
        "The cat sat on the mat.",
        "The cat sat on the mat."
    ]

    results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=False,
        return_per_sample=True,
        show_progress=False
    )

    # Check sample 0: high precision, low recall
    sample0 = results.per_sample_scores[0]['rouge1']
    assert sample0['precision'] > sample0['recall'], \
        f"Sample 0 should have precision > recall, got P={sample0['precision']:.2f}, R={sample0['recall']:.2f}"

    # Check sample 1: low precision, high recall
    sample1 = results.per_sample_scores[1]['rouge1']
    assert sample1['recall'] >= sample1['precision'], \
        f"Sample 1 should have recall >= precision, got P={sample1['precision']:.2f}, R={sample1['recall']:.2f}"

    # Check sample 2: perfect match
    sample2 = results.per_sample_scores[2]['rouge1']
    assert abs(sample2['precision'] - 1.0) < 1e-6, f"Perfect match precision should be 1.0, got {sample2['precision']}"
    assert abs(sample2['recall'] - 1.0) < 1e-6, f"Perfect match recall should be 1.0, got {sample2['recall']}"
    assert abs(sample2['f1'] - 1.0) < 1e-6, f"Perfect match F1 should be 1.0, got {sample2['f1']}"

    print("✓ Precision and recall computed correctly")
    print(f"  High precision/low recall: P={sample0['precision']:.3f}, R={sample0['recall']:.3f}")
    print(f"  Low precision/high recall: P={sample1['precision']:.3f}, R={sample1['recall']:.3f}")
    print(f"  Perfect match: P={sample2['precision']:.3f}, R={sample2['recall']:.3f}")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("ROUGE Metrics Implementation Validation")
    print("=" * 60)

    tests = [
        ("Perfect Match", test_perfect_match),
        ("No Overlap", test_no_overlap),
        ("Empty Strings", test_empty_strings),
        ("Unicode Text", test_unicode_text),
        ("Bootstrap CI", test_bootstrap_confidence_intervals),
        ("Batch Efficiency", test_batch_processing_efficiency),
        ("Per-Sample Scores", test_per_sample_scores),
        ("XSUM Format", test_xsum_format_compatibility),
        ("Multi-Model Batch", test_multi_model_batch),
        ("Official Comparison", compare_with_official_implementation),
        ("Precision/Recall", test_precision_recall_computation)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("✓ All validation tests passed successfully!")
        print("✓ ROUGE implementation is correct and efficient")
    else:
        print(f"✗ {failed} tests failed - please review")

    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)