#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for verifying all evaluation metrics in LatentWire.

Tests:
1. Accuracy (classification) - SST-2, AG News
2. F1 score (QA) - SQuAD, HotpotQA
3. ROUGE (generation) - XSum
4. Latency metrics (p50, p95, p99)
5. Memory usage tracking
6. Compression ratio calculations

Each test includes:
- Mathematical correctness verification
- Edge case handling (div by zero, empty inputs)
- Aggregation across seeds
- Unit consistency
- Comparison with known ground truth
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple
import unittest
from unittest.mock import patch, MagicMock
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentwire.core_utils import (
    f1, em, squad_em_f1, batch_metrics,
    gsm8k_accuracy, extract_gsm8k_answer,
    compute_wire_metrics
)


class TestAccuracyMetrics(unittest.TestCase):
    """Test classification accuracy computation."""

    def test_basic_accuracy(self):
        """Test simple accuracy calculation."""
        predictions = ["positive", "negative", "positive", "negative"]
        labels = ["positive", "negative", "negative", "positive"]

        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(predictions)

        self.assertEqual(accuracy, 0.5)  # 2/4 correct

    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        predictions = []
        labels = []

        # Should handle gracefully without division by zero
        accuracy = 0.0 if len(predictions) == 0 else sum(1 for p, l in zip(predictions, labels) if p == l) / len(predictions)

        self.assertEqual(accuracy, 0.0)

    def test_per_class_accuracy(self):
        """Test per-class accuracy tracking."""
        predictions = ["positive", "positive", "negative", "negative"]
        labels = ["positive", "negative", "positive", "negative"]

        # Track per class
        pos_preds = [p for p, l in zip(predictions, labels) if l == "positive"]
        pos_correct = sum(1 for p, l in zip(predictions, labels) if l == "positive" and p == "positive")
        pos_total = sum(1 for l in labels if l == "positive")

        neg_preds = [p for p, l in zip(predictions, labels) if l == "negative"]
        neg_correct = sum(1 for p, l in zip(predictions, labels) if l == "negative" and p == "negative")
        neg_total = sum(1 for l in labels if l == "negative")

        pos_acc = pos_correct / max(pos_total, 1)
        neg_acc = neg_correct / max(neg_total, 1)

        self.assertEqual(pos_acc, 0.5)  # 1/2 correct
        self.assertEqual(neg_acc, 0.5)  # 1/2 correct


class TestF1Metrics(unittest.TestCase):
    """Test F1 and EM score computation for QA tasks."""

    def test_f1_exact_match(self):
        """Test F1 score when prediction exactly matches."""
        pred = "The quick brown fox"
        truth = "The quick brown fox"

        score = f1(pred, truth)
        self.assertEqual(score, 1.0)

    def test_f1_partial_match(self):
        """Test F1 score with partial overlap."""
        pred = "The quick brown"
        truth = "The brown fox"

        score = f1(pred, truth)
        # Common tokens: "the", "brown" (normalized to lowercase)
        # Precision: 2/3, Recall: 2/3, F1: 2*2/3*2/3 / (2/3+2/3) = 2/3
        self.assertAlmostEqual(score, 0.666666, places=4)

    def test_f1_no_match(self):
        """Test F1 score with no overlap."""
        pred = "cat dog"
        truth = "fox wolf"

        score = f1(pred, truth)
        self.assertEqual(score, 0.0)

    def test_f1_empty_strings(self):
        """Test F1 handling of empty strings."""
        # Both empty - should be 1.0
        score = f1("", "")
        self.assertEqual(score, 1.0)

        # One empty - should be 0.0
        score = f1("text", "")
        self.assertEqual(score, 0.0)

    def test_em_score(self):
        """Test exact match scoring."""
        # Exact match after normalization
        self.assertEqual(em("The Answer!", "the answer"), 1.0)

        # No match
        self.assertEqual(em("wrong", "right"), 0.0)

    def test_squad_em_f1(self):
        """Test SQuAD-style EM/F1 with multiple references."""
        preds = ["The capital of France", "Berlin", "Madrid"]
        truths = [
            ["The capital of France", "Paris"],
            ["Berlin", "The capital of Germany"],
            ["The capital of Spain", "Madrid"]
        ]

        em_score, f1_score = squad_em_f1(preds, truths)

        # EM: pred[0] matches truth[0][0], pred[1] matches truth[1][0], pred[2] matches truth[2][1]
        self.assertEqual(em_score, 1.0)  # 3/3 exact matches
        self.assertEqual(f1_score, 1.0)  # All have perfect F1


class TestROUGEMetrics(unittest.TestCase):
    """Test ROUGE metric computation."""

    def setUp(self):
        """Set up ROUGE scorer mock."""
        # Import at test time to handle missing dependency gracefully
        try:
            from telepathy.rouge_metrics import compute_rouge, RougeResults
            self.compute_rouge = compute_rouge
            self.RougeResults = RougeResults
        except ImportError:
            self.skipTest("rouge_score not installed")

    def test_rouge_perfect_match(self):
        """Test ROUGE with identical text."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        results = self.compute_rouge(
            predictions, references,
            compute_confidence_intervals=False,
            show_progress=False
        )

        # Perfect match should give ROUGE-1/2/L = 1.0
        self.assertAlmostEqual(results.rouge1_f1_mean, 1.0, places=4)
        self.assertAlmostEqual(results.rouge2_f1_mean, 1.0, places=4)
        self.assertAlmostEqual(results.rougeL_f1_mean, 1.0, places=4)

    def test_rouge_no_match(self):
        """Test ROUGE with no overlap."""
        predictions = ["cat dog fish"]
        references = ["car bike plane"]

        results = self.compute_rouge(
            predictions, references,
            compute_confidence_intervals=False,
            show_progress=False
        )

        # No overlap should give ROUGE = 0.0
        self.assertEqual(results.rouge1_f1_mean, 0.0)
        self.assertEqual(results.rouge2_f1_mean, 0.0)
        self.assertEqual(results.rougeL_f1_mean, 0.0)

    def test_rouge_empty_handling(self):
        """Test ROUGE handling of empty strings."""
        # Should handle gracefully
        predictions = ["", "text"]
        references = ["text", ""]

        results = self.compute_rouge(
            predictions, references,
            compute_confidence_intervals=False,
            show_progress=False
        )

        # Should not crash, scores should be low but valid
        self.assertGreaterEqual(results.rouge1_f1_mean, 0.0)
        self.assertLessEqual(results.rouge1_f1_mean, 1.0)


class TestLatencyMetrics(unittest.TestCase):
    """Test latency percentile calculations."""

    def test_percentile_calculation(self):
        """Test p50, p95, p99 calculation."""
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        self.assertEqual(p50, 55.0)  # Median
        self.assertEqual(p95, 95.5)  # 95th percentile
        self.assertEqual(p99, 99.1)  # 99th percentile

    def test_empty_latencies(self):
        """Test handling of empty latency list."""
        latencies = []

        # Should handle gracefully
        if len(latencies) > 0:
            p50 = np.percentile(latencies, 50)
        else:
            p50 = 0.0

        self.assertEqual(p50, 0.0)

    def test_single_latency(self):
        """Test with single latency value."""
        latencies = [42.5]

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # All percentiles should be the same value
        self.assertEqual(p50, 42.5)
        self.assertEqual(p95, 42.5)
        self.assertEqual(p99, 42.5)


class TestMemoryMetrics(unittest.TestCase):
    """Test memory usage tracking."""

    def test_bytes_to_mb_gb(self):
        """Test conversion between bytes, MB, and GB."""
        bytes_val = 1024 * 1024 * 1024  # 1 GB in bytes

        mb = bytes_val / (1024 * 1024)
        gb = bytes_val / (1024 * 1024 * 1024)

        self.assertEqual(mb, 1024.0)  # 1024 MB
        self.assertEqual(gb, 1.0)  # 1 GB

    def test_tensor_memory_calculation(self):
        """Test memory calculation for tensors."""
        # Float32 tensor: 100x100 = 10,000 elements * 4 bytes = 40,000 bytes
        tensor = torch.randn(100, 100, dtype=torch.float32)
        memory_bytes = tensor.element_size() * tensor.nelement()

        self.assertEqual(memory_bytes, 40000)

        # Float16 tensor: half the size
        tensor_fp16 = tensor.half()
        memory_bytes_fp16 = tensor_fp16.element_size() * tensor_fp16.nelement()

        self.assertEqual(memory_bytes_fp16, 20000)


class TestCompressionMetrics(unittest.TestCase):
    """Test compression ratio calculations."""

    def test_compression_ratio(self):
        """Test basic compression ratio."""
        original_size = 1000  # bytes
        compressed_size = 250  # bytes

        ratio = original_size / compressed_size
        self.assertEqual(ratio, 4.0)  # 4x compression

    def test_wire_metrics(self):
        """Test wire cost computation."""
        llama_prompts = ["This is a test prompt"]
        qwen_prompts = ["This is a test prompt"]
        latents = torch.randn(1, 32, 256)  # [batch, seq_len, hidden]

        metrics = compute_wire_metrics(
            llama_prompts, qwen_prompts, latents,
            group_size=32, scale_bits=16, selected_bits=8
        )

        # Check required fields
        self.assertIn("prompt_chars", metrics)
        self.assertIn("latent_bytes", metrics)
        self.assertIn("latent_shape", metrics)

        # Check calculations
        num_elements = latents.numel()  # 1 * 32 * 256 = 8192
        self.assertEqual(metrics["latent_bytes"]["fp32"], num_elements * 4)
        self.assertEqual(metrics["latent_bytes"]["fp16"], num_elements * 2)

    def test_quantization_overhead(self):
        """Test quantization with scale overhead."""
        latents = torch.randn(2, 64, 128)  # [batch, seq, hidden]

        metrics = compute_wire_metrics(
            ["prompt1", "prompt2"],
            ["prompt1", "prompt2"],
            latents,
            group_size=32,
            scale_bits=16,
            selected_bits=4  # 4-bit quantization
        )

        # Check quantized size calculation
        self.assertIn("selected_latent_bytes", metrics)
        self.assertIsNotNone(metrics["selected_latent_bytes"])

        # Quantized should be smaller than fp16
        self.assertLess(
            metrics["selected_latent_bytes"],
            metrics["latent_bytes"]["fp16"]
        )


class TestAggregationAcrossSeeds(unittest.TestCase):
    """Test metric aggregation across multiple random seeds."""

    def test_mean_std_aggregation(self):
        """Test computing mean and std across seeds."""
        # Results from 3 seeds
        seed_results = [
            {"accuracy": 0.85, "f1": 0.82},
            {"accuracy": 0.87, "f1": 0.84},
            {"accuracy": 0.86, "f1": 0.83},
        ]

        accuracies = [r["accuracy"] for r in seed_results]
        f1_scores = [r["f1"] for r in seed_results]

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        self.assertAlmostEqual(mean_acc, 0.86, places=4)
        self.assertAlmostEqual(mean_f1, 0.83, places=4)

        # Std should be small for consistent results
        self.assertLess(std_acc, 0.02)
        self.assertLess(std_f1, 0.02)

    def test_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        values = np.array([0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.81, 0.82])

        # Bootstrap CI
        n_bootstrap = 1000
        bootstrap_means = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # CI should contain true mean
        true_mean = np.mean(values)
        self.assertGreaterEqual(true_mean, ci_lower)
        self.assertLessEqual(true_mean, ci_upper)


class TestGSM8KMetrics(unittest.TestCase):
    """Test GSM8K-specific metrics."""

    def test_answer_extraction(self):
        """Test extracting numerical answers from GSM8K format."""
        # Standard format
        text1 = "The calculation is 5 * 10 = 50. #### 50"
        self.assertEqual(extract_gsm8k_answer(text1), "50")

        # With decimals
        text2 = "The answer is #### 3.14"
        self.assertEqual(extract_gsm8k_answer(text2), "3.14")

        # Negative number
        text3 = "Result: #### -42"
        self.assertEqual(extract_gsm8k_answer(text3), "-42")

        # No marker, extract last number
        text4 = "First we get 10, then 20, finally 30"
        self.assertEqual(extract_gsm8k_answer(text4), "30")

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy calculation."""
        predictions = ["#### 42", "#### 3.14", "#### 100"]
        truths = ["#### 42", "#### 3.14", "#### 99"]

        accuracy = gsm8k_accuracy(predictions, truths)
        self.assertAlmostEqual(accuracy, 2/3, places=4)  # 2 out of 3 correct


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_mismatched_lengths(self):
        """Test handling of mismatched prediction/label lengths."""
        preds = ["a", "b", "c"]
        labels = ["a", "b"]  # Shorter

        # Should handle by using minimum length
        min_len = min(len(preds), len(labels))
        correct = sum(1 for i in range(min_len) if preds[i] == labels[i])
        accuracy = correct / min_len if min_len > 0 else 0.0

        self.assertEqual(accuracy, 1.0)  # Both match for first 2

    def test_unicode_handling(self):
        """Test metrics with Unicode characters."""
        pred = "café résumé"
        truth = "café résumé"

        score = f1(pred, truth)
        self.assertEqual(score, 1.0)

    def test_very_long_texts(self):
        """Test with very long inputs."""
        # Create long text (10K tokens)
        long_text = " ".join(["word"] * 10000)

        # Should handle without overflow
        score = f1(long_text, long_text)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)