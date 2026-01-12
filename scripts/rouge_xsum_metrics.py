#!/usr/bin/env python3
"""
ROUGE metrics implementation for XSUM summarization tasks.

This module provides a robust and production-ready ROUGE scoring system
specifically designed for XSUM dataset evaluation, with proper error handling,
batch processing, and statistical analysis capabilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import warnings
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try multiple ROUGE implementations for robustness
ROUGE_BACKEND = None

try:
    from rouge_score import rouge_scorer
    from rouge_score.scoring import Score
    ROUGE_BACKEND = 'rouge_score'
    logger.info("Using rouge_score library for ROUGE computation")
except ImportError:
    logger.warning("rouge_score not available, trying evaluate library...")
    try:
        import evaluate
        ROUGE_BACKEND = 'evaluate'
        logger.info("Using evaluate library for ROUGE computation")
    except ImportError:
        logger.error("No ROUGE library available. Install with: pip install rouge-score or pip install evaluate")
        raise ImportError("Please install rouge-score or evaluate library for ROUGE metrics")

# Suppress tokenization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rouge_score")


@dataclass
class XSumRougeResults:
    """Container for XSUM-specific ROUGE evaluation results."""

    # Core ROUGE metrics
    rouge1_f1: float
    rouge1_precision: float
    rouge1_recall: float

    rouge2_f1: float
    rouge2_precision: float
    rouge2_recall: float

    rougeL_f1: float
    rougeL_precision: float
    rougeL_recall: float

    rougeLsum_f1: float
    rougeLsum_precision: float
    rougeLsum_recall: float

    # Statistical measures
    rouge1_f1_std: float = 0.0
    rouge2_f1_std: float = 0.0
    rougeL_f1_std: float = 0.0
    rougeLsum_f1_std: float = 0.0

    # Confidence intervals
    rouge1_f1_ci: Optional[Tuple[float, float]] = None
    rouge2_f1_ci: Optional[Tuple[float, float]] = None
    rougeL_f1_ci: Optional[Tuple[float, float]] = None
    rougeLsum_f1_ci: Optional[Tuple[float, float]] = None

    # Additional metadata
    num_samples: int = 0
    per_sample_scores: Optional[List[Dict[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'metrics': {
                'rouge1': {
                    'f1': self.rouge1_f1,
                    'precision': self.rouge1_precision,
                    'recall': self.rouge1_recall,
                    'f1_std': self.rouge1_f1_std,
                    'f1_ci': self.rouge1_f1_ci
                },
                'rouge2': {
                    'f1': self.rouge2_f1,
                    'precision': self.rouge2_precision,
                    'recall': self.rouge2_recall,
                    'f1_std': self.rouge2_f1_std,
                    'f1_ci': self.rouge2_f1_ci
                },
                'rougeL': {
                    'f1': self.rougeL_f1,
                    'precision': self.rougeL_precision,
                    'recall': self.rougeL_recall,
                    'f1_std': self.rougeL_f1_std,
                    'f1_ci': self.rougeL_f1_ci
                },
                'rougeLsum': {
                    'f1': self.rougeLsum_f1,
                    'precision': self.rougeLsum_precision,
                    'recall': self.rougeLsum_recall,
                    'f1_std': self.rougeLsum_f1_std,
                    'f1_ci': self.rougeLsum_f1_ci
                }
            },
            'num_samples': self.num_samples,
            'per_sample_scores': self.per_sample_scores
        }

    def summary_string(self) -> str:
        """Generate human-readable summary."""
        return f"""
XSUM ROUGE Evaluation Results ({self.num_samples} samples):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ROUGE-1:    F1={self.rouge1_f1:.4f} (±{self.rouge1_f1_std:.4f})  P={self.rouge1_precision:.4f}  R={self.rouge1_recall:.4f}
  ROUGE-2:    F1={self.rouge2_f1:.4f} (±{self.rouge2_f1_std:.4f})  P={self.rouge2_precision:.4f}  R={self.rouge2_recall:.4f}
  ROUGE-L:    F1={self.rougeL_f1:.4f} (±{self.rougeL_f1_std:.4f})  P={self.rougeL_precision:.4f}  R={self.rougeL_recall:.4f}
  ROUGE-Lsum: F1={self.rougeLsum_f1:.4f} (±{self.rougeLsum_f1_std:.4f})  P={self.rougeLsum_precision:.4f}  R={self.rougeLsum_recall:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def compute_rouge_xsum(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True,
    compute_confidence_intervals: bool = True,
    n_bootstrap: int = 1000,
    return_per_sample: bool = False
) -> XSumRougeResults:
    """
    Compute ROUGE scores optimized for XSUM summarization evaluation.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        use_stemmer: Whether to use Porter stemmer (standard for XSUM)
        compute_confidence_intervals: Whether to compute bootstrap CIs
        n_bootstrap: Number of bootstrap samples
        return_per_sample: Whether to include per-sample scores

    Returns:
        XSumRougeResults object with all metrics
    """
    if len(predictions) != len(references):
        raise ValueError(f"Mismatched lengths: {len(predictions)} predictions vs {len(references)} references")

    if not predictions:
        raise ValueError("Empty prediction list")

    # Clean inputs
    predictions = [_clean_text(p) for p in predictions]
    references = [_clean_text(r) for r in references]

    if ROUGE_BACKEND == 'rouge_score':
        return _compute_with_rouge_score(predictions, references, use_stemmer,
                                        compute_confidence_intervals, n_bootstrap, return_per_sample)
    elif ROUGE_BACKEND == 'evaluate':
        return _compute_with_evaluate(predictions, references, use_stemmer,
                                     compute_confidence_intervals, n_bootstrap, return_per_sample)
    else:
        raise RuntimeError("No ROUGE backend available")


def _clean_text(text: str) -> str:
    """Clean and normalize text for ROUGE computation."""
    if not text or not isinstance(text, str):
        return "empty"

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Handle empty after cleaning
    if not text:
        return "empty"

    return text


def _compute_with_rouge_score(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool,
    compute_ci: bool,
    n_bootstrap: int,
    return_per_sample: bool
) -> XSumRougeResults:
    """Compute ROUGE using rouge_score library."""

    # Initialize scorer with XSUM-relevant metrics
    scorer = rouge_scorer.RougeScorer(
        rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        use_stemmer=use_stemmer
    )

    # Compute scores
    all_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        all_scores.append(scores)

    # Extract arrays for statistics
    r1_f1 = np.array([s['rouge1'].fmeasure for s in all_scores])
    r1_p = np.array([s['rouge1'].precision for s in all_scores])
    r1_r = np.array([s['rouge1'].recall for s in all_scores])

    r2_f1 = np.array([s['rouge2'].fmeasure for s in all_scores])
    r2_p = np.array([s['rouge2'].precision for s in all_scores])
    r2_r = np.array([s['rouge2'].recall for s in all_scores])

    rL_f1 = np.array([s['rougeL'].fmeasure for s in all_scores])
    rL_p = np.array([s['rougeL'].precision for s in all_scores])
    rL_r = np.array([s['rougeL'].recall for s in all_scores])

    rLsum_f1 = np.array([s['rougeLsum'].fmeasure for s in all_scores])
    rLsum_p = np.array([s['rougeLsum'].precision for s in all_scores])
    rLsum_r = np.array([s['rougeLsum'].recall for s in all_scores])

    # Create results
    results = XSumRougeResults(
        rouge1_f1=float(np.mean(r1_f1)),
        rouge1_precision=float(np.mean(r1_p)),
        rouge1_recall=float(np.mean(r1_r)),
        rouge1_f1_std=float(np.std(r1_f1)),

        rouge2_f1=float(np.mean(r2_f1)),
        rouge2_precision=float(np.mean(r2_p)),
        rouge2_recall=float(np.mean(r2_r)),
        rouge2_f1_std=float(np.std(r2_f1)),

        rougeL_f1=float(np.mean(rL_f1)),
        rougeL_precision=float(np.mean(rL_p)),
        rougeL_recall=float(np.mean(rL_r)),
        rougeL_f1_std=float(np.std(rL_f1)),

        rougeLsum_f1=float(np.mean(rLsum_f1)),
        rougeLsum_precision=float(np.mean(rLsum_p)),
        rougeLsum_recall=float(np.mean(rLsum_r)),
        rougeLsum_f1_std=float(np.std(rLsum_f1)),

        num_samples=len(predictions)
    )

    # Compute confidence intervals
    if compute_ci and len(predictions) > 1:
        results.rouge1_f1_ci = _bootstrap_ci(r1_f1, n_bootstrap)
        results.rouge2_f1_ci = _bootstrap_ci(r2_f1, n_bootstrap)
        results.rougeL_f1_ci = _bootstrap_ci(rL_f1, n_bootstrap)
        results.rougeLsum_f1_ci = _bootstrap_ci(rLsum_f1, n_bootstrap)

    # Add per-sample scores
    if return_per_sample:
        per_sample = []
        for i, scores in enumerate(all_scores):
            per_sample.append({
                'index': i,
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure,
                'rougeLsum_f1': scores['rougeLsum'].fmeasure
            })
        results.per_sample_scores = per_sample

    return results


def _compute_with_evaluate(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool,
    compute_ci: bool,
    n_bootstrap: int,
    return_per_sample: bool
) -> XSumRougeResults:
    """Compute ROUGE using evaluate library."""
    import evaluate

    # Load ROUGE metric
    rouge = evaluate.load('rouge')

    # Compute scores
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=use_stemmer,
        use_aggregator=False  # Get per-sample scores
    )

    # Convert to numpy arrays
    r1 = np.array(scores['rouge1'])
    r2 = np.array(scores['rouge2'])
    rL = np.array(scores['rougeL'])
    rLsum = np.array(scores['rougeLsum'])

    # Create results
    results = XSumRougeResults(
        rouge1_f1=float(np.mean(r1)),
        rouge1_precision=float(np.mean(r1)),  # evaluate doesn't separate P/R
        rouge1_recall=float(np.mean(r1)),
        rouge1_f1_std=float(np.std(r1)),

        rouge2_f1=float(np.mean(r2)),
        rouge2_precision=float(np.mean(r2)),
        rouge2_recall=float(np.mean(r2)),
        rouge2_f1_std=float(np.std(r2)),

        rougeL_f1=float(np.mean(rL)),
        rougeL_precision=float(np.mean(rL)),
        rougeL_recall=float(np.mean(rL)),
        rougeL_f1_std=float(np.std(rL)),

        rougeLsum_f1=float(np.mean(rLsum)),
        rougeLsum_precision=float(np.mean(rLsum)),
        rougeLsum_recall=float(np.mean(rLsum)),
        rougeLsum_f1_std=float(np.std(rLsum)),

        num_samples=len(predictions)
    )

    # Compute confidence intervals
    if compute_ci and len(predictions) > 1:
        results.rouge1_f1_ci = _bootstrap_ci(r1, n_bootstrap)
        results.rouge2_f1_ci = _bootstrap_ci(r2, n_bootstrap)
        results.rougeL_f1_ci = _bootstrap_ci(rL, n_bootstrap)
        results.rougeLsum_f1_ci = _bootstrap_ci(rLsum, n_bootstrap)

    # Add per-sample scores
    if return_per_sample:
        per_sample = []
        for i in range(len(predictions)):
            per_sample.append({
                'index': i,
                'rouge1_f1': float(r1[i]),
                'rouge2_f1': float(r2[i]),
                'rougeL_f1': float(rL[i]),
                'rougeLsum_f1': float(rLsum[i])
            })
        results.per_sample_scores = per_sample

    return results


def _bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    n = len(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return (float(lower), float(upper))


def evaluate_xsum_model(
    model_predictions: List[str],
    gold_summaries: List[str],
    model_name: str = "model",
    output_dir: Optional[Path] = None,
    save_results: bool = True
) -> XSumRougeResults:
    """
    Evaluate a model on XSUM data and optionally save results.

    Args:
        model_predictions: List of generated summaries
        gold_summaries: List of reference summaries
        model_name: Name of the model being evaluated
        output_dir: Directory to save results
        save_results: Whether to save results to disk

    Returns:
        XSumRougeResults object
    """
    logger.info(f"Evaluating {model_name} on {len(model_predictions)} XSUM samples...")

    # Compute ROUGE scores
    results = compute_rouge_xsum(
        predictions=model_predictions,
        references=gold_summaries,
        use_stemmer=True,  # Standard for XSUM
        compute_confidence_intervals=True,
        return_per_sample=False  # Set to True for detailed analysis
    )

    # Print summary
    print(results.summary_string())

    # Save results if requested
    if save_results and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{model_name}_rouge_results.json"

        with open(output_file, 'w') as f:
            json.dump({
                'model': model_name,
                'results': results.to_dict(),
                'summary': results.summary_string()
            }, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    return results


def compare_models_xsum(
    model_predictions_dict: Dict[str, List[str]],
    gold_summaries: List[str],
    output_dir: Optional[Path] = None
) -> Dict[str, XSumRougeResults]:
    """
    Compare multiple models on XSUM evaluation.

    Args:
        model_predictions_dict: Dict mapping model names to predictions
        gold_summaries: List of reference summaries
        output_dir: Directory to save comparison results

    Returns:
        Dict mapping model names to results
    """
    results = {}

    for model_name, predictions in model_predictions_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print('='*50)

        results[model_name] = evaluate_xsum_model(
            predictions,
            gold_summaries,
            model_name,
            output_dir,
            save_results=True
        )

    # Create comparison table
    print("\n" + "="*70)
    print("XSUM Model Comparison Summary")
    print("="*70)
    print(f"{'Model':<20} {'ROUGE-1':<15} {'ROUGE-2':<15} {'ROUGE-L':<15}")
    print("-"*70)

    for model_name, res in results.items():
        r1 = f"{res.rouge1_f1:.4f}±{res.rouge1_f1_std:.4f}"
        r2 = f"{res.rouge2_f1:.4f}±{res.rouge2_f1_std:.4f}"
        rL = f"{res.rougeL_f1:.4f}±{res.rougeL_f1_std:.4f}"
        print(f"{model_name:<20} {r1:<15} {r2:<15} {rL:<15}")

    print("="*70)

    # Save comparison
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_file = output_dir / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            comparison_data = {
                model: res.to_dict() for model, res in results.items()
            }
            json.dump(comparison_data, f, indent=2)

        logger.info(f"Comparison saved to {comparison_file}")

    return results


# Testing functionality
if __name__ == "__main__":
    # Test with sample XSUM-style data
    test_predictions = [
        "Scientists discover new treatment for rare disease",
        "Government announces new climate policy measures",
        "Tech company reports record quarterly earnings"
    ]

    test_references = [
        "Researchers have found a breakthrough treatment for a rare genetic disorder affecting thousands",
        "The government unveiled comprehensive climate change policies including renewable energy targets",
        "Technology giant posts best-ever quarterly results exceeding analyst expectations"
    ]

    print("Testing ROUGE implementation for XSUM...")
    results = compute_rouge_xsum(
        predictions=test_predictions,
        references=test_references,
        compute_confidence_intervals=True,
        return_per_sample=True
    )

    print(results.summary_string())

    if results.per_sample_scores:
        print("\nPer-sample scores:")
        for score in results.per_sample_scores:
            print(f"  Sample {score['index']}: R1={score['rouge1_f1']:.3f}, R2={score['rouge2_f1']:.3f}")

    print("\n✅ ROUGE implementation test completed successfully!")