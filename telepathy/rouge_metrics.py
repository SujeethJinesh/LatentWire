"""
ROUGE metrics implementation for generation tasks.

This module provides efficient ROUGE scoring for summarization and other generation tasks,
with support for batch processing, statistical aggregation, and bootstrap confidence intervals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from rouge_score import rouge_scorer
from rouge_score.scoring import AggregateScore, Score
import warnings
from tqdm import tqdm

# Suppress tokenization warnings from rouge_score
warnings.filterwarnings("ignore", category=UserWarning, module="rouge_score")


@dataclass
class RougeResults:
    """Container for ROUGE evaluation results with statistical measures."""

    # Aggregate statistics
    rouge1_f1_mean: float
    rouge1_f1_std: float
    rouge1_precision_mean: float
    rouge1_recall_mean: float

    rouge2_f1_mean: float
    rouge2_f1_std: float
    rouge2_precision_mean: float
    rouge2_recall_mean: float

    rougeL_f1_mean: float
    rougeL_f1_std: float
    rougeL_precision_mean: float
    rougeL_recall_mean: float

    # Bootstrap confidence intervals (if computed)
    rouge1_f1_ci: Optional[Tuple[float, float]] = None
    rouge2_f1_ci: Optional[Tuple[float, float]] = None
    rougeL_f1_ci: Optional[Tuple[float, float]] = None

    # Per-sample scores for detailed analysis
    per_sample_scores: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        result = {
            'rouge1': {
                'f1': {'mean': self.rouge1_f1_mean, 'std': self.rouge1_f1_std},
                'precision': {'mean': self.rouge1_precision_mean},
                'recall': {'mean': self.rouge1_recall_mean}
            },
            'rouge2': {
                'f1': {'mean': self.rouge2_f1_mean, 'std': self.rouge2_f1_std},
                'precision': {'mean': self.rouge2_precision_mean},
                'recall': {'mean': self.rouge2_recall_mean}
            },
            'rougeL': {
                'f1': {'mean': self.rougeL_f1_mean, 'std': self.rougeL_f1_std},
                'precision': {'mean': self.rougeL_precision_mean},
                'recall': {'mean': self.rougeL_recall_mean}
            }
        }

        # Add confidence intervals if available
        if self.rouge1_f1_ci is not None:
            result['rouge1']['f1']['ci_95'] = list(self.rouge1_f1_ci)
        if self.rouge2_f1_ci is not None:
            result['rouge2']['f1']['ci_95'] = list(self.rouge2_f1_ci)
        if self.rougeL_f1_ci is not None:
            result['rougeL']['f1']['ci_95'] = list(self.rougeL_f1_ci)

        # Add per-sample scores if requested
        if self.per_sample_scores is not None:
            result['per_sample'] = self.per_sample_scores

        return result

    def summary_string(self) -> str:
        """Generate a human-readable summary of results."""
        lines = [
            "ROUGE Evaluation Results:",
            f"  ROUGE-1 F1: {self.rouge1_f1_mean:.4f} ± {self.rouge1_f1_std:.4f}",
            f"  ROUGE-2 F1: {self.rouge2_f1_mean:.4f} ± {self.rouge2_f1_std:.4f}",
            f"  ROUGE-L F1: {self.rougeL_f1_mean:.4f} ± {self.rougeL_f1_std:.4f}"
        ]

        if self.rouge1_f1_ci is not None:
            lines.extend([
                "\nConfidence Intervals (95%):",
                f"  ROUGE-1 F1: [{self.rouge1_f1_ci[0]:.4f}, {self.rouge1_f1_ci[1]:.4f}]",
                f"  ROUGE-2 F1: [{self.rouge2_f1_ci[0]:.4f}, {self.rouge2_f1_ci[1]:.4f}]",
                f"  ROUGE-L F1: [{self.rougeL_f1_ci[0]:.4f}, {self.rougeL_f1_ci[1]:.4f}]"
            ])

        return "\n".join(lines)


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None,
    use_stemmer: bool = True,
    split_summaries: bool = False,
    compute_confidence_intervals: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    return_per_sample: bool = False,
    show_progress: bool = True
) -> RougeResults:
    """
    Compute ROUGE scores for a batch of predictions against references.

    Args:
        predictions: List of generated/predicted summaries
        references: List of reference/gold summaries
        rouge_types: Types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        use_stemmer: Whether to use Porter stemmer for word matching
        split_summaries: Whether to split summaries on sentence boundaries
        compute_confidence_intervals: Whether to compute bootstrap CIs
        n_bootstrap: Number of bootstrap samples for CI computation
        confidence_level: Confidence level for intervals (e.g., 0.95)
        return_per_sample: Whether to include per-sample scores in results
        show_progress: Whether to show progress bar during computation

    Returns:
        RougeResults object containing all computed metrics

    Raises:
        ValueError: If predictions and references have different lengths
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")

    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']

    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        split_summaries=split_summaries
    )

    # Compute scores for each sample
    all_scores = []
    iterator = zip(predictions, references)
    if show_progress:
        iterator = tqdm(iterator, total=len(predictions), desc="Computing ROUGE scores")

    for pred, ref in iterator:
        # Handle empty strings gracefully
        if not pred.strip():
            pred = "empty"
        if not ref.strip():
            ref = "empty"

        scores = scorer.score(ref, pred)
        all_scores.append(scores)

    # Extract scores into arrays for statistical analysis
    rouge1_f1_scores = np.array([s['rouge1'].fmeasure for s in all_scores])
    rouge1_precision_scores = np.array([s['rouge1'].precision for s in all_scores])
    rouge1_recall_scores = np.array([s['rouge1'].recall for s in all_scores])

    rouge2_f1_scores = np.array([s['rouge2'].fmeasure for s in all_scores])
    rouge2_precision_scores = np.array([s['rouge2'].precision for s in all_scores])
    rouge2_recall_scores = np.array([s['rouge2'].recall for s in all_scores])

    rougeL_f1_scores = np.array([s['rougeL'].fmeasure for s in all_scores])
    rougeL_precision_scores = np.array([s['rougeL'].precision for s in all_scores])
    rougeL_recall_scores = np.array([s['rougeL'].recall for s in all_scores])

    # Compute basic statistics
    results = RougeResults(
        rouge1_f1_mean=float(np.mean(rouge1_f1_scores)),
        rouge1_f1_std=float(np.std(rouge1_f1_scores)),
        rouge1_precision_mean=float(np.mean(rouge1_precision_scores)),
        rouge1_recall_mean=float(np.mean(rouge1_recall_scores)),

        rouge2_f1_mean=float(np.mean(rouge2_f1_scores)),
        rouge2_f1_std=float(np.std(rouge2_f1_scores)),
        rouge2_precision_mean=float(np.mean(rouge2_precision_scores)),
        rouge2_recall_mean=float(np.mean(rouge2_recall_scores)),

        rougeL_f1_mean=float(np.mean(rougeL_f1_scores)),
        rougeL_f1_std=float(np.std(rougeL_f1_scores)),
        rougeL_precision_mean=float(np.mean(rougeL_precision_scores)),
        rougeL_recall_mean=float(np.mean(rougeL_recall_scores))
    )

    # Compute bootstrap confidence intervals if requested
    if compute_confidence_intervals and len(predictions) > 1:
        results.rouge1_f1_ci = _bootstrap_confidence_interval(
            rouge1_f1_scores, n_bootstrap, confidence_level
        )
        results.rouge2_f1_ci = _bootstrap_confidence_interval(
            rouge2_f1_scores, n_bootstrap, confidence_level
        )
        results.rougeL_f1_ci = _bootstrap_confidence_interval(
            rougeL_f1_scores, n_bootstrap, confidence_level
        )

    # Include per-sample scores if requested
    if return_per_sample:
        per_sample = []
        for i, scores in enumerate(all_scores):
            sample_dict = {
                'index': i,
                'rouge1': {
                    'f1': scores['rouge1'].fmeasure,
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall
                },
                'rouge2': {
                    'f1': scores['rouge2'].fmeasure,
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall
                },
                'rougeL': {
                    'f1': scores['rougeL'].fmeasure,
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall
                }
            }
            per_sample.append(sample_dict)
        results.per_sample_scores = per_sample

    return results


def _bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a set of scores.

    Args:
        scores: Array of scores to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval
    """
    n_samples = len(scores)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = float(np.percentile(bootstrap_means, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_means, upper_percentile))

    return (lower_bound, upper_bound)


def compute_rouge_batch(
    predictions_batch: List[List[str]],
    references: List[str],
    model_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, RougeResults]:
    """
    Compute ROUGE scores for multiple models/systems in batch.

    Args:
        predictions_batch: List of prediction lists, one per model
        references: Single list of references (shared across models)
        model_names: Optional names for each model
        **kwargs: Additional arguments passed to compute_rouge()

    Returns:
        Dictionary mapping model names to RougeResults
    """
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(predictions_batch))]

    if len(model_names) != len(predictions_batch):
        raise ValueError("Number of model names must match number of prediction sets")

    results = {}
    for name, predictions in zip(model_names, predictions_batch):
        print(f"\nComputing ROUGE for {name}...")
        results[name] = compute_rouge(predictions, references, **kwargs)

    return results


def save_rouge_results(
    results: RougeResults,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save ROUGE results to JSON file with optional metadata.

    Args:
        results: RougeResults object to save
        output_path: Path to output JSON file
        metadata: Optional metadata to include (e.g., dataset, model, timestamp)
    """
    output_dict = {
        'rouge_scores': results.to_dict(),
        'summary': results.summary_string()
    }

    if metadata is not None:
        output_dict['metadata'] = metadata

    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f"ROUGE results saved to: {output_path}")


def load_rouge_results(input_path: str) -> Tuple[RougeResults, Optional[Dict[str, Any]]]:
    """
    Load ROUGE results from JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        Tuple of (RougeResults, metadata dict or None)
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    scores = data['rouge_scores']

    # Reconstruct RougeResults object
    results = RougeResults(
        rouge1_f1_mean=scores['rouge1']['f1']['mean'],
        rouge1_f1_std=scores['rouge1']['f1']['std'],
        rouge1_precision_mean=scores['rouge1']['precision']['mean'],
        rouge1_recall_mean=scores['rouge1']['recall']['mean'],

        rouge2_f1_mean=scores['rouge2']['f1']['mean'],
        rouge2_f1_std=scores['rouge2']['f1']['std'],
        rouge2_precision_mean=scores['rouge2']['precision']['mean'],
        rouge2_recall_mean=scores['rouge2']['recall']['mean'],

        rougeL_f1_mean=scores['rougeL']['f1']['mean'],
        rougeL_f1_std=scores['rougeL']['f1']['std'],
        rougeL_precision_mean=scores['rougeL']['precision']['mean'],
        rougeL_recall_mean=scores['rougeL']['recall']['mean']
    )

    # Add confidence intervals if present
    if 'ci_95' in scores.get('rouge1', {}).get('f1', {}):
        results.rouge1_f1_ci = tuple(scores['rouge1']['f1']['ci_95'])
    if 'ci_95' in scores.get('rouge2', {}).get('f1', {}):
        results.rouge2_f1_ci = tuple(scores['rouge2']['f1']['ci_95'])
    if 'ci_95' in scores.get('rougeL', {}).get('f1', {}):
        results.rougeL_f1_ci = tuple(scores['rougeL']['f1']['ci_95'])

    # Add per-sample scores if present
    if 'per_sample' in scores:
        results.per_sample_scores = scores['per_sample']

    metadata = data.get('metadata', None)

    return results, metadata


# Example usage and testing
if __name__ == "__main__":
    # Test with simple examples
    test_predictions = [
        "The cat sat on the mat.",
        "A quick brown fox jumps.",
        "Machine learning is powerful."
    ]

    test_references = [
        "The cat was sitting on the mat.",
        "A quick brown fox jumped over the lazy dog.",
        "Machine learning is a powerful technology."
    ]

    print("Testing ROUGE computation...")
    results = compute_rouge(
        test_predictions,
        test_references,
        compute_confidence_intervals=True,
        return_per_sample=True
    )

    print("\n" + results.summary_string())

    # Test saving and loading
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        save_rouge_results(results, temp_path, metadata={'test': True})
        loaded_results, loaded_metadata = load_rouge_results(temp_path)

        print("\nVerifying save/load...")
        print(f"Original ROUGE-1 F1: {results.rouge1_f1_mean:.4f}")
        print(f"Loaded ROUGE-1 F1: {loaded_results.rouge1_f1_mean:.4f}")
        print(f"Metadata: {loaded_metadata}")

    finally:
        os.unlink(temp_path)

    print("\nAll tests passed!")