"""Consolidated metrics module for LatentWire evaluation.

This module combines metrics from core_utils.py and rouge_xsum_metrics.py
to provide a unified interface for computing evaluation metrics.
"""

from typing import List, Tuple, Dict, Any, Union, Sequence
import re
import string
from collections import Counter

# Import ROUGE metrics if available
try:
    from rouge_xsum_metrics import (
        compute_rouge_scores,
        compute_rouge_with_stemming,
        RougeScorer,
        _compute_rouge_from_lists,
        _preprocess_text,
        _compute_lcs,
        _rouge_l_score
    )
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# ---------------------------------------------------------------------------
# String normalization for metrics
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """
    Normalize a string for metric calculation.

    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Collapse whitespace
    """
    s = s.lower()

    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)

    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))

    # Collapse whitespace
    s = ' '.join(s.split())

    return s

# ---------------------------------------------------------------------------
# Exact Match (EM) metric
# ---------------------------------------------------------------------------

def em(pred: str, truth: str) -> float:
    """
    Compute exact match score between prediction and truth.

    Args:
        pred: Predicted string
        truth: Ground truth string

    Returns:
        1.0 if normalized strings match exactly, 0.0 otherwise
    """
    return float(_normalize(pred) == _normalize(truth))

# ---------------------------------------------------------------------------
# F1 Score metric
# ---------------------------------------------------------------------------

def f1(pred: str, truth: str) -> float:
    """
    Compute token-level F1 score between prediction and truth.

    Args:
        pred: Predicted string
        truth: Ground truth string

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = _normalize(pred).split()
    truth_tokens = _normalize(truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    # Use Counter to properly handle token frequencies
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    # Count overlapping tokens (minimum of counts for each token)
    common_count = sum((pred_counter & truth_counter).values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(pred_tokens)
    recall = common_count / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)

# ---------------------------------------------------------------------------
# SQuAD-style EM/F1 (handles multiple reference answers)
# ---------------------------------------------------------------------------

def squad_em_f1(preds: List[str], truths: List[List[str]]) -> Tuple[float, float]:
    """
    Compute SQuAD-style EM and F1 scores with multiple reference answers.

    Args:
        preds: List of predictions
        truths: List of lists of acceptable answers

    Returns:
        Tuple of (EM score, F1 score)
    """
    total_em = 0.0
    total_f1 = 0.0
    n = len(preds)

    for p, ts in zip(preds, truths):
        # Take max score across all reference answers
        total_em += max(em(p, t) for t in ts)
        total_f1 += max(f1(p, t) for t in ts)

    denom = max(n, 1)
    return total_em / denom, total_f1 / denom

# ---------------------------------------------------------------------------
# Batch metrics computation
# ---------------------------------------------------------------------------

def batch_metrics(
    preds: Sequence[str],
    golds: Sequence[Union[Sequence[str], str]]
) -> Tuple[float, float]:
    """
    Compute batch EM and F1 metrics.

    Args:
        preds: Predictions
        golds: Gold answers (can be single string or list of strings per example)

    Returns:
        Tuple of (EM score, F1 score)
    """
    total_em = 0.0
    total_f1 = 0.0
    count = 0

    for pred, gold in zip(preds, golds):
        if isinstance(gold, str):
            gold = [gold]

        # Take max score across all reference answers
        em_score = max(em(pred, g) for g in gold)
        f1_score = max(f1(pred, g) for g in gold)

        total_em += em_score
        total_f1 += f1_score
        count += 1

    if count == 0:
        return 0.0, 0.0

    return total_em / count, total_f1 / count

# ---------------------------------------------------------------------------
# ROUGE metrics wrapper
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: List[str],
    references: List[str],
    use_stemming: bool = False
) -> Dict[str, float]:
    """
    Compute ROUGE scores for predictions vs references.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        use_stemming: Whether to use Porter stemming

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    if not ROUGE_AVAILABLE:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

    if use_stemming:
        return compute_rouge_with_stemming(predictions, references)
    else:
        return compute_rouge_scores(predictions, references)

# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------

def accuracy(preds: List[str], truths: List[str]) -> float:
    """
    Compute accuracy for classification tasks.

    Args:
        preds: List of predictions
        truths: List of ground truth labels

    Returns:
        Accuracy score between 0 and 1
    """
    if not preds or not truths:
        return 0.0

    correct = sum(1 for p, t in zip(preds, truths) if p.strip().lower() == t.strip().lower())
    return correct / len(preds)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def dump_metrics(path: str, metrics: Dict[str, Any]) -> None:
    """Save metrics to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# All-in-one metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    task_type: str = "qa"
) -> Dict[str, float]:
    """
    Compute all relevant metrics based on task type.

    Args:
        predictions: Model predictions
        references: Ground truth (can be multiple per example for QA)
        task_type: One of "qa", "summarization", "classification"

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}

    if task_type == "qa":
        # Handle multiple reference answers
        if references and isinstance(references[0], str):
            references = [[r] for r in references]

        em_score, f1_score = squad_em_f1(predictions, references)
        metrics["em"] = em_score
        metrics["f1"] = f1_score

    elif task_type == "summarization":
        # Flatten references if needed
        flat_refs = []
        for r in references:
            if isinstance(r, list):
                flat_refs.append(r[0] if r else "")
            else:
                flat_refs.append(r)

        if ROUGE_AVAILABLE:
            rouge_scores = compute_rouge(predictions, flat_refs)
            metrics.update(rouge_scores)

        # Also compute F1 for summarization
        em_score, f1_score = batch_metrics(predictions, references)
        metrics["f1"] = f1_score

    elif task_type == "classification":
        # For classification, compute accuracy
        flat_refs = []
        for r in references:
            if isinstance(r, list):
                flat_refs.append(r[0] if r else "")
            else:
                flat_refs.append(r)

        metrics["accuracy"] = accuracy(predictions, flat_refs)

    else:
        # Default to QA-style metrics
        if references and isinstance(references[0], str):
            references = [[r] for r in references]

        em_score, f1_score = squad_em_f1(predictions, references)
        metrics["em"] = em_score
        metrics["f1"] = f1_score

    return metrics

# Export main functions
__all__ = [
    "em",
    "f1",
    "squad_em_f1",
    "batch_metrics",
    "compute_rouge",
    "accuracy",
    "compute_all_metrics",
    "dump_metrics",
    "load_metrics",
    "_normalize"
]