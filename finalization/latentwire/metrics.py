# -*- coding: utf-8 -*-
"""Metrics module for LatentWire evaluation.

This module provides evaluation metrics including:
- Exact Match (EM)
- F1 Score
- Negative Log-Likelihood (NLL)
- Token-level accuracy
"""

import re
import string
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(pred: str, gold: str) -> float:
    """Compute F1 score between prediction and gold answer."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()

    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_em(pred: str, gold: str) -> float:
    """Compute Exact Match score."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_nll_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Tuple[float, int]:
    """Compute negative log-likelihood per token.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Label value to ignore in loss computation

    Returns:
        Tuple of (total_nll, num_tokens)
    """
    # Flatten for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute cross entropy
    losses = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction='none'
    )

    # Count valid tokens
    valid_mask = (labels_flat != ignore_index)
    num_tokens = valid_mask.sum().item()

    if num_tokens == 0:
        return 0.0, 0

    # Sum losses for valid tokens
    total_nll = losses[valid_mask].sum().item()

    return total_nll, num_tokens


def compute_token_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Compute token-level accuracy.

    Args:
        predictions: Predicted token IDs [batch_size, seq_len]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Label value to ignore

    Returns:
        Token accuracy as a float
    """
    valid_mask = (labels != ignore_index)

    if valid_mask.sum() == 0:
        return 0.0

    correct = (predictions == labels) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().float()

    return accuracy.item()


def compute_first_token_accuracy(
    predictions: List[str],
    gold_answers: List[str],
    tokenizer=None
) -> float:
    """Compute first-token accuracy for generation tasks.

    Args:
        predictions: List of predicted strings
        gold_answers: List of gold answer strings
        tokenizer: Optional tokenizer for token-level comparison

    Returns:
        First-token accuracy as a float
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        if tokenizer is not None:
            # Token-level comparison
            pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
            gold_tokens = tokenizer.encode(gold, add_special_tokens=False)

            if len(pred_tokens) > 0 and len(gold_tokens) > 0:
                if pred_tokens[0] == gold_tokens[0]:
                    correct += 1
        else:
            # Character-level comparison (fallback)
            pred_words = pred.strip().split()
            gold_words = gold.strip().split()

            if len(pred_words) > 0 and len(gold_words) > 0:
                if pred_words[0].lower() == gold_words[0].lower():
                    correct += 1

    return correct / len(predictions)


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple evaluations.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Aggregated metrics with mean and std
    """
    if not metrics_list:
        return {}

    aggregated = {}

    # Collect all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    values.append(value)

        if values:
            import numpy as np
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

    return aggregated