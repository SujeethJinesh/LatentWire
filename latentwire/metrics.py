"""Evaluation helpers for EM / F1 style QA metrics."""

import os
import json
import re
import string
from typing import Dict, List, Sequence, Tuple, Union

__all__ = [
    "batch_metrics",
    "_normalize",
    "em",
    "f1",
    "squad_em_f1",
    "dump_metrics",
]

def _normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _normalize(text: str) -> str:
    return _normalize_answer(text)


def em(pred: str, truth: str) -> float:
    """Exact-match score."""
    return float(_normalize(pred) == _normalize(truth))


def f1(pred: str, truth: str) -> float:
    """Token-level F1 using normalized answers."""
    pred_tokens = _normalize(pred).split()
    truth_tokens = _normalize(truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall + 1e-8)

def squad_em_f1(preds: List[str], truths: List[List[str]]) -> Tuple[float, float]:
    total_em = 0.0
    total_f1 = 0.0
    n = len(preds)
    for p, ts in zip(preds, truths):
        total_em += max(em(p, t) for t in ts)
        total_f1 += max(f1(p, t) for t in ts)
    denom = max(n, 1)
    return total_em / denom, total_f1 / denom


def batch_metrics(preds: Sequence[str], golds: Sequence[Union[Sequence[str], str]]) -> Tuple[float, float]:
    """Compute EM/F1 for batch predictions against references."""
    total_em = 0.0
    total_f1 = 0.0
    count = 0
    for pred, gold in zip(preds, golds):
        refs: List[str]
        if isinstance(gold, str):
            refs = [gold]
        else:
            refs = list(gold)
        total_em += max(em(pred, ref) for ref in refs)
        total_f1 += max(f1(pred, ref) for ref in refs)
        count += 1
    if count == 0:
        return 0.0, 0.0
    return total_em / count, total_f1 / count

def dump_metrics(path: str, metrics: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
