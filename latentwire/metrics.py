# latentwire/metrics.py
# Basic SQuAD-style EM/F1 utilities with light normalization.
import re
import string
from typing import List, Tuple

_ARTICLES = {"a","an","the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # Remove punctuation
    s = s.translate(_PUNC_TABLE)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Drop articles as standalone tokens
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def em(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0

def f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_counts = {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    overlap = 0
    for t in g:
        if p_counts.get(t, 0) > 0:
            overlap += 1
            p_counts[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(p))
    rec  = overlap / max(1, len(g))
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def batch_metrics(preds: List[str], golds: List[str]) -> Tuple[float, float]:
    assert len(preds) == len(golds)
    em_sum = 0.0
    f1_sum = 0.0
    n = len(preds)
    for p, g in zip(preds, golds):
        em_sum += em(p, g)
        f1_sum += f1(p, g)
    return em_sum / n if n else 0.0, f1_sum / n if n else 0.0