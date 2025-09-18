import os, json, re, string
from typing import List, Dict, Tuple

def _normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score(pred: str, truth: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    truth_tokens = _normalize_answer(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(pred_tokens) == 0 or len(truth_tokens) == 0: return float(pred_tokens == truth_tokens)
    if not common: return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec + 1e-8)

def _em(pred: str, truth: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(truth))

def squad_em_f1(preds: List[str], truths: List[List[str]]) -> Tuple[float, float]:
    em = 0.0; f1 = 0.0
    n = len(preds)
    for p, ts in zip(preds, truths):
        em += max(_em(p, t) for t in ts)
        f1 += max(_f1_score(p, t) for t in ts)
    return em / max(n,1), f1 / max(n,1)

def dump_metrics(path: str, metrics: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
