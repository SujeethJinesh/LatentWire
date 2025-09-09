import re
import string
from typing import List, Tuple

def _normalize(s: str) -> str:
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

def em(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))

def f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def batch_metrics(preds: List[str], golds: List[str]) -> Tuple[float, float]:
    ems, f1s = [], []
    for p, g in zip(preds, golds):
        ems.append(em(p, g))
        f1s.append(f1(p, g))
    return sum(ems)/len(ems), sum(f1s)/len(f1s)

