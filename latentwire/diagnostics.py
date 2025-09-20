import os
import sys
import json
import logging
import platform
import datetime
import re
import string
from typing import Dict, List, Sequence, Tuple, Union

import torch

LOG = logging.getLogger("latentwire.diagnostics")

__all__ = [
    "capture_env_snapshot",
    "capture_stats",
    "batch_metrics",
    "_normalize",
    "em",
    "f1",
    "squad_em_f1",
    "dump_metrics",
]


def _tensor_rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt().item())


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_import_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def capture_env_snapshot(out_dir: str, extras=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    snap = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "torch": _safe_import_version("torch"),
            "transformers": _safe_import_version("transformers"),
            "datasets": _safe_import_version("datasets"),
            "sentence_transformers": _safe_import_version("sentence_transformers"),
            "bitsandbytes": _safe_import_version("bitsandbytes"),
        },
        "argv": sys.argv,
    }
    if extras:
        snap.update(extras)
    dst = os.path.join(out_dir, "env_snapshot.json")
    _save_json(dst, snap)
    LOG.info(f"[diag] wrote {dst}")
    return dst


@torch.no_grad()
def capture_stats(run_dir: str, model_name: str, lm_embed_weight: torch.Tensor, adapter_out: torch.Tensor, z: torch.Tensor, extra: Dict=None):
    stats = {
        "model": model_name,
        "embed_weight_rms": _tensor_rms(lm_embed_weight),
        "adapter_out_rms": _tensor_rms(adapter_out),
        "z_rms": _tensor_rms(z),
        "adapter_out_mean": float(adapter_out.float().mean()),
        "adapter_out_std": float(adapter_out.float().std()),
        "z_mean": float(z.float().mean()),
        "z_std": float(z.float().std()),
    }
    if extra:
        stats.update(extra)
    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_stats.json")
    _save_json(path, stats)
    LOG.info(f"[diag] wrote {path}")
    return stats


# ---------------------------------------------------------------------------
# Metrics helpers (EM / F1)
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text: str) -> str:
    return _normalize_answer(text)


def em(pred: str, truth: str) -> float:
    return float(_normalize(pred) == _normalize(truth))


def f1(pred: str, truth: str) -> float:
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
