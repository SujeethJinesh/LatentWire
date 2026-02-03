"""KVComm-like baseline: token-level selection heuristics."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from rosetta.utils.kv_select import compute_token_scores, select_topk


def select_tokens(k: torch.Tensor, v: torch.Tensor, proportion: float, mode: str = "vnorm_topk") -> Tuple[torch.Tensor, Dict[str, float]]:
    scores = compute_token_scores(k, v, mode=mode)
    idx = select_topk(scores, proportion=proportion, min_tokens=1)
    stats = {
        "selected_tokens": int(idx.numel()),
        "total_tokens": int(k.shape[2]),
    }
    if idx.numel() > 0:
        sel = scores[idx]
        stats["score_mean"] = float(sel.mean().item())
    return idx, stats
