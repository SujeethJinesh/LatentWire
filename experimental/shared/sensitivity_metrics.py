"""Small metrics used by Mac-local preregistered gates."""

from __future__ import annotations

import torch


def rel_l2(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-8) -> float:
    return float(torch.linalg.norm(reference.float() - candidate.float()) / torch.linalg.norm(reference.float()).clamp_min(eps))


def max_abs(tensor: torch.Tensor) -> float:
    return float(torch.amax(torch.abs(tensor.float())))


def kurtosis(tensor: torch.Tensor, eps: float = 1e-8) -> float:
    values = tensor.float().reshape(-1)
    centered = values - values.mean()
    variance = torch.mean(centered * centered).clamp_min(eps)
    fourth = torch.mean(centered ** 4)
    return float(fourth / (variance ** 2))


def symmetric_kl_from_logits(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    ref_logp = torch.log_softmax(reference_logits.float(), dim=-1)
    cand_logp = torch.log_softmax(candidate_logits.float(), dim=-1)
    ref_p = ref_logp.exp()
    cand_p = cand_logp.exp()
    kl_ref_cand = torch.sum(ref_p * (ref_logp - cand_logp), dim=-1)
    kl_cand_ref = torch.sum(cand_p * (cand_logp - ref_logp), dim=-1)
    return float(torch.mean(0.5 * (kl_ref_cand + kl_cand_ref)))


def spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel():
        raise ValueError("x and y must contain the same number of elements")
    if x.numel() < 2:
        raise ValueError("need at least two values")
    x_rank = torch.argsort(torch.argsort(x.float().reshape(-1))).float()
    y_rank = torch.argsort(torch.argsort(y.float().reshape(-1))).float()
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = torch.linalg.norm(x_centered) * torch.linalg.norm(y_centered)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(x_centered, y_centered) / denom)
