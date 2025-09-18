from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def first_token_ce_loss(logits: torch.Tensor, target_ids: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    CE on the first generated position only. 
    logits: [B, T, V]; target_ids: [B, T]
    """
    if logits.size(1) == 0:
        return torch.tensor(0.0, device=logits.device)
    first_logits = logits[:, 0, :]        # [B, V]
    first_target = target_ids[:, 0]       # [B]
    return weight * F.cross_entropy(first_logits, first_target, reduction="mean")

def k_ce_loss(logits: torch.Tensor, target_ids: torch.Tensor, k: int = 6, weight: float = 1.0) -> torch.Tensor:
    """
    Cross-entropy on the first k positions (or up to T if shorter).
    """
    T = min(k, logits.size(1))
    if T == 0:
        return torch.tensor(0.0, device=logits.device)
    loss = F.cross_entropy(logits[:, :T, :].reshape(-1, logits.size(-1)),
                           target_ids[:, :T].reshape(-1), reduction="mean")
    return weight * loss

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float = 1.0, weight: float = 1.0) -> torch.Tensor:
    """
    KL(student || teacher) with temperature tau, averaged over time.
    """
    if student_logits.size(1) == 0:
        return torch.tensor(0.0, device=student_logits.device)
    s = F.log_softmax(student_logits / tau, dim=-1)
    t = F.softmax(teacher_logits / tau, dim=-1)
    kl = F.kl_div(s, t, reduction="batchmean") * (tau * tau)
    return weight * kl

def l2_on_scale(z: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0, device=z.device)
    return weight * (z.float().pow(2).mean())

def manifold_stat_reg(z: torch.Tensor, target_mean: float = 0.0, target_std: float = 0.01, weight: float = 0.0) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0, device=z.device)
    m = z.float().mean()
    s = z.float().std().clamp_min(1e-6)
    return weight * ((m - target_mean).abs() + (s - target_std).abs())
