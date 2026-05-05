"""CPU reference for anchor/phase-protected low-bit retention.

This is a Phase 2/4 semantic primitive for ThoughtFlow-FP8. It tests the policy
core: preserve anchor and phase-transition positions, keep high-importance
positions, and quantize retained values. It is not a native FP8 kernel.
"""

from __future__ import annotations

import torch


def anchor_phase_quantize_reference(
    values: torch.Tensor,
    importance: torch.Tensor,
    anchor_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    *,
    threshold: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int8 quantized values and boolean keep mask."""

    if values.shape != importance.shape:
        raise ValueError("values and importance must have the same shape")
    if anchor_mask.shape != values.shape:
        raise ValueError("anchor_mask must match values")
    if phase_mask.shape != values.shape:
        raise ValueError("phase_mask must match values")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    keep = anchor_mask.bool() | phase_mask.bool() | (importance.to(torch.float32) >= threshold)
    scaled = values.to(torch.float32) / scale
    rounded = torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5)
    quantized = torch.clamp(rounded, -127, 127).to(torch.int8)
    return torch.where(keep, quantized, torch.zeros_like(quantized)), keep
