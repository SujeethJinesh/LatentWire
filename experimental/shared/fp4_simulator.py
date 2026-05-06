"""Deterministic low-precision simulation helpers for Mac-local gates.

The functions here emulate cast-and-cast-back quantization in PyTorch. They are
for ranking hypotheses before GPU work, not for claiming native FP4/FP8 kernel
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationResult:
    """Quantized tensor plus metadata useful for audit logs."""

    dequantized: torch.Tensor
    scale: torch.Tensor
    codebook: torch.Tensor | None
    format_name: str
    block_size: int


def _flatten_blocks(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, tuple[int, ...], int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    original_shape = tuple(tensor.shape)
    flat = tensor.float().reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.reshape(-1, block_size), original_shape, pad


def _restore_blocks(blocks: torch.Tensor, original_shape: tuple[int, ...], pad: int) -> torch.Tensor:
    flat = blocks.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(original_shape)


def e2m1_codebook(device: torch.device | None = None) -> torch.Tensor:
    """Return a signed FP4 E2M1-style codebook normalized to max magnitude 6."""

    values = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(values, dtype=torch.float32, device=device)


def block_codebook_quantize(
    tensor: torch.Tensor,
    *,
    codebook: torch.Tensor,
    block_size: int = 32,
    format_name: str = "codebook",
) -> QuantizationResult:
    """Quantize each block by nearest normalized codebook value."""

    blocks, original_shape, pad = _flatten_blocks(tensor, block_size)
    codebook = codebook.to(device=blocks.device, dtype=torch.float32)
    max_code = torch.max(torch.abs(codebook)).clamp_min(1e-8)
    scale = torch.amax(torch.abs(blocks), dim=1, keepdim=True).clamp_min(1e-8) / max_code
    normalized = blocks / scale
    distances = torch.abs(normalized[..., None] - codebook)
    indices = torch.argmin(distances, dim=-1)
    dequantized_blocks = codebook[indices] * scale
    return QuantizationResult(
        dequantized=_restore_blocks(dequantized_blocks, original_shape, pad).to(tensor.dtype),
        scale=scale.squeeze(-1),
        codebook=codebook.detach().cpu(),
        format_name=format_name,
        block_size=block_size,
    )


def simulate_mxfp4_e2m1(tensor: torch.Tensor, *, block_size: int = 32) -> QuantizationResult:
    """Simulate MXFP4 E2M1-style block-scaled quantization."""

    return block_codebook_quantize(
        tensor,
        codebook=e2m1_codebook(tensor.device),
        block_size=block_size,
        format_name="mxfp4_e2m1_sim",
    )


def simulate_symmetric_int(
    tensor: torch.Tensor,
    *,
    bits: int,
    block_size: int = 32,
    format_name: str | None = None,
) -> QuantizationResult:
    """Simulate signed symmetric integer quantization with per-block scales."""

    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = float((2 ** (bits - 1)) - 1)
    blocks, original_shape, pad = _flatten_blocks(tensor, block_size)
    scale = torch.amax(torch.abs(blocks), dim=1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(blocks / scale).clamp(-qmax, qmax)
    dequantized_blocks = quantized * scale
    return QuantizationResult(
        dequantized=_restore_blocks(dequantized_blocks, original_shape, pad).to(tensor.dtype),
        scale=scale.squeeze(-1),
        codebook=None,
        format_name=format_name or f"int{bits}_sym_sim",
        block_size=block_size,
    )


def protect_positions(
    original: torch.Tensor,
    quantized: torch.Tensor,
    *,
    protected_positions: list[int] | tuple[int, ...],
    position_dim: int = -2,
) -> torch.Tensor:
    """Restore selected positions from `original` into a quantized tensor."""

    if original.shape != quantized.shape:
        raise ValueError("original and quantized tensors must have the same shape")
    result = quantized.clone()
    dim = position_dim % original.ndim
    for position in protected_positions:
        index = [slice(None)] * original.ndim
        index[dim] = position
        result[tuple(index)] = original[tuple(index)]
    return result


def gap_recovery_ratio(
    *,
    bf16_score: float,
    uniform_score: float,
    protected_score: float,
    lower_is_better: bool = True,
) -> float:
    """Return the fraction of uniform-quantization degradation that remains.

    Values <= 0.5 mean the protected variant recovered at least half of the
    quality gap introduced by uniform quantization.
    """

    if lower_is_better:
        denominator = uniform_score - bf16_score
        numerator = protected_score - bf16_score
    else:
        denominator = bf16_score - uniform_score
        numerator = bf16_score - protected_score
    if denominator <= 0:
        raise ValueError("uniform_score must be worse than bf16_score")
    return numerator / denominator
