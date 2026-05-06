"""Reference SinkKV policies for Mac-local preregistered gates.

These helpers emulate policy behavior only. They do not implement native FP4
packing, GPU kernels, or any shortcut that skips query-dependent sink attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experimental.shared.fp4_simulator import (
    protect_positions,
    simulate_mxfp4_e2m1,
    simulate_symmetric_int,
)


@dataclass(frozen=True)
class SinkKVPolicyResult:
    key: torch.Tensor
    value: torch.Tensor
    budget_bits_per_element: float
    protected_positions: tuple[int, ...]
    policy_name: str


def _position_mask(seq_len: int, positions: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for position in positions:
        if position < 0 or position >= seq_len:
            raise ValueError(f"protected position {position} is outside sequence length {seq_len}")
        mask[position] = True
    return mask


def uniform_mxfp4_kv(key: torch.Tensor, value: torch.Tensor, *, block_size: int = 32) -> SinkKVPolicyResult:
    """Quantize all K/V positions with simulated MXFP4."""

    return SinkKVPolicyResult(
        key=simulate_mxfp4_e2m1(key, block_size=block_size).dequantized,
        value=simulate_mxfp4_e2m1(value, block_size=block_size).dequantized,
        budget_bits_per_element=4.0,
        protected_positions=(),
        policy_name="uniform_mxfp4_kv",
    )


def budget_matched_protected_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    protected_positions: tuple[int, ...],
    block_size: int = 32,
) -> SinkKVPolicyResult:
    """Protect selected positions while matching a 4-bit average budget.

    Protected positions are restored from full precision. To keep the average
    element budget equal to uniform 4-bit storage, a deterministic subset of
    unprotected positions uses simulated 4-bit quantization and the remainder
    uses simulated 3-bit quantization. This is a reference accounting policy for
    Mac gates, not a proposed production packing format.
    """

    if key.shape != value.shape:
        raise ValueError("key and value tensors must have the same shape")
    seq_len = key.shape[-2]
    protected = tuple(sorted(protected_positions))
    if not protected:
        return uniform_mxfp4_kv(key, value, block_size=block_size)

    unprotected_count = seq_len - len(protected)
    if unprotected_count <= 0:
        raise ValueError("at least one unprotected position is required")

    # Solve 16*p + 4*m + 3*(u-m) = 4*(p+u), where p is protected count and
    # u is unprotected count. Rounding down never exceeds the 4-bit budget.
    protected_count = len(protected)
    four_bit_tail_count = max(0, min(unprotected_count, seq_len - 13 * protected_count))

    protected_mask = _position_mask(seq_len, protected, device=key.device)
    unprotected_positions = [idx for idx in range(seq_len) if not bool(protected_mask[idx])]
    four_bit_positions = set(unprotected_positions[:four_bit_tail_count])

    key_4 = simulate_mxfp4_e2m1(key, block_size=block_size).dequantized
    value_4 = simulate_mxfp4_e2m1(value, block_size=block_size).dequantized
    key_3 = simulate_symmetric_int(key, bits=3, block_size=block_size).dequantized
    value_3 = simulate_symmetric_int(value, bits=3, block_size=block_size).dequantized

    mixed_key = key_3.clone()
    mixed_value = value_3.clone()
    for position in four_bit_positions:
        index = [slice(None)] * key.ndim
        index[-2] = position
        mixed_key[tuple(index)] = key_4[tuple(index)]
        mixed_value[tuple(index)] = value_4[tuple(index)]

    mixed_key = protect_positions(key, mixed_key, protected_positions=protected, position_dim=-2)
    mixed_value = protect_positions(value, mixed_value, protected_positions=protected, position_dim=-2)

    total_bits = 16 * protected_count + 4 * four_bit_tail_count + 3 * (unprotected_count - four_bit_tail_count)
    budget_bits = total_bits / seq_len

    return SinkKVPolicyResult(
        key=mixed_key,
        value=mixed_value,
        budget_bits_per_element=budget_bits,
        protected_positions=protected,
        policy_name="protected_budget_matched_kv",
    )
