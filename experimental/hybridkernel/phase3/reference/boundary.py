"""CPU reference for a hybrid attention/SSM boundary blend primitive.

This is a small semantic primitive for Phase 4 kernel correctness. It is not a
claim that the full hybrid model boundary is implemented or optimized.
"""

from __future__ import annotations

import torch


def hybrid_boundary_blend_reference(
    attention_state: torch.Tensor,
    ssm_state: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Blend attention and SSM states at a layer boundary.

    All tensors must be broadcast-identical after flattening. The formula is:

    out = gate * attention_state + (1 - gate) * ssm_state + bias
    """

    if attention_state.shape != ssm_state.shape:
        raise ValueError("attention_state and ssm_state must have the same shape")
    if gate.shape != attention_state.shape:
        raise ValueError("gate must have the same shape as attention_state")
    if bias.shape != attention_state.shape:
        raise ValueError("bias must have the same shape as attention_state")

    attention_state = attention_state.to(torch.float32)
    ssm_state = ssm_state.to(torch.float32)
    gate = gate.to(torch.float32)
    bias = bias.to(torch.float32)
    return gate * attention_state + (1.0 - gate) * ssm_state + bias
