"""Prefix and anchor utilities shared between training and evaluation."""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch

LOG = logging.getLogger("latentwire.prefix_utils")

__all__ = [
    "tensor_rms",
    "tensor_rms_d",
    "calibrate_to_embed_rms",
    "bos_policy",
    "first_non_bos",
    "build_scaffold_ids",
    "anchor_token_ids",
    "combine_latents",
    "quantize_dequantize",
    "compute_wire_metrics",
    "build_anchor_prefix_text",
]


def tensor_rms(t: torch.Tensor) -> float:
    """Return the root-mean-square of *t* as a Python float (detached)."""
    if t.numel() == 0:
        return 0.0
    return float(t.detach().float().pow(2).mean().sqrt().item())


def tensor_rms_d(t: torch.Tensor) -> torch.Tensor:
    """Differentiable RMS of *t* (keeps gradients)."""
    if t.numel() == 0:
        return t.new_zeros(())
    return t.pow(2).mean().sqrt()


def calibrate_to_embed_rms(
    prefix: torch.Tensor,
    wrapper,
    mode: str = "embed_rms",
    gain: float = 1.0,
) -> torch.Tensor:
    """Scale *prefix* so each example matches the LM embedding RMS."""
    if mode == "none" or mode is None:
        return prefix
    target_rms = float(wrapper.input_embedding_rms()) * float(gain)
    if target_rms <= 0:
        return prefix
    current = prefix.float().pow(2).mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-6)
    scale = prefix.new_tensor(target_rms).view(1, 1, 1) / current
    return prefix * scale


def bos_policy(mode: str, anchor_ids: Optional[Union[Sequence[int], torch.Tensor]]) -> Optional[bool]:
    """Determine whether to append BOS after the latent prefix."""
    mode = (mode or "auto").lower()
    has_anchor = anchor_ids is not None and len(anchor_ids) > 0
    if mode == "yes":
        return True
    if mode == "no":
        return False
    if mode == "auto":
        return not has_anchor
    raise ValueError(f"Unknown BOS policy '{mode}'")


def first_non_bos(tokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Extract the first content token (skip PAD/BOS)."""
    device = token_ids.device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)

    if pad_id is None:
        mask = torch.ones_like(token_ids, dtype=torch.bool, device=device)
    else:
        mask = token_ids.ne(int(pad_id))

    first_idx = mask.float().argmax(dim=1)
    batch = torch.arange(token_ids.size(0), device=device)
    first_tok = token_ids[batch, first_idx]

    if bos_id is not None:
        bos_id = int(bos_id)
        need_next = first_tok.eq(bos_id)
        if need_next.any():
            next_idx = torch.clamp(first_idx + 1, max=token_ids.size(1) - 1)
            first_tok = torch.where(need_next, token_ids[batch, next_idx], first_tok)
    return first_tok


def build_scaffold_ids(
    tokenizer,
    texts: Sequence[str],
    anchor_text: str,
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Tokenize teacher prompts for KD (prompt + anchor)."""
    if anchor_text and not anchor_text.endswith(" "):
        anchor = anchor_text + " "
    else:
        anchor = anchor_text
    payloads = [f"{text}{anchor}" for text in texts]
    enc = tokenizer(
        payloads,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    return enc["input_ids"].to(device)


def anchor_token_ids(wrapper_or_tokenizer, anchor_text: str) -> List[int]:
    """Encode the anchor string without adding generation tokens."""
    if not anchor_text:
        return []
    text = anchor_text if anchor_text.endswith(" ") else anchor_text + " "
    if hasattr(wrapper_or_tokenizer, "_encode_anchor_text"):
        return list(wrapper_or_tokenizer._encode_anchor_text(text))  # type: ignore[attr-defined]
    tokenizer = getattr(wrapper_or_tokenizer, "tokenizer", wrapper_or_tokenizer)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        ids = enc.get("input_ids", [])
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
    return list(ids)


def combine_latents(components: Dict[str, torch.Tensor | Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    """Concatenate shared and private latents for a specific model *key*."""
    shared = components.get("shared")
    private = components.get("private") or {}
    if shared is None or not isinstance(shared, torch.Tensor):
        raise KeyError("components['shared'] must be a tensor")
    if key not in private:
        raise KeyError(f"No private latents for key '{key}'")
    return torch.cat([shared, private[key]], dim=1)


def quantize_dequantize(latents: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    """Symmetric per-group linear quantization followed by dequantization."""
    if bits is None or bits <= 0:
        return latents
    if bits >= 16:
        return latents
    if group_size <= 0:
        group_size = latents.size(-1)

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    out = latents.clone()
    flat = out.view(out.size(0), -1)
    elems = flat.size(1)

    for start in range(0, elems, group_size):
        end = min(start + group_size, elems)
        chunk = flat[:, start:end]
        scale = chunk.abs().amax(dim=1, keepdim=True) / max(qmax, 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(chunk / scale), qmin, qmax)
        chunk.copy_(q * scale)
    return out.view_as(latents)


def compute_wire_metrics(
    llama_prompts: Sequence[str],
    qwen_prompts: Sequence[str],
    latents: torch.Tensor,
    *,
    group_size: int = 32,
    scale_bits: int = 16,
    selected_bits: Optional[int] = None,
) -> Dict[str, object]:
    """Estimate payload sizes for transmitting latent prefixes."""
    num_latent = latents.numel()
    metrics: Dict[str, object] = {
        "prompt_chars": {
            "llama": sum(len(p) for p in llama_prompts),
            "qwen": sum(len(p) for p in qwen_prompts),
        },
        "prompt_count": len(llama_prompts),
        "latent_shape": list(latents.shape),
        "latent_bytes": {
            "fp32": int(num_latent * 4),
            "fp16": int(num_latent * 2),
        },
        "group_size": int(group_size),
        "scale_bits": int(scale_bits),
    }

    if selected_bits is not None and selected_bits > 0:
        B, M, D = latents.shape
        values_per_sample = M * D
        groups_per_sample = math.ceil(values_per_sample / max(group_size, 1))
        scale_bytes = groups_per_sample * scale_bits / 8.0 * B
        data_bytes = num_latent * selected_bits / 8.0
        total_bytes = int(math.ceil(scale_bytes + data_bytes))
        metrics["selected_bits"] = int(selected_bits)
        metrics["selected_latent_bytes"] = total_bytes
        metrics["latent_bytes"]["quantized"] = int(math.ceil(data_bytes))
        metrics["latent_bytes"]["quantized_with_scales"] = total_bytes
    else:
        metrics["selected_latent_bytes"] = None
    return metrics


def build_anchor_prefix_text(anchor_text: str = "Answer: ", append_bos_after_prefix: str = "no") -> Dict[str, object]:
    """Utility kept for backwards compatibility with older CLI flows."""
    if anchor_text and not anchor_text.endswith(" "):
        LOG.warning("Anchor text should have trailing space; appending one.")
        anchor_text = anchor_text + " "
    return {"anchor_text": anchor_text, "append_bos_after_prefix": append_bos_after_prefix}
