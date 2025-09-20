"""Core utility helpers for LatentWire (prompt building, anchors, data shims)."""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from datasets import load_dataset

LOG = logging.getLogger("latentwire.core_utils")

__all__ = [
    "STOP_STRINGS",
    "clean_pred",
    "SYSTEM_PROMPT",
    "NEUTRAL_SYSTEM_PROMPT",
    "build_chat_prompts",
    "build_neutral_encoder_texts",
    "truncate_chat_to_k_tokens",
    "content_only_m_token_chat_prompt",
    "build_token_budget_prompts",
    "_ByteTokenizerShim",
    "collate_bytes",
    "assistant_header_anchor",
    "make_anchor_text",
    "infer_anchor_mode_and_text",
    "apply_anchor_normalization",
    "apply_anchor_and_bos",
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
    "patch_dataloader_defaults",
    "load_squad_split",
]

# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

STOP_STRINGS = [
    "<|eot_id|>", "<|im_end|>", "</s>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "\n\n\n", "\n\nAssistant:", "\nAssistant:",
]


def clean_pred(s: str) -> str:
    """Normalize short-span generations to a clean answer phrase."""
    if not s:
        return s
    for ss in STOP_STRINGS:
        idx = s.find(ss)
        if idx >= 0:
            s = s[:idx]
    s = re.sub(r"^\s*(assistant|assistant:|Assistant:)\s*", "", s)
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    return s.strip(" \t\r\n.:;,'\"-–—")

# ---------------------------------------------------------------------------
# Prompt builders and token shims
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a concise QA assistant. Use the context to answer with a short phrase only. "
    "Answer in English. Respond with the answer phrase only."
)
NEUTRAL_SYSTEM_PROMPT = "You are a concise QA assistant. Use the context to answer with a short phrase only."


def build_chat_prompts(tokenizer, raw_sources: List[str]) -> List[str]:
    outs = []
    for s in raw_sources:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outs.append(text)
    return outs


def build_neutral_encoder_texts(raw_sources: List[str]) -> List[str]:
    return [f"System: {NEUTRAL_SYSTEM_PROMPT}\nUser: {s}\nAssistant:" for s in raw_sources]


def truncate_chat_to_k_tokens(tokenizer, chat_prompts: List[str], k: int) -> List[str]:
    outs = []
    for cp in chat_prompts:
        enc = tokenizer(cp, add_special_tokens=False, return_attention_mask=False)
        ids_k = enc["input_ids"][:k]
        outs.append(tokenizer.decode(ids_k, skip_special_tokens=True))
    return outs


def content_only_m_token_chat_prompt(tokenizer, raw_source: str, k: int) -> str:
    enc = tokenizer(raw_source, add_special_tokens=True, return_attention_mask=False)
    ids_k = enc["input_ids"][:k]
    truncated_content = tokenizer.decode(ids_k, skip_special_tokens=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": truncated_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_token_budget_prompts(tokenizer, raw_sources: List[str], chat_prompts: List[str], k: int, mode: str) -> List[str]:
    if mode == "chat_full":
        return truncate_chat_to_k_tokens(tokenizer, chat_prompts, k)
    return [content_only_m_token_chat_prompt(tokenizer, s, k) for s in raw_sources]


class _ByteTokenizerShim:
    """Type shim so the signature matches the ByteTokenizer in models."""
    def __init__(self, max_bytes: int = 512):
        self.max_bytes = max_bytes

    def encode(self, text: str) -> torch.Tensor:
        b = text.encode("utf-8")[: self.max_bytes]
        return torch.tensor(list(b), dtype=torch.long)


def collate_bytes(texts: List[str], byte_tok, device: Union[str, torch.device]) -> torch.Tensor:
    ids = [byte_tok.encode(t) for t in texts]
    maxT = max([x.size(0) for x in ids]) if ids else 0
    if maxT == 0:
        return torch.zeros((len(texts), 1), dtype=torch.long, device=device)
    batch = torch.stack([
        torch.cat([x, torch.zeros(maxT - x.size(0), dtype=torch.long)], dim=0)
        for x in ids
    ], dim=0)
    return batch.to(device)

# ---------------------------------------------------------------------------
# Anchors and BOS helpers
# ---------------------------------------------------------------------------


def assistant_header_anchor(tokenizer) -> str:
    """Extract the assistant header string for this model family, if available."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return text or ""
    except Exception:
        return ""


def make_anchor_text(mode: str, wrapper, explicit_text: str) -> str:
    if mode == "none":
        return ""
    if mode == "text":
        return explicit_text or ""
    if mode == "chat":
        return assistant_header_anchor(wrapper.tokenizer)
    raise ValueError(f"Unknown latent_anchor_mode: {mode}")


def infer_anchor_mode_and_text(wrapper, cfg: dict, cli_mode: str, cli_text: str):
    if cli_mode != "auto":
        return cli_mode, cli_text
    train_anchor = (cfg.get("warm_anchor_text") or "").strip()
    if train_anchor:
        return "text", train_anchor
    return "chat", ""


def apply_anchor_normalization(args):  # pragma: no cover - compatibility shim
    return None


def apply_anchor_and_bos(prefix_embeds: torch.Tensor, anchor_embeds: torch.Tensor, append_bos_after_prefix: str = "no"):
    if prefix_embeds.dim() == 2:
        prefix_embeds = prefix_embeds.unsqueeze(0)
    if anchor_embeds.dim() == 2:
        anchor_embeds = anchor_embeds.unsqueeze(0)
    out = torch.cat([prefix_embeds, anchor_embeds], dim=1)
    LOG.debug("[anchor] prefix+anchor -> %s", tuple(out.shape))
    return out

# ---------------------------------------------------------------------------
# Prefix / latent utilities
# ---------------------------------------------------------------------------


def tensor_rms(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    return float(t.detach().float().pow(2).mean().sqrt().item())


def tensor_rms_d(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return t.new_zeros(())
    return t.pow(2).mean().sqrt()


def calibrate_to_embed_rms(prefix: torch.Tensor, wrapper, mode: str = "embed_rms", gain: float = 1.0) -> torch.Tensor:
    if mode == "none" or mode is None:
        return prefix
    target_rms = float(wrapper.input_embedding_rms()) * float(gain)
    if target_rms <= 0:
        return prefix
    current = prefix.float().pow(2).mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-6)
    scale = prefix.new_tensor(target_rms).view(1, 1, 1) / current
    return prefix * scale


def bos_policy(mode: str, anchor_ids: Optional[Union[Sequence[int], torch.Tensor]]) -> Optional[bool]:
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
    device = token_ids.device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    mask = torch.ones_like(token_ids, dtype=torch.bool, device=device) if pad_id is None else token_ids.ne(int(pad_id))
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


def build_scaffold_ids(tokenizer, texts: Sequence[str], anchor_text: str, device: Union[torch.device, str]) -> torch.Tensor:
    anchor = anchor_text + " " if anchor_text and not anchor_text.endswith(" ") else anchor_text
    payloads = [f"{text}{anchor}" for text in texts]
    enc = tokenizer(payloads, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
    return enc["input_ids"].to(device)


def anchor_token_ids(wrapper_or_tokenizer, anchor_text: str) -> List[int]:
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
    shared = components.get("shared")
    private = components.get("private") or {}
    if shared is None or not isinstance(shared, torch.Tensor):
        raise KeyError("components['shared'] must be a tensor")
    if key not in private:
        raise KeyError(f"No private latents for key '{key}'")
    return torch.cat([shared, private[key]], dim=1)


def quantize_dequantize(latents: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    if bits is None or bits <= 0 or bits >= 16:
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
    if anchor_text and not anchor_text.endswith(" "):
        LOG.warning("Anchor text should have trailing space; appending one.")
        anchor_text = anchor_text + " "
    return {"anchor_text": anchor_text, "append_bos_after_prefix": append_bos_after_prefix}

# ---------------------------------------------------------------------------
# Data loader patches
# ---------------------------------------------------------------------------


def patch_dataloader_defaults() -> None:  # pragma: no cover - compatibility shim
    return None


def load_squad_split(split: str = "train", samples: Optional[int] = None):
    ds = load_dataset("squad", split=split)
    if samples is not None:
        ds = ds.select(range(min(len(ds), samples)))
    return ds
