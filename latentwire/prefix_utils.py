import math
from typing import Any, Dict, List, Optional, Sequence

import torch


def calibrate_to_embed_rms(prefix: torch.Tensor, wrapper) -> torch.Tensor:
    """Scale each example to match the wrapper's input embedding RMS."""
    target = prefix.new_tensor(wrapper.input_embedding_rms())
    cur = prefix.float().pow(2).mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-8)
    gain = (target / cur).to(prefix.dtype)
    return prefix * gain


def bos_policy(flag: str, anchor_ids: Sequence[int]) -> Optional[bool]:
    """Resolve user BOS policy flag into a boolean understood by LMWrapper."""
    if flag == "auto":
        return None
    return flag == "yes"


def first_non_bos(tokenizer, ids: torch.Tensor) -> torch.Tensor:
    """Return the first non-padding, non-BOS token per row."""
    device = ids.device
    pad = getattr(tokenizer, "pad_token_id", None)
    bos = getattr(tokenizer, "bos_token_id", None)

    if pad is None:
        non_pad = torch.ones_like(ids, dtype=torch.bool, device=device)
    else:
        non_pad = ids.ne(int(pad))

    first_idx = non_pad.float().argmax(dim=1)
    row = torch.arange(ids.size(0), device=device)
    first_tok = ids[row, first_idx]

    if bos is not None:
        is_bos = first_tok.eq(int(bos))
        next_idx = torch.clamp(first_idx + 1, max=ids.size(1) - 1)
        first_tok = torch.where(is_bos, ids[row, next_idx], first_tok)

    return first_tok


def build_scaffold_ids(tokenizer, texts: Sequence[str], anchor_text: str, device: str) -> torch.Tensor:
    suffix = anchor_text or ""
    combo = [t + suffix for t in texts]
    enc = tokenizer(combo, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
    return enc["input_ids"].to(device)


def anchor_token_ids(wrapper, text: Optional[str]) -> List[int]:
    if text:
        return wrapper._encode_anchor_text(text)
    return []


def tensor_rms(x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(x.float().pow(2).mean().sqrt().item())


def tensor_rms_d(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()


def combine_latents(encoded: Dict[str, torch.Tensor], model_key: str) -> torch.Tensor:
    shared = encoded.get("shared")
    private_dict = encoded.get("private", {})
    parts = []
    if shared is not None:
        parts.append(shared)
    if model_key in private_dict:
        parts.append(private_dict[model_key])
    if parts:
        return torch.cat(parts, dim=1)
    raise KeyError(f"No latent components found for model key '{model_key}'")


@torch.no_grad()
def quantize_dequantize(Z: torch.Tensor, bits: Optional[int], group_size: int = 32) -> torch.Tensor:
    """Symmetric per-group quantize/dequantize helper used at eval time."""
    if bits is None or bits >= 16:
        return Z
    if bits not in (8, 6, 4):
        raise ValueError(f"Unsupported quantization bits: {bits}")

    qmax = (1 << (bits - 1)) - 1
    flat = Z.detach().to(torch.float32).contiguous().view(-1)
    if flat.numel() == 0:
        return Z

    out = flat.clone()
    gs = max(1, int(group_size))
    for start in range(0, flat.numel(), gs):
        end = min(start + gs, flat.numel())
        seg = flat[start:end]
        amax = seg.abs().max()
        scale = (amax / qmax) if float(amax) > 0.0 else torch.tensor(1e-8, dtype=seg.dtype, device=seg.device)
        q = torch.clamp(torch.round(seg / scale), min=-qmax, max=qmax)
        out[start:end] = q * scale
    return out.view_as(Z).to(Z.dtype)


def _latent_bits_count(M: int, d_latent: int, bits_per_param: int,
                       group_size: Optional[int] = None, scale_bits: int = 16) -> int:
    core = int(M) * int(d_latent) * int(bits_per_param)
    overhead = 0
    if group_size and group_size > 0:
        groups = math.ceil((int(M) * int(d_latent)) / int(group_size))
        overhead = groups * int(scale_bits)
    header_bits = 8 * 16  # 16 bytes metadata/header
    return core + overhead + header_bits


def latent_bytes_dict(M: int, d_latent: int, group_size: int = 32, scale_bits: int = 16) -> Dict[str, int]:
    return {
        "fp32": math.ceil(_latent_bits_count(M, d_latent, 32) / 8),
        "fp16": math.ceil(_latent_bits_count(M, d_latent, 16) / 8),
        "int8": math.ceil(_latent_bits_count(M, d_latent, 8, group_size, scale_bits) / 8),
        "int6": math.ceil(_latent_bits_count(M, d_latent, 6, group_size, scale_bits) / 8),
        "int4": math.ceil(_latent_bits_count(M, d_latent, 4, group_size, scale_bits) / 8),
    }


def compute_wire_metrics(
    llama_chat_prompts: Sequence[str],
    qwen_chat_prompts: Sequence[str],
    Z: torch.Tensor,
    group_size: int = 32,
    scale_bits: int = 16,
    selected_bits: Optional[int] = None,
) -> Dict[str, Any]:
    avg_text_bytes_llama = (
        int(sum(len(p.encode("utf-8")) for p in llama_chat_prompts) / max(1, len(llama_chat_prompts)))
        if llama_chat_prompts else 0
    )
    avg_text_bytes_qwen = (
        int(sum(len(p.encode("utf-8")) for p in qwen_chat_prompts) / max(1, len(qwen_chat_prompts)))
        if qwen_chat_prompts else 0
    )
    M = int(Z.size(1))
    d = int(Z.size(2))
    lat_bytes = latent_bytes_dict(M, d, group_size=group_size, scale_bits=scale_bits)
    max_onecopy = max(avg_text_bytes_llama, avg_text_bytes_qwen)

    selected_key = {16: "fp16", 8: "int8", 6: "int6", 4: "int4"}.get(selected_bits)
    selected = lat_bytes.get(selected_key) if selected_key else None

    def _safe(x: Optional[int], y: int) -> Optional[float]:
        return (float(x) / float(y)) if (x and y) else None

    return {
        "text_bytes_onecopy": {
            "llama_avg": avg_text_bytes_llama,
            "qwen_avg": avg_text_bytes_qwen,
            "max_avg": max_onecopy,
        },
        "text_bytes_twocopies": {"sum_avg": avg_text_bytes_llama + avg_text_bytes_qwen},
        "latent_bytes": lat_bytes,
        "selected_latent_bytes": selected,
        "wire_ratio": {
            "latent_over_onecopy_fp16": _safe(lat_bytes["fp16"], max_onecopy),
            "latent_over_onecopy_fp32": _safe(lat_bytes["fp32"], max_onecopy),
            "selected_over_onecopy": _safe(selected, max_onecopy) if selected else None,
        },
    }
