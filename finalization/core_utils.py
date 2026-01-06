"""Core utility helpers for LatentWire (prompt building, anchors, data shims)."""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import platform
import re
import string
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    "capture_env_snapshot",
    "capture_stats",
    "batch_metrics",
    "_normalize",
    "em",
    "f1",
    "squad_em_f1",
    "dump_metrics",
    "build_chat_for_qa",
    "apply_lora",
    "maybe_merge_lora",
    "apply_prefix_tuning",
    "apply_prompt_tuning",
    "split_user_and_anchor",
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
    s = re.sub(r"^\s*(answer\s*:)", "", s, flags=re.IGNORECASE)
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    s = re.sub(r"^(<\|assistant\|>|</s>|<\|im_start\|>assistant|Assistant:)+\s*", "", s, flags=re.IGNORECASE)
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
    # Determine the mode (from CLI or config)
    if cli_mode != "auto":
        mode = cli_mode
        # For chat mode, always ignore provided text (will be derived from tokenizer)
        explicit_text = "" if cli_mode == "chat" else cli_text
    else:
        mode = (cfg.get("warm_anchor_mode") or "auto").lower()
        explicit_text = (cfg.get("warm_anchor_text") or "").strip()

    # Apply mode-specific logic (consistent for CLI and config)
    if mode == "none":
        return "none", ""
    if mode == "text":
        return "text", explicit_text
    if mode == "chat":
        # Always derive assistant header from tokenizer; ignore any provided text
        # CRITICAL: Never use explicit_text for chat mode (breaks first-token contract)
        return "chat", ""

    # auto/legacy fallback: prefer explicit text, otherwise chat header
    if explicit_text:
        return "text", explicit_text
    return "chat", ""


def split_user_and_anchor(text: str, anchor_literal: str):
    """Split raw prompt into user text and trailing anchor literal (if present)."""
    if not anchor_literal:
        return text, ""
    marker = anchor_literal.strip()
    raw = text.rstrip()
    if marker and raw.endswith(marker):
        idx = raw.rfind(marker)
        user = raw[:idx].rstrip()
        return user, anchor_literal
    newline_marker = f"\n{marker}"
    if marker and newline_marker in text:
        parts = text.split(newline_marker)
        user = newline_marker.join(parts[:-1]).rstrip()
        return user, anchor_literal
    return text, anchor_literal


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
    enc = tokenizer(payloads, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
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


# ---------------------------------------------------------------------------
# Diagnostics (env snapshot + prefix stats)
# ---------------------------------------------------------------------------


def _tensor_rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt().item())


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_import_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def capture_env_snapshot(out_dir: str, extras=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    snap = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "torch": _safe_import_version("torch"),
            "transformers": _safe_import_version("transformers"),
            "datasets": _safe_import_version("datasets"),
            "sentence_transformers": _safe_import_version("sentence_transformers"),
            "bitsandbytes": _safe_import_version("bitsandbytes"),
        },
        "argv": sys.argv,
    }
    if extras:
        snap.update(extras)
    dst = os.path.join(out_dir, "env_snapshot.json")
    _save_json(dst, snap)
    LOG.info("[core_utils] wrote %s", dst)
    return dst


@torch.no_grad()
def capture_stats(
    run_dir: str,
    model_name: str,
    lm_embed_weight: torch.Tensor,
    adapter_out: torch.Tensor,
    z: torch.Tensor,
    extra: Optional[Dict] = None,
):
    stats = {
        "model": model_name,
        "embed_weight_rms": _tensor_rms(lm_embed_weight),
        "adapter_out_rms": _tensor_rms(adapter_out),
        "z_rms": _tensor_rms(z),
        "adapter_out_mean": float(adapter_out.float().mean()),
        "adapter_out_std": float(adapter_out.float().std()),
        "z_mean": float(z.float().mean()),
        "z_std": float(z.float().std()),
    }
    if extra:
        stats.update(extra)
    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_stats.json")
    _save_json(path, stats)
    LOG.info("[core_utils] wrote %s", path)
    return stats


# ---------------------------------------------------------------------------
# Metrics (EM / F1)
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text: str) -> str:
    return _normalize_answer(text)


def em(pred: str, truth: str) -> float:
    return float(_normalize(pred) == _normalize(truth))


def f1(pred: str, truth: str) -> float:
    pred_tokens = _normalize(pred).split()
    truth_tokens = _normalize(truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    # Use Counter to properly handle token frequencies
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    # Count overlapping tokens (minimum of counts for each token)
    common_count = sum((pred_counter & truth_counter).values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(pred_tokens)
    recall = common_count / len(truth_tokens)
    return 2 * precision * recall / (precision + recall + 1e-8)


def squad_em_f1(preds: List[str], truths: List[List[str]]) -> Tuple[float, float]:
    total_em = 0.0
    total_f1 = 0.0
    n = len(preds)
    for p, ts in zip(preds, truths):
        total_em += max(em(p, t) for t in ts)
        total_f1 += max(f1(p, t) for t in ts)
    denom = max(n, 1)
    return total_em / denom, total_f1 / denom


def batch_metrics(preds: Sequence[str], golds: Sequence[Union[Sequence[str], str]]) -> Tuple[float, float]:
    total_em = 0.0
    total_f1 = 0.0
    count = 0
    for pred, gold in zip(preds, golds):
        refs: List[str] = [gold] if isinstance(gold, str) else list(gold)
        total_em += max(em(pred, ref) for ref in refs)
        total_f1 += max(f1(pred, ref) for ref in refs)
        count += 1
    if count == 0:
        return 0.0, 0.0
    return total_em / count, total_f1 / count


def extract_gsm8k_answer(text: str) -> str:
    """Extract numerical answer from GSM8K response.

    GSM8K answers are in format: 'explanation #### numerical_answer'
    This extracts the numerical part after ####, or tries to find numbers in the text.
    """
    # First try to find #### marker
    if "####" in text:
        parts = text.split("####")
        answer_part = parts[-1].strip()
        # Extract just the number
        import re
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers[0]

    # Fallback: look for numbers in the text
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]  # Return last number found

    return text.strip()


def gsm8k_accuracy(preds: List[str], truths: List[str]) -> float:
    """Compute accuracy for GSM8K by comparing numerical answers."""
    correct = 0
    total = len(preds)

    for pred, truth in zip(preds, truths):
        pred_num = extract_gsm8k_answer(pred)
        truth_num = extract_gsm8k_answer(truth)

        # Compare as strings after normalization
        if pred_num == truth_num:
            correct += 1

    return correct / max(total, 1)


def dump_metrics(path: str, metrics: Dict) -> None:
    _save_json(path, metrics)


# ---------------------------------------------------------------------------
# Chat templating helper (exported for runners if needed)
# ---------------------------------------------------------------------------


def build_chat_for_qa(
    tokenizer,
    question: str,
    context: str,
    system: Optional[str] = None,
    add_generation_prompt: bool = True,
    return_tensors: str = "pt",
) -> Tuple[Dict[str, str], Optional[str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": (
            "Use the context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer in one short span."
        ),
    })

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return model_inputs, prompt_text


# ---------------------------------------------------------------------------
# PEFT integration helpers (LoRA / Prefix / Prompt tuning)
# ---------------------------------------------------------------------------


def _ensure_peft():
    try:
        import peft  # noqa
    except Exception:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "peft>=0.12.0", "accelerate>=0.33.0"]
        )
    import peft  # noqa
    return peft


def _infer_default_targets(model):
    names = set()
    for n, _ in model.named_modules():
        if any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
            names.add(n.split(".")[-1])
    if not names:
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}
    return sorted(list(names))


def _parse_target_modules(arg: Union[str, Iterable[str]], model):
    if isinstance(arg, (list, tuple, set)):
        return list(arg), None
    s = str(arg).strip()
    firstN = None
    if "firstN:" in s:
        s, n = s.split("firstN:", 1)
        try:
            firstN = int(n)
        except Exception:
            firstN = None
    if s == "attn":
        mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif s == "attn_mlp":
        mods = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif s == "auto":
        mods = _infer_default_targets(model)
    else:
        mods = [m.strip() for m in s.split(",") if m.strip()]
    return mods, firstN


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Union[str, Iterable[str]] = "attn_mlp_firstN:16",
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    peft = _ensure_peft()
    from peft import LoraConfig, get_peft_model, TaskType

    tm, firstN = _parse_target_modules(target_modules, model)
    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=tm,
        bias=bias,
        task_type=getattr(TaskType, task_type),
    )
    try:
        kwargs["layers_to_transform"] = list(range(firstN)) if firstN else None
    except Exception:
        pass
    cfg = LoraConfig(**{k: v for k, v in kwargs.items() if v is not None})
    lora_model = get_peft_model(model, cfg)
    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass
    return lora_model


def maybe_merge_lora(model):
    """Merge LoRA adapters into base weights if present; otherwise return as-is."""
    try:
        from peft import PeftModel
    except Exception:
        return model
    if isinstance(model, PeftModel):
        model = model.merge_and_unload(safe_merge=True)
    return model


def apply_prefix_tuning(
    model,
    num_virtual_tokens: int = 16,
    projection: bool = True,
    encoder_hidden_size: Optional[int] = None,
    task_type: str = "CAUSAL_LM",
):
    peft = _ensure_peft()
    from peft import PrefixTuningConfig, get_peft_model, TaskType

    cfg = PrefixTuningConfig(
        task_type=getattr(TaskType, task_type),
        num_virtual_tokens=int(num_virtual_tokens),
        prefix_projection=bool(projection),
        encoder_hidden_size=encoder_hidden_size,
    )
    pt_model = get_peft_model(model, cfg)
    try:
        pt_model.print_trainable_parameters()
    except Exception:
        pass
    return pt_model


def apply_prompt_tuning(
    model,
    num_virtual_tokens: int = 16,
    tokenizer=None,
    task_type: str = "CAUSAL_LM",
):
    peft = _ensure_peft()
    from peft import PromptTuningConfig, get_peft_model, TaskType

    cfg = PromptTuningConfig(
        task_type=getattr(TaskType, task_type),
        num_virtual_tokens=int(num_virtual_tokens),
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
    )
    pm = get_peft_model(model, cfg)
    try:
        pm.print_trainable_parameters()
    except Exception:
        pass
    return pm
