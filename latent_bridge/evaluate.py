"""
Evaluate RotAlign-KV and baseline communication protocols.

Supports:
  * Multiple-choice JSONL:
      {"question": "...", "choices": ["A", "B"], "answer": 0}
  * Generation / exact-match JSONL:
      {"question": "...", "answer": "42"}
      {"prompt": "...", "answer_text": "...", "aliases": ["..."]}

The key difference from the earlier harness is that RotAlign now supports
multiple communication protocols:
  * translated_only  - translated source KV only
  * fused            - target KV fused with translated source KV
  * text_kv_hybrid   - text hint plus fused KV
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import pathlib
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rotalign import RotAlignKVTranslator


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class MCQExample:
    question: str
    choices: list[str]
    answer: int


@dataclass
class GenerationExample:
    prompt: str
    answers: list[str]


@dataclass
class PrefixState:
    past_key_values: Any
    last_token: torch.Tensor
    prefix_len: int


@dataclass
class GenerationTrace:
    text: str
    num_generated_tokens: int
    ttft_sec: float
    elapsed_sec: float


def load_mcq(path: str) -> list[MCQExample]:
    items: list[MCQExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                MCQExample(
                    question=obj["question"],
                    choices=list(obj["choices"]),
                    answer=int(obj["answer"]),
                )
            )
    return items


def load_generation(path: str) -> list[GenerationExample]:
    items: list[GenerationExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("question")
            if prompt is None:
                raise ValueError("Generation examples require `prompt` or `question`")
            raw_answer = obj.get("answer_text", obj.get("answer", obj.get("target")))
            if raw_answer is None:
                raise ValueError("Generation examples require an answer field")
            aliases = obj.get("aliases", [])
            answers = [str(raw_answer), *[str(alias) for alias in aliases]]
            items.append(GenerationExample(prompt=str(prompt), answers=answers))
    return items


def _stable_example_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _mcq_example_id(example: MCQExample) -> str:
    return _stable_example_id(
        {
            "question": example.question,
            "choices": example.choices,
            "answer": example.answer,
        }
    )


def _generation_example_id(example: GenerationExample) -> str:
    return _stable_example_id(
        {
            "prompt": example.prompt,
            "answers": example.answers,
        }
    )


def infer_task_type(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            return "mcq" if "choices" in obj else "generation"
    raise ValueError(f"No examples found in {path}")


def load_prompt_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _optional_bool_from_arg(value: str) -> bool | None:
    lowered = value.lower()
    if lowered == "auto":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Unknown tri-state bool value: {value}")


def _format_prompt_for_tokenizer(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
    enable_thinking: bool | None,
) -> str:
    prompt = prompt.strip()
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _normalize_cache(pkv: Any) -> tuple | None:
    if pkv is None:
        return None
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()
    return tuple((layer[0], layer[1]) for layer in pkv)


def _cache_seq_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].shape[2])


def _cache_for_model(model, past_key_values: Any):
    if past_key_values is None or hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return past_key_values
    try:
        return DynamicCache.from_legacy_cache(past_key_values)
    except Exception:
        try:
            return DynamicCache(past_key_values)
        except Exception:
            return past_key_values


def _ones(length: int, device: str) -> torch.Tensor:
    return torch.ones((1, length), device=device, dtype=torch.long)


def _split_prompt_prefix(prompt_ids: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
    if prompt_ids.shape[1] == 0:
        raise ValueError("Prompt tokenized to an empty sequence")
    if prompt_ids.shape[1] == 1:
        return None, prompt_ids[:, -1:]
    return prompt_ids[:, :-1], prompt_ids[:, -1:]


@torch.no_grad()
def _prepare_prefix_state(
    model,
    tokenizer,
    prompt: str,
    device: str,
    *,
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
) -> PrefixState:
    formatted_prompt = _format_prompt_for_tokenizer(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    prefix_ids, last_token = _split_prompt_prefix(prompt_ids)
    if prefix_ids is None:
        return PrefixState(past_key_values=None, last_token=last_token, prefix_len=1)
    out = model(input_ids=prefix_ids, attention_mask=_ones(prefix_ids.shape[1], device), use_cache=True)
    return PrefixState(
        past_key_values=out.past_key_values,
        last_token=last_token,
        prefix_len=prompt_ids.shape[1],
    )


@torch.no_grad()
def _step_with_past(model, token_ids: torch.Tensor, past_key_values: tuple | None, device: str):
    model_cache = _cache_for_model(model, past_key_values)
    prefix_len = _cache_seq_length(model_cache)
    out = model(
        input_ids=token_ids,
        attention_mask=_ones(prefix_len + token_ids.shape[1], device),
        past_key_values=model_cache,
        use_cache=True,
    )
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def _last_token_attention_maps(
    model,
    token_ids: torch.Tensor,
    past_key_values: Any,
    device: str,
    *,
    translator: RotAlignKVTranslator | None = None,
) -> list[torch.Tensor]:
    model_cache = _cache_for_model(model, past_key_values)
    prefix_len = _cache_seq_length(model_cache)
    out = model(
        input_ids=token_ids,
        attention_mask=_ones(prefix_len + token_ids.shape[1], device),
        past_key_values=model_cache,
        use_cache=False,
        output_attentions=True,
    )
    attentions = getattr(out, "attentions", None)
    if attentions is None:
        raise ValueError("Target model did not return attentions for attention-based position selection")

    layer_maps: list[torch.Tensor] = []
    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            raise ValueError(f"Missing attention tensor for layer {layer_idx}")
        if attn.shape[-1] < prefix_len:
            raise ValueError(
                f"Attention tensor for layer {layer_idx} has length {attn.shape[-1]} < prefix length {prefix_len}"
            )
        score = attn[0, :, -1, :prefix_len].float()
        if translator is not None and layer_idx < translator.config.num_tgt_layers:
            target_heads = translator.config.tgt_num_heads
            if score.shape[0] != target_heads and score.shape[0] % target_heads == 0:
                group = score.shape[0] // target_heads
                score = score.view(target_heads, group, prefix_len).mean(dim=1)
            head_mask = translator.head_selected_mask[layer_idx].to(device=score.device)
            if head_mask.numel() == score.shape[0] and bool(head_mask.any()):
                score = score[head_mask]
        layer_maps.append(score)
    return layer_maps


@torch.no_grad()
def _last_token_attention_scores(
    model,
    token_ids: torch.Tensor,
    past_key_values: Any,
    device: str,
    *,
    translator: RotAlignKVTranslator | None = None,
) -> list[torch.Tensor]:
    return [
        score.mean(dim=0)
        for score in _last_token_attention_maps(
            model,
            token_ids,
            past_key_values,
            device,
            translator=translator,
        )
    ]


def _model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Unsupported model architecture for query extraction")


@torch.no_grad()
def _last_token_query_heads(
    model,
    token_ids: torch.Tensor,
    past_key_values: Any,
    device: str,
    *,
    translator: RotAlignKVTranslator,
) -> list[torch.Tensor]:
    model_cache = _cache_for_model(model, past_key_values)
    prefix_len = _cache_seq_length(model_cache)
    out = model(
        input_ids=token_ids,
        attention_mask=_ones(prefix_len + token_ids.shape[1], device),
        past_key_values=model_cache,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Target model did not return hidden_states for QK-fidelity scoring")
    layers = _model_layers(model)
    query_heads: list[torch.Tensor] = []
    for layer_idx in range(min(len(layers), len(hidden_states) - 1, translator.config.num_tgt_layers)):
        layer = layers[layer_idx]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            attn = getattr(layer, "attention", None)
        if attn is None or not hasattr(attn, "q_proj"):
            raise ValueError(f"Layer {layer_idx} does not expose a q_proj module")
        hidden = hidden_states[layer_idx][:, -1, :]
        q = attn.q_proj(hidden).float()
        head_dim = int(getattr(attn, "head_dim", translator.config.tgt_head_dim))
        num_query_heads = getattr(attn, "num_heads", None)
        if num_query_heads is None:
            num_query_heads = getattr(attn, "num_attention_heads", None)
        if num_query_heads is None:
            num_query_heads = q.shape[-1] // max(head_dim, 1)
        num_query_heads = int(num_query_heads)
        if q.shape[-1] % num_query_heads != 0:
            raise ValueError(
                f"Layer {layer_idx} query projection width {q.shape[-1]} is not divisible by num_query_heads={num_query_heads}"
            )
        head_dim = q.shape[-1] // num_query_heads
        q = q.view(q.shape[0], num_query_heads, head_dim)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q)
        target_heads = translator.config.tgt_num_heads
        if q.shape[1] != target_heads and q.shape[1] % target_heads == 0:
            group = q.shape[1] // target_heads
            q = q.view(q.shape[0], target_heads, group, head_dim).mean(dim=2)
        if q.shape[1] != target_heads:
            raise ValueError(
                f"Layer {layer_idx} query head count {q.shape[1]} does not match target KV heads {target_heads}"
            )
        query_heads.append(q[0].detach())
    return query_heads


def _translator_selected_head_indices(
    translator: RotAlignKVTranslator,
    tgt_layer_idx: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.nonzero(
        translator.head_selected_mask[tgt_layer_idx].to(device=device),
        as_tuple=False,
    ).flatten()
    if indices.numel() == 0:
        return torch.arange(translator.config.tgt_num_heads, device=device)
    return indices


def _selected_head_count(head_count: int, ratio: float) -> int:
    if head_count <= 0:
        return 0
    if ratio >= 1.0:
        return head_count
    if ratio <= 0.0:
        return 1
    return max(1, min(head_count, int(round(head_count * ratio))))


def _runtime_head_scores(
    attention_map: torch.Tensor,
    *,
    metric: str,
    layer_idx: int,
) -> torch.Tensor:
    scores = attention_map.float()
    probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if metric == "attention_peak":
        return scores.max(dim=-1).values
    if metric == "attention_entropy":
        return (probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    if metric == "attention_margin":
        if probs.shape[-1] <= 1:
            return probs.max(dim=-1).values.log()
        top2 = torch.topk(probs, k=2, dim=-1, largest=True).values.clamp_min(1e-8)
        return top2[:, 0].log() - top2[:, 1].log()
    if metric == "retrieval_peak":
        if scores.shape[-1] <= 1:
            return scores.max(dim=-1).values
        positions = torch.linspace(0.0, 1.0, steps=scores.shape[-1], device=scores.device, dtype=scores.dtype)
        centroid = (probs * positions).sum(dim=-1)
        peak = probs.max(dim=-1).values
        return centroid * peak
    if metric == "random":
        gen = torch.Generator(device="cpu").manual_seed(91_001 + int(layer_idx))
        return torch.rand(scores.shape[0], generator=gen, dtype=torch.float32).to(device=scores.device)
    raise ValueError(f"Unknown runtime_head_selection_metric: {metric}")


def _expected_attention_head_scores(
    attention_map: torch.Tensor,
    position_prior: torch.Tensor,
) -> torch.Tensor:
    probs = attention_map.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    prior = _resample_position_profile(position_prior.to(device=probs.device), probs.shape[-1]).view(1, -1)
    return (probs * prior).sum(dim=-1)


def _attention_fidelity_head_scores(
    attention_map: torch.Tensor,
    target_keys: torch.Tensor,
    translated_keys: torch.Tensor,
) -> torch.Tensor:
    if target_keys.ndim == 4:
        if target_keys.shape[0] != 1:
            raise ValueError(f"target_keys batch dimension must be 1, got shape {tuple(target_keys.shape)}")
        target_keys = target_keys.squeeze(0)
    if translated_keys.ndim == 4:
        if translated_keys.shape[0] != 1:
            raise ValueError(
                f"translated_keys batch dimension must be 1, got shape {tuple(translated_keys.shape)}"
            )
        translated_keys = translated_keys.squeeze(0)
    if attention_map.ndim != 2:
        raise ValueError(f"attention_map must be rank-2 [heads, positions], got shape {tuple(attention_map.shape)}")
    if target_keys.ndim != 3 or translated_keys.ndim != 3:
        raise ValueError(
            "target_keys and translated_keys must be rank-3 [heads, positions, dim], "
            f"got {tuple(target_keys.shape)} and {tuple(translated_keys.shape)}"
        )
    if target_keys.shape[0] != attention_map.shape[0] or translated_keys.shape[0] != attention_map.shape[0]:
        raise ValueError(
            "attention_map, target_keys, and translated_keys must agree on head count, "
            f"got {attention_map.shape[0]}, {target_keys.shape[0]}, and {translated_keys.shape[0]}"
        )
    prefix_len = min(attention_map.shape[-1], target_keys.shape[1], translated_keys.shape[1])
    if prefix_len <= 0:
        raise ValueError("attention_fidelity requires at least one shared prefix position")
    probs = attention_map[:, -prefix_len:].float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = target_keys[:, -prefix_len:, :].float()
    trans = translated_keys[:, -prefix_len:, :].float()
    cosine = F.cosine_similarity(trans, tgt, dim=-1).clamp(-1.0, 1.0)
    fidelity = 0.5 * (cosine + 1.0)
    return (probs * fidelity).sum(dim=-1)


def _attention_procrustes_head_scores(
    attention_map: torch.Tensor,
    target_keys: torch.Tensor,
    translated_keys: torch.Tensor,
) -> torch.Tensor:
    if target_keys.ndim == 4:
        if target_keys.shape[0] != 1:
            raise ValueError(f"target_keys batch dimension must be 1, got shape {tuple(target_keys.shape)}")
        target_keys = target_keys.squeeze(0)
    if translated_keys.ndim == 4:
        if translated_keys.shape[0] != 1:
            raise ValueError(
                f"translated_keys batch dimension must be 1, got shape {tuple(translated_keys.shape)}"
            )
        translated_keys = translated_keys.squeeze(0)
    if attention_map.ndim != 2:
        raise ValueError(f"attention_map must be rank-2 [heads, positions], got shape {tuple(attention_map.shape)}")
    if target_keys.ndim != 3 or translated_keys.ndim != 3:
        raise ValueError(
            "target_keys and translated_keys must be rank-3 [heads, positions, dim], "
            f"got {tuple(target_keys.shape)} and {tuple(translated_keys.shape)}"
        )
    if target_keys.shape[0] != attention_map.shape[0] or translated_keys.shape[0] != attention_map.shape[0]:
        raise ValueError(
            "attention_map, target_keys, and translated_keys must agree on head count, "
            f"got {attention_map.shape[0]}, {target_keys.shape[0]}, and {translated_keys.shape[0]}"
        )
    prefix_len = min(attention_map.shape[-1], target_keys.shape[1], translated_keys.shape[1])
    if prefix_len <= 0:
        raise ValueError("attention_procrustes requires at least one shared prefix position")
    probs = attention_map[:, -prefix_len:].float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = target_keys[:, -prefix_len:, :].float()
    trans = translated_keys[:, -prefix_len:, :].float()
    scores: list[torch.Tensor] = []
    for head_idx in range(probs.shape[0]):
        weight = probs[head_idx].sqrt().unsqueeze(-1)
        tgt_head = tgt[head_idx]
        trans_head = trans[head_idx]
        tgt_mean = (probs[head_idx].unsqueeze(-1) * tgt_head).sum(dim=0, keepdim=True)
        trans_mean = (probs[head_idx].unsqueeze(-1) * trans_head).sum(dim=0, keepdim=True)
        x = weight * (tgt_head - tgt_mean)
        y = weight * (trans_head - trans_mean)
        if float(x.norm().item()) < 1e-8 or float(y.norm().item()) < 1e-8:
            scores.append(torch.tensor(0.0, device=tgt.device))
            continue
        cross = x @ y.transpose(0, 1)
        # Per-head cross matrices are tiny; CPU SVD avoids missing MPS kernels
        # without materially affecting runtime.
        cross_cpu = cross.to("cpu")
        nuclear_norm = torch.linalg.svdvals(cross_cpu).sum().to(tgt.device)
        score = nuclear_norm / (x.norm() * y.norm()).clamp_min(1e-8)
        scores.append(score.clamp_min(0.0))
    return torch.stack(scores)


def _attention_qk_fidelity_head_scores(
    query_heads: torch.Tensor,
    target_keys: torch.Tensor,
    translated_keys: torch.Tensor,
) -> torch.Tensor:
    tgt_probs, trans_probs = _attention_qk_fidelity_probabilities(
        query_heads,
        target_keys,
        translated_keys,
    )
    mid = 0.5 * (tgt_probs + trans_probs)
    js = 0.5 * (
        (tgt_probs * (tgt_probs.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum(dim=-1)
        + (trans_probs * (trans_probs.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum(dim=-1)
    )
    return (1.0 - js.clamp_min(0.0)).clamp_min(0.0)


def _attention_qk_fidelity_probabilities(
    query_heads: torch.Tensor,
    target_keys: torch.Tensor,
    translated_keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_heads.ndim == 3:
        if query_heads.shape[0] != 1:
            raise ValueError(f"query_heads batch dimension must be 1, got shape {tuple(query_heads.shape)}")
        query_heads = query_heads.squeeze(0)
    if target_keys.ndim == 4:
        if target_keys.shape[0] != 1:
            raise ValueError(f"target_keys batch dimension must be 1, got shape {tuple(target_keys.shape)}")
        target_keys = target_keys.squeeze(0)
    if translated_keys.ndim == 4:
        if translated_keys.shape[0] != 1:
            raise ValueError(
                f"translated_keys batch dimension must be 1, got shape {tuple(translated_keys.shape)}"
            )
        translated_keys = translated_keys.squeeze(0)
    if query_heads.ndim != 2:
        raise ValueError(f"query_heads must be rank-2 [heads, dim], got shape {tuple(query_heads.shape)}")
    if target_keys.ndim != 3 or translated_keys.ndim != 3:
        raise ValueError(
            "target_keys and translated_keys must be rank-3 [heads, positions, dim], "
            f"got {tuple(target_keys.shape)} and {tuple(translated_keys.shape)}"
        )
    if (
        query_heads.shape[0] != target_keys.shape[0]
        or query_heads.shape[0] != translated_keys.shape[0]
    ):
        raise ValueError(
            "query_heads, target_keys, and translated_keys must agree on head count, "
            f"got {query_heads.shape[0]}, {target_keys.shape[0]}, and {translated_keys.shape[0]}"
        )
    prefix_len = min(target_keys.shape[1], translated_keys.shape[1])
    if prefix_len <= 0:
        raise ValueError("attention_qk_fidelity requires at least one shared prefix position")
    q = query_heads.float()
    tgt = target_keys[:, -prefix_len:, :].float()
    trans = translated_keys[:, -prefix_len:, :].float()
    scale = math.sqrt(max(int(tgt.shape[-1]), 1))
    tgt_logits = torch.einsum("hd,hpd->hp", q, tgt) / scale
    trans_logits = torch.einsum("hd,hpd->hp", q, trans) / scale
    tgt_probs = torch.softmax(tgt_logits, dim=-1)
    trans_probs = torch.softmax(trans_logits, dim=-1)
    return tgt_probs, trans_probs


def _attention_qk_fidelity_token_scores(
    query_heads: torch.Tensor,
    target_keys: torch.Tensor,
    translated_keys: torch.Tensor,
) -> torch.Tensor:
    tgt_probs, trans_probs = _attention_qk_fidelity_probabilities(
        query_heads,
        target_keys,
        translated_keys,
    )
    return torch.minimum(tgt_probs, trans_probs).clamp_min(0.0)


def _match_prior_scores_to_live_order(
    live_scores: torch.Tensor,
    prior_scores: torch.Tensor,
    *,
    layer_idx: int,
    shuffled: bool = False,
) -> torch.Tensor:
    live = _normalize_selection_scores(live_scores).view(-1)
    fixed = _normalize_selection_scores(prior_scores).view(-1)
    if live.numel() != fixed.numel():
        raise ValueError(
            f"live_scores and prior_scores must have the same length, got {live.numel()} and {fixed.numel()}"
        )
    if live.numel() <= 1:
        return fixed
    live_order = torch.argsort(live, descending=True)
    if shuffled:
        matched_values = _deterministic_score_permutation(fixed, layer_idx=layer_idx, seed_offset=11_417)
    else:
        matched_values = fixed[torch.argsort(fixed, descending=True)]
    matched = torch.empty_like(fixed)
    matched[live_order] = matched_values
    return _normalize_selection_scores(matched)


def _normalize_selection_scores(scores: torch.Tensor) -> torch.Tensor:
    values = scores.float()
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if float((max_value - min_value).abs()) < 1e-8:
        return torch.ones_like(values)
    return (values - min_value) / (max_value - min_value).clamp_min(1e-8)


def _per_head_gate_override_from_scores(
    base_gate: float,
    scores: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    values = _normalize_selection_scores(scores.float().view(-1))
    if values.numel() == 0:
        return values
    mean_value = values.mean().clamp_min(1e-8)
    scaled = values / mean_value
    blend = (1.0 - float(strength)) + float(strength) * scaled
    return (float(base_gate) * blend).clamp(0.0, 1.0)


def _tokenwise_gate_override_from_scores(
    base_gate: torch.Tensor,
    scores: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must be rank-2 [heads, positions], got shape {tuple(scores.shape)}")
    base = base_gate.float().view(-1, 1)
    if base.shape[0] != scores.shape[0]:
        raise ValueError(
            "base_gate and scores must agree on head count, "
            f"got {base.shape[0]} and {scores.shape[0]}"
        )
    values = scores.float()
    if values.numel() == 0:
        return values.view(1, 0, 0, 1)
    mean_value = values.mean(dim=-1, keepdim=True)
    scaled = torch.where(
        mean_value > 1e-8,
        values / mean_value.clamp_min(1e-8),
        torch.ones_like(values),
    )
    blend = (1.0 - float(strength)) + float(strength) * scaled
    return (base * blend).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(-1)


def _deterministic_score_permutation(
    scores: torch.Tensor,
    *,
    layer_idx: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    values = scores.float().view(-1)
    if values.numel() <= 1:
        return values
    gen = torch.Generator(device="cpu").manual_seed(101_113 + seed_offset + int(layer_idx))
    perm = torch.randperm(values.numel(), generator=gen)
    identity = torch.arange(values.numel())
    if torch.equal(perm, identity):
        perm = torch.roll(perm, shifts=1)
    return values[perm.to(device=values.device)]


def _resample_head_profile(profile: torch.Tensor, target_heads: int) -> torch.Tensor:
    if target_heads <= 0:
        raise ValueError("target_heads must be positive")
    profile = profile.float().view(-1)
    if profile.numel() == target_heads:
        out = profile
    else:
        out = F.interpolate(
            profile.view(1, 1, -1),
            size=target_heads,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return out / out.sum().clamp_min(1e-8)


@torch.no_grad()
def _mean_head_prior_from_prompts(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    translator: RotAlignKVTranslator | None = None,
    metric: str = "attention_peak",
) -> list[torch.Tensor]:
    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for prompt in prompts:
        prefix_state = _prepare_prefix_state(model, tokenizer, prompt, device)
        if prefix_state.prefix_len <= 1:
            continue
        layer_maps = _last_token_attention_maps(
            model,
            prefix_state.last_token,
            prefix_state.past_key_values,
            device,
            translator=translator,
        )
        layer_scores = [
            _runtime_head_scores(attention_map, metric=metric, layer_idx=layer_idx)
            for layer_idx, attention_map in enumerate(layer_maps)
        ]
        if layer_sums is None:
            layer_sums = [scores.float().clone() for scores in layer_scores]
        else:
            for layer_idx, scores in enumerate(layer_scores):
                layer_sums[layer_idx] += scores.float()
        used_prompts += 1

    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build head prior from calibration prompts")
    return [
        _normalize_selection_scores(scores / float(used_prompts)).cpu()
        for scores in layer_sums
    ]


@torch.no_grad()
def _mean_head_attention_templates_from_prompts(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    translator: RotAlignKVTranslator | None = None,
    bins: int = 128,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for prompt in prompts:
        prefix_state = _prepare_prefix_state(model, tokenizer, prompt, device)
        if prefix_state.prefix_len <= 1:
            continue
        layer_maps = _last_token_attention_maps(
            model,
            prefix_state.last_token,
            prefix_state.past_key_values,
            device,
            translator=translator,
        )
        layer_templates = []
        for attention_map in layer_maps:
            resampled = torch.stack(
                [_resample_position_profile(head_scores, bins) for head_scores in attention_map.float()],
                dim=0,
            )
            resampled = resampled / resampled.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            layer_templates.append(resampled)
        if layer_sums is None:
            layer_sums = [templates.clone() for templates in layer_templates]
        else:
            for layer_idx, templates in enumerate(layer_templates):
                layer_sums[layer_idx] += templates
        used_prompts += 1

    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build head attention templates from calibration prompts")
    out: list[torch.Tensor] = []
    for templates in layer_sums:
        averaged = templates / float(used_prompts)
        averaged = averaged / averaged.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(averaged.cpu())
    return out


def _qk_position_distributions(
    query_heads: torch.Tensor,
    keys: torch.Tensor,
    *,
    bins: int | None = None,
) -> torch.Tensor:
    if query_heads.ndim == 3:
        if query_heads.shape[0] != 1:
            raise ValueError(f"query_heads batch dimension must be 1, got shape {tuple(query_heads.shape)}")
        query_heads = query_heads.squeeze(0)
    if keys.ndim == 4:
        if keys.shape[0] != 1:
            raise ValueError(f"keys batch dimension must be 1, got shape {tuple(keys.shape)}")
        keys = keys.squeeze(0)
    if query_heads.ndim != 2:
        raise ValueError(f"query_heads must be rank-2 [heads, dim], got shape {tuple(query_heads.shape)}")
    if keys.ndim != 3:
        raise ValueError(f"keys must be rank-3 [heads, positions, dim], got shape {tuple(keys.shape)}")
    if query_heads.shape[0] != keys.shape[0]:
        raise ValueError(
            "query_heads and keys must agree on head count, "
            f"got {query_heads.shape[0]} and {keys.shape[0]}"
        )
    prefix_len = keys.shape[1]
    if prefix_len <= 0:
        raise ValueError("QK templates require at least one prefix position")
    scale = math.sqrt(max(int(keys.shape[-1]), 1))
    logits = torch.einsum("hd,hpd->hp", query_heads.float(), keys.float()) / scale
    probs = torch.softmax(logits, dim=-1)
    if bins is not None:
        if bins <= 0:
            raise ValueError("bins must be positive")
        probs = torch.stack(
            [_resample_position_profile(head_probs, bins) for head_probs in probs],
            dim=0,
        )
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _query_feature_vector(
    query_heads: torch.Tensor,
    *,
    target_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if query_heads.ndim == 3:
        if query_heads.shape[0] != 1:
            raise ValueError(f"query_heads batch dimension must be 1, got shape {tuple(query_heads.shape)}")
        query_heads = query_heads.squeeze(0)
    if query_heads.ndim != 2:
        raise ValueError(f"query_heads must be rank-2 [heads, dim], got shape {tuple(query_heads.shape)}")
    if query_heads.shape[0] != target_heads:
        raise ValueError(
            f"query_heads head count {query_heads.shape[0]} does not match target_heads {target_heads}"
        )
    if query_heads.shape[1] != head_dim:
        raise ValueError(
            f"query_heads dim {query_heads.shape[1]} does not match head_dim {head_dim}"
        )
    return query_heads.float().reshape(1, 1, target_heads * head_dim)


@torch.no_grad()
def _mean_head_qk_templates_from_prompts(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    translator: RotAlignKVTranslator,
    bins: int = 128,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for prompt in prompts:
        prefix_state = _prepare_prefix_state(model, tokenizer, prompt, device)
        if prefix_state.prefix_len <= 1:
            continue
        query_heads = _last_token_query_heads(
            model,
            prefix_state.last_token,
            prefix_state.past_key_values,
            device,
            translator=translator,
        )
        normalized_cache = _normalize_cache(prefix_state.past_key_values)
        if normalized_cache is None:
            continue
        layer_templates: list[torch.Tensor] = []
        layer_limit = min(len(query_heads), len(normalized_cache), translator.config.num_tgt_layers)
        for layer_idx in range(layer_limit):
            keys = normalized_cache[layer_idx][0]
            layer_templates.append(_qk_position_distributions(query_heads[layer_idx], keys, bins=bins))
        if layer_sums is None:
            layer_sums = [templates.clone() for templates in layer_templates]
        else:
            for layer_idx, templates in enumerate(layer_templates):
                layer_sums[layer_idx] += templates
        used_prompts += 1

    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build head QK templates from calibration prompts")
    out: list[torch.Tensor] = []
    for templates in layer_sums:
        averaged = templates / float(used_prompts)
        averaged = averaged / averaged.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(averaged.cpu())
    return out


@torch.no_grad()
def _head_qk_template_bank_from_prompts(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    translator: RotAlignKVTranslator,
    bins: int = 128,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    layer_bank: list[list[torch.Tensor]] | None = None
    for prompt in prompts:
        prefix_state = _prepare_prefix_state(model, tokenizer, prompt, device)
        if prefix_state.prefix_len <= 1:
            continue
        query_heads = _last_token_query_heads(
            model,
            prefix_state.last_token,
            prefix_state.past_key_values,
            device,
            translator=translator,
        )
        normalized_cache = _normalize_cache(prefix_state.past_key_values)
        if normalized_cache is None:
            continue
        layer_limit = min(len(query_heads), len(normalized_cache), translator.config.num_tgt_layers)
        layer_templates: list[torch.Tensor] = []
        for layer_idx in range(layer_limit):
            keys = normalized_cache[layer_idx][0]
            layer_templates.append(_qk_position_distributions(query_heads[layer_idx], keys, bins=bins).cpu())
        if layer_bank is None:
            layer_bank = [[templates] for templates in layer_templates]
        else:
            for layer_idx, templates in enumerate(layer_templates):
                layer_bank[layer_idx].append(templates)

    if layer_bank is None or not layer_bank:
        raise ValueError("Could not build head QK template bank from calibration prompts")
    return [torch.stack(templates, dim=0) for templates in layer_bank]


def _attention_template_transport_scores(
    attention_map: torch.Tensor,
    head_templates: torch.Tensor,
    prior_scores: torch.Tensor,
    *,
    layer_idx: int,
    shuffled: bool = False,
    temperature: float = 0.1,
) -> torch.Tensor:
    if attention_map.ndim != 2:
        raise ValueError(f"attention_map must be rank-2 [heads, positions], got shape {tuple(attention_map.shape)}")
    if head_templates.ndim != 2:
        raise ValueError(f"head_templates must be rank-2 [heads, bins], got shape {tuple(head_templates.shape)}")
    if prior_scores.ndim != 1:
        raise ValueError(f"prior_scores must be rank-1 [heads], got shape {tuple(prior_scores.shape)}")
    if head_templates.shape[0] != prior_scores.numel():
        raise ValueError(
            "head_templates and prior_scores must agree on head count, "
            f"got {head_templates.shape[0]} and {prior_scores.numel()}"
        )
    template_count = head_templates.shape[0]
    live_count = attention_map.shape[0]
    if live_count <= 0:
        raise ValueError("attention_template_transport requires at least one live head")
    weights = _normalize_selection_scores(prior_scores).view(-1)
    templates = head_templates.float()
    if live_count != template_count:
        keep = min(live_count, template_count)
        order = torch.argsort(weights, descending=True)[:keep]
        templates = templates[order]
        weights = weights[order]
    if shuffled:
        weights = _deterministic_score_permutation(weights, layer_idx=layer_idx, seed_offset=13_211)
    bins = templates.shape[-1]
    live = torch.stack(
        [_resample_position_profile(head_scores.float(), bins) for head_scores in attention_map],
        dim=0,
    )
    live = live / live.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    templates = templates.to(device=live.device)
    weights = weights.to(device=live.device)
    p = templates[:, None, :]
    q = live[None, :, :]
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
        + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
    )
    kernel = torch.softmax(-js / max(float(temperature), 1e-4), dim=-1)
    aligned = (weights[:, None] * kernel).sum(dim=0)
    return _normalize_selection_scores(aligned)


def _attention_qk_bank_transport_scores(
    query_heads: torch.Tensor,
    target_keys: torch.Tensor,
    qk_template_bank: torch.Tensor,
    prior_scores: torch.Tensor,
    *,
    layer_idx: int,
    shuffled: bool = False,
    temperature: float = 0.1,
    prompt_temperature: float = 0.1,
) -> torch.Tensor:
    if qk_template_bank.ndim != 3:
        raise ValueError(
            f"qk_template_bank must be rank-3 [prompts, heads, bins], got shape {tuple(qk_template_bank.shape)}"
        )
    if prior_scores.ndim != 1:
        raise ValueError(f"prior_scores must be rank-1 [heads], got shape {tuple(prior_scores.shape)}")
    if qk_template_bank.shape[1] != prior_scores.numel():
        raise ValueError(
            "qk_template_bank and prior_scores must agree on head count, "
            f"got {qk_template_bank.shape[1]} and {prior_scores.numel()}"
        )
    live = _qk_position_distributions(query_heads, target_keys, bins=qk_template_bank.shape[-1])
    prompt_count, template_count, _ = qk_template_bank.shape
    live_count = live.shape[0]
    if live_count <= 0:
        raise ValueError("attention_qk_bank_transport requires at least one live head")
    weights = _normalize_selection_scores(prior_scores).view(-1)
    bank = qk_template_bank.float()
    if live_count != template_count:
        keep = min(live_count, template_count)
        order = torch.argsort(weights, descending=True)[:keep]
        bank = bank[:, order, :]
        weights = weights[order]
    if shuffled:
        weights = _deterministic_score_permutation(weights, layer_idx=layer_idx, seed_offset=16_241)
    bank = bank.to(device=live.device)
    weights = weights.to(device=live.device)
    p = bank
    q = live.unsqueeze(0)
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
        + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
    )
    prompt_score = -js.mean(dim=-1)
    prompt_weights = torch.softmax(prompt_score / max(float(prompt_temperature), 1e-4), dim=0)
    templates = (prompt_weights[:, None, None] * bank).sum(dim=0)
    return _attention_qk_template_transport_scores(
        query_heads,
        target_keys,
        templates,
        weights,
        layer_idx=layer_idx,
        temperature=temperature,
    )


def _attention_qk_template_transport_scores(
    query_heads: torch.Tensor,
    target_keys: torch.Tensor,
    qk_templates: torch.Tensor,
    prior_scores: torch.Tensor,
    *,
    layer_idx: int,
    shuffled: bool = False,
    temperature: float = 0.1,
) -> torch.Tensor:
    if qk_templates.ndim != 2:
        raise ValueError(f"qk_templates must be rank-2 [heads, bins], got shape {tuple(qk_templates.shape)}")
    if prior_scores.ndim != 1:
        raise ValueError(f"prior_scores must be rank-1 [heads], got shape {tuple(prior_scores.shape)}")
    if qk_templates.shape[0] != prior_scores.numel():
        raise ValueError(
            "qk_templates and prior_scores must agree on head count, "
            f"got {qk_templates.shape[0]} and {prior_scores.numel()}"
        )
    live = _qk_position_distributions(query_heads, target_keys, bins=qk_templates.shape[-1])
    template_count = qk_templates.shape[0]
    live_count = live.shape[0]
    if live_count <= 0:
        raise ValueError("attention_qk_template_transport requires at least one live head")
    weights = _normalize_selection_scores(prior_scores).view(-1)
    templates = qk_templates.float()
    if live_count != template_count:
        keep = min(live_count, template_count)
        order = torch.argsort(weights, descending=True)[:keep]
        templates = templates[order]
        weights = weights[order]
    if shuffled:
        weights = _deterministic_score_permutation(weights, layer_idx=layer_idx, seed_offset=14_407)
    templates = templates.to(device=live.device)
    weights = weights.to(device=live.device)
    p = templates[:, None, :]
    q = live[None, :, :]
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
        + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
    )
    kernel = torch.softmax(-js / max(float(temperature), 1e-4), dim=-1)
    aligned = (weights[:, None] * kernel).sum(dim=0)
    return _normalize_selection_scores(aligned)


def _attention_sinkhorn_transport_scores(
    attention_map: torch.Tensor,
    head_templates: torch.Tensor,
    prior_scores: torch.Tensor,
    *,
    layer_idx: int,
    shuffled: bool = False,
    temperature: float = 0.1,
    iterations: int = 4,
) -> torch.Tensor:
    if attention_map.ndim != 2:
        raise ValueError(f"attention_map must be rank-2 [heads, positions], got shape {tuple(attention_map.shape)}")
    if head_templates.ndim != 2:
        raise ValueError(f"head_templates must be rank-2 [heads, bins], got shape {tuple(head_templates.shape)}")
    if prior_scores.ndim != 1:
        raise ValueError(f"prior_scores must be rank-1 [heads], got shape {tuple(prior_scores.shape)}")
    if head_templates.shape[0] != prior_scores.numel():
        raise ValueError(
            "head_templates and prior_scores must agree on head count, "
            f"got {head_templates.shape[0]} and {prior_scores.numel()}"
        )
    template_count = head_templates.shape[0]
    live_count = attention_map.shape[0]
    if live_count <= 0:
        raise ValueError("attention_sinkhorn requires at least one live head")
    weights = _normalize_selection_scores(prior_scores).view(-1)
    templates = head_templates.float()
    if live_count != template_count:
        keep = min(live_count, template_count)
        order = torch.argsort(weights, descending=True)[:keep]
        templates = templates[order]
        weights = weights[order]
    if shuffled:
        weights = _deterministic_score_permutation(weights, layer_idx=layer_idx, seed_offset=15_173)
    bins = templates.shape[-1]
    live = torch.stack(
        [_resample_position_profile(head_scores.float(), bins) for head_scores in attention_map],
        dim=0,
    )
    live = live / live.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    templates = templates.to(device=live.device)
    weights = weights.to(device=live.device)
    live_peak = _normalize_selection_scores(_runtime_head_scores(attention_map, metric="attention_peak", layer_idx=layer_idx))
    live_peak = live_peak.to(device=live.device)
    p = templates[:, None, :]
    q = live[None, :, :]
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
        + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1)
    )
    similarity = torch.exp(-js / max(float(temperature), 1e-4))
    kernel = similarity.clamp_min(1e-8)
    u = torch.ones_like(weights)
    v = torch.ones_like(live_peak)
    for _ in range(max(int(iterations), 1)):
        u = weights / (kernel @ v).clamp_min(1e-8)
        v = live_peak / (kernel.transpose(0, 1) @ u).clamp_min(1e-8)
    transport = u[:, None] * kernel * v[None, :]
    aligned = (transport * similarity).sum(dim=0)
    return _normalize_selection_scores(aligned)


def _resample_head_profile_stack(
    profiles: list[torch.Tensor],
    *,
    target_layers: int,
) -> list[torch.Tensor]:
    if target_layers <= 0:
        raise ValueError("target_layers must be positive")
    if not profiles:
        raise ValueError("profiles must be non-empty")
    normalized = [torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu() for profile in profiles]
    if len(normalized) == target_layers:
        return [profile.clone() for profile in normalized]
    if len(normalized) == 1:
        return [normalized[0].clone() for _ in range(target_layers)]

    positions = torch.linspace(0, len(normalized) - 1, steps=target_layers, dtype=torch.float32)
    out: list[torch.Tensor] = []
    for pos in positions.tolist():
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(normalized[lo].clone())
            continue
        alpha = float(pos - lo)
        lo_profile = normalized[lo]
        hi_profile = normalized[hi]
        target_heads = max(lo_profile.numel(), hi_profile.numel())
        lo_profile = _resample_head_profile(lo_profile, target_heads)
        hi_profile = _resample_head_profile(hi_profile, target_heads)
        out.append(((1.0 - alpha) * lo_profile + alpha * hi_profile).cpu())
    return out


def _save_head_profile_bundle(
    path: str,
    profiles: list[torch.Tensor],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    bundle = {
        "profiles": [torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu() for profile in profiles],
        "metadata": dict(metadata or {}),
    }
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)


def _load_head_profile_bundle(
    path: str,
    *,
    target_layers: int | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metadata: dict[str, Any]
    if isinstance(payload, dict) and "profiles" in payload:
        raw_profiles = payload["profiles"]
        metadata = dict(payload.get("metadata") or {})
    elif isinstance(payload, list):
        raw_profiles = payload
        metadata = {}
    else:
        raise ValueError(f"Unsupported head profile bundle format: {type(payload)!r}")
    profiles = [torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu() for profile in raw_profiles]
    if not profiles:
        raise ValueError("Head profile bundle contains no profiles")
    metadata.setdefault("bundle_path", str(path))
    metadata.setdefault("stored_layer_count", len(profiles))
    metadata.setdefault("stored_head_counts", [int(profile.numel()) for profile in profiles])
    if target_layers is not None and len(profiles) != target_layers:
        profiles = _resample_head_profile_stack(profiles, target_layers=target_layers)
        metadata["resampled_to_layer_count"] = int(target_layers)
    return profiles, metadata


def _shrink_head_profiles(
    profiles: list[torch.Tensor],
    *,
    strength: float,
    target: str,
) -> list[torch.Tensor]:
    if not profiles:
        raise ValueError("profiles must be non-empty")
    amount = float(strength)
    if amount <= 0.0:
        return [torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu().clone() for profile in profiles]
    if amount > 1.0:
        raise ValueError("strength must be in [0, 1]")

    normalized = [
        _normalize_selection_scores(torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu())
        for profile in profiles
    ]
    if target == "uniform":
        targets = [torch.ones_like(profile) for profile in normalized]
    elif target == "global":
        max_heads = max(int(profile.numel()) for profile in normalized)
        global_profile = torch.stack(
            [_resample_head_profile(profile, max_heads) for profile in normalized],
            dim=0,
        ).mean(dim=0)
        global_profile = _normalize_selection_scores(global_profile)
        targets = [
            _normalize_selection_scores(_resample_head_profile(global_profile, int(profile.numel())))
            for profile in normalized
        ]
    else:
        raise ValueError(f"Unknown shrink target: {target}")

    out: list[torch.Tensor] = []
    for profile, target_profile in zip(normalized, targets):
        blended = (1.0 - amount) * profile + amount * target_profile
        out.append(_normalize_selection_scores(blended).cpu())
    return out


def _head_profile_summary(profiles: list[torch.Tensor]) -> dict[str, float]:
    entropies: list[float] = []
    top1_masses: list[float] = []
    for profile in profiles:
        weights = torch.as_tensor(profile, dtype=torch.float32).view(-1).clamp_min(0.0)
        if float(weights.sum().abs()) < 1e-8:
            weights = torch.ones_like(weights) / float(max(weights.numel(), 1))
        else:
            weights = weights / weights.sum().clamp_min(1e-8)
        entropies.append(float((-(weights * weights.clamp_min(1e-8).log()).sum()).item()))
        top1_masses.append(float(weights.max().item()))
    return {
        "profile_entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "profile_top1_mass_mean": float(sum(top1_masses) / max(len(top1_masses), 1)),
    }


def _head_profile_topk_overlap(
    left: list[torch.Tensor],
    right: list[torch.Tensor],
    *,
    keep_fraction: float = 0.25,
) -> float:
    overlaps: list[float] = []
    for lhs, rhs in zip(left, right):
        lhs = torch.as_tensor(lhs, dtype=torch.float32).view(-1)
        rhs = torch.as_tensor(rhs, dtype=torch.float32).view(-1)
        count = max(1, min(lhs.numel(), int(round(lhs.numel() * keep_fraction))))
        lhs_keep = set(torch.topk(lhs, k=count, largest=True).indices.detach().cpu().tolist())
        rhs_keep = set(torch.topk(rhs, k=count, largest=True).indices.detach().cpu().tolist())
        overlaps.append(float(len(lhs_keep & rhs_keep) / max(len(lhs_keep | rhs_keep), 1)))
    return float(sum(overlaps) / max(len(overlaps), 1))


def _head_trace(
    scores: torch.Tensor,
    keep_local_indices: torch.Tensor,
    original_head_indices: torch.Tensor,
    *,
    keep: int,
    prior_local_indices: torch.Tensor | None = None,
    attention_map: torch.Tensor | None = None,
) -> dict[str, Any]:
    selected_head_ids = sorted(
        int(idx)
        for idx in original_head_indices[keep_local_indices].detach().cpu().tolist()
    )
    probs = scores.float().detach()
    probs = probs / probs.sum().clamp_min(1e-8)
    sorted_scores = probs.sort(descending=True).values
    trace = {
        "head_keep": int(keep),
        "head_keep_fraction": float(keep / max(scores.numel(), 1)),
        "selected_head_ids": selected_head_ids[:16],
        "selected_head_count_truncated": int(max(len(selected_head_ids) - 16, 0)),
        "head_score_entropy": float((-(probs * probs.clamp_min(1e-8).log()).sum()).item()),
        "head_score_top": float(sorted_scores[0].item()),
        "head_score_gap": float((sorted_scores[0] - sorted_scores[1]).item()) if sorted_scores.numel() > 1 else float(sorted_scores[0].item()),
    }
    if prior_local_indices is not None:
        selected = set(int(idx) for idx in keep_local_indices.detach().cpu().tolist())
        prior = set(int(idx) for idx in prior_local_indices.detach().cpu().tolist())
        union = len(selected | prior)
        trace["head_prior_overlap_jaccard"] = float(len(selected & prior) / max(union, 1))
        prior_head_ids = sorted(int(idx) for idx in original_head_indices[prior_local_indices].detach().cpu().tolist())
        trace["prior_head_ids"] = prior_head_ids[:16]
        trace["prior_head_count_truncated"] = int(max(len(prior_head_ids) - 16, 0))
    if attention_map is not None and attention_map.shape[-1] > 0 and keep_local_indices.numel() > 0:
        selected_attention = attention_map[keep_local_indices].float()
        probs = selected_attention / selected_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        positions = torch.linspace(
            0.0,
            1.0,
            steps=selected_attention.shape[-1],
            device=selected_attention.device,
            dtype=selected_attention.dtype,
        )
        centroid = (probs * positions).sum(dim=-1)
        trace["selected_head_centroid_mean"] = float(centroid.mean().item())
        trace["selected_head_peak_mean"] = float(probs.max(dim=-1).values.mean().item())
    return trace


def _runtime_head_scores_with_prior(
    attention_map: torch.Tensor | None,
    *,
    metric: str,
    layer_idx: int,
    prior_scores: torch.Tensor | None = None,
    position_prior: torch.Tensor | None = None,
    target_keys: torch.Tensor | None = None,
    translated_keys: torch.Tensor | None = None,
    query_heads: torch.Tensor | None = None,
    head_templates: torch.Tensor | None = None,
    qk_templates: torch.Tensor | None = None,
    qk_template_bank: torch.Tensor | None = None,
    prior_alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prior_topk: torch.Tensor | None = None
    if metric in {"attention_peak", "attention_entropy", "attention_margin", "retrieval_peak", "random"}:
        if attention_map is None:
            raise ValueError(f"{metric} runtime head selection requires target attention maps")
        return _runtime_head_scores(attention_map, metric=metric, layer_idx=layer_idx), None
    if metric == "attention_expected":
        if attention_map is None:
            raise ValueError("attention_expected runtime head selection requires target attention maps")
        if position_prior is None:
            raise ValueError("attention_expected runtime head selection requires fixed_position_profiles")
        return _expected_attention_head_scores(attention_map, position_prior), None
    if metric == "attention_expected_shuffled":
        if attention_map is None:
            raise ValueError("attention_expected_shuffled runtime head selection requires target attention maps")
        if position_prior is None:
            raise ValueError("attention_expected_shuffled runtime head selection requires fixed_position_profiles")
        shuffled_prior = _deterministic_score_permutation(
            _resample_position_profile(position_prior, attention_map.shape[-1]),
            layer_idx=layer_idx,
            seed_offset=8_117,
        )
        return _expected_attention_head_scores(attention_map, shuffled_prior), None
    if metric == "attention_fidelity":
        if attention_map is None:
            raise ValueError("attention_fidelity runtime head selection requires target attention maps")
        if target_keys is None or translated_keys is None:
            raise ValueError("attention_fidelity runtime head selection requires target_keys and translated_keys")
        return _attention_fidelity_head_scores(attention_map, target_keys, translated_keys), None
    if metric == "attention_fidelity_shuffled":
        if attention_map is None:
            raise ValueError("attention_fidelity_shuffled runtime head selection requires target attention maps")
        if target_keys is None or translated_keys is None:
            raise ValueError("attention_fidelity_shuffled runtime head selection requires target_keys and translated_keys")
        scores = _attention_fidelity_head_scores(attention_map, target_keys, translated_keys)
        return _deterministic_score_permutation(scores, layer_idx=layer_idx, seed_offset=12_019), None
    if metric == "attention_procrustes":
        if attention_map is None:
            raise ValueError("attention_procrustes runtime head selection requires target attention maps")
        if target_keys is None or translated_keys is None:
            raise ValueError("attention_procrustes runtime head selection requires target_keys and translated_keys")
        return _attention_procrustes_head_scores(attention_map, target_keys, translated_keys), None
    if metric == "attention_procrustes_shuffled":
        if attention_map is None:
            raise ValueError("attention_procrustes_shuffled runtime head selection requires target attention maps")
        if target_keys is None or translated_keys is None:
            raise ValueError("attention_procrustes_shuffled runtime head selection requires target_keys and translated_keys")
        scores = _attention_procrustes_head_scores(attention_map, target_keys, translated_keys)
        return _deterministic_score_permutation(scores, layer_idx=layer_idx, seed_offset=12_427), None
    if metric == "attention_qk_fidelity":
        if query_heads is None or target_keys is None or translated_keys is None:
            raise ValueError(
                "attention_qk_fidelity runtime head selection requires query_heads, target_keys, and translated_keys"
            )
        return _attention_qk_fidelity_head_scores(query_heads, target_keys, translated_keys), None
    if metric == "attention_qk_fidelity_shuffled":
        if query_heads is None or target_keys is None or translated_keys is None:
            raise ValueError(
                "attention_qk_fidelity_shuffled runtime head selection requires query_heads, target_keys, and translated_keys"
            )
        scores = _attention_qk_fidelity_head_scores(query_heads, target_keys, translated_keys)
        return _deterministic_score_permutation(scores, layer_idx=layer_idx, seed_offset=12_881), None
    if metric == "attention_qk_template_transport":
        if query_heads is None or target_keys is None:
            raise ValueError(
                "attention_qk_template_transport runtime head selection requires query_heads and target_keys"
            )
        if prior_scores is None or qk_templates is None:
            raise ValueError(
                "attention_qk_template_transport requires fixed_head_profiles and fixed_head_qk_templates"
            )
        return _attention_qk_template_transport_scores(
            query_heads,
            target_keys,
            qk_templates,
            prior_scores,
            layer_idx=layer_idx,
        ), None
    if metric == "attention_qk_template_transport_shuffled":
        if query_heads is None or target_keys is None:
            raise ValueError(
                "attention_qk_template_transport_shuffled runtime head selection requires query_heads and target_keys"
            )
        if prior_scores is None or qk_templates is None:
            raise ValueError(
                "attention_qk_template_transport_shuffled requires fixed_head_profiles and fixed_head_qk_templates"
            )
        return _attention_qk_template_transport_scores(
            query_heads,
            target_keys,
            qk_templates,
            prior_scores,
            layer_idx=layer_idx,
            shuffled=True,
        ), None
    if metric == "attention_qk_bank_transport":
        if query_heads is None or target_keys is None:
            raise ValueError(
                "attention_qk_bank_transport runtime head selection requires query_heads and target_keys"
            )
        if prior_scores is None or qk_template_bank is None:
            raise ValueError(
                "attention_qk_bank_transport requires fixed_head_profiles and fixed_head_qk_template_bank"
            )
        return _attention_qk_bank_transport_scores(
            query_heads,
            target_keys,
            qk_template_bank,
            prior_scores,
            layer_idx=layer_idx,
        ), None
    if metric == "attention_qk_bank_transport_shuffled":
        if query_heads is None or target_keys is None:
            raise ValueError(
                "attention_qk_bank_transport_shuffled runtime head selection requires query_heads and target_keys"
            )
        if prior_scores is None or qk_template_bank is None:
            raise ValueError(
                "attention_qk_bank_transport_shuffled requires fixed_head_profiles and fixed_head_qk_template_bank"
            )
        return _attention_qk_bank_transport_scores(
            query_heads,
            target_keys,
            qk_template_bank,
            prior_scores,
            layer_idx=layer_idx,
            shuffled=True,
        ), None
    if metric == "attention_prior":
        if prior_scores is None:
            raise ValueError("attention_prior runtime head selection requires fixed_head_profiles")
        return _normalize_selection_scores(prior_scores), None
    if metric == "attention_prior_shuffled":
        if prior_scores is None:
            raise ValueError("attention_prior_shuffled runtime head selection requires fixed_head_profiles")
        shuffled_scores = _deterministic_score_permutation(prior_scores, layer_idx=layer_idx, seed_offset=7_009)
        return _normalize_selection_scores(shuffled_scores), None
    if metric == "attention_template_transport":
        if attention_map is None:
            raise ValueError("attention_template_transport runtime head selection requires target attention maps")
        if prior_scores is None or head_templates is None:
            raise ValueError("attention_template_transport requires fixed_head_profiles and fixed_head_templates")
        return _attention_template_transport_scores(
            attention_map,
            head_templates,
            prior_scores,
            layer_idx=layer_idx,
        ), None
    if metric == "attention_template_transport_shuffled":
        if attention_map is None:
            raise ValueError("attention_template_transport_shuffled runtime head selection requires target attention maps")
        if prior_scores is None or head_templates is None:
            raise ValueError("attention_template_transport_shuffled requires fixed_head_profiles and fixed_head_templates")
        return _attention_template_transport_scores(
            attention_map,
            head_templates,
            prior_scores,
            layer_idx=layer_idx,
            shuffled=True,
        ), None
    if metric == "attention_sinkhorn":
        if attention_map is None:
            raise ValueError("attention_sinkhorn runtime head selection requires target attention maps")
        if prior_scores is None or head_templates is None:
            raise ValueError("attention_sinkhorn requires fixed_head_profiles and fixed_head_templates")
        return _attention_sinkhorn_transport_scores(
            attention_map,
            head_templates,
            prior_scores,
            layer_idx=layer_idx,
        ), None
    if metric == "attention_sinkhorn_shuffled":
        if attention_map is None:
            raise ValueError("attention_sinkhorn_shuffled runtime head selection requires target attention maps")
        if prior_scores is None or head_templates is None:
            raise ValueError("attention_sinkhorn_shuffled requires fixed_head_profiles and fixed_head_templates")
        return _attention_sinkhorn_transport_scores(
            attention_map,
            head_templates,
            prior_scores,
            layer_idx=layer_idx,
            shuffled=True,
        ), None
    if metric == "attention_match":
        if attention_map is None:
            raise ValueError("attention_match runtime head selection requires target attention maps")
        if prior_scores is None:
            raise ValueError("attention_match runtime head selection requires fixed_head_profiles")
        live_scores = _runtime_head_scores(attention_map, metric="attention_peak", layer_idx=layer_idx)
        return _match_prior_scores_to_live_order(live_scores, prior_scores, layer_idx=layer_idx), None
    if metric == "attention_match_shuffled":
        if attention_map is None:
            raise ValueError("attention_match_shuffled runtime head selection requires target attention maps")
        if prior_scores is None:
            raise ValueError("attention_match_shuffled runtime head selection requires fixed_head_profiles")
        live_scores = _runtime_head_scores(attention_map, metric="attention_peak", layer_idx=layer_idx)
        return _match_prior_scores_to_live_order(
            live_scores,
            prior_scores,
            layer_idx=layer_idx,
            shuffled=True,
        ), None
    if metric == "attention_blend":
        if attention_map is None:
            raise ValueError("attention_blend runtime head selection requires target attention maps")
        if prior_scores is None:
            raise ValueError("attention_blend runtime head selection requires fixed_head_profiles")
        live_scores = _normalize_selection_scores(
            _runtime_head_scores(attention_map, metric="attention_peak", layer_idx=layer_idx)
        )
        fixed_scores = _normalize_selection_scores(prior_scores)
        blend = (1.0 - float(prior_alpha)) * live_scores + float(prior_alpha) * fixed_scores
        prior_topk = torch.topk(fixed_scores, k=min(blend.numel(), fixed_scores.numel()), largest=True).indices
        return blend, prior_topk
    raise ValueError(f"Unknown runtime_head_selection_metric: {metric}")


def _apply_runtime_head_selection(
    translated_kv: torch.Tensor,
    target_kv: torch.Tensor,
    full_head_mask: torch.Tensor,
    *,
    protocol: str,
) -> torch.Tensor:
    mask = full_head_mask.view(1, -1, 1, 1).to(device=translated_kv.device)
    fill = torch.zeros_like(translated_kv) if protocol == "translated_only" else target_kv
    return torch.where(mask, translated_kv, fill)


def _resample_position_profile(profile: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    profile = profile.float()
    if profile.numel() == target_len:
        out = profile
    else:
        out = F.interpolate(
            profile.view(1, 1, -1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return out / out.sum().clamp_min(1e-8)


def _translate_layer_with_optional_runtime_attention(
    translator: RotAlignKVTranslator,
    K_s: torch.Tensor,
    V_s: torch.Tensor,
    *,
    tgt_layer_idx: int,
    quantize: bool,
    quantization_control: str,
    runtime_attention_profile: torch.Tensor | None,
    runtime_query_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return translator.translate_layer(
            K_s,
            V_s,
            tgt_layer_idx=tgt_layer_idx,
            quantize=quantize,
            quantization_control=quantization_control,
            runtime_attention_profile=runtime_attention_profile,
            runtime_query_features=runtime_query_features,
        )
    except TypeError as exc:
        if "runtime_attention_profile" not in str(exc) and "runtime_query_features" not in str(exc):
            raise
        return translator.translate_layer(
            K_s,
            V_s,
            tgt_layer_idx=tgt_layer_idx,
            quantize=quantize,
            quantization_control=quantization_control,
        )


@torch.no_grad()
def _mean_attention_prior_from_prompts(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    *,
    translator: RotAlignKVTranslator | None = None,
    bins: int = 128,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for prompt in prompts:
        prefix_state = _prepare_prefix_state(model, tokenizer, prompt, device)
        if prefix_state.prefix_len <= 1:
            continue
        layer_scores = _last_token_attention_scores(
            model,
            prefix_state.last_token,
            prefix_state.past_key_values,
            device,
            translator=translator,
        )
        layer_scores = [_resample_position_profile(scores, bins) for scores in layer_scores]
        if layer_sums is None:
            layer_sums = [scores.clone() for scores in layer_scores]
        else:
            for layer_idx, scores in enumerate(layer_scores):
                layer_sums[layer_idx] += scores
        used_prompts += 1

    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build attention prior from calibration prompts")
    return [(scores / float(used_prompts)).cpu() for scores in layer_sums]


def _uniform_attention_prior(
    *,
    layer_count: int,
    bins: int,
) -> list[torch.Tensor]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if bins <= 0:
        raise ValueError("bins must be positive")
    prior = torch.full((bins,), 1.0 / float(bins), dtype=torch.float32)
    return [prior.clone() for _ in range(layer_count)]


@torch.no_grad()
def _logprob_tokens_from_prefix_state(
    model,
    prefix_state: PrefixState,
    continuation_ids: torch.Tensor,
    device: str,
) -> float:
    if continuation_ids.shape[1] == 0:
        return 0.0
    logits, past = _step_with_past(model, prefix_state.last_token, prefix_state.past_key_values, device)
    log_probs = logits.log_softmax(dim=-1)
    total = float(log_probs[0, continuation_ids[0, 0]])
    current = continuation_ids[:, :1]
    for idx in range(1, continuation_ids.shape[1]):
        logits, past = _step_with_past(model, current, past, device)
        log_probs = logits.log_softmax(dim=-1)
        current = continuation_ids[:, idx : idx + 1]
        total += float(log_probs[0, current.item()])
    return total


@torch.no_grad()
def _greedy_generate_from_prefix_state(
    model,
    tokenizer,
    prefix_state: PrefixState,
    device: str,
    max_new_tokens: int,
) -> str:
    return _greedy_generate_with_stats(
        model,
        tokenizer,
        prefix_state,
        device,
        max_new_tokens,
    ).text


@torch.no_grad()
def _greedy_generate_with_stats(
    model,
    tokenizer,
    prefix_state: PrefixState,
    device: str,
    max_new_tokens: int,
) -> GenerationTrace:
    started = time.perf_counter()
    logits, past = _step_with_past(model, prefix_state.last_token, prefix_state.past_key_values, device)
    ttft_sec = time.perf_counter() - started
    generated: list[int] = []
    next_token = int(logits.argmax(dim=-1).item())
    generated.append(next_token)
    current = torch.tensor([[next_token]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens - 1):
        if next_token == tokenizer.eos_token_id:
            break
        logits, past = _step_with_past(model, current, past, device)
        next_token = int(logits.argmax(dim=-1).item())
        generated.append(next_token)
        current = torch.tensor([[next_token]], dtype=torch.long, device=device)
    return GenerationTrace(
        text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
        num_generated_tokens=len(generated),
        ttft_sec=ttft_sec,
        elapsed_sec=time.perf_counter() - started,
    )


@torch.no_grad()
def score_choice_loglik(model, tokenizer, context: str, choice: str, device: str) -> float:
    """Return sum log p(choice_tokens | context) under the model."""
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(context + " " + choice, return_tensors="pt").input_ids.to(device)
    out = model(full_ids, labels=full_ids)
    logits = out.logits[0, :-1, :].log_softmax(dim=-1)
    targets = full_ids[0, 1:]
    choice_start = max(ctx_ids.shape[1] - 1, 0)
    token_logp = logits[choice_start:].gather(1, targets[choice_start:].unsqueeze(-1)).squeeze(-1)
    return float(token_logp.sum())


@torch.no_grad()
def score_with_injected_kv(
    target_model,
    target_tokenizer,
    injected_pkv: tuple,
    injected_attention_mask: torch.Tensor,
    choice: str,
    device: str,
) -> float:
    """Backwards-compatible helper retained for unit tests."""
    choice_ids = target_tokenizer(choice, return_tensors="pt").input_ids.to(device)
    new_mask = torch.cat([injected_attention_mask, torch.ones_like(choice_ids)], dim=1)
    out = target_model(
        input_ids=choice_ids,
        attention_mask=new_mask,
        past_key_values=injected_pkv,
        use_cache=True,
    )
    logits = out.logits[0].log_softmax(dim=-1)
    if choice_ids.shape[1] < 2:
        return 0.0
    scored = logits[:-1].gather(1, choice_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(scored.sum())


def _score_mcq_with_prefix_state(model, tokenizer, prefix_state: PrefixState, choice: str, device: str) -> float:
    # Match `score_choice_loglik`, which evaluates "context + space + choice".
    continuation_ids = tokenizer(" " + choice, return_tensors="pt").input_ids.to(device)
    return _logprob_tokens_from_prefix_state(model, prefix_state, continuation_ids, device)


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_EXPLICIT_NUMERIC_PATTERNS = (
    re.compile(r"####\s*([-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(
        r"(?:final answer|answer|result|total)\s*(?:is|=|:)?\s*"
        r"([-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)


def _normalize_generation_text(text: str) -> str:
    norm = " ".join(text.strip().lower().split())
    norm = re.sub(r"^[`'\"“”‘’\(\[]+", "", norm)
    norm = re.sub(r"[`'\"“”‘’\)\]\.!,?:;]+$", "", norm)
    return norm


def _normalize_numeric_string(text: str) -> str | None:
    cleaned = text.strip()
    cleaned = cleaned.replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".,!?;: ")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def _extract_reference_numeric_answer(answer: str) -> str | None:
    for pattern in _EXPLICIT_NUMERIC_PATTERNS:
        matches = pattern.findall(answer)
        if matches:
            candidate = matches[-1]
            if isinstance(candidate, tuple):
                candidate = candidate[-1]
            numeric = _normalize_numeric_string(str(candidate))
            if numeric is not None:
                return numeric
    return _normalize_numeric_string(answer)


def _extract_prediction_numeric_answer(prediction: str) -> str | None:
    for pattern in _EXPLICIT_NUMERIC_PATTERNS:
        matches = pattern.findall(prediction)
        if matches:
            candidate = matches[-1]
            if isinstance(candidate, tuple):
                candidate = candidate[-1]
            numeric = _normalize_numeric_string(str(candidate))
            if numeric is not None:
                return numeric

    candidates = _NUMERIC_TOKEN_RE.findall(prediction)
    for candidate in reversed(candidates):
        numeric = _normalize_numeric_string(candidate)
        if numeric is not None:
            return numeric
    return None


def _generation_match(prediction: str, answers: list[str]) -> bool:
    norm_pred = _normalize_generation_text(prediction)
    normalized_answers = {_normalize_generation_text(answer) for answer in answers}
    if norm_pred in normalized_answers:
        return True

    numeric_answers = {
        numeric
        for answer in answers
        if (numeric := _extract_reference_numeric_answer(answer)) is not None
    }
    if not numeric_answers:
        return False

    pred_numeric = _extract_prediction_numeric_answer(prediction)
    return pred_numeric is not None and pred_numeric in numeric_answers


def _generation_metrics(
    *,
    correct: int,
    num_examples: int,
    total_generated_tokens: int,
    total_ttft_sec: float,
    total_elapsed_sec: float,
) -> dict[str, float]:
    count = max(num_examples, 1)
    elapsed = max(total_elapsed_sec, 1e-9)
    return {
        "accuracy": correct / count,
        "ttft_sec": total_ttft_sec / count,
        "tokens_per_sec": total_generated_tokens / elapsed,
        "examples_per_sec": num_examples / elapsed,
        "latency_sec": total_elapsed_sec / count,
        "generated_tokens_avg": total_generated_tokens / count,
    }


def _append_prediction_record(
    records: list[dict[str, Any]] | None,
    *,
    index: int,
    example_id: str | None = None,
    method: str,
    prediction: Any,
    answer: Any,
    correct: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    if records is None:
        return
    record = {
        "index": int(index),
        "method": method,
        "prediction": prediction,
        "answer": answer,
        "correct": bool(correct),
    }
    if example_id is not None:
        record["example_id"] = example_id
    if extra:
        record.update(extra)
    records.append(record)


def write_prediction_records(path: str, records: list[dict[str, Any]]) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def prediction_record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), []).append(record)

    summary: dict[str, Any] = {}
    for method, rows in by_method.items():
        correct = sum(bool(row.get("correct")) for row in rows)
        method_summary: dict[str, Any] = {
            "count": len(rows),
            "accuracy": correct / max(len(rows), 1),
        }
        bits = [float(row["bits"]) for row in rows if row.get("bits") is not None]
        if bits:
            method_summary["avg_bits"] = sum(bits) / len(bits)
            method_summary["avg_bytes"] = method_summary["avg_bits"] / 8.0
        payload_bits = [float(row["payload_bits"]) for row in rows if row.get("payload_bits") is not None]
        if payload_bits:
            method_summary["payload_bits_avg"] = sum(payload_bits) / len(payload_bits)
            method_summary["payload_bytes_avg"] = method_summary["payload_bits_avg"] / 8.0
        selector_bits = [float(row["selector_bits"]) for row in rows if row.get("selector_bits") is not None]
        if selector_bits:
            method_summary["selector_bits_avg"] = sum(selector_bits) / len(selector_bits)
            method_summary["selector_bytes_avg"] = method_summary["selector_bits_avg"] / 8.0
        metadata_bits = [float(row["metadata_bits"]) for row in rows if row.get("metadata_bits") is not None]
        if metadata_bits:
            method_summary["metadata_bits_avg"] = sum(metadata_bits) / len(metadata_bits)
            method_summary["metadata_bytes_avg"] = method_summary["metadata_bits_avg"] / 8.0
        traces = [row.get("selector_trace") for row in rows if row.get("selector_trace")]
        if traces:
            flattened = [layer for trace in traces for layer in trace]
            if flattened:
                keep_fractions = [float(layer["keep_fraction"]) for layer in flattened if "keep_fraction" in layer]
                entropies = [float(layer["score_entropy"]) for layer in flattened if "score_entropy" in layer]
                method_summary["selector_layers_logged"] = len(flattened)
                if keep_fractions:
                    method_summary["selector_keep_fraction_avg"] = sum(keep_fractions) / len(keep_fractions)
                if entropies:
                    method_summary["selector_entropy_avg"] = sum(entropies) / len(entropies)
        head_traces = [row.get("head_trace") for row in rows if row.get("head_trace")]
        if head_traces:
            flattened_heads = [layer for trace in head_traces for layer in trace]
            if flattened_heads:
                keep_fractions = [
                    float(layer["head_keep_fraction"])
                    for layer in flattened_heads
                    if "head_keep_fraction" in layer
                ]
                entropies = [
                    float(layer["head_score_entropy"])
                    for layer in flattened_heads
                    if "head_score_entropy" in layer
                ]
                prior_overlaps = [
                    float(layer["head_prior_overlap_jaccard"])
                    for layer in flattened_heads
                    if "head_prior_overlap_jaccard" in layer
                ]
                method_summary["head_layers_logged"] = len(flattened_heads)
                if keep_fractions:
                    method_summary["head_keep_fraction_avg"] = sum(keep_fractions) / len(keep_fractions)
                if entropies:
                    method_summary["head_score_entropy_avg"] = sum(entropies) / len(entropies)
                if prior_overlaps:
                    method_summary["head_prior_overlap_jaccard_avg"] = sum(prior_overlaps) / len(prior_overlaps)
        head_budget_traces = [row.get("head_budget_trace") for row in rows if row.get("head_budget_trace")]
        if head_budget_traces:
            flattened_budgets = [layer for trace in head_budget_traces for layer in trace]
            if flattened_budgets:
                keep_fractions = [
                    float(layer["head_budget_keep_fraction"])
                    for layer in flattened_budgets
                    if "head_budget_keep_fraction" in layer
                ]
                nonzero_fractions = [
                    float(layer["head_budget_nonzero_fraction"])
                    for layer in flattened_budgets
                    if "head_budget_nonzero_fraction" in layer
                ]
                method_summary["head_budget_layers_logged"] = len(flattened_budgets)
                if keep_fractions:
                    method_summary["head_budget_keep_fraction_avg"] = sum(keep_fractions) / len(keep_fractions)
                if nonzero_fractions:
                    method_summary["head_budget_nonzero_fraction_avg"] = sum(nonzero_fractions) / len(nonzero_fractions)
        summary[method] = method_summary
    return summary


def write_prediction_sidecar(
    path: str,
    records: list[dict[str, Any]],
    results: dict[str, float],
    run_config: dict[str, Any],
) -> None:
    output_path = pathlib.Path(path)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "run_config": run_config,
        "method_summary": prediction_record_summary(records),
        "paired_summary": {
            key: value
            for key, value in results.items()
            if key.startswith("paired_")
        },
        "metric_summary": {
            key: value
            for key, value in results.items()
            if not key.startswith("paired_")
        },
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def paired_prediction_metrics(
    records: list[dict[str, Any]],
    method: str,
    baseline: str,
    *,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    by_method: dict[str, dict[int, bool]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), {})[int(record["index"])] = bool(record["correct"])
    method_rows = by_method.get(method, {})
    baseline_rows = by_method.get(baseline, {})
    indices = sorted(set(method_rows) & set(baseline_rows))
    if not indices:
        return {}

    diffs = [
        (1.0 if method_rows[idx] else 0.0) - (1.0 if baseline_rows[idx] else 0.0)
        for idx in indices
    ]
    method_only = sum(method_rows[idx] and not baseline_rows[idx] for idx in indices)
    baseline_only = sum(baseline_rows[idx] and not method_rows[idx] for idx in indices)
    denom = method_only + baseline_only
    if denom:
        chi2 = (max(abs(method_only - baseline_only) - 1.0, 0.0) ** 2) / denom
        # McNemar with one degree of freedom: survival function = erfc(sqrt(x/2)).
        p_value = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        p_value = 1.0

    gen = torch.Generator(device="cpu").manual_seed(0)
    diff_tensor = torch.tensor(diffs, dtype=torch.float32)
    boot = []
    for _ in range(n_bootstrap):
        sample_idx = torch.randint(0, len(diffs), (len(diffs),), generator=gen)
        boot.append(float(diff_tensor[sample_idx].mean()))
    boot.sort()
    lo_idx = int(0.025 * (len(boot) - 1))
    hi_idx = int(0.975 * (len(boot) - 1))
    return {
        "paired_n": float(len(indices)),
        "delta_accuracy": float(sum(diffs) / len(diffs)),
        "method_only": float(method_only),
        "baseline_only": float(baseline_only),
        "both_correct": float(sum(method_rows[idx] and baseline_rows[idx] for idx in indices)),
        "both_wrong": float(sum((not method_rows[idx]) and (not baseline_rows[idx]) for idx in indices)),
        "mcnemar_chi2": float(chi2 if denom else 0.0),
        "mcnemar_p": float(p_value),
        "bootstrap_delta_low": float(boot[lo_idx]),
        "bootstrap_delta_high": float(boot[hi_idx]),
    }


def _metric_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def add_paired_prediction_summary(
    results: dict[str, float],
    records: list[dict[str, Any]],
) -> None:
    methods = sorted({str(record["method"]) for record in records})
    for method in methods:
        if method in {"target_alone", "text_to_text", "source_alone", "routing"}:
            continue
        for baseline in ("target_alone", "text_to_text"):
            stats = paired_prediction_metrics(records, method, baseline)
            if not stats:
                continue
            prefix = f"paired_{_metric_slug(method)}_vs_{baseline}"
            for key, value in stats.items():
                results[f"{prefix}_{key}"] = value


def _tail_align_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq = min(left.shape[2], right.shape[2])
    return left[:, :, -seq:, :], right[:, :, -seq:, :]


def _apply_source_kv_control(
    K_s: torch.Tensor,
    V_s: torch.Tensor,
    mode: str,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative-control perturbations for source-side KV communication."""
    if mode == "real":
        return K_s, V_s
    if mode == "zero":
        return torch.zeros_like(K_s), torch.zeros_like(V_s)
    if mode == "shuffle_positions":
        if K_s.shape[2] <= 1:
            return K_s, V_s
        return K_s.flip(dims=[2]), V_s.flip(dims=[2])
    if mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(17_171 + int(layer_idx))

        def matched_noise(x: torch.Tensor) -> torch.Tensor:
            x_cpu = x.detach().to("cpu", dtype=torch.float32)
            noise = torch.randn(x_cpu.shape, generator=gen, dtype=torch.float32)
            return (noise * x_cpu.std().clamp_min(1e-6) + x_cpu.mean()).to(
                device=x.device,
                dtype=x.dtype,
            )

        return matched_noise(K_s), matched_noise(V_s)
    raise ValueError(f"Unknown source_kv_control: {mode}")


def _apply_translated_kv_control(
    K_hat: torch.Tensor,
    V_hat: torch.Tensor,
    mode: str,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative controls applied after translation into target KV space."""
    if mode == "real":
        return K_hat, V_hat
    if mode == "zero":
        return torch.zeros_like(K_hat), torch.zeros_like(V_hat)
    if mode == "shuffle_positions":
        if K_hat.shape[2] <= 1:
            return K_hat, V_hat
        return K_hat.flip(dims=[2]), V_hat.flip(dims=[2])
    if mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(27_217 + int(layer_idx))

        def matched_noise(x: torch.Tensor) -> torch.Tensor:
            x_cpu = x.detach().to("cpu", dtype=torch.float32)
            noise = torch.randn(x_cpu.shape, generator=gen, dtype=torch.float32)
            return (noise * x_cpu.std().clamp_min(1e-6) + x_cpu.mean()).to(
                device=x.device,
                dtype=x.dtype,
            )

        return matched_noise(K_hat), matched_noise(V_hat)
    raise ValueError(f"Unknown translated_kv_control: {mode}")


def _selected_token_count(seq_len: int, position_selection_ratio: float) -> int:
    if seq_len <= 0:
        return 0
    ratio = float(position_selection_ratio)
    if ratio >= 1.0:
        return seq_len
    if ratio <= 0.0:
        return 1
    return max(1, min(seq_len, int(round(seq_len * ratio))))


def _selected_total_token_budget(seq_len: int, head_count: int, position_selection_ratio: float) -> int:
    if seq_len <= 0 or head_count <= 0:
        return 0
    ratio = float(position_selection_ratio)
    total = seq_len * head_count
    if ratio >= 1.0:
        return total
    if ratio <= 0.0:
        return 1
    return max(1, min(total, int(round(total * ratio))))


def _position_selection_index_bits(seq_len: int, selected_tokens: int) -> float:
    if seq_len <= 1 or selected_tokens >= seq_len:
        return 0.0
    return float(selected_tokens * max(1, math.ceil(math.log2(seq_len))))


def _bits_per_kv_tensor(
    translator: RotAlignKVTranslator,
    seq_len: int,
    quantize: bool,
    *,
    selected_heads: int | None = None,
) -> float:
    selected_heads = translator.config.tgt_num_heads if selected_heads is None else max(0, selected_heads)
    d_t = selected_heads * translator.config.tgt_head_dim
    if d_t <= 0:
        return 0.0
    if quantize:
        return float(seq_len * (translator.config.quant_bits * d_t + 32))
    return float(seq_len * 32 * d_t)


def _translated_bit_breakdown(
    translator: RotAlignKVTranslator,
    seq_len: int,
    quantize: bool,
    *,
    active_k_head_counts: list[int] | None = None,
    active_v_head_counts: list[int] | None = None,
    active_k_token_counts: list[list[int]] | None = None,
    active_v_token_counts: list[list[int]] | None = None,
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
) -> tuple[float, float]:
    transport_k = kv_transport in {"both", "k_only"}
    transport_v = kv_transport in {"both", "v_only"}
    selected_layers = translator.selected_layer_indices()
    if active_k_head_counts is None and transport_k:
        active_k_head_counts = [translator.selected_head_count(layer_idx) for layer_idx in selected_layers]
    if active_v_head_counts is None and transport_v:
        active_v_head_counts = [translator.selected_head_count(layer_idx) for layer_idx in selected_layers]
    active_k_head_counts = [] if not transport_k or active_k_head_counts is None else active_k_head_counts
    active_v_head_counts = [] if not transport_v or active_v_head_counts is None else active_v_head_counts
    if active_k_token_counts is not None or active_v_token_counts is not None:
        active_k_token_counts = [] if not transport_k or active_k_token_counts is None else active_k_token_counts
        active_v_token_counts = [] if not transport_v or active_v_token_counts is None else active_v_token_counts
        payload_bits = float(
            sum(
                _bits_per_kv_tensor(translator, int(token_count), quantize, selected_heads=1)
                for layer_counts in active_k_token_counts
                for token_count in layer_counts
                if int(token_count) > 0
            )
            + sum(
                _bits_per_kv_tensor(translator, int(token_count), quantize, selected_heads=1)
                for layer_counts in active_v_token_counts
                for token_count in layer_counts
                if int(token_count) > 0
            )
        )
        selector_bits = float(
            sum(
                _position_selection_index_bits(seq_len, int(token_count))
                for layer_counts in active_k_token_counts
                for token_count in layer_counts
                if int(token_count) > 0
            )
            + sum(
                _position_selection_index_bits(seq_len, int(token_count))
                for layer_counts in active_v_token_counts
                for token_count in layer_counts
                if int(token_count) > 0
            )
        )
        return payload_bits, selector_bits
    selected_tokens = _selected_token_count(seq_len, position_selection_ratio)
    payload_bits = float(
        sum(_bits_per_kv_tensor(translator, selected_tokens, quantize, selected_heads=count) for count in active_k_head_counts)
        + sum(_bits_per_kv_tensor(translator, selected_tokens, quantize, selected_heads=count) for count in active_v_head_counts)
    )
    layer_count = max(len(active_k_head_counts), len(active_v_head_counts))
    selector_bits = float(layer_count) * _position_selection_index_bits(seq_len, selected_tokens)
    return payload_bits, selector_bits


def _translated_bits(
    translator: RotAlignKVTranslator,
    seq_len: int,
    quantize: bool,
    *,
    active_k_head_counts: list[int] | None = None,
    active_v_head_counts: list[int] | None = None,
    active_k_token_counts: list[list[int]] | None = None,
    active_v_token_counts: list[list[int]] | None = None,
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
) -> float:
    payload_bits, selector_bits = _translated_bit_breakdown(
        translator,
        seq_len,
        quantize,
        active_k_head_counts=active_k_head_counts,
        active_v_head_counts=active_v_head_counts,
        active_k_token_counts=active_k_token_counts,
        active_v_token_counts=active_v_token_counts,
        kv_transport=kv_transport,
        position_selection_ratio=position_selection_ratio,
    )
    return payload_bits + selector_bits


def _communication_bits(
    translator: RotAlignKVTranslator,
    seq_len: int,
    quantize: bool,
    translated_kv_control: str,
    protocol: str = "fused",
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
    *,
    active_k_head_counts: list[int] | None = None,
    active_v_head_counts: list[int] | None = None,
    active_k_token_counts: list[list[int]] | None = None,
    active_v_token_counts: list[list[int]] | None = None,
) -> float:
    if translated_kv_control != "real":
        return 0.0
    if protocol == "translated_only":
        return _translated_bits(
            translator,
            seq_len,
            quantize,
            active_k_head_counts=active_k_head_counts,
            active_v_head_counts=active_v_head_counts,
            active_k_token_counts=active_k_token_counts,
            active_v_token_counts=active_v_token_counts,
            kv_transport=kv_transport,
            position_selection_ratio=position_selection_ratio,
        )
    transport_k = kv_transport in {"both", "k_only"}
    transport_v = kv_transport in {"both", "v_only"}
    if active_k_head_counts is None:
        active_k_head_counts = []
        for layer_idx in translator.selected_layer_indices():
            gate_k, _ = translator.gate_value(layer_idx)
            if transport_k and abs(gate_k) > 1e-5:
                active_k_head_counts.append(translator.selected_head_count(layer_idx))
    if active_v_head_counts is None:
        active_v_head_counts = []
        for layer_idx in translator.selected_layer_indices():
            _, gate_v = translator.gate_value(layer_idx)
            if transport_v and abs(gate_v) > 1e-5:
                active_v_head_counts.append(translator.selected_head_count(layer_idx))
    return _translated_bits(
        translator,
        seq_len,
        quantize,
        active_k_head_counts=active_k_head_counts,
        active_v_head_counts=active_v_head_counts,
        active_k_token_counts=active_k_token_counts,
        active_v_token_counts=active_v_token_counts,
        kv_transport=kv_transport,
        position_selection_ratio=position_selection_ratio,
    )


def _apply_kv_transport(
    K_t: torch.Tensor,
    V_t: torch.Tensor,
    K_hat: torch.Tensor,
    V_hat: torch.Tensor,
    *,
    protocol: str,
    kv_transport: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_transport == "both":
        return K_hat, V_hat
    if kv_transport == "k_only":
        if protocol == "translated_only":
            return K_hat, torch.zeros_like(V_hat)
        return K_hat, V_t
    if kv_transport == "v_only":
        if protocol == "translated_only":
            return torch.zeros_like(K_hat), V_hat
        return K_t, V_hat
    raise ValueError(f"Unknown kv_transport: {kv_transport}")


def _position_selection_scores(
    K_t: torch.Tensor,
    V_t: torch.Tensor,
    K_hat: torch.Tensor,
    V_hat: torch.Tensor,
    *,
    kv_transport: str,
    position_selection_metric: str,
    position_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if position_selection_metric == "energy":
        if kv_transport == "k_only":
            active = K_hat.float()
            return active.pow(2).mean(dim=(0, 1, 3))
        if kv_transport == "v_only":
            active = V_hat.float()
            return active.pow(2).mean(dim=(0, 1, 3))
        return K_hat.float().pow(2).mean(dim=(0, 1, 3)) + V_hat.float().pow(2).mean(dim=(0, 1, 3))
    if position_selection_metric == "disagreement":
        if kv_transport == "k_only":
            return (K_hat.float() - K_t.float()).pow(2).mean(dim=(0, 1, 3))
        if kv_transport == "v_only":
            return (V_hat.float() - V_t.float()).pow(2).mean(dim=(0, 1, 3))
        return (
            (K_hat.float() - K_t.float()).pow(2).mean(dim=(0, 1, 3))
            + (V_hat.float() - V_t.float()).pow(2).mean(dim=(0, 1, 3))
        )
    if position_selection_metric == "random":
        if kv_transport == "k_only":
            salt = float(K_hat.float().sum().detach().cpu())
        elif kv_transport == "v_only":
            salt = float(V_hat.float().sum().detach().cpu())
        else:
            salt = float((K_hat.float().sum() + V_hat.float().sum()).detach().cpu())
        seed = 13_579 + (int(abs(salt) * 1_000) % 1_000_003)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.rand(K_hat.shape[2], generator=gen, dtype=torch.float32).to(device=K_hat.device)
    if position_selection_metric == "recency":
        return torch.arange(1, K_hat.shape[2] + 1, device=K_hat.device, dtype=torch.float32)
    if position_selection_metric == "attention":
        if position_scores is None:
            raise ValueError("attention-based position selection requires explicit position scores")
        return position_scores.float()
    if position_selection_metric == "attention_disagreement":
        if position_scores is None:
            raise ValueError("attention_disagreement position selection requires explicit position scores")
        attention_scores = position_scores.float()
        attention_scores = attention_scores / attention_scores.sum().clamp_min(1e-8)
        disagreement_scores = _position_selection_scores(
            K_t,
            V_t,
            K_hat,
            V_hat,
            kv_transport=kv_transport,
            position_selection_metric="disagreement",
        ).float()
        disagreement_scores = disagreement_scores / disagreement_scores.sum().clamp_min(1e-8)
        combined = attention_scores * disagreement_scores
        return combined / combined.sum().clamp_min(1e-8)
    if position_selection_metric == "attention_shuffled":
        if position_scores is None:
            raise ValueError("attention_shuffled position selection requires explicit position scores")
        salt = float(K_hat.float().sum().detach().cpu())
        seed = 24_681 + (int(abs(salt) * 1_000) % 1_000_003)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(position_scores.numel(), generator=gen)
        if position_scores.numel() > 1 and torch.equal(perm, torch.arange(position_scores.numel())):
            perm = torch.roll(perm, shifts=1)
        return position_scores.float()[perm.to(device=position_scores.device)]
    if position_selection_metric == "source_attention":
        if position_scores is None:
            raise ValueError("source_attention position selection requires explicit position scores")
        return position_scores.float()
    if position_selection_metric == "attention_prior":
        if position_scores is None:
            raise ValueError("attention_prior position selection requires explicit position scores")
        return position_scores.float()
    raise ValueError(f"Unknown position_selection_metric: {position_selection_metric}")


def _selector_trace(
    scores: torch.Tensor | None,
    keep_indices: torch.Tensor,
    seq_len: int,
    keep: int,
) -> dict[str, Any]:
    positions = sorted(int(idx) for idx in keep_indices.detach().cpu().tolist())
    trace: dict[str, Any] = {
        "seq_len": int(seq_len),
        "keep": int(keep),
        "keep_fraction": float(keep / max(seq_len, 1)),
        "selected_positions": positions[:16],
        "selected_count_truncated": int(max(len(positions) - 16, 0)),
    }
    if positions:
        trace["selected_min_pos"] = int(positions[0])
        trace["selected_max_pos"] = int(positions[-1])
        trace["selected_mean_pos"] = float(sum(positions) / len(positions))
    if scores is not None:
        probs = scores.float().detach()
        probs = probs / probs.sum().clamp_min(1e-8)
        sorted_scores = probs.sort(descending=True).values
        trace["score_entropy"] = float((-(probs * probs.clamp_min(1e-8).log()).sum()).item())
        trace["score_top"] = float(sorted_scores[0].item())
        trace["score_gap"] = float((sorted_scores[0] - sorted_scores[1]).item()) if sorted_scores.numel() > 1 else float(sorted_scores[0].item())
    return trace


def _allocate_per_head_token_budgets(
    head_scores: torch.Tensor,
    *,
    seq_len: int,
    position_selection_ratio: float,
    score_normalization: str = "minmax",
) -> torch.Tensor:
    head_scores = head_scores.float().view(-1)
    if head_scores.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=head_scores.device)
    total_keep = _selected_total_token_budget(seq_len, head_scores.numel(), position_selection_ratio)
    if total_keep >= seq_len * head_scores.numel():
        return torch.full((head_scores.numel(),), seq_len, dtype=torch.long, device=head_scores.device)

    if score_normalization == "minmax":
        weights = _normalize_selection_scores(head_scores)
    elif score_normalization == "sum":
        weights = head_scores.clamp_min(0.0)
    else:
        raise ValueError(f"Unknown score_normalization: {score_normalization}")
    if float(weights.sum().abs()) < 1e-8:
        weights = torch.ones_like(weights) / float(max(weights.numel(), 1))
    else:
        weights = weights / weights.sum().clamp_min(1e-8)

    raw = weights * float(total_keep)
    keep = torch.floor(raw).to(dtype=torch.long)
    keep = keep.clamp(min=0, max=seq_len)
    remainder = int(total_keep - int(keep.sum().item()))
    if remainder > 0:
        fractional = raw - keep.float()
        order = torch.argsort(fractional, descending=True)
        for idx in order.detach().cpu().tolist():
            if remainder <= 0:
                break
            if int(keep[idx].item()) >= seq_len:
                continue
            keep[idx] += 1
            remainder -= 1
    return keep


def _head_budget_trace(
    head_scores: torch.Tensor,
    head_keep_counts: torch.Tensor,
    active_head_indices: torch.Tensor,
    *,
    seq_len: int,
) -> dict[str, Any]:
    counts = [int(value) for value in head_keep_counts.detach().cpu().tolist()]
    active_heads = [int(idx) for idx in active_head_indices.detach().cpu().tolist()]
    total_keep = int(sum(counts))
    weights = _normalize_selection_scores(head_scores.float())
    weights = weights / weights.sum().clamp_min(1e-8)
    keep_weights = head_keep_counts.float()
    if float(keep_weights.sum().abs()) > 1e-8:
        keep_probs = keep_weights / keep_weights.sum().clamp_min(1e-8)
        keep_entropy = float((-(keep_probs * keep_probs.clamp_min(1e-8).log()).sum()).item())
    else:
        keep_entropy = 0.0
    preview = [
        {"head": head, "keep": keep}
        for head, keep in zip(active_heads[:8], counts[:8])
    ]
    return {
        "head_budget_total_keep": total_keep,
        "head_budget_keep_fraction": float(total_keep / max(seq_len * max(len(counts), 1), 1)),
        "head_budget_nonzero_heads": int(sum(1 for value in counts if value > 0)),
        "head_budget_nonzero_fraction": float(sum(1 for value in counts if value > 0) / max(len(counts), 1)),
        "head_budget_score_entropy": float((-(weights * weights.clamp_min(1e-8).log()).sum()).item()),
        "head_budget_keep_entropy": keep_entropy,
        "head_budget_preview": preview,
        "head_budget_preview_truncated": int(max(len(counts) - len(preview), 0)),
    }


def _apply_per_head_position_selection(
    K_t: torch.Tensor,
    V_t: torch.Tensor,
    K_hat: torch.Tensor,
    V_hat: torch.Tensor,
    *,
    protocol: str,
    kv_transport: str,
    position_selection_ratio: float,
    head_scores: torch.Tensor,
    per_head_position_scores: torch.Tensor,
    active_head_indices: torch.Tensor,
    score_normalization: str = "minmax",
    return_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
    seq_len = K_hat.shape[2]
    keep_counts = _allocate_per_head_token_budgets(
        head_scores,
        seq_len=seq_len,
        position_selection_ratio=position_selection_ratio,
        score_normalization=score_normalization,
    )
    if int(keep_counts.sum().item()) >= seq_len * max(int(active_head_indices.numel()), 1):
        trace = _head_budget_trace(head_scores, keep_counts, active_head_indices, seq_len=seq_len)
        return (K_hat, V_hat, trace, keep_counts) if return_trace else (K_hat, V_hat)

    full_mask = torch.zeros(
        K_hat.shape[1],
        seq_len,
        dtype=torch.bool,
        device=K_hat.device,
    )
    for local_idx, keep in enumerate(keep_counts.detach().cpu().tolist()):
        if keep <= 0:
            continue
        positions = torch.topk(per_head_position_scores[local_idx], k=int(keep), largest=True).indices
        full_mask[active_head_indices[local_idx], positions] = True
    mask = full_mask.view(1, full_mask.shape[0], seq_len, 1)

    if protocol == "translated_only":
        fill_k = torch.zeros_like(K_hat)
        fill_v = torch.zeros_like(V_hat)
    else:
        fill_k = K_t
        fill_v = V_t
    selected_k = torch.where(mask, K_hat, fill_k)
    selected_v = torch.where(mask, V_hat, fill_v)
    trace = _head_budget_trace(head_scores, keep_counts, active_head_indices, seq_len=seq_len)
    return (selected_k, selected_v, trace, keep_counts) if return_trace else (selected_k, selected_v)


def _apply_position_selection(
    K_t: torch.Tensor,
    V_t: torch.Tensor,
    K_hat: torch.Tensor,
    V_hat: torch.Tensor,
    *,
    protocol: str,
    kv_transport: str,
    position_selection_ratio: float,
    position_selection_metric: str,
    position_scores: torch.Tensor | None = None,
    return_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    seq_len = K_hat.shape[2]
    keep = _selected_token_count(seq_len, position_selection_ratio)
    if keep >= seq_len:
        trace = _selector_trace(None, torch.arange(seq_len, device=K_hat.device), seq_len, seq_len)
        return (K_hat, V_hat, trace) if return_trace else (K_hat, V_hat)

    scores = _position_selection_scores(
        K_t,
        V_t,
        K_hat,
        V_hat,
        kv_transport=kv_transport,
        position_selection_metric=position_selection_metric,
        position_scores=position_scores,
    )
    keep_indices = torch.topk(scores, k=keep, largest=True).indices
    mask = torch.zeros(seq_len, dtype=torch.bool, device=K_hat.device)
    mask[keep_indices] = True
    mask = mask.view(1, 1, seq_len, 1)

    if protocol == "translated_only":
        fill_k = torch.zeros_like(K_hat)
        fill_v = torch.zeros_like(V_hat)
    else:
        fill_k = K_t
        fill_v = V_t
    selected_k = torch.where(mask, K_hat, fill_k)
    selected_v = torch.where(mask, V_hat, fill_v)
    trace = _selector_trace(scores, keep_indices, seq_len, keep)
    return (selected_k, selected_v, trace) if return_trace else (selected_k, selected_v)


@torch.no_grad()
def _build_rotalign_prefix_state(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    translator: RotAlignKVTranslator,
    source_prompt: str,
    target_prompt: str,
    device: str,
    quantize: bool,
    protocol: str,
    source_kv_control: str = "real",
    quantization_control: str = "real",
    translated_kv_control: str = "real",
    fusion_rule: str = "static",
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
    position_selection_metric: str = "energy",
    fixed_position_profiles: list[torch.Tensor] | None = None,
    runtime_head_selection_ratio: float = 1.0,
    runtime_head_selection_metric: str = "attention_peak",
    fixed_head_profiles: list[torch.Tensor] | None = None,
    fixed_head_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_template_bank: list[torch.Tensor] | None = None,
    runtime_head_prior_alpha: float = 0.5,
    runtime_head_gate_metric: str = "none",
    runtime_head_gate_strength: float = 1.0,
    per_head_position_budget_mode: str = "none",
    source_use_chat_template: bool = False,
    target_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_enable_thinking: bool | None = None,
    drop_target_layers: set[int] | None = None,
    drop_target_layer_mode: str = "none",
) -> tuple[PrefixState, dict[str, float]]:
    formatted_target_prompt = _format_prompt_for_tokenizer(
        target_tokenizer,
        target_prompt,
        use_chat_template=target_use_chat_template,
        enable_thinking=target_enable_thinking,
    )
    tgt_prompt_ids = target_tokenizer(formatted_target_prompt, return_tensors="pt").input_ids.to(device)
    tgt_prefix_ids, tgt_last_token = _split_prompt_prefix(tgt_prompt_ids)
    if tgt_prefix_ids is None:
        return PrefixState(None, tgt_last_token, 1), {"bits": 0.0}
    tgt_out = target_model(
        input_ids=tgt_prefix_ids,
        attention_mask=_ones(tgt_prefix_ids.shape[1], device),
        use_cache=True,
    )
    tgt_pkv = list(_normalize_cache(tgt_out.past_key_values))

    formatted_source_prompt = _format_prompt_for_tokenizer(
        source_tokenizer,
        source_prompt,
        use_chat_template=source_use_chat_template,
        enable_thinking=source_enable_thinking,
    )
    src_ids = source_tokenizer(formatted_source_prompt, return_tensors="pt").input_ids.to(device)
    src_prefix_ids, src_last_token = _split_prompt_prefix(src_ids)
    if src_prefix_ids is None:
        src_pkv = []
    else:
        src_out = source_model(
            input_ids=src_prefix_ids,
            attention_mask=_ones(src_prefix_ids.shape[1], device),
            use_cache=True,
        )
        src_pkv = list(_normalize_cache(src_out.past_key_values))

    layer_attention_maps: list[torch.Tensor] | None = None
    layer_query_heads: list[torch.Tensor] | None = None
    runtime_head_attention_metrics = {
        "attention_peak",
        "attention_entropy",
        "attention_margin",
        "retrieval_peak",
        "attention_fidelity",
        "attention_fidelity_shuffled",
        "attention_procrustes",
        "attention_procrustes_shuffled",
        "attention_qk_fidelity",
        "attention_qk_fidelity_shuffled",
        "attention_qk_bank_transport",
        "attention_qk_bank_transport_shuffled",
        "attention_template_transport",
        "attention_template_transport_shuffled",
        "attention_sinkhorn",
        "attention_sinkhorn_shuffled",
        "attention_expected",
        "attention_expected_shuffled",
        "attention_match",
        "attention_match_shuffled",
        "attention_blend",
    }
    if (
        position_selection_metric in {"attention", "attention_disagreement", "attention_shuffled"}
        and position_selection_ratio < 1.0
    ) or (
        runtime_head_selection_ratio < 1.0
        and runtime_head_selection_metric in runtime_head_attention_metrics
    ) or (
        runtime_head_gate_metric in runtime_head_attention_metrics
    ) or (
        per_head_position_budget_mode != "none"
    ):
        layer_attention_maps = _last_token_attention_maps(
            target_model,
            tgt_last_token,
            tuple(tgt_pkv),
            device,
            translator=translator,
        )
    if (
        runtime_head_selection_ratio < 1.0
        and runtime_head_selection_metric in {
            "attention_qk_fidelity",
            "attention_qk_fidelity_shuffled",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
        }
    ) or runtime_head_gate_metric in {
        "attention_qk_fidelity",
        "attention_qk_fidelity_shuffled",
        "attention_qk_fidelity_tokenwise",
        "attention_qk_template_transport",
        "attention_qk_template_transport_shuffled",
        "attention_qk_bank_transport",
        "attention_qk_bank_transport_shuffled",
    } or per_head_position_budget_mode in {
        "attention_qk_fidelity",
        "attention_qk_fidelity_shuffled",
        "attention_qk_template_transport",
        "attention_qk_template_transport_shuffled",
        "attention_qk_bank_transport",
        "attention_qk_bank_transport_shuffled",
    } or translator.config.quantization_correction in {"bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank", "bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter"}:
        layer_query_heads = _last_token_query_heads(
            target_model,
            tgt_last_token,
            tuple(tgt_pkv),
            device,
            translator=translator,
        )
    source_layer_position_scores: list[torch.Tensor] | None = None
    if position_selection_metric == "source_attention" and position_selection_ratio < 1.0:
        if src_prefix_ids is None or src_last_token is None:
            raise ValueError("source_attention position selection requires a non-empty source prefix")
        source_layer_position_scores = _last_token_attention_scores(
            source_model,
            src_last_token,
            tuple(src_pkv),
            device,
            translator=None,
        )
    if position_selection_metric == "attention_prior" and position_selection_ratio < 1.0 and fixed_position_profiles is None:
        raise ValueError("attention_prior position selection requires fixed_position_profiles")
    if (
        runtime_head_selection_ratio < 1.0
        and runtime_head_selection_metric in {
            "attention_prior",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
            "attention_match",
            "attention_match_shuffled",
            "attention_blend",
        }
        and fixed_head_profiles is None
    ):
        raise ValueError(
            f"{runtime_head_selection_metric} runtime head selection requires fixed_head_profiles"
        )
    if (
        runtime_head_selection_ratio < 1.0
        and runtime_head_selection_metric in {
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
        }
        and fixed_head_templates is None
        and fixed_head_qk_templates is None
        and fixed_head_qk_template_bank is None
    ):
        raise ValueError(
            f"{runtime_head_selection_metric} runtime head selection requires fixed_head_templates, fixed_head_qk_templates, or fixed_head_qk_template_bank"
        )
    if per_head_position_budget_mode in {
        "attention_prior",
        "attention_prior_shuffled",
        "attention_qk_template_transport",
        "attention_qk_template_transport_shuffled",
        "attention_qk_bank_transport",
        "attention_qk_bank_transport_shuffled",
        "attention_template_transport",
        "attention_template_transport_shuffled",
        "attention_sinkhorn",
        "attention_sinkhorn_shuffled",
        "attention_match",
        "attention_match_shuffled",
        "attention_blend",
    } and fixed_head_profiles is None:
        raise ValueError(
            f"{per_head_position_budget_mode} per-head position budgets require fixed_head_profiles"
        )
    if per_head_position_budget_mode in {
        "attention_qk_template_transport",
        "attention_qk_template_transport_shuffled",
        "attention_qk_bank_transport",
        "attention_qk_bank_transport_shuffled",
        "attention_template_transport",
        "attention_template_transport_shuffled",
        "attention_sinkhorn",
        "attention_sinkhorn_shuffled",
    } and fixed_head_templates is None and fixed_head_qk_templates is None and fixed_head_qk_template_bank is None:
        raise ValueError(
            f"{per_head_position_budget_mode} per-head position budgets require fixed_head_templates, fixed_head_qk_templates, or fixed_head_qk_template_bank"
        )

    fused_pkv: list[tuple[torch.Tensor, torch.Tensor]] = []
    selector_layers: list[dict[str, Any]] = []
    head_layers: list[dict[str, Any]] = []
    head_budget_layers: list[dict[str, Any]] = []
    head_gate_layers: list[dict[str, Any]] = []
    layer_runtime_head_counts = [
        translator.selected_head_count(layer_idx)
        for layer_idx in range(translator.config.num_tgt_layers)
    ]
    active_k_token_counts: list[list[int]] = []
    active_v_token_counts: list[list[int]] = []
    dropped_layer_set = set(drop_target_layers or set())
    for tgt_l in range(translator.config.num_tgt_layers):
        src_l = translator.layer_map[tgt_l]
        K_s, V_s = src_pkv[src_l]
        K_s, V_s = _apply_source_kv_control(K_s, V_s, source_kv_control, tgt_l)
        K_t, V_t = tgt_pkv[tgt_l]
        runtime_attention_profile = None
        runtime_query_features = None
        if translator.config.quantization_correction in {"bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
            if layer_query_heads is None:
                raise ValueError(f"{translator.config.quantization_correction} requires live query heads")
            runtime_attention_profile = _qk_position_distributions(
                layer_query_heads[tgt_l],
                K_t.to(device=device, dtype=torch.float32),
                bins=translator.config.transport_template_bins,
            ).mean(dim=0)
            if translator.config.quantization_correction in {"bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
                runtime_query_features = _query_feature_vector(
                    layer_query_heads[tgt_l],
                    target_heads=translator.config.tgt_num_heads,
                    head_dim=translator.config.tgt_head_dim,
                )
        elif translator.config.quantization_correction in {"bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter"}:
            if layer_query_heads is None:
                raise ValueError(f"{translator.config.quantization_correction} requires live query heads")
            runtime_query_features = _query_feature_vector(
                layer_query_heads[tgt_l],
                target_heads=translator.config.tgt_num_heads,
                head_dim=translator.config.tgt_head_dim,
            )
        elif layer_attention_maps is not None:
            runtime_attention_profile = _resample_position_profile(
                layer_attention_maps[tgt_l].mean(dim=0),
                translator.config.transport_template_bins,
            )
        K_hat, V_hat = _translate_layer_with_optional_runtime_attention(
            translator,
            K_s.to(device=device, dtype=torch.float32),
            V_s.to(device=device, dtype=torch.float32),
            tgt_layer_idx=tgt_l,
            quantize=quantize,
            quantization_control=quantization_control,
            runtime_attention_profile=runtime_attention_profile,
            runtime_query_features=runtime_query_features,
        )
        K_hat, V_hat = _apply_translated_kv_control(
            K_hat,
            V_hat,
            translated_kv_control,
            tgt_l,
        )
        K_t_aligned, K_hat = _tail_align_pair(K_t, K_hat.to(dtype=K_t.dtype))
        V_t_aligned, V_hat = _tail_align_pair(V_t, V_hat.to(dtype=V_t.dtype))
        layer_dropped = drop_target_layer_mode != "none" and tgt_l in dropped_layer_set
        if layer_dropped:
            if drop_target_layer_mode == "target":
                K_hat = K_t_aligned.clone()
                V_hat = V_t_aligned.clone()
            elif drop_target_layer_mode == "zero":
                K_hat = torch.zeros_like(K_hat)
                V_hat = torch.zeros_like(V_hat)
            else:
                raise ValueError(f"Unknown drop_target_layer_mode: {drop_target_layer_mode}")
            layer_runtime_head_counts[tgt_l] = 0
        runtime_position_scores: torch.Tensor | None = None
        active_head_indices = _translator_selected_head_indices(
            translator,
            tgt_l,
            device=K_t_aligned.device,
        )
        active_attention_map = (
            layer_attention_maps[tgt_l][:, -K_t_aligned.shape[2] :]
            if layer_attention_maps is not None
            else None
        )
        prior_scores = None
        position_prior_scores = None
        if fixed_head_profiles is not None:
            layer_head_profile = fixed_head_profiles[tgt_l].to(device=K_t_aligned.device)
            if layer_head_profile.numel() == translator.config.tgt_num_heads:
                prior_scores = _normalize_selection_scores(layer_head_profile[active_head_indices])
            else:
                prior_scores = _resample_head_profile(
                    layer_head_profile,
                    active_head_indices.numel(),
                )
        if fixed_position_profiles is not None:
            position_prior_scores = fixed_position_profiles[tgt_l].to(device=K_t_aligned.device)
        template_scores = None
        if fixed_head_templates is not None:
            layer_templates = fixed_head_templates[tgt_l].to(device=K_t_aligned.device)
            if layer_templates.shape[0] == translator.config.tgt_num_heads:
                template_scores = layer_templates[active_head_indices]
            else:
                template_scores = layer_templates
        qk_template_scores = None
        if fixed_head_qk_templates is not None:
            layer_qk_templates = fixed_head_qk_templates[tgt_l].to(device=K_t_aligned.device)
            if layer_qk_templates.shape[0] == translator.config.tgt_num_heads:
                qk_template_scores = layer_qk_templates[active_head_indices]
            else:
                qk_template_scores = layer_qk_templates
        qk_template_bank_scores = None
        if fixed_head_qk_template_bank is not None:
            layer_qk_template_bank = fixed_head_qk_template_bank[tgt_l].to(device=K_t_aligned.device)
            if layer_qk_template_bank.shape[1] == translator.config.tgt_num_heads:
                qk_template_bank_scores = layer_qk_template_bank[:, active_head_indices, :]
            else:
                qk_template_bank_scores = layer_qk_template_bank
        active_target_keys = K_t_aligned[:, active_head_indices]
        active_translated_keys = K_hat[:, active_head_indices]
        active_query_heads = (
            layer_query_heads[tgt_l][active_head_indices]
            if layer_query_heads is not None
            else None
        )
        head_gate_override_k: torch.Tensor | None = None
        head_gate_override_v: torch.Tensor | None = None
        if runtime_head_selection_ratio < 1.0:
            attention_map = active_attention_map
            original_head_indices = active_head_indices
            available_heads = attention_map.shape[0] if attention_map is not None else original_head_indices.numel()
            keep_heads = _selected_head_count(available_heads, runtime_head_selection_ratio)
            head_scores, prior_ranking = _runtime_head_scores_with_prior(
                attention_map,
                metric=runtime_head_selection_metric,
                layer_idx=tgt_l,
                prior_scores=prior_scores,
                position_prior=position_prior_scores,
                target_keys=active_target_keys,
                translated_keys=active_translated_keys,
                query_heads=active_query_heads,
                head_templates=template_scores,
                qk_templates=qk_template_scores,
                qk_template_bank=qk_template_bank_scores,
                prior_alpha=runtime_head_prior_alpha,
            )
            keep_heads = min(keep_heads, int(head_scores.numel()))
            keep_local = torch.topk(head_scores, k=keep_heads, largest=True).indices
            prior_keep_local = None
            if prior_scores is not None:
                prior_keep_local = torch.topk(prior_scores, k=keep_heads, largest=True).indices
            full_head_mask = torch.zeros(
                translator.config.tgt_num_heads,
                dtype=torch.bool,
                device=K_t_aligned.device,
            )
            full_head_mask[original_head_indices[keep_local]] = True
            K_hat = _apply_runtime_head_selection(
                K_hat,
                K_t_aligned,
                full_head_mask,
                protocol=protocol,
            )
            V_hat = _apply_runtime_head_selection(
                V_hat,
                V_t_aligned,
                full_head_mask,
                protocol=protocol,
            )
            if attention_map is not None:
                runtime_position_scores = attention_map[keep_local].mean(dim=0)
            head_trace = _head_trace(
                head_scores,
                keep_local,
                original_head_indices,
                keep=keep_heads,
                prior_local_indices=prior_keep_local,
                attention_map=attention_map,
            )
            head_trace["target_layer"] = int(tgt_l)
            head_trace["source_layer"] = int(src_l)
            head_trace["metric"] = runtime_head_selection_metric
            if prior_keep_local is not None:
                head_trace["prior_metric"] = "attention_peak"
            if prior_scores is not None:
                head_trace["prior_alpha"] = float(runtime_head_prior_alpha)
            head_layers.append(head_trace)
            layer_runtime_head_counts[tgt_l] = int(full_head_mask.sum().item())
            active_head_indices = original_head_indices[keep_local]
            if attention_map is not None:
                active_attention_map = attention_map[keep_local]
            active_target_keys = K_t_aligned[:, active_head_indices]
            active_translated_keys = K_hat[:, active_head_indices]
            if prior_scores is not None:
                prior_scores = prior_scores[keep_local]
            if template_scores is not None:
                template_scores = template_scores[keep_local]
            if qk_template_scores is not None:
                qk_template_scores = qk_template_scores[keep_local]
            if qk_template_bank_scores is not None:
                qk_template_bank_scores = qk_template_bank_scores[:, keep_local, :]
            if active_query_heads is not None:
                active_query_heads = active_query_heads[keep_local]
        elif layer_attention_maps is not None:
            runtime_position_scores = layer_attention_maps[tgt_l][-K_t_aligned.shape[2] :].mean(dim=0)
        gate_k_now, gate_v_now = translator.gate_value(tgt_l)
        if runtime_head_gate_metric != "none" and translator.is_layer_selected(tgt_l):
            token_gate_scores: torch.Tensor | None = None
            if runtime_head_gate_metric == "attention_qk_fidelity_tokenwise":
                if active_query_heads is None:
                    raise ValueError(
                        "attention_qk_fidelity_tokenwise runtime head gating requires query_heads, target_keys, and translated_keys"
                    )
                gate_scores = _attention_qk_fidelity_head_scores(
                    active_query_heads,
                    active_target_keys,
                    active_translated_keys,
                )
                token_gate_scores = _attention_qk_fidelity_token_scores(
                    active_query_heads,
                    active_target_keys,
                    active_translated_keys,
                )
            else:
                gate_scores, _ = _runtime_head_scores_with_prior(
                    active_attention_map,
                    metric=runtime_head_gate_metric,
                    layer_idx=tgt_l,
                    prior_scores=prior_scores,
                    position_prior=position_prior_scores,
                    target_keys=active_target_keys,
                    translated_keys=active_translated_keys,
                    query_heads=active_query_heads,
                    head_templates=template_scores,
                    qk_templates=qk_template_scores,
                    qk_template_bank=qk_template_bank_scores,
                    prior_alpha=runtime_head_prior_alpha,
                )
            active_gate_k = _per_head_gate_override_from_scores(
                gate_k_now,
                gate_scores,
                strength=runtime_head_gate_strength,
            ).to(device=K_t_aligned.device, dtype=K_t_aligned.dtype)
            active_gate_v = _per_head_gate_override_from_scores(
                gate_v_now,
                gate_scores,
                strength=runtime_head_gate_strength,
            ).to(device=K_t_aligned.device, dtype=K_t_aligned.dtype)
            if token_gate_scores is None:
                full_gate = torch.zeros(
                    translator.config.tgt_num_heads,
                    dtype=K_t_aligned.dtype,
                    device=K_t_aligned.device,
                )
                if kv_transport in {"both", "k_only"}:
                    full_gate[active_head_indices] = active_gate_k
                    head_gate_override_k = full_gate
                if kv_transport in {"both", "v_only"}:
                    full_gate_v = torch.zeros_like(full_gate)
                    full_gate_v[active_head_indices] = active_gate_v
                    head_gate_override_v = full_gate_v
            else:
                full_gate = torch.zeros(
                    1,
                    translator.config.tgt_num_heads,
                    K_t_aligned.shape[2],
                    1,
                    dtype=K_t_aligned.dtype,
                    device=K_t_aligned.device,
                )
                if kv_transport in {"both", "k_only"}:
                    token_gate = _tokenwise_gate_override_from_scores(
                        active_gate_k,
                        token_gate_scores,
                        strength=runtime_head_gate_strength,
                    ).to(device=full_gate.device, dtype=full_gate.dtype)
                    full_gate[:, active_head_indices, :, :] = token_gate
                    head_gate_override_k = full_gate
                if kv_transport in {"both", "v_only"}:
                    full_gate_v = torch.zeros_like(full_gate)
                    token_gate_v = _tokenwise_gate_override_from_scores(
                        active_gate_v,
                        token_gate_scores,
                        strength=runtime_head_gate_strength,
                    ).to(device=full_gate_v.device, dtype=full_gate_v.dtype)
                    full_gate_v[:, active_head_indices, :, :] = token_gate_v
                    head_gate_override_v = full_gate_v
            top_local = torch.topk(gate_scores, k=min(int(gate_scores.numel()), 3), largest=True).indices
            head_gate_layers.append(
                {
                    "target_layer": int(tgt_l),
                    "source_layer": int(src_l),
                    "metric": runtime_head_gate_metric,
                    "strength": float(runtime_head_gate_strength),
                    "base_gate_k": float(gate_k_now),
                    "base_gate_v": float(gate_v_now),
                    "mean_score": float(gate_scores.mean().item()),
                    "score_gap": float((gate_scores.max() - gate_scores.min()).item()),
                    "tokenwise": bool(token_gate_scores is not None),
                    "top_target_heads": [int(active_head_indices[idx].item()) for idx in top_local],
                }
            )
        if per_head_position_budget_mode != "none" and position_selection_ratio < 1.0:
            if active_attention_map is None:
                raise ValueError("per-head position budgets require target attention maps")
            head_budget_scores, _ = _runtime_head_scores_with_prior(
                active_attention_map,
                metric=per_head_position_budget_mode,
                layer_idx=tgt_l,
                prior_scores=prior_scores,
                position_prior=position_prior_scores,
                target_keys=active_target_keys,
                translated_keys=active_translated_keys,
                query_heads=active_query_heads,
                head_templates=template_scores,
                qk_templates=qk_template_scores,
                qk_template_bank=qk_template_bank_scores,
                prior_alpha=runtime_head_prior_alpha,
            )
            K_hat, V_hat, head_budget_trace, keep_counts = _apply_per_head_position_selection(
                K_t_aligned,
                V_t_aligned,
                K_hat,
                V_hat,
                protocol=protocol,
                kv_transport=kv_transport,
                position_selection_ratio=position_selection_ratio,
                head_scores=head_budget_scores,
                per_head_position_scores=active_attention_map,
                active_head_indices=active_head_indices,
                score_normalization=(
                    "sum"
                    if per_head_position_budget_mode in {
                        "attention_prior",
                        "attention_prior_shuffled",
                        "attention_procrustes",
                        "attention_procrustes_shuffled",
                        "attention_qk_fidelity",
                        "attention_qk_fidelity_shuffled",
                        "attention_qk_template_transport",
                        "attention_qk_template_transport_shuffled",
                        "attention_template_transport",
                        "attention_template_transport_shuffled",
                        "attention_sinkhorn",
                        "attention_sinkhorn_shuffled",
                        "attention_match",
                        "attention_match_shuffled",
                        "attention_blend",
                        "attention_expected",
                        "attention_expected_shuffled",
                    }
                    else "minmax"
                ),
                return_trace=True,
            )
            head_budget_trace["target_layer"] = int(tgt_l)
            head_budget_trace["source_layer"] = int(src_l)
            head_budget_trace["metric"] = per_head_position_budget_mode
            head_budget_layers.append(head_budget_trace)
            selector_layers.append(
                {
                    "target_layer": int(tgt_l),
                    "source_layer": int(src_l),
                    "metric": f"per_head_{per_head_position_budget_mode}",
                    "keep_fraction": float(
                        keep_counts.sum().item()
                        / max(K_t_aligned.shape[2] * max(active_head_indices.numel(), 1), 1)
                    ),
                    "target_layer_drop_mode": drop_target_layer_mode if layer_dropped else "none",
                }
            )
            keep_list = [int(value) for value in keep_counts.detach().cpu().tolist()]
            if (
                not layer_dropped
                and translator.is_layer_selected(tgt_l)
                and kv_transport in {"both", "k_only"}
                and (protocol == "translated_only" or abs(gate_k_now) > 1e-5)
            ):
                active_k_token_counts.append(keep_list)
            if (
                not layer_dropped
                and translator.is_layer_selected(tgt_l)
                and kv_transport in {"both", "v_only"}
                and (protocol == "translated_only" or abs(gate_v_now) > 1e-5)
            ):
                active_v_token_counts.append(keep_list)
        else:
            K_hat, V_hat, selector_trace = _apply_position_selection(
                K_t_aligned,
                V_t_aligned,
                K_hat,
                V_hat,
                protocol=protocol,
                kv_transport=kv_transport,
                position_selection_ratio=position_selection_ratio,
                position_selection_metric=position_selection_metric,
                position_scores=(
                    runtime_position_scores
                    if runtime_position_scores is not None
                    else (
                        source_layer_position_scores[src_l][-K_t_aligned.shape[2] :]
                        if source_layer_position_scores is not None
                        else (
                            _resample_position_profile(
                                fixed_position_profiles[tgt_l].to(device=K_t_aligned.device),
                                K_t_aligned.shape[2],
                            )
                            if fixed_position_profiles is not None
                            else None
                        )
                    )
                ),
                return_trace=True,
            )
            selector_trace["target_layer"] = int(tgt_l)
            selector_trace["source_layer"] = int(src_l)
            selector_trace["metric"] = position_selection_metric
            selector_trace["target_layer_drop_mode"] = drop_target_layer_mode if layer_dropped else "none"
            selector_layers.append(selector_trace)
        K_hat, V_hat = _apply_kv_transport(
            K_t_aligned,
            V_t_aligned,
            K_hat,
            V_hat,
            protocol=protocol,
            kv_transport=kv_transport,
        )

        if protocol == "translated_only":
            if translator.is_layer_selected(tgt_l):
                fused_pkv.append(
                    (
                        translator.apply_head_selection(K_hat, tgt_l),
                        translator.apply_head_selection(V_hat, tgt_l),
                    )
                )
            else:
                fused_pkv.append((torch.zeros_like(K_t_aligned), torch.zeros_like(V_t_aligned)))
        elif protocol in {"fused", "text_kv_hybrid"}:
            if translator.is_layer_selected(tgt_l):
                fused_pkv.append(
                    translator.fuse_layer(
                        K_t_aligned,
                        V_t_aligned,
                        K_hat,
                        V_hat,
                        tgt_l,
                        fusion_rule=fusion_rule,
                        head_gate_override_K=head_gate_override_k,
                        head_gate_override_V=head_gate_override_v,
                    )
                )
            else:
                fused_pkv.append((K_t_aligned, V_t_aligned))
        else:
            raise ValueError(f"Unknown RotAlign protocol: {protocol}")

    seq_len = fused_pkv[0][0].shape[2] if fused_pkv else 0
    active_k_head_counts: list[int] = []
    active_v_head_counts: list[int] = []
    for layer_idx in translator.selected_layer_indices():
        if layer_idx in dropped_layer_set and drop_target_layer_mode != "none":
            continue
        gate_k, gate_v = translator.gate_value(layer_idx)
        selected_heads = layer_runtime_head_counts[layer_idx]
        if kv_transport in {"both", "k_only"} and (protocol == "translated_only" or abs(gate_k) > 1e-5):
            active_k_head_counts.append(selected_heads)
        if kv_transport in {"both", "v_only"} and (protocol == "translated_only" or abs(gate_v) > 1e-5):
            active_v_head_counts.append(selected_heads)
    payload_bits, selector_bits = (0.0, 0.0)
    if translated_kv_control == "real":
        payload_bits, selector_bits = _translated_bit_breakdown(
            translator,
            seq_len,
            quantize,
            active_k_head_counts=active_k_head_counts,
            active_v_head_counts=active_v_head_counts,
            active_k_token_counts=active_k_token_counts if active_k_token_counts else None,
            active_v_token_counts=active_v_token_counts if active_v_token_counts else None,
            kv_transport=kv_transport,
            position_selection_ratio=position_selection_ratio,
        )
    prefix_state = PrefixState(
        past_key_values=tuple(fused_pkv),
        last_token=tgt_last_token,
        prefix_len=seq_len + 1,
    )
    total_bits = float(payload_bits + selector_bits)
    return prefix_state, {
        "bits": total_bits,
        "bytes": total_bits / 8.0,
        "payload_bits": float(payload_bits),
        "selector_bits": float(selector_bits),
        "metadata_bits": float(selector_bits),
        "selector_trace": selector_layers,
        "head_trace": head_layers,
        "head_gate_trace": head_gate_layers,
        "head_budget_trace": head_budget_layers,
        "selected_target_layers": [int(layer) for layer in translator.selected_layer_indices()],
        "dropped_target_layers": sorted(int(layer) for layer in dropped_layer_set),
        "drop_target_layer_mode": drop_target_layer_mode,
    }


def _limit_examples(examples, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(examples):
        return list(examples)
    return list(examples[:limit])


def _parse_target_layer_set(value: str | None) -> set[int]:
    if value is None or not value.strip():
        return set()
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if layer < 0:
            raise ValueError("--drop-target-layers must contain non-negative integers")
        layers.add(layer)
    return layers


def _search_per_layer_gates(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    translator: RotAlignKVTranslator,
    examples,
    device,
    quantize: bool,
    protocol: str,
    source_reasoning_mode: str,
    gate_values: list[float],
    source_kv_control: str = "real",
    quantization_control: str = "real",
    translated_kv_control: str = "real",
    fusion_rule: str = "static",
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
    position_selection_metric: str = "energy",
    fixed_position_profiles: list[torch.Tensor] | None = None,
    runtime_head_selection_ratio: float = 1.0,
    runtime_head_selection_metric: str = "attention_peak",
    fixed_head_profiles: list[torch.Tensor] | None = None,
    fixed_head_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_template_bank: list[torch.Tensor] | None = None,
    runtime_head_prior_alpha: float = 0.5,
    runtime_head_gate_metric: str = "none",
    runtime_head_gate_strength: float = 1.0,
    per_head_position_budget_mode: str = "none",
) -> dict[str, list[float]]:
    """Coordinate-descent line search over per-layer K/V fusion gates.

    The objective is held-out task accuracy on the supplied examples. We tune
    K and V separately, one selected layer at a time, while keeping the other
    gates fixed to their current values.
    """
    if not examples:
        raise ValueError("Gate search requires at least one held-out example")

    candidates = [float(value) for value in gate_values]
    layer_indices = translator.selected_layer_indices()
    if not layer_indices:
        layer_indices = list(range(translator.config.num_tgt_layers))
    is_generation = bool(examples) and isinstance(examples[0], GenerationExample)
    search_k = kv_transport in {"both", "k_only"}
    search_v = kv_transport in {"both", "v_only"}

    def _score_current_gates() -> float:
        eval_kwargs: dict[str, Any] = {}
        if fixed_position_profiles is not None:
            eval_kwargs["fixed_position_profiles"] = fixed_position_profiles
        if fixed_head_profiles is not None:
            eval_kwargs["fixed_head_profiles"] = fixed_head_profiles
        if fixed_head_templates is not None:
            eval_kwargs["fixed_head_templates"] = fixed_head_templates
        if fixed_head_qk_templates is not None:
            eval_kwargs["fixed_head_qk_templates"] = fixed_head_qk_templates
        if fixed_head_qk_template_bank is not None:
            eval_kwargs["fixed_head_qk_template_bank"] = fixed_head_qk_template_bank
        eval_kwargs["runtime_head_prior_alpha"] = runtime_head_prior_alpha
        eval_kwargs["runtime_head_gate_metric"] = runtime_head_gate_metric
        eval_kwargs["runtime_head_gate_strength"] = runtime_head_gate_strength
        eval_kwargs["per_head_position_budget_mode"] = per_head_position_budget_mode
        if is_generation:
            return _eval_generation_rotalign(
                source_model,
                source_tokenizer,
                target_model,
                target_tokenizer,
                translator,
                examples,
                device,
                max_new_tokens=64,
                quantize=quantize,
                protocol=protocol,
                source_reasoning_mode=source_reasoning_mode,
                source_kv_control=source_kv_control,
                quantization_control=quantization_control,
                translated_kv_control=translated_kv_control,
                fusion_rule=fusion_rule,
                kv_transport=kv_transport,
                position_selection_ratio=position_selection_ratio,
                position_selection_metric=position_selection_metric,
                runtime_head_selection_ratio=runtime_head_selection_ratio,
                runtime_head_selection_metric=runtime_head_selection_metric,
                **eval_kwargs,
            )[0]
        return eval_rotalign_kv(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            translator,
            examples,
            device,
            quantize=quantize,
            protocol=protocol,
            source_reasoning_mode=source_reasoning_mode,
            source_kv_control=source_kv_control,
            quantization_control=quantization_control,
            translated_kv_control=translated_kv_control,
            fusion_rule=fusion_rule,
            kv_transport=kv_transport,
            position_selection_ratio=position_selection_ratio,
            position_selection_metric=position_selection_metric,
            runtime_head_selection_ratio=runtime_head_selection_ratio,
            runtime_head_selection_metric=runtime_head_selection_metric,
            **eval_kwargs,
        )

    for tgt_layer_idx in layer_indices:
        current_k, current_v = translator.gate_value(tgt_layer_idx)

        best_k = current_k
        best_k_score: float | None = None
        if search_k:
            best_k_score = float("-inf")
            for candidate in candidates:
                translator.set_layer_gates(tgt_layer_idx, alpha_k=candidate, alpha_v=current_v)
                score = _score_current_gates()
                if score > best_k_score:
                    best_k_score = score
                    best_k = candidate

        translator.set_layer_gates(tgt_layer_idx, alpha_k=best_k, alpha_v=current_v)

        best_v = current_v
        best_v_score: float | None = None
        if search_v:
            best_v_score = float("-inf")
            for candidate in candidates:
                translator.set_layer_gates(tgt_layer_idx, alpha_k=best_k, alpha_v=candidate)
                score = _score_current_gates()
                if score > best_v_score:
                    best_v_score = score
                    best_v = candidate

        translator.set_layer_gates(tgt_layer_idx, alpha_k=best_k, alpha_v=best_v)
        k_score_display = "skipped" if best_k_score is None else f"{best_k_score:.4f}"
        v_score_display = "skipped" if best_v_score is None else f"{best_v_score:.4f}"
        print(
            f"[gate search] layer {tgt_layer_idx:>2d}: "
            f"K={best_k:.3f} (score={k_score_display})  "
            f"V={best_v:.3f} (score={v_score_display})"
        )

    return {
        "gate_K": [translator.gate_value(idx)[0] for idx in range(translator.config.num_tgt_layers)],
        "gate_V": [translator.gate_value(idx)[1] for idx in range(translator.config.num_tgt_layers)],
    }


def _restore_gate_values(
    translator: RotAlignKVTranslator, gate_values: list[tuple[float, float]]
) -> None:
    for layer_idx, (gate_k, gate_v) in enumerate(gate_values):
        translator.set_layer_gates(layer_idx, alpha_k=gate_k, alpha_v=gate_v)


def _mcq_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _source_reasoning_prompt(prompt: str, mode: str) -> str:
    prompt = prompt.strip()
    if mode == "plain":
        return prompt
    if mode == "brief_analysis":
        return (
            "Briefly analyze this problem in one sentence:\n"
            f"{prompt}\nAnalysis:"
        )
    if mode == "cot":
        return f"Let's think step by step.\n{prompt}\nReasoning:"
    if mode == "scratchpad":
        return f"Use a scratchpad to work through the problem carefully.\n{prompt}\nScratchpad:"
    raise ValueError(f"Unknown source reasoning mode: {mode}")


def _generate_source_hint(
    source_model,
    source_tokenizer,
    prompt: str,
    device: str,
    source_reasoning_mode: str,
    *,
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
) -> str:
    hint_prompt = _source_reasoning_prompt(prompt, source_reasoning_mode)
    formatted_hint_prompt = _format_prompt_for_tokenizer(
        source_tokenizer,
        hint_prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    hint_ids = source_tokenizer(formatted_hint_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = source_model.generate(
            hint_ids,
            attention_mask=torch.ones_like(hint_ids),
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=source_tokenizer.eos_token_id,
        )
    return source_tokenizer.decode(out[0, hint_ids.shape[1] :], skip_special_tokens=True).strip()


def eval_target_alone(target_model, target_tokenizer, examples, device, records: list[dict[str, Any]] | None = None):
    correct = 0
    for idx, ex in enumerate(examples):
        prompt = _mcq_prompt(ex.question)
        scores = [
            score_choice_loglik(target_model, target_tokenizer, prompt, c, device)
            for c in ex.choices
        ]
        pred = int(torch.tensor(scores).argmax())
        is_correct = pred == ex.answer
        _append_prediction_record(
            records,
            index=idx,
            example_id=_mcq_example_id(ex),
            method="target_alone",
            prediction=pred,
            answer=ex.answer,
            correct=is_correct,
        )
        if is_correct:
            correct += 1
    return correct / len(examples)


def eval_source_alone(source_model, source_tokenizer, examples, device, records: list[dict[str, Any]] | None = None):
    correct = 0
    for idx, ex in enumerate(examples):
        prompt = _mcq_prompt(ex.question)
        scores = [
            score_choice_loglik(source_model, source_tokenizer, prompt, c, device)
            for c in ex.choices
        ]
        pred = int(torch.tensor(scores).argmax())
        is_correct = pred == ex.answer
        _append_prediction_record(
            records,
            index=idx,
            example_id=_mcq_example_id(ex),
            method="source_alone",
            prediction=pred,
            answer=ex.answer,
            correct=is_correct,
        )
        if is_correct:
            correct += 1
    return correct / len(examples)


def eval_routing(
    source_model, source_tokenizer, target_model, target_tokenizer, examples, device, records: list[dict[str, Any]] | None = None
):
    correct = 0
    for idx, ex in enumerate(examples):
        prompt = _mcq_prompt(ex.question)
        src_scores = [
            score_choice_loglik(source_model, source_tokenizer, prompt, c, device)
            for c in ex.choices
        ]
        tgt_scores = [
            score_choice_loglik(target_model, target_tokenizer, prompt, c, device)
            for c in ex.choices
        ]
        src_sorted = sorted(src_scores, reverse=True)
        tgt_sorted = sorted(tgt_scores, reverse=True)
        src_margin = src_sorted[0] - src_sorted[1] if len(src_sorted) > 1 else src_sorted[0]
        tgt_margin = tgt_sorted[0] - tgt_sorted[1] if len(tgt_sorted) > 1 else tgt_sorted[0]
        pred = int(torch.tensor(src_scores if src_margin > tgt_margin else tgt_scores).argmax())
        is_correct = pred == ex.answer
        _append_prediction_record(
            records,
            index=idx,
            example_id=_mcq_example_id(ex),
            method="routing",
            prediction=pred,
            answer=ex.answer,
            correct=is_correct,
            extra={"src_margin": float(src_margin), "target_margin": float(tgt_margin)},
        )
        if is_correct:
            correct += 1
    return correct / len(examples)


def eval_text_to_text(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    examples,
    device,
    source_reasoning_mode: str = "brief_analysis",
    records: list[dict[str, Any]] | None = None,
):
    correct = 0
    for idx, ex in enumerate(examples):
        hint = _generate_source_hint(
            source_model,
            source_tokenizer,
            _mcq_prompt(ex.question),
            device,
            source_reasoning_mode,
        )
        augmented = f"Analysis: {hint.strip()}\nQuestion: {ex.question}\nAnswer:"
        scores = [
            score_choice_loglik(target_model, target_tokenizer, augmented, c, device)
            for c in ex.choices
        ]
        pred = int(torch.tensor(scores).argmax())
        is_correct = pred == ex.answer
        _append_prediction_record(
            records,
            index=idx,
            example_id=_mcq_example_id(ex),
            method="text_to_text",
            prediction=pred,
            answer=ex.answer,
            correct=is_correct,
            extra={"hint": hint},
        )
        if is_correct:
            correct += 1
    return correct / len(examples)


def eval_rotalign_kv(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    translator,
    examples,
    device,
    quantize: bool = True,
    protocol: str = "fused",
    source_reasoning_mode: str = "brief_analysis",
    source_kv_control: str = "real",
    quantization_control: str = "real",
    translated_kv_control: str = "real",
    fusion_rule: str = "static",
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
    position_selection_metric: str = "energy",
    fixed_position_profiles: list[torch.Tensor] | None = None,
    runtime_head_selection_ratio: float = 1.0,
    runtime_head_selection_metric: str = "attention_peak",
    fixed_head_profiles: list[torch.Tensor] | None = None,
    fixed_head_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_template_bank: list[torch.Tensor] | None = None,
    runtime_head_prior_alpha: float = 0.5,
    runtime_head_gate_metric: str = "none",
    runtime_head_gate_strength: float = 1.0,
    per_head_position_budget_mode: str = "none",
    drop_target_layers: set[int] | None = None,
    drop_target_layer_mode: str = "none",
    records: list[dict[str, Any]] | None = None,
    method_name: str = "rotalign_kv",
) -> float:
    translator = translator.to(device).eval()
    correct = 0
    for idx, ex in enumerate(examples):
        source_prompt = _source_reasoning_prompt(_mcq_prompt(ex.question), source_reasoning_mode)
        target_prompt = _mcq_prompt(ex.question)
        build_kwargs: dict[str, Any] = {}
        if fixed_position_profiles is not None:
            build_kwargs["fixed_position_profiles"] = fixed_position_profiles
        if fixed_head_profiles is not None:
            build_kwargs["fixed_head_profiles"] = fixed_head_profiles
        if fixed_head_templates is not None:
            build_kwargs["fixed_head_templates"] = fixed_head_templates
        if fixed_head_qk_templates is not None:
            build_kwargs["fixed_head_qk_templates"] = fixed_head_qk_templates
        if fixed_head_qk_template_bank is not None:
            build_kwargs["fixed_head_qk_template_bank"] = fixed_head_qk_template_bank
        build_kwargs["runtime_head_selection_ratio"] = runtime_head_selection_ratio
        build_kwargs["runtime_head_selection_metric"] = runtime_head_selection_metric
        build_kwargs["runtime_head_prior_alpha"] = runtime_head_prior_alpha
        build_kwargs["runtime_head_gate_metric"] = runtime_head_gate_metric
        build_kwargs["runtime_head_gate_strength"] = runtime_head_gate_strength
        build_kwargs["per_head_position_budget_mode"] = per_head_position_budget_mode
        build_kwargs["drop_target_layers"] = drop_target_layers
        build_kwargs["drop_target_layer_mode"] = drop_target_layer_mode
        prefix_state, stats = _build_rotalign_prefix_state(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            translator,
            source_prompt,
            target_prompt,
            device,
            quantize,
            protocol,
            source_kv_control,
            quantization_control,
            translated_kv_control,
            fusion_rule,
            kv_transport,
            position_selection_ratio,
            position_selection_metric,
            **build_kwargs,
        )
        scores = [
            _score_mcq_with_prefix_state(target_model, target_tokenizer, prefix_state, c, device)
            for c in ex.choices
        ]
        pred = int(torch.tensor(scores).argmax())
        is_correct = pred == ex.answer
        _append_prediction_record(
            records,
            index=idx,
            example_id=_mcq_example_id(ex),
            method=method_name,
            prediction=pred,
            answer=ex.answer,
            correct=is_correct,
            extra={
                "protocol": protocol,
                "source_kv_control": source_kv_control,
                "quantization_control": quantization_control,
                "translated_kv_control": translated_kv_control,
                "fusion_rule": fusion_rule,
                "kv_transport": kv_transport,
                "position_selection_ratio": position_selection_ratio,
                "position_selection_metric": position_selection_metric,
                "runtime_head_selection_ratio": runtime_head_selection_ratio,
                "runtime_head_selection_metric": runtime_head_selection_metric,
                "runtime_head_prior_alpha": runtime_head_prior_alpha,
                "runtime_head_gate_metric": runtime_head_gate_metric,
                "runtime_head_gate_strength": runtime_head_gate_strength,
                "per_head_position_budget_mode": per_head_position_budget_mode,
                "drop_target_layers": stats.get("dropped_target_layers"),
                "drop_target_layer_mode": stats.get("drop_target_layer_mode"),
                "bits": stats.get("bits"),
                "bytes": stats.get("bytes", float(stats.get("bits", 0.0)) / 8.0),
                "communication_bits": stats.get("bits"),
                "communication_bytes": stats.get("bytes", float(stats.get("bits", 0.0)) / 8.0),
                "payload_bits": stats.get("payload_bits"),
                "selector_bits": stats.get("selector_bits"),
                "metadata_bits": stats.get("metadata_bits"),
                "selected_target_layers": stats.get("selected_target_layers"),
                "selector_trace": stats.get("selector_trace"),
                "head_trace": stats.get("head_trace"),
                "head_gate_trace": stats.get("head_gate_trace"),
                "head_budget_trace": stats.get("head_budget_trace"),
            },
        )
        if is_correct:
            correct += 1
    return correct / len(examples)


def _eval_generation_target_alone(model, tokenizer, examples, device, max_new_tokens: int) -> float:
    return _eval_generation_target_alone_with_stats(
        model,
        tokenizer,
        examples,
        device,
        max_new_tokens,
    )["accuracy"]


def _eval_generation_target_alone_with_stats(
    model,
    tokenizer,
    examples,
    device,
    max_new_tokens: int,
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
    records: list[dict[str, Any]] | None = None,
    method_name: str = "target_alone",
) -> dict[str, float]:
    correct = 0
    total_tokens = 0
    total_ttft = 0.0
    total_elapsed = 0.0
    for idx, ex in enumerate(examples):
        prep_started = time.perf_counter()
        prefix_state = _prepare_prefix_state(
            model,
            tokenizer,
            ex.prompt,
            device,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )
        prep_elapsed = time.perf_counter() - prep_started
        trace = _greedy_generate_with_stats(
            model,
            tokenizer,
            prefix_state,
            device,
            max_new_tokens,
        )
        is_correct = _generation_match(trace.text, ex.answers)
        correct += int(is_correct)
        _append_prediction_record(
            records,
            index=idx,
            example_id=_generation_example_id(ex),
            method=method_name,
            prediction=trace.text,
            answer=ex.answers,
            correct=is_correct,
            extra={"generated_tokens": trace.num_generated_tokens},
        )
        total_tokens += trace.num_generated_tokens
        total_ttft += prep_elapsed + trace.ttft_sec
        total_elapsed += prep_elapsed + trace.elapsed_sec
    return _generation_metrics(
        correct=correct,
        num_examples=len(examples),
        total_generated_tokens=total_tokens,
        total_ttft_sec=total_ttft,
        total_elapsed_sec=total_elapsed,
    )


def _eval_generation_text_to_text(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    examples,
    device,
    max_new_tokens: int,
    source_reasoning_mode: str = "brief_analysis",
) -> float:
    return _eval_generation_text_to_text_with_stats(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        examples,
        device,
        max_new_tokens,
        source_reasoning_mode,
    )["accuracy"]


def _eval_generation_text_to_text_with_stats(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    examples,
    device,
    max_new_tokens: int,
    source_reasoning_mode: str = "brief_analysis",
    source_use_chat_template: bool = False,
    target_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_enable_thinking: bool | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    correct = 0
    total_tokens = 0
    total_ttft = 0.0
    total_elapsed = 0.0
    for idx, ex in enumerate(examples):
        example_started = time.perf_counter()
        hint = _generate_source_hint(
            source_model,
            source_tokenizer,
            ex.prompt,
            device,
            source_reasoning_mode,
            use_chat_template=source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        augmented = f"Analysis: {hint}\n{ex.prompt}"
        prefix_state = _prepare_prefix_state(
            target_model,
            target_tokenizer,
            augmented,
            device,
            use_chat_template=target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        trace = _greedy_generate_with_stats(
            target_model,
            target_tokenizer,
            prefix_state,
            device,
            max_new_tokens,
        )
        example_elapsed = time.perf_counter() - example_started
        is_correct = _generation_match(trace.text, ex.answers)
        correct += int(is_correct)
        _append_prediction_record(
            records,
            index=idx,
            example_id=_generation_example_id(ex),
            method="text_to_text",
            prediction=trace.text,
            answer=ex.answers,
            correct=is_correct,
            extra={"hint": hint, "generated_tokens": trace.num_generated_tokens},
        )
        total_tokens += trace.num_generated_tokens
        total_ttft += example_elapsed - trace.elapsed_sec + trace.ttft_sec
        total_elapsed += example_elapsed
    return _generation_metrics(
        correct=correct,
        num_examples=len(examples),
        total_generated_tokens=total_tokens,
        total_ttft_sec=total_ttft,
        total_elapsed_sec=total_elapsed,
    )


def _eval_generation_rotalign(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    translator,
    examples,
    device,
    max_new_tokens: int,
    quantize: bool,
    protocol: str,
    source_reasoning_mode: str = "brief_analysis",
    source_kv_control: str = "real",
    quantization_control: str = "real",
    translated_kv_control: str = "real",
    fusion_rule: str = "static",
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
    position_selection_metric: str = "energy",
    fixed_position_profiles: list[torch.Tensor] | None = None,
    runtime_head_selection_ratio: float = 1.0,
    runtime_head_selection_metric: str = "attention_peak",
    fixed_head_profiles: list[torch.Tensor] | None = None,
    fixed_head_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_template_bank: list[torch.Tensor] | None = None,
    runtime_head_prior_alpha: float = 0.5,
    runtime_head_gate_metric: str = "none",
    runtime_head_gate_strength: float = 1.0,
    per_head_position_budget_mode: str = "none",
    drop_target_layers: set[int] | None = None,
    drop_target_layer_mode: str = "none",
) -> tuple[float, float, float]:
    eval_kwargs: dict[str, Any] = {}
    if fixed_position_profiles is not None:
        eval_kwargs["fixed_position_profiles"] = fixed_position_profiles
    if fixed_head_profiles is not None:
        eval_kwargs["fixed_head_profiles"] = fixed_head_profiles
    if fixed_head_templates is not None:
        eval_kwargs["fixed_head_templates"] = fixed_head_templates
    if fixed_head_qk_templates is not None:
        eval_kwargs["fixed_head_qk_templates"] = fixed_head_qk_templates
    if fixed_head_qk_template_bank is not None:
        eval_kwargs["fixed_head_qk_template_bank"] = fixed_head_qk_template_bank
    eval_kwargs["runtime_head_prior_alpha"] = runtime_head_prior_alpha
    eval_kwargs["runtime_head_gate_metric"] = runtime_head_gate_metric
    eval_kwargs["runtime_head_gate_strength"] = runtime_head_gate_strength
    eval_kwargs["per_head_position_budget_mode"] = per_head_position_budget_mode
    eval_kwargs["drop_target_layers"] = drop_target_layers
    eval_kwargs["drop_target_layer_mode"] = drop_target_layer_mode
    stats = _eval_generation_rotalign_with_stats(
        source_model,
        source_tokenizer,
        target_model,
        target_tokenizer,
        translator,
        examples,
        device,
        max_new_tokens,
        quantize,
        protocol,
        source_reasoning_mode,
        source_kv_control,
        quantization_control,
        translated_kv_control,
        fusion_rule=fusion_rule,
        kv_transport=kv_transport,
        position_selection_ratio=position_selection_ratio,
        position_selection_metric=position_selection_metric,
        runtime_head_selection_ratio=runtime_head_selection_ratio,
        runtime_head_selection_metric=runtime_head_selection_metric,
        **eval_kwargs,
    )
    return stats["accuracy"], stats["bits"], stats["latency_sec"]


def _eval_generation_rotalign_with_stats(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    translator,
    examples,
    device,
    max_new_tokens: int,
    quantize: bool,
    protocol: str,
    source_reasoning_mode: str = "brief_analysis",
    source_kv_control: str = "real",
    quantization_control: str = "real",
    translated_kv_control: str = "real",
    fusion_rule: str = "static",
    kv_transport: str = "both",
    position_selection_ratio: float = 1.0,
    position_selection_metric: str = "energy",
    fixed_position_profiles: list[torch.Tensor] | None = None,
    runtime_head_selection_ratio: float = 1.0,
    runtime_head_selection_metric: str = "attention_peak",
    fixed_head_profiles: list[torch.Tensor] | None = None,
    fixed_head_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_templates: list[torch.Tensor] | None = None,
    fixed_head_qk_template_bank: list[torch.Tensor] | None = None,
    runtime_head_prior_alpha: float = 0.5,
    runtime_head_gate_metric: str = "none",
    runtime_head_gate_strength: float = 1.0,
    per_head_position_budget_mode: str = "none",
    drop_target_layers: set[int] | None = None,
    drop_target_layer_mode: str = "none",
    source_use_chat_template: bool = False,
    target_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_enable_thinking: bool | None = None,
    records: list[dict[str, Any]] | None = None,
    method_name: str = "rotalign_kv",
) -> dict[str, float]:
    translator = translator.to(device).eval()
    correct = 0
    total_bits = 0.0
    total_tokens = 0
    total_ttft = 0.0
    total_elapsed = 0.0
    for idx, ex in enumerate(examples):
        example_started = time.perf_counter()
        source_prompt = _source_reasoning_prompt(ex.prompt, source_reasoning_mode)
        target_prompt = ex.prompt
        if protocol == "text_kv_hybrid":
            hint = _generate_source_hint(
                source_model,
                source_tokenizer,
                ex.prompt,
                device,
                source_reasoning_mode,
                use_chat_template=source_use_chat_template,
                enable_thinking=source_enable_thinking,
            )
            target_prompt = f"Analysis: {hint}\n{ex.prompt}"

        build_kwargs: dict[str, Any] = {}
        if fixed_position_profiles is not None:
            build_kwargs["fixed_position_profiles"] = fixed_position_profiles
        if fixed_head_profiles is not None:
            build_kwargs["fixed_head_profiles"] = fixed_head_profiles
        if fixed_head_templates is not None:
            build_kwargs["fixed_head_templates"] = fixed_head_templates
        if fixed_head_qk_templates is not None:
            build_kwargs["fixed_head_qk_templates"] = fixed_head_qk_templates
        if fixed_head_qk_template_bank is not None:
            build_kwargs["fixed_head_qk_template_bank"] = fixed_head_qk_template_bank
        build_kwargs["runtime_head_selection_ratio"] = runtime_head_selection_ratio
        build_kwargs["runtime_head_selection_metric"] = runtime_head_selection_metric
        build_kwargs["runtime_head_prior_alpha"] = runtime_head_prior_alpha
        build_kwargs["runtime_head_gate_metric"] = runtime_head_gate_metric
        build_kwargs["runtime_head_gate_strength"] = runtime_head_gate_strength
        build_kwargs["per_head_position_budget_mode"] = per_head_position_budget_mode
        build_kwargs["drop_target_layers"] = drop_target_layers
        build_kwargs["drop_target_layer_mode"] = drop_target_layer_mode
        build_kwargs["source_use_chat_template"] = source_use_chat_template
        build_kwargs["target_use_chat_template"] = target_use_chat_template
        build_kwargs["source_enable_thinking"] = source_enable_thinking
        build_kwargs["target_enable_thinking"] = target_enable_thinking
        prefix_state, stats = _build_rotalign_prefix_state(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            translator,
            source_prompt,
            target_prompt,
            device,
            quantize,
            protocol,
            source_kv_control,
            quantization_control,
            translated_kv_control,
            fusion_rule,
            kv_transport,
            position_selection_ratio,
            position_selection_metric,
            **build_kwargs,
        )
        trace = _greedy_generate_with_stats(
            target_model,
            target_tokenizer,
            prefix_state,
            device,
            max_new_tokens,
        )
        example_elapsed = time.perf_counter() - example_started
        is_correct = _generation_match(trace.text, ex.answers)
        correct += int(is_correct)
        _append_prediction_record(
            records,
            index=idx,
            example_id=_generation_example_id(ex),
            method=method_name,
            prediction=trace.text,
            answer=ex.answers,
            correct=is_correct,
            extra={
                "protocol": protocol,
                "source_kv_control": source_kv_control,
                "quantization_control": quantization_control,
                "translated_kv_control": translated_kv_control,
                    "fusion_rule": fusion_rule,
                    "kv_transport": kv_transport,
                    "position_selection_ratio": position_selection_ratio,
                    "position_selection_metric": position_selection_metric,
                    "runtime_head_selection_ratio": runtime_head_selection_ratio,
                    "runtime_head_selection_metric": runtime_head_selection_metric,
                    "runtime_head_prior_alpha": runtime_head_prior_alpha,
                    "runtime_head_gate_metric": runtime_head_gate_metric,
                    "runtime_head_gate_strength": runtime_head_gate_strength,
                    "per_head_position_budget_mode": per_head_position_budget_mode,
                    "drop_target_layers": stats.get("dropped_target_layers"),
                    "drop_target_layer_mode": stats.get("drop_target_layer_mode"),
                    "bits": stats["bits"],
                    "bytes": stats.get("bytes", float(stats.get("bits", 0.0)) / 8.0),
                    "payload_bits": stats.get("payload_bits"),
                    "selector_bits": stats.get("selector_bits"),
                    "metadata_bits": stats.get("metadata_bits"),
                    "generated_tokens": trace.num_generated_tokens,
                    "selected_target_layers": stats.get("selected_target_layers"),
                    "selector_trace": stats.get("selector_trace"),
                    "head_trace": stats.get("head_trace"),
                    "head_gate_trace": stats.get("head_gate_trace"),
                    "head_budget_trace": stats.get("head_budget_trace"),
                },
        )
        total_bits += stats["bits"]
        total_tokens += trace.num_generated_tokens
        total_ttft += example_elapsed - trace.elapsed_sec + trace.ttft_sec
        total_elapsed += example_elapsed
    metrics = _generation_metrics(
        correct=correct,
        num_examples=len(examples),
        total_generated_tokens=total_tokens,
        total_ttft_sec=total_ttft,
        total_elapsed_sec=total_elapsed,
    )
    metrics["bits"] = total_bits / max(len(examples), 1)
    metrics["bytes"] = metrics["bits"] / 8.0
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--translator", required=True)
    p.add_argument("--source-model", required=True)
    p.add_argument("--target-model", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--task-type", choices=["auto", "mcq", "generation"], default="auto")
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="brief_analysis",
        help="Prompt template used when the source model generates a hint and when source KVs are captured",
    )
    p.add_argument(
        "--source-use-chat-template",
        action="store_true",
        help="Format source prompts through tokenizer.apply_chat_template when available.",
    )
    p.add_argument(
        "--target-use-chat-template",
        action="store_true",
        help="Format target prompts through tokenizer.apply_chat_template when available.",
    )
    p.add_argument(
        "--source-enable-thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help="Thinking flag passed into the source tokenizer chat template when supported.",
    )
    p.add_argument(
        "--target-enable-thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help="Thinking flag passed into the target tokenizer chat template when supported.",
    )
    p.add_argument("--no-quantize", action="store_true",
                   help="Disable Lloyd-Max quantization during translation (ablation)")
    p.add_argument(
        "--source-kv-control",
        choices=["real", "zero", "random", "shuffle_positions"],
        default="real",
        help="Negative-control perturbation applied to source KVs before translation",
    )
    p.add_argument(
        "--quantization-control",
        choices=["real", "matched_noise"],
        default="real",
        help="Use real quantize/dequantize or inject matched Gaussian noise instead.",
    )
    p.add_argument(
        "--translated-kv-control",
        choices=["real", "zero", "random", "shuffle_positions"],
        default="real",
        help="Negative-control perturbation applied after translation into target KV space.",
    )
    p.add_argument(
        "--fusion-rule",
        choices=[
            "static",
            "cosine",
            "cosine_shifted",
            "js_shrinkage",
            "kalman",
            "learned_affine",
            "learned_head_ridge",
            "cosine_tokenwise",
            "cosine_shifted_tokenwise",
            "js_shrinkage_tokenwise",
            "kalman_tokenwise",
        ],
        default="static",
        help="Static scalar gates or cosine-based runtime attenuation of translated KV.",
    )
    p.add_argument(
        "--kv-transport",
        choices=["both", "k_only", "v_only"],
        default="both",
        help="Transmit both translated tensors or ablate to keys-only / values-only transport.",
    )
    p.add_argument(
        "--position-selection-ratio",
        type=float,
        default=1.0,
        help="Fraction of prefix positions to keep from translated KV before transport/fusion.",
    )
    p.add_argument(
        "--position-selection-metric",
        choices=[
            "energy",
            "disagreement",
            "random",
            "recency",
            "attention",
            "attention_disagreement",
            "attention_shuffled",
            "source_attention",
            "attention_prior",
        ],
        default="energy",
        help="How to rank translated positions when position_selection_ratio < 1.0.",
    )
    p.add_argument(
        "--position-selection-prior-file",
        default=None,
        help="Optional calibration prompt file used to build a fixed query-blind attention prior.",
    )
    p.add_argument(
        "--position-selection-prior-source",
        choices=["calibration_mean_attention", "uniform"],
        default="calibration_mean_attention",
        help="How to build fixed query-blind attention priors when position_selection_metric=attention_prior.",
    )
    p.add_argument(
        "--position-selection-prior-bins",
        type=int,
        default=128,
        help="Number of normalized position bins used for fixed attention priors.",
    )
    p.add_argument(
        "--runtime-head-selection-ratio",
        type=float,
        default=1.0,
        help="Fraction of statically selected target heads to keep at evaluation time.",
    )
    p.add_argument(
        "--runtime-head-selection-metric",
        choices=[
            "attention_peak",
            "attention_entropy",
            "attention_margin",
            "retrieval_peak",
            "random",
            "attention_fidelity",
            "attention_fidelity_shuffled",
            "attention_procrustes",
            "attention_procrustes_shuffled",
            "attention_qk_fidelity",
            "attention_qk_fidelity_shuffled",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_expected",
            "attention_expected_shuffled",
            "attention_prior",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
            "attention_match",
            "attention_match_shuffled",
            "attention_blend",
        ],
        default="attention_peak",
        help="How to rank heads when runtime_head_selection_ratio < 1.0.",
    )
    p.add_argument(
        "--runtime-head-gate-metric",
        choices=[
            "none",
            "attention_peak",
            "attention_entropy",
            "attention_margin",
            "retrieval_peak",
            "random",
            "attention_fidelity",
            "attention_fidelity_shuffled",
            "attention_procrustes",
            "attention_procrustes_shuffled",
            "attention_qk_fidelity",
            "attention_qk_fidelity_shuffled",
            "attention_qk_fidelity_tokenwise",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_expected",
            "attention_expected_shuffled",
            "attention_prior",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
            "attention_match",
            "attention_match_shuffled",
            "attention_blend",
        ],
        default="none",
        help="Use a runtime per-head soft gate on top of the existing fusion gate instead of only hard head selection.",
    )
    p.add_argument(
        "--runtime-head-gate-strength",
        type=float,
        default=1.0,
        help="Blend strength in [0,1] for runtime per-head gate overrides; 0 keeps the scalar gate unchanged.",
    )
    p.add_argument(
        "--runtime-head-prior-file",
        default=None,
        help="Optional calibration prompt file used to build a fixed head prior.",
    )
    p.add_argument(
        "--runtime-head-prior-load",
        default=None,
        help="Optional saved fixed head-profile bundle to load instead of rebuilding from prompts.",
    )
    p.add_argument(
        "--runtime-head-prior-save",
        default=None,
        help="Optional path to save the built fixed head-profile bundle for reuse across runs.",
    )
    p.add_argument(
        "--runtime-head-prior-metric",
        choices=["attention_peak", "attention_entropy", "attention_margin", "retrieval_peak"],
        default="attention_peak",
        help="How to score heads when building a fixed head prior from calibration prompts.",
    )
    p.add_argument(
        "--runtime-head-prior-alpha",
        type=float,
        default=0.5,
        help="Blend weight for the fixed head prior when runtime_head_selection_metric=attention_blend.",
    )
    p.add_argument(
        "--runtime-head-prior-shrinkage",
        type=float,
        default=0.0,
        help="Optional shrinkage strength in [0,1] applied to fixed head priors before use.",
    )
    p.add_argument(
        "--runtime-head-prior-shrink-target",
        choices=["uniform", "global"],
        default="uniform",
        help="Shrink fixed head priors toward a uniform or global cross-layer prior.",
    )
    p.add_argument(
        "--per-head-position-budget-mode",
        choices=[
            "none",
            "attention_peak",
            "attention_entropy",
            "attention_margin",
            "retrieval_peak",
            "random",
            "attention_fidelity",
            "attention_fidelity_shuffled",
            "attention_procrustes",
            "attention_procrustes_shuffled",
            "attention_qk_fidelity",
            "attention_qk_fidelity_shuffled",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_expected",
            "attention_expected_shuffled",
            "attention_prior",
            "attention_prior_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
            "attention_match",
            "attention_match_shuffled",
            "attention_blend",
        ],
        default="none",
        help="Allocate the position budget unevenly across active heads instead of using one flat per-layer position ratio.",
    )
    p.add_argument(
        "--drop-target-layers",
        default=None,
        help="Comma-separated target layer indices whose translated KV should be ablated for layer-knockout diagnostics.",
    )
    p.add_argument(
        "--drop-target-layer-mode",
        choices=["none", "target", "zero"],
        default="none",
        help="Layer knockout mode: target replaces translated KV with target KV; zero replaces translated KV with zeros.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["target", "t2t", "rotalign"],
        choices=[
            "target",
            "source",
            "routing",
            "t2t",
            "rotalign",
            "rotalign_translated",
            "rotalign_fused",
            "rotalign_text_kv",
        ],
    )
    p.add_argument(
        "--gate-mode",
        choices=["checkpoint", "fixed", "search", "sweep"],
        default="checkpoint",
    )
    p.add_argument("--fixed-gate", type=float, default=0.5)
    p.add_argument("--gate-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument(
        "--gate-search-file",
        default=None,
        help="Held-out JSONL file used to line-search K/V gates before final evaluation",
    )
    p.add_argument(
        "--gate-search-limit",
        type=int,
        default=16,
        help="Maximum number of held-out examples to use for gate search",
    )
    p.add_argument(
        "--prediction-output",
        default=None,
        help="Optional JSONL path for per-example predictions and correctness.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fusion_rule = getattr(args, "fusion_rule", "static")
    kv_transport = getattr(args, "kv_transport", "both")
    position_selection_ratio = float(getattr(args, "position_selection_ratio", 1.0))
    position_selection_metric = getattr(args, "position_selection_metric", "energy")
    runtime_head_selection_ratio = float(getattr(args, "runtime_head_selection_ratio", 1.0))
    runtime_head_selection_metric = getattr(args, "runtime_head_selection_metric", "attention_peak")
    runtime_head_prior_alpha = float(getattr(args, "runtime_head_prior_alpha", 0.5))
    runtime_head_gate_metric = getattr(args, "runtime_head_gate_metric", "none")
    runtime_head_gate_strength = float(getattr(args, "runtime_head_gate_strength", 1.0))
    runtime_head_prior_file = getattr(args, "runtime_head_prior_file", None)
    runtime_head_prior_load = getattr(args, "runtime_head_prior_load", None)
    runtime_head_prior_save = getattr(args, "runtime_head_prior_save", None)
    runtime_head_prior_shrinkage = float(getattr(args, "runtime_head_prior_shrinkage", 0.0))
    runtime_head_prior_shrink_target = getattr(args, "runtime_head_prior_shrink_target", "uniform")
    per_head_position_budget_mode = getattr(args, "per_head_position_budget_mode", "none")
    drop_target_layers = _parse_target_layer_set(getattr(args, "drop_target_layers", None))
    drop_target_layer_mode = getattr(args, "drop_target_layer_mode", "none")
    source_use_chat_template = bool(getattr(args, "source_use_chat_template", False))
    target_use_chat_template = bool(getattr(args, "target_use_chat_template", False))
    source_enable_thinking = _optional_bool_from_arg(getattr(args, "source_enable_thinking", "auto"))
    target_enable_thinking = _optional_bool_from_arg(getattr(args, "target_enable_thinking", "auto"))
    if not 0.0 <= runtime_head_prior_shrinkage <= 1.0:
        raise ValueError("--runtime-head-prior-shrinkage must be in [0, 1]")
    if not 0.0 <= runtime_head_gate_strength <= 1.0:
        raise ValueError("--runtime-head-gate-strength must be in [0, 1]")
    if drop_target_layers and drop_target_layer_mode == "none":
        drop_target_layer_mode = "target"

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    task_type = infer_task_type(args.eval_file) if args.task_type == "auto" else args.task_type
    if task_type == "mcq":
        examples = load_mcq(args.eval_file)
    else:
        examples = load_generation(args.eval_file)
    print(f"Loaded {len(examples)} {task_type} examples from {args.eval_file}")

    attention_selector_metrics = {
        "attention",
        "attention_disagreement",
        "attention_shuffled",
        "source_attention",
        "attention_prior",
    }
    runtime_head_attention_metrics = {
        "attention_peak",
        "attention_entropy",
        "attention_margin",
        "retrieval_peak",
        "attention_fidelity",
        "attention_fidelity_shuffled",
        "attention_procrustes",
        "attention_procrustes_shuffled",
        "attention_qk_fidelity",
        "attention_qk_fidelity_shuffled",
        "attention_qk_template_transport",
        "attention_qk_template_transport_shuffled",
        "attention_qk_bank_transport",
        "attention_qk_bank_transport_shuffled",
        "attention_expected",
        "attention_expected_shuffled",
        "attention_prior",
        "attention_template_transport",
        "attention_template_transport_shuffled",
        "attention_sinkhorn",
        "attention_sinkhorn_shuffled",
        "attention_match",
        "attention_match_shuffled",
        "attention_blend",
    }

    print(f"\nLoading source: {args.source_model}")
    tok_s = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tok_s.pad_token_id is None:
        tok_s.pad_token = tok_s.eos_token
    source_model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if position_selection_metric == "source_attention":
        source_model_kwargs["attn_implementation"] = "eager"
    src = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        **source_model_kwargs,
    ).to(args.device).eval()

    print(f"Loading target: {args.target_model}")
    tok_t = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tok_t.pad_token_id is None:
        tok_t.pad_token = tok_t.eos_token
    target_model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if (
        position_selection_metric in attention_selector_metrics
        or (
            runtime_head_selection_ratio < 1.0
            and runtime_head_selection_metric in runtime_head_attention_metrics
        )
        or runtime_head_gate_metric in runtime_head_attention_metrics
        or per_head_position_budget_mode != "none"
    ):
        target_model_kwargs["attn_implementation"] = "eager"
    tgt = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        **target_model_kwargs,
    ).to(args.device).eval()

    print(f"Loading translator from {args.translator}")
    translator = RotAlignKVTranslator.load(args.translator, map_location=args.device)
    initial_gate_values = translator.gate_values()
    fixed_position_profiles = None
    fixed_head_profiles = None
    fixed_head_templates = None
    fixed_head_qk_templates = None
    fixed_head_qk_template_bank = None
    loaded_head_prior_metadata: dict[str, Any] | None = None
    needs_fixed_position_prior = (
        position_selection_metric == "attention_prior"
        or (
            runtime_head_selection_ratio < 1.0
            and runtime_head_selection_metric in {"attention_expected", "attention_expected_shuffled"}
        )
        or runtime_head_gate_metric in {"attention_expected", "attention_expected_shuffled"}
        or per_head_position_budget_mode in {"attention_expected", "attention_expected_shuffled"}
    )
    if needs_fixed_position_prior:
        if args.position_selection_prior_source == "uniform":
            print(
                "Building uniform fixed attention prior "
                f"(layers={translator.config.num_tgt_layers}, bins={args.position_selection_prior_bins})"
            )
            fixed_position_profiles = _uniform_attention_prior(
                layer_count=translator.config.num_tgt_layers,
                bins=args.position_selection_prior_bins,
            )
        else:
            if not args.position_selection_prior_file:
                raise ValueError(
                    "--position-selection-prior-file is required when a fixed attention prior "
                    "is needed and --position-selection-prior-source "
                    "calibration_mean_attention is selected"
                )
            prior_prompts = load_prompt_lines(args.position_selection_prior_file)
            print(
                f"Building fixed attention prior from {len(prior_prompts)} calibration prompts "
                f"({args.position_selection_prior_file})"
            )
            fixed_position_profiles = _mean_attention_prior_from_prompts(
                tgt,
                tok_t,
                prior_prompts,
                args.device,
                translator=translator,
                bins=args.position_selection_prior_bins,
            )
    needs_fixed_head_prior = (
        (
            runtime_head_selection_ratio < 1.0
            and runtime_head_selection_metric in {
                "attention_prior",
                "attention_qk_template_transport",
                "attention_qk_template_transport_shuffled",
                "attention_qk_bank_transport",
                "attention_qk_bank_transport_shuffled",
                "attention_template_transport",
                "attention_template_transport_shuffled",
                "attention_sinkhorn",
                "attention_sinkhorn_shuffled",
                "attention_match",
                "attention_match_shuffled",
                "attention_blend",
            }
        )
        or runtime_head_gate_metric in {
            "attention_prior",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
            "attention_match",
            "attention_match_shuffled",
            "attention_blend",
        }
        or per_head_position_budget_mode in {
            "attention_prior",
            "attention_prior_shuffled",
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
            "attention_match",
            "attention_match_shuffled",
            "attention_blend",
        }
    )
    needs_fixed_head_templates = (
        (
            runtime_head_selection_ratio < 1.0
            and runtime_head_selection_metric in {
                "attention_qk_template_transport",
                "attention_qk_template_transport_shuffled",
                "attention_template_transport",
                "attention_template_transport_shuffled",
                "attention_sinkhorn",
                "attention_sinkhorn_shuffled",
            }
        )
        or runtime_head_gate_metric in {
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
        }
        or per_head_position_budget_mode in {
            "attention_qk_template_transport",
            "attention_qk_template_transport_shuffled",
            "attention_template_transport",
            "attention_template_transport_shuffled",
            "attention_sinkhorn",
            "attention_sinkhorn_shuffled",
        }
    )
    needs_fixed_head_qk_template_bank = (
        (
            runtime_head_selection_ratio < 1.0
            and runtime_head_selection_metric in {
                "attention_qk_bank_transport",
                "attention_qk_bank_transport_shuffled",
            }
        )
        or runtime_head_gate_metric in {
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
        }
        or per_head_position_budget_mode in {
            "attention_qk_bank_transport",
            "attention_qk_bank_transport_shuffled",
        }
    )
    if needs_fixed_head_prior:
        if runtime_head_prior_load:
            print(f"Loading fixed head prior bundle from {runtime_head_prior_load}")
            fixed_head_profiles, loaded_head_prior_metadata = _load_head_profile_bundle(
                runtime_head_prior_load,
                target_layers=translator.config.num_tgt_layers,
            )
        else:
            if not runtime_head_prior_file:
                raise ValueError(
                    "--runtime-head-prior-file or --runtime-head-prior-load is required when "
                    "a fixed head prior is required"
                )
            head_prior_prompts = load_prompt_lines(runtime_head_prior_file)
            print(
                f"Building fixed head prior from {len(head_prior_prompts)} calibration prompts "
                f"({runtime_head_prior_file})"
            )
            fixed_head_profiles = _mean_head_prior_from_prompts(
                tgt,
                tok_t,
                head_prior_prompts,
                args.device,
                translator=translator,
                metric=args.runtime_head_prior_metric,
            )
        if runtime_head_prior_shrinkage > 0.0:
            original_head_profiles = [
                torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu().clone()
                for profile in fixed_head_profiles
            ]
            print(
                "Applying fixed head prior shrinkage "
                f"(strength={runtime_head_prior_shrinkage:.3f}, target={runtime_head_prior_shrink_target})"
            )
            fixed_head_profiles = _shrink_head_profiles(
                fixed_head_profiles,
                strength=runtime_head_prior_shrinkage,
                target=runtime_head_prior_shrink_target,
            )
            if loaded_head_prior_metadata is None:
                loaded_head_prior_metadata = {}
            loaded_head_prior_metadata["applied_shrinkage"] = float(runtime_head_prior_shrinkage)
            loaded_head_prior_metadata["applied_shrink_target"] = runtime_head_prior_shrink_target
            loaded_head_prior_metadata.update(
                {
                    f"{key}_before": value
                    for key, value in _head_profile_summary(original_head_profiles).items()
                }
            )
            loaded_head_prior_metadata.update(
                {
                    f"{key}_after": value
                    for key, value in _head_profile_summary(fixed_head_profiles).items()
                }
            )
            loaded_head_prior_metadata["profile_topk_overlap_jaccard"] = _head_profile_topk_overlap(
                original_head_profiles,
                fixed_head_profiles,
            )
        if runtime_head_prior_save:
            _save_head_profile_bundle(
                runtime_head_prior_save,
                fixed_head_profiles,
                metadata={
                    "source_model": args.source_model,
                    "target_model": args.target_model,
                    "translator": args.translator,
                    "metric": args.runtime_head_prior_metric,
                    "prompt_file": runtime_head_prior_file,
                    "layer_count": len(fixed_head_profiles),
                    "head_counts": [int(profile.numel()) for profile in fixed_head_profiles],
                    "shrinkage": float(runtime_head_prior_shrinkage),
                    "shrink_target": runtime_head_prior_shrink_target,
                },
            )
            print(f"Saved fixed head prior bundle to {runtime_head_prior_save}")
    elif runtime_head_prior_save:
        if not runtime_head_prior_file:
            raise ValueError(
                "--runtime-head-prior-file is required when saving a newly built fixed head prior"
            )
        head_prior_prompts = load_prompt_lines(runtime_head_prior_file)
        print(
            f"Building fixed head prior from {len(head_prior_prompts)} calibration prompts "
            f"({runtime_head_prior_file}) for bundle export"
        )
        fixed_head_profiles = _mean_head_prior_from_prompts(
            tgt,
            tok_t,
            head_prior_prompts,
            args.device,
            translator=translator,
            metric=args.runtime_head_prior_metric,
        )
        if runtime_head_prior_shrinkage > 0.0:
            original_head_profiles = [
                torch.as_tensor(profile, dtype=torch.float32).view(-1).cpu().clone()
                for profile in fixed_head_profiles
            ]
            print(
                "Applying fixed head prior shrinkage "
                f"(strength={runtime_head_prior_shrinkage:.3f}, target={runtime_head_prior_shrink_target})"
            )
            fixed_head_profiles = _shrink_head_profiles(
                fixed_head_profiles,
                strength=runtime_head_prior_shrinkage,
                target=runtime_head_prior_shrink_target,
            )
        _save_head_profile_bundle(
            runtime_head_prior_save,
            fixed_head_profiles,
            metadata={
                "source_model": args.source_model,
                "target_model": args.target_model,
                "translator": args.translator,
                "metric": args.runtime_head_prior_metric,
                "prompt_file": runtime_head_prior_file,
                "layer_count": len(fixed_head_profiles),
                "head_counts": [int(profile.numel()) for profile in fixed_head_profiles],
                "shrinkage": float(runtime_head_prior_shrinkage),
                "shrink_target": runtime_head_prior_shrink_target,
                **(
                    {
                        f"{key}_before": value
                        for key, value in _head_profile_summary(original_head_profiles).items()
                    }
                    if runtime_head_prior_shrinkage > 0.0
                    else {}
                ),
                **(
                    {
                        f"{key}_after": value
                        for key, value in _head_profile_summary(fixed_head_profiles).items()
                    }
                    if runtime_head_prior_shrinkage > 0.0
                    else {}
                ),
                **(
                    {
                        "profile_topk_overlap_jaccard": _head_profile_topk_overlap(
                            original_head_profiles,
                            fixed_head_profiles,
                        )
                    }
                    if runtime_head_prior_shrinkage > 0.0
                    else {}
                ),
            },
        )
        print(f"Saved fixed head prior bundle to {runtime_head_prior_save}")
    if needs_fixed_head_templates:
        if not runtime_head_prior_file:
            raise ValueError(
                "--runtime-head-prior-file is required when building fixed head attention templates"
            )
        head_template_prompts = load_prompt_lines(runtime_head_prior_file)
        print(
            f"Building fixed head attention templates from {len(head_template_prompts)} calibration prompts "
            f"({runtime_head_prior_file})"
        )
        fixed_head_templates = _mean_head_attention_templates_from_prompts(
            tgt,
            tok_t,
            head_template_prompts,
            args.device,
            translator=translator,
            bins=args.position_selection_prior_bins,
        )
    if needs_fixed_head_templates:
        if not runtime_head_prior_file:
            raise ValueError(
                "--runtime-head-prior-file is required when building fixed head QK templates"
            )
        head_qk_template_prompts = load_prompt_lines(runtime_head_prior_file)
        print(
            f"Building fixed head QK templates from {len(head_qk_template_prompts)} calibration prompts "
            f"({runtime_head_prior_file})"
        )
        fixed_head_qk_templates = _mean_head_qk_templates_from_prompts(
            tgt,
            tok_t,
            head_qk_template_prompts,
            args.device,
            translator=translator,
            bins=args.position_selection_prior_bins,
        )
    if needs_fixed_head_qk_template_bank:
        if not runtime_head_prior_file:
            raise ValueError(
                "--runtime-head-prior-file is required when building fixed head QK template banks"
            )
        head_qk_template_prompts = load_prompt_lines(runtime_head_prior_file)
        print(
            f"Building fixed head QK template bank from {len(head_qk_template_prompts)} calibration prompts "
            f"({runtime_head_prior_file})"
        )
        fixed_head_qk_template_bank = _head_qk_template_bank_from_prompts(
            tgt,
            tok_t,
            head_qk_template_prompts,
            args.device,
            translator=translator,
            bins=args.position_selection_prior_bins,
        )

    if args.gate_mode == "fixed":
        translator.set_fixed_gates(args.fixed_gate)
    search_examples = None
    search_task_type = None
    if args.gate_mode == "search":
        if not args.gate_search_file:
            raise ValueError("--gate-search-file is required when --gate-mode search is used")
        search_task_type = infer_task_type(args.gate_search_file)
        if search_task_type == "mcq":
            search_examples = load_mcq(args.gate_search_file)
        else:
            search_examples = load_generation(args.gate_search_file)
        search_examples = _limit_examples(search_examples, args.gate_search_limit)

    results: dict[str, float] = {}
    prediction_records: list[dict[str, Any]] | None = [] if args.prediction_output else None
    if task_type == "mcq":
        if "target" in args.methods:
            results["target_alone"] = eval_target_alone(tgt, tok_t, examples, args.device, prediction_records)
        if "source" in args.methods:
            results["source_alone"] = eval_source_alone(src, tok_s, examples, args.device, prediction_records)
        if "routing" in args.methods:
            results["routing"] = eval_routing(src, tok_s, tgt, tok_t, examples, args.device, prediction_records)
        if "t2t" in args.methods:
            results["text_to_text"] = eval_text_to_text(src, tok_s, tgt, tok_t, examples, args.device, records=prediction_records)

        rotalign_modes = []
        if "rotalign" in args.methods:
            rotalign_modes.append(("rotalign_kv", "fused"))
        if "rotalign_translated" in args.methods:
            rotalign_modes.append(("rotalign_translated_only", "translated_only"))
        if "rotalign_fused" in args.methods:
            rotalign_modes.append(("rotalign_fused", "fused"))
        if "rotalign_text_kv" in args.methods:
            rotalign_modes.append(("rotalign_text_kv_hybrid", "text_kv_hybrid"))

        gate_values = args.gate_values if args.gate_mode == "sweep" else [args.fixed_gate]
        if args.gate_mode in {"checkpoint", "search"}:
            gate_values = [None]
        for key, protocol in rotalign_modes:
            if args.gate_mode == "search":
                assert search_examples is not None
                _restore_gate_values(translator, initial_gate_values)
                print(
                    f"Searching per-layer gates on {len(search_examples)} held-out "
                    f"{search_task_type} examples from {args.gate_search_file} "
                    f"for protocol={protocol}"
                )
                search_kwargs: dict[str, Any] = {}
                if fixed_position_profiles is not None:
                    search_kwargs["fixed_position_profiles"] = fixed_position_profiles
                if fixed_head_profiles is not None:
                    search_kwargs["fixed_head_profiles"] = fixed_head_profiles
                if fixed_head_templates is not None:
                    search_kwargs["fixed_head_templates"] = fixed_head_templates
                if fixed_head_qk_templates is not None:
                    search_kwargs["fixed_head_qk_templates"] = fixed_head_qk_templates
                if fixed_head_qk_template_bank is not None:
                    search_kwargs["fixed_head_qk_template_bank"] = fixed_head_qk_template_bank
                search_stats = _search_per_layer_gates(
                    src,
                    tok_s,
                    tgt,
                    tok_t,
                    translator,
                    search_examples,
                    args.device,
                    quantize=not args.no_quantize,
                    protocol=protocol,
                    source_reasoning_mode=args.source_reasoning_mode,
                    gate_values=args.gate_values,
                    source_kv_control=args.source_kv_control,
                    quantization_control=args.quantization_control,
                    translated_kv_control=args.translated_kv_control,
                    fusion_rule=fusion_rule,
                    kv_transport=kv_transport,
                    position_selection_ratio=position_selection_ratio,
                    position_selection_metric=position_selection_metric,
                    runtime_head_selection_ratio=runtime_head_selection_ratio,
                    runtime_head_selection_metric=runtime_head_selection_metric,
                    runtime_head_prior_alpha=runtime_head_prior_alpha,
                    runtime_head_gate_metric=runtime_head_gate_metric,
                    runtime_head_gate_strength=runtime_head_gate_strength,
                    per_head_position_budget_mode=per_head_position_budget_mode,
                    **search_kwargs,
                )
                print(
                    "Selected gate means: "
                    f"K={sum(search_stats['gate_K']) / max(len(search_stats['gate_K']), 1):.3f}, "
                    f"V={sum(search_stats['gate_V']) / max(len(search_stats['gate_V']), 1):.3f}"
                )
            for gate in gate_values:
                if gate is not None:
                    translator.set_fixed_gates(gate)
                metric_key = key if gate is None else f"{key}_gate_{gate:.2f}"
                start = time.perf_counter()
                score = eval_rotalign_kv(
                    src,
                    tok_s,
                    tgt,
                    tok_t,
                    translator,
                    examples,
                    args.device,
                    quantize=not args.no_quantize,
                    protocol=protocol,
                    source_reasoning_mode=args.source_reasoning_mode,
                    source_kv_control=args.source_kv_control,
                    quantization_control=args.quantization_control,
                    translated_kv_control=args.translated_kv_control,
                    fusion_rule=fusion_rule,
                    kv_transport=kv_transport,
                    position_selection_ratio=position_selection_ratio,
                    position_selection_metric=position_selection_metric,
                    fixed_position_profiles=fixed_position_profiles,
                    runtime_head_selection_ratio=runtime_head_selection_ratio,
                    runtime_head_selection_metric=runtime_head_selection_metric,
                    fixed_head_profiles=fixed_head_profiles,
                    fixed_head_templates=fixed_head_templates,
                    fixed_head_qk_templates=fixed_head_qk_templates,
                    fixed_head_qk_template_bank=fixed_head_qk_template_bank,
                    runtime_head_prior_alpha=runtime_head_prior_alpha,
                    runtime_head_gate_metric=runtime_head_gate_metric,
                    runtime_head_gate_strength=runtime_head_gate_strength,
                    per_head_position_budget_mode=per_head_position_budget_mode,
                    drop_target_layers=drop_target_layers,
                    drop_target_layer_mode=drop_target_layer_mode,
                    records=prediction_records,
                    method_name=metric_key,
                )
                elapsed = time.perf_counter() - start
                results[metric_key] = score
                results[f"{metric_key}_latency_sec"] = elapsed / max(len(examples), 1)
                results[f"{metric_key}_bits"] = _communication_bits(
                    translator,
                    seq_len=1,
                    quantize=not args.no_quantize,
                    translated_kv_control=args.translated_kv_control,
                    kv_transport=kv_transport,
                    protocol=protocol,
                    position_selection_ratio=position_selection_ratio,
                    active_k_head_counts=None,
                    active_v_head_counts=None,
                )
                results[f"{metric_key}_bytes"] = results[f"{metric_key}_bits"] / 8.0
    else:
        if "target" in args.methods:
            target_stats = _eval_generation_target_alone_with_stats(
                tgt,
                tok_t,
                examples,
                args.device,
                args.max_new_tokens,
                use_chat_template=target_use_chat_template,
                enable_thinking=target_enable_thinking,
                records=prediction_records,
            )
            results["target_alone"] = target_stats["accuracy"]
            results["target_alone_ttft_sec"] = target_stats["ttft_sec"]
            results["target_alone_tokens_per_sec"] = target_stats["tokens_per_sec"]
            results["target_alone_examples_per_sec"] = target_stats["examples_per_sec"]
            results["target_alone_latency_sec"] = target_stats["latency_sec"]
        if "source" in args.methods:
            source_stats = _eval_generation_target_alone_with_stats(
                src,
                tok_s,
                examples,
                args.device,
                args.max_new_tokens,
                use_chat_template=source_use_chat_template,
                enable_thinking=source_enable_thinking,
                records=prediction_records,
                method_name="source_alone",
            )
            results["source_alone"] = source_stats["accuracy"]
            results["source_alone_ttft_sec"] = source_stats["ttft_sec"]
            results["source_alone_tokens_per_sec"] = source_stats["tokens_per_sec"]
            results["source_alone_examples_per_sec"] = source_stats["examples_per_sec"]
            results["source_alone_latency_sec"] = source_stats["latency_sec"]
        if "routing" in args.methods:
            raise ValueError("routing baseline is only implemented for MCQ tasks")
        if "t2t" in args.methods:
            t2t_stats = _eval_generation_text_to_text_with_stats(
                src,
                tok_s,
                tgt,
                tok_t,
                examples,
                args.device,
                args.max_new_tokens,
                source_reasoning_mode=args.source_reasoning_mode,
                source_use_chat_template=source_use_chat_template,
                target_use_chat_template=target_use_chat_template,
                source_enable_thinking=source_enable_thinking,
                target_enable_thinking=target_enable_thinking,
                records=prediction_records,
            )
            results["text_to_text"] = t2t_stats["accuracy"]
            results["text_to_text_ttft_sec"] = t2t_stats["ttft_sec"]
            results["text_to_text_tokens_per_sec"] = t2t_stats["tokens_per_sec"]
            results["text_to_text_examples_per_sec"] = t2t_stats["examples_per_sec"]
            results["text_to_text_latency_sec"] = t2t_stats["latency_sec"]

        rotalign_modes = []
        if "rotalign" in args.methods:
            rotalign_modes.append(("rotalign_kv", "fused"))
        if "rotalign_translated" in args.methods:
            rotalign_modes.append(("rotalign_translated_only", "translated_only"))
        if "rotalign_fused" in args.methods:
            rotalign_modes.append(("rotalign_fused", "fused"))
        if "rotalign_text_kv" in args.methods:
            rotalign_modes.append(("rotalign_text_kv_hybrid", "text_kv_hybrid"))

        gate_values = args.gate_values if args.gate_mode == "sweep" else [args.fixed_gate]
        if args.gate_mode in {"checkpoint", "search"}:
            gate_values = [None]
        for key, protocol in rotalign_modes:
            if args.gate_mode == "search":
                assert search_examples is not None
                _restore_gate_values(translator, initial_gate_values)
                print(
                    f"Searching per-layer gates on {len(search_examples)} held-out "
                    f"{search_task_type} examples from {args.gate_search_file} "
                    f"for protocol={protocol}"
                )
                search_kwargs: dict[str, Any] = {}
                if fixed_position_profiles is not None:
                    search_kwargs["fixed_position_profiles"] = fixed_position_profiles
                if fixed_head_profiles is not None:
                    search_kwargs["fixed_head_profiles"] = fixed_head_profiles
                if fixed_head_templates is not None:
                    search_kwargs["fixed_head_templates"] = fixed_head_templates
                if fixed_head_qk_templates is not None:
                    search_kwargs["fixed_head_qk_templates"] = fixed_head_qk_templates
                if fixed_head_qk_template_bank is not None:
                    search_kwargs["fixed_head_qk_template_bank"] = fixed_head_qk_template_bank
                search_stats = _search_per_layer_gates(
                    src,
                    tok_s,
                    tgt,
                    tok_t,
                    translator,
                    search_examples,
                    args.device,
                    quantize=not args.no_quantize,
                    protocol=protocol,
                    source_reasoning_mode=args.source_reasoning_mode,
                    gate_values=args.gate_values,
                    source_kv_control=args.source_kv_control,
                    quantization_control=args.quantization_control,
                    translated_kv_control=args.translated_kv_control,
                    fusion_rule=fusion_rule,
                    kv_transport=kv_transport,
                    position_selection_ratio=position_selection_ratio,
                    position_selection_metric=position_selection_metric,
                    runtime_head_selection_ratio=runtime_head_selection_ratio,
                    runtime_head_selection_metric=runtime_head_selection_metric,
                    runtime_head_prior_alpha=runtime_head_prior_alpha,
                    runtime_head_gate_metric=runtime_head_gate_metric,
                    runtime_head_gate_strength=runtime_head_gate_strength,
                    per_head_position_budget_mode=per_head_position_budget_mode,
                    **search_kwargs,
                )
                print(
                    "Selected gate means: "
                    f"K={sum(search_stats['gate_K']) / max(len(search_stats['gate_K']), 1):.3f}, "
                    f"V={sum(search_stats['gate_V']) / max(len(search_stats['gate_V']), 1):.3f}"
                )
            for gate in gate_values:
                if gate is not None:
                    translator.set_fixed_gates(gate)
                metric_key = key if gate is None else f"{key}_gate_{gate:.2f}"
                rotalign_stats = _eval_generation_rotalign_with_stats(
                    src,
                    tok_s,
                    tgt,
                    tok_t,
                    translator,
                    examples,
                    args.device,
                    args.max_new_tokens,
                    quantize=not args.no_quantize,
                    protocol=protocol,
                    source_reasoning_mode=args.source_reasoning_mode,
                    source_kv_control=args.source_kv_control,
                    quantization_control=args.quantization_control,
                    translated_kv_control=args.translated_kv_control,
                    fusion_rule=fusion_rule,
                    kv_transport=kv_transport,
                    position_selection_ratio=position_selection_ratio,
                    position_selection_metric=position_selection_metric,
                    fixed_position_profiles=fixed_position_profiles,
                    runtime_head_selection_ratio=runtime_head_selection_ratio,
                    runtime_head_selection_metric=runtime_head_selection_metric,
                    fixed_head_profiles=fixed_head_profiles,
                    fixed_head_templates=fixed_head_templates,
                    fixed_head_qk_templates=fixed_head_qk_templates,
                    fixed_head_qk_template_bank=fixed_head_qk_template_bank,
                    runtime_head_prior_alpha=runtime_head_prior_alpha,
                    runtime_head_gate_metric=runtime_head_gate_metric,
                    runtime_head_gate_strength=runtime_head_gate_strength,
                    per_head_position_budget_mode=per_head_position_budget_mode,
                    drop_target_layers=drop_target_layers,
                    drop_target_layer_mode=drop_target_layer_mode,
                    source_use_chat_template=source_use_chat_template,
                    target_use_chat_template=target_use_chat_template,
                    source_enable_thinking=source_enable_thinking,
                    target_enable_thinking=target_enable_thinking,
                    records=prediction_records,
                    method_name=metric_key,
                )
                results[metric_key] = rotalign_stats["accuracy"]
                results[f"{metric_key}_bits"] = rotalign_stats["bits"]
                results[f"{metric_key}_bytes"] = rotalign_stats["bytes"]
                results[f"{metric_key}_ttft_sec"] = rotalign_stats["ttft_sec"]
                results[f"{metric_key}_tokens_per_sec"] = rotalign_stats["tokens_per_sec"]
                results[f"{metric_key}_examples_per_sec"] = rotalign_stats["examples_per_sec"]
                results[f"{metric_key}_latency_sec"] = rotalign_stats["latency_sec"]

    if prediction_records is not None:
        add_paired_prediction_summary(results, prediction_records)
        write_prediction_records(args.prediction_output, prediction_records)
        write_prediction_sidecar(
            args.prediction_output,
            prediction_records,
            results,
            {
                "translator": args.translator,
                "source_model": args.source_model,
                "target_model": args.target_model,
                "eval_file": args.eval_file,
                "task_type": task_type,
                "device": args.device,
                "dtype": args.dtype,
                "max_new_tokens": getattr(args, "max_new_tokens", None),
                "source_reasoning_mode": args.source_reasoning_mode,
                "source_use_chat_template": source_use_chat_template,
                "target_use_chat_template": target_use_chat_template,
                "source_enable_thinking": getattr(args, "source_enable_thinking", "auto"),
                "target_enable_thinking": getattr(args, "target_enable_thinking", "auto"),
                "methods": list(args.methods),
                "gate_mode": args.gate_mode,
                "fixed_gate": args.fixed_gate,
                "gate_values": list(args.gate_values),
                "gate_search_file": args.gate_search_file,
                "gate_search_limit": args.gate_search_limit,
                "quantize": not args.no_quantize,
                "source_kv_control": args.source_kv_control,
                "quantization_control": args.quantization_control,
                "translated_kv_control": args.translated_kv_control,
                "fusion_rule": fusion_rule,
                "kv_transport": kv_transport,
                "position_selection_ratio": position_selection_ratio,
                "position_selection_metric": position_selection_metric,
                "position_selection_prior_file": args.position_selection_prior_file,
                "position_selection_prior_source": args.position_selection_prior_source,
                "position_selection_prior_bins": args.position_selection_prior_bins,
                "runtime_head_selection_ratio": runtime_head_selection_ratio,
                "runtime_head_selection_metric": runtime_head_selection_metric,
                "runtime_head_prior_file": runtime_head_prior_file,
                "runtime_head_prior_load": runtime_head_prior_load,
                "runtime_head_prior_save": runtime_head_prior_save,
                "runtime_head_prior_metric": args.runtime_head_prior_metric,
                "runtime_head_prior_alpha": runtime_head_prior_alpha,
                "runtime_head_gate_metric": runtime_head_gate_metric,
                "runtime_head_gate_strength": runtime_head_gate_strength,
                "runtime_head_prior_shrinkage": runtime_head_prior_shrinkage,
                "runtime_head_prior_shrink_target": runtime_head_prior_shrink_target,
                "runtime_head_prior_bundle_metadata": loaded_head_prior_metadata,
                "per_head_position_budget_mode": per_head_position_budget_mode,
                "drop_target_layers": sorted(int(layer) for layer in drop_target_layers),
                "drop_target_layer_mode": drop_target_layer_mode,
            },
        )

    print("\n=== Summary ===")
    for key, value in results.items():
        print(f"  {key:>26s}: {value:.6f}")


if __name__ == "__main__":
    main()
