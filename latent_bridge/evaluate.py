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


def infer_task_type(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            return "mcq" if "choices" in obj else "generation"
    raise ValueError(f"No examples found in {path}")


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
        return DynamicCache(past_key_values, config=model.config)
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
def _prepare_prefix_state(model, tokenizer, prompt: str, device: str) -> PrefixState:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
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
    if extra:
        record.update(extra)
    records.append(record)


def write_prediction_records(path: str, records: list[dict[str, Any]]) -> None:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


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


def _translated_bits(
    translator: RotAlignKVTranslator,
    seq_len: int,
    quantize: bool,
    *,
    active_k_head_counts: list[int] | None = None,
    active_v_head_counts: list[int] | None = None,
) -> float:
    selected_layers = translator.selected_layer_indices()
    if active_k_head_counts is None:
        active_k_head_counts = [translator.selected_head_count(layer_idx) for layer_idx in selected_layers]
    if active_v_head_counts is None:
        active_v_head_counts = [translator.selected_head_count(layer_idx) for layer_idx in selected_layers]
    return float(
        sum(_bits_per_kv_tensor(translator, seq_len, quantize, selected_heads=count) for count in active_k_head_counts)
        + sum(_bits_per_kv_tensor(translator, seq_len, quantize, selected_heads=count) for count in active_v_head_counts)
    )


def _communication_bits(
    translator: RotAlignKVTranslator,
    seq_len: int,
    quantize: bool,
    translated_kv_control: str,
    protocol: str = "fused",
) -> float:
    if translated_kv_control != "real":
        return 0.0
    if protocol == "translated_only":
        return _translated_bits(translator, seq_len, quantize)
    active_k_head_counts: list[int] = []
    active_v_head_counts: list[int] = []
    for layer_idx in translator.selected_layer_indices():
        gate_k, gate_v = translator.gate_value(layer_idx)
        if abs(gate_k) > 1e-5:
            active_k_head_counts.append(translator.selected_head_count(layer_idx))
        if abs(gate_v) > 1e-5:
            active_v_head_counts.append(translator.selected_head_count(layer_idx))
    return _translated_bits(
        translator,
        seq_len,
        quantize,
        active_k_head_counts=active_k_head_counts,
        active_v_head_counts=active_v_head_counts,
    )


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
) -> tuple[PrefixState, dict[str, float]]:
    tgt_prompt_ids = target_tokenizer(target_prompt, return_tensors="pt").input_ids.to(device)
    tgt_prefix_ids, tgt_last_token = _split_prompt_prefix(tgt_prompt_ids)
    if tgt_prefix_ids is None:
        return PrefixState(None, tgt_last_token, 1), {"bits": 0.0}
    tgt_out = target_model(
        input_ids=tgt_prefix_ids,
        attention_mask=_ones(tgt_prefix_ids.shape[1], device),
        use_cache=True,
    )
    tgt_pkv = list(_normalize_cache(tgt_out.past_key_values))

    src_ids = source_tokenizer(source_prompt, return_tensors="pt").input_ids.to(device)
    src_prefix_ids, _ = _split_prompt_prefix(src_ids)
    if src_prefix_ids is None:
        src_pkv = []
    else:
        src_out = source_model(
            input_ids=src_prefix_ids,
            attention_mask=_ones(src_prefix_ids.shape[1], device),
            use_cache=True,
        )
        src_pkv = list(_normalize_cache(src_out.past_key_values))

    fused_pkv: list[tuple[torch.Tensor, torch.Tensor]] = []
    for tgt_l in range(translator.config.num_tgt_layers):
        src_l = translator.layer_map[tgt_l]
        K_s, V_s = src_pkv[src_l]
        K_s, V_s = _apply_source_kv_control(K_s, V_s, source_kv_control, tgt_l)
        K_t, V_t = tgt_pkv[tgt_l]
        K_hat, V_hat = translator.translate_layer(
            K_s.to(device=device, dtype=torch.float32),
            V_s.to(device=device, dtype=torch.float32),
            tgt_layer_idx=tgt_l,
            quantize=quantize,
            quantization_control=quantization_control,
        )
        K_hat, V_hat = _apply_translated_kv_control(
            K_hat,
            V_hat,
            translated_kv_control,
            tgt_l,
        )
        K_t_aligned, K_hat = _tail_align_pair(K_t, K_hat.to(dtype=K_t.dtype))
        V_t_aligned, V_hat = _tail_align_pair(V_t, V_hat.to(dtype=V_t.dtype))

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
                    )
                )
            else:
                fused_pkv.append((K_t_aligned, V_t_aligned))
        else:
            raise ValueError(f"Unknown RotAlign protocol: {protocol}")

    seq_len = fused_pkv[0][0].shape[2] if fused_pkv else 0
    prefix_state = PrefixState(
        past_key_values=tuple(fused_pkv),
        last_token=tgt_last_token,
        prefix_len=seq_len + 1,
    )
    return prefix_state, {
        "bits": _communication_bits(
            translator,
            seq_len,
            quantize,
            translated_kv_control,
            protocol,
        )
    }


def _limit_examples(examples, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(examples):
        return list(examples)
    return list(examples[:limit])


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

    def _score_current_gates() -> float:
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
        )

    for tgt_layer_idx in layer_indices:
        current_k, current_v = translator.gate_value(tgt_layer_idx)

        best_k = current_k
        best_k_score = float("-inf")
        for candidate in candidates:
            translator.set_layer_gates(tgt_layer_idx, alpha_k=candidate, alpha_v=current_v)
            score = _score_current_gates()
            if score > best_k_score:
                best_k_score = score
                best_k = candidate

        translator.set_layer_gates(tgt_layer_idx, alpha_k=best_k, alpha_v=current_v)

        best_v = current_v
        best_v_score = float("-inf")
        for candidate in candidates:
            translator.set_layer_gates(tgt_layer_idx, alpha_k=best_k, alpha_v=candidate)
            score = _score_current_gates()
            if score > best_v_score:
                best_v_score = score
                best_v = candidate

        translator.set_layer_gates(tgt_layer_idx, alpha_k=best_k, alpha_v=best_v)
        print(
            f"[gate search] layer {tgt_layer_idx:>2d}: "
            f"K={best_k:.3f} (score={best_k_score:.4f})  "
            f"V={best_v:.3f} (score={best_v_score:.4f})"
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
) -> str:
    hint_prompt = _source_reasoning_prompt(prompt, source_reasoning_mode)
    hint_ids = source_tokenizer(hint_prompt, return_tensors="pt").input_ids.to(device)
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
    records: list[dict[str, Any]] | None = None,
    method_name: str = "rotalign_kv",
) -> float:
    translator = translator.to(device).eval()
    correct = 0
    for idx, ex in enumerate(examples):
        source_prompt = _source_reasoning_prompt(_mcq_prompt(ex.question), source_reasoning_mode)
        target_prompt = _mcq_prompt(ex.question)
        prefix_state, _ = _build_rotalign_prefix_state(
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
    records: list[dict[str, Any]] | None = None,
    method_name: str = "target_alone",
) -> dict[str, float]:
    correct = 0
    total_tokens = 0
    total_ttft = 0.0
    total_elapsed = 0.0
    for idx, ex in enumerate(examples):
        prep_started = time.perf_counter()
        prefix_state = _prepare_prefix_state(model, tokenizer, ex.prompt, device)
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
        )
        augmented = f"Analysis: {hint}\n{ex.prompt}"
        prefix_state = _prepare_prefix_state(target_model, target_tokenizer, augmented, device)
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
) -> tuple[float, float, float]:
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
        fusion_rule,
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
            )
            target_prompt = f"Analysis: {hint}\n{ex.prompt}"

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
                "bits": stats["bits"],
                "generated_tokens": trace.num_generated_tokens,
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
        choices=["static", "cosine", "cosine_shifted", "js_shrinkage", "kalman"],
        default="static",
        help="Static scalar gates or cosine-based runtime attenuation of translated KV.",
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

    print(f"\nLoading source: {args.source_model}")
    tok_s = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tok_s.pad_token_id is None:
        tok_s.pad_token = tok_s.eos_token
    src = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    print(f"Loading target: {args.target_model}")
    tok_t = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tok_t.pad_token_id is None:
        tok_t.pad_token = tok_t.eos_token
    tgt = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    print(f"Loading translator from {args.translator}")
    translator = RotAlignKVTranslator.load(args.translator, map_location=args.device)
    initial_gate_values = translator.gate_values()

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
                    protocol=protocol,
                )
                results[f"{metric_key}_bytes"] = results[f"{metric_key}_bits"] / 8.0
    else:
        if "target" in args.methods:
            target_stats = _eval_generation_target_alone_with_stats(
                tgt, tok_t, examples, args.device, args.max_new_tokens, prediction_records
            )
            results["target_alone"] = target_stats["accuracy"]
            results["target_alone_ttft_sec"] = target_stats["ttft_sec"]
            results["target_alone_tokens_per_sec"] = target_stats["tokens_per_sec"]
            results["target_alone_examples_per_sec"] = target_stats["examples_per_sec"]
            results["target_alone_latency_sec"] = target_stats["latency_sec"]
        if "source" in args.methods:
            source_stats = _eval_generation_target_alone_with_stats(
                src, tok_s, examples, args.device, args.max_new_tokens, prediction_records, "source_alone"
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
                src, tok_s, tgt, tok_t, examples, args.device, args.max_new_tokens
                , source_reasoning_mode=args.source_reasoning_mode, records=prediction_records
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

    print("\n=== Summary ===")
    for key, value in results.items():
        print(f"  {key:>26s}: {value:.6f}")


if __name__ == "__main__":
    main()
