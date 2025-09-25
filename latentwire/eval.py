# latentwire/eval.py
# Deterministic, hardened evaluation for LatentWire.
# Adds first-token accuracy diagnostics and preserves robust BOS/anchor policy.

import os
import time
import json
import argparse
import gc
import ast
from typing import List, Dict, Any, Tuple, Optional

import torch
import math

from latentwire.models import (
    InterlinguaEncoder,
    Adapter,
    LMWrapper,
    LMConfig,
    ByteTokenizer,
    SimpleEncoder,
    STQueryEncoder,
)
from latentwire.core_utils import (
    patch_dataloader_defaults,
    apply_anchor_normalization,
    quantize_dequantize,
    compute_wire_metrics,
    tensor_rms,
    combine_latents,
    clean_pred,
    build_chat_prompts,
    build_neutral_encoder_texts,
    truncate_chat_to_k_tokens,
    content_only_m_token_chat_prompt,
    build_token_budget_prompts,
    collate_bytes,
    make_anchor_text,
    infer_anchor_mode_and_text,
    SYSTEM_PROMPT,
    bos_policy,
    split_user_and_anchor,
)
from latentwire.data import load_examples
from latentwire.core_utils import batch_metrics, _normalize, em, f1

# ---------------------------
# Defaults / hardening toggles
# ---------------------------

EVAL_FIXED_SEED = 12345  # deterministic eval
DEFAULT_ANSWER_PREFIX = "Answer: "


def _parse_device_map(spec: Optional[str]):
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    if s.lower() == "auto":
        return "auto"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, int):
            return {"": int(parsed)}
        if isinstance(parsed, str) and parsed.isdigit():
            return {"": int(parsed)}
        return parsed
    except Exception:
        pass
    if s.isdigit():
        return {"": int(s)}
    return {"": s}


def _parse_device_csv(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None:
        return None
    s = str(spec).strip()
    if not s or s.lower() == "auto":
        return None
    out: List[int] = []
    for chunk in s.split(","):
        token = chunk.strip()
        if not token:
            continue
        if not token.isdigit():
            return None
        out.append(int(token))
    return out or None


def _parse_models_arg(spec: Optional[str]) -> List[str]:
    valid = ["llama", "qwen"]
    if not spec:
        return valid
    models = []
    for chunk in spec.split(','):
        name = chunk.strip().lower()
        if name in valid and name not in models:
            models.append(name)
    return models or valid


def _primary_device(wrapper: LMWrapper) -> torch.device:
    return next(wrapper.model.parameters()).device


def _safe_load(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # PyTorch >= 2.4
    except TypeError:
        return torch.load(path, map_location=map_location)

def _to_long(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device, dtype=torch.long)

def _ensure_dir(p: Optional[str]):
    if p:
        os.makedirs(p, exist_ok=True)

def _is_finite(x: float) -> bool:
    return (x == x) and math.isfinite(x)

def _per_example_calibrate(prefix: torch.Tensor, target_rms: float) -> torch.Tensor:
    """Per-example calibration to a scalar target RMS."""
    cur = prefix.float().pow(2).mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-8)
    gain = (prefix.new_tensor(float(target_rms)) / cur).to(prefix.dtype)
    return prefix * gain

def _calibrate_prefix(prefix: torch.Tensor, wrapper: LMWrapper, mode: str, fixed_rms: Optional[float], stats: Optional[dict], model_key: str) -> Tuple[torch.Tensor, float, float]:
    """Returns (calibrated_prefix, pre_calib_rms_scalar, target_rms_scalar)."""
    pre_scalar = tensor_rms(prefix)  # for logging
    # decide scalar target
    if mode == "none":
        tgt = pre_scalar
    elif mode == "embed_rms":
        tgt = wrapper.input_embedding_rms()
    elif mode == "fixed":
        tgt = float(fixed_rms or 0.015)
    elif mode == "train_stats":
        if stats and model_key in stats:
            # prefer calibrated rms mean if available
            tgt = float(stats[model_key].get("rms_mean_cal", stats[model_key].get("rms_mean", wrapper.input_embedding_rms())))
        else:
            tgt = wrapper.input_embedding_rms()
    else:
        tgt = wrapper.input_embedding_rms()

    # per-example calibration
    if mode != "none":
        prefix = _per_example_calibrate(prefix, tgt)
    return prefix, pre_scalar, float(tgt)


def _answer_lengths_eval(wrapper: LMWrapper, answers: List[str], max_answer_tokens: int, device: str) -> torch.Tensor:
    tokens = wrapper.tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_answer_tokens,
        add_special_tokens=True,
    )
    ids = tokens["input_ids"].to(device)
    pad_id = getattr(wrapper.tokenizer, "pad_token_id", None)
    if pad_id is None:
        lengths = torch.full((ids.size(0),), ids.size(1), device=device)
    else:
        lengths = ids.ne(int(pad_id)).sum(dim=1)
    return lengths



# ---------------------------
# Encoder text alignment (auto)
# ---------------------------

def _encoder_texts_for_mode(mode: str, wrapper: Optional[LMWrapper], raw_sources: List[str]) -> List[str]:
    if mode == "raw" or wrapper is None:
        return raw_sources
    elif mode == "neutral_chat":
        return build_neutral_encoder_texts(raw_sources)
    elif mode in ("llama_chat", "qwen_chat"):
        return build_chat_prompts(wrapper.tokenizer, raw_sources)
    else:
        raise ValueError(f"Unknown encoder_text_mode: {mode}")

@torch.no_grad()
def _compute_Z_for_mode(
    encoder_type: str,
    encoder,
    mode: str,
    wrapper: Optional[LMWrapper],
    raw_sources: List[str],
    device: str,
    byte_max: Optional[int],
) -> torch.Tensor:
    if encoder_type.startswith("simple"):
        texts = _encoder_texts_for_mode(mode, wrapper, raw_sources)
        return encoder(texts)
    else:
        byte_tok = ByteTokenizer(max_bytes=byte_max or 512)
        z_bytes = collate_bytes(raw_sources, byte_tok, device)
        return encoder(z_bytes)

def _maybe_save_Z(out_dir: Optional[str], tag: str, Z: torch.Tensor):
    if out_dir:
        path = os.path.join(out_dir, f"Z_{tag}.pt")
        torch.save(Z.to("cpu"), path)
        print(f"Saved Z[{tag}] to {path}")

def _pick_fallback_mode_from_cfg(cfg: dict) -> str:
    return "neutral_chat" if bool(cfg.get("encoder_use_chat_template", False)) else "raw"


def format_with_chat_template(tokenizer, user_text: str, system_text: Optional[str], assistant_prefill: Optional[str]) -> str:
    """Apply chat template with optional assistant prefill handled correctly for chat LLMs."""
    messages: List[Dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    base_prefill = assistant_prefill or ""
    if base_prefill and not base_prefill.endswith(" "):
        base_prefill = base_prefill + " "
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not rendered:
            raise ValueError("empty rendering")
        if base_prefill:
            rendered = rendered + base_prefill
        return rendered
    except Exception:
        # Retry by explicitly continuing an assistant turn (covers tokenizers that require an explicit assistant message).
        fallback_msgs = list(messages)
        fallback_msgs.append({"role": "assistant", "content": base_prefill})
        fallback_kwargs = {"tokenize": False, "add_generation_prompt": False, "continue_final_message": True}
        try:
            rendered = tokenizer.apply_chat_template(fallback_msgs, **fallback_kwargs)
            if rendered:
                return rendered
        except Exception:
            pass
        # Final fallback: synthesize a minimal chat prompt ourselves (system/user + assistant header)
        system_block = f"System: {system_text}\n" if system_text else ""
        assistant_hdr = "Assistant:" if not base_prefill else "Assistant: " + base_prefill
        return f"{system_block}User: {user_text}\n\n{assistant_hdr}"

def _best_mode_from_scores(candidates: List[str], scores: Dict[str, float], cfg: dict) -> str:
    finite = {k: v for k, v in scores.items() if _is_finite(v)}
    if finite:
        vals = list(finite.values())
        if (max(vals) - min(vals)) < 1e-3:
            return _pick_fallback_mode_from_cfg(cfg)
        return min(finite, key=lambda k: finite[k])
    return _pick_fallback_mode_from_cfg(cfg)

@torch.no_grad()
def _select_best_encoder_mode_for_model(
    encoder_type: str,
    encoder,
    model_name: str,
    wrapper: LMWrapper,
    adapter: Adapter,
    raw_sources: List[str],
    golds: List[str],
    device: str,
    byte_max: Optional[int],
    out_dir: Optional[str],
    debug: bool,
    cfg: dict,
    anchor_token_text: Optional[str],
    calibration_mode: str,
    prefix_target_rms: Optional[float],
    train_stats: Optional[dict],
) -> Tuple[str, torch.Tensor, Dict[str, float]]:
    candidates = ["raw"]
    if encoder_type.startswith("simple"):
        candidates.extend(["neutral_chat", f"{model_name}_chat"])

    scores: Dict[str, float] = {}
    Z_cache: Dict[str, torch.Tensor] = {}

    for mode in candidates:
        Z = _compute_Z_for_mode(encoder_type, encoder, mode, wrapper, raw_sources, device, byte_max)
        Z_cache[mode] = Z
        with torch.no_grad():
            prefix = adapter(Z)
            prefix, _, _ = _calibrate_prefix(prefix, wrapper, calibration_mode, prefix_target_rms, train_stats, model_name)
        nll = avg_nll_latent(wrapper, prefix, golds, wrapper.tokenizer, device, anchor_token_text=anchor_token_text)
        scores[mode] = float("nan") if (nll is None) else float(nll)

    best_mode = _best_mode_from_scores(candidates, scores, cfg)
    Z_best = Z_cache[best_mode]
    if debug:
        print(f"[align:{model_name}] encoder_text_mode candidates={candidates} | nlls={scores} | picked={best_mode}")
    _maybe_save_Z(out_dir, f"{model_name}_{best_mode}", Z_best)
    return best_mode, Z_best, scores

# ---------------------------
# Loss/NLL helpers
# ---------------------------

@torch.no_grad()
def avg_nll_text(wrapper: LMWrapper, prompts_text: List[str], answers: List[str], tokenizer, device: str) -> Optional[float]:
    if wrapper is None:
        return None
    tot_w, tot_tok, skipped = 0.0, 0, 0
    for i in range(len(prompts_text)):
        enc_p = _to_long(tokenizer(prompts_text[i], return_tensors="pt", add_special_tokens=False).input_ids, device)
        enc_a = _to_long(tokenizer(answers[i], return_tensors="pt", add_special_tokens=True).input_ids, device)
        loss, n_tok = wrapper.loss_with_text_prompt(enc_p, enc_a)
        if (loss is None) or (not torch.isfinite(loss)):
            skipped += 1
            continue
        n = int(n_tok) if (n_tok is not None and int(n_tok) > 0) else (enc_a.size(1) - 1)
        tot_w += float(loss.item()) * n
        tot_tok += n
    if tot_tok == 0:
        return None
    return tot_w / tot_tok

@torch.no_grad()
def avg_nll_latent(
    wrapper: LMWrapper,
    prefix: torch.Tensor,
    answers: List[str],
    tokenizer,
    device: str,
    anchor_token_text: Optional[str] = None,
) -> Optional[float]:
    if wrapper is None:
        return None

    anchor_ids = None
    if anchor_token_text:
        try:
            anchor_ids = tokenizer.encode(anchor_token_text, add_special_tokens=False)
        except Exception:
            enc = tokenizer(anchor_token_text, add_special_tokens=False, return_attention_mask=False)
            anchor_ids = enc.get("input_ids", [])
            if isinstance(anchor_ids, list) and anchor_ids and isinstance(anchor_ids[0], list):
                anchor_ids = anchor_ids[0]

    tot_w, tot_tok, skipped = 0.0, 0, 0
    for i, a in enumerate(answers):
        a_ids = _to_long(tokenizer(a, return_tensors="pt", add_special_tokens=True).input_ids, device)
        loss = wrapper.forward_with_prefix_loss(prefix[i:i+1], a_ids, anchor_token_ids=anchor_ids)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        n_tok = max(1, a_ids.size(1) - 1)
        tot_w += float(loss.item()) * n_tok
        tot_tok += int(n_tok)
    return None if tot_tok == 0 else (tot_w / tot_tok)

# ---------------------------
# First-token accuracy diagnostics
# ---------------------------

def _first_content_token_ids(tokenizer, golds: List[str], device: str) -> torch.Tensor:
    """
    Robustly extract the first *content* token id per example:
      - ignore left PAD
      - skip BOS if present at the first non-PAD position
    """
    ids = tokenizer(golds, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).input_ids.to(device)
    B, T = ids.size()
    pad = getattr(tokenizer, "pad_token_id", None)
    bos = getattr(tokenizer, "bos_token_id", None)

    if pad is None:
        nonpad = torch.ones_like(ids, dtype=torch.bool, device=device)
    else:
        nonpad = ids.ne(int(pad))

    # first non-PAD index
    first_idx = nonpad.float().argmax(dim=1)
    row = torch.arange(B, device=device)
    first_tok = ids[row, first_idx]

    if bos is not None:
        is_bos = first_tok.eq(int(bos))
        next_idx = torch.clamp(first_idx + 1, max=T - 1)
        first_tok = torch.where(is_bos, ids[row, next_idx], first_tok)

    return first_tok

@torch.no_grad()
def first_token_topk_acc(
    wrapper: LMWrapper,
    prefix: torch.Tensor,
    golds: List[str],
    anchor_token_text: Optional[str],
    append_bos_after_prefix: Optional[bool],
    k_values=(1, 5),
    skip: bool = False,
    chunk_size: int = 16,
) -> Dict[str, float]:
    """
    Compute top-k accuracy for the very first answer token predicted from
    (prefix + optional anchor + optional BOS).
    """
    if skip:
        return {f"first_token_top{k}": float("nan") for k in k_values}

    device = next(wrapper.model.parameters()).device
    gold_first = _first_content_token_ids(wrapper.tokenizer, golds, device)

    total = prefix.size(0)
    chunk = max(int(chunk_size or 1), 1)
    correct = {int(k): 0.0 for k in k_values}
    counts = 0

    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        prefix_chunk = prefix[start:end]
        logits = wrapper.first_token_logits_from_prefix(
            prefix_chunk,
            anchor_token_text=anchor_token_text,
            append_bos_after_prefix=append_bos_after_prefix,
        )
        probs = torch.softmax(logits, dim=-1)
        V = probs.size(-1)
        gold_chunk = gold_first[start:end]
        for k in k_values:
            kk = max(1, min(int(k), V))
            topk_ids = probs.topk(k=kk, dim=-1).indices
            match = (topk_ids == gold_chunk.unsqueeze(-1)).any(dim=-1).float()
            correct[int(k)] += float(match.sum().item())
        counts += end - start

    accs = {f"first_token_top{int(k)}": (correct[int(k)] / max(counts, 1)) for k in k_values}
    return accs

# ---------------------------
# Debug helpers
# ---------------------------

def _latent_debug_stats(name: str, Z: torch.Tensor, prefix: torch.Tensor, adapter: Adapter, wrapper: LMWrapper) -> Dict[str, float]:
    with torch.no_grad():
        z_std = float(Z.std().item())
        z_mean_norm = float(Z.norm(dim=-1).mean().item())
        p_std = float(prefix.std().item())
        p_mean_norm = float(prefix.norm(dim=-1).mean().item())
        try:
            scale = float(adapter.scale.detach().cpu().item())
        except Exception:
            scale = float('nan')
        try:
            emb_rms = wrapper.input_embedding_rms()
        except Exception:
            emb_rms = float('nan')
    print(f"[debug:{name}] adapter.scale={scale:.4f} | Z.std={z_std:.4f} Z.mean||={z_mean_norm:.4f} | "
          f"prefix.std={p_std:.4f} prefix.mean||={p_mean_norm:.4f} | embed.RMS={emb_rms:.4f}")
    return {
        "adapter_scale": scale,
        "Z_std": z_std,
        "Z_mean_norm": z_mean_norm,
        "prefix_std": p_std,
        "prefix_mean_norm": p_mean_norm,
        "embed_rms": emb_rms
    }

# ---------------------------
# Chunked evaluation helpers
# ---------------------------

@torch.no_grad()
def evaluate_model_chunked_text(
    wrapper: LMWrapper,
    prompts: list,
    max_new_tokens: int,
    chunk_size: int,
    tag: str = "",
    lengths: Optional[torch.Tensor] = None,
):
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(prompts) if len(prompts) > 0 else 1

    preds = []
    t0 = time.time()
    for i in range(0, len(prompts), chunk_size):
        batch = prompts[i:i + chunk_size]
        cap = max_new_tokens
        if lengths is not None and lengths.numel() > 0:
            chunk_len = lengths[i:i + chunk_size]
            if chunk_len.numel() > 0:
                cap = max(1, min(max_new_tokens, int(chunk_len.max().item())))
        out_ids = wrapper.generate_from_text(batch, max_new_tokens=cap, temperature=0.0)
        preds.extend(wrapper.decode_batch_then_clean(out_ids))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total = time.time() - t0
    return preds, t_total

@torch.no_grad()
def evaluate_model_chunked_latent(
    wrapper: LMWrapper,
    prefix_embeds: torch.Tensor,
    max_new_tokens: int,
    chunk_size: int,
    tag: str = "",
    anchor_token_text: str = None,
    min_new_tokens: int = 0,
    eos_ban_steps: int = 0,
    first_token_top_p: float = 1.0,
    first_token_temperature: float = 0.0,
    append_bos_after_prefix: Optional[bool] = None,
    lengths: Optional[torch.Tensor] = None,
):
    N = prefix_embeds.size(0)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = N if N > 0 else 1

    preds = []
    t0 = time.time()
    for i in range(0, N, chunk_size):
        pb = prefix_embeds[i:i + chunk_size]
        cap = max_new_tokens
        if lengths is not None and lengths.numel() > 0:
            chunk_len = lengths[i:i + chunk_size]
            if chunk_len.numel() > 0:
                cap = max(1, min(max_new_tokens, int(chunk_len.max().item())))
        out_ids = wrapper.generate_from_prefix(
            pb,
            max_new_tokens=cap,
            temperature=0.0,
            top_p=1.0,
            anchor_token_text=anchor_token_text,
            min_new_tokens=min_new_tokens,
            eos_ban_steps=eos_ban_steps,
            first_token_top_p=first_token_top_p,
            first_token_temperature=first_token_temperature,
            append_bos_after_prefix=append_bos_after_prefix,
        )
        preds.extend(wrapper.decode_batch_then_clean(out_ids))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total = time.time() - t0
    return preds, t_total

# ---------------------------
# Standard evaluation
# ---------------------------


def _run_text_path(
    wrapper: LMWrapper,
    chat_prompts: List[str],
    prompts_raw: List[str],
    golds: List[str],
    args,
    name: str,
    answer_lengths: Optional[torch.Tensor],
):
    preds, t_text = evaluate_model_chunked_text(
        wrapper,
        chat_prompts,
        args.max_new_tokens,
        args.chunk_size,
        name,
        lengths=answer_lengths,
    )
    em_score, f1_score = batch_metrics(preds, golds)
    nll = avg_nll_text(wrapper, chat_prompts, golds, wrapper.tokenizer, wrapper.model.device)
    prompt_tok = wrapper.tokenizer(
        chat_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
    )
    pad_id = wrapper.tokenizer.pad_token_id
    avg_prompt_tokens = float((prompt_tok["input_ids"] != pad_id).sum().item()) / max(1, len(prompts_raw))
    return {
        "preds": preds,
        "metrics": {"em": em_score, "f1": f1_score, "nll": nll},
        "time": t_text,
        "avg_prompt_tokens": avg_prompt_tokens,
    }


def _run_latent_path(
    args,
    name: str,
    wrapper: LMWrapper,
    prefix: torch.Tensor,
    anchor_text: Optional[str],
    append_bos: Optional[bool],
    prompts_raw: List[str],
    chat_prompts: List[str],
    golds: List[str],
    latent_len: int,
    answer_lengths: Optional[torch.Tensor],
):
    acc = first_token_topk_acc(
        wrapper,
        prefix,
        golds,
        anchor_text or None,
        append_bos_after_prefix=append_bos,
        skip=bool(getattr(args, "skip_prefix_acc", False)),
        chunk_size=max(1, getattr(args, "chunk_size", 8)),
    )

    latent_preds, t_latent = evaluate_model_chunked_latent(
        wrapper,
        prefix,
        args.max_new_tokens,
        args.chunk_size,
        name,
        anchor_token_text=anchor_text or None,
        min_new_tokens=args.min_new_tokens,
        eos_ban_steps=args.eos_ban_steps,
        first_token_top_p=args.first_token_top_p,
        first_token_temperature=args.first_token_temperature,
        append_bos_after_prefix=append_bos,
        lengths=answer_lengths,
    )

    if args.debug and args.debug_print_first > 0:
        print(f"\n[DEBUG] First generations ({name}, latent):")
        for i, pred in enumerate(latent_preds[:args.debug_print_first]):
            print(f"  {i}: '{pred}'")

    k_budget = args.token_budget_k or latent_len
    trunc_prompts = build_token_budget_prompts(
        wrapper.tokenizer, prompts_raw, chat_prompts, k_budget, args.token_budget_mode
    )
    trunc_preds, t_trunc = evaluate_model_chunked_text(
        wrapper,
        trunc_prompts,
        args.max_new_tokens,
        args.chunk_size,
        name,
        lengths=answer_lengths,
    )

    latent_em, latent_f1 = batch_metrics(latent_preds, golds)
    trunc_em, trunc_f1 = batch_metrics(trunc_preds, golds)

    latent_nll = avg_nll_latent(wrapper, prefix, golds, wrapper.tokenizer, prefix.device, anchor_text or None)

    return {
        "latent": {
            "preds": latent_preds,
            "metrics": {"em": latent_em, "f1": latent_f1, "nll": latent_nll, **acc},
            "time": t_latent,
        },
        "trunc": {
            "preds": trunc_preds,
            "metrics": {"em": trunc_em, "f1": trunc_f1},
            "time": t_trunc,
        },
    }

def run_standard_eval(args, device, dtype, encoded_latents, prompts_raw, golds,
                      llama_id, qwen_id, latent_len, d_z, cfg, train_stats,
                      models: Optional[List[str]] = None):
    print("\n[Standard Evaluation Mode]\n(Use --sequential_eval to enable per-model encoder text auto-alignment.)")

    ckpt_path = args.ckpt
    ckpt_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else ckpt_path

    llama_map_spec = args.llama_device_map if args.llama_device_map is not None else cfg.get("llama_device_map")
    qwen_map_spec = args.qwen_device_map if args.qwen_device_map is not None else cfg.get("qwen_device_map")
    llama_devices_spec = args.llama_devices if args.llama_devices is not None else cfg.get("llama_devices")
    qwen_devices_spec = args.qwen_devices if args.qwen_devices is not None else cfg.get("qwen_devices")
    gpu_mem_budget = args.gpu_mem_gib if args.gpu_mem_gib is not None else cfg.get("gpu_mem_gib", 78.0)

    llama_device_ids = _parse_device_csv(llama_devices_spec)
    if llama_device_ids is None:
        llama_device_ids = _parse_device_csv(llama_map_spec)
    qwen_device_ids = _parse_device_csv(qwen_devices_spec)
    if qwen_device_ids is None:
        qwen_device_ids = _parse_device_csv(qwen_map_spec)

    def _build_max_memory(devices: Optional[List[int]], budget_gib: float):
        if not devices or not torch.cuda.is_available():
            return None
        budget = f"{int(budget_gib)}GiB"
        table: Dict[int, str] = {}
        for idx in range(torch.cuda.device_count()):
            table[idx] = budget if idx in devices else "0GiB"
        return table

    llama_max_memory = _build_max_memory(llama_device_ids, gpu_mem_budget)
    qwen_max_memory = _build_max_memory(qwen_device_ids, gpu_mem_budget)

    llama_map = _parse_device_map(llama_map_spec)
    qwen_map = _parse_device_map(qwen_map_spec)

    if llama_map is None and llama_max_memory is not None and device == "cuda":
        llama_map = "auto"
    if qwen_map is None and qwen_max_memory is not None and device == "cuda":
        qwen_map = "auto"

    requested = models or ["llama", "qwen"]
    wrappers: Dict[str, LMWrapper] = {}

    if "llama" in requested:
        llama = LMWrapper(LMConfig(
            model_id=llama_id,
            device=device,
            dtype=dtype,
            load_4bit=args.load_4bit,
            device_map=llama_map,
            max_memory=llama_max_memory,
        ))
        try:
            if hasattr(llama.model.config, "use_cache"):
                llama.model.config.use_cache = False
        except Exception:
            pass
        wrappers["llama"] = llama

    if "qwen" in requested:
        qwen = LMWrapper(LMConfig(
            model_id=qwen_id,
            device=device,
            dtype=dtype,
            load_4bit=args.load_4bit,
            device_map=qwen_map,
            max_memory=qwen_max_memory,
        ))
        try:
            if hasattr(qwen.model.config, "use_cache"):
                qwen.model.config.use_cache = False
        except Exception:
            pass
        wrappers["qwen"] = qwen

    if not wrappers:
        raise ValueError("No models selected for evaluation.")

    max_answer_tokens = int(cfg.get("max_answer_tokens", getattr(args, "max_answer_tokens", 32)))

    adapter_hidden_mult = int(cfg.get("adapter_hidden_mult", 1))
    adapter_colorize = bool(cfg.get("adapter_colorize", False))
    adapter_dropout = float(args.adapter_dropout) if args.adapter_dropout is not None else float(cfg.get("adapter_dropout", 0.0))
    adapter_enable_metadata = bool(cfg.get("adapter_enable_metadata", True))

    per_model_latent_len = int(cfg.get("latent_shared_len", latent_len)) + int(cfg.get("latent_private_len", 0))
    if per_model_latent_len <= 0:
        per_model_latent_len = latent_len

    model_contexts = {name: {"wrapper": wrapper} for name, wrapper in wrappers.items()}

    answer_lengths = {}
    for name, ctx in model_contexts.items():
        target_device = _primary_device(ctx["wrapper"])
        answer_lengths[name] = _answer_lengths_eval(ctx["wrapper"], golds, max_answer_tokens, target_device)

    strip_literal = cfg.get("strip_anchor_text") or DEFAULT_ANSWER_PREFIX
    if strip_literal and not strip_literal.endswith(" "):
        strip_literal = strip_literal + " "

    anchor_info = {}
    for name, ctx in model_contexts.items():
        mode, anchor_text_src = infer_anchor_mode_and_text(
            ctx["wrapper"], cfg, args.latent_anchor_mode, args.latent_anchor_text
        )
        anchor = make_anchor_text(mode, ctx["wrapper"], anchor_text_src)
        if mode != "text":
            anchor = None
        has_anchor = bool(anchor)
        bos_flag = bos_policy(args.append_bos_after_prefix, [0] if has_anchor else [])
        if mode == "chat":
            bos_flag = False
        anchor_info[name] = {
            "mode": mode,
            "text": anchor_text_src,
            "anchor": anchor,
            "bos": bos_flag,
        }

    use_chat_template_flag = str(getattr(args, "use_chat_template", "yes")).lower()
    apply_chat_template = use_chat_template_flag != "no"
    for name, ctx in model_contexts.items():
        if apply_chat_template:
            info = anchor_info.get(name, {})
            assistant_prefill = strip_literal or None
            chat_user = [split_user_and_anchor(raw, strip_literal or "")[0] for raw in prompts_raw]
            ctx["chat"] = [
                format_with_chat_template(
                    ctx["wrapper"].tokenizer,
                    user_text=user_text,
                    system_text=SYSTEM_PROMPT,
                    assistant_prefill=assistant_prefill,
                )
                for user_text in chat_user
            ]
        else:
            ctx["chat"] = build_chat_prompts(ctx["wrapper"].tokenizer, prompts_raw)

    text_results = {}
    text_wall = 0.0
    for name, ctx in model_contexts.items():
        res = _run_text_path(
            ctx["wrapper"],
            ctx["chat"],
            prompts_raw,
            golds,
            args,
            name,
            answer_lengths[name],
        )
        text_results[name] = res
        text_wall += res["time"]
    # Capture text baseline summary for logging
    print("\n— Text baseline summary:")
    for name, ctx in model_contexts.items():
        metrics = text_results[name]["metrics"]
        print(f"{name}: EM={metrics['em']:.3f} F1={metrics['f1']:.3f}")

    # Reattach Prefix-Tuning adapters if available (for latent runs)
    try:
        from peft import PeftModel  # type: ignore

        prefix_paths = {
            "llama": os.path.join(ckpt_dir, "prefix_llama"),
            "qwen": os.path.join(ckpt_dir, "prefix_qwen"),
        }
        for name, path in prefix_paths.items():
            if name not in wrappers:
                continue
            if os.path.isdir(path):
                wrappers[name].model = PeftModel.from_pretrained(wrappers[name].model, path).eval()
                print(f"✓ Loaded Prefix-Tuning adapters for {name}")
    except Exception as exc:
        print(f"[WARN] Prefix-Tuning reload skipped: {exc}")

    adapters = {}
    for name in list(wrappers.keys()):
        adapter = Adapter(
            d_z=d_z,
            d_model=wrappers[name].d_model,
            latent_length=per_model_latent_len,
            enable_metadata=adapter_enable_metadata,
            length_norm=max_answer_tokens,
            hidden_mult=adapter_hidden_mult,
            colorize=adapter_colorize,
            dropout=adapter_dropout,
        ).to(_primary_device(wrappers[name])).eval()
        path = os.path.join(ckpt_dir, f"adapter_{name}.pt")
        state = _safe_load(path, map_location=device)
        try:
            adapter.load_state_dict(state, strict=True)
        except Exception as exc:
            print(f"⚠️  Adapter({name}) strict load failed; retrying with strict=False ({exc})")
            adapter.load_state_dict(state, strict=False)
        if adapter_colorize and hasattr(adapter, "install_color_from_wrapper"):
            try:
                adapter.install_color_from_wrapper(wrappers[name])
            except Exception:
                pass
        adapters[name] = adapter
        model_contexts[name]["adapter"] = adapter

    combined_latents = {
        name: combine_latents(encoded_latents, name) for name in model_contexts
    }
    quant_bits = getattr(args, "latent_quant_bits", None)
    quant_group = getattr(args, "latent_quant_group_size", 32)
    prefix_map = {}
    debug_map = {name: {} for name in model_contexts}
    with torch.no_grad():
        for name, ctx in model_contexts.items():
            latents = combined_latents[name]
            if quant_bits is not None:
                latents = quantize_dequantize(latents, quant_bits, group_size=quant_group)
                combined_latents[name] = latents
            target_device = _primary_device(ctx["wrapper"])
            latents_for_adapter = latents.to(target_device, non_blocking=True)
            lengths_for_adapter = answer_lengths[name].to(target_device, non_blocking=True)
            prefix = ctx["adapter"](latents_for_adapter, answer_lengths=lengths_for_adapter)
            prefix, rms_val, tgt_val = _calibrate_prefix(
                prefix,
                ctx["wrapper"],
                args.calibration,
                args.prefix_target_rms,
                train_stats,
                name,
            )
            if args.debug:
                print(f"[calib:{name}] mode={args.calibration} prefix_rms={rms_val:.5f} -> target={tgt_val:.5f}")
            prefix = prefix * args.prefix_gain
            prefix_map[name] = prefix
            if args.debug:
                debug = _latent_debug_stats(name, latents, prefix, ctx["adapter"], ctx["wrapper"])
                debug.update({
                    "encoder_text_mode": "standard",
                    "calibration_mode": args.calibration,
                    "append_bos_after_prefix": args.append_bos_after_prefix,
                    "latent_anchor_mode": anchor_info[name]["mode"],
                    "latent_anchor_text": anchor_info[name]["anchor"],
                    "model_id": ctx["wrapper"].cfg.model_id,
                })
                debug_map[name] = debug

    latent_results = {}
    latent_wall = 0.0
    trunc_wall = 0.0
    for name, ctx in model_contexts.items():
        anchor_payload = anchor_info[name]["anchor"] if anchor_info[name]["mode"] == "text" else None
        append_bos = anchor_info[name]["bos"]
        res = _run_latent_path(
            args,
            name,
            ctx["wrapper"],
            prefix_map[name],
            anchor_payload,
            append_bos,
            prompts_raw,
            ctx["chat"],
            golds,
            latent_len,
            answer_lengths[name],
        )
        latent_results[name] = res
        latent_wall += res["latent"]["time"]
        trunc_wall += res["trunc"]["time"]

    model_outputs = {}
    for name in model_contexts.keys():
        model_outputs[name] = {
            "avg_prompt_tokens": text_results[name]["avg_prompt_tokens"],
            "text_preds": text_results[name]["preds"],
            "latent_preds": latent_results[name]["latent"]["preds"],
            "trunc_preds": latent_results[name]["trunc"]["preds"],
            "times": {
                "text": text_results[name]["time"],
                "latent": latent_results[name]["latent"]["time"],
                "trunc": latent_results[name]["trunc"]["time"],
            },
            "metrics": {
                "text": text_results[name]["metrics"],
            "latent": latent_results[name]["latent"]["metrics"],
            "trunc": latent_results[name]["trunc"]["metrics"],
        },
            "chat_prompts": model_contexts[name]["chat"],
            "debug": debug_map.get(name, {}),
        }

    joint_em = joint_f1 = agreement_rate = float("nan")
    joint_preds = []
    if all(name in model_outputs for name in ("llama", "qwen")):
        agree = 0
        anchor_ids = {
            name: (wrappers[name]._encode_anchor_text(anchor_info[name]["anchor"]) if anchor_info[name]["mode"] == "text" and anchor_info[name]["anchor"] else None)
            for name in ("llama", "qwen")
        }

        for i, (candA, candB) in enumerate(zip(model_outputs["llama"]["latent_preds"], model_outputs["qwen"]["latent_preds"])):
            prefix_ll = prefix_map["llama"][i:i+1]
            prefix_qw = prefix_map["qwen"][i:i+1]
            A_ids_L = _to_long(wrappers["llama"].tokenizer(candA, return_tensors="pt", add_special_tokens=True).input_ids, device)
            A_ids_Q = _to_long(wrappers["qwen"].tokenizer(candA,  return_tensors="pt", add_special_tokens=True).input_ids, device)
            B_ids_L = _to_long(wrappers["llama"].tokenizer(candB, return_tensors="pt", add_special_tokens=True).input_ids, device)
            B_ids_Q = _to_long(wrappers["qwen"].tokenizer(candB,  return_tensors="pt", add_special_tokens=True).input_ids, device)
            scoreA = wrappers["llama"].score_prefix_logprob(prefix_ll, A_ids_L, anchor_token_ids=anchor_ids["llama"]) + \
                     wrappers["qwen"].score_prefix_logprob(prefix_qw, A_ids_Q, anchor_token_ids=anchor_ids["qwen"])
            scoreB = wrappers["llama"].score_prefix_logprob(prefix_ll, B_ids_L, anchor_token_ids=anchor_ids["llama"]) + \
                     wrappers["qwen"].score_prefix_logprob(prefix_qw, B_ids_Q, anchor_token_ids=anchor_ids["qwen"])
            pick = candA if scoreA >= scoreB else candB
            joint_preds.append(pick)
            if _normalize(candA) == _normalize(candB):
                agree += 1

        joint_em, joint_f1 = batch_metrics(joint_preds, golds)
        agreement_rate = agree / len(golds)

    wire = {}
    bytes_per_latent = 0
    group_size = getattr(args, "latent_quant_group_size", 32)
    scale_bits = getattr(args, "latent_quant_scale_bits", 16)
    selected_bytes: Optional[int] = None
    if all(name in model_outputs for name in ("llama", "qwen")):
        wire = compute_wire_metrics(
            model_outputs["llama"]["chat_prompts"],
            model_outputs["qwen"]["chat_prompts"],
            combined_latents["llama"],
            group_size=group_size,
            scale_bits=scale_bits,
            selected_bits=getattr(args, "latent_quant_bits", None),
        )
        llama_latents = combined_latents["llama"]
        base_bytes_per_latent = int(llama_latents.element_size() * llama_latents.size(1) * llama_latents.size(2))
        selected_bytes = wire.get("selected_latent_bytes")
        bytes_per_latent = int(selected_bytes) if selected_bytes is not None else base_bytes_per_latent
        wire["base_latent_bytes"] = base_bytes_per_latent
    else:
        name, latent_tensor = next(iter(combined_latents.items()))
        prompts = model_outputs[name]["chat_prompts"]
        num_latent = latent_tensor.numel()
        base_bytes_per_latent = int(num_latent * 4)
        bytes_per_latent = base_bytes_per_latent
        wire = {
            "prompt_chars": {name: sum(len(p) for p in prompts)},
            "prompt_count": len(prompts),
            "latent_shape": list(latent_tensor.shape),
            "latent_bytes": {
                "fp32": base_bytes_per_latent,
                "fp16": int(num_latent * 2),
            },
            "group_size": int(group_size),
            "scale_bits": int(scale_bits),
            "selected_latent_bytes": None,
        }

    oracle_em = 0.0
    oracle_f1 = 0.0
    if all(name in model_outputs for name in ("llama", "qwen")):
        for candA, candB, gold in zip(
            model_outputs["llama"]["latent_preds"],
            model_outputs["qwen"]["latent_preds"],
            golds,
        ):
            oracle_em += max(em(candA, gold), em(candB, gold))
            oracle_f1 += max(f1(candA, gold), f1(candB, gold))
        oracle_em /= len(golds)
        oracle_f1 /= len(golds)

    if not all(name in model_outputs for name in ("llama", "qwen")):
        joint_summary = {
            "em": None,
            "f1": None,
            "agreement": None,
            "oracle": {"em": oracle_em, "f1": oracle_f1},
        }
    else:
        joint_summary = {
            "em": joint_em,
            "f1": joint_f1,
            "agreement": agreement_rate,
            "oracle": {"em": oracle_em, "f1": oracle_f1},
        }

    summary = {
        "samples": len(prompts_raw),
        "max_new_tokens": args.max_new_tokens,
        "latent_len": latent_len,
        "device": device,
        "dtype": str(dtype),
        "avg_prompt_tokens": {name: text_results[name]["avg_prompt_tokens"] for name in model_contexts},
        "compression": {name: text_results[name]["avg_prompt_tokens"] / max(latent_len, 1) for name in model_contexts},
        "payload_bytes": bytes_per_latent,
        "payload_bytes_detail": {
            "fp32": wire["latent_bytes"]["fp32"],
            "fp16": wire["latent_bytes"]["fp16"],
            "selected": selected_bytes,
        },
        "wire": wire,
        "text": {
            name: {
                "em": text_results[name]["metrics"]["em"],
                "f1": text_results[name]["metrics"]["f1"],
                "nll_token": text_results[name]["metrics"]["nll"],
            }
            for name in model_contexts
        },
        "latent": {
            name: {
                **latent_results[name]["latent"]["metrics"],
                "nll_token": latent_results[name]["latent"]["metrics"].get("nll"),
            }
            for name in model_contexts
        },
        "token_budget": {
            "mode": args.token_budget_mode,
            "k": args.token_budget_k or latent_len,
            **{name: latent_results[name]["trunc"]["metrics"] for name in model_contexts},
        },
        "joint": joint_summary,
        "debug": {name: model_outputs[name]["debug"] for name in model_contexts},
        "oracle": {"em": oracle_em, "f1": oracle_f1},
    }
    summary["text"]["wall_clock_sec"] = text_wall
    summary.setdefault("latent", {})
    summary["latent"]["wall_clock_sec"] = latent_wall
    summary.setdefault("token_budget", {})
    summary["token_budget"]["wall_clock_sec"] = trunc_wall
    try:
        summary.setdefault("wire", {}).setdefault("wire_ratio", {}).update(wire.get("wire_ratio", {}))
    except Exception:
        pass
    summary.setdefault("debug", {})
    summary["debug"]["settings"] = {
        "latent_anchor_mode": args.latent_anchor_mode,
        "latent_anchor_text": args.latent_anchor_text,
        "prefix_gain": args.prefix_gain,
        "calibration_mode": args.calibration,
        "append_bos_after_prefix": args.append_bos_after_prefix,
        "decode": {
            "min_new_tokens": args.min_new_tokens,
            "eos_ban_steps": args.eos_ban_steps,
            "first_token_top_p": args.first_token_top_p,
            "first_token_temperature": args.first_token_temperature,
        },
    }

    preds_dump = []
    for i in range(len(prompts_raw)):
        entry = {
            "prompt_raw": prompts_raw[i],
            "gold": golds[i],
        }
        for name, outputs in model_outputs.items():
            entry.setdefault("chat_prompts", {})[name] = outputs["chat_prompts"][i]
            entry[f"text_pred_{name}"] = outputs["text_preds"][i]
            entry[f"latent_pred_{name}"] = outputs["latent_preds"][i]
            entry[f"trunc_pred_{name}"] = outputs["trunc_preds"][i]
        preds_dump.append(entry)

    for wrapper in wrappers.values():
        del wrapper
    for adapter in adapters.values():
        del adapter
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return summary, preds_dump

def run_sequential_eval(args, device, dtype, encoded_latents, prompts_raw, golds,
                        llama_id, qwen_id, latent_len, d_z, encoder_type,
                        byte_max, cfg, train_stats, models: Optional[List[str]] = None):
    """Fallback to standard evaluation logic; kept for API compatibility."""
    return run_standard_eval(args, device, dtype, encoded_latents, prompts_raw, golds, llama_id, qwen_id, latent_len, d_z, cfg, train_stats, models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--llama_id", type=str, default=None)
    ap.add_argument("--qwen_id", type=str, default=None)
    ap.add_argument("--llama_device_map", type=str, default=None,
                    help="Device map for the Llama wrapper during eval (e.g., 0, 'auto', or JSON dict).")
    ap.add_argument("--qwen_device_map", type=str, default=None,
                    help="Device map for the Qwen wrapper during eval (e.g., 1, 'auto', or JSON dict).")
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot","squad","squad_v2"])
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--hotpot_config", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--token_budget_mode", type=str, default="content_only", choices=["chat_full", "content_only"])
    ap.add_argument("--token_budget_k", type=int, default=None)
    ap.add_argument("--llama_devices", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--qwen_devices", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gpu_mem_gib", type=float, default=78.0,
                    help="Per-GPU memory budget (GiB) used to constrain auto device maps.")

    # Anchors / BOS controls (hardened defaults)
    ap.add_argument("--latent_anchor_mode", type=str, default="auto", choices=["auto","chat","text","none"])
    ap.add_argument("--latent_anchor_text", type=str, default="Answer: ")
    ap.add_argument("--append_bos_after_prefix", type=str, default="auto", choices=["auto","yes","no"])
    ap.add_argument("--skip_prefix_acc", action="store_true", help="Skip first-token accuracy evaluation (saves memory)")
    ap.add_argument("--use_chat_template", type=str, default="yes", choices=["yes","no"],
                    help="Toggle chat template application when constructing text prompts.")

    ap.add_argument("--sequential_eval", action="store_true")
    ap.add_argument("--models", type=str, default="llama,qwen",
                    help="Comma-separated list of models to evaluate (subset of llama,qwen)")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--hf_encoder_id", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_enc_tokens", type=int, default=1024)
    ap.add_argument("--fresh_eval", action="store_true")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_print_first", type=int, default=0)
    ap.add_argument("--debug_topk", type=int, default=0)
    ap.add_argument("--debug_topk_examples", type=int, default=2)

    # new decode controls (latent) - deterministic by default
    ap.add_argument("--min_new_tokens", type=int, default=1)
    ap.add_argument("--eos_ban_steps", type=int, default=0)
    ap.add_argument("--first_token_top_p", type=float, default=1.0)
    ap.add_argument("--first_token_temperature", type=float, default=0.0)
    # Optional latent quantization sweeps (applied before adapters)
    ap.add_argument("--latent_quant_bits", type=int, default=None, choices=[16, 8, 6, 4],
                    help="If set, quantize→dequantize latent Z to this bitwidth for eval.")
    ap.add_argument("--latent_quant_group_size", type=int, default=32,
                    help="Group size for symmetric per-group quantization of Z.")
    ap.add_argument("--latent_quant_scale_bits", type=int, default=16,
                    help="Metadata bits allocated for each group's scale during wire accounting.")

    # eval-time amplitude control
    ap.add_argument("--prefix_gain", type=float, default=1.0)

    # calibration modes
    ap.add_argument("--calibration", type=str, default="embed_rms", choices=["none","embed_rms","fixed","train_stats"])
    ap.add_argument("--prefix_target_rms", type=float, default=None)

    ap.add_argument("--adapter_dropout", type=float, default=None,
                    help="Override adapter dropout probability when loading adapters (defaults to training config).")

    # encoder input alignment
    ap.add_argument("--encoder_text_mode", type=str, default="auto",
                    choices=["auto","raw","neutral_chat","llama_chat","qwen_chat"])

    # deterministic eval seed
    ap.add_argument("--seed", type=int, default=EVAL_FIXED_SEED)

    args = ap.parse_args()
    patch_dataloader_defaults()
    apply_anchor_normalization(args)

    selected_models = _parse_models_arg(getattr(args, "models", ""))

    # Deterministic by default
    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device + dtype
    if args.device:
        device = args.device
        print(f"Using forced device: {device}")
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Auto-detected device: {device}")

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
        print("Using fp16 precision on MPS for memory efficiency")
    else:
        dtype = torch.float32
        print("Using fp32 precision on CPU")

    _ensure_dir(args.out_dir)

    # Load run config (from training)
    # Resolve checkpoint directory if a file path (e.g., state.pt) was provided
    ckpt_path = args.ckpt
    if os.path.isfile(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_path)
    else:
        ckpt_dir = ckpt_path

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)

    encoder_type = cfg.get("encoder_type", "byte")
    trained_used_neutral = bool(cfg.get("encoder_use_chat_template", False))
    if (not args.sequential_eval) and (not args.fresh_eval) and encoder_type.startswith("simple") and trained_used_neutral:
        print("⚠️  Detected SimpleEncoder trained with a chat-style wrapper, but Standard eval is about to reuse a cached Z.pt.")
        print("    Add --fresh_eval (or use --sequential_eval) so Z is recomputed with the correct wrapper.")

    llama_id = args.llama_id or cfg["llama_id"]
    qwen_id  = args.qwen_id  or cfg["qwen_id"]
    latent_len = int(cfg.get("latent_len", getattr(args, "latent_len", 0) or 0))
    latent_shared_len = int(cfg.get("latent_shared_len", latent_len))
    latent_private_len = int(cfg.get("latent_private_len", 0))
    if selected_models:
        model_keys = tuple(selected_models)
    else:
        model_keys = tuple(cfg.get("models", ["llama", "qwen"]))
        if not model_keys:
            model_keys = ("llama", "qwen")
    if latent_private_len > 0 and latent_shared_len + latent_private_len * len(model_keys) != latent_len:
        latent_len = latent_shared_len + latent_private_len * len(model_keys)
    d_z = int(cfg["d_z"])
    byte_max = cfg.get("byte_max", 512)

    # Try to load training-time prefix stats (optional)
    train_stats_path = os.path.join(ckpt_dir, "training_stats.json")
    train_stats = None
    if os.path.isfile(train_stats_path):
        try:
            with open(train_stats_path, "r") as f:
                train_stats = json.load(f)
            print(f"Loaded training_stats.json: {train_stats_path}")
        except Exception as e:
            print(f"⚠️  Failed to read training_stats.json: {e}")

    # Load eval examples
    if args.dataset.startswith("squad"):
        eval_examples = load_examples(dataset=args.dataset, split="validation", samples=args.samples, seed=42)
        dataset_detail = args.dataset
    else:
        eval_examples = load_examples(dataset="hotpot", split="validation", samples=args.samples, seed=42,
                                      config=(args.hotpot_config or "fullwiki"))
        dataset_detail = f"hotpot:{args.hotpot_config or 'fullwiki'}"

    prompts_raw = [e["source"] for e in eval_examples]
    golds       = [e["answer"] for e in eval_examples]

    # Determine encoder inputs (match training preprocessing)
    strip_literal = cfg.get("strip_anchor_text") or DEFAULT_ANSWER_PREFIX
    if strip_literal and not strip_literal.endswith(" "):
        strip_literal = strip_literal + " "
    # Remove the anchor literal the encoder never saw during training
    encoder_sources = [
        split_user_and_anchor(src, strip_literal or "")[0] for src in prompts_raw
    ]

    encoder_text_mode = (args.encoder_text_mode or "auto").lower()
    trained_neutral_chat = bool(cfg.get("encoder_use_chat_template", False))
    if encoder_text_mode == "auto":
        encoder_text_mode = "neutral_chat" if trained_neutral_chat else "raw"

    if encoder_text_mode == "neutral_chat":
        encoder_inputs = build_neutral_encoder_texts(encoder_sources)
    else:
        # Fall back to raw inputs (already anchor-stripped). Additional modes (llama/qwen chat)
        # are handled elsewhere when sequential alignment is requested.
        encoder_inputs = encoder_sources

    print(f"Encoder input alignment: mode={encoder_text_mode} | strip_anchor={'yes' if strip_literal else 'no'} | samples={len(encoder_inputs)}")

    # Build a Z for wire metrics
    Z_path = os.path.join(args.out_dir, "Z.pt") if args.out_dir else None
    if Z_path and (not args.fresh_eval) and os.path.exists(Z_path):
        print(f"Loading Z from {Z_path}")
        encoded_latents = torch.load(Z_path, map_location=device)
    else:
        print("Building encoder and computing Z...")
        if encoder_type == "byte":
            encoder_wire = InterlinguaEncoder(
                d_z=d_z,
                latent_shared_len=latent_shared_len,
                latent_private_len=latent_private_len,
                model_keys=model_keys,
            ).to(device).eval()
            encoder_wire.load_state_dict(_safe_load(os.path.join(ckpt_dir, "encoder.pt"), map_location=device))
            with torch.no_grad():
                byte_tok = ByteTokenizer(max_bytes=byte_max)
                z_bytes = collate_bytes(encoder_inputs, byte_tok, device)
                encoded_latents = encoder_wire(z_bytes, return_components=True)
        elif encoder_type == "stq":
            encoder_wire = STQueryEncoder(d_z=d_z, latent_len=latent_len,
                                         hf_encoder_id=(args.hf_encoder_id or cfg.get('hf_encoder_id','sentence-transformers/all-MiniLM-L6-v2')),
                                         max_tokens=(args.max_enc_tokens or cfg.get('max_enc_tokens',1024))).to(device).eval()
            encoder_wire.load_state_dict(_safe_load(os.path.join(ckpt_dir, "encoder.pt"), map_location=device))
            with torch.no_grad():
                raw = encoder_wire(encoder_inputs)
                shared = raw[:, :latent_shared_len] if latent_shared_len > 0 else raw.new_zeros(raw.size(0), 0, raw.size(-1))
                private = {}
                start = latent_shared_len
                for key in model_keys:
                    if latent_private_len > 0:
                        private[key] = raw[:, start:start + latent_private_len]
                    else:
                        private[key] = raw.new_zeros(raw.size(0), 0, raw.size(-1))
                    start += latent_private_len
                encoded_latents = {"shared": shared, "private": private}
        else:
            encoder_wire = SimpleEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
            encoder_wire.load_state_dict(_safe_load(os.path.join(ckpt_dir, "encoder.pt"), map_location=device))
            with torch.no_grad():
                raw = encoder_wire(encoder_inputs)
                shared = raw[:, :latent_shared_len] if latent_shared_len > 0 else raw.new_zeros(raw.size(0), 0, raw.size(-1))
                private = {}
                start = latent_shared_len
                for key in model_keys:
                    if latent_private_len > 0:
                        private[key] = raw[:, start:start + latent_private_len]
                    else:
                        private[key] = raw.new_zeros(raw.size(0), 0, raw.size(-1))
                    start += latent_private_len
                encoded_latents = {"shared": shared, "private": private}
        if Z_path:
            torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else {kk: vv.cpu() for kk, vv in v.items()} for k, v in encoded_latents.items()}, Z_path)
            print(f"Saved Z to {Z_path}")

    # === EVALUATION MODES ===
    if args.sequential_eval:
        summary, preds_dump = run_sequential_eval(
            args=args, device=device, dtype=dtype,
            encoded_latents=encoded_latents, prompts_raw=prompts_raw, golds=golds,
            llama_id=llama_id, qwen_id=qwen_id,
            latent_len=latent_len, d_z=d_z, encoder_type=encoder_type, byte_max=byte_max,
            cfg=cfg, train_stats=train_stats, models=selected_models
        )
    else:
        summary, preds_dump = run_standard_eval(
            args=args, device=device, dtype=dtype,
            encoded_latents=encoded_latents, prompts_raw=prompts_raw, golds=golds,
            llama_id=llama_id, qwen_id=qwen_id,
            latent_len=latent_len, d_z=d_z, cfg=cfg, train_stats=train_stats,
            models=selected_models
        )

    # Tag dataset in the summary
    summary["dataset"] = args.dataset
    if args.dataset == "hotpot":
        summary["dataset_detail"] = {"config": (args.hotpot_config or "fullwiki")}

    print("\n==== LatentWire Evaluation ====")
    print(f"Dataset: {dataset_detail}")
    print(f"Samples: {summary['samples']}  |  Max new tokens: {summary['max_new_tokens']}")
    print(f"Device: {summary['device']}  |  Dtype: {summary['dtype'].split('.')[-1] if isinstance(summary['dtype'], str) else 'unknown'}")
    def _fmt_float(value, default="-"):
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return default

    llama_tokens = _fmt_float(summary['avg_prompt_tokens'].get('llama')) if isinstance(summary.get('avg_prompt_tokens'), dict) else "-"
    qwen_tokens = _fmt_float(summary['avg_prompt_tokens'].get('qwen')) if isinstance(summary.get('avg_prompt_tokens'), dict) else "-"
    llama_comp = summary.get('compression', {})
    llama_comp_val = _fmt_float(llama_comp.get('llama')) if isinstance(llama_comp, dict) else "-"
    qwen_comp_val = _fmt_float(llama_comp.get('qwen')) if isinstance(llama_comp, dict) else "-"

    print(f"Avg prompt tokens (Llama): {llama_tokens} | (Qwen): {qwen_tokens} | Latent length M: {summary['latent_len']}")
    print(f"Compression ratio (Llama): {llama_comp_val}x | (Qwen): {qwen_comp_val}x")
    payload_detail = summary.get('payload_bytes_detail', {})
    selected_bytes = payload_detail.get('selected')
    base_bytes = summary['wire'].get('base_latent_bytes', summary['payload_bytes'])
    payload_line = (
        f"Approx interlingua payload per example: {summary['payload_bytes']} bytes"
        + (f" ({args.latent_quant_bits}-bit selected)" if selected_bytes is not None else " (fp32)")
        + f"; fp16 reference: {payload_detail.get('fp16', 'n/a')} bytes; fp32 reference: {payload_detail.get('fp32', base_bytes)} bytes"
    )
    print(payload_line)
    wire_ratio = summary.get('wire', {}).get('wire_ratio', {}) if isinstance(summary.get('wire'), dict) else {}
    ratio_val = wire_ratio.get('latent_over_onecopy_fp16') if isinstance(wire_ratio, dict) else None
    if ratio_val is not None:
        print(f"latent/text bytes (one-copy, fp16): {float(ratio_val):.2f}x")
    else:
        print("latent/text bytes (one-copy, fp16): n/a")

    print("\n— Baseline: Text prompting")
    if summary['text'].get('llama') is not None:
        print(f"Llama  EM: {summary['text']['llama']['em']:.3f}  F1: {summary['text']['llama']['f1']:.3f}  |  NLL/token (gold): {summary['text']['llama']['nll_token']}")
    if summary['text'].get('qwen') is not None:
        print(f"Qwen   EM: {summary['text']['qwen']['em']:.3f}   F1: {summary['text']['qwen']['f1']:.3f}   |  NLL/token (gold): {summary['text']['qwen']['nll_token']}")
    print(f"Wall clock: {summary['text']['wall_clock_sec']:.2f}s")

    print("\n— Latent prompting (shared interlingua)")
    if summary['latent'].get('llama') is not None:
        print(f"Llama  EM: {summary['latent']['llama']['em']:.3f}  F1: {summary['latent']['llama']['f1']:.3f}  |  NLL/token (gold): {summary['latent']['llama']['nll_token']}")
        if "first_token_top1" in summary['latent']['llama']:
            print(f"       First-token acc: top1={summary['latent']['llama']['first_token_top1']:.3f}  top5={summary['latent']['llama']['first_token_top5']:.3f}")
    if summary['latent'].get('qwen') is not None:
        print(f"Qwen   EM: {summary['latent']['qwen']['em']:.3f}   F1: {summary['latent']['qwen']['f1']:.3f}  |  NLL/token (gold): {summary['latent']['qwen']['nll_token']}")
        if "first_token_top1" in summary['latent']['qwen']:
            print(f"       First-token acc: top1={summary['latent']['qwen']['first_token_top1']:.3f}  top5={summary['latent']['qwen']['first_token_top5']:.3f}")
    print(f"Wall clock: {summary['latent']['wall_clock_sec']:.2f}s")

    print(f"\n— Token-budget baseline (mode: {summary['token_budget'].get('mode','content_only')})")
    if summary['token_budget'].get('llama') is not None:
        print(f"Llama  EM: {summary['token_budget']['llama']['em']:.3f}  F1: {summary['token_budget']['llama']['f1']:.3f}")
    if summary['token_budget'].get('qwen') is not None:
        print(f"Qwen   EM: {summary['token_budget']['qwen']['em']:.3f}   F1: {summary['token_budget']['qwen']['f1']:.3f}")
    print(f"Wall clock: {summary['token_budget']['wall_clock_sec']:.2f}s")

    print("\n— 2-LLM joint (rescored pick on latent runs)")
    joint_block = summary.get('joint', {})
    if joint_block.get('em') is not None:
        print(f"Joint  EM: {joint_block['em']:.3f}  F1: {joint_block['f1']:.3f}")
        if joint_block.get('agreement') is not None:
            print(f"Inter-model agreement (normalized): {joint_block['agreement']:.3f}")
        print(f"Oracle upper bound:  EM {summary['oracle']['em']:.3f}  F1 {summary['oracle']['f1']:.3f}")
    else:
        print("Joint metrics unavailable (single-model evaluation).")

    print("\n==== METRICS_JSON ====")
    print(json.dumps(summary, indent=2))

    if args.out_dir:
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        # CSV (flat)
        import csv as _csv
        with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["group","model","EM","F1","NLL/token","wall_clock_sec","compression","payload_bytes","samples","M","token_budget_mode","token_budget_k","dataset"])
            for mdl in ["llama","qwen"]:
                if summary["text"].get(mdl) is not None:
                    w.writerow(["text", mdl, summary["text"][mdl]["em"], summary["text"][mdl]["f1"], summary["text"][mdl]["nll_token"],
                                summary["text"]["wall_clock_sec"], summary["compression"].get(mdl,""), summary["payload_bytes"],
                                summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            for mdl in ["llama","qwen"]:
                if summary["latent"].get(mdl) is not None:
                    w.writerow(["latent", mdl, summary["latent"][mdl]["em"], summary["latent"][mdl]["f1"], summary["latent"][mdl]["nll_token"],
                                summary["latent"]["wall_clock_sec"], summary["compression"].get(mdl,""), summary["payload_bytes"],
                                summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            for mdl in ["llama","qwen"]:
                if summary["token_budget"].get(mdl) is not None:
                    w.writerow(["token_budget", mdl, summary["token_budget"][mdl]["em"], summary["token_budget"][mdl]["f1"], "",
                                summary["token_budget"]["wall_clock_sec"], summary["compression"].get(mdl,""), summary["payload_bytes"],
                                summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            if summary.get("joint", {}).get("em") is not None:
                w.writerow(["joint","both", summary["joint"]["em"], summary["joint"]["f1"], "", "", "", summary["payload_bytes"],
                            summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])

        dump_path = os.path.join(args.out_dir, "predictions.jsonl")
        with open(dump_path, "w") as f:
            for rec in preds_dump:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote per-example predictions to {dump_path}")

if __name__ == "__main__":
    main()
