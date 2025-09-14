# latentwire/eval.py
import os
import time
import json
import argparse
import gc
from typing import List, Dict, Any, Tuple, Optional

import torch
import math

from latentwire.models import (
    InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
)
from latentwire.data import load_examples
from latentwire.metrics import batch_metrics, _normalize, em, f1

# Shared helpers (deduped from models/train/eval)
from latentwire.common import (
    clean_pred,
    build_chat_prompts, build_neutral_encoder_texts,
    truncate_chat_to_k_tokens, content_only_m_token_chat_prompt, build_token_budget_prompts,
    collate_bytes, make_anchor_text, infer_anchor_mode_and_text, SYSTEM_PROMPT
)

# ---------------------------
# Utilities
# ---------------------------

def _safe_load(path: str, map_location=None):
    """torch.load with weights_only=True when available; fall back otherwise."""
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

def _tensor_rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt().item())

def _calibrate_prefix(prefix: torch.Tensor, wrapper: LMWrapper, mode: str, fixed_rms: Optional[float], stats: Optional[dict], model_key: str) -> Tuple[torch.Tensor, float, float]:
    """
    Returns (scaled_prefix, current_rms, target_rms)
    mode: 'none' | 'embed_rms' | 'fixed' | 'train_stats'
    """
    cur = _tensor_rms(prefix)
    if mode == "none":
        tgt = cur
    elif mode == "embed_rms":
        tgt = wrapper.input_embedding_rms()
    elif mode == "fixed":
        tgt = float(fixed_rms or 0.015)
    elif mode == "train_stats":
        if stats and model_key in stats and "rms_mean" in stats[model_key]:
            tgt = float(stats[model_key]["rms_mean"])
        else:
            tgt = wrapper.input_embedding_rms()
    else:
        tgt = wrapper.input_embedding_rms()

    if cur > 0 and tgt > 0:
        gain = tgt / cur
        prefix = prefix * gain
    return prefix, cur, tgt

def compute_wire_metrics(llama_chat_prompts: List[str], qwen_chat_prompts: List[str], Z: torch.Tensor) -> Dict[str, Any]:
    avg_text_bytes_llama = int(sum(len(p.encode("utf-8")) for p in llama_chat_prompts) / max(1, len(llama_chat_prompts))) if llama_chat_prompts else 0
    avg_text_bytes_qwen  = int(sum(len(p.encode("utf-8")) for p in qwen_chat_prompts)  / max(1, len(qwen_chat_prompts))) if qwen_chat_prompts else 0
    bytes_fp32 = int(Z.size(1) * Z.size(2) * 4)
    bytes_fp16 = int(Z.size(1) * Z.size(2) * 2)
    max_onecopy = max(avg_text_bytes_llama, avg_text_bytes_qwen)
    return {
        "text_bytes_onecopy": {"llama_avg": avg_text_bytes_llama, "qwen_avg": avg_text_bytes_qwen, "max_avg": max_onecopy},
        "text_bytes_twocopies": {"sum_avg": avg_text_bytes_llama + avg_text_bytes_qwen},
        "latent_bytes": {"fp32": bytes_fp32, "fp16": bytes_fp16},
        "wire_compression": {
            "vs_onecopy_fp16": (float(max_onecopy) / bytes_fp16) if bytes_fp16 > 0 else None,
            "vs_onecopy_fp32": (float(max_onecopy) / bytes_fp32) if bytes_fp32 > 0 else None,
        }
    }

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
        # byte encoder
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

def _best_mode_from_scores(candidates: List[str], scores: Dict[str, float], cfg: dict) -> str:
    # Lower NLL is better
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
        enc_p = _to_long(tokenizer(prompts_text[i], return_tensors="pt", add_special_tokens=True).input_ids, device)
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
    # Pre-encode anchor once
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
    tag: str = ""
):
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(prompts) if len(prompts) > 0 else 1

    preds = []
    t0 = time.time()
    for i in range(0, len(prompts), chunk_size):
        batch = prompts[i:i + chunk_size]
        out_ids = wrapper.generate_from_text(batch, max_new_tokens=max_new_tokens, temperature=0.0)
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
):
    N = prefix_embeds.size(0)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = N if N > 0 else 1

    preds = []
    t0 = time.time()
    for i in range(0, N, chunk_size):
        pb = prefix_embeds[i:i + chunk_size]
        out_ids = wrapper.generate_from_prefix(
            pb,
            max_new_tokens=max_new_tokens,
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
# Standard evaluation (both models loaded; no auto-align)
# ---------------------------

def run_standard_eval(args, device, dtype, Z, prompts_raw, golds, llama_id, qwen_id, latent_len, d_z, cfg, train_stats):
    print("\n[Standard Evaluation Mode - both models loaded]\n(Use --sequential_eval to enable per-model encoder text auto-alignment.)")

    llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))

    adp_llama = Adapter(d_z=d_z, d_model=llama.d_model).to(device).eval()
    adp_qwen  = Adapter(d_z=d_z, d_model=qwen.d_model).to(device).eval()
    adp_llama.load_state_dict(_safe_load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device), strict=True)
    adp_qwen.load_state_dict(_safe_load(os.path.join(args.ckpt, "adapter_qwen.pt"),  map_location=device), strict=True)

    llama_chat = build_chat_prompts(llama.tokenizer, prompts_raw)
    qwen_chat  = build_chat_prompts(qwen.tokenizer,  prompts_raw)

    llama_prompt_tok = llama.tokenizer(llama_chat, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    qwen_prompt_tok  = qwen.tokenizer(qwen_chat,  return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    avg_prompt_tokens_llama = float((llama_prompt_tok["input_ids"] != llama.tokenizer.pad_token_id).sum().item()) / max(1,len(prompts_raw))
    avg_prompt_tokens_qwen  = float((qwen_prompt_tok["input_ids"]  != qwen.tokenizer.pad_token_id).sum().item())  / max(1,len(prompts_raw))

    llama_text_preds, t_text_llama = evaluate_model_chunked_text(llama, llama_chat, args.max_new_tokens, args.chunk_size, "llama")
    qwen_text_preds,  t_text_qwen  = evaluate_model_chunked_text(qwen,  qwen_chat,  args.max_new_tokens, args.chunk_size, "qwen")
    t_text = t_text_llama + t_text_qwen
    llama_text_em, llama_text_f1 = batch_metrics(llama_text_preds, golds)
    qwen_text_em,  qwen_text_f1  = batch_metrics(qwen_text_preds, golds)

    # Anchors: infer from training cfg unless user overrides
    mode_ll, text_ll = infer_anchor_mode_and_text(llama, cfg, args.latent_anchor_mode, args.latent_anchor_text)
    mode_qw, text_qw = infer_anchor_mode_and_text(qwen,  cfg, args.latent_anchor_mode, args.latent_anchor_text)
    anchor_ll = make_anchor_text(mode_ll, llama, text_ll)
    anchor_qw = make_anchor_text(mode_qw, qwen,  text_qw)

    with torch.no_grad():
        prefix_llama = adp_llama(Z)
        prefix_qwen  = adp_qwen(Z)
        prefix_llama, rmsL, tgtL = _calibrate_prefix(prefix_llama, llama, args.calibration, args.prefix_target_rms, train_stats, "llama")
        prefix_qwen,  rmsQ, tgtQ = _calibrate_prefix(prefix_qwen,  qwen,  args.calibration, args.prefix_target_rms, train_stats, "qwen")
        if args.debug:
            print(f"[calib:llama] mode={args.calibration} prefix_rms={rmsL:.5f} -> target={tgtL:.5f}")
            print(f"[calib:qwen]  mode={args.calibration} prefix_rms={rmsQ:.5f} -> target={tgtQ:.5f}")
        prefix_llama = prefix_llama * args.prefix_gain
        prefix_qwen  = prefix_qwen  * args.prefix_gain

    debug_llama = _latent_debug_stats("llama", Z, prefix_llama, adp_llama, llama) if args.debug else {}
    debug_qwen  = _latent_debug_stats("qwen",  Z, prefix_qwen,  adp_qwen,  qwen)  if args.debug else {}
    if args.debug:
        debug_llama.update({"encoder_text_mode": "standard", "calibration_mode": args.calibration,
                            "append_bos_after_prefix": args.append_bos_after_prefix,
                            "latent_anchor_mode": mode_ll, "latent_anchor_text": anchor_ll, "model_id": llama.cfg.model_id})
        debug_qwen.update({"encoder_text_mode": "standard", "calibration_mode": args.calibration,
                           "append_bos_after_prefix": args.append_bos_after_prefix,
                           "latent_anchor_mode": mode_qw, "latent_anchor_text": anchor_qw, "model_id": qwen.cfg.model_id})

    llama_latent_preds, t_latent_llama = evaluate_model_chunked_latent(
        llama, prefix_llama, args.max_new_tokens, args.chunk_size, "llama", anchor_ll or None,
        min_new_tokens=args.min_new_tokens, eos_ban_steps=args.eos_ban_steps,
        first_token_top_p=args.first_token_top_p, first_token_temperature=args.first_token_temperature,
        append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes"))
    )
    qwen_latent_preds,  t_latent_qwen  = evaluate_model_chunked_latent(
        qwen,  prefix_qwen,  args.max_new_tokens, args.chunk_size, "qwen",  anchor_qw or None,
        min_new_tokens=args.min_new_tokens, eos_ban_steps=args.eos_ban_steps,
        first_token_top_p=args.first_token_top_p, first_token_temperature=args.first_token_temperature,
        append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes"))
    )
    if args.debug and args.debug_print_first > 0:
        print("\n[DEBUG] First generations (Llama, latent):")
        for i, pred in enumerate(llama_latent_preds[:args.debug_print_first]):
            print(f"  {i}: '{pred}'")
        print("\n[DEBUG] First generations (Qwen, latent):")
        for i, pred in enumerate(qwen_latent_preds[:args.debug_print_first]):
            print(f"  {i}: '{pred}'")
    if args.debug and args.debug_topk > 0:
        try:
            topkL = llama.peek_first_step_from_prefix(
                prefix_llama[:args.debug_topk_examples], anchor_ll or None,
                append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes")),
                topk=args.debug_topk)
            print("\n[DEBUG] First-step top-k (Llama):")
            for i, rows in enumerate(topkL):
                pretty = ", ".join([(tok.strip().replace('\n','\\n') or '<NL>') + f":{prob:.3f}" for _, prob, tok in rows])
                print(f"  ex{i}: {pretty}")
        except Exception as e:
            print(f"[DEBUG] Llama top-k failed: {e}")
        try:
            topkQ = qwen.peek_first_step_from_prefix(
                prefix_qwen[:args.debug_topk_examples], anchor_qw or None,
                append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes")),
                topk=args.debug_topk)
            print("\n[DEBUG] First-step top-k (Qwen):")
            for i, rows in enumerate(topkQ):
                pretty = ", ".join([(tok.strip().replace('\n','\\n') or '<NL>') + f":{prob:.3f}" for _, prob, tok in rows])
                print(f"  ex{i}: {pretty}")
        except Exception as e:
            print(f"[DEBUG] Qwen top-k failed: {e}")

    t_latent = t_latent_llama + t_latent_qwen

    llama_latent_em, llama_latent_f1 = batch_metrics(llama_latent_preds, golds)
    qwen_latent_em,  qwen_latent_f1  = batch_metrics(qwen_latent_preds, golds)

    k_budget = args.token_budget_k or latent_len
    llama_trunc = build_token_budget_prompts(llama.tokenizer, prompts_raw, llama_chat, k_budget, args.token_budget_mode)
    qwen_trunc  = build_token_budget_prompts(qwen.tokenizer,  prompts_raw, qwen_chat,  k_budget, args.token_budget_mode)

    llama_trunc_preds, t_trunc_llama = evaluate_model_chunked_text(llama, llama_trunc, args.max_new_tokens, args.chunk_size, "llama")
    qwen_trunc_preds,  t_trunc_qwen  = evaluate_model_chunked_text(qwen,  qwen_trunc,  args.max_new_tokens, args.chunk_size, "qwen")
    t_trunc = t_trunc_llama + t_trunc_qwen

    llama_trunc_em, llama_trunc_f1 = batch_metrics(llama_trunc_preds, golds)
    qwen_trunc_em,  qwen_trunc_f1  = batch_metrics(qwen_trunc_preds,  golds)

    llama_latent_nll = avg_nll_latent(llama, prefix_llama, golds, llama.tokenizer, device, anchor_token_text=anchor_ll or None)
    qwen_latent_nll  = avg_nll_latent(qwen,  prefix_qwen,  golds, qwen.tokenizer, device, anchor_token_text=anchor_qw or None)
    llama_text_nll   = avg_nll_text(llama, llama_chat, golds, llama.tokenizer, device)
    qwen_text_nll    = avg_nll_text(qwen,  qwen_chat,  golds, qwen.tokenizer, device)

    # Joint rescoring (simple)
    joint_preds = []
    agree = 0
    anchor_ids_llama = llama._encode_anchor_text(anchor_ll) if anchor_ll else None
    anchor_ids_qwen  = qwen._encode_anchor_text(anchor_qw)  if anchor_qw else None

    for i in range(len(prompts_raw)):
        candA = llama_latent_preds[i]
        candB = qwen_latent_preds[i]
        A_ids_L = _to_long(llama.tokenizer(candA, return_tensors="pt", add_special_tokens=True).input_ids, device)
        A_ids_Q = _to_long(qwen.tokenizer(candA,  return_tensors="pt", add_special_tokens=True).input_ids, device)
        B_ids_L = _to_long(llama.tokenizer(candB, return_tensors="pt", add_special_tokens=True).input_ids, device)
        B_ids_Q = _to_long(qwen.tokenizer(candB,  return_tensors="pt", add_special_tokens=True).input_ids, device)
        scoreA = llama.score_prefix_logprob(prefix_llama[i:i+1], A_ids_L, anchor_token_ids=anchor_ids_llama) + \
                 qwen.score_prefix_logprob(prefix_qwen[i:i+1],  A_ids_Q, anchor_token_ids=anchor_ids_qwen)
        scoreB = llama.score_prefix_logprob(prefix_llama[i:i+1], B_ids_L, anchor_token_ids=anchor_ids_llama) + \
                 qwen.score_prefix_logprob(prefix_qwen[i:i+1],  B_ids_Q, anchor_token_ids=anchor_ids_qwen)
        pick = candA if scoreA >= scoreB else candB
        joint_preds.append(pick)
        if _normalize(candA) == _normalize(candB):
            agree += 1

    joint_em, joint_f1 = batch_metrics(joint_preds, golds)
    agreement_rate = agree / len(prompts_raw)

    wire = compute_wire_metrics(llama_chat, qwen_chat, Z)
    bytes_per_latent = int(Z.element_size() * Z.size(1) * Z.size(2))

    # Oracle upper bound
    oracle_em = 0.0
    oracle_f1 = 0.0
    for pA, pB, g in zip(llama_latent_preds, qwen_latent_preds, golds):
        oracle_em += max(em(pA, g), em(pB, g))
        oracle_f1 += max(f1(pA, g), f1(pB, g))
    oracle_em /= len(golds)
    oracle_f1 /= len(golds)

    summary = {
        "samples": len(prompts_raw),
        "max_new_tokens": args.max_new_tokens,
        "latent_len": latent_len,
        "device": device,
        "dtype": str(dtype),
        "avg_prompt_tokens": {"llama": avg_prompt_tokens_llama, "qwen": avg_prompt_tokens_qwen},
        "compression": {"llama": avg_prompt_tokens_llama/latent_len, "qwen": avg_prompt_tokens_qwen/latent_len},
        "payload_bytes": bytes_per_latent,
        "wire": wire,
        "text": {
            "llama": {"em": llama_text_em, "f1": llama_text_f1, "nll_token": llama_text_nll},
            "qwen":  {"em": qwen_text_em,  "f1": qwen_text_f1,  "nll_token": qwen_text_nll},
            "wall_clock_sec": t_text,
        },
        "latent": {
            "llama": {"em": llama_latent_em, "f1": llama_latent_f1, "nll_token": llama_latent_nll},
            "qwen":  {"em": qwen_latent_em,  "f1": qwen_latent_f1, "nll_token": qwen_latent_nll},
            "wall_clock_sec": t_latent,
        },
        "token_budget": {
            "mode": args.token_budget_mode,
            "k": k_budget,
            "llama": {"em": llama_trunc_em, "f1": llama_trunc_f1},
            "qwen":  {"em": qwen_trunc_em,  "f1": qwen_trunc_f1},
            "wall_clock_sec": t_trunc,
        },
        "joint": {
            "em": joint_em, "f1": joint_f1, "agreement": agreement_rate,
            "oracle": {"em": None, "f1": None},
        },
        "debug": {
            "llama": debug_llama, "qwen": debug_qwen,
            "latent_anchor_mode": args.latent_anchor_mode,
            "latent_anchor_text": anchor_ll if anchor_ll else (anchor_qw or ""),
            "prefix_gain": args.prefix_gain,
            "calibration_mode": args.calibration,
            "append_bos_after_prefix": args.append_bos_after_prefix,
            "decode": {
                "min_new_tokens": args.min_new_tokens,
                "eos_ban_steps": args.eos_ban_steps,
                "first_token_top_p": args.first_token_top_p,
                "first_token_temperature": args.first_token_temperature
            }
        },
        "oracle": {"em": oracle_em, "f1": oracle_f1},
    }

    preds_dump = []
    for i in range(len(prompts_raw)):
        preds_dump.append({
            "prompt_raw": prompts_raw[i],
            "prompt_llama_chat": llama_chat[i],
            "prompt_qwen_chat":  qwen_chat[i],
            "gold": golds[i],
            "text_pred_llama":   llama_text_preds[i],
            "text_pred_qwen":    qwen_text_preds[i],
            "latent_pred_llama": llama_latent_preds[i],
            "latent_pred_qwen":  qwen_latent_preds[i],
            "trunc_pred_llama":  llama_trunc_preds[i],
            "trunc_pred_qwen":   qwen_trunc_preds[i],
        })

    del llama, qwen, adp_llama, adp_qwen
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return summary, preds_dump

# ---------------------------
# Sequential evaluation (with per-model encoder auto-alignment)
# ---------------------------

def run_sequential_eval(args, device, dtype, Z_for_wire, prompts_raw, golds, llama_id, qwen_id, latent_len, d_z, encoder_type, byte_max, cfg, train_stats):
    print("\n[Sequential Evaluation Mode - one model at a time]")

    llama_file = os.path.join(args.out_dir, "llama_results.json") if args.out_dir else None
    qwen_file  = os.path.join(args.out_dir, "qwen_results.json") if args.out_dir else None

    have_llama = os.path.isfile(os.path.join(args.ckpt, "adapter_llama.pt"))
    have_qwen  = os.path.isfile(os.path.join(args.ckpt, "adapter_qwen.pt"))

    # Build encoder (shared)
    if encoder_type.startswith("simple"):
        encoder = SimpleEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
    else:
        encoder = InterlinguaEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
    encoder.load_state_dict(_safe_load(os.path.join(args.ckpt, "encoder.pt"), map_location=device))

    L, Q = {}, {}
    avg_prompt_tokens_llama = 0.0
    avg_prompt_tokens_qwen  = 0.0

    # ----- Llama phase
    if have_llama:
        if (not args.fresh_eval) and llama_file and os.path.exists(llama_file):
            print(f"Loading cached Llama results: {llama_file}")
            with open(llama_file, "r") as f:
                L = json.load(f)
            avg_prompt_tokens_llama = float(L.get("avg_prompt_tokens", 0.0))
        else:
            print("\nEvaluating Llama...")
            llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
            adp_llama = Adapter(d_z=d_z, d_model=llama.d_model).to(device).eval()
            adp_llama.load_state_dict(_safe_load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device), strict=True)

            llama_chat = build_chat_prompts(llama.tokenizer, prompts_raw)
            llama_prompt_tok = llama.tokenizer(llama_chat, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            avg_prompt_tokens_llama = float((llama_prompt_tok["input_ids"] != llama.tokenizer.pad_token_id).sum().item()) / max(1,len(prompts_raw))

            llama_text_preds, t_text_llama = evaluate_model_chunked_text(llama, llama_chat, args.max_new_tokens, args.chunk_size, "llama")

            # Anchors: infer from training cfg unless overridden
            mode_ll, text_ll = infer_anchor_mode_and_text(llama, cfg, args.latent_anchor_mode, args.latent_anchor_text)
            anchor_ll = make_anchor_text(mode_ll, llama, text_ll)

            # Encoder text auto-alignment
            if args.encoder_text_mode == "auto":
                mode, Z, scores = _select_best_encoder_mode_for_model(
                    encoder_type, encoder, "llama", llama, adp_llama,
                    prompts_raw, golds, device, byte_max, args.out_dir, args.debug, cfg=cfg,
                    anchor_token_text=anchor_ll,
                    calibration_mode=args.calibration, prefix_target_rms=args.prefix_target_rms, train_stats=train_stats
                )
            else:
                mode = args.encoder_text_mode
                Z = _compute_Z_for_mode(encoder_type, encoder, mode, llama, prompts_raw, device, byte_max)
                if args.debug:
                    print(f"[align:llama] forced encoder_text_mode={mode}")

            with torch.no_grad():
                prefix_llama = adp_llama(Z)
                prefix_llama, p_rms, e_rms = _calibrate_prefix(prefix_llama, llama, args.calibration, args.prefix_target_rms, train_stats, "llama")
                if args.debug:
                    print(f"[calib:llama] mode={args.calibration} prefix_rms={p_rms:.5f} -> target={e_rms:.5f}")
                prefix_llama = prefix_llama * args.prefix_gain

            debug_llama = _latent_debug_stats("llama", Z, prefix_llama, adp_llama, llama) if args.debug else {}
            if args.debug:
                debug_llama.update({
                    "encoder_text_mode": mode, "calibration_mode": args.calibration,
                    "append_bos_after_prefix": args.append_bos_after_prefix,
                    "latent_anchor_mode": mode_ll, "latent_anchor_text": anchor_ll, "model_id": llama.cfg.model_id
                })

            llama_latent_preds, t_latent_llama = evaluate_model_chunked_latent(
                llama, prefix_llama, args.max_new_tokens, args.chunk_size, "llama", anchor_ll or None,
                min_new_tokens=args.min_new_tokens, eos_ban_steps=args.eos_ban_steps,
                first_token_top_p=args.first_token_top_p, first_token_temperature=args.first_token_temperature,
                append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes"))
            )
            if args.debug and args.debug_print_first > 0:
                print("\n[DEBUG] First generations (Llama, latent):")
                for i, pred in enumerate(llama_latent_preds[:args.debug_print_first]):
                    print(f"  {i}: '{pred}'")
            if args.debug and args.debug_topk > 0:
                try:
                    topkL = llama.peek_first_step_from_prefix(prefix_llama[:args.debug_topk_examples], anchor_ll or None,
                                                              append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes")),
                                                              topk=args.debug_topk)
                    print("\n[DEBUG] First-step top-k (Llama):")
                    for i, rows in enumerate(topkL):
                        pretty = ", ".join([(tok.strip().replace('\n','\\n') or '<NL>') + f":{prob:.3f}" for _, prob, tok in rows])
                        print(f"  ex{i}: {pretty}")
                except Exception as e:
                    print(f"[DEBUG] Llama top-k failed: {e}")

            k_budget = args.token_budget_k or latent_len
            llama_trunc = build_token_budget_prompts(llama.tokenizer, prompts_raw, llama_chat, k_budget, args.token_budget_mode)
            llama_trunc_preds, t_trunc_llama = evaluate_model_chunked_text(llama, llama_trunc, args.max_new_tokens, args.chunk_size, "llama")

            llama_text_em, llama_text_f1     = batch_metrics(llama_text_preds, golds)
            llama_latent_em, llama_latent_f1 = batch_metrics(llama_latent_preds, golds)
            llama_trunc_em, llama_trunc_f1   = batch_metrics(llama_trunc_preds, golds)

            llama_text_nll   = avg_nll_text(llama, llama_chat, golds, llama.tokenizer, device)
            llama_latent_nll = avg_nll_latent(llama, prefix_llama, golds, llama.tokenizer, device, anchor_ll or None)

            L = {
                "avg_prompt_tokens": avg_prompt_tokens_llama,
                "text_preds": llama_text_preds,
                "latent_preds": llama_latent_preds,
                "trunc_preds": llama_trunc_preds,
                "times": {"text": t_text_llama, "latent": t_latent_llama, "trunc": t_trunc_llama},
                "metrics": {
                    "text": {"em": llama_text_em, "f1": llama_text_f1, "nll": llama_text_nll},
                    "latent": {"em": llama_latent_em, "f1": llama_latent_f1, "nll": llama_latent_nll},
                    "trunc": {"em": llama_trunc_em, "f1": llama_trunc_f1}
                },
                "chat_prompts": llama_chat,
                "debug": debug_llama,
            }
            if llama_file:
                with open(llama_file, "w") as f:
                    json.dump(L, f)
                print(f"Saved Llama results to {llama_file}")

            del llama, adp_llama, prefix_llama, llama_prompt_tok, Z
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache(); torch.mps.synchronize(); time.sleep(0.5)

    # ----- Qwen phase
    if have_qwen:
        if (not args.fresh_eval) and qwen_file and os.path.exists(qwen_file):
            print(f"Loading cached Qwen results: {qwen_file}")
            with open(qwen_file, "r") as f:
                Q = json.load(f)
            avg_prompt_tokens_qwen = float(Q.get("avg_prompt_tokens", 0.0))
        else:
            print("\nEvaluating Qwen...")
            qwen = LMWrapper(LMConfig(model_id=qwen_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
            adp_qwen = Adapter(d_z=d_z, d_model=qwen.d_model).to(device).eval()
            adp_qwen.load_state_dict(_safe_load(os.path.join(args.ckpt, "adapter_qwen.pt"), map_location=device), strict=True)

            qwen_chat = build_chat_prompts(qwen.tokenizer, prompts_raw)
            qwen_prompt_tok = qwen.tokenizer(qwen_chat, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            avg_prompt_tokens_qwen = float((qwen_prompt_tok["input_ids"] != qwen.tokenizer.pad_token_id).sum().item()) / max(1,len(prompts_raw))

            qwen_text_preds, t_text_qwen = evaluate_model_chunked_text(qwen, qwen_chat, args.max_new_tokens, args.chunk_size, "qwen")

            mode_qw, text_qw = infer_anchor_mode_and_text(qwen, cfg, args.latent_anchor_mode, args.latent_anchor_text)
            anchor_qw = make_anchor_text(mode_qw, qwen, text_qw)

            if args.encoder_text_mode == "auto":
                mode, Z, scores = _select_best_encoder_mode_for_model(
                    encoder_type, encoder, "qwen", qwen, adp_qwen,
                    prompts_raw, golds, device, byte_max, args.out_dir, args.debug, cfg=cfg,
                    anchor_token_text=anchor_qw,
                    calibration_mode=args.calibration, prefix_target_rms=args.prefix_target_rms, train_stats=train_stats
                )
            else:
                mode = args.encoder_text_mode
                Z = _compute_Z_for_mode(encoder_type, encoder, mode, qwen, prompts_raw, device, byte_max)
                if args.debug:
                    print(f"[align:qwen] forced encoder_text_mode={mode}")

            with torch.no_grad():
                prefix_qwen = adp_qwen(Z)
                prefix_qwen, p_rms, e_rms = _calibrate_prefix(prefix_qwen, qwen, args.calibration, args.prefix_target_rms, train_stats, "qwen")
                if args.debug:
                    print(f"[calib:qwen]  mode={args.calibration} prefix_rms={p_rms:.5f} -> target={e_rms:.5f}")
                prefix_qwen = prefix_qwen * args.prefix_gain

            debug_qwen = _latent_debug_stats("qwen", Z, prefix_qwen, adp_qwen, qwen) if args.debug else {}
            if args.debug:
                debug_qwen.update({
                    "encoder_text_mode": mode, "calibration_mode": args.calibration,
                    "append_bos_after_prefix": args.append_bos_after_prefix,
                    "latent_anchor_mode": mode_qw, "latent_anchor_text": anchor_qw, "model_id": qwen.cfg.model_id
                })

            qwen_latent_preds, t_latent_qwen = evaluate_model_chunked_latent(
                qwen, prefix_qwen, args.max_new_tokens, args.chunk_size, "qwen", anchor_qw or None,
                min_new_tokens=args.min_new_tokens, eos_ban_steps=args.eos_ban_steps,
                first_token_top_p=args.first_token_top_p, first_token_temperature=args.first_token_temperature,
                append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes"))
            )
            if args.debug and args.debug_print_first > 0:
                print("\n[DEBUG] First generations (Qwen, latent):")
                for i, pred in enumerate(qwen_latent_preds[:args.debug_print_first]):
                    print(f"  {i}: '{pred}'")
            if args.debug and args.debug_topk > 0:
                try:
                    topkQ = qwen.peek_first_step_from_prefix(prefix_qwen[:args.debug_topk_examples], anchor_qw or None,
                                                             append_bos_after_prefix=(None if args.append_bos_after_prefix == "auto" else (args.append_bos_after_prefix == "yes")),
                                                             topk=args.debug_topk)
                    print("\n[DEBUG] First-step top-k (Qwen):")
                    for i, rows in enumerate(topkQ):
                        pretty = ", ".join([(tok.strip().replace('\n','\\n') or '<NL>') + f":{prob:.3f}" for _, prob, tok in rows])
                        print(f"  ex{i}: {pretty}")
                except Exception as e:
                    print(f"[DEBUG] Qwen top-k failed: {e}")

            k_budget = args.token_budget_k or latent_len
            qwen_trunc = build_token_budget_prompts(qwen.tokenizer, prompts_raw, qwen_chat, k_budget, args.token_budget_mode)
            qwen_trunc_preds, t_trunc_qwen = evaluate_model_chunked_text(qwen, qwen_trunc, args.max_new_tokens, args.chunk_size, "qwen")

            qwen_text_em, qwen_text_f1     = batch_metrics(qwen_text_preds, golds)
            qwen_latent_em, qwen_latent_f1 = batch_metrics(qwen_latent_preds, golds)
            qwen_trunc_em, qwen_trunc_f1   = batch_metrics(qwen_trunc_preds, golds)

            qwen_text_nll   = avg_nll_text(qwen, qwen_chat, golds, qwen.tokenizer, device)
            qwen_latent_nll = avg_nll_latent(qwen, prefix_qwen, golds, qwen.tokenizer, device, anchor_qw or None)

            Q = {
                "avg_prompt_tokens": avg_prompt_tokens_qwen,
                "text_preds": qwen_text_preds,
                "latent_preds": qwen_latent_preds,
                "trunc_preds": qwen_trunc_preds,
                "times": {"text": t_text_qwen, "latent": t_latent_qwen, "trunc": t_trunc_qwen},
                "metrics": {
                    "text": {"em": qwen_text_em, "f1": qwen_text_f1, "nll": qwen_text_nll},
                    "latent": {"em": qwen_latent_em, "f1": qwen_latent_f1, "nll": qwen_latent_nll},
                    "trunc": {"em": qwen_trunc_em, "f1": qwen_trunc_f1}
                },
                "chat_prompts": qwen_chat,
                "debug": debug_qwen,
            }
            if qwen_file:
                with open(qwen_file, "w") as f:
                    json.dump(Q, f)
                print(f"Saved Qwen results to {qwen_file}")

            del qwen, adp_qwen, prefix_qwen, qwen_prompt_tok, Z
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache(); torch.mps.synchronize(); time.sleep(0.5)

    # Joint section & summary
    if not (have_llama or have_qwen):
        raise RuntimeError("No adapters found (neither adapter_llama.pt nor adapter_qwen.pt).")

    joint_section = {"em": None, "f1": None, "agreement": 0.0, "oracle": {"em": None, "f1": None}}

    llama_avg_tok = float(L.get("avg_prompt_tokens", 0.0)) if have_llama else 0.0
    qwen_avg_tok  = float(Q.get("avg_prompt_tokens", 0.0))  if have_qwen  else 0.0

    wire = compute_wire_metrics(L.get("chat_prompts", []) if have_llama else [],
                                Q.get("chat_prompts", []) if have_qwen else [], Z_for_wire)

    bytes_per_latent = int(Z_for_wire.element_size() * Z_for_wire.size(1) * Z_for_wire.size(2))
    text_block = {"wall_clock_sec": 0.0}
    latent_block = {"wall_clock_sec": 0.0}
    token_budget_block = {"mode": args.token_budget_mode, "k": (args.token_budget_k or latent_len)}

    if have_llama:
        text_block["llama"] = {"em": L["metrics"]["text"]["em"], "f1": L["metrics"]["text"]["f1"], "nll_token": L["metrics"]["text"]["nll"]}
        latent_block["llama"] = {"em": L["metrics"]["latent"]["em"], "f1": L["metrics"]["latent"]["f1"], "nll_token": L["metrics"]["latent"]["nll"]}
        token_budget_block["llama"] = {"em": L["metrics"]["trunc"]["em"], "f1": L["metrics"]["trunc"]["f1"]}
        text_block["wall_clock_sec"] += L["times"]["text"]
        latent_block["wall_clock_sec"] += L["times"]["latent"]
        token_budget_block["wall_clock_sec"] = token_budget_block.get("wall_clock_sec", 0.0) + L["times"]["trunc"]

    if have_qwen:
        text_block["qwen"] = {"em": Q["metrics"]["text"]["em"], "f1": Q["metrics"]["text"]["f1"], "nll_token": Q["metrics"]["text"]["nll"]}
        latent_block["qwen"] = {"em": Q["metrics"]["latent"]["em"], "f1": Q["metrics"]["latent"]["f1"], "nll_token": Q["metrics"]["latent"]["nll"]}
        token_budget_block["qwen"] = {"em": Q["metrics"]["trunc"]["em"], "f1": Q["metrics"]["trunc"]["f1"]}
        text_block["wall_clock_sec"] += Q["times"]["text"]
        latent_block["wall_clock_sec"] += Q["times"]["latent"]
        token_budget_block["wall_clock_sec"] = token_budget_block.get("wall_clock_sec", 0.0) + Q["times"]["trunc"]

    compression = {}
    if have_llama: compression["llama"] = llama_avg_tok / latent_len
    if have_qwen:  compression["qwen"]  = qwen_avg_tok / latent_len

    # Anchored joint rescoring (sequential scoring)
    joint_em = joint_f1 = agreement_rate = None
    oracle_em = oracle_f1 = None
    if have_llama and have_qwen:
        print("\nJoint rescoring...")
        llama = LMWrapper(LMConfig(model_id=(L.get("debug", {}).get("model_id") or llama_id), device=device, dtype=dtype, load_4bit=args.load_4bit))
        adp_llama = Adapter(d_z=d_z, d_model=llama.d_model).to(device).eval()
        adp_llama.load_state_dict(_safe_load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device), strict=True)
        mode_L = L.get("debug", {}).get("encoder_text_mode", "raw")
        ZL = _compute_Z_for_mode(encoder_type, encoder, mode_L, llama, prompts_raw, device, byte_max)
        with torch.no_grad():
            prefixL = adp_llama(ZL)
            prefixL, _, _ = _calibrate_prefix(prefixL, llama, args.calibration, args.prefix_target_rms, train_stats, "llama")
            prefixL = prefixL * args.prefix_gain
        mode_ll, text_ll = infer_anchor_mode_and_text(llama, cfg, args.latent_anchor_mode, args.latent_anchor_text)
        anchor_ll = make_anchor_text(mode_ll, llama, text_ll)
        anchor_ids_llama = llama._encode_anchor_text(anchor_ll) if anchor_ll else None

        qwen = LMWrapper(LMConfig(model_id=(Q.get("debug", {}).get("model_id") or qwen_id), device=device, dtype=dtype, load_4bit=args.load_4bit))
        adp_qwen = Adapter(d_z=d_z, d_model=qwen.d_model).to(device).eval()
        adp_qwen.load_state_dict(_safe_load(os.path.join(args.ckpt, "adapter_qwen.pt"), map_location=device), strict=True)
        mode_Q = Q.get("debug", {}).get("encoder_text_mode", "raw")
        ZQ = _compute_Z_for_mode(encoder_type, encoder, mode_Q, qwen, prompts_raw, device, byte_max)
        with torch.no_grad():
            prefixQ = adp_qwen(ZQ)
            prefixQ, _, _ = _calibrate_prefix(prefixQ, qwen, args.calibration, args.prefix_target_rms, train_stats, "qwen")
            prefixQ = prefixQ * args.prefix_gain
        mode_qw, text_qw = infer_anchor_mode_and_text(qwen, cfg, args.latent_anchor_mode, args.latent_anchor_text)
        anchor_qw = make_anchor_text(mode_qw, qwen, text_qw)
        anchor_ids_qwen = qwen._encode_anchor_text(anchor_qw) if anchor_qw else None

        joint_preds, agree = [], 0
        for i in range(len(prompts_raw)):
            candA = L["latent_preds"][i]
            candB = Q["latent_preds"][i]
            if _normalize(candA) == _normalize(candB):
                agree += 1
            A_ids_L = _to_long(llama.tokenizer(candA, return_tensors="pt", add_special_tokens=True).input_ids, device)
            A_ids_Q = _to_long(qwen.tokenizer(candA,  return_tensors="pt", add_special_tokens=True).input_ids, device)
            B_ids_L = _to_long(llama.tokenizer(candB, return_tensors="pt", add_special_tokens=True).input_ids, device)
            B_ids_Q = _to_long(qwen.tokenizer(candB,  return_tensors="pt", add_special_tokens=True).input_ids, device)
            scoreA = llama.score_prefix_logprob(prefixL[i:i+1], A_ids_L, anchor_token_ids=anchor_ids_llama) + \
                     qwen.score_prefix_logprob(prefixQ[i:i+1],  A_ids_Q, anchor_token_ids=anchor_ids_qwen)
            scoreB = llama.score_prefix_logprob(prefixL[i:i+1], B_ids_L, anchor_token_ids=anchor_ids_llama) + \
                     qwen.score_prefix_logprob(prefixQ[i:i+1],  B_ids_Q, anchor_token_ids=anchor_ids_qwen)
            joint_preds.append(candA if scoreA >= scoreB else candB)

        joint_em, joint_f1 = batch_metrics(joint_preds, golds)
        agreement_rate = agree / len(prompts_raw)

        # Oracle bound
        oracle_em = 0.0
        oracle_f1 = 0.0
        for pA, pB, g in zip(L.get("latent_preds", []), Q.get("latent_preds", []), golds):
            oracle_em += max(em(pA, g), em(pB, g))
            oracle_f1 += max(f1(pA, g), f1(pB, g))
        oracle_em /= len(golds)
        oracle_f1 /= len(golds)

        del llama, qwen, adp_llama, adp_qwen, prefixL, prefixQ, ZL, ZQ
        if device == "cuda":
            torch.cuda.empty_cache()

        joint_section = {"em": joint_em, "f1": joint_f1, "agreement": agreement_rate, "oracle": {"em": None, "f1": None}}

    summary = {
        "samples": len(prompts_raw),
        "max_new_tokens": args.max_new_tokens,
        "latent_len": latent_len,
        "device": device,
        "dtype": str(dtype),
        "avg_prompt_tokens": {"llama": llama_avg_tok if have_llama else 0.0, "qwen": qwen_avg_tok if have_qwen else 0.0},
        "compression": compression,
        "payload_bytes": bytes_per_latent,
        "wire": wire,
        "text": text_block,
        "latent": latent_block,
        "token_budget": token_budget_block,
        "joint": joint_section,
        "debug": {
            "llama": L.get("debug", {}) if have_llama else {},
            "qwen":  Q.get("debug", {}) if have_qwen else {},
            "latent_anchor_mode": args.latent_anchor_mode,
            "latent_anchor_text": L.get("debug", {}).get("latent_anchor_text","") if have_llama else (Q.get("debug", {}).get("latent_anchor_text","") if have_qwen else ""),
            "prefix_gain": args.prefix_gain,
            "calibration_mode": args.calibration,
            "append_bos_after_prefix": args.append_bos_after_prefix,
            "decode": {
                "min_new_tokens": args.min_new_tokens,
                "eos_ban_steps": args.eos_ban_steps,
                "first_token_top_p": args.first_token_top_p,
                "first_token_temperature": args.first_token_temperature
            }
        },
        "oracle": {"em": oracle_em, "f1": oracle_f1},
    }

    preds_dump = []
    llama_chat = L.get("chat_prompts", []) if have_llama else ["" for _ in range(len(prompts_raw))]
    qwen_chat  = Q.get("chat_prompts", []) if have_qwen  else ["" for _ in range(len(prompts_raw))]
    llama_text = L.get("text_preds",   [""]*len(prompts_raw)) if have_llama else ["" for _ in range(len(prompts_raw))]
    qwen_text  = Q.get("text_preds",   [""]*len(prompts_raw)) if have_qwen  else ["" for _ in range(len(prompts_raw))]
    llama_trunc= L.get("trunc_preds",  [""]*len(prompts_raw)) if have_llama else ["" for _ in range(len(prompts_raw))]
    qwen_trunc = Q.get("trunc_preds",  [""]*len(prompts_raw)) if have_qwen  else ["" for _ in range(len(prompts_raw))]
    llama_lat  = L.get("latent_preds", [""]*len(prompts_raw)) if have_llama else ["" for _ in range(len(prompts_raw))]
    qwen_lat   = Q.get("latent_preds", [""]*len(prompts_raw)) if have_qwen  else ["" for _ in range(len(prompts_raw))]

    for i in range(len(prompts_raw)):
        preds_dump.append({
            "prompt_raw": prompts_raw[i],
            "prompt_llama_chat": llama_chat[i] if i < len(llama_chat) else "",
            "prompt_qwen_chat":  qwen_chat[i]  if i < len(qwen_chat) else "",
            "gold": golds[i],
            "text_pred_llama":   llama_text[i],
            "text_pred_qwen":    qwen_text[i],
            "latent_pred_llama": llama_lat[i],
            "latent_pred_qwen":  qwen_lat[i],
            "trunc_pred_llama":  llama_trunc[i],
            "trunc_pred_qwen":   qwen_trunc[i],
        })

    return summary, preds_dump

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--llama_id", type=str, default=None)
    ap.add_argument("--qwen_id", type=str, default=None)
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot","squad","squad_v2"])
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--hotpot_config", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--token_budget_mode", type=str, default="content_only", choices=["chat_full", "content_only"])
    ap.add_argument("--token_budget_k", type=int, default=None)

    # Anchors / BOS controls
    ap.add_argument("--latent_anchor_mode", type=str, default="auto", choices=["auto","chat","text","none"],
                    help="What to use as latent anchor. 'auto' matches training cfg (warm_anchor_text => text; else chat).")
    ap.add_argument("--latent_anchor_text", type=str, default="Answer: ",
                    help="Used when --latent_anchor_mode=text (or as a hint for 'auto' if training cfg has none).")
    ap.add_argument("--append_bos_after_prefix", type=str, default="auto", choices=["auto","yes","no"],
                    help="Append BOS after latent prefix (+anchor). 'auto' = only when no anchor is provided.")

    ap.add_argument("--sequential_eval", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--fresh_eval", action="store_true")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_print_first", type=int, default=0)
    ap.add_argument("--debug_topk", type=int, default=0)
    ap.add_argument("--debug_topk_examples", type=int, default=2)

    # new decode controls (latent)
    ap.add_argument("--min_new_tokens", type=int, default=3)
    ap.add_argument("--eos_ban_steps", type=int, default=6)
    ap.add_argument("--first_token_top_p", type=float, default=1.0)
    ap.add_argument("--first_token_temperature", type=float, default=0.0)

    # eval-time amplitude control
    ap.add_argument("--prefix_gain", type=float, default=1.0, help="Multiply adapter output by this gain at eval (does not change weights)")

    # calibration modes
    ap.add_argument("--calibration", type=str, default="embed_rms", choices=["none","embed_rms","fixed","train_stats"])
    ap.add_argument("--prefix_target_rms", type=float, default=None)

    # encoder input alignment
    ap.add_argument("--encoder_text_mode", type=str, default="auto",
                    choices=["auto","raw","neutral_chat","llama_chat","qwen_chat"])

    args = ap.parse_args()

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
    with open(os.path.join(args.ckpt, "config.json")) as f:
        cfg = json.load(f)

    encoder_type = cfg.get("encoder_type", "byte")
    trained_used_neutral = bool(cfg.get("encoder_use_chat_template", False))
    if (not args.sequential_eval) and (not args.fresh_eval) and encoder_type.startswith("simple") and trained_used_neutral:
        print("⚠️  Detected SimpleEncoder trained with a chat-style wrapper, but Standard eval is about to reuse a cached Z.pt.")
        print("    Add --fresh_eval (or use --sequential_eval) so Z is recomputed with the correct wrapper.")

    llama_id = args.llama_id or cfg["llama_id"]
    qwen_id  = args.qwen_id  or cfg["qwen_id"]
    latent_len = int(cfg["latent_len"])
    d_z = int(cfg["d_z"])
    byte_max = cfg.get("byte_max", 512)

    # Try to load training-time prefix stats (optional)
    train_stats_path = os.path.join(args.ckpt, "training_stats.json")
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

    # Build a Z for wire metrics
    Z_path = os.path.join(args.out_dir, "Z.pt") if args.out_dir else None
    if Z_path and (not args.fresh_eval) and os.path.exists(Z_path):
        print(f"Loading Z from {Z_path}")
        Z_for_wire = torch.load(Z_path, map_location=device)
    else:
        print("Building encoder and computing Z...")
        if encoder_type == "byte":
            encoder_wire = InterlinguaEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
            encoder_wire.load_state_dict(_safe_load(os.path.join(args.ckpt, "encoder.pt"), map_location=device))
            with torch.no_grad():
                byte_tok = ByteTokenizer(max_bytes=byte_max)
                z_bytes = collate_bytes(prompts_raw, byte_tok, device)
                Z_for_wire = encoder_wire(z_bytes)
        else:
            encoder_wire = SimpleEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
            encoder_wire.load_state_dict(_safe_load(os.path.join(args.ckpt, "encoder.pt"), map_location=device))
            with torch.no_grad():
                Z_for_wire = encoder_wire(prompts_raw)
        if Z_path:
            torch.save(Z_for_wire.to("cpu"), Z_path)
            print(f"Saved Z to {Z_path}")

    # === EVALUATION MODES ===
    if args.sequential_eval:
        summary, preds_dump = run_sequential_eval(
            args=args, device=device, dtype=dtype,
            Z_for_wire=Z_for_wire, prompts_raw=prompts_raw, golds=golds,
            llama_id=llama_id, qwen_id=qwen_id,
            latent_len=latent_len, d_z=d_z, encoder_type=encoder_type, byte_max=byte_max,
            cfg=cfg, train_stats=train_stats
        )
    else:
        summary, preds_dump = run_standard_eval(
            args=args, device=device, dtype=dtype,
            Z=Z_for_wire, prompts_raw=prompts_raw, golds=golds,
            llama_id=llama_id, qwen_id=qwen_id,
            latent_len=latent_len, d_z=d_z, cfg=cfg, train_stats=train_stats
        )

    # Tag dataset in the summary
    summary["dataset"] = args.dataset
    if args.dataset == "hotpot":
        summary["dataset_detail"] = {"config": (args.hotpot_config or "fullwiki")}

    print("\n==== LatentWire Evaluation ====")
    print(f"Dataset: {dataset_detail}")
    print(f"Samples: {summary['samples']}  |  Max new tokens: {summary['max_new_tokens']}")
    print(f"Device: {summary['device']}  |  Dtype: {summary['dtype'].split('.')[-1] if isinstance(summary['dtype'], str) else 'unknown'}")
    print(f"Avg prompt tokens (Llama): {summary['avg_prompt_tokens'].get('llama','-'):.1f} | "
          f"(Qwen): {summary['avg_prompt_tokens'].get('qwen','-'):.1f} | Latent length M: {summary['latent_len']}")
    print(f"Compression ratio (Llama): {summary['compression'].get('llama','-'):.1f}x | "
          f"(Qwen): {summary['compression'].get('qwen','-'):.1f}x")
    print(f"Approx interlingua payload per example: {summary['payload_bytes']} bytes (fp32), "
          f"and {summary['wire']['latent_bytes']['fp16']} bytes (fp16); "
          f"wire compression vs one-copy text (fp16): {summary['wire']['wire_compression']['vs_onecopy_fp16']:.2f}x")

    print("\n— Baseline: Text prompting")
    if summary['text'].get('llama') is not None:
        print(f"Llama  EM: {summary['text']['llama']['em']:.3f}  F1: {summary['text']['llama']['f1']:.3f}  |  NLL/token (gold): {summary['text']['llama']['nll_token']}")
    if summary['text'].get('qwen') is not None:
        print(f"Qwen   EM: {summary['text']['qwen']['em']:.3f}   F1: {summary['text']['qwen']['f1']:.3f}   |  NLL/token (gold): {summary['text']['qwen']['nll_token']}")
    print(f"Wall clock: {summary['text']['wall_clock_sec']:.2f}s")

    print("\n— Latent prompting (shared interlingua)")
    if summary['latent'].get('llama') is not None:
        print(f"Llama  EM: {summary['latent']['llama']['em']:.3f}  F1: {summary['latent']['llama']['f1']:.3f}  |  NLL/token (gold): {summary['latent']['llama']['nll_token']}")
    if summary['latent'].get('qwen') is not None:
        print(f"Qwen   EM: {summary['latent']['qwen']['em']:.3f}   F1: {summary['latent']['qwen']['f1']:.3f}   |  NLL/token (gold): {summary['latent']['qwen']['nll_token']}")
    print(f"Wall clock: {summary['latent']['wall_clock_sec']:.2f}s")

    print(f"\n— Token-budget baseline (mode: {summary['token_budget'].get('mode','content_only')})")
    if summary['token_budget'].get('llama') is not None:
        print(f"Llama  EM: {summary['token_budget']['llama']['em']:.3f}  F1: {summary['token_budget']['llama']['f1']:.3f}")
    if summary['token_budget'].get('qwen') is not None:
        print(f"Qwen   EM: {summary['token_budget']['qwen']['em']:.3f}   F1: {summary['token_budget']['qwen']['f1']:.3f}")
    print(f"Wall clock: {summary['token_budget']['wall_clock_sec']:.2f}s")

    print("\n— 2-LLM joint (rescored pick on latent runs)")
    if summary['joint']['em'] is not None:
        print(f"Joint  EM: {summary['joint']['em']:.3f}  F1: {summary['joint']['f1']:.3f}")
        print(f"Inter-model agreement (normalized): {summary['joint']['agreement']:.3f}")
        print(f"Oracle upper bound:  EM {summary['oracle']['em']:.3f}  F1 {summary['oracle']['f1']:.3f}")

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
            if summary["joint"]["em"] is not None:
                w.writerow(["joint","both", summary["joint"]["em"], summary["joint"]["f1"], "", "", "", summary["payload_bytes"],
                            summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])

        dump_path = os.path.join(args.out_dir, "predictions.jsonl")
        with open(dump_path, "w") as f:
            for rec in preds_dump:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote per-example predictions to {dump_path}")

if __name__ == "__main__":
    main()
