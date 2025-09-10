# latentwire/eval.py
import os
import time
import json
import argparse
import gc
import re
from typing import List, Dict, Any, Tuple, Optional

import torch

from latentwire.models import (
    InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
)
from latentwire.data import load_examples
from latentwire.metrics import batch_metrics, _normalize, em, f1


SYSTEM_PROMPT = "You are a concise QA assistant. Use the context to answer with a short phrase only. Answer in English. Respond with the answer phrase only."
STOP_STRINGS = ["<|eot_id|>", "<|im_end|>", "</s>", "\n\n\n", "\n\nAssistant:", "\nAssistant:"]

def safe_clean_pred(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    for ss in STOP_STRINGS:
        idx = s.find(ss)
        if idx >= 0:
            s = s[:idx]
    s = re.sub(r"^\s*(assistant|assistant:|Assistant:)\s*", "", s)
    lines = [ln for ln in s.splitlines() if ln.strip() != ""]
    if not lines:
        return ""
    s = lines[0]
    return s.strip(" \t\r\n.:;,'\"-–—")

def decode_ids_list(tokenizer, batch_ids: List[List[int]]) -> List[str]:
    outs: List[str] = []
    for ids in batch_ids:
        try:
            text = tokenizer.decode(ids or [], skip_special_tokens=True)
        except Exception:
            text = ""
        outs.append(safe_clean_pred(text))
    return outs


# ---------- utilities ----------

def _to_long(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device, dtype=torch.long)

def _human_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60); s = int(sec % 60)
    return f"{m}m{s:02d}s"

def _ensure_dir(p: Optional[str]):
    if p:
        os.makedirs(p, exist_ok=True)

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
    else:
        return [content_only_m_token_chat_prompt(tokenizer, s, k) for s in raw_sources]

def collate_bytes(texts: List[str], byte_tok: ByteTokenizer, device: str):
    ids = [byte_tok.encode(t) for t in texts]
    maxT = max([x.size(0) for x in ids])
    batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.size(0), dtype=torch.long)], dim=0) for x in ids], dim=0)
    return batch.to(device)


# ---------- generation helpers ----------

@torch.no_grad()
def generate_text_baseline(wrapper: LMWrapper, prompts: List[str], max_new_tokens: int, min_new_tokens: int, eos_ban_steps: int) -> List[str]:
    out_ids = wrapper.generate_from_text(
        prompts, max_new_tokens=max_new_tokens, temperature=0.0,
        min_new_tokens=min_new_tokens, eos_ban_steps=eos_ban_steps
    )
    return decode_ids_list(wrapper.tokenizer, out_ids)

@torch.no_grad()
def generate_latent(wrapper: LMWrapper, prefix_embeds: torch.Tensor, max_new_tokens: int, anchor: Optional[str], min_new_tokens: int, eos_ban_steps: int) -> List[str]:
    out_ids = wrapper.generate_from_prefix(
        prefix_embeds, max_new_tokens=max_new_tokens, temperature=0.0,
        anchor_token_text=anchor, min_new_tokens=min_new_tokens, eos_ban_steps=eos_ban_steps
    )
    return decode_ids_list(wrapper.tokenizer, out_ids)


# ---------- loss/NLL ----------

def avg_nll_latent(wrapper: LMWrapper, prefix: torch.Tensor, answers: List[str], tokenizer, device: str) -> float:
    tot_nll, tot_tok = 0.0, 0
    for i, a in enumerate(answers):
        a_ids = _to_long(tokenizer(a, return_tensors="pt", add_special_tokens=True).input_ids, device)
        loss = wrapper.forward_with_prefix_loss(prefix[i:i+1], a_ids)
        n_tok = a_ids.size(1) - 1
        tot_nll += float(loss.item()) * n_tok
        tot_tok += int(n_tok)
    return tot_nll / max(1, tot_tok)

def avg_nll_text(wrapper: LMWrapper, prompts_text: List[str], answers: List[str], tokenizer, device: str) -> float:
    tot_nll, tot_tok = 0.0, 0
    for i in range(len(prompts_text)):
        enc_p = _to_long(tokenizer(prompts_text[i], return_tensors="pt", add_special_tokens=True).input_ids, device)
        enc_a = _to_long(tokenizer(answers[i],      return_tensors="pt", add_special_tokens=True).input_ids, device)
        loss, n_tok = wrapper.loss_with_text_prompt(enc_p, enc_a)
        tot_nll += float(loss.item()) * n_tok
        tot_tok += int(n_tok)
    return tot_nll / max(1, tot_tok)


# ---------- debug ----------

def _latent_debug_stats(name: str, Z: torch.Tensor, prefix: torch.Tensor, adapter: Adapter) -> Dict[str, float]:
    with torch.no_grad():
        z_std = float(Z.std().item())
        z_mean_norm = float(Z.norm(dim=-1).mean().item())
        p_std = float(prefix.std().item())
        p_mean_norm = float(prefix.norm(dim=-1).mean().item())
        try:
            scale = float(adapter.scale.detach().cpu().item())
        except Exception:
            scale = float('nan')
    print(f"[debug:{name}] adapter.scale={scale:.4f} | Z.std={z_std:.4f} Z.mean||={z_mean_norm:.4f} | "
          f"prefix.std={p_std:.4f} prefix.mean||={p_mean_norm:.4f}")
    return {
        "adapter_scale": scale,
        "Z_std": z_std,
        "Z_mean_norm": z_mean_norm,
        "prefix_std": p_std,
        "prefix_mean_norm": p_mean_norm,
    }


# ---------- chunked evaluators ----------

def evaluate_model_chunked_text(wrapper: LMWrapper, chat_prompts: List[str], max_new_tokens: int,
                                chunk_size: int, log_prefix: str, min_new_tokens: int, eos_ban_steps: int) -> Tuple[List[str], float]:
    preds = []
    t0 = time.time()
    for i in range(0, len(chat_prompts), chunk_size):
        batch = chat_prompts[i:i+chunk_size]
        preds.extend(generate_text_baseline(wrapper, batch, max_new_tokens, min_new_tokens, eos_ban_steps))
        done = i + len(batch)
        elapsed = time.time() - t0
        rate = done / max(1e-9, elapsed)
        remain = (len(chat_prompts) - done) / max(1e-9, rate)
        print(f"  [{log_prefix}] text: {done}/{len(chat_prompts)} | {rate:.2f} ex/s | elapsed={_human_time(elapsed)} | eta={_human_time(remain)}")
    return preds, time.time() - t0

def evaluate_model_chunked_latent(wrapper: LMWrapper, prefix: torch.Tensor, max_new_tokens: int,
                                  chunk_size: int, log_prefix: str, anchor: Optional[str], min_new_tokens: int, eos_ban_steps: int) -> Tuple[List[str], float]:
    preds = []
    B = prefix.size(0)
    t0 = time.time()
    for i in range(0, B, chunk_size):
        batch = prefix[i:i+chunk_size]
        preds.extend(generate_latent(wrapper, batch, max_new_tokens, anchor, min_new_tokens, eos_ban_steps))
        done = i + batch.size(0)
        elapsed = time.time() - t0
        rate = done / max(1e-9, elapsed)
        remain = (B - done) / max(1e-9, rate)
        print(f"  [{log_prefix}] latent: {done}/{B} | {rate:.2f} ex/s | elapsed={_human_time(elapsed)} | eta={_human_time(remain)}")
    return preds, time.time() - t0


# ---------- wire metrics ----------

def compute_wire_metrics(llama_chat_prompts: List[str], qwen_chat_prompts: List[str], Z: torch.Tensor) -> Dict[str, Any]:
    avg_text_bytes_llama = int(sum(len(p.encode("utf-8")) for p in llama_chat_prompts) / max(1, len(llama_chat_prompts)))
    avg_text_bytes_qwen  = int(sum(len(p.encode("utf-8")) for p in qwen_chat_prompts)  / max(1, len(qwen_chat_prompts)))
    bytes_fp32 = int(Z.size(1) * Z.size(2) * 4)
    bytes_fp16 = int(Z.size(1) * Z.size(2) * 2)
    return {
        "text_bytes_onecopy": {
            "llama_avg": avg_text_bytes_llama,
            "qwen_avg": avg_text_bytes_qwen,
            "max_avg": max(avg_text_bytes_llama, avg_text_bytes_qwen)
        },
        "text_bytes_twocopies": {
            "sum_avg": avg_text_bytes_llama + avg_text_bytes_qwen
        },
        "latent_bytes": {"fp32": bytes_fp32, "fp16": bytes_fp16}
    }


# ---------- standard eval ----------

def run_standard_eval(args, device, dtype, Z, prompts_raw, golds, llama_id, qwen_id, latent_len, d_z):
    print("\n[Standard Evaluation Mode - both models loaded]")

    llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))

    adp_llama = Adapter(d_z=d_z, d_model=llama.d_model).to(device).eval()
    adp_qwen  = Adapter(d_z=d_z, d_model=qwen.d_model).to(device).eval()
    adp_llama.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device), strict=False)
    adp_qwen.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_qwen.pt"),  map_location=device), strict=False)

    # Calibrate scales to match embedding norm using this Z batch
    Z_cal = Z[: min(128, Z.size(0))].to(device)
    llama.calibrate_adapter(adp_llama, Z_sample=Z_cal)
    qwen.calibrate_adapter(adp_qwen,  Z_sample=Z_cal)

    llama_chat = build_chat_prompts(llama.tokenizer, prompts_raw)
    qwen_chat  = build_chat_prompts(qwen.tokenizer,  prompts_raw)

    llama_prompt_tok = llama.tokenizer(llama_chat, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    qwen_prompt_tok  = qwen.tokenizer(qwen_chat,  return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    avg_prompt_tokens_llama = float((llama_prompt_tok["input_ids"] != llama.tokenizer.pad_token_id).sum().item()) / len(prompts_raw)
    avg_prompt_tokens_qwen  = float((qwen_prompt_tok["input_ids"]  != qwen.tokenizer.pad_token_id).sum().item())  / len(prompts_raw)

    llama_text_preds, t_text_llama = evaluate_model_chunked_text(llama, llama_chat, args.max_new_tokens, args.chunk_size, "llama", args.min_new_tokens, args.eos_ban_steps)
    qwen_text_preds,  t_text_qwen  = evaluate_model_chunked_text(qwen,  qwen_chat,  args.max_new_tokens, args.chunk_size, "qwen",  args.min_new_tokens, args.eos_ban_steps)
    t_text = t_text_llama + t_text_qwen
    llama_text_em, llama_text_f1 = batch_metrics(llama_text_preds, golds)
    qwen_text_em,  qwen_text_f1  = batch_metrics(qwen_text_preds, golds)

    with torch.no_grad():
        prefix_llama = adp_llama(Z.to(device))
        prefix_qwen  = adp_qwen(Z.to(device))

    debug_llama = _latent_debug_stats("llama", Z, prefix_llama, adp_llama) if args.debug else {}
    debug_qwen  = _latent_debug_stats("qwen",  Z, prefix_qwen,  adp_qwen)  if args.debug else {}

    llama_latent_preds, t_latent_llama = evaluate_model_chunked_latent(
        llama, prefix_llama, args.max_new_tokens, args.chunk_size, "llama", args.latent_anchor_text or None, args.min_new_tokens, args.eos_ban_steps
    )
    qwen_latent_preds,  t_latent_qwen  = evaluate_model_chunked_latent(
        qwen,  prefix_qwen,  args.max_new_tokens, args.chunk_size, "qwen",  args.latent_anchor_text or None, args.min_new_tokens, args.eos_ban_steps
    )
    t_latent = t_latent_llama + t_latent_qwen

    llama_latent_em, llama_latent_f1 = batch_metrics(llama_latent_preds, golds)
    qwen_latent_em,  qwen_latent_f1  = batch_metrics(qwen_latent_preds, golds)

    k_budget = args.token_budget_k or latent_len
    llama_trunc = build_token_budget_prompts(llama.tokenizer, prompts_raw, llama_chat, k_budget, args.token_budget_mode)
    qwen_trunc  = build_token_budget_prompts(qwen.tokenizer,  prompts_raw, qwen_chat,  k_budget, args.token_budget_mode)

    llama_trunc_preds, t_trunc_llama = evaluate_model_chunked_text(llama, llama_trunc, args.max_new_tokens, args.chunk_size, "llama", args.min_new_tokens, args.eos_ban_steps)
    qwen_trunc_preds,  t_trunc_qwen  = evaluate_model_chunked_text(qwen,  qwen_trunc,  args.max_new_tokens, args.chunk_size, "qwen",  args.min_new_tokens, args.eos_ban_steps)
    t_trunc = t_trunc_llama + t_trunc_qwen

    llama_trunc_em, llama_trunc_f1 = batch_metrics(llama_trunc_preds, golds)
    qwen_trunc_em,  qwen_trunc_f1  = batch_metrics(qwen_trunc_preds,  golds)

    llama_latent_nll = avg_nll_latent(llama, prefix_llama, golds, llama.tokenizer, device)
    qwen_latent_nll  = avg_nll_latent(qwen,  prefix_qwen,  golds, qwen.tokenizer, device)
    llama_text_nll   = avg_nll_text(llama, llama_chat, golds, llama.tokenizer, device)
    qwen_text_nll    = avg_nll_text(qwen,  qwen_chat,  golds, qwen.tokenizer, device)

    joint_preds = []
    agree = 0
    for i in range(len(prompts_raw)):
        candA = llama_latent_preds[i]
        candB = qwen_latent_preds[i]

        A_ids_L = _to_long(llama.tokenizer(candA, return_tensors="pt", add_special_tokens=True).input_ids, device)
        A_ids_Q = _to_long(qwen.tokenizer(candA,  return_tensors="pt", add_special_tokens=True).input_ids, device)
        B_ids_L = _to_long(llama.tokenizer(candB, return_tensors="pt", add_special_tokens=True).input_ids, device)
        B_ids_Q = _to_long(qwen.tokenizer(candB,  return_tensors="pt", add_special_tokens=True).input_ids, device)

        scoreA = llama.score_prefix_logprob(prefix_llama[i:i+1], A_ids_L) + qwen.score_prefix_logprob(prefix_qwen[i:i+1], A_ids_Q)
        scoreB = llama.score_prefix_logprob(prefix_llama[i:i+1], B_ids_L) + qwen.score_prefix_logprob(prefix_qwen[i:i+1], B_ids_Q)

        pick = candA if scoreA >= scoreB else candB
        joint_preds.append(pick)
        if _normalize(candA) == _normalize(candB):
            agree += 1

    joint_em, joint_f1 = batch_metrics(joint_preds, golds)
    agreement_rate = agree / len(prompts_raw)
    wire = compute_wire_metrics(llama_chat, qwen_chat, Z)

    oracle_em = 0.0
    oracle_f1 = 0.0
    for pA, pB, g in zip(llama_latent_preds, qwen_latent_preds, golds):
        oracle_em += max(em(pA, g), em(pB, g))
        oracle_f1 += max(f1(pA, g), f1(pB, g))
    oracle_em /= len(golds)
    oracle_f1 /= len(golds)

    bytes_per_latent = int(Z.element_size() * Z.size(1) * Z.size(2))
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
        "joint": {"em": joint_em, "f1": joint_f1, "agreement": agreement_rate, "oracle": {"em": None, "f1": None}},
        "debug": {
            "llama": debug_llama,
            "qwen": debug_qwen,
            "latent_anchor_text": args.latent_anchor_text or "",
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


# ---------- sequential eval ----------

def run_sequential_eval(args, device, dtype, Z, prompts_raw, golds, llama_id, qwen_id, latent_len, d_z):
    print("\n[Sequential Evaluation Mode - one model at a time]")

    llama_file = os.path.join(args.out_dir, "llama_results.json") if args.out_dir else None
    qwen_file  = os.path.join(args.out_dir, "qwen_results.json") if args.out_dir else None

    # Phase A: Llama
    if (not args.fresh_eval) and llama_file and os.path.exists(llama_file):
        print(f"Loading cached Llama results: {llama_file}")
        with open(llama_file, "r") as f:
            L = json.load(f)
    else:
        print("\nEvaluating Llama...")
        llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
        adp_llama = Adapter(d_z=d_z, d_model=llama.d_model).to(device).eval()
        adp_llama.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device), strict=False)

        # Calibrate
        Z_cal = Z[: min(128, Z.size(0))].to(device)
        llama.calibrate_adapter(adp_llama, Z_sample=Z_cal)

        llama_chat = build_chat_prompts(llama.tokenizer, prompts_raw)
        llama_prompt_tok = llama.tokenizer(llama_chat, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        avg_prompt_tokens_llama = float((llama_prompt_tok["input_ids"] != llama.tokenizer.pad_token_id).sum().item()) / len(prompts_raw)

        llama_text_preds, t_text_llama = evaluate_model_chunked_text(llama, llama_chat, args.max_new_tokens, args.chunk_size, "llama", args.min_new_tokens, args.eos_ban_steps)

        with torch.no_grad():
            prefix_llama = adp_llama(Z.to(device))

        debug_llama = _latent_debug_stats("llama", Z, prefix_llama, adp_llama) if args.debug else {}

        llama_latent_preds, t_latent_llama = evaluate_model_chunked_latent(
            llama, prefix_llama, args.max_new_tokens, args.chunk_size, "llama", args.latent_anchor_text or None, args.min_new_tokens, args.eos_ban_steps
        )

        k_budget = args.token_budget_k or latent_len
        llama_trunc = build_token_budget_prompts(llama.tokenizer, prompts_raw, llama_chat, k_budget, args.token_budget_mode)
        llama_trunc_preds, t_trunc_llama = evaluate_model_chunked_text(llama, llama_trunc, args.max_new_tokens, args.chunk_size, "llama", args.min_new_tokens, args.eos_ban_steps)

        llama_text_em, llama_text_f1     = batch_metrics(llama_text_preds, golds)
        llama_latent_em, llama_latent_f1 = batch_metrics(llama_latent_preds, golds)
        llama_trunc_em, llama_trunc_f1   = batch_metrics(llama_trunc_preds, golds)

        llama_text_nll   = avg_nll_text(llama, llama_chat, golds, llama.tokenizer, device)
        llama_latent_nll = avg_nll_latent(llama, prefix_llama, golds, llama.tokenizer, device)

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

        del llama, adp_llama, prefix_llama, llama_prompt_tok
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            time.sleep(2)

    # Phase B: Qwen
    if (not args.fresh_eval) and qwen_file and os.path.exists(qwen_file):
        print(f"Loading cached Qwen results: {qwen_file}")
        with open(qwen_file, "r") as f:
            Q = json.load(f)
    else:
        print("\nEvaluating Qwen...")
        qwen = LMWrapper(LMConfig(model_id=qwen_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
        adp_qwen = Adapter(d_z=d_z, d_model=qwen.d_model).to(device).eval()
        adp_qwen.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_qwen.pt"), map_location=device), strict=False)

        Z_cal = Z[: min(128, Z.size(0))].to(device)
        qwen.calibrate_adapter(adp_qwen, Z_sample=Z_cal)

        qwen_chat = build_chat_prompts(qwen.tokenizer, prompts_raw)
        qwen_prompt_tok = qwen.tokenizer(qwen_chat, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        avg_prompt_tokens_qwen = float((qwen_prompt_tok["input_ids"] != qwen.tokenizer.pad_token_id).sum().item()) / len(prompts_raw)

        qwen_text_preds, t_text_qwen = evaluate_model_chunked_text(qwen, qwen_chat, args.max_new_tokens, args.chunk_size, "qwen", args.min_new_tokens, args.eos_ban_steps)

        with torch.no_grad():
            prefix_qwen = adp_qwen(Z.to(device))

        debug_qwen = _latent_debug_stats("qwen", Z, prefix_qwen, adp_qwen) if args.debug else {}

        qwen_latent_preds, t_latent_qwen = evaluate_model_chunked_latent(
            qwen, prefix_qwen, args.max_new_tokens, args.chunk_size, "qwen", args.latent_anchor_text or None, args.min_new_tokens, args.eos_ban_steps
        )

        k_budget = args.token_budget_k or latent_len
        qwen_trunc = build_token_budget_prompts(qwen.tokenizer, prompts_raw, qwen_chat, k_budget, args.token_budget_mode)
        qwen_trunc_preds, t_trunc_qwen = evaluate_model_chunked_text(qwen, qwen_trunc, args.max_new_tokens, args.chunk_size, "qwen", args.min_new_tokens, args.eos_ban_steps)

        qwen_text_em, qwen_text_f1     = batch_metrics(qwen_text_preds, golds)
        qwen_latent_em, qwen_latent_f1 = batch_metrics(qwen_latent_preds, golds)
        qwen_trunc_em, qwen_trunc_f1   = batch_metrics(qwen_trunc_preds, golds)

        qwen_text_nll   = avg_nll_text(qwen, qwen_chat, golds, qwen.tokenizer, device)
        qwen_latent_nll = avg_nll_latent(qwen, prefix_qwen, golds, qwen.tokenizer, device)

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

        del qwen, adp_qwen, prefix_qwen, qwen_prompt_tok
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            time.sleep(2)

    # Phase C: Joint rescoring
    print("\nJoint rescoring...")
    llama = LMWrapper(LMConfig(model_id=llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))
    adp_llama = Adapter(d_z=d_z, d_model=llama.d_model).to(device).eval()
    adp_qwen  = Adapter(d_z=d_z, d_model=qwen.d_model).to(device).eval()
    adp_llama.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_llama.pt"), map_location=device), strict=False)
    adp_qwen.load_state_dict(torch.load(os.path.join(args.ckpt, "adapter_qwen.pt"),  map_location=device), strict=False)

    Z_cal = Z[: min(128, Z.size(0))].to(device)
    llama.calibrate_adapter(adp_llama, Z_sample=Z_cal)
    qwen.calibrate_adapter(adp_qwen,  Z_sample=Z_cal)

    with torch.no_grad():
        prefix_llama = adp_llama(Z.to(device))
        prefix_qwen  = adp_qwen(Z.to(device))

    joint_preds = []
    agree = 0
    for i in range(len(prompts_raw)):
        candA = L["latent_preds"][i]
        candB = Q["latent_preds"][i]

        A_ids_L = _to_long(llama.tokenizer(candA, return_tensors="pt", add_special_tokens=True).input_ids, device)
        A_ids_Q = _to_long(qwen.tokenizer(candA,  return_tensors="pt", add_special_tokens=True).input_ids, device)
        B_ids_L = _to_long(llama.tokenizer(candB, return_tensors="pt", add_special_tokens=True).input_ids, device)
        B_ids_Q = _to_long(qwen.tokenizer(candB,  return_tensors="pt", add_special_tokens=True).input_ids, device)

        scoreA = llama.score_prefix_logprob(prefix_llama[i:i+1], A_ids_L) + qwen.score_prefix_logprob(prefix_qwen[i:i+1], A_ids_Q)
        scoreB = llama.score_prefix_logprob(prefix_llama[i:i+1], B_ids_L) + qwen.score_prefix_logprob(prefix_qwen[i:i+1], B_ids_Q)

        joint_preds.append(candA if scoreA >= scoreB else candB)
        if _normalize(candA) == _normalize(candB):
            agree += 1

    joint_em, joint_f1 = batch_metrics(joint_preds, golds)
    agreement_rate = agree / len(prompts_raw)

    wire = compute_wire_metrics(L["chat_prompts"], Q["chat_prompts"], Z)
    bytes_per_latent = int(Z.element_size() * Z.size(1) * Z.size(2))

    oracle_em = 0.0
    oracle_f1 = 0.0
    for pA, pB, g in zip(L["latent_preds"], Q["latent_preds"], golds):
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
        "avg_prompt_tokens": {"llama": L["avg_prompt_tokens"], "qwen": Q["avg_prompt_tokens"]},
        "compression": {"llama": L["avg_prompt_tokens"]/latent_len, "qwen": Q["avg_prompt_tokens"]/latent_len},
        "payload_bytes": bytes_per_latent,
        "wire": wire,
        "text": {
            "llama": {"em": L["metrics"]["text"]["em"], "f1": L["metrics"]["text"]["f1"], "nll_token": L["metrics"]["text"]["nll"]},
            "qwen":  {"em": Q["metrics"]["text"]["em"], "f1": Q["metrics"]["text"]["f1"], "nll_token": Q["metrics"]["text"]["nll"]},
            "wall_clock_sec": L["times"]["text"] + Q["times"]["text"],
        },
        "latent": {
            "llama": {"em": L["metrics"]["latent"]["em"], "f1": L["metrics"]["latent"]["f1"], "nll_token": L["metrics"]["latent"]["nll"]},
            "qwen":  {"em": Q["metrics"]["latent"]["em"], "f1": Q["metrics"]["latent"]["f1"], "nll_token": Q["metrics"]["latent"]["nll"]},
            "wall_clock_sec": L["times"]["latent"] + Q["times"]["latent"],
        },
        "token_budget": {
            "mode": args.token_budget_mode,
            "k": (args.token_budget_k or latent_len),
            "llama": {"em": L["metrics"]["trunc"]["em"], "f1": L["metrics"]["trunc"]["f1"]},
            "qwen":  {"em": Q["metrics"]["trunc"]["em"], "f1": Q["metrics"]["trunc"]["f1"]},
            "wall_clock_sec": L["times"]["trunc"] + Q["times"]["trunc"],
        },
        "joint": {"em": joint_em, "f1": joint_f1, "agreement": agreement_rate, "oracle": {"em": None, "f1": None}},
        "debug": {"llama": L.get("debug", {}), "qwen":  Q.get("debug", {}), "latent_anchor_text": args.latent_anchor_text or ""},
        "oracle": {"em": oracle_em, "f1": oracle_f1},
    }

    preds_dump = []
    llama_chat = L["chat_prompts"]; qwen_chat  = Q["chat_prompts"]
    llama_text = L["text_preds"];    qwen_text = Q["text_preds"]
    llama_trunc = L["trunc_preds"];  qwen_trunc = Q["trunc_preds"]
    for i in range(len(prompts_raw)):
        preds_dump.append({
            "prompt_raw": prompts_raw[i],
            "prompt_llama_chat": llama_chat[i],
            "prompt_qwen_chat":  qwen_chat[i],
            "gold": golds[i],
            "text_pred_llama":   llama_text[i],
            "text_pred_qwen":    qwen_text[i],
            "latent_pred_llama": L["latent_preds"][i],
            "latent_pred_qwen":  Q["latent_preds"][i],
            "trunc_pred_llama":  llama_trunc[i],
            "trunc_pred_qwen":   qwen_trunc[i],
        })

    del llama, qwen, adp_llama, adp_qwen
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return summary, preds_dump


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--llama_id", type=str, default=None)
    ap.add_argument("--qwen_id", type=str, default=None)
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot","squad","squad_v2"])
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--min_new_tokens", type=int, default=2)
    ap.add_argument("--eos_ban_steps", type=int, default=4)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda, mps, or cpu")
    ap.add_argument("--hotpot_config", type=str, default=None, help="HotpotQA config (fullwiki/distractor)")
    ap.add_argument("--out_dir", type=str, default=None, help="Write metrics/predictions here if set (enables resume)")
    ap.add_argument("--token_budget_mode", type=str, default="content_only", choices=["chat_full", "content_only"])
    ap.add_argument("--token_budget_k", type=int, default=None, help="Override token budget K; default: latent_len")
    ap.add_argument("--latent_anchor_text", type=str, default="", help="Optional warm-start text, e.g. 'Answer:' or 'Answer: '")
    ap.add_argument("--sequential_eval", action="store_true", help="Load/unload models sequentially to save memory")
    ap.add_argument("--chunk_size", type=int, default=8, help="Eval batch size")
    ap.add_argument("--fresh_eval", action="store_true", help="Ignore any resume files in out_dir and start fresh.")
    ap.add_argument("--debug", action="store_true", help="Print latent/adapter norm/var debug stats")
    args = ap.parse_args()

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

    with open(os.path.join(args.ckpt, "config.json")) as f:
        cfg = json.load(f)
    llama_id = args.llama_id or cfg["llama_id"]
    qwen_id  = args.qwen_id  or cfg["qwen_id"]
    latent_len = int(cfg["latent_len"])
    encoder_type = cfg.get("encoder_type", "byte")
    d_z = int(cfg["d_z"])

    if args.dataset.startswith("squad"):
        eval_examples = load_examples(dataset=args.dataset, split="validation", samples=args.samples, seed=42)
        dataset_detail = args.dataset
    else:
        eval_examples = load_examples(dataset="hotpot", split="validation", samples=args.samples, seed=42,
                                      config=(args.hotpot_config or "fullwiki"))
        dataset_detail = f"hotpot:{args.hotpot_config or 'fullwiki'}"

    prompts_raw = [e["source"] for e in eval_examples]
    golds       = [e["answer"] for e in eval_examples]

    Z_path = os.path.join(args.out_dir, "Z.pt") if args.out_dir else None
    if Z_path and (not args.fresh_eval) and os.path.exists(Z_path):
        print(f"Loading Z from {Z_path}")
        Z = torch.load(Z_path, map_location=device)
    else:
        print("Building encoder and computing Z...")
        if encoder_type == "byte":
            encoder = InterlinguaEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
            encoder.load_state_dict(torch.load(os.path.join(args.ckpt, "encoder.pt"), map_location=device))
            with torch.no_grad():
                byte_tok = ByteTokenizer(max_bytes=cfg["byte_max"])
                z_bytes = collate_bytes(prompts_raw, byte_tok, device)
                Z = encoder(z_bytes)
        else:
            encoder = SimpleEncoder(d_z=d_z, latent_len=latent_len).to(device).eval()
            encoder.load_state_dict(torch.load(os.path.join(args.ckpt, "encoder.pt"), map_location=device))
            with torch.no_grad():
                Z = encoder(prompts_raw)
        if Z_path:
            torch.save(Z.to("cpu"), Z_path)
            print(f"Saved Z to {Z_path}")

    if args.sequential_eval:
        summary, preds_dump = run_sequential_eval(
            args=args, device=device, dtype=dtype,
            Z=Z, prompts_raw=prompts_raw, golds=golds,
            llama_id=llama_id, qwen_id=qwen_id,
            latent_len=latent_len, d_z=d_z
        )
    else:
        summary, preds_dump = run_standard_eval(
            args=args, device=device, dtype=dtype,
            Z=Z, prompts_raw=prompts_raw, golds=golds,
            llama_id=llama_id, qwen_id=qwen_id,
            latent_len=latent_len, d_z=d_z
        )

    summary["dataset"] = args.dataset
    if args.dataset == "hotpot":
        summary["dataset_detail"] = {"config": (args.hotpot_config or "fullwiki")}

    print("\n==== LatentWire Evaluation ====")
    print(f"Dataset: {dataset_detail}")
    print(f"Samples: {summary['samples']}  |  Max new tokens: {summary['max_new_tokens']}")
    print(f"Device: {summary['device']}  |  Dtype: {summary['dtype']}")
    print(f"Avg prompt tokens (Llama): {summary['avg_prompt_tokens']['llama']:.1f} | (Qwen): {summary['avg_prompt_tokens']['qwen']:.1f} | Latent length M: {summary['latent_len']}")
    print(f"Compression ratio (Llama): {summary['compression']['llama']:.1f}x | (Qwen): {summary['compression']['qwen']:.1f}x")
    print(f"Approx interlingua payload per example: {summary['payload_bytes']} bytes (fp32), and {summary['wire']['latent_bytes']['fp16']} bytes (fp16)")

    print("\n— Baseline: Text prompting")
    print(f"Llama  EM: {summary['text']['llama']['em']:.3f}  F1: {summary['text']['llama']['f1']:.3f}  |  NLL/token (gold): {summary['text']['llama']['nll_token']:.3f}")
    print(f"Qwen   EM: {summary['text']['qwen']['em']:.3f}   F1: {summary['text']['qwen']['f1']:.3f}   |  NLL/token (gold): {summary['text']['qwen']['nll_token']:.3f}")
    print(f"Wall clock: {summary['text']['wall_clock_sec']:.2f}s")

    print("\n— Latent prompting (shared interlingua)")
    print(f"Llama  EM: {summary['latent']['llama']['em']:.3f}  F1: {summary['latent']['llama']['f1']:.3f}  |  NLL/token (gold): {summary['latent']['llama']['nll_token']:.3f}")
    print(f"Qwen   EM: {summary['latent']['qwen']['em']:.3f}   F1: {summary['latent']['qwen']['f1']:.3f}   |  NLL/token (gold): {summary['latent']['qwen']['nll_token']:.3f}")
    print(f"Wall clock: {summary['latent']['wall_clock_sec']:.2f}s")

    print(f"\n— Token-budget baseline (mode: {summary['token_budget'].get('mode','content_only')})")
    print(f"Llama  EM: {summary['token_budget']['llama']['em']:.3f}  F1: {summary['token_budget']['llama']['f1']:.3f}")
    print(f"Qwen   EM: {summary['token_budget']['qwen']['em']:.3f}   F1: {summary['token_budget']['qwen']['f1']:.3f}")
    print(f"Wall clock: {summary['token_budget']['wall_clock_sec']:.2f}s")

    print("\n— 2-LLM joint (rescored pick on latent runs)")
    print(f"Joint  EM: {summary['joint']['em']:.3f}  F1: {summary['joint']['f1']:.3f}")
    print(f"Inter-model agreement (normalized): {summary['joint']['agreement']:.3f}")
    print(f"Oracle upper bound:  EM {summary['oracle']['em']:.3f}  F1 {summary['oracle']['f1']:.3f}")

    print("\n==== METRICS_JSON ====")
    print(json.dumps(summary, indent=2))

    if args.out_dir:
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
        import csv as _csv
        with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["group","model","EM","F1","NLL/token","wall_clock_sec","compression","payload_bytes","samples","M","token_budget_mode","token_budget_k","dataset"])
            w.writerow(["text","llama", summary["text"]["llama"]["em"], summary["text"]["llama"]["f1"], summary["text"]["llama"]["nll_token"], summary["text"]["wall_clock_sec"], summary["compression"]["llama"], summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            w.writerow(["text","qwen",  summary["text"]["qwen"]["em"],  summary["text"]["qwen"]["f1"],  summary["text"]["qwen"]["nll_token"],  summary["text"]["wall_clock_sec"], summary["compression"]["qwen"],  summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            w.writerow(["latent","llama", summary["latent"]["llama"]["em"], summary["latent"]["llama"]["f1"], summary["latent"]["llama"]["nll_token"], summary["latent"]["wall_clock_sec"], summary["compression"]["llama"], summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            w.writerow(["latent","qwen",  summary["latent"]["qwen"]["em"],  summary["latent"]["qwen"]["f1"],  summary["latent"]["qwen"]["nll_token"],  summary["latent"]["wall_clock_sec"], summary["compression"]["qwen"],  summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            w.writerow(["token_budget","llama", summary["token_budget"]["llama"]["em"], summary["token_budget"]["llama"]["f1"], "", summary["token_budget"]["wall_clock_sec"], summary["compression"]["llama"], summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            w.writerow(["token_budget","qwen",  summary["token_budget"]["qwen"]["em"],  summary["token_budget"]["qwen"]["f1"],  "", summary["token_budget"]["wall_clock_sec"], summary["compression"]["qwen"],  summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])
            w.writerow(["joint","both", summary["joint"]["em"], summary["joint"]["f1"], "", "", "", summary["payload_bytes"], summary["samples"], summary["latent_len"], summary["token_budget"]["mode"], summary["token_budget"].get("k", None), args.dataset])

        dump_path = os.path.join(args.out_dir, "predictions.jsonl")
        with open(dump_path, "w") as f:
            for rec in preds_dump:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote per-example predictions to {dump_path}")


if __name__ == "__main__":
    main()
