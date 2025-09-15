# latentwire/train.py
import os
import re
import time
import json
import argparse
import random
from typing import Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim

from latentwire.models import (
    InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
)
from latentwire.checkpointing import save_latest_checkpoint
from latentwire.data import load_examples
from latentwire.common import collate_bytes  # deduped

DEFAULT_SEED = 42


@contextmanager
def _temp_padding_side(tokenizer, side: str):
    old = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = side
        yield
    finally:
        try:
            tokenizer.padding_side = old
        except Exception:
            pass

# ---------------------------
# Checkpoint helpers
# ---------------------------

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    if not save_dir:
        return None
    if os.path.isfile(save_dir):
        return save_dir
    if not os.path.isdir(save_dir):
        return None

    for name in ["state.pt", "last.pt"]:
        p = os.path.join(save_dir, name)
        if os.path.isfile(p):
            return p

    candidates = []
    for fn in os.listdir(save_dir):
        m = re.match(r"state_step(\d+)\.pt$", fn)
        if m:
            candidates.append((int(m.group(1)), os.path.join(save_dir, fn)))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


def _safe_load(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _maybe_to_device_optimizer_state(optimizer: optim.Optimizer, device: str):
    for p in optimizer.state.values():
        for k, v in p.items():
            if torch.is_tensor(v):
                p[k] = v.to(device)


def load_checkpoint(
    path: str,
    encoder: InterlinguaEncoder,
    adp_llama: Adapter,
    adp_qwen: Adapter,
    optimizer: Optional[optim.Optimizer] = None,
    strict: bool = True,
    device: str = "cpu"
) -> Tuple[int, int]:
    state = _safe_load(path, map_location="cpu") if path and os.path.isfile(path) else {}
    ckpt_dir = os.path.dirname(path) if path and os.path.isfile(path) else None

    enc_loaded = False
    if isinstance(state, dict) and all(k in state for k in ["encoder", "adp_llama", "adp_qwen"]):
        try:
            encoder.load_state_dict(state["encoder"], strict=strict)
            adp_llama.load_state_dict(state["adp_llama"], strict=strict)
            adp_qwen.load_state_dict(state["adp_qwen"], strict=strict)
            enc_loaded = True
            print("   -> loaded encoder/adapters FROM state.pt")
        except Exception as e:
            print(f"   -> failed to load weights from state.pt ({e}); will try .pt files")

    if not enc_loaded and ckpt_dir:
        enc_path = os.path.join(ckpt_dir, "encoder.pt")
        llm_path = os.path.join(ckpt_dir, "adapter_llama.pt")
        qwn_path = os.path.join(ckpt_dir, "adapter_qwen.pt")
        if all(os.path.isfile(p) for p in [enc_path, llm_path, qwn_path]):
            encoder.load_state_dict(_safe_load(enc_path, map_location=device), strict=strict)
            adp_llama.load_state_dict(_safe_load(llm_path, map_location=device), strict=strict)
            adp_qwen.load_state_dict(_safe_load(qwn_path, map_location=device), strict=strict)
            print("   -> loaded encoder/adapters FROM encoder.pt + adapter_{llama,qwen}.pt")
        else:
            raise FileNotFoundError(
                "No weights found to resume: neither state.pt contained weights nor did encoder.pt/adapter_*.pt exist."
            )

    if optimizer is not None and isinstance(state, dict):
        opt_state = state.get("optimizer", None) or state.get("optim", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            _maybe_to_device_optimizer_state(optimizer, device)
            print("   -> restored optimizer state")

    if isinstance(state, dict) and "rng" in state:
        try:
            rng = state["rng"]
            if "torch" in rng and isinstance(rng["torch"], torch.ByteTensor):
                torch.set_rng_state(rng["torch"])
            elif "torch" in rng:
                torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
            print("   -> restored RNG state")
        except Exception as e:
            print(f"   -> RNG restore skipped ({e})")

    epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
    global_step = int(state.get("global_step", 0)) if isinstance(state, dict) else 0
    return epoch, global_step


# ---------------------------
# Small helpers
# ---------------------------

def _tensor_rms(x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(x.float().pow(2).mean().sqrt().item())

def _tensor_rms_d(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()

def _calibrate_to_embed_rms(prefix: torch.Tensor, wrapper: LMWrapper) -> torch.Tensor:
    """
    Per-example calibration of the latent prefix to the model's input-embedding RMS.
    This exactly matches eval-time behavior and avoids batch-level amplitude drift.
    """
    # scalar target
    tgt = prefix.new_tensor(wrapper.input_embedding_rms())
    # per-example RMS across [M, D]
    cur = prefix.float().pow(2).mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-8)
    gain = (tgt / cur).to(prefix.dtype)  # [B,1,1]
    return prefix * gain

def _first_non_bos(tokenizer, ids: torch.Tensor) -> torch.Tensor:
    """
    Returns the first *content* token id per row:
      - skips all left PADs (padding_side='left')
      - skips BOS if present at the first non-PAD position
    ids: [B, T]
    """
    device = ids.device
    B, T = ids.size()
    pad = getattr(tokenizer, "pad_token_id", None)
    bos = getattr(tokenizer, "bos_token_id", None)

    # mask of non-PAD positions
    if pad is None:
        nonpad = torch.ones_like(ids, dtype=torch.bool, device=device)
    else:
        nonpad = ids.ne(int(pad))

    # first non-PAD index per row
    first_idx = nonpad.float().argmax(dim=1)

    # gather that token
    row = torch.arange(B, device=device)
    first_tok = ids[row, first_idx]

    # if that token is BOS, move one step to the right (clamped)
    if bos is not None:
        is_bos = first_tok.eq(int(bos))
        next_idx = torch.clamp(first_idx + 1, max=T - 1)
        first_tok = torch.where(is_bos, ids[row, next_idx], first_tok)

    return first_tok


def main():
    ap = argparse.ArgumentParser()
    # Models & data
    ap.add_argument("--llama_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--qwen_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot", "squad", "squad_v2"])
    ap.add_argument("--hotpot_config", type=str, default="fullwiki")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)

    # Repro / randomness
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global seed for RNGs & epoch permutations.")
    ap.add_argument("--data_seed", type=int, default=DEFAULT_SEED, help="Seed for picking dataset subset.")

    # Interlingua / encoder
    ap.add_argument("--latent_len", type=int, default=8)
    ap.add_argument("--d_z", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=512)
    ap.add_argument("--encoder_type", type=str, default="byte", choices=["byte", "simple-st"])
    ap.add_argument("--encoder_use_chat_template", action="store_true",
                    help="Wrap encoder input with a neutral chat-style header (SimpleEncoder only).")
    ap.add_argument("--encoder_backbone", type=str, default=None,
                    help="Optional SentenceTransformer backbone when --encoder_type=simple-st")

    # Training & stability
    ap.add_argument("--max_answer_tokens", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale_l2", type=float, default=0.05,
                    help="L2 penalty weight to keep adapter.scale near 1.0; set 0 to disable.")
    ap.add_argument("--adapter_rms_l2", type=float, default=0.0,
                    help="Optional penalty to pull RAW adapter RMS toward input embedding RMS.")
    ap.add_argument("--max_grad_norm", type=float, default=0.0,
                    help="If >0, clip grad norm to this value.")
    ap.add_argument("--adapter_freeze_scale", action="store_true",
                    help="If set, fix adapter.scale at 1.0 (no learning).")
    ap.add_argument("--first_token_ce_weight", type=float, default=0.5,
                    help="Weight for the first-token CE objective; 0 disables.")
    ap.add_argument("--train_append_bos_after_prefix", type=str, default="auto",
                    choices=["auto","yes","no"],
                    help="Controls BOS appending when computing first-token CE (train).")

    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--sequential_models", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--fp16_mps", action="store_true")
    ap.add_argument("--warm_anchor_text", type=str, default="",
                    help="Optional anchor tokens AFTER latent prefix during training, e.g. 'Answer: '")
    ap.add_argument("--debug", action="store_true")

    # Checkpointing
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save the latest checkpoint every N steps and prune old files.")
    ap.add_argument("--resume_from", type=str, default=None, help="Path to state.pt OR a directory containing it.")
    ap.add_argument("--auto_resume", action="store_true")
    ap.add_argument("--no_load_optimizer", action="store_true")

    # Training stats (for eval-time calibration)
    ap.add_argument("--save_training_stats", action="store_true", help="Record running mean of prefix RMS per model and save to training_stats.json")

    args = ap.parse_args()

    # Device + dtype
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16 if args.fp16_mps else torch.float32
    else:
        dtype = torch.float32

    # ===== Repro =====
    random.seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except Exception:
        pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ===== Data =====
    print("Loading dataset subset...")
    if args.dataset.startswith("squad"):
        print("Loading SQuAD subset...")
        examples = load_examples(dataset=args.dataset, split="train", samples=args.samples, seed=args.data_seed)
    else:
        print("Loading HotpotQA subset...")
        examples = load_examples(dataset="hotpot", split="train", samples=args.samples, seed=args.data_seed, config=args.hotpot_config)

    if len(examples) == 0:
        raise RuntimeError("No training examples loaded.")

    texts = [e["source"] for e in examples]
    answers = [e["answer"] for e in examples]

    # ===== Models =====
    llama = LMWrapper(LMConfig(model_id=args.llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=args.qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))
    print(f"Llama hidden size: {llama.d_model}, Qwen hidden size: {qwen.d_model}")

    if args.warm_anchor_text:
        anchor_llama_ids = llama.tokenizer.encode(args.warm_anchor_text, add_special_tokens=False) if args.warm_anchor_text else []
        anchor_qwen_ids  = qwen.tokenizer.encode(args.warm_anchor_text,  add_special_tokens=False) if args.warm_anchor_text else []
    else:
        anchor_llama_ids, anchor_qwen_ids = [], []

    if args.grad_ckpt:
        llama.enable_gradient_checkpointing()
        qwen.enable_gradient_checkpointing()

    # ===== Encoder =====
    if args.encoder_type == "byte":
        encoder = InterlinguaEncoder(d_z=args.d_z, latent_len=args.latent_len).to(device)
        byte_tok = ByteTokenizer(max_bytes=args.max_bytes)
        def encode_fn(batch_texts):
            z_bytes = collate_bytes(batch_texts, byte_tok, device)
            return encoder(z_bytes)
    else:
        encoder = SimpleEncoder(d_z=args.d_z, latent_len=args.latent_len,
                                backbone=(args.encoder_backbone or "sentence-transformers/all-MiniLM-L6-v2")).to(device)
        def _neutral_chat_wrap(s: str) -> str:
            system = "You are a concise QA assistant. Use the context to answer with a short phrase only."
            return f"System: {system}\nUser: {s}\nAssistant:"
        def encode_fn(batch_texts):
            if args.encoder_use_chat_template:
                batch_texts = [_neutral_chat_wrap(t) for t in batch_texts]
            return encoder(batch_texts)

    # ===== Adapters =====
    adp_llama = Adapter(d_z=args.d_z, d_model=llama.d_model).to(device)
    adp_qwen  = Adapter(d_z=args.d_z, d_model=qwen.d_model).to(device)

    if args.adapter_freeze_scale:
        adp_llama.scale.requires_grad_(False)
        adp_qwen.scale.requires_grad_(False)
        with torch.no_grad():
            adp_llama.scale.fill_(1.0)
            adp_qwen.scale.fill_(1.0)

    # ===== Optimizer =====
    optim_groups = list(encoder.parameters()) + list(adp_llama.parameters()) + list(adp_qwen.parameters())
    optimizer = optim.AdamW([p for p in optim_groups if p.requires_grad], lr=args.lr)

    # ===== Tokenize answers (teacher forcing) =====
    with _temp_padding_side(llama.tokenizer, "right"):
        llama_tok = llama.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_answer_tokens, add_special_tokens=True
        )
    llama_ids = llama_tok["input_ids"].to(device)

    with _temp_padding_side(qwen.tokenizer, "right"):
        qwen_tok = qwen.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_answer_tokens, add_special_tokens=True
        )
    qwen_ids = qwen_tok["input_ids"].to(device)

    # First gold token ids (skip left PADs and BOS) for the CE on the first step
    llama_first_ids_all = _first_non_bos(llama.tokenizer, llama_ids)
    qwen_first_ids_all  = _first_non_bos(qwen.tokenizer,  qwen_ids)

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    # ===== Resume (optional) =====
    start_epoch = 0
    global_step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_path = None
    if args.resume_from or args.auto_resume:
        ckpt_path = args.resume_from if args.resume_from else find_latest_checkpoint(args.save_dir)
        if ckpt_path and (os.path.isfile(ckpt_path) or os.path.isdir(os.path.dirname(ckpt_path))):
            print(f"⏪ Resuming from: {ckpt_path}")
            epoch_loaded, global_loaded = load_checkpoint(
                ckpt_path, encoder, adp_llama, adp_qwen,
                optimizer=None if args.no_load_optimizer else optimizer,
                strict=True, device=device
            )
            start_epoch = epoch_loaded
            global_step = global_loaded
            print(f"   -> start_epoch={start_epoch}, global_step={global_step}")
        else:
            print("⚠️  No valid checkpoint found to resume; starting fresh.")

    # ===== Training stats trackers =====
    class _RunningMean:
        def __init__(self):
            self.n = 0
            self.sum = 0.0
        def update(self, value: float):
            self.n += 1
            self.sum += float(value)
        @property
        def mean(self):
            return (self.sum / self.n) if self.n > 0 else 0.0

    rms_llama_raw = _RunningMean()
    rms_qwen_raw  = _RunningMean()
    rms_llama_cal = _RunningMean()
    rms_qwen_cal  = _RunningMean()

    embed_rms_llama = llama.input_embedding_rms()
    embed_rms_qwen  = qwen.input_embedding_rms()

    # Helper: map train flag to boolean for BOS policy
    def _bos_flag_from_str(flag: str, has_anchor: bool) -> Optional[bool]:
        if flag == "auto":
            return None  # let wrapper auto-policy decide
        return (flag == "yes")

    # ===== Train =====
    ema_step_time = None

    def scale_penalty(adapter: Adapter) -> torch.Tensor:
        if args.scale_l2 <= 0.0 or (adapter.scale is None) or (not adapter.scale.requires_grad):
            return torch.zeros((), device=device)
        return (adapter.scale - 1.0).pow(2).mean()

    def rms_raw_penalty(prefix_raw: torch.Tensor, wrapper: LMWrapper) -> torch.Tensor:
        if args.adapter_rms_l2 <= 0.0:
            return torch.zeros((), device=device)
        tgt = prefix_raw.new_tensor(wrapper.input_embedding_rms())
        cur = _tensor_rms_d(prefix_raw)
        return (cur - tgt).pow(2)

    params_for_clip = [p for p in optim_groups if p.requires_grad]

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        g = torch.Generator(device="cpu")
        g.manual_seed(int(args.seed) + int(epoch))
        perm = torch.randperm(N, generator=g)

        for step in range(steps_per_epoch):
            t0 = time.time()
            idx = perm[step*args.batch_size : (step+1)*args.batch_size]
            batch_texts = [texts[i] for i in idx.tolist()]

            optimizer.zero_grad(set_to_none=True)

            # Encode sources once when possible
            z = encode_fn(batch_texts)

            # ----- Llama path -----
            prefix_llama_raw = adp_llama(z)
            prefix_llama = _calibrate_to_embed_rms(prefix_llama_raw, llama)
            y_llama = llama_ids[idx]
            loss_llama_tf = llama.forward_with_prefix_loss(prefix_llama, y_llama, anchor_token_ids=anchor_llama_ids)

            # First-token CE (supervise the very first token)
            if args.first_token_ce_weight and args.first_token_ce_weight > 0.0:
                logits_first_ll = llama.first_token_logits_from_prefix(
                    prefix_llama,
                    anchor_token_text=(args.warm_anchor_text or None),
                    append_bos_after_prefix=_bos_flag_from_str(args.train_append_bos_after_prefix, has_anchor=bool(anchor_llama_ids))
                )
                y_first_ll = llama_first_ids_all[idx]
                loss_first_ll = nn.functional.cross_entropy(logits_first_ll.float(), y_first_ll)
            else:
                loss_first_ll = torch.zeros((), device=device)

            # ----- Qwen path -----
            prefix_qwen_raw = adp_qwen(z)
            prefix_qwen = _calibrate_to_embed_rms(prefix_qwen_raw, qwen)
            y_qwen = qwen_ids[idx]
            loss_qwen_tf = qwen.forward_with_prefix_loss(prefix_qwen, y_qwen, anchor_token_ids=anchor_qwen_ids)

            if args.first_token_ce_weight and args.first_token_ce_weight > 0.0:
                logits_first_qw = qwen.first_token_logits_from_prefix(
                    prefix_qwen,
                    anchor_token_text=(args.warm_anchor_text or None),
                    append_bos_after_prefix=_bos_flag_from_str(args.train_append_bos_after_prefix, has_anchor=bool(anchor_qwen_ids))
                )
                y_first_qw = qwen_first_ids_all[idx]
                loss_first_qw = nn.functional.cross_entropy(logits_first_qw.float(), y_first_qw)
            else:
                loss_first_qw = torch.zeros((), device=device)

            # Combine losses
            penalty = scale_penalty(adp_llama) + scale_penalty(adp_qwen)
            rms_pen  = rms_raw_penalty(prefix_llama_raw, llama) + rms_raw_penalty(prefix_qwen_raw, qwen)

            loss_ll = loss_llama_tf + args.first_token_ce_weight * loss_first_ll
            loss_qw = loss_qwen_tf  + args.first_token_ce_weight * loss_first_qw
            loss = 0.5 * (loss_ll + loss_qw) + args.scale_l2 * penalty + args.adapter_rms_l2 * rms_pen

            if not torch.isfinite(loss):
                print("NaN/Inf loss; skipping step")
                continue

            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))
            optimizer.step()

            # Track RMS (raw + calibrated)
            llama_rms_raw_val = _tensor_rms(prefix_llama_raw)
            qwen_rms_raw_val  = _tensor_rms(prefix_qwen_raw)
            llama_rms_cal_val = _tensor_rms(prefix_llama)
            qwen_rms_cal_val  = _tensor_rms(prefix_qwen)

            if args.save_training_stats:
                try: rms_llama_raw.update(llama_rms_raw_val)
                except Exception: pass
                try: rms_qwen_raw.update(qwen_rms_raw_val)
                except Exception: pass
                try: rms_llama_cal.update(llama_rms_cal_val)
                except Exception: pass
                try: rms_qwen_cal.update(qwen_rms_cal_val)
                except Exception: pass

            # Grad norm (monitor) – encoder only as a proxy
            try:
                total_norm = float(torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float("inf")))
            except Exception:
                total_norm = float("nan")

            global_step += 1
            dt = time.time() - t0
            ema_step_time = dt if (locals().get("ema_step_time", None) is None) else (0.9 * locals()["ema_step_time"] + 0.1 * dt)

            if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                pen_L = float(scale_penalty(adp_llama).item()) if args.scale_l2 > 0 else 0.0
                pen_Q = float(scale_penalty(adp_qwen).item())  if args.scale_l2 > 0 else 0.0
                msg = (
                    f"  step  {step+1}/{steps_per_epoch} | "
                    f"loss_L={float(loss_llama_tf.item()):.4f} | loss_Q={float(loss_qwen_tf.item()):.4f} | "
                    f"firstCE_L={float((loss_first_ll if torch.is_tensor(loss_first_ll) else 0.0)):.4f} | "
                    f"firstCE_Q={float((loss_first_qw if torch.is_tensor(loss_first_qw) else 0.0)):.4f} | "
                    f"scale_pen(L)={pen_L:.4e} | scale_pen(Q)={pen_Q:.4e} | "
                    f"grad_norm={total_norm:.2f} | sec/step~{dt:.2f}"
                )
                if args.save_training_stats:
                    msg += (f" | rms_raw(L)~{rms_llama_raw.mean:.4f} rms_raw(Q)~{rms_qwen_raw.mean:.4f}"
                            f" | rms_cal(L)~{rms_llama_cal.mean:.4f} rms_cal(Q)~{rms_qwen_cal.mean:.4f}"
                            f" | embed_rms(L)~{embed_rms_llama:.5f} embed_rms(Q)~{embed_rms_qwen:.5f}")
                print(msg)

            # ---- Periodic checkpoint: save + prune
            if args.save_every and (global_step % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                cfg = {
                    "d_z": args.d_z,
                    "latent_len": args.latent_len,
                    "byte_max": args.max_bytes,
                    "llama_id": args.llama_id,
                    "qwen_id": args.qwen_id,
                    "encoder_type": args.encoder_type,
                    "encoder_use_chat_template": bool(args.encoder_use_chat_template),
                    "encoder_backbone": (args.encoder_backbone or ""),
                    "warm_anchor_text": args.warm_anchor_text,
                    "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
                    "first_token_ce_weight": args.first_token_ce_weight,
                    "seed": args.seed,
                    "data_seed": args.data_seed,
                }
                state_blob = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                    "optimizer": optimizer.state_dict(),
                    "adapter_scale": {
                        "llama": float(adp_llama.scale.detach().cpu().item()),
                        "qwen":  float(adp_qwen.scale.detach().cpu().item()),
                    },
                    "encoder": encoder.state_dict(),
                    "adp_llama": adp_llama.state_dict(),
                    "adp_qwen": adp_qwen.state_dict(),
                }
                artifacts = {
                    "encoder.pt":       encoder.state_dict(),
                    "adapter_llama.pt": adp_llama.state_dict(),
                    "adapter_qwen.pt":  adp_qwen.state_dict(),
                    "state.pt":         state_blob,
                    "config.json":      cfg,
                }
                save_latest_checkpoint(args.save_dir, artifacts, pre_prune=True, post_prune=True, verbose=True)
                print(f"  ✅ Saved (and pruned to) latest at step {global_step}")

    # ===== Final save =====
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {
        "d_z": args.d_z,
        "latent_len": args.latent_len,
        "byte_max": args.max_bytes,
        "llama_id": args.llama_id,
        "qwen_id": args.qwen_id,
        "encoder_type": args.encoder_type,
        "encoder_use_chat_template": bool(args.encoder_use_chat_template),
        "encoder_backbone": (args.encoder_backbone or ""),
        "warm_anchor_text": args.warm_anchor_text,
        "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
        "first_token_ce_weight": args.first_token_ce_weight,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }
    state_blob = {
        "epoch": epoch + 1 if 'epoch' in locals() else None,
        "global_step": global_step if 'global_step' in locals() else None,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "optimizer": optimizer.state_dict(),
        "adapter_scale": {
            "llama": float(adp_llama.scale.detach().cpu().item()),
            "qwen":  float(adp_qwen.scale.detach().cpu().item()),
        },
        "encoder": encoder.state_dict(),
        "adp_llama": adp_llama.state_dict(),
        "adp_qwen": adp_qwen.state_dict(),
    }
    artifacts = {
        "encoder.pt":       encoder.state_dict(),
        "adapter_llama.pt": adp_llama.state_dict(),
        "adapter_qwen.pt":  adp_qwen.state_dict(),
        "state.pt":         state_blob,
        "config.json":      cfg,
    }
    save_latest_checkpoint(args.save_dir, artifacts, pre_prune=True, post_prune=True, verbose=True)
    print(f"✅ Saved latest checkpoint to {args.save_dir}")

    if args.save_training_stats:
        stats = {
            "llama": {
                "rms_mean_raw": rms_llama_raw.mean, "rms_mean_cal": rms_llama_cal.mean,
                "embed_rms": embed_rms_llama, "count": rms_llama_cal.n
            },
            "qwen":  {
                "rms_mean_raw": rms_qwen_raw.mean,  "rms_mean_cal": rms_qwen_cal.mean,
                "embed_rms": embed_rms_qwen,  "count": rms_qwen_cal.n
            },
        }
        with open(os.path.join(args.save_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📝 Saved training_stats.json: {stats}")


if __name__ == "__main__":
    main()
