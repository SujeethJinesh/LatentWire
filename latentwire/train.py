# latentwire/train.py
import os
import re
import time
import json
import argparse

import torch
import torch.optim as optim

from latentwire.models import (
    InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
)
from latentwire.checkpointing import save_latest_checkpoint, prune_save_dir
from latentwire.data import load_examples
from latentwire.common import collate_bytes  # deduped

def find_latest_checkpoint(save_dir: str):
    if not os.path.isdir(save_dir):
        return None
    state_path = os.path.join(save_dir, "state.pt")
    if os.path.isfile(state_path):
        return state_path
    last_path = os.path.join(save_dir, "last.pt")
    if os.path.isfile(last_path):
        return last_path
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

def load_checkpoint(path, encoder, adp_llama, adp_qwen, optimizer=None, strict=True, device="cpu"):
    state = _safe_load(path, map_location="cpu")
    if "encoder" in state and "adp_llama" in state and "adp_qwen" in state:
        encoder.load_state_dict(state["encoder"], strict=strict)
        adp_llama.load_state_dict(state["adp_llama"], strict=strict)
        adp_qwen.load_state_dict(state["adp_qwen"], strict=strict)
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            for p in optimizer.state.values():
                for k, v in p.items():
                    if torch.is_tensor(v):
                        p[k] = v.to(device)
        epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        return epoch, global_step

    epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        for p in optimizer.state.values():
            for k, v in p.items():
                if torch.is_tensor(v):
                    p[k] = v.to(device)
    return epoch, global_step

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

def _tensor_rms(x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(x.float().pow(2).mean().sqrt().item())

def _tensor_rms_d(x: torch.Tensor) -> torch.Tensor:
    # differentiable RMS
    return x.pow(2).mean().sqrt()

def _calibrate_to_embed_rms(prefix: torch.Tensor, wrapper: LMWrapper) -> torch.Tensor:
    with torch.no_grad():
        cur = prefix.float().pow(2).mean().sqrt()
        tgt = wrapper.input_embedding_rms()
        gain = float(tgt) / float(cur) if float(cur) > 0 else 1.0
    return prefix * gain

def main():
    ap = argparse.ArgumentParser()
    # Models & data
    ap.add_argument("--llama_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--qwen_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot","squad","squad_v2"])
    ap.add_argument("--hotpot_config", type=str, default="fullwiki")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)

    # Interlingua / encoder
    ap.add_argument("--latent_len", type=int, default=8)
    ap.add_argument("--d_z", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=512)
    ap.add_argument("--encoder_type", type=str, default="byte", choices=["byte", "simple-st"])
    ap.add_argument("--encoder_use_chat_template", action="store_true",
                    help="Wrap encoder input with a neutral chat-style header (SimpleEncoder only).")

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
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--sequential_models", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--fp16_mps", action="store_true")
    ap.add_argument("--warm_anchor_text", type=str, default="",
                    help="Optional anchor tokens AFTER latent prefix during training, e.g. 'Answer:'")
    ap.add_argument("--debug", action="store_true")

    # Checkpointing
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save the latest checkpoint every N steps and prune old files.")
    ap.add_argument("--resume_from", type=str, default=None)
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

    # ===== Data =====
    print("Loading dataset subset...")
    if args.dataset.startswith("squad"):
        print("Loading SQuAD subset...")
        examples = load_examples(dataset=args.dataset, split="train", samples=args.samples, seed=0)
    else:
        print("Loading HotpotQA subset...")
        examples = load_examples(dataset="hotpot", split="train", samples=args.samples, seed=0, config=args.hotpot_config)

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
        encoder = SimpleEncoder(d_z=args.d_z, latent_len=args.latent_len).to(device)
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

    # ===== Tokenize answers =====
    llama_tok = llama.tokenizer(
        answers, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_answer_tokens, add_special_tokens=True
    )
    llama_ids = llama_tok["input_ids"].to(device)
    llama_attention_mask = llama_tok["attention_mask"].to(device)
    
    # Create labels - mask padding AND special tokens
    llama_labels = llama_ids.clone()
    # Mask padding
    llama_labels[llama_attention_mask == 0] = -100
    # Mask BOS token if present (usually first token)
    if llama.tokenizer.bos_token_id is not None:
        llama_labels[llama_ids == llama.tokenizer.bos_token_id] = -100
    # Mask EOS token if present
    if llama.tokenizer.eos_token_id is not None:
        llama_labels[llama_ids == llama.tokenizer.eos_token_id] = -100

    qwen_tok = qwen.tokenizer(
        answers, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_answer_tokens, add_special_tokens=True
    )
    qwen_ids = qwen_tok["input_ids"].to(device)
    qwen_attention_mask = qwen_tok["attention_mask"].to(device)
    
    # Create labels - mask padding AND special tokens
    qwen_labels = qwen_ids.clone()
    # Mask padding
    qwen_labels[qwen_attention_mask == 0] = -100
    # Mask BOS token if present
    if qwen.tokenizer.bos_token_id is not None:
        qwen_labels[qwen_ids == qwen.tokenizer.bos_token_id] = -100
    # Mask EOS token if present
    if qwen.tokenizer.eos_token_id is not None:
        qwen_labels[qwen_ids == qwen.tokenizer.eos_token_id] = -100

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    # ===== Resume (optional) =====
    start_epoch = 0
    global_step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        prune_save_dir(args.save_dir)
    except Exception:
        pass

    if args.resume_from or args.auto_resume:
        ckpt_path = args.resume_from or find_latest_checkpoint(args.save_dir)
        if ckpt_path and os.path.isfile(ckpt_path):
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
    rms_llama = _RunningMean()
    rms_qwen  = _RunningMean()

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
        perm = torch.randperm(N)
        for step in range(steps_per_epoch):
            t0 = time.time()
            idx = perm[step*args.batch_size : (step+1)*args.batch_size]
            batch_texts = [texts[i] for i in idx.tolist()]

            optimizer.zero_grad(set_to_none=True)

            if args.sequential_models:
                z = encode_fn(batch_texts)

                # ---- Llama path
                prefix_llama_raw = adp_llama(z)
                prefix_llama = _calibrate_to_embed_rms(prefix_llama_raw, llama)
                y_llama = llama_ids[idx]  # Use original IDs, not labels with -100
                loss_llama = llama.forward_with_prefix_loss(prefix_llama, y_llama, anchor_token_ids=anchor_llama_ids)
                loss_llama_total = loss_llama + args.scale_l2 * scale_penalty(adp_llama) + args.adapter_rms_l2 * rms_raw_penalty(prefix_llama_raw, llama)
                if torch.isfinite(loss_llama_total):
                    loss_llama_total.backward()

                # ---- Qwen path
                z = encode_fn(batch_texts)
                prefix_qwen_raw = adp_qwen(z)
                prefix_qwen = _calibrate_to_embed_rms(prefix_qwen_raw, qwen)
                y_qwen = qwen_ids[idx]  # Use original IDs, not labels with -100
                loss_qwen = qwen.forward_with_prefix_loss(prefix_qwen, y_qwen, anchor_token_ids=anchor_qwen_ids)
                loss_qwen_total = loss_qwen + args.scale_l2 * scale_penalty(adp_qwen) + args.adapter_rms_l2 * rms_raw_penalty(prefix_qwen_raw, qwen)
                if torch.isfinite(loss_qwen_total):
                    loss_qwen_total.backward()

                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))
                optimizer.step()
                loss_llama_value = float(loss_llama.item())
                loss_qwen_value  = float(loss_qwen.item())
                if args.save_training_stats:
                    try: rms_llama.update(_tensor_rms(prefix_llama_raw))
                    except Exception: pass
                    try: rms_qwen.update(_tensor_rms(prefix_qwen_raw))
                    except Exception: pass

            else:
                z = encode_fn(batch_texts)
                prefix_llama_raw = adp_llama(z)
                prefix_qwen_raw  = adp_qwen(z)
                # Calibrate both before loss
                prefix_llama = _calibrate_to_embed_rms(prefix_llama_raw, llama)
                prefix_qwen  = _calibrate_to_embed_rms(prefix_qwen_raw,  qwen)
                y_llama = llama_ids[idx]; y_qwen = qwen_ids[idx]
                loss_llama = llama.forward_with_prefix_loss(prefix_llama, y_llama, anchor_token_ids=anchor_llama_ids)
                loss_qwen  = qwen.forward_with_prefix_loss(prefix_qwen,  y_qwen,  anchor_token_ids=anchor_qwen_ids)
                penalty = scale_penalty(adp_llama) + scale_penalty(adp_qwen)
                rms_pen  = rms_raw_penalty(prefix_llama_raw, llama) + rms_raw_penalty(prefix_qwen_raw, qwen)
                loss = 0.5 * (loss_llama + loss_qwen) + args.scale_l2 * penalty + args.adapter_rms_l2 * rms_pen
                if not torch.isfinite(loss):
                    print("NaN/Inf loss; skipping step")
                    continue
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))
                optimizer.step()
                loss_llama_value = float(loss_llama.item())
                loss_qwen_value  = float(loss_qwen.item())
                if args.save_training_stats:
                    try: rms_llama.update(_tensor_rms(prefix_llama_raw))
                    except Exception: pass
                    try: rms_qwen.update(_tensor_rms(prefix_qwen_raw))
                    except Exception: pass

            # Grad norm (monitor)
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
                    f"  step {step+1}/{steps_per_epoch} | "
                    f"loss_L={loss_llama_value:.4f} | loss_Q={loss_qwen_value:.4f} | "
                    f"scale_pen(L)= {pen_L:.4e} | scale_pen(Q)= {pen_Q:.4e} | "
                    f"grad_norm={total_norm:.2f} | sec/step~{dt:.2f}"
                )
                if args.save_training_stats and (rms_llama.n > 0 or rms_qwen.n > 0):
                    msg += f" | rms_L~{rms_llama.mean:.4f} rms_Q~{rms_qwen.mean:.4f}"
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
                    "warm_anchor_text": args.warm_anchor_text,
                }
                artifacts = {
                    "encoder.pt":       encoder.state_dict(),
                    "adapter_llama.pt": adp_llama.state_dict(),
                    "adapter_qwen.pt":  adp_qwen.state_dict(),
                    "state.pt": {
                        "epoch": epoch,
                        "global_step": global_step,
                        "optim": optimizer.state_dict(),
                        "adapter_scale": {
                            "llama": float(adp_llama.scale.detach().cpu().item()),
                            "qwen":  float(adp_qwen.scale.detach().cpu().item()),
                        },
                    },
                    "config.json": cfg,
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
        "warm_anchor_text": args.warm_anchor_text
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

    # Save training-time prefix RMS stats (optional, RAW only)
    if args.save_training_stats:
        stats = {
            "llama": {"rms_mean": rms_llama.mean, "count": rms_llama.n},
            "qwen":  {"rms_mean": rms_qwen.mean,  "count": rms_qwen.n},
        }
        with open(os.path.join(args.save_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📝 Saved training_stats.json: {stats}")

if __name__ == "__main__":
    main()
