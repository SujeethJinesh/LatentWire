import os
import re
import time
import json
import argparse

import torch
import torch.optim as optim

from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
from latentwire.data import load_hotpot_subset


def collate_bytes(texts, byte_tok: ByteTokenizer, device: str):
    ids = [byte_tok.encode(t) for t in texts]
    maxT = max([x.size(0) for x in ids])
    batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.size(0), dtype=torch.long)], dim=0) for x in ids], dim=0)
    return batch.to(device)


def find_latest_checkpoint(save_dir: str):
    """Return path to the numerically-latest state_step*.pt in save_dir, else None."""
    if not os.path.isdir(save_dir):
        return None
    cand = []
    for fn in os.listdir(save_dir):
        m = re.match(r"state_step(\d+)\.pt$", fn)
        if m:
            cand.append((int(m.group(1)), os.path.join(save_dir, fn)))
    if not cand:
        # also support 'last.pt'
        last_path = os.path.join(save_dir, "last.pt")
        return last_path if os.path.isfile(last_path) else None
    cand.sort()
    return cand[-1][1]


def save_checkpoint(path, encoder, adp_llama, adp_qwen, optimizer, epoch, global_step, args):
    state = {
        "encoder": encoder.state_dict(),
        "adp_llama": adp_llama.state_dict(),
        "adp_qwen": adp_qwen.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, path)
    # also update pointer
    last_path = os.path.join(os.path.dirname(path), "last.pt")
    try:
        torch.save(state, last_path)
    except Exception:
        pass


def load_checkpoint(path, encoder, adp_llama, adp_qwen, optimizer=None, strict=True, device="cpu"):
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state["encoder"], strict=strict)
    adp_llama.load_state_dict(state["adp_llama"], strict=strict)
    adp_qwen.load_state_dict(state["adp_qwen"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        # Move optimizer state to correct device
        for p in optimizer.state.values():
            for k, v in p.items():
                if torch.is_tensor(v):
                    p[k] = v.to(device)
    epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    return epoch, global_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--qwen_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--latent_len", type=int, default=8)
    ap.add_argument("--d_z", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=512)
    ap.add_argument("--max_answer_tokens", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--load_4bit", action="store_true", help="load LMs in 4-bit nf4 (Linux + CUDA required)")
    ap.add_argument("--hotpot_config", type=str, default="fullwiki", help="HotpotQA config: fullwiki or distractor")
    ap.add_argument("--sequential_models", action="store_true", help="backprop through Llama and Qwen sequentially to reduce peak memory")
    ap.add_argument("--grad_ckpt", action="store_true", help="enable HF gradient checkpointing on both LMs to save memory")
    ap.add_argument("--fp16_mps", action="store_true", help="use float16 compute on MPS (Apple Silicon) to reduce memory")
    ap.add_argument("--encoder_type", type=str, default="byte", choices=["byte", "simple-st"], help="choose byte encoder or SimpleEncoder")
    ap.add_argument("--save_every", type=int, default=0, help="save checkpoints every N steps (0=only final)")
    ap.add_argument(
        "--encoder_use_chat_template",
        action="store_true",
        help="If set with --encoder_type simple-st, wrap inputs in a neutral chat-style header before encoding (off by default).",
    )
    ap.add_argument("--resume_from", type=str, default=None, help="path to a state_step*.pt or last.pt to resume from")
    ap.add_argument("--auto_resume", action="store_true", help="if set, auto-pick the latest checkpoint in save_dir")
    ap.add_argument("--no_load_optimizer", action="store_true", help="if set, do not restore optimizer state")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16 if args.fp16_mps else torch.float32
    else:
        dtype = torch.float32

    # Data
    print("Loading HotpotQA subset...")
    examples = load_hotpot_subset(split="train", samples=args.samples, seed=0, config=args.hotpot_config)
    texts = [e["source"] for e in examples]
    answers = [e["answer"] for e in examples]

    # Models
    llama = LMWrapper(LMConfig(model_id=args.llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit))
    qwen  = LMWrapper(LMConfig(model_id=args.qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit))
    print(f"Llama hidden size: {llama.d_model}, Qwen hidden size: {qwen.d_model}")

    if args.grad_ckpt:
        llama.enable_gradient_checkpointing()
        qwen.enable_gradient_checkpointing()

    # Encoder selection
    if args.encoder_type == "byte":
        encoder = InterlinguaEncoder(d_z=args.d_z, latent_len=args.latent_len).to(device)
        byte_tok = ByteTokenizer(max_bytes=args.max_bytes)
        def encode_fn(batch_texts):
            z_bytes = collate_bytes(batch_texts, byte_tok, device)
            return encoder(z_bytes)
    else:
        encoder = SimpleEncoder(d_z=args.d_z, latent_len=args.latent_len).to(device)
        def _neutral_chat_wrap(s: str) -> str:
            # Neutral, model-agnostic chat-style wrapping without inserting model-specific tokens.
            system = (
                "You are a concise QA assistant. Use the context to answer with a short phrase only."
            )
            return f"System: {system}\nUser: {s}\nAssistant:"

        def encode_fn(batch_texts):
            # If requested, apply neutral chat wrapping before SimpleEncoder to better match model prompts.
            if args.encoder_use_chat_template:
                batch_texts = [_neutral_chat_wrap(t) for t in batch_texts]
            return encoder(batch_texts)

    adp_llama = Adapter(d_z=args.d_z, d_model=llama.d_model).to(device)
    adp_qwen  = Adapter(d_z=args.d_z, d_model=qwen.d_model).to(device)

    optim_groups = list(encoder.parameters()) + list(adp_llama.parameters()) + list(adp_qwen.parameters())
    optimizer = optim.AdamW(optim_groups, lr=args.lr)

    # Answer tokenization
    llama_ans = llama.tokenizer(answers, return_tensors="pt", padding=True, truncation=True, max_length=args.max_answer_tokens, add_special_tokens=True)
    qwen_ans  = qwen.tokenizer(answers,  return_tensors="pt", padding=True, truncation=True, max_length=args.max_answer_tokens, add_special_tokens=True)
    llama_ids = llama_ans["input_ids"].to(device)
    qwen_ids  = qwen_ans["input_ids"].to(device)

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    # ===== Resume logic =====
    start_epoch = 0
    global_step = 0
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

    ema_step_time = None
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        perm = torch.randperm(N)
        for step in range(steps_per_epoch):
            t0 = time.time()
            idx = perm[step*args.batch_size : (step+1)*args.batch_size]
            batch_texts = [texts[i] for i in idx.tolist()]

            optimizer.zero_grad(set_to_none=True)
            if args.sequential_models:
                # Backprop through each LM separately to reduce peak memory
                z = encode_fn(batch_texts)
                prefix_llama = adp_llama(z)
                y_llama = llama_ids[idx]
                loss_llama = llama.forward_with_prefix_loss(prefix_llama, y_llama)
                if not torch.isfinite(loss_llama):
                    print("NaN/Inf loss_llama; skipping step")
                    continue
                loss_llama.backward()

                # Recompute encoder forward for Qwen to avoid retaining graph
                z = encode_fn(batch_texts)
                prefix_qwen = adp_qwen(z)
                y_qwen = qwen_ids[idx]
                loss_qwen = qwen.forward_with_prefix_loss(prefix_qwen, y_qwen)
                if not torch.isfinite(loss_qwen):
                    print("NaN/Inf loss_qwen; skipping step")
                    continue
                loss_qwen.backward()
                # Optional: free MPS cache
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                optimizer.step()
            else:
                z = encode_fn(batch_texts)       # [B, M, d_z]
                prefix_llama = adp_llama(z)      # [B, M, d_model_L]
                prefix_qwen  = adp_qwen(z)       # [B, M, d_model_Q]

                y_llama = llama_ids[idx]
                y_qwen  = qwen_ids[idx]

                loss_llama = llama.forward_with_prefix_loss(prefix_llama, y_llama)
                loss_qwen  = qwen.forward_with_prefix_loss(prefix_qwen,  y_qwen)
                loss = (loss_llama + loss_qwen) * 0.5
                if not torch.isfinite(loss):
                    print("NaN/Inf loss; skipping step")
                    continue
                loss.backward()
                optimizer.step()

            # Grad norm monitor (does not clip)
            try:
                total_norm = float(torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float("inf")))
            except Exception:
                total_norm = float("nan")

            global_step += 1
            dt = time.time() - t0
            ema_step_time = dt if ema_step_time is None else (0.9 * ema_step_time + 0.1 * dt)

            if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                print(f"  step {step+1}/{steps_per_epoch} | loss_llama={loss_llama.item():.4f} | "
                      f"loss_qwen={loss_qwen.item():.4f} | grad_norm={total_norm:.2f} | sec/step~{ema_step_time:.2f}")

            # Periodic checkpointing: light and full
            if args.save_every and (global_step % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                # Light (compat)
                torch.save(encoder.state_dict(), os.path.join(args.save_dir, f"encoder_step{global_step}.pt"))
                torch.save(adp_llama.state_dict(), os.path.join(args.save_dir, f"adapter_llama_step{global_step}.pt"))
                torch.save(adp_qwen.state_dict(), os.path.join(args.save_dir, f"adapter_qwen_step{global_step}.pt"))
                # Full (resume-ready)
                save_checkpoint(os.path.join(args.save_dir, f"state_step{global_step}.pt"),
                                encoder, adp_llama, adp_qwen, optimizer, epoch, global_step, args)
                print(f"  ✅ Saved checkpoint at step {global_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    # Final light files
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
    torch.save(adp_llama.state_dict(), os.path.join(args.save_dir, "adapter_llama.pt"))
    torch.save(adp_qwen.state_dict(), os.path.join(args.save_dir, "adapter_qwen.pt"))
    # Final full state
    save_checkpoint(os.path.join(args.save_dir, "state_final.pt"),
                    encoder, adp_llama, adp_qwen, optimizer, args.epochs-1, global_step, args)

    cfg = {
        "d_z": args.d_z,
        "latent_len": args.latent_len,
        "byte_max": args.max_bytes,
        "llama_id": args.llama_id,
        "qwen_id": args.qwen_id,
        "encoder_type": args.encoder_type
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved encoder, adapters, and state to", args.save_dir)


if __name__ == "__main__":
    main()
