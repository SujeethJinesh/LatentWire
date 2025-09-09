import os
import time
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
        def encode_fn(batch_texts):
            return encoder(batch_texts)
    adp_llama = Adapter(d_z=args.d_z, d_model=llama.d_model).to(device)
    adp_qwen  = Adapter(d_z=args.d_z, d_model=qwen.d_model).to(device)

    optim_groups = list(encoder.parameters()) + list(adp_llama.parameters()) + list(adp_qwen.parameters())
    optimizer = optim.AdamW(optim_groups, lr=args.lr)

    llama_ans = llama.tokenizer(answers, return_tensors="pt", padding=True, truncation=True, max_length=args.max_answer_tokens, add_special_tokens=True)
    qwen_ans  = qwen.tokenizer(answers,  return_tensors="pt", padding=True, truncation=True, max_length=args.max_answer_tokens, add_special_tokens=True)
    llama_ids = llama_ans["input_ids"].to(device)
    qwen_ids  = qwen_ans["input_ids"].to(device)

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    global_step = 0
    ema_step_time = None
    for epoch in range(args.epochs):
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
                loss_qwen  = qwen.forward_with_prefix_loss(prefix_qwen, y_qwen)
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
                print(f"  step {step+1}/{steps_per_epoch} | loss_llama={loss_llama.item():.4f} | loss_qwen={loss_qwen.item():.4f} | grad_norm={total_norm:.2f} | sec/step~{ema_step_time:.2f}")

            # Periodic checkpointing
            if args.save_every and (global_step % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(encoder.state_dict(), os.path.join(args.save_dir, f"encoder_step{global_step}.pt"))
                torch.save(adp_llama.state_dict(), os.path.join(args.save_dir, f"adapter_llama_step{global_step}.pt"))
                torch.save(adp_qwen.state_dict(), os.path.join(args.save_dir, f"adapter_qwen_step{global_step}.pt"))
                print(f"  ✅ Saved checkpoint at step {global_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
    torch.save(adp_llama.state_dict(), os.path.join(args.save_dir, "adapter_llama.pt"))
    torch.save(adp_qwen.state_dict(), os.path.join(args.save_dir, "adapter_qwen.pt"))

    cfg = {"d_z": args.d_z, "latent_len": args.latent_len, "byte_max": args.max_bytes, "llama_id": args.llama_id, "qwen_id": args.qwen_id, "encoder_type": args.encoder_type}
    import json
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved encoder and adapters to", args.save_dir)


if __name__ == "__main__":
    main()
