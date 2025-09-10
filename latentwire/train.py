# latentwire/train.py
import os
import re
import time
import json
import argparse

import torch
import torch.optim as optim

from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, LMConfig, ByteTokenizer, SimpleEncoder
from latentwire.data import load_examples


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
        last_path = os.path.join(save_dir, "last.pt")
        return last_path if os.path.isfile(last_path) else None
    cand.sort()
    return cand[-1][1]


def save_checkpoint(path, encoder, adp_llama, adp_qwen, optimizer, epoch, global_step, args):
    state = {
        "encoder": encoder.state_dict(),
        "adp_llama": adp_llama.state_dict() if adp_llama is not None else None,
        "adp_qwen": adp_qwen.state_dict() if adp_qwen is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, path)
    try:
        torch.save(state, os.path.join(os.path.dirname(path), "last.pt"))
    except Exception:
        pass


def load_checkpoint(path, encoder, adp_llama, adp_qwen, optimizer=None, strict=True, device="cpu"):
    state = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state["encoder"], strict=strict)
    if adp_llama is not None and state.get("adp_llama"):
        adp_llama.load_state_dict(state["adp_llama"], strict=strict)
    if adp_qwen is not None and state.get("adp_qwen"):
        adp_qwen.load_state_dict(state["adp_qwen"], strict=strict)
    if optimizer is not None and "optimizer" in state and state["optimizer"]:
        optimizer.load_state_dict(state["optimizer"])
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
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--load_4bit", action="store_true", help="load LMs in 4-bit nf4 (Linux + CUDA required)")
    ap.add_argument("--hotpot_config", type=str, default="fullwiki", help="HotpotQA config: fullwiki or distractor")
    ap.add_argument("--sequential_models", action="store_true", help="backprop through Llama and Qwen sequentially to reduce peak memory")
    ap.add_argument("--grad_ckpt", action="store_true", help="enable HF gradient checkpointing on both LMs to save memory")
    ap.add_argument("--fp16_mps", action="store_true", help="use float16 compute on MPS (Apple Silicon) to reduce memory")
    ap.add_argument("--encoder_type", type=str, default="byte", choices=["byte", "simple-st"], help="choose byte encoder or SimpleEncoder")
    ap.add_argument("--encoder_use_chat_template", action="store_true", help="Wrap inputs to encoder in a neutral chat-style header before encoding (SimpleEncoder only).")
    ap.add_argument("--save_every", type=int, default=0, help="save checkpoints every N steps (0=only final)")
    ap.add_argument("--resume_from", type=str, default=None, help="path to a state_step*.pt or last.pt to resume from")
    ap.add_argument("--auto_resume", action="store_true", help="if set, auto-pick the latest checkpoint in save_dir")
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot","squad","squad_v2"], help="Which dataset to use")
    ap.add_argument("--warm_anchor_text", type=str, default="", help="Optional anchor tokens after latent during training, e.g. 'Answer: '")
    ap.add_argument("--debug", action="store_true", help="Print adapter scale and prefix norms intermittently")
    ap.add_argument("--lambda_llama", type=float, default=1.0, help="Weight for Llama loss (0 to disable).")
    ap.add_argument("--lambda_qwen", type=float, default=1.0, help="Weight for Qwen loss (0 to disable).")
    ap.add_argument("--acceptance_reg", type=float, default=0.05, help="Strength of prefix norm regularizer toward embedding norm.")
    ap.add_argument("--fix_adapter_scale", action="store_true", help="Keep adapter.scale frozen at 1.0 during training.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16 if args.fp16_mps else torch.float32
    else:
        dtype = torch.float32

    # Data
    print("Loading dataset subset...")
    if args.dataset.startswith("squad"):
        print("Loading SQuAD subset...")
        examples = load_examples(dataset=args.dataset, split="train", samples=args.samples, seed=0)
    else:
        print("Loading HotpotQA subset...")
        examples = load_examples(dataset="hotpot", split="train", samples=args.samples, seed=0, config=args.hotpot_config)
    texts = [e["source"] for e in examples]
    answers = [e["answer"] for e in examples]

    # Models
    use_llama = args.lambda_llama > 0.0
    use_qwen  = args.lambda_qwen  > 0.0

    llama = LMWrapper(LMConfig(model_id=args.llama_id, device=device, dtype=dtype, load_4bit=args.load_4bit)) if use_llama else None
    qwen  = LMWrapper(LMConfig(model_id=args.qwen_id,  device=device, dtype=dtype, load_4bit=args.load_4bit)) if use_qwen  else None
    if use_llama and use_qwen:
        print(f"Llama hidden size: {llama.d_model}, Qwen hidden size: {qwen.d_model}")
    elif use_llama:
        print(f"Llama hidden size: {llama.d_model}")
    else:
        print(f"Qwen hidden size: {qwen.d_model}")

    # Anchors (safe defaults)
    anchor_llama_ids = llama.tokenizer.encode(args.warm_anchor_text, add_special_tokens=False) if (use_llama and args.warm_anchor_text) else []
    anchor_qwen_ids  = qwen.tokenizer.encode(args.warm_anchor_text,  add_special_tokens=False) if (use_qwen  and args.warm_anchor_text) else []

    if args.grad_ckpt:
        if use_llama: llama.enable_gradient_checkpointing()
        if use_qwen:  qwen.enable_gradient_checkpointing()

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
            system = "You are a concise QA assistant. Use the context to answer with a short phrase only."
            return f"System: {system}\nUser: {s}\nAssistant:"
        def encode_fn(batch_texts):
            if args.encoder_use_chat_template:
                batch_texts = [_neutral_chat_wrap(t) for t in batch_texts]
            return encoder(batch_texts)

    adp_llama = Adapter(d_z=args.d_z, d_model=llama.d_model).to(device) if use_llama else None
    adp_qwen  = Adapter(d_z=args.d_z, d_model=qwen.d_model).to(device)  if use_qwen  else None

    if args.fix_adapter_scale:
        if use_llama: adp_llama.scale.requires_grad_(False); adp_llama.scale.data.fill_(1.0)
        if use_qwen:  adp_qwen.scale.requires_grad_(False);  adp_qwen.scale.data.fill_(1.0)

    # Optimizer
    params = list(encoder.parameters())
    if use_llama: params += list(adp_llama.parameters())
    if use_qwen:  params += list(adp_qwen.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Answer tokenization
    if use_llama:
        llama_ans = llama.tokenizer(answers, return_tensors="pt", padding=True, truncation=True, max_length=args.max_answer_tokens, add_special_tokens=True)
        llama_ids = llama_ans["input_ids"].to(device)
    if use_qwen:
        qwen_ans  = qwen.tokenizer(answers,  return_tensors="pt", padding=True, truncation=True, max_length=args.max_answer_tokens, add_special_tokens=True)
        qwen_ids  = qwen_ans["input_ids"].to(device)

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    # Target norms for acceptance regularizer
    target_norm_llama = llama.get_embedding_mean_norm() if use_llama else None
    target_norm_qwen  = qwen.get_embedding_mean_norm()  if use_qwen  else None

    # ===== Resume logic =====
    start_epoch = 0
    global_step = 0
    if args.resume_from or args.auto_resume:
        ckpt_path = args.resume_from or find_latest_checkpoint(args.save_dir)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"⏪ Resuming from: {ckpt_path}")
            epoch_loaded, global_loaded = load_checkpoint(
                ckpt_path, encoder, adp_llama, adp_qwen,
                optimizer=optimizer,
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
            z = encode_fn(batch_texts)  # compute once per step

            total_loss = 0.0
            # LLAMA path
            if use_llama and args.lambda_llama > 0.0:
                prefix_llama = adp_llama(z)
                y_llama = llama_ids[idx]
                loss_llama = llama.forward_with_prefix_loss(prefix_llama, y_llama, anchor_token_ids=anchor_llama_ids)
                # acceptance regularizer (mean norm toward token-embedding mean norm)
                if args.acceptance_reg > 0.0:
                    mean_norm_L = prefix_llama.norm(dim=-1).mean()
                    tgt = torch.tensor(target_norm_llama, device=mean_norm_L.device, dtype=mean_norm_L.dtype)
                    loss_llama = loss_llama + args.acceptance_reg * (mean_norm_L - tgt).pow(2)
                (args.lambda_llama * loss_llama).backward()
                total_loss += float(loss_llama.item())

            # QWEN path
            if use_qwen and args.lambda_qwen > 0.0:
                prefix_qwen = adp_qwen(z)
                y_qwen = qwen_ids[idx]
                loss_qwen = qwen.forward_with_prefix_loss(prefix_qwen, y_qwen, anchor_token_ids=anchor_qwen_ids)
                if args.acceptance_reg > 0.0:
                    mean_norm_Q = prefix_qwen.norm(dim=-1).mean()
                    tgt = torch.tensor(target_norm_qwen, device=mean_norm_Q.device, dtype=mean_norm_Q.dtype)
                    loss_qwen = loss_qwen + args.acceptance_reg * (mean_norm_Q - tgt).pow(2)
                (args.lambda_qwen * loss_qwen).backward()
                total_loss += float(loss_qwen.item())

            # grad clip + step
            try:
                torch.nn.utils.clip_grad_norm_(params, max_norm=args.max_grad_norm)
            except Exception:
                pass
            optimizer.step()

            global_step += 1
            dt = time.time() - t0
            ema_step_time = dt if ema_step_time is None else (0.9 * ema_step_time + 0.1 * dt)

            if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                if use_llama and use_qwen and (args.lambda_llama > 0 and args.lambda_qwen > 0):
                    print(f"  step {step+1}/{steps_per_epoch} | loss_total={total_loss:.4f} | sec/step~{ema_step_time:.2f}")
                else:
                    print(f"  step {step+1}/{steps_per_epoch} | loss={total_loss:.4f} | sec/step~{ema_step_time:.2f}")

            if args.debug and ((step + 1) % 100 == 0 or (step + 1) == steps_per_epoch):
                with torch.no_grad():
                    try:
                        z_std = float(z.detach().std().item())
                        z_mean_norm = float(z.detach().norm(dim=-1).mean().item())
                        if use_llama:
                            pL = prefix_llama.detach()
                            pL_std = float(pL.std().item())
                            pL_mean_norm = float(pL.norm(dim=-1).mean().item())
                            print(f"  [debug:L] a.scale={float(adp_llama.scale.detach().cpu()):.4f} "
                                  f"| Z.std={z_std:.4f} Z.mean||={z_mean_norm:.4f} "
                                  f"| p.std={pL_std:.4f} p.mean||={pL_mean_norm:.4f}")
                        if use_qwen:
                            pQ = prefix_qwen.detach()
                            pQ_std = float(pQ.std().item())
                            pQ_mean_norm = float(pQ.norm(dim=-1).mean().item())
                            print(f"  [debug:Q] a.scale={float(adp_qwen.scale.detach().cpu()):.4f} "
                                  f"| Z.std={z_std:.4f} Z.mean||={z_mean_norm:.4f} "
                                  f"| p.std={pQ_std:.4f} p.mean||={pQ_mean_norm:.4f}")
                    except Exception as _e:
                        print("  [debug] norm check failed:", _e)

            # Periodic checkpointing
            if args.save_every and (global_step % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(encoder.state_dict(), os.path.join(args.save_dir, f"encoder_step{global_step}.pt"))
                if use_llama:
                    torch.save(adp_llama.state_dict(), os.path.join(args.save_dir, f"adapter_llama_step{global_step}.pt"))
                if use_qwen:
                    torch.save(adp_qwen.state_dict(), os.path.join(args.save_dir, f"adapter_qwen_step{global_step}.pt"))
                save_checkpoint(os.path.join(args.save_dir, f"state_step{global_step}.pt"),
                                encoder, adp_llama, adp_qwen, optimizer, epoch, global_step, args)
                print(f"  ✅ Saved checkpoint at step {global_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    if use_llama:
        torch.save(adp_llama.state_dict(), os.path.join(args.save_dir, "adapter_llama.pt"))
    if use_qwen:
        torch.save(adp_qwen.state_dict(), os.path.join(args.save_dir, "adapter_qwen.pt"))
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pt"))
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
