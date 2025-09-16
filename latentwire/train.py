# latentwire/train.py
import os
import re
import time
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union
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
from latentwire.losses import k_token_ce_from_prefix, kd_first_k_prefix_vs_text
from latentwire.prefix_utils import (
    calibrate_to_embed_rms,
    bos_policy,
    first_non_bos,
    build_scaffold_ids,
    anchor_token_ids,
    tensor_rms,
    tensor_rms_d,
)

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

def _to_float(value: Union[torch.Tensor, float, int]) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


@dataclass
class ModelTrainContext:
    name: str
    wrapper: LMWrapper
    adapter: Adapter
    token_ids: torch.Tensor
    first_token_ids: torch.Tensor
    anchor_ids: List[int]
    bos_flag: Optional[bool]


def _assert_t0_alignment(tokenizer, answer_prefix: str = "Answer: "):
    """Sanity check ensuring token t=0 matches after the anchor."""
    try:
        q = "Q: Capital of France?"
        c = "C: Paris is the capital of France."
        g = "Paris"
        prompt = f"{c}\n\n{q}\n{answer_prefix}"
        ids_all = tokenizer(prompt + g, add_special_tokens=False).input_ids
        ids_pref = tokenizer(prompt, add_special_tokens=False).input_ids
        ids_gold = tokenizer(g, add_special_tokens=False).input_ids
        assert ids_all[len(ids_pref)] == ids_gold[0], "t=0 alignment failed"
        print(f"[OK] t=0 alignment for {getattr(tokenizer, 'name_or_path', 'tokenizer')}")
    except Exception as exc:
        print(f"[WARN] t=0 alignment could not be verified: {exc}")


def _build_scaffold_ids(tokenizer, texts: List[str], anchor_text: str, device: str) -> torch.Tensor:
    """Construct teacher text scaffolds (prompt + anchor) for KD."""
    suffix = anchor_text or ""
    combo = [t + suffix for t in texts]
    enc = tokenizer(combo, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
    return enc["input_ids"].to(device)


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
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Number of micro-batches to accumulate before an optimizer step.")

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
    # K-token supervision + KD
    ap.add_argument("--K", type=int, default=4, help="Number of early tokens to supervise (A1/A2).")
    ap.add_argument("--k_ce_weight", type=float, default=0.5,
                    help="Aux weight for K-token CE on first K steps.")
    ap.add_argument("--kd_first_k_weight", type=float, default=1.0,
                    help="Weight for prefix KD vs text teacher (first K steps).")
    ap.add_argument("--kd_tau", type=float, default=1.0, help="Temperature for KD.")

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
    env_dtype = os.environ.get("TORCH_DTYPE")
    if env_dtype:
        dtype_lookup = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = dtype_lookup.get(env_dtype.lower(), torch.bfloat16 if device == "cuda" else torch.float32)
    elif device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16 if args.fp16_mps else torch.float32
    else:
        dtype = torch.float32

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    model_keys = ["llama", "qwen"]
    if args.latent_shared_len is not None:
        latent_shared_len = int(args.latent_shared_len)
        latent_private_len = max(0, int(args.latent_private_len))
        total_latent_len = latent_shared_len + latent_private_len * len(model_keys)
    else:
        latent_private_len = max(0, int(args.latent_private_len))
        total_latent_len = int(args.latent_len)
        latent_shared_len = max(total_latent_len - latent_private_len * len(model_keys), 0)
    total_latent_len = max(total_latent_len, 0)

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

    # === A0 sanity check ===
    try:
        _assert_t0_alignment(llama.tokenizer, args.warm_anchor_text or "Answer: ")
        _assert_t0_alignment(qwen.tokenizer,  args.warm_anchor_text or "Answer: ")
    except Exception as exc:
        print(f"[WARN] A0 sanity check skipped/failed: {exc}")

    anchor_llama_ids = anchor_token_ids(llama, args.warm_anchor_text)
    anchor_qwen_ids = anchor_token_ids(qwen, args.warm_anchor_text)

    if args.grad_ckpt:
        llama.enable_gradient_checkpointing()
        qwen.enable_gradient_checkpointing()

    # ===== Encoder =====
    def _structure_latents(raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = raw[:, :latent_shared_len] if latent_shared_len > 0 else raw.new_zeros(raw.size(0), 0, raw.size(-1))
        private = {}
        start = latent_shared_len
        for key in model_keys:
            if latent_private_len > 0:
                private[key] = raw[:, start:start + latent_private_len]
            else:
                private[key] = raw.new_zeros(raw.size(0), 0, raw.size(-1))
            start += latent_private_len
        return {"shared": shared, "private": private}

    if args.encoder_type == "byte":
        encoder = InterlinguaEncoder(
            d_z=args.d_z,
            latent_shared_len=latent_shared_len,
            latent_private_len=latent_private_len,
            model_keys=tuple(model_keys),
        ).to(device)
        byte_tok = ByteTokenizer(max_bytes=args.max_bytes)

        def encode_fn(batch_texts):
            z_bytes = collate_bytes(batch_texts, byte_tok, device)
            return encoder(z_bytes, return_components=True)
    else:
        encoder = SimpleEncoder(d_z=args.d_z, latent_len=total_latent_len,
                                backbone=(args.encoder_backbone or "sentence-transformers/all-MiniLM-L6-v2")).to(device)

        def _neutral_chat_wrap(s: str) -> str:
            system = "You are a concise QA assistant. Use the context to answer with a short phrase only."
            return f"System: {system}\nUser: {s}\nAssistant:"

        def encode_fn(batch_texts):
            if args.encoder_use_chat_template:
                batch_texts = [_neutral_chat_wrap(t) for t in batch_texts]
            raw = encoder(batch_texts)
            return _structure_latents(raw)

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
    llama_first_ids_all = first_non_bos(llama.tokenizer, llama_ids)
    qwen_first_ids_all  = first_non_bos(qwen.tokenizer,  qwen_ids)

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

    stats_trackers = {
        "llama": {
            "rms_raw": _RunningMean(),
            "rms_cal": _RunningMean(),
            "embed_rms": llama.input_embedding_rms(),
        },
        "qwen": {
            "rms_raw": _RunningMean(),
            "rms_cal": _RunningMean(),
            "embed_rms": qwen.input_embedding_rms(),
        },
    }

    model_contexts: List[ModelTrainContext] = [
        ModelTrainContext(
            name="llama",
            wrapper=llama,
            adapter=adp_llama,
            token_ids=llama_ids,
            first_token_ids=llama_first_ids_all,
            anchor_ids=anchor_llama_ids,
            bos_flag=bos_policy(args.train_append_bos_after_prefix, anchor_llama_ids),
        ),
        ModelTrainContext(
            name="qwen",
            wrapper=qwen,
            adapter=adp_qwen,
            token_ids=qwen_ids,
            first_token_ids=qwen_first_ids_all,
            anchor_ids=anchor_qwen_ids,
            bos_flag=bos_policy(args.train_append_bos_after_prefix, anchor_qwen_ids),
        ),
    ]

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
        cur = tensor_rms_d(prefix_raw)
        return (cur - tgt).pow(2)

    params_for_clip = [p for p in optim_groups if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        g = torch.Generator(device="cpu")
        g.manual_seed(int(args.seed) + int(epoch))
        perm = torch.randperm(N, generator=g)

        for step in range(steps_per_epoch):
            t0 = time.time()
            idx = perm[step*args.batch_size : (step+1)*args.batch_size]
            batch_texts = [texts[i] for i in idx.tolist()]

            scaffolds = {
                ctx.name: build_scaffold_ids(
                    ctx.wrapper.tokenizer, batch_texts, args.warm_anchor_text or "", device
                )
                for ctx in model_contexts
            }

            encoded_latents = encode_fn(batch_texts)
            shared_latents = encoded_latents["shared"]
            private_latents = encoded_latents["private"]
            model_latents = {
                name: torch.cat([shared_latents, private_latents[name]], dim=1)
                for name in model_keys
            }

            per_model_losses: Dict[str, Dict[str, torch.Tensor]] = {}
            total_model_loss = 0.0
            penalty = torch.zeros((), device=device)
            rms_pen = torch.zeros((), device=device)

            for ctx in model_contexts:
                prefix_raw = ctx.adapter(model_latents[ctx.name])
                prefix = calibrate_to_embed_rms(prefix_raw, ctx.wrapper)
                targets = ctx.token_ids[idx]
                loss_tf = ctx.wrapper.forward_with_prefix_loss(
                    prefix, targets, anchor_token_ids=ctx.anchor_ids
                )

                if args.first_token_ce_weight and args.first_token_ce_weight > 0.0:
                    logits_first = ctx.wrapper.first_token_logits_from_prefix(
                        prefix,
                        anchor_token_text=(args.warm_anchor_text or None),
                        append_bos_after_prefix=ctx.bos_flag,
                    )
                    first_targets = ctx.first_token_ids[idx]
                    loss_first = nn.functional.cross_entropy(logits_first.float(), first_targets)
                else:
                    loss_first = torch.zeros((), device=device)

                if args.k_ce_weight and args.k_ce_weight > 0.0:
                    loss_kce = k_token_ce_from_prefix(
                        ctx.wrapper,
                        prefix,
                        targets,
                        K=args.K,
                        anchor_ids=ctx.anchor_ids,
                        append_bos_after_prefix=ctx.bos_flag,
                    )
                else:
                    loss_kce = torch.zeros((), device=device)

                if args.kd_first_k_weight and args.kd_first_k_weight > 0.0:
                    loss_kd = kd_first_k_prefix_vs_text(
                        ctx.wrapper,
                        ctx.wrapper,
                        prefix,
                        scaffolds[ctx.name],
                        targets,
                        K=args.K,
                        tau=args.kd_tau,
                        anchor_ids=ctx.anchor_ids,
                        append_bos_after_prefix=ctx.bos_flag,
                    )
                else:
                    loss_kd = torch.zeros((), device=device)

                model_loss = (
                    loss_tf
                    + args.first_token_ce_weight * loss_first
                    + args.k_ce_weight * loss_kce
                    + args.kd_first_k_weight * loss_kd
                )
                total_model_loss = total_model_loss + model_loss

                penalty = penalty + scale_penalty(ctx.adapter)
                rms_pen = rms_pen + rms_raw_penalty(prefix_raw, ctx.wrapper)

                rms_raw_val = tensor_rms(prefix_raw)
                rms_cal_val = tensor_rms(prefix)
                stats_trackers[ctx.name]["rms_raw"].update(rms_raw_val)
                stats_trackers[ctx.name]["rms_cal"].update(rms_cal_val)

                per_model_losses[ctx.name] = {
                    "tf": loss_tf,
                    "first": loss_first,
                    "kce": loss_kce,
                    "kd": loss_kd,
                    "rms_raw": rms_raw_val,
                    "rms_cal": rms_cal_val,
                }

            loss = (
                total_model_loss / float(len(model_contexts))
                + args.scale_l2 * penalty
                + args.adapter_rms_l2 * rms_pen
            )

            if not torch.isfinite(loss):
                print("NaN/Inf loss; skipping step")
                continue

            loss_backward = loss / float(grad_accum_steps)
            loss_backward.backward()

            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)
            if should_step:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    total_norm = float(
                        torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))
                    )
                else:
                    total_norm = float("nan")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                total_norm = float("nan")

            # Grad norm (monitor) – encoder only as a proxy
            global_step += 1
            dt = time.time() - t0
            ema_step_time = dt if (locals().get("ema_step_time", None) is None) else (0.9 * locals()["ema_step_time"] + 0.1 * dt)

            if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                parts = [
                    f"  step  {step+1}/{steps_per_epoch}",
                    f"grad_norm={total_norm:.2f}",
                    f"sec/step~{dt:.2f}",
                ]
                for ctx in model_contexts:
                    metrics = per_model_losses[ctx.name]
                    parts.append(
                        f"{ctx.name}: tf={_to_float(metrics['tf']):.4f}"
                        f" first={_to_float(metrics['first']):.4f}"
                        f" kCE={_to_float(metrics['kce']):.4f}"
                        f" KD={_to_float(metrics['kd']):.4f}"
                    )
                    if args.scale_l2 > 0.0:
                        parts.append(f"scale_pen({ctx.name})={scale_penalty(ctx.adapter).item():.4e}")
                parts.append(f"K={args.K} tau={args.kd_tau:.2f}")
                if args.save_training_stats:
                    stats_msgs = []
                    for ctx in model_contexts:
                        tracker = stats_trackers[ctx.name]
                        stats_msgs.append(
                            f"{ctx.name}: rms_raw~{tracker['rms_raw'].mean:.4f}"
                            f" rms_cal~{tracker['rms_cal'].mean:.4f}"
                            f" embed_rms~{tracker['embed_rms']:.5f}"
                        )
                    parts.append("stats=[" + "; ".join(stats_msgs) + "]")
                print(" | ".join(parts))

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
                    "K": args.K,
                    "k_ce_weight": args.k_ce_weight,
                    "kd_first_k_weight": args.kd_first_k_weight,
                    "kd_tau": args.kd_tau,
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
        "K": args.K,
        "k_ce_weight": args.k_ce_weight,
        "kd_first_k_weight": args.kd_first_k_weight,
        "kd_tau": args.kd_tau,
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
            name: {
                "rms_mean_raw": tracker["rms_raw"].mean,
                "rms_mean_cal": tracker["rms_cal"].mean,
                "embed_rms": tracker["embed_rms"],
                "count": tracker["rms_cal"].n,
            }
            for name, tracker in stats_trackers.items()
        }
        with open(os.path.join(args.save_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📝 Saved training_stats.json: {stats}")


if __name__ == "__main__":
    main()
