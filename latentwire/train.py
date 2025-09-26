# latentwire/train.py
import os
import re
import time
import json
import math
import argparse
import random
import ast
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any, Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
from latentwire.core_utils import (
    capture_env_snapshot,
    patch_dataloader_defaults,
    apply_anchor_normalization,
    collate_bytes,
    calibrate_to_embed_rms,
    bos_policy,
    first_non_bos,
    anchor_token_ids,
    tensor_rms,
    tensor_rms_d,
    assistant_header_anchor,
    SYSTEM_PROMPT,
    split_user_and_anchor,
)

from latentwire.models import (
    InterlinguaEncoder,
    Adapter,
    LatentRefiner,
    GistReconstructionHead,
    LMWrapper,
    LMConfig,
    ByteTokenizer,
    SimpleEncoder,
    STQueryEncoder,
    DeepPrefixGenerator,
    apply_lora_if_requested,
    apply_prefix_if_requested,
)
from latentwire.checkpointing import save_latest_checkpoint

def _align_optimizer_state_to_param_devices(optimizer):
    """Ensure optimizer state tensors live on the same device as their params (multi-GPU safety)."""
    try:
        for param, st in optimizer.state.items():
            if not isinstance(st, dict):
                continue
            pdev = getattr(param, "device", None)
            if pdev is None:
                continue
            for k, v in list(st.items()):
                try:
                    if torch.is_tensor(v) and v.device != pdev:
                        st[k] = v.to(pdev, non_blocking=True)
                except Exception:
                    pass
    except Exception:
        pass
from latentwire.data import load_examples
from latentwire.losses import (
    k_token_ce_from_prefix,
    kd_first_k_prefix_vs_text,
    kd_hidden_states_first_k,
)

DEFAULT_SEED = 42
DEFAULT_ANSWER_PREFIX = "Answer: "


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


def _debug_print_optimizer_state_devices(optimizer: optim.Optimizer, limit: int = 8) -> None:
    if getattr(_debug_print_optimizer_state_devices, "_printed", False):
        return
    try:
        lines = []
        for idx, (param, state) in enumerate(optimizer.state.items()):
            param_dev = str(getattr(param, "device", "None"))
            state_devs = {}
            if isinstance(state, dict):
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state_devs[key] = str(value.device)
            lines.append(f"param_dev={param_dev} state_devs={state_devs}")
            if idx + 1 >= limit:
                break
        if lines:
            print("[DEBUG] Optimizer state devices:\n  " + "\n  ".join(lines), flush=True)
    except Exception:
        pass
    _debug_print_optimizer_state_devices._printed = True


def load_checkpoint(
    path: str,
    encoder: InterlinguaEncoder,
    adapters: Dict[str, Adapter],
    refiner: Optional[LatentRefiner] = None,
    deep_prefix_generators: Optional[Dict[str, DeepPrefixGenerator]] = None,
    gist_heads: Optional[Dict[str, GistReconstructionHead]] = None,
    optimizer: Optional[optim.Optimizer] = None,
    strict: bool = True,
    device: str = "cpu",
) -> Tuple[int, int]:
    if path and os.path.isfile(path):
        state = _safe_load(path, map_location="cpu")
        ckpt_dir = os.path.dirname(path)
    elif path and os.path.isdir(path):
        state = {}
        ckpt_dir = path
    else:
        state = {}
        ckpt_dir = None

    enc_loaded = False
    refiner_loaded = refiner is None
    adapters_loaded: Dict[str, bool] = {name: False for name in adapters.keys()}
    deep_prefix_loaded: Dict[str, bool] = (
        {name: False for name in (deep_prefix_generators or {}).keys()}
        if deep_prefix_generators else {}
    )
    gist_loaded: Dict[str, bool] = (
        {name: False for name in (gist_heads or {}).keys()}
        if gist_heads else {}
    )
    if isinstance(state, dict) and "encoder" in state:
        try:
            encoder.load_state_dict(state["encoder"], strict=strict)
            enc_loaded = True
        except Exception as exc:
            print(f"   -> failed to load encoder weights from state.pt ({exc}); will try .pt files")
            enc_loaded = False

        if enc_loaded:
            for name, adapter in adapters.items():
                key = f"adp_{name}"
                alt_key = {
                    "llama": "adp_llama",
                    "qwen": "adp_qwen",
                }.get(name, key)
                state_key = key if key in state else alt_key
                if state_key in state:
                    try:
                        adapter.load_state_dict(state[state_key], strict=strict)
                        adapters_loaded[name] = True
                    except Exception as exc:
                        print(f"   -> adapter '{name}' from state.pt failed ({exc}); will retry from disk")
                        adapters_loaded[name] = False
                else:
                    print(f"   -> adapter '{name}' missing in state.pt; will retry from disk")
            if deep_prefix_generators:
                for name, generator in deep_prefix_generators.items():
                    key = f"deep_prefix_{name}"
                    if key in state:
                        try:
                            generator.load_state_dict(state[key], strict=strict)
                            deep_prefix_loaded[name] = True
                        except Exception as exc:
                            print(f"   -> deep prefix '{name}' from state.pt failed ({exc}); will retry from disk")
                            deep_prefix_loaded[name] = False
                    else:
                        print(f"   -> deep prefix '{name}' missing in state.pt; will retry from disk")
            if gist_heads:
                for name, head in gist_heads.items():
                    key = f"gist_{name}"
                    if key in state:
                        try:
                            head.load_state_dict(state[key], strict=strict)
                            gist_loaded[name] = True
                        except Exception as exc:
                            print(f"   -> gist head '{name}' from state.pt failed ({exc}); will retry from disk")
                            gist_loaded[name] = False
                    else:
                        print(f"   -> gist head '{name}' missing in state.pt; will retry from disk")
        if refiner is not None and "refiner" in state:
            try:
                refiner.load_state_dict(state["refiner"], strict=strict)
                refiner_loaded = True
            except Exception as exc:
                print(f"   -> refiner from state.pt failed ({exc}); will retry from disk")
                refiner_loaded = False
        else:
            refiner_loaded = refiner is None
    deep_prefix_ok = True if not deep_prefix_generators else all(deep_prefix_loaded.values())
    gist_ok = True if not gist_heads else all(gist_loaded.values())
    if enc_loaded and all(adapters_loaded.values()) and refiner_loaded and deep_prefix_ok and gist_ok:
        suffix = "encoder/adapters"
        if deep_prefix_generators:
            suffix += "/deep_prefix"
        if refiner is not None:
            suffix += "/refiner"
        if gist_heads:
            suffix += "/gist"
        print(f"   -> loaded {suffix} FROM state.pt")

    if (not enc_loaded or not all(adapters_loaded.values()) or not refiner_loaded) and ckpt_dir:
        enc_path = os.path.join(ckpt_dir, "encoder.pt")
        missing: List[str] = []
        if not enc_loaded:
            if os.path.isfile(enc_path):
                encoder.load_state_dict(_safe_load(enc_path, map_location=device), strict=strict)
                enc_loaded = True
            else:
                missing.append(enc_path)

        for name, adapter in adapters.items():
            if adapters_loaded.get(name):
                continue
            adapter_path = os.path.join(ckpt_dir, f"adapter_{name}.pt")
            legacy_path = os.path.join(ckpt_dir, f"adapter_{name if name in ('llama','qwen') else name}.pt")
            path_to_use = adapter_path if os.path.isfile(adapter_path) else legacy_path
            if os.path.isfile(path_to_use):
                adapter.load_state_dict(_safe_load(path_to_use, map_location=device), strict=strict)
                adapters_loaded[name] = True
            else:
                missing.append(adapter_path)

        if deep_prefix_generators:
            for name, generator in deep_prefix_generators.items():
                if deep_prefix_loaded.get(name):
                    continue
                prefix_path = os.path.join(ckpt_dir, f"deep_prefix_{name}.pt")
                if os.path.isfile(prefix_path):
                    generator.load_state_dict(_safe_load(prefix_path, map_location=device), strict=strict)
                    deep_prefix_loaded[name] = True
                else:
                    missing.append(prefix_path)

        if refiner is not None and not refiner_loaded:
            refiner_path = os.path.join(ckpt_dir, "refiner.pt")
            if os.path.isfile(refiner_path):
                refiner.load_state_dict(_safe_load(refiner_path, map_location=device), strict=strict)
                refiner_loaded = True
            else:
                missing.append(refiner_path)
        if gist_heads:
            for name, head in gist_heads.items():
                if gist_loaded.get(name):
                    continue
                gist_path = os.path.join(ckpt_dir, f"gist_{name}.pt")
                if os.path.isfile(gist_path):
                    head.load_state_dict(_safe_load(gist_path, map_location=device), strict=strict)
                    gist_loaded[name] = True
                else:
                    missing.append(gist_path)

        if missing:
            raise FileNotFoundError(
                "Missing checkpoint artifacts: " + ", ".join(missing)
            )
        else:
            suffix = "encoder/adapters"
            if deep_prefix_generators:
                suffix += "/deep_prefix"
            if refiner is not None:
                suffix += "/refiner"
            if gist_heads:
                suffix += "/gist"
            print(f"   -> loaded {suffix} FROM encoder.pt + adapter_*.pt")

    if optimizer is not None and isinstance(state, dict):
        opt_state = state.get("optimizer", None) or state.get("optim", None)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except ValueError as exc:
                print(f"[WARN] Optimizer state incompatible; continuing with fresh optimizer ({exc})")
            # Important: keep optimizer state tensors on the same device as *their* params.
            # Do NOT mass-move to a single device, because adapters live on different GPUs.
            _align_optimizer_state_to_param_devices(optimizer)
            _debug_print_optimizer_state_devices(optimizer)
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


@dataclass
class ModelTrainContext:
    name: str
    wrapper: LMWrapper
    adapter: Adapter
    token_ids: torch.Tensor
    first_token_ids: torch.Tensor
    anchor_ids: List[int]
    bos_flag: Optional[bool]
    answer_lengths: torch.Tensor
    anchor_text: str
    anchor_mode: str


def _primary_device(wrapper: LMWrapper) -> torch.device:
    return next(wrapper.model.parameters()).device


def _assert_t0_alignment(tokenizer, answer_prefix: str = "Answer: ", skip_if_chat: bool = False):
    """Sanity check: the first gold token should appear immediately after the anchor."""
    if skip_if_chat:
        return
    try:
        q = "Q: Capital of France?"
        c = "C: Paris is the capital of France."
        g = "Paris"
        prompt = f"{c}\n\n{q}\n{answer_prefix}"

        ids_all = tokenizer.encode(prompt + g, add_special_tokens=False)
        ids_pref = tokenizer.encode(prompt, add_special_tokens=False)
        ids_gold = tokenizer.encode(g, add_special_tokens=False)

        if len(ids_all) == len(ids_pref):  # tokenizer swallowed the answer prefix boundary
            prompt_sp = prompt + " "
            ids_all = tokenizer.encode(prompt_sp + g, add_special_tokens=False)
            ids_pref = tokenizer.encode(prompt_sp, add_special_tokens=False)

        assert ids_gold, "gold tokenized to empty"
        assert len(ids_all) > len(ids_pref), "concatenation produced no new tokens"
        got = ids_all[len(ids_pref)]
        expect = ids_gold[0]
        assert got == expect, f"t=0 mismatch: got {got}, expected {expect}"
        print(f"[OK] t=0 alignment for {getattr(tokenizer, 'name_or_path', 'tokenizer')}")
    except Exception as exc:
        print(f"[WARN] t=0 alignment failed: {exc}")


def _render_chat_prompt(tokenizer, user_text: str, system_prompt: Optional[str]) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered:
            return rendered
    except Exception:
        pass
    system_block = f"System: {system_prompt}\n" if system_prompt else ""
    return f"{system_block}User: {user_text}\nAssistant:"


def _answer_lengths(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        lengths = torch.full((token_ids.size(0),), token_ids.size(1), device=token_ids.device)
    else:
        lengths = token_ids.ne(int(pad_id)).sum(dim=1)
    return lengths


def main():
    ap = argparse.ArgumentParser()
    # Models & data
    ap.add_argument("--llama_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--qwen_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--llama_device_map", type=str, default=None,
                    help=("Device map for the Llama wrapper (e.g., 0, 'auto', or JSON dict). "
                          "Pins the model to a subset of GPUs when running multi-model training."))
    ap.add_argument("--qwen_device_map", type=str, default=None,
                    help="Device map for the Qwen wrapper (e.g., 1, 'auto', or JSON dict).")
    ap.add_argument("--require_cuda", type=str, default="yes", choices=["yes", "no"],
                    help="Abort immediately if CUDA is not available (default: yes).")
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot", "squad", "squad_v2"])
    ap.add_argument("--models", type=str, default="llama,qwen",
                    help="Comma-separated subset of models to train (subset of llama,qwen).")
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
    ap.add_argument("--latent_shared_len", type=int, default=None,
                    help="Optional explicit shared latent length; overrides derived value.")
    ap.add_argument("--latent_private_len", type=int, default=0,
                    help="Per-model private latent length (combined with shared length).")
    ap.add_argument("--d_z", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=512)
    ap.add_argument("--encoder_type", type=str, default="byte", choices=["byte", "simple-st", "stq"])
    ap.add_argument("--encoder_use_chat_template", action="store_true",
                    help="Wrap encoder input with a neutral chat-style header (SimpleEncoder only).")
    ap.add_argument("--encoder_backbone", type=str, default=None,
                    help="Optional SentenceTransformer backbone when --encoder_type=simple-st")
    ap.add_argument("--hf_encoder_id", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HF encoder id for --encoder_type=stq (frozen).")
    ap.add_argument("--max_enc_tokens", type=int, default=1024,
                    help="Max source tokens for the HF encoder when --encoder_type=stq.")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Freeze encoder parameters (e.g., during Stage B prefix-tuning).")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Apply tokenizer.apply_chat_template when constructing teacher scaffolds and anchors.")

    # Training & stability
    ap.add_argument("--max_answer_tokens", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale_l2", type=float, default=0.05,
                    help="L2 penalty weight to keep adapter.scale near 1.0; set 0 to disable.")
    ap.add_argument("--adapter_rms_l2", type=float, default=0.0,
                    help="Optional penalty to pull RAW adapter RMS toward input embedding RMS.")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Clip grad norm to this value (set <=0 to disable).")
    ap.add_argument("--grad_diag_interval", type=int, default=0,
                    help="If >0, compute gradient-norm diagnostics every N global steps (0 disables).")
    ap.add_argument("--grad_diag_components", type=str,
                    default="tf,first,kce,kd,align,latent_align,latent_prefix_align",
                    help="Comma-separated loss names to include in gradient diagnostics (e.g., 'tf,first,kd').")
    ap.add_argument("--diagnostic_log", type=str, default="",
                    help="Optional JSONL path to append diagnostic summaries each log interval.")
    ap.add_argument("--adapter_freeze_scale", action="store_true",
                    help="If set, fix adapter.scale at 1.0 (no learning).")
    ap.add_argument("--first_token_ce_weight", type=float, default=0.5,
                    help="Weight for the first-token CE objective; 0 disables.")
    ap.add_argument("--first_token_ce_schedule", type=str, default="none", choices=["none", "cosine"],
                    help="Optional schedule for first-token CE weights (default: none).")
    ap.add_argument("--first_token_ce_peak", type=float, default=None,
                    help="Peak first-token CE weight during warmup when using a schedule.")
    ap.add_argument("--first_token_ce_warmup_frac", type=float, default=0.4,
                    help="Fraction of total steps to hold the peak first-token CE weight before cosine decay.")
    ap.add_argument("--first_token_autoscale", type=str, default="yes", choices=["yes", "no"],
                    help="If 'yes', dynamically boost first-token CE weight when latent first-token loss stays larger than the teacher-forced loss.")
    ap.add_argument("--train_append_bos_after_prefix", type=str, default="no",
                    choices=["auto","yes","no"],
                    help="Controls BOS appending when computing first-token CE (train).")
    ap.add_argument("--adapter_hidden_mult", type=int, default=2,
                    help="Hidden width multiplier for the adapter MLP.")
    ap.add_argument("--adapter_dropout", type=float, default=0.0,
                    help="Dropout probability for adapter MLP hidden states (0 disables).")
    ap.add_argument("--adapter_colorize", action="store_true",
                    help="If set, add per-dim colorizer to align adapter outputs with LM embeddings.")
    ap.add_argument("--no_adapter_metadata", action="store_false", dest="adapter_metadata",
                    help="Disable positional/answer-length metadata injection in the adapter.")
    ap.set_defaults(adapter_metadata=True)
    ap.add_argument("--manifold_stat_weight", type=float, default=0.0,
                    help="Optional weight for μ/σ matching loss (1e-3..5e-3 recommended).")
    ap.add_argument("--state_kd_weight", type=float, default=0.0,
                    help="Weight for hidden-state KD on first K steps (0 disables).")
    ap.add_argument("--state_kd_layers", type=str, default="0,1,2",
                    help="Comma-separated transformer layer indices for hidden-state KD.")
    ap.add_argument("--use_gist_head", action="store_true",
                    help="Enable gist reconstruction head that rebuilds teacher prompts from the latent wire.")
    ap.add_argument("--gist_target_len", type=int, default=48,
                    help="Number of tokens to reconstruct with the gist head.")
    ap.add_argument("--gist_hidden", type=int, default=512,
                    help="Hidden dimension inside the gist head MLP.")
    ap.add_argument("--gist_layers", type=int, default=2,
                    help="Number of residual MLP blocks in the gist head.")
    ap.add_argument("--gist_dropout", type=float, default=0.1,
                    help="Dropout applied inside the gist head.")
    ap.add_argument("--gist_weight", type=float, default=0.0,
                    help="Weight for the gist reconstruction loss (0 disables).")
    ap.add_argument("--gist_mask_prob", type=float, default=0.15,
                    help="Probability of masking each gist target token when computing the loss (simulates gist masking).")
    # PEFT toggles
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="auto",
                    help="Comma-separated module list or presets (auto, attn_mlp_firstN:12, ...).")
    ap.add_argument("--use_prefix", action="store_true")
    ap.add_argument("--prefix_tokens", type=int, default=16)
    ap.add_argument("--prefix_projection", action="store_true")
    ap.add_argument("--peft_prefix_all_layers", type=str, default="yes",
                    help="yes/no toggle to apply prefix adapters across every transformer layer.")
    ap.add_argument("--use_deep_prefix", action="store_true",
                    help="Enable learned per-layer prefixes derived from the latent interlingua.")
    ap.add_argument("--deep_prefix_len", type=int, default=None,
                    help="Number of latent slots used to seed the deep prefixes (defaults to shared latent length).")
    ap.add_argument("--deep_prefix_dropout", type=float, default=0.1,
                    help="Dropout probability applied inside the deep prefix generator.")
    # K-token supervision + KD
    ap.add_argument("--K", type=int, default=4, help="Number of early tokens to supervise (A1/A2).")
    ap.add_argument("--adaptive_k_start", type=int, default=None,
                    help="Optional starting K for curriculum (defaults to --K).")
    ap.add_argument("--adaptive_k_end", type=int, default=None,
                    help="Optional final K for curriculum (defaults to --K).")
    ap.add_argument("--latent_keep_start", type=float, default=1.0,
                    help="Starting keep probability for latent dropout curriculum (1.0 = keep all).")
    ap.add_argument("--latent_keep_end", type=float, default=1.0,
                    help="Final keep probability for latent dropout curriculum.")
    ap.add_argument("--latent_keep_power", type=float, default=1.0,
                    help="Exponent controlling schedule shape (1.0 linear, >1 later drop).")
    ap.add_argument("--warmup_text_latent_steps", type=int, default=0,
                    help="Number of initial optimizer steps to alternate text vs latent teacher forcing (0 disables).")
    ap.add_argument("--warmup_text_latent_epochs", type=float, default=0.0,
                    help="Alternative way to specify warm-up length via epochs (e.g., 1.0 for first epoch).")
    ap.add_argument("--warmup_align_tokens", type=int, default=1,
                    help="During warm-up text steps, align this many leading answer tokens (0 disables alignment).")
    ap.add_argument("--warmup_align_weight", type=float, default=1.0,
                    help="Weight for the warm-up embedding alignment loss (text-mode steps only).")
    ap.add_argument("--warmup_text_teacher_weight", type=float, default=1.0,
                    help="Weight for the teacher-forced text loss during warm-up text steps.")
    ap.add_argument("--warmup_text_latent_weight", type=float, default=0.2,
                    help="Multiplier applied to latent losses on warm-up text batches (0 disables latent CE on those steps).")
    ap.add_argument("--warmup_text_latent_weight_end", type=float, default=1.0,
                    help="Target latent-loss multiplier once warm-up completes (tail text batches use this value).")
    ap.add_argument("--warmup_tail_prob", type=float, default=0.0,
                    help="After the warm-up window, continue sampling text batches with this probability (0 disables).")
    ap.add_argument("--latent_align_weight", type=float, default=0.0,
                    help="Weight for matching latent prefix embeddings to the teacher's first token embedding during latent batches.")
    ap.add_argument("--latent_prefix_align_weight", type=float, default=0.0,
                    help="Weight for aligning the entire latent prefix to the teacher's token embeddings (first slots).")
    ap.add_argument("--latent_align_metric", type=str, default="cosine",
                    choices=["mse", "cosine", "both"],
                    help="Distance metric for latent alignment losses (default cosine).")
    ap.add_argument("--k_ce_weight", type=float, default=0.5,
                    help="Aux weight for K-token CE on first K steps.")
    ap.add_argument("--kd_first_k_weight", type=float, default=1.0,
                    help="Weight for prefix KD vs text teacher (first K steps).")
    ap.add_argument("--kd_tau", type=float, default=1.0, help="Temperature for KD.")
    ap.add_argument("--latent_refiner_layers", type=int, default=0,
                    help="If >0, use a Transformer refiner with this many layers on latent slots before adapters.")
    ap.add_argument("--latent_refiner_heads", type=int, default=4,
                    help="Number of attention heads for the latent refiner (when enabled).")

    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--sequential_models", action="store_true")
    ap.add_argument("--llama_devices", type=str, default=None,
                    help="Comma-separated CUDA device ids reserved for Llama (e.g., '0,1').")
    ap.add_argument("--qwen_devices", type=str, default=None,
                    help="Comma-separated CUDA device ids reserved for Qwen (e.g., '2,3').")
    ap.add_argument("--gpu_mem_gib", type=float, default=78.0,
                    help="Per-GPU memory budget (GiB) when constraining auto device maps.")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--fp16_mps", action="store_true")
    ap.add_argument("--warm_anchor_text", type=str, default="",
                    help="Optional anchor tokens AFTER latent prefix during training (text mode).")
    ap.add_argument(
        "--warm_anchor_mode",
        type=str,
        default="auto",
        choices=["auto", "text", "chat", "none"],
        help="How to choose the training anchor: 'text' uses --warm_anchor_text, 'chat' injects the"
             " tokenizer's assistant header, 'none' disables anchors, 'auto' matches legacy behaviour.",
    )
    ap.add_argument(
        "--max_anchor_tokens",
        type=int,
        default=32,
        help="Upper bound on the number of discrete anchor tokens to inject (prevents chat headers"
             " from ballooning the prefix). Set <=0 to disable truncation.",
    )
    ap.add_argument("--debug", action="store_true")

    # Checkpointing
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save the latest checkpoint every N steps and prune old files.")
    ap.add_argument("--resume_from", type=str, default="", help="Path to state.pt or directory containing checkpoints. Empty string disables resume.")
    ap.add_argument("--auto_resume", action="store_true")
    ap.add_argument("--no_load_optimizer", action="store_true")
    ap.add_argument("--no_load_lr_scheduler", action="store_true")
    ap.add_argument("--reset_epoch", action="store_true", help="When resuming, ignore stored epoch/step counters and restart from zero.")

    # Training stats (for eval-time calibration)
    ap.add_argument("--save_training_stats", action="store_true", help="Record running mean of prefix RMS per model and save to training_stats.json")

    args = ap.parse_args()
    # global runtime patches
    patch_dataloader_defaults()
    apply_anchor_normalization(args)

    grad_diag_interval = max(0, int(getattr(args, "grad_diag_interval", 0)))
    grad_diag_components = [
        token.strip().lower()
        for token in (getattr(args, "grad_diag_components", "") or "").split(",")
        if token.strip()
    ]
    grad_diag_components = list(dict.fromkeys(grad_diag_components))  # preserve order, dedupe
    grad_diag_component_set = set(grad_diag_components)
    diagnostic_log_path = (getattr(args, "diagnostic_log", "") or "").strip()
    if diagnostic_log_path:
        diag_dir = os.path.dirname(diagnostic_log_path)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        else:
            os.makedirs(".", exist_ok=True)
        try:
            with open(diagnostic_log_path, "a") as _f:
                pass
        except Exception as exc:
            print(f"[WARN] Unable to open diagnostic log '{diagnostic_log_path}': {exc}")
            diagnostic_log_path = ""

    # Device + dtype
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if args.require_cuda.lower() != "no":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            print("[FATAL] CUDA unavailable. torch.cuda.is_available()=", torch.cuda.is_available(),
                  "count=", torch.cuda.device_count())
            print("  CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
            try:
                subprocess.run(["nvidia-smi"], check=False)
            except Exception as exc:
                print("  nvidia-smi not runnable:", exc)
            raise SystemExit(2)
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

    supported_models = ["llama", "qwen"]
    requested_models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not requested_models:
        requested_models = supported_models
    for name in requested_models:
        if name not in supported_models:
            raise ValueError(f"Unknown model '{name}'. Choose from {supported_models}.")
    model_keys = requested_models

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    adaptive_k_start = int(args.adaptive_k_start) if args.adaptive_k_start is not None else args.K
    adaptive_k_end = int(args.adaptive_k_end) if args.adaptive_k_end is not None else args.K
    latent_keep_start = float(args.latent_keep_start)
    latent_keep_end = float(args.latent_keep_end)
    latent_keep_power = max(1e-6, float(args.latent_keep_power))
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
    def _build_max_memory(devices_csv: Optional[str], budget_gib: float):
        if devices_csv is None or not torch.cuda.is_available():
            return None
        devs: List[int] = []
        for item in devices_csv.split(","):
            name = item.strip()
            if not name:
                continue
            try:
                dev_id = int(name)
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device id '{name}' in '{devices_csv}'") from exc
            devs.append(dev_id)
        if not devs:
            return None
        max_mem = {}
        for idx in range(torch.cuda.device_count()):
            max_mem[idx] = f"{int(budget_gib)}GiB" if idx in devs else "0GiB"
        return max_mem

    llama_device_map = _parse_device_map(args.llama_device_map)
    qwen_device_map = _parse_device_map(args.qwen_device_map)

    llama_max_memory = _build_max_memory(args.llama_devices, args.gpu_mem_gib)
    qwen_max_memory = _build_max_memory(args.qwen_devices, args.gpu_mem_gib)

    if llama_device_map is None and llama_max_memory is not None and device == "cuda":
        llama_device_map = "auto"
    if qwen_device_map is None and qwen_max_memory is not None and device == "cuda":
        qwen_device_map = "auto"

    llama = None
    if "llama" in model_keys:
        llama = LMWrapper(LMConfig(
            model_id=args.llama_id,
            device=device,
            dtype=dtype,
            load_4bit=args.load_4bit,
            device_map=llama_device_map,
            max_memory=llama_max_memory,
        ))
    qwen = None
    if "qwen" in model_keys:
        qwen = LMWrapper(LMConfig(
            model_id=args.qwen_id,
            device=device,
            dtype=dtype,
            load_4bit=args.load_4bit,
            device_map=qwen_device_map,
            max_memory=qwen_max_memory,
        ))

    wrappers_in_use: List[LMWrapper] = [w for w in (llama, qwen) if w is not None]
    for wrapper in wrappers_in_use:
        try:
            if hasattr(wrapper.model.config, "use_cache"):
                wrapper.model.config.use_cache = False
        except Exception:
            pass

    def _collect_trainable(module: nn.Module) -> List[nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    extra_llama_params: List[nn.Parameter] = []
    extra_qwen_params: List[nn.Parameter] = []

    if args.use_lora:
        lora_cfg = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        }
        if llama is not None:
            llama.model = apply_lora_if_requested(llama.model, lora_cfg, args.llama_id)
            extra_llama_params.extend(_collect_trainable(llama.model))
            llama.model.train()
            llama.input_embed = llama.model.get_input_embeddings()
        if qwen is not None:
            qwen.model = apply_lora_if_requested(qwen.model, lora_cfg, args.qwen_id)
            extra_qwen_params.extend(_collect_trainable(qwen.model))
            qwen.model.train()
            qwen.input_embed = qwen.model.get_input_embeddings()

    if args.use_prefix:
        prefix_cfg = {
            "tokens": args.prefix_tokens,
            "projection": args.prefix_projection,
            "all_layers": str(args.peft_prefix_all_layers).lower() != "no",
        }
        if llama is not None:
            llama.model = apply_prefix_if_requested(llama.model, prefix_cfg, llama.tokenizer)
            extra_llama_params = _collect_trainable(llama.model)
            llama.model.train()
            llama.input_embed = llama.model.get_input_embeddings()
        if qwen is not None:
            qwen.model = apply_prefix_if_requested(qwen.model, prefix_cfg, qwen.tokenizer)
            extra_qwen_params = _collect_trainable(qwen.model)
            qwen.model.train()
            qwen.input_embed = qwen.model.get_input_embeddings()

    if (args.use_lora or args.use_prefix) and not (extra_llama_params or extra_qwen_params):
        raise RuntimeError("No trainable PEFT parameters detected – check LoRA/Prefix flags")

    if llama is not None and qwen is not None:
        print(f"Llama hidden size: {llama.d_model}, Qwen hidden size: {qwen.d_model}")
    elif llama is not None:
        print(f"Llama hidden size: {llama.d_model}")
    elif qwen is not None:
        print(f"Qwen hidden size: {qwen.d_model}")
    try:
        if llama is not None:
            print("[DeviceMap] Llama:", getattr(llama.model, "hf_device_map", None))
        if qwen is not None:
            print("[DeviceMap] Qwen :", getattr(qwen.model, "hf_device_map", None))
    except Exception:
        pass

    strip_anchor_literal = args.warm_anchor_text if args.warm_anchor_text else DEFAULT_ANSWER_PREFIX
    if strip_anchor_literal and not strip_anchor_literal.endswith(" "):
        strip_anchor_literal = strip_anchor_literal + " "

    embed_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    if llama is not None:
        embed_stats["llama"] = llama.embedding_stats()
    if qwen is not None:
        embed_stats["qwen"] = qwen.embedding_stats()

    def _anchor_text_for(wrapper, fallback: str) -> Tuple[str, str]:
        requested = (getattr(args, "warm_anchor_mode", "auto") or "auto").lower()
        mode = requested

        def _ensure_trailing_space(txt: str) -> str:
            if txt and not txt.endswith(" "):
                return txt + " "
            return txt

        if mode == "none":
            return "", "none"
        if mode == "chat":
            header = assistant_header_anchor(wrapper.tokenizer) or ""
            literal = DEFAULT_ANSWER_PREFIX if DEFAULT_ANSWER_PREFIX else ""
            if literal and not literal.endswith(" "):
                literal = literal + " "
            return header + literal, "chat"
        if mode == "text":
            base = fallback or ""
            if not base and args.use_chat_template:
                base = "Answer: "
            return _ensure_trailing_space(base), "text"

        # auto (legacy behaviour): prefer explicit text, otherwise chat header when templates are used
        base = fallback or ""
        if base:
            return _ensure_trailing_space(base), "text"
        if args.use_chat_template:
            anchor = assistant_header_anchor(wrapper.tokenizer) or "Answer: "
            return anchor, "chat"
        return "", "none"

    def _truncate_anchor(wrapper, text: str) -> str:
        max_tok = int(getattr(args, "max_anchor_tokens", 0) or 0)
        if max_tok <= 0 or not text:
            return text
        try:
            ids = wrapper._encode_anchor_text(text)
        except Exception:
            return text
        if len(ids) <= max_tok:
            return text
        kept = ids[:max_tok]
        truncated = wrapper.tokenizer.decode(kept, skip_special_tokens=False)
        if text.endswith(" ") and not truncated.endswith(" "):
            truncated += " "
        print(f"[WARN] Anchor trimmed from {len(ids)} to {len(kept)} tokens for {wrapper.cfg.model_id}")
        return truncated

    anchor_texts: Dict[str, str] = {}
    anchor_modes: Dict[str, str] = {}
    anchor_token_lists: Dict[str, List[int]] = {}
    bos_flags: Dict[str, Optional[bool]] = {}

    for name, wrapper in (('llama', llama), ('qwen', qwen)):
        if wrapper is None:
            continue
        text, mode = _anchor_text_for(wrapper, args.warm_anchor_text)
        if mode == "text":
            text = _truncate_anchor(wrapper, text)
        elif mode == "chat":
            text = strip_anchor_literal
        anchor_texts[name] = text
        anchor_modes[name] = mode

        try:
            skip_chat = bool(args.use_chat_template)
            _assert_t0_alignment(wrapper.tokenizer, text or "Answer: ", skip_if_chat=skip_chat)
        except Exception as exc:
            print(f"[WARN] A0 sanity check skipped/failed for {name}: {exc}")

        anchor_tokens_source = text if mode == "text" else strip_anchor_literal
        anchor_ids = anchor_token_ids(wrapper, anchor_tokens_source) if anchor_tokens_source else []
        anchor_token_lists[name] = anchor_ids
        if anchor_ids:
            print(f"[INFO] {name} anchor tokens: {len(anchor_ids)}")
        flag = bos_policy(args.train_append_bos_after_prefix, anchor_ids)
        if mode == "chat":
            flag = False
        bos_flags[name] = flag

    if 'llama' in anchor_texts and 'qwen' in anchor_texts:
        if anchor_texts['llama'] != anchor_texts['qwen']:
            print("[WARN] Anchor strings differ between models; using Llama variant for shared config.")

    if args.grad_ckpt:
        if llama is not None:
            llama.enable_gradient_checkpointing()
            try:
                llama.model.config.use_cache = False
            except Exception:
                pass
        if qwen is not None:
            qwen.enable_gradient_checkpointing()
            try:
                qwen.model.config.use_cache = False
            except Exception:
                pass

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

    def _neutral_chat_wrap(s: str) -> str:
        system = "You are a concise QA assistant. Use the context to answer with a short phrase only."
        return f"System: {system}\nUser: {s}\nAssistant:"

    def _maybe_chat_texts(batch_texts: List[str]) -> List[str]:
        if not args.encoder_use_chat_template:
            return batch_texts
        return [_neutral_chat_wrap(t) for t in batch_texts]

    if args.encoder_type == "byte":
        encoder = InterlinguaEncoder(
            d_z=args.d_z,
            latent_shared_len=latent_shared_len,
            latent_private_len=latent_private_len,
            model_keys=tuple(model_keys),
        ).to(device)
        byte_tok = ByteTokenizer(max_bytes=args.max_bytes)

        def encode_fn(batch_texts):
            texts = _maybe_chat_texts(batch_texts)
            z_bytes = collate_bytes(texts, byte_tok, device)
            return encoder(z_bytes, return_components=True)

    elif args.encoder_type == "stq":
        encoder = STQueryEncoder(
            d_z=args.d_z,
            latent_len=total_latent_len,
            hf_encoder_id=(args.hf_encoder_id or "sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=args.max_enc_tokens,
        ).to(device)

        def encode_fn(batch_texts):
            texts = _maybe_chat_texts(batch_texts)
            raw = encoder(texts)
            return _structure_latents(raw)

    else:
        encoder = SimpleEncoder(
            d_z=args.d_z,
            latent_len=total_latent_len,
            backbone=(args.encoder_backbone or "sentence-transformers/all-MiniLM-L6-v2"),
        ).to(device)

        def encode_fn(batch_texts):
            texts = _maybe_chat_texts(batch_texts)
            raw = encoder(texts)
            return _structure_latents(raw)

    latent_refiner = None
    if int(args.latent_refiner_layers) > 0:
        latent_refiner = LatentRefiner(
            d_z=args.d_z,
            num_layers=int(args.latent_refiner_layers),
            num_heads=int(max(args.latent_refiner_heads, 1)),
        ).to(device)

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad_(False)
        print("[INFO] Encoder frozen (--freeze_encoder)")

    # ===== Adapters =====
    adapters: Dict[str, Adapter] = {}
    deep_prefix_generators: Dict[str, DeepPrefixGenerator] = {}
    gist_heads: Dict[str, GistReconstructionHead] = {}
    adp_llama: Optional[Adapter] = None
    adp_qwen: Optional[Adapter] = None

    if llama is not None:
        adp_llama = Adapter(
            d_z=args.d_z,
            d_model=llama.d_model,
            latent_length=latent_shared_len + latent_private_len,
            enable_metadata=bool(args.adapter_metadata),
            length_norm=float(args.max_answer_tokens),
            hidden_mult=args.adapter_hidden_mult,
            colorize=bool(args.adapter_colorize),
            dropout=float(args.adapter_dropout),
        ).to(_primary_device(llama))
        adapters["llama"] = adp_llama

    if qwen is not None:
        adp_qwen = Adapter(
            d_z=args.d_z,
            d_model=qwen.d_model,
            latent_length=latent_shared_len + latent_private_len,
            enable_metadata=bool(args.adapter_metadata),
            length_norm=float(args.max_answer_tokens),
            hidden_mult=args.adapter_hidden_mult,
            colorize=bool(args.adapter_colorize),
            dropout=float(args.adapter_dropout),
        ).to(_primary_device(qwen))
        adapters["qwen"] = adp_qwen

    if args.adapter_colorize:
        for name, adapter in adapters.items():
            wrapper = llama if name == "llama" else qwen
            if wrapper is None:
                continue
            try:
                adapter.install_color_from_wrapper(wrapper)
                print(f"Initialized adapter colorizer for {name} from LM embedding stats.")
            except Exception as exc:
                print(f"[WARN] Adapter colorizer initialization skipped for {name}: {exc}")

    if args.adapter_freeze_scale:
        for adapter in adapters.values():
            if adapter.scale is None:
                continue
            adapter.scale.requires_grad_(False)
            with torch.no_grad():
                adapter.scale.fill_(1.0)

    if args.use_deep_prefix:
        cfg_prefix_len = args.deep_prefix_len if args.deep_prefix_len is not None else (latent_shared_len + latent_private_len)
        if cfg_prefix_len <= 0:
            raise ValueError("--use_deep_prefix requires deep_prefix_len > 0 or non-zero latent length")
        prefix_len = int(cfg_prefix_len)
        dropout = float(max(args.deep_prefix_dropout, 0.0))
        for name, wrapper in (('llama', llama), ('qwen', qwen)):
            if wrapper is None:
                continue
            num_layers = getattr(wrapper.model.config, "num_hidden_layers", None)
            num_attention_heads = getattr(wrapper.model.config, "num_attention_heads", getattr(wrapper.model.config, "n_head", None))
            num_kv_heads = getattr(wrapper.model.config, "num_key_value_heads", None)
            if num_layers is None or num_attention_heads is None:
                print(f"[WARN] Skipping deep prefix for {name}: model config missing layer/head counts")
                continue
            if num_kv_heads is None:
                num_kv_heads = num_attention_heads
            head_dim = wrapper.d_model // int(num_attention_heads)
            generator = DeepPrefixGenerator(
                d_z=wrapper.d_model,
                prefix_len=prefix_len,
                num_layers=int(num_layers),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                dropout=dropout,
            ).to(_primary_device(wrapper))
            generator.train()
            deep_prefix_generators[name] = generator

    if args.use_gist_head and args.gist_weight > 0.0:
        gist_len = max(1, int(args.gist_target_len))
        for name, wrapper in (('llama', llama), ('qwen', qwen)):
            if wrapper is None:
                continue
            head = GistReconstructionHead(
                d_latent=args.d_z,
                d_model=wrapper.d_model,
                target_len=gist_len,
                hidden=int(max(args.gist_hidden, args.d_z)),
                num_layers=int(max(args.gist_layers, 0)),
                dropout=float(max(args.gist_dropout, 0.0)),
            ).to(_primary_device(wrapper))
            gist_heads[name] = head

    # ===== Optimizer =====
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    llama_params = [p for p in adp_llama.parameters() if p.requires_grad] if adp_llama is not None else []
    qwen_params = [p for p in adp_qwen.parameters() if p.requires_grad] if adp_qwen is not None else []
    refiner_params = [p for p in latent_refiner.parameters() if p.requires_grad] if latent_refiner is not None else []
    deep_prefix_params: List[torch.nn.Parameter] = []
    for generator in deep_prefix_generators.values():
        deep_prefix_params.extend([p for p in generator.parameters() if p.requires_grad])
    gist_params: List[torch.nn.Parameter] = []
    for head in gist_heads.values():
        gist_params.extend([p for p in head.parameters() if p.requires_grad])

    optim_groups = []
    if enc_params:
        optim_groups.append({"params": enc_params, "lr": args.lr})
    if llama_params:
        optim_groups.append({"params": llama_params, "lr": args.lr})
    if qwen_params:
        optim_groups.append({"params": qwen_params, "lr": args.lr})
    if extra_llama_params:
        optim_groups.append({"params": extra_llama_params, "lr": args.lr})
    if extra_qwen_params:
        optim_groups.append({"params": extra_qwen_params, "lr": args.lr})
    if refiner_params:
        optim_groups.append({"params": refiner_params, "lr": args.lr})
    if deep_prefix_params:
        optim_groups.append({"params": deep_prefix_params, "lr": args.lr})
    if gist_params:
        optim_groups.append({"params": gist_params, "lr": args.lr})

    optimizer = optim.AdamW(optim_groups, lr=args.lr, foreach=False)

    # ===== Tokenize answers (teacher forcing) =====
    token_ids_map: Dict[str, torch.Tensor] = {}
    answer_lengths_map: Dict[str, torch.Tensor] = {}
    first_token_ids_map: Dict[str, torch.Tensor] = {}

    for name, wrapper in (('llama', llama), ('qwen', qwen)):
        if wrapper is None:
            continue
        with _temp_padding_side(wrapper.tokenizer, "right"):
            tok = wrapper.tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_answer_tokens,
                add_special_tokens=True,
            )
        ids = tok["input_ids"].to(device)
        token_ids_map[name] = ids
        answer_lengths_map[name] = _answer_lengths(ids, wrapper.tokenizer)
        first_token_ids_map[name] = first_non_bos(wrapper.tokenizer, ids)

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    # ===== Resume (optional) =====
    start_epoch = 0
    global_step = 0
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        capture_env_snapshot(args.save_dir, extras={"phase":"train"})
    except Exception:
        pass

    ckpt_path = None
    if args.resume_from:
        if os.path.isdir(args.resume_from):
            ckpt_path = find_latest_checkpoint(args.resume_from)
        else:
            ckpt_path = args.resume_from
    elif args.auto_resume:
        ckpt_path = find_latest_checkpoint(args.save_dir)

    if ckpt_path:
        print(f"⏪ Resuming from: {ckpt_path}")
        epoch_loaded, global_loaded = load_checkpoint(
            ckpt_path,
            encoder,
            adapters,
            refiner=latent_refiner,
            deep_prefix_generators=deep_prefix_generators,
            gist_heads=gist_heads,
            optimizer=None if args.no_load_optimizer else optimizer,
            strict=True,
            device=device,
        )
        start_epoch = epoch_loaded
        global_step = global_loaded
        if args.reset_epoch:
            start_epoch = 0
            global_step = 0
            print("   -> reset epoch/global_step to zero as requested")
        print(f"   -> start_epoch={start_epoch}, global_step={global_step}")
        if args.adapter_colorize:
            try:
                for name, adapter in adapters.items():
                    wrapper = llama if name == "llama" else qwen
                    if wrapper is None:
                        continue
                    adapter.install_color_from_wrapper(wrapper)
            except Exception as exc:
                print(f"[WARN] Adapter colorizer re-install skipped after resume: {exc}")
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

    stats_trackers: Dict[str, Dict[str, Any]] = {}
    if llama is not None:
        stats_trackers["llama"] = {
            "rms_raw": _RunningMean(),
            "rms_cal": _RunningMean(),
            "embed_rms": llama.input_embedding_rms(),
        }
    if qwen is not None:
        stats_trackers["qwen"] = {
            "rms_raw": _RunningMean(),
            "rms_cal": _RunningMean(),
            "embed_rms": qwen.input_embedding_rms(),
        }

    model_contexts: List[ModelTrainContext] = []

    if llama is not None:
        model_contexts.append(
            ModelTrainContext(
                name="llama",
                wrapper=llama,
                adapter=adp_llama,
                token_ids=token_ids_map["llama"],
                first_token_ids=first_token_ids_map["llama"],
                anchor_ids=anchor_token_lists["llama"],
                bos_flag=bos_flags.get("llama"),
                answer_lengths=answer_lengths_map["llama"],
                anchor_text=anchor_texts.get("llama", ""),
                anchor_mode=anchor_modes.get("llama", "none"),
            )
        )

    if qwen is not None:
        model_contexts.append(
            ModelTrainContext(
                name="qwen",
                wrapper=qwen,
                adapter=adp_qwen,
                token_ids=token_ids_map["qwen"],
                first_token_ids=first_token_ids_map["qwen"],
                anchor_ids=anchor_token_lists["qwen"],
                bos_flag=bos_flags.get("qwen"),
                answer_lengths=answer_lengths_map["qwen"],
                anchor_text=anchor_texts.get("qwen", ""),
                anchor_mode=anchor_modes.get("qwen", "none"),
            )
        )

    if not model_contexts:
        raise RuntimeError("No models selected for training. Use --models to include at least one backend.")

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

    def manifold_stat_loss(prefix: torch.Tensor, model_key: str) -> torch.Tensor:
        if args.manifold_stat_weight <= 0.0:
            return torch.zeros((), device=prefix.device)
        mu, sd = embed_stats[model_key]
        mu = mu.to(prefix.device, dtype=prefix.dtype)
        sd = sd.to(prefix.device, dtype=prefix.dtype)
        cur_mu = prefix.float().mean(dim=[0, 1])
        cur_sd = prefix.float().std(dim=[0, 1]).clamp_min(1e-6)
        return (cur_mu - mu).pow(2).mean() + (cur_sd - sd).pow(2).mean()

    def alignment_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pred.numel() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        diff = (pred - target).pow(2)
        if mask is not None:
            mask_base = mask.to(pred.device, dtype=pred.dtype)
            diff = diff * mask_base.unsqueeze(-1)
            denom = (mask_base.sum().clamp_min(1.0)) * pred.size(-1)
        else:
            denom = torch.tensor(float(diff.numel()), device=pred.device, dtype=pred.dtype)
        return diff.sum() / denom

    def _parse_layers_arg(value: str) -> Tuple[int, ...]:
        try:
            items = [int(v) for v in re.split(r"[\s,]+", value.strip()) if v != ""]
            return tuple(items) if items else (0, 1, 2)
        except Exception:
            return (0, 1, 2)

    state_kd_layers = _parse_layers_arg(args.state_kd_layers)

    params_for_clip = enc_params + llama_params + qwen_params

    def _grad_norm(params) -> float:
        norms = []
        for p in params:
            grad = getattr(p, "grad", None)
            if grad is None:
                continue
            g = grad.detach()
            if not torch.isfinite(g).all():
                return float("nan")
            norms.append(g.float().norm(2).cpu())
        if not norms:
            return 0.0
        stacked = torch.stack(norms)
        return float(torch.norm(stacked, 2).item())

    optimizer.zero_grad(set_to_none=True)

    total_batches = steps_per_epoch * args.epochs
    warmup_steps_from_epochs = int(round(max(float(args.warmup_text_latent_epochs), 0.0) * steps_per_epoch))
    warmup_total_steps = max(int(args.warmup_text_latent_steps), warmup_steps_from_epochs)
    warmup_total_steps = max(0, min(warmup_total_steps, total_batches))
    if warmup_total_steps > 0:
        print(f"[warmup] alternating text/latent for first {warmup_total_steps} steps")

    first_ce_schedule = str(getattr(args, "first_token_ce_schedule", "none")).lower()
    peak_override = args.first_token_ce_peak
    autoscale_first = str(getattr(args, "first_token_autoscale", "yes")).lower() != "no"

    def _first_token_weight_for_step(step_idx: int) -> float:
        base = max(float(args.first_token_ce_weight), 0.0)
        if first_ce_schedule == "none" or total_batches <= 0:
            return base
        total = max(int(total_batches), 1)
        peak = peak_override if (peak_override is not None and peak_override > 0.0) else max(8.0, 2.0 * max(base, 1e-6))
        warm_frac = min(max(float(args.first_token_ce_warmup_frac), 0.0), 1.0)
        warm_steps = int(round(total * warm_frac))
        warm_steps = min(warm_steps, total)
        if step_idx < warm_steps:
            return peak
        if step_idx >= total or total == warm_steps:
            return base
        t = (step_idx - warm_steps) / max(1, total - warm_steps)
        t = min(max(t, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return base + (peak - base) * cosine
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        g = torch.Generator(device="cpu")
        g.manual_seed(int(args.seed) + int(epoch))
        perm = torch.randperm(N, generator=g)

        for step in range(steps_per_epoch):
            t0 = time.time()
            idx = perm[step*args.batch_size : (step+1)*args.batch_size]
            batch_texts = [texts[i] for i in idx.tolist()]
            if args.use_chat_template:
                batch_user_texts = [split_user_and_anchor(raw, strip_anchor_literal)[0] for raw in batch_texts]
            else:
                batch_user_texts = batch_texts

            global_batch_idx = epoch * steps_per_epoch + step
            if total_batches > 1:
                progress = min(max(global_batch_idx / (total_batches - 1), 0.0), 1.0)
            else:
                progress = 1.0
            progress_pow = progress ** latent_keep_power
            keep_prob = latent_keep_start + (latent_keep_end - latent_keep_start) * progress_pow
            keep_prob = float(min(max(keep_prob, 0.0), 1.0))
            current_K = int(round(adaptive_k_start + (adaptive_k_end - adaptive_k_start) * progress_pow))
            current_K = max(1, min(current_K, args.max_answer_tokens))

            current_first_weight = _first_token_weight_for_step(global_step)
            enable_first_token_loss = current_first_weight > 0.0

            scaffolds = {}
            for ctx in model_contexts:
                if args.use_chat_template:
                    pad_token = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                    if pad_token is None:
                        pad_token = int(getattr(ctx.wrapper.tokenizer, "eos_token_id", 0))
                    ids_list = []
                    for user_text in batch_user_texts:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_text},
                        ]
                        rendered = ctx.wrapper.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        assistant_prefill = ctx.anchor_text if ctx.anchor_mode == "text" else strip_anchor_literal
                        if assistant_prefill:
                            rendered = rendered + assistant_prefill
                        toks = ctx.wrapper.tokenizer(
                            rendered,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                            add_special_tokens=False,
                        )
                        ids = toks["input_ids"][0].to(device)
                        ids_list.append(ids)
                    scaffolds[ctx.name] = torch.nn.utils.rnn.pad_sequence(
                        ids_list, batch_first=True, padding_value=int(pad_token)
                    )
                else:
                    anchor_suffix = ctx.anchor_text if ctx.anchor_mode == "text" else strip_anchor_literal
                    texts_with_anchor = [f"{text}{anchor_suffix}" for text in batch_texts]
                    tok = ctx.wrapper.tokenizer(
                        texts_with_anchor,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                        add_special_tokens=False,
                    )
                    scaffolds[ctx.name] = tok["input_ids"].to(device)

            effective_texts = batch_user_texts if args.use_chat_template else batch_texts
            warmup_active = warmup_total_steps > 0 and global_step < warmup_total_steps
            training_mode = "latent"
            if warmup_active:
                training_mode = "text"
            elif args.warmup_tail_prob > 0.0 and random.random() < float(args.warmup_tail_prob):
                training_mode = "text"

            if training_mode == "text":
                if warmup_active and global_step < 10:
                    print(f"[warmup] step={global_step} mode=text (warm-up)")
                elif not warmup_active and global_step < warmup_total_steps + 50:
                    print(f"[warmup] step={global_step} mode=text (tail)")

            per_model_losses: Dict[str, Dict[str, torch.Tensor]] = {}
            total_model_loss = torch.zeros((), device=device)
            penalty = torch.zeros((), device=device)
            rms_pen = torch.zeros((), device=device)

            encoded_latents = encode_fn(effective_texts)
            shared_latents = encoded_latents["shared"]
            private_latents = encoded_latents["private"]
            dropout_keep = keep_prob if training_mode == "latent" else 1.0
            if shared_latents.size(1) > 0 and dropout_keep < 1.0:
                mask = (torch.rand(shared_latents.shape[:2], device=shared_latents.device) < dropout_keep).float()
                need_fix = mask.sum(dim=1) == 0
                if need_fix.any():
                    mask[need_fix, 0] = 1.0
                mask = mask.unsqueeze(-1)
                shared_latents = shared_latents * mask / max(dropout_keep, 1e-3)
            model_latents = {
                name: torch.cat([shared_latents, private_latents[name]], dim=1)
                for name in model_keys
            }
            if latent_refiner is not None:
                for name in model_keys:
                    model_latents[name] = latent_refiner(model_latents[name])

            for ctx in model_contexts:
                target_device = _primary_device(ctx.wrapper)
                targets = ctx.token_ids[idx].to(target_device, non_blocking=True)
                scaffold = scaffolds[ctx.name].to(target_device, non_blocking=True)

                losses_record: Dict[str, torch.Tensor] = {}
                latents_for_adapter = model_latents[ctx.name].to(target_device, non_blocking=True)
                answer_lengths = ctx.answer_lengths[idx].to(target_device, non_blocking=True)

                prefix_raw = ctx.adapter(latents_for_adapter, answer_lengths=answer_lengths)
                prefix = calibrate_to_embed_rms(prefix_raw, ctx.wrapper)
                deep_prefix_cache: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None
                if ctx.name in deep_prefix_generators:
                    deep_prefix_cache = deep_prefix_generators[ctx.name](prefix)
                if args.debug and epoch == start_epoch and step == 0:
                    print(
                        f"[DEBUG:{ctx.name}] prefix_len={prefix.shape[1]} anchor_ids={len(ctx.anchor_ids)} tf_len={targets.size(1)}",
                        flush=True,
                    )
                    print(
                        f"[DEBUG:{ctx.name}] scaffold_len={scaffold.size(1)} anchor_mode={ctx.anchor_mode}",
                        flush=True,
                    )
                loss_tf_latent = ctx.wrapper.forward_with_prefix_loss(
                    prefix,
                    targets,
                    anchor_token_ids=ctx.anchor_ids,
                    deep_prefix_past=deep_prefix_cache,
                )

                first_anchor_text = ctx.anchor_text if ctx.anchor_mode == "text" else strip_anchor_literal
                if enable_first_token_loss:
                    logits_first = ctx.wrapper.first_token_logits_from_prefix(
                        prefix,
                        anchor_token_text=first_anchor_text,
                        append_bos_after_prefix=ctx.bos_flag,
                        deep_prefix_past=deep_prefix_cache,
                    )
                    first_targets = ctx.first_token_ids[idx].to(target_device)
                    loss_first_raw = nn.functional.cross_entropy(logits_first.float(), first_targets)
                    with torch.no_grad():
                        first_pred = logits_first.argmax(dim=-1)
                        first_acc_raw = (first_pred == first_targets).float().mean()
                else:
                    loss_first_raw = torch.zeros((), device=target_device)
                    first_acc_raw = torch.zeros((), device=target_device)

                if args.k_ce_weight and args.k_ce_weight > 0.0:
                    loss_kce_raw = k_token_ce_from_prefix(
                        ctx.wrapper,
                        prefix,
                        targets,
                        K=current_K,
                        anchor_ids=ctx.anchor_ids,
                        append_bos_after_prefix=ctx.bos_flag,
                        deep_prefix_past=deep_prefix_cache,
                    )
                else:
                    loss_kce_raw = torch.zeros((), device=target_device)

                if args.kd_first_k_weight and args.kd_first_k_weight > 0.0:
                    loss_kd_raw = kd_first_k_prefix_vs_text(
                        ctx.wrapper,
                        ctx.wrapper,
                        prefix,
                        scaffold,
                        targets,
                        K=current_K,
                        tau=args.kd_tau,
                        anchor_ids=ctx.anchor_ids,
                        append_bos_after_prefix=ctx.bos_flag,
                        deep_prefix_past=deep_prefix_cache,
                    )
                else:
                    loss_kd_raw = torch.zeros((), device=target_device)

                if args.state_kd_weight and args.state_kd_weight > 0.0:
                    loss_state_raw = kd_hidden_states_first_k(
                        ctx.wrapper,
                        prefix,
                        scaffold,
                        targets,
                        K=current_K,
                        layers=state_kd_layers,
                        append_bos_after_prefix=ctx.bos_flag,
                        anchor_ids=ctx.anchor_ids,
                        deep_prefix_past=deep_prefix_cache,
                    )
                else:
                    loss_state_raw = torch.zeros((), device=target_device)

                gist_loss_raw = torch.zeros((), device=target_device)
                if args.use_gist_head and args.gist_weight > 0.0:
                    head = gist_heads.get(ctx.name)
                    if head is not None:
                        gist_len = head.target_len
                        scaffold_slice = scaffold[:, :gist_len]
                        gist_pred = head(latents_for_adapter)
                        pad_id = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                        valid = torch.ones_like(scaffold_slice, dtype=torch.bool, device=target_device)
                        if pad_id is not None:
                            valid = valid & scaffold_slice.ne(int(pad_id))
                        if args.gist_mask_prob > 0.0:
                            mask_rand = torch.rand_like(scaffold_slice.float(), device=target_device)
                            valid = valid & (mask_rand >= float(args.gist_mask_prob))
                        gist_targets = ctx.wrapper.input_embed(scaffold_slice)
                        mask = valid.unsqueeze(-1).float()
                        denom = mask.sum().clamp_min(1.0)
                        diff = (gist_pred - gist_targets) * mask
                        gist_loss_raw = diff.pow(2).sum() / denom

                grad_diag_values: Dict[str, torch.Tensor] = {}
                if (
                    grad_diag_interval > 0
                    and grad_diag_component_set
                    and prefix_raw.requires_grad
                    and (global_step % grad_diag_interval == 0)
                ):
                    diag_sources: Dict[str, torch.Tensor] = {
                        "tf": loss_tf_latent,
                        "first": loss_first_raw,
                        "kce": loss_kce_raw,
                        "kd": loss_kd_raw,
                        "state": loss_state_raw,
                        "align": align_loss,
                        "latent_align": latent_align_loss,
                        "latent_prefix_align": latent_prefix_align_loss,
                        "gist": gist_loss_raw,
                    }
                    for name in grad_diag_components:
                        term = diag_sources.get(name)
                        if term is None:
                            continue
                        try:
                            grads = torch.autograd.grad(
                                term,
                                prefix_raw,
                                retain_graph=True,
                                allow_unused=True,
                            )
                        except RuntimeError as exc:
                            if args.debug:
                                print(f"[WARN] grad_diag({ctx.name}:{name}) failed: {exc}")
                            continue
                        grad_tensor = grads[0] if grads else None
                        if grad_tensor is None:
                            value = torch.zeros((), device=target_device)
                        else:
                            value = grad_tensor.float().pow(2).sum().sqrt()
                        grad_diag_values[f"grad_{name}"] = value.detach()

                latent_scale = 1.0
                if training_mode == "text":
                    start_scale = float(max(args.warmup_text_latent_weight, 0.0))
                    end_scale = float(max(args.warmup_text_latent_weight_end, 0.0))
                    if warmup_active and warmup_total_steps > 0:
                        frac = min(1.0, max(0.0, global_step / float(max(1, warmup_total_steps))))
                        latent_scale = start_scale + (end_scale - start_scale) * frac
                    else:
                        latent_scale = end_scale

                effective_first_weight = float(current_first_weight)
                loss_tf = latent_scale * loss_tf_latent
                loss_first = latent_scale * loss_first_raw
                loss_kce = latent_scale * loss_kce_raw
                loss_kd = latent_scale * loss_kd_raw
                loss_state = latent_scale * loss_state_raw
                gist_weight = float(max(args.gist_weight, 0.0)) if args.use_gist_head else 0.0
                gist_loss = gist_weight * gist_loss_raw

                if (
                    enable_first_token_loss
                    and autoscale_first
                    and training_mode == "latent"
                    and latent_scale > 0.0
                ):
                    denom_val = float(loss_tf_latent.detach().abs().clamp_min(1e-4).item())
                    numer_val = float(loss_first_raw.detach().abs().item())
                    if denom_val > 0.0:
                        ratio = numer_val / denom_val
                        if ratio > 1.0:
                            effective_first_weight *= max(min(ratio, 8.0), 1.0)

                manifold_loss = manifold_stat_loss(prefix, ctx.name)

                text_teacher_loss = torch.zeros((), device=target_device)
                align_loss = torch.zeros((), device=target_device)
                latent_align_loss = torch.zeros((), device=target_device)
                latent_prefix_align_loss = torch.zeros((), device=target_device)
                if training_mode == "text" and args.warmup_align_tokens > 0 and args.warmup_align_weight > 0.0:
                    max_align = min(int(args.warmup_align_tokens), prefix.shape[1])
                    pad_id = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                    bos_id = getattr(ctx.wrapper.tokenizer, "bos_token_id", None)
                    if max_align > 0 and prefix.shape[1] > 0:
                        teacher_ids = ctx.token_ids[idx].to(target_device, non_blocking=True)
                        start = 0
                        if bos_id is not None and teacher_ids.size(1) > 0 and (teacher_ids[:, 0] == int(bos_id)).all():
                            start = 1
                        stop = min(start + max_align, teacher_ids.size(1))
                        token_slice = teacher_ids[:, start:stop]
                        if token_slice.numel() > 0:
                            mask = None
                            if pad_id is not None:
                                mask = token_slice.ne(int(pad_id))
                            teacher_embeds = ctx.wrapper.input_embed(token_slice)
                            prefix_slice = prefix[:, : token_slice.size(1), :]
                            align_loss = alignment_mse(prefix_slice, teacher_embeds, mask)
                            align_loss = align_loss * float(max(args.warmup_align_weight, 0.0))
                if training_mode == "latent" and args.latent_align_weight > 0.0 and prefix.shape[1] > 0:
                    teacher_first_ids = ctx.first_token_ids[idx].to(target_device, non_blocking=True)
                    teacher_first_ids = teacher_first_ids.view(-1, 1)
                    teacher_emb = ctx.wrapper.input_embed(teacher_first_ids).squeeze(1).to(prefix.dtype)
                    latent_embed = prefix[:, 0, :]
                    if args.latent_align_metric in ("cosine", "both"):
                        cos = 1.0 - nn.functional.cosine_similarity(latent_embed, teacher_emb, dim=-1)
                        latent_align_loss = latent_align_loss + cos.mean()
                    if args.latent_align_metric in ("mse", "both"):
                        latent_align_loss = latent_align_loss + nn.functional.mse_loss(latent_embed, teacher_emb)
                    latent_align_loss = latent_align_loss * float(max(args.latent_align_weight, 0.0))
                if training_mode == "latent" and args.latent_prefix_align_weight > 0.0 and prefix.shape[1] > 0:
                    prefix_len = prefix.shape[1]
                    teacher_prefix_ids = ctx.token_ids[idx].to(target_device, non_blocking=True)
                    teacher_prefix_emb = ctx.wrapper.input_embed(teacher_prefix_ids).to(prefix.dtype)
                    teacher_prefix_emb = teacher_prefix_emb[:, :prefix_len]
                    overlap = min(prefix_len, teacher_prefix_emb.size(1))
                    if overlap > 0:
                        latent_prefix_align_loss = torch.zeros((), device=target_device)
                        if args.latent_align_metric in ("cosine", "both"):
                            cos = 1.0 - nn.functional.cosine_similarity(
                                prefix[:, :overlap, :], teacher_prefix_emb[:, :overlap, :], dim=-1
                            )
                            latent_prefix_align_loss = latent_prefix_align_loss + cos.mean()
                        if args.latent_align_metric in ("mse", "both"):
                            latent_prefix_align_loss = latent_prefix_align_loss + nn.functional.mse_loss(
                                prefix[:, :overlap, :], teacher_prefix_emb[:, :overlap, :]
                            )
                        latent_prefix_align_loss = latent_prefix_align_loss * float(max(args.latent_prefix_align_weight, 0.0))
                if training_mode == "text":
                    try:
                        text_teacher_loss, _ = ctx.wrapper.loss_with_text_prompt(scaffold, targets)
                    except Exception:
                        text_teacher_loss = torch.zeros((), device=target_device)

                model_loss = (
                    loss_tf
                    + effective_first_weight * loss_first
                    + args.k_ce_weight * loss_kce
                    + args.kd_first_k_weight * loss_kd
                    + args.state_kd_weight * loss_state
                    + args.manifold_stat_weight * manifold_loss
                    + align_loss
                    + latent_align_loss
                    + latent_prefix_align_loss
                    + float(max(args.warmup_text_teacher_weight, 0.0)) * text_teacher_loss
                    + gist_loss
                ).to(device)
                total_model_loss = total_model_loss + model_loss

                penalty = penalty + scale_penalty(ctx.adapter).to(device)
                rms_pen = rms_pen + rms_raw_penalty(prefix_raw, ctx.wrapper).to(device)

                rms_raw_val = tensor_rms(prefix_raw)
                rms_cal_val = tensor_rms(prefix)
                stats_trackers[ctx.name]["rms_raw"].update(rms_raw_val)
                stats_trackers[ctx.name]["rms_cal"].update(rms_cal_val)

                losses_record.update({
                    "tf": loss_tf,
                    "first": loss_first,
                    "first_weight": torch.tensor(float(effective_first_weight), device=target_device),
                    "first_acc": first_acc_raw,
                    "kce": loss_kce,
                    "kd": loss_kd,
                    "state": loss_state,
                    "manifold": manifold_loss,
                    "rms_raw": rms_raw_val,
                    "rms_cal": rms_cal_val,
                    "align": align_loss,
                    "latent_align": latent_align_loss,
                    "latent_prefix_align": latent_prefix_align_loss,
                    "text_tf": text_teacher_loss,
                    "latent_scale": torch.tensor(float(latent_scale), device=target_device),
                    "gist": gist_loss_raw,
                })
                for key, value in grad_diag_values.items():
                    losses_record[key] = value
                per_model_losses[ctx.name] = losses_record
                per_model_losses[ctx.name]["mode"] = "latent"
                if training_mode == "text":
                    per_model_losses[ctx.name]["mode"] = "text"

            if training_mode == "text":
                parts_text = [
                    f"  step  {step+1}/{steps_per_epoch}",
                    "(warm-up text)" if warmup_active else "(tail text)",
                    f"align={_to_float(align_loss):.4f}",
                    f"text_tf={_to_float(text_teacher_loss):.4f}",
                    f"latent_scale={latent_scale:.2f}",
                ]
                print(" | ".join(parts_text))

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

            grad_norm_val = _grad_norm(params_for_clip)
            if not math.isfinite(grad_norm_val):
                print("⚠️  Non-finite gradient detected; skipping optimizer step for this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)
            if should_step:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))
                _align_optimizer_state_to_param_devices(optimizer)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_norm = grad_norm_val

            # Grad norm (monitor) – encoder only as a proxy
            global_step += 1
            dt = time.time() - t0
            ema_step_time = dt if (locals().get("ema_step_time", None) is None) else (0.9 * locals()["ema_step_time"] + 0.1 * dt)

            if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                parts = [
                    f"  step  {step+1}/{steps_per_epoch}",
                    f"grad_norm={total_norm:.2f}",
                    f"sec/step~{dt:.2f}",
                    f"keep={keep_prob:.2f}",
                    f"K={current_K}",
                ]
                if first_ce_schedule != "none":
                    parts.append(f"first_w={current_first_weight:.2f}")
                for ctx in model_contexts:
                    metrics = per_model_losses[ctx.name]
                    mode_tag = metrics.get("mode", "latent")
                    mode_label = "T" if mode_tag == "text" else "L"
                    msg_ctx = (
                        f"{ctx.name}({mode_label}): tf={_to_float(metrics['tf']):.4f}"
                        f" first={_to_float(metrics['first']):.4f}"
                        f" kCE={_to_float(metrics['kce']):.4f}"
                        f" KD={_to_float(metrics['kd']):.4f}"
                    )
                    msg_ctx += f" acc={_to_float(metrics.get('first_acc', 0.0)):.3f}"
                    if args.state_kd_weight > 0.0:
                        msg_ctx += f" state={_to_float(metrics['state']):.4f}"
                    if args.manifold_stat_weight > 0.0:
                        msg_ctx += f" man={_to_float(metrics['manifold']):.4f}"
                    if args.warmup_align_weight > 0.0:
                        msg_ctx += f" align={_to_float(metrics['align']):.4f}"
                    if args.latent_align_weight > 0.0:
                        msg_ctx += f" latA={_to_float(metrics['latent_align']):.4f}"
                    if args.latent_prefix_align_weight > 0.0:
                        msg_ctx += f" latP={_to_float(metrics['latent_prefix_align']):.4f}"
                    if args.use_gist_head and args.gist_weight > 0.0:
                        msg_ctx += f" gist={_to_float(metrics['gist']):.4f}"
                    if grad_diag_components:
                        for diag_name in grad_diag_components:
                            key = f"grad_{diag_name}"
                            if key in metrics:
                                msg_ctx += f" {key}={_to_float(metrics[key]):.3e}"
                    parts.append(msg_ctx)
                    if args.scale_l2 > 0.0:
                        parts.append(f"scale_pen({ctx.name})={scale_penalty(ctx.adapter).item():.4e}")
                parts.append(f"K={current_K} tau={args.kd_tau:.2f}")
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

                if diagnostic_log_path and ((step + 1) % 10 == 0 or (step + 1) == steps_per_epoch):
                    diag_entry: Dict[str, Any] = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "epoch_step": step + 1,
                        "mode": training_mode,
                        "keep_prob": keep_prob,
                        "K": current_K,
                        "first_weight": float(current_first_weight),
                        "grad_norm": float(total_norm),
                        "sec_per_step": float(dt),
                        "latent_scale": float(latent_scale),
                        "models": {},
                    }
                    for ctx in model_contexts:
                        metrics = per_model_losses[ctx.name]
                        diag_entry["models"][ctx.name] = {
                            "mode": metrics.get("mode", "latent"),
                            "tf": _to_float(metrics.get("tf", 0.0)),
                            "first": _to_float(metrics.get("first", 0.0)),
                            "first_acc": _to_float(metrics.get("first_acc", 0.0)),
                            "kce": _to_float(metrics.get("kce", 0.0)),
                            "kd": _to_float(metrics.get("kd", 0.0)),
                            "state": _to_float(metrics.get("state", 0.0)),
                            "align": _to_float(metrics.get("align", 0.0)),
                            "latent_align": _to_float(metrics.get("latent_align", 0.0)),
                            "latent_prefix_align": _to_float(metrics.get("latent_prefix_align", 0.0)),
                            "gist": _to_float(metrics.get("gist", 0.0)),
                        }
                        for diag_name in grad_diag_components:
                            key = f"grad_{diag_name}"
                            if key in metrics:
                                diag_entry["models"][ctx.name][key] = _to_float(metrics[key])
                    try:
                        with open(diagnostic_log_path, "a") as diag_f:
                            json.dump(diag_entry, diag_f)
                            diag_f.write("\n")
                    except Exception as exc:
                        if args.debug:
                            print(f"[WARN] Failed to append diagnostic log: {exc}")

            # ---- Periodic checkpoint: save + prune
            if args.save_every and (global_step % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                cfg = {
                    "d_z": args.d_z,
                    "latent_len": total_latent_len,
                    "latent_shared_len": latent_shared_len,
                    "latent_private_len": latent_private_len,
                    "byte_max": args.max_bytes,
                    "llama_id": args.llama_id,
                    "qwen_id": args.qwen_id,
                    "encoder_type": args.encoder_type,
                    "encoder_use_chat_template": bool(args.encoder_use_chat_template),
                    "hf_encoder_id": (args.hf_encoder_id if hasattr(args, "hf_encoder_id") else ""),
                    "max_enc_tokens": (args.max_enc_tokens if hasattr(args, "max_enc_tokens") else 1024),
                    "encoder_backbone": (args.encoder_backbone or ""),
                    "freeze_encoder": bool(args.freeze_encoder),
                    "use_chat_template": bool(args.use_chat_template),
                    "warm_anchor_mode": args.warm_anchor_mode,
                    "strip_anchor_text": strip_anchor_literal,
                    "max_anchor_tokens": args.max_anchor_tokens,
                    "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
                    "first_token_ce_weight": args.first_token_ce_weight,
                    "first_token_ce_schedule": args.first_token_ce_schedule,
                    "first_token_ce_peak": args.first_token_ce_peak,
                    "first_token_ce_warmup_frac": args.first_token_ce_warmup_frac,
                    "adapter_hidden_mult": args.adapter_hidden_mult,
                    "adapter_dropout": args.adapter_dropout,
                    "adapter_colorize": bool(args.adapter_colorize),
                    "adapter_enable_metadata": bool(args.adapter_metadata),
                    "llama_device_map": args.llama_device_map,
                    "qwen_device_map": args.qwen_device_map,
                    "llama_devices": args.llama_devices,
                    "qwen_devices": args.qwen_devices,
                    "gpu_mem_gib": args.gpu_mem_gib,
                    "use_lora": bool(args.use_lora),
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "use_prefix": bool(args.use_prefix),
                    "prefix_tokens": args.prefix_tokens,
                    "prefix_projection": bool(args.prefix_projection),
                    "peft_prefix_all_layers": str(getattr(args, "peft_prefix_all_layers", "yes")),
                    "manifold_stat_weight": args.manifold_stat_weight,
                    "state_kd_weight": args.state_kd_weight,
                    "state_kd_layers": args.state_kd_layers,
                    "K": args.K,
                    "k_ce_weight": args.k_ce_weight,
                    "kd_first_k_weight": args.kd_first_k_weight,
                    "kd_tau": args.kd_tau,
                    "max_answer_tokens": args.max_answer_tokens,
                    "grad_accum_steps": grad_accum_steps,
                    "seed": args.seed,
                    "data_seed": args.data_seed,
                    "models": model_keys,
                    "warmup_text_latent_steps": warmup_total_steps,
                    "warmup_text_latent_weight": args.warmup_text_latent_weight,
                    "warmup_text_latent_weight_end": args.warmup_text_latent_weight_end,
                    "warmup_text_teacher_weight": args.warmup_text_teacher_weight,
                    "warmup_tail_prob": args.warmup_tail_prob,
                    "warmup_align_tokens": args.warmup_align_tokens,
                    "warmup_align_weight": args.warmup_align_weight,
                    "grad_diag_interval": grad_diag_interval,
                    "grad_diag_components": args.grad_diag_components,
                    "gist_head": {
                        "enabled": bool(args.use_gist_head),
                        "target_len": int(args.gist_target_len),
                        "hidden": int(args.gist_hidden),
                        "layers": int(args.gist_layers),
                        "dropout": float(args.gist_dropout),
                        "weight": float(args.gist_weight),
                        "mask_prob": float(args.gist_mask_prob),
                    },
                }
                dp_len_cfg = int(
                    args.deep_prefix_len
                    if (args.use_deep_prefix and args.deep_prefix_len is not None)
                    else (latent_shared_len + latent_private_len)
                ) if args.use_deep_prefix else 0
                cfg["deep_prefix"] = {
                    "enabled": bool(args.use_deep_prefix),
                    "len": dp_len_cfg,
                    "dropout": float(args.deep_prefix_dropout),
                }
                cfg["warm_anchor_text"] = anchor_texts.get("llama", "")
                cfg["warm_anchor_texts"] = anchor_texts
                cfg["warm_anchor_modes"] = anchor_modes

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
                        name: float(adapter.scale.detach().cpu().item())
                        for name, adapter in adapters.items()
                        if getattr(adapter, "scale", None) is not None
                    },
                    "encoder": encoder.state_dict(),
                }
                for name, adapter in adapters.items():
                    state_blob[f"adp_{name}"] = adapter.state_dict()
                for name, gen in deep_prefix_generators.items():
                    state_blob[f"deep_prefix_{name}"] = gen.state_dict()
                if latent_refiner is not None:
                    state_blob["refiner"] = latent_refiner.state_dict()
                artifacts = {
                    "encoder.pt": encoder.state_dict(),
                    "state.pt": state_blob,
                    "config.json": cfg,
                }
                for name, adapter in adapters.items():
                    artifacts[f"adapter_{name}.pt"] = adapter.state_dict()
                for name, gen in deep_prefix_generators.items():
                    artifacts[f"deep_prefix_{name}.pt"] = gen.state_dict()
                if latent_refiner is not None:
                    artifacts["refiner.pt"] = latent_refiner.state_dict()
                save_latest_checkpoint(args.save_dir, artifacts, pre_prune=True, post_prune=True, verbose=True)
                print(f"  ✅ Saved (and pruned to) latest at step {global_step}")

    # ===== Final save =====
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {
        "d_z": args.d_z,
        "latent_len": total_latent_len,
        "latent_shared_len": latent_shared_len,
        "latent_private_len": latent_private_len,
        "byte_max": args.max_bytes,
        "llama_id": args.llama_id,
        "qwen_id": args.qwen_id,
        "encoder_type": args.encoder_type,
        "encoder_use_chat_template": bool(args.encoder_use_chat_template),
        "hf_encoder_id": (args.hf_encoder_id if hasattr(args, "hf_encoder_id") else ""),
        "max_enc_tokens": (args.max_enc_tokens if hasattr(args, "max_enc_tokens") else 1024),
        "encoder_backbone": (args.encoder_backbone or ""),
        "freeze_encoder": bool(args.freeze_encoder),
        "use_chat_template": bool(args.use_chat_template),
        "warm_anchor_text": anchor_texts.get("llama", ""),
        "warm_anchor_texts": anchor_texts,
        "warm_anchor_modes": anchor_modes,
        "strip_anchor_text": strip_anchor_literal,
        "max_anchor_tokens": args.max_anchor_tokens,
        "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
        "first_token_ce_weight": args.first_token_ce_weight,
        "first_token_ce_schedule": args.first_token_ce_schedule,
        "first_token_ce_peak": args.first_token_ce_peak,
        "first_token_ce_warmup_frac": args.first_token_ce_warmup_frac,
        "adapter_hidden_mult": args.adapter_hidden_mult,
        "adapter_dropout": args.adapter_dropout,
        "adapter_colorize": bool(args.adapter_colorize),
        "adapter_enable_metadata": bool(args.adapter_metadata),
        "llama_device_map": args.llama_device_map,
        "qwen_device_map": args.qwen_device_map,
        "llama_devices": args.llama_devices,
        "qwen_devices": args.qwen_devices,
        "gpu_mem_gib": args.gpu_mem_gib,
        "use_lora": bool(args.use_lora),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "use_prefix": bool(args.use_prefix),
        "prefix_tokens": args.prefix_tokens,
        "prefix_projection": bool(args.prefix_projection),
        "peft_prefix_all_layers": str(getattr(args, "peft_prefix_all_layers", "yes")),
        "manifold_stat_weight": args.manifold_stat_weight,
        "state_kd_weight": args.state_kd_weight,
        "state_kd_layers": args.state_kd_layers,
        "K": args.K,
        "k_ce_weight": args.k_ce_weight,
        "kd_first_k_weight": args.kd_first_k_weight,
        "kd_tau": args.kd_tau,
        "max_answer_tokens": args.max_answer_tokens,
        "grad_accum_steps": grad_accum_steps,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "models": model_keys,
        "warmup_text_latent_steps": warmup_total_steps,
        "warmup_text_latent_weight": args.warmup_text_latent_weight,
        "warmup_text_latent_weight_end": args.warmup_text_latent_weight_end,
        "warmup_text_teacher_weight": args.warmup_text_teacher_weight,
        "warmup_tail_prob": args.warmup_tail_prob,
        "warmup_align_tokens": args.warmup_align_tokens,
        "warmup_align_weight": args.warmup_align_weight,
        "grad_diag_interval": grad_diag_interval,
        "grad_diag_components": args.grad_diag_components,
        "gist_head": {
            "enabled": bool(args.use_gist_head),
            "target_len": int(args.gist_target_len),
            "hidden": int(args.gist_hidden),
            "layers": int(args.gist_layers),
            "dropout": float(args.gist_dropout),
            "weight": float(args.gist_weight),
            "mask_prob": float(args.gist_mask_prob),
        },
    }
    dp_len_cfg = int(
        args.deep_prefix_len
        if (args.use_deep_prefix and args.deep_prefix_len is not None)
        else (latent_shared_len + latent_private_len)
    ) if args.use_deep_prefix else 0
    cfg["deep_prefix"] = {
        "enabled": bool(args.use_deep_prefix),
        "len": dp_len_cfg,
        "dropout": float(args.deep_prefix_dropout),
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
            name: float(adapter.scale.detach().cpu().item())
            for name, adapter in adapters.items()
            if getattr(adapter, "scale", None) is not None
        },
        "encoder": encoder.state_dict(),
    }
    for name, adapter in adapters.items():
        state_blob[f"adp_{name}"] = adapter.state_dict()
    for name, gen in deep_prefix_generators.items():
        state_blob[f"deep_prefix_{name}"] = gen.state_dict()
    if latent_refiner is not None:
        state_blob["refiner"] = latent_refiner.state_dict()
    for name, head in gist_heads.items():
        state_blob[f"gist_{name}"] = head.state_dict()

    artifacts = {
        "encoder.pt": encoder.state_dict(),
        "state.pt": state_blob,
        "config.json": cfg,
    }
    for name, adapter in adapters.items():
        artifacts[f"adapter_{name}.pt"] = adapter.state_dict()
    for name, gen in deep_prefix_generators.items():
        artifacts[f"deep_prefix_{name}.pt"] = gen.state_dict()
    if latent_refiner is not None:
        artifacts["refiner.pt"] = latent_refiner.state_dict()
    for name, head in gist_heads.items():
        artifacts[f"gist_{name}.pt"] = head.state_dict()
    save_latest_checkpoint(args.save_dir, artifacts, pre_prune=True, post_prune=True, verbose=True)
    print(f"✅ Saved latest checkpoint to {args.save_dir}")

    # Persist PEFT adapters (LoRA / Prefix) if present
    try:
        from peft import PeftModel  # type: ignore

        def _save_peft(model: nn.Module, path: str) -> None:
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)

        if isinstance(getattr(llama, "model", None), PeftModel):
            if args.use_lora:
                _save_peft(llama.model, os.path.join(args.save_dir, "lora_llama"))
                print("📝 Saved LoRA adapters for Llama")
            if args.use_prefix:
                _save_peft(llama.model, os.path.join(args.save_dir, "prefix_llama"))
                print("📝 Saved Prefix-Tuning adapters for Llama")
        if isinstance(getattr(qwen, "model", None), PeftModel):
            if args.use_lora:
                _save_peft(qwen.model, os.path.join(args.save_dir, "lora_qwen"))
                print("📝 Saved LoRA adapters for Qwen")
            if args.use_prefix:
                _save_peft(qwen.model, os.path.join(args.save_dir, "prefix_qwen"))
                print("📝 Saved Prefix-Tuning adapters for Qwen")
    except Exception as exc:
        print(f"[WARN] Skipped PEFT adapter save: {exc}")

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
