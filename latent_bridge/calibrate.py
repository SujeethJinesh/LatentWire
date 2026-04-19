"""
Calibrate a RotAlignKVTranslator on real HuggingFace models.

This script:
  1) Loads a source and target causal LM from Hugging Face.
  2) Runs both on a list of calibration prompts (one per line in a text file),
     capturing their past_key_values.
  3) Fits per-layer Procrustes/ridge alignments in closed form.
  4) Saves the fitted translator to disk.

Usage:
    python scripts/calibrate.py \
        --source-model Qwen/Qwen2.5-0.5B-Instruct \
        --target-model Qwen/Qwen3-0.6B \
        --calibration-file data/calibration_prompts.txt \
        --output checkpoints/qwen25_to_qwen3.pt \
        --bits 4

Phase-1 targets to swap in after the control pair:
    - Qwen/Qwen3.5-0.8B
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    - google/gemma-4-E2B-it
    - Qwen/Qwen3.5-4B (stretch)

Cross-tokenizer caveat: the calibration code below masks out padding tokens and
only fits positions up to the shorter tokenizer-specific valid length for each
prompt. If the two tokenizers fragment the same text differently, the
token-wise pairing is still approximate. This is a known open problem for
cross-family pairs; see method.md §5.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Iterable, Sequence

# Allow running from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rotalign import RotAlignKVTranslator, TranslatorConfig


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", required=True, help="HF model ID for sender LLM")
    p.add_argument("--target-model", required=True, help="HF model ID for receiver LLM")
    p.add_argument(
        "--calibration-file",
        required=True,
        help="Text file with one calibration prompt per line",
    )
    p.add_argument("--output", required=True, help="Where to save the fitted translator")
    p.add_argument("--bits", type=int, default=4, help="Quantization bit rate")
    p.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Pad/truncate all prompts to this token length",
    )
    p.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for KV collection"
    )
    p.add_argument(
        "--alignment",
        choices=[
            "auto",
            "identity",
            "procrustes",
            "procrustes_rand",
            "ridge",
            "cca",
            "reduced_rank",
            "grouped_auto",
            "grouped_identity",
            "grouped_procrustes",
            "grouped_procrustes_rand",
            "grouped_ridge",
            "grouped_cca",
            "grouped_reduced_rank",
            "grouped_transport",
            "grouped_permutation",
            "grouped_signature_transport",
            "grouped_subspace_transport",
            "grouped_canonical_transport",
            "grouped_covariance_transport",
            "grouped_template_transport",
            "grouped_template_subspace_transport",
            "broadcast_template_transport",
            "broadcast_template_ot_transport",
            "broadcast_peak_template_ot_transport",
            "broadcast_retrieval_spectrum_ot_transport",
        ],
        default="auto",
    )
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument(
        "--alignment-rank",
        type=int,
        default=None,
        help="Rank for CCA / reduced-rank regression (defaults to min(d_in, d_out))",
    )
    p.add_argument(
        "--transport-residual-rank",
        type=int,
        default=None,
        help="Optional low-rank residual on top of grouped soft transport",
    )
    p.add_argument(
        "--transport-temperature",
        type=float,
        default=1.0,
        help="Temperature for grouped soft transport weights",
    )
    p.add_argument(
        "--transport-sinkhorn-iters",
        type=int,
        default=8,
        help="Number of Sinkhorn row/column normalization steps for grouped soft transport",
    )
    p.add_argument(
        "--transport-signature-rank",
        type=int,
        default=8,
        help="Top-k singular-value signature size for geometry-aware grouped transport",
    )
    p.add_argument(
        "--transport-signature-weight",
        type=float,
        default=0.0,
        help="Penalty weight for mismatched grouped spectral signatures during grouped transport",
    )
    p.add_argument(
        "--transport-template-bins",
        type=int,
        default=64,
        help="Number of bins for grouped attention-template transport",
    )
    p.add_argument(
        "--canonical-subspace-rank",
        type=int,
        default=None,
        help="Shared low-rank basis size for grouped canonical transport blocks",
    )
    p.add_argument(
        "--rotation",
        choices=["identity", "orthogonal", "hadamard", "dct"],
        default="orthogonal",
        help="Identity, random orthogonal (O(d^2)), randomized Hadamard (O(d log d)), or DCT",
    )
    p.add_argument(
        "--whitening",
        action="store_true",
        help="Apply ZCA whitening of source coords before alignment",
    )
    p.add_argument(
        "--target-whitening",
        action="store_true",
        help="Canonicalize target rotated coordinates too, then dewhiten after projection",
    )
    p.add_argument(
        "--layer-pairing",
        choices=["interp", "cka", "reverse", "shifted", "random"],
        default="interp",
        help="Linear interpolation, SemAlign-style CKA ranking, or negative-control pairing",
    )
    p.add_argument(
        "--layer-selection-topk",
        type=int,
        default=None,
        help="Only transmit the top-k target layers ranked by calibration quality",
    )
    p.add_argument(
        "--layer-selection-ratio",
        type=float,
        default=1.0,
        help="Fraction of target layers to transmit when top-k is unset",
    )
    p.add_argument(
        "--layer-selection-metric",
        choices=["mean_cosine_similarity", "negative_error"],
        default="mean_cosine_similarity",
        help="Metric used to rank layers for selective transmission",
    )
    p.add_argument(
        "--head-selection-topk",
        type=int,
        default=None,
        help="Only transmit the top-k aligned target head-groups ranked by calibration quality",
    )
    p.add_argument(
        "--head-selection-ratio",
        type=float,
        default=1.0,
        help="Fraction of aligned target head-groups to transmit when top-k is unset",
    )
    p.add_argument(
        "--head-selection-metric",
        choices=["mean_cosine_similarity", "negative_error"],
        default="mean_cosine_similarity",
        help="Metric used to rank head-groups for selective transmission",
    )
    p.add_argument(
        "--pre-quant-rank",
        type=int,
        default=None,
        help="Optional rank for target-space low-rank filtering before quantization",
    )
    p.add_argument(
        "--pre-quant-shrinkage",
        type=float,
        default=0.0,
        help="Optional shrinkage strength for the pre-quant low-rank filter",
    )
    p.add_argument(
        "--quantization-correction",
        choices=["none", "affine", "ridge"],
        default="none",
        help="Optional decoder-side correction applied after quantize/dequantize",
    )
    p.add_argument(
        "--learned-fusion-dropout",
        type=float,
        default=0.0,
        help="Dropout rate used when fitting the tiny coordinatewise learned fusion layer",
    )
    p.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="plain",
        help="Prompt template used on the source side during calibration",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _source_reasoning_prompt(prompt: str, mode: str) -> str:
    prompt = prompt.strip()
    if mode == "plain":
        return prompt
    if mode == "brief_analysis":
        return (
            "Briefly analyze this problem in one sentence:\n"
            f"{prompt}\nAnalysis:"
        )
    if mode == "cot":
        return f"Let's think step by step.\n{prompt}\nReasoning:"
    if mode == "scratchpad":
        return f"Use a scratchpad to work through the problem carefully.\n{prompt}\nScratchpad:"
    raise ValueError(f"Unknown source reasoning mode: {mode}")


def batched(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _normalize_past_key_values(pkv):
    """Convert HF cache objects to plain (K, V) tuples."""
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in pkv]


def _append_valid_token_positions(
    per_layer_K: list[list[torch.Tensor]],
    per_layer_V: list[list[torch.Tensor]],
    kv_layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    attention_mask: torch.Tensor,
) -> None:
    """Append only non-padding token positions, one token per batch item.

    This avoids accidentally treating padded sequence positions as calibration
    samples when fitting the closed-form alignment.
    """
    mask_cpu = attention_mask.to("cpu")
    kv_layers_cpu = [
        (
            K.detach().to("cpu", dtype=torch.float32),
            V.detach().to("cpu", dtype=torch.float32),
        )
        for K, V in kv_layers
    ]
    for batch_idx in range(mask_cpu.shape[0]):
        valid_len = int(mask_cpu[batch_idx].sum().item())
        if valid_len <= 0:
            continue
        for pos in range(valid_len):
            for layer_idx, (K, V) in enumerate(kv_layers_cpu):
                per_layer_K[layer_idx].append(
                    K[batch_idx : batch_idx + 1, :, pos : pos + 1, :]
                )
                per_layer_V[layer_idx].append(
                    V[batch_idx : batch_idx + 1, :, pos : pos + 1, :]
                )


def collect_kvs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int,
    batch_size: int,
    device: str,
    reasoning_mode: str = "plain",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run the model on all prompts and concatenate only valid token KVs.

    Returns: list of length num_layers, each (K, V) with K,V shaped
             [N, num_kv_heads, 1, head_dim]. Each batch item corresponds to a
             real token position, so padded positions are never used as
             calibration samples.
    """
    # Make sure pad token is defined (some tokenizers lack it)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    per_layer_K: list[list[torch.Tensor]] = []
    per_layer_V: list[list[torch.Tensor]] = []

    for batch in batched(prompts, batch_size):
        batch = [_source_reasoning_prompt(prompt, reasoning_mode) for prompt in batch]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, use_cache=True, return_dict=True)
        pkv = _normalize_past_key_values(out.past_key_values)
        n_layers = len(pkv)

        if not per_layer_K:
            per_layer_K = [[] for _ in range(n_layers)]
            per_layer_V = [[] for _ in range(n_layers)]

        _append_valid_token_positions(per_layer_K, per_layer_V, pkv, enc["attention_mask"])

    return [
        (torch.cat(per_layer_K[l], dim=0), torch.cat(per_layer_V[l], dim=0))
        for l in range(len(per_layer_K))
    ]


def collect_aligned_kv_pairs(
    src_model: AutoModelForCausalLM,
    src_tokenizer: AutoTokenizer,
    tgt_model: AutoModelForCausalLM,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int,
    batch_size: int,
    device: str,
    source_reasoning_mode: str = "plain",
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Collect source/target KVs while masking pads and aligning prompt lengths.

    For each prompt, only positions up to the shorter tokenizer-specific valid
    length are retained. Each retained token position is emitted as a separate
    batch item with seq_len=1 so downstream fitting never sees padding tokens.
    """
    if src_tokenizer.pad_token_id is None:
        src_tokenizer.pad_token = src_tokenizer.eos_token
    if tgt_tokenizer.pad_token_id is None:
        tgt_tokenizer.pad_token = tgt_tokenizer.eos_token

    src_per_layer_K: list[list[torch.Tensor]] = []
    src_per_layer_V: list[list[torch.Tensor]] = []
    tgt_per_layer_K: list[list[torch.Tensor]] = []
    tgt_per_layer_V: list[list[torch.Tensor]] = []

    for batch in batched(prompts, batch_size):
        src_batch = [_source_reasoning_prompt(prompt, source_reasoning_mode) for prompt in batch]
        tgt_batch = list(batch)
        src_enc = src_tokenizer(
            src_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        tgt_enc = tgt_tokenizer(
            tgt_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            src_out = src_model(**src_enc, use_cache=True, return_dict=True)
            tgt_out = tgt_model(**tgt_enc, use_cache=True, return_dict=True)

        src_pkv = _normalize_past_key_values(src_out.past_key_values)
        tgt_pkv = _normalize_past_key_values(tgt_out.past_key_values)
        src_pkv_cpu = [
            (
                K.detach().to("cpu", dtype=torch.float32),
                V.detach().to("cpu", dtype=torch.float32),
            )
            for K, V in src_pkv
        ]
        tgt_pkv_cpu = [
            (
                K.detach().to("cpu", dtype=torch.float32),
                V.detach().to("cpu", dtype=torch.float32),
            )
            for K, V in tgt_pkv
        ]

        if not src_per_layer_K:
            src_per_layer_K = [[] for _ in range(len(src_pkv))]
            src_per_layer_V = [[] for _ in range(len(src_pkv))]
        if not tgt_per_layer_K:
            tgt_per_layer_K = [[] for _ in range(len(tgt_pkv))]
            tgt_per_layer_V = [[] for _ in range(len(tgt_pkv))]

        src_mask = src_enc["attention_mask"].to("cpu")
        tgt_mask = tgt_enc["attention_mask"].to("cpu")

        for batch_idx in range(src_mask.shape[0]):
            valid_len = min(
                int(src_mask[batch_idx].sum().item()),
                int(tgt_mask[batch_idx].sum().item()),
            )
            if valid_len <= 0:
                continue
            for pos in range(valid_len):
                for layer_idx, (K, V) in enumerate(src_pkv_cpu):
                    src_per_layer_K[layer_idx].append(
                        K[batch_idx : batch_idx + 1, :, pos : pos + 1, :]
                    )
                    src_per_layer_V[layer_idx].append(
                        V[batch_idx : batch_idx + 1, :, pos : pos + 1, :]
                    )
                for layer_idx, (K, V) in enumerate(tgt_pkv_cpu):
                    tgt_per_layer_K[layer_idx].append(
                        K[batch_idx : batch_idx + 1, :, pos : pos + 1, :]
                    )
                    tgt_per_layer_V[layer_idx].append(
                        V[batch_idx : batch_idx + 1, :, pos : pos + 1, :]
                    )

    src_kvs = [
        (torch.cat(src_per_layer_K[l], dim=0), torch.cat(src_per_layer_V[l], dim=0))
        for l in range(len(src_per_layer_K))
    ]
    tgt_kvs = [
        (torch.cat(tgt_per_layer_K[l], dim=0), torch.cat(tgt_per_layer_V[l], dim=0))
        for l in range(len(tgt_per_layer_K))
    ]
    return src_kvs, tgt_kvs


def _resample_position_profile(profile: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    values = profile.float().view(-1)
    if values.numel() == target_len:
        out = values
    else:
        out = F.interpolate(
            values.view(1, 1, -1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return out / out.sum().clamp_min(1e-8)


def _reduce_attention_heads_to_kv_heads(
    attention_map: torch.Tensor,
    *,
    kv_heads: int,
) -> torch.Tensor:
    if kv_heads <= 0:
        raise ValueError("kv_heads must be positive")
    if attention_map.shape[0] == kv_heads:
        return attention_map
    if attention_map.shape[0] % kv_heads == 0:
        factor = attention_map.shape[0] // kv_heads
        return attention_map.view(kv_heads, factor, attention_map.shape[-1]).mean(dim=1)
    interpolated = F.interpolate(
        attention_map.transpose(0, 1).unsqueeze(0),
        size=kv_heads,
        mode="linear",
        align_corners=False,
    ).squeeze(0).transpose(0, 1)
    return interpolated


@torch.no_grad()
def collect_group_attention_templates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: str,
    kv_heads: int,
    group_count: int,
    bins: int,
    template_mode: str = "mean",
    reasoning_mode: str = "plain",
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if kv_heads % group_count != 0:
        raise ValueError(f"kv_heads={kv_heads} must be divisible by group_count={group_count}")
    if template_mode not in {"mean", "peak"}:
        raise ValueError(f"Unknown template_mode: {template_mode}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    group_heads = kv_heads // group_count
    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for batch in batched(prompts, batch_size):
        batch_text = [_source_reasoning_prompt(prompt, reasoning_mode) for prompt in batch]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=False, output_attentions=True, return_dict=True)
        attentions = getattr(out, "attentions", None)
        if attentions is None:
            raise ValueError("Model did not return attentions while building grouped templates")
        if layer_sums is None:
            layer_sums = [torch.zeros(group_count, bins, dtype=torch.float32) for _ in range(len(attentions))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            valid_len = int(mask_cpu[batch_idx].sum().item())
            if valid_len <= 1:
                continue
            for layer_idx, attn in enumerate(attentions):
                head_scores = attn[batch_idx, :, valid_len - 1, : valid_len - 1].detach().to("cpu", dtype=torch.float32)
                head_scores = _reduce_attention_heads_to_kv_heads(head_scores, kv_heads=kv_heads)
                head_scores = head_scores / head_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                grouped = head_scores.reshape(group_count, group_heads, valid_len - 1).mean(dim=1)
                if template_mode == "mean":
                    grouped = torch.stack(
                        [_resample_position_profile(group_scores, bins) for group_scores in grouped],
                        dim=0,
                    )
                else:
                    peak_templates = []
                    for group_scores in grouped:
                        peak_idx = int(torch.argmax(group_scores).item())
                        peak_bin = min(bins - 1, int(peak_idx * bins / max(group_scores.numel(), 1)))
                        one_hot = torch.zeros(bins, dtype=torch.float32)
                        one_hot[peak_bin] = 1.0
                        peak_templates.append(one_hot)
                    grouped = torch.stack(peak_templates, dim=0)
                grouped = grouped / grouped.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                layer_sums[layer_idx] += grouped
            used_prompts += 1
    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build grouped attention templates from calibration prompts")
    return [(templates / float(used_prompts)).cpu() for templates in layer_sums]


def _weighted_key_spectrum(
    head_keys: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    weights = head_weights.float()
    if weights.numel() == 0:
        raise ValueError("head_weights must be non-empty")
    weights = weights / weights.sum().clamp_min(1e-8)
    keys = head_keys.float()
    mean = (weights[:, None] * keys).sum(dim=0, keepdim=True)
    centered = (keys - mean) * weights.sqrt()[:, None]
    spectrum = torch.linalg.svdvals(centered)
    if float(spectrum.sum()) <= 1e-8:
        spectrum = torch.linalg.svdvals(keys * weights.sqrt()[:, None])
    out = torch.zeros(rank, dtype=torch.float32)
    take = min(rank, int(spectrum.numel()))
    if take > 0:
        out[:take] = spectrum[:take]
    return out / out.sum().clamp_min(1e-8)


@torch.no_grad()
def collect_group_key_signatures(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: str,
    kv_heads: int,
    group_count: int,
    rank: int,
    reasoning_mode: str = "plain",
) -> list[torch.Tensor]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if kv_heads % group_count != 0:
        raise ValueError(f"kv_heads={kv_heads} must be divisible by group_count={group_count}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    group_heads = kv_heads // group_count
    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for batch in batched(prompts, batch_size):
        batch_text = [_source_reasoning_prompt(prompt, reasoning_mode) for prompt in batch]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, output_attentions=True, return_dict=True)
        attentions = getattr(out, "attentions", None)
        if attentions is None:
            raise ValueError("Model did not return attentions while building grouped key signatures")
        pkv = _normalize_past_key_values(out.past_key_values)
        if layer_sums is None:
            layer_sums = [torch.zeros(group_count, rank, dtype=torch.float32) for _ in range(len(attentions))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            valid_len = int(mask_cpu[batch_idx].sum().item())
            if valid_len <= 1:
                continue
            for layer_idx, attn in enumerate(attentions):
                head_scores = attn[batch_idx, :, valid_len - 1, : valid_len - 1].detach().to("cpu", dtype=torch.float32)
                head_scores = _reduce_attention_heads_to_kv_heads(head_scores, kv_heads=kv_heads)
                head_scores = head_scores / head_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                layer_keys = pkv[layer_idx][0][batch_idx, :, : valid_len - 1, :].detach().to("cpu", dtype=torch.float32)
                group_signatures = []
                for group_idx in range(group_count):
                    start = group_idx * group_heads
                    stop = start + group_heads
                    head_sigs = [
                        _weighted_key_spectrum(layer_keys[head_idx], head_scores[head_idx], rank=rank)
                        for head_idx in range(start, stop)
                    ]
                    group_signatures.append(torch.stack(head_sigs, dim=0).mean(dim=0))
                layer_sums[layer_idx] += torch.stack(group_signatures, dim=0)
            used_prompts += 1
    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build grouped key signatures from calibration prompts")
    return [(signatures / float(used_prompts)).cpu() for signatures in layer_sums]


def main() -> None:
    args = parse_args()
    dtype = torch_dtype(args.dtype)

    prompts = load_prompts(args.calibration_file)
    print(f"Loaded {len(prompts)} calibration prompts from {args.calibration_file}")

    print(f"\nLoading source model: {args.source_model}")
    tok_s = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    src = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    print(f"Loading target model: {args.target_model}")
    tok_t = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    tgt = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    print("\nCollecting aligned source/target KVs...")
    src_kvs, tgt_kvs = collect_aligned_kv_pairs(
        src,
        tok_s,
        tgt,
        tok_t,
        prompts,
        args.max_length,
        args.batch_size,
        args.device,
        source_reasoning_mode=args.source_reasoning_mode,
    )

    # Inspect shapes to build the config.
    K_s0 = src_kvs[0][0]  # [N, h_s, S, d_s]
    K_t0 = tgt_kvs[0][0]
    print(f"\nSource KV shape: {tuple(K_s0.shape)}  "
          f"(n_samples, num_kv_heads, seq_len, head_dim)")
    print(f"Target KV shape: {tuple(K_t0.shape)}")
    print(f"Source layers: {len(src_kvs)}  |  Target layers: {len(tgt_kvs)}")

    # Check seq-length mismatch from cross-tokenizer fragmentation
    if K_s0.shape[2] != K_t0.shape[2]:
        min_seq = min(K_s0.shape[2], K_t0.shape[2])
        print(
            f"WARNING: seq-length mismatch (src={K_s0.shape[2]}, tgt={K_t0.shape[2]}); "
            f"truncating both to {min_seq}. See method.md §5 for the cross-tokenizer "
            f"caveat."
        )
        src_kvs = [(K[:, :, :min_seq, :], V[:, :, :min_seq, :]) for K, V in src_kvs]
        tgt_kvs = [(K[:, :, :min_seq, :], V[:, :, :min_seq, :]) for K, V in tgt_kvs]

    config = TranslatorConfig(
        src_head_dim=K_s0.shape[-1],
        src_num_heads=K_s0.shape[1],
        num_src_layers=len(src_kvs),
        tgt_head_dim=K_t0.shape[-1],
        tgt_num_heads=K_t0.shape[1],
        num_tgt_layers=len(tgt_kvs),
        quant_bits=args.bits,
        rotation_kind=args.rotation,
        use_whitening=args.whitening,
        use_target_whitening=args.target_whitening,
        alignment_method=args.alignment,
        ridge_lambda=args.ridge_lambda,
        alignment_rank=args.alignment_rank,
        transport_residual_rank=args.transport_residual_rank,
        transport_temperature=args.transport_temperature,
        transport_sinkhorn_iters=args.transport_sinkhorn_iters,
        transport_signature_rank=args.transport_signature_rank,
        transport_signature_weight=args.transport_signature_weight,
        transport_template_bins=args.transport_template_bins,
        canonical_subspace_rank=args.canonical_subspace_rank,
        layer_pairing=args.layer_pairing,
        layer_selection_topk=args.layer_selection_topk,
        layer_selection_ratio=args.layer_selection_ratio,
        layer_selection_metric=args.layer_selection_metric,
        head_selection_topk=args.head_selection_topk,
        head_selection_ratio=args.head_selection_ratio,
        head_selection_metric=args.head_selection_metric,
        pre_quant_rank=args.pre_quant_rank,
        pre_quant_shrinkage=args.pre_quant_shrinkage,
        quantization_correction=args.quantization_correction,
        learned_fusion_dropout=args.learned_fusion_dropout,
        seed=args.seed,
    )
    print(f"\nBuilding translator with config:\n  {config}")
    translator = RotAlignKVTranslator(config)

    if args.alignment in {
        "grouped_template_transport",
        "grouped_template_subspace_transport",
        "broadcast_template_transport",
        "broadcast_template_ot_transport",
        "broadcast_peak_template_ot_transport",
        "broadcast_retrieval_spectrum_ot_transport",
    }:
        group_count = math.gcd(config.src_num_heads, config.tgt_num_heads)
        is_broadcast = args.alignment in {
            "broadcast_template_transport",
            "broadcast_template_ot_transport",
            "broadcast_peak_template_ot_transport",
            "broadcast_retrieval_spectrum_ot_transport",
        }
        template_mode = "peak" if args.alignment == "broadcast_peak_template_ot_transport" else "mean"
        src_template_groups = config.src_num_heads if is_broadcast else group_count
        tgt_template_groups = config.tgt_num_heads if is_broadcast else group_count
        if args.alignment == "broadcast_retrieval_spectrum_ot_transport":
            print(
                "\nBuilding grouped retrieval-weighted key signatures from calibration prompts "
                f"(source groups={src_template_groups}, target groups={tgt_template_groups}, rank={args.transport_signature_rank})..."
            )
            translator._transport_src_group_templates = collect_group_key_signatures(
                src,
                tok_s,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.src_num_heads,
                group_count=src_template_groups,
                rank=args.transport_signature_rank,
                reasoning_mode=args.source_reasoning_mode,
            )
            translator._transport_tgt_group_templates = collect_group_key_signatures(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=tgt_template_groups,
                rank=args.transport_signature_rank,
                reasoning_mode="plain",
            )
            print(
                "Built grouped key signatures: "
                f"source layers={len(translator._transport_src_group_templates)}, "
                f"target layers={len(translator._transport_tgt_group_templates)}"
            )
        else:
            print(
                "\nBuilding grouped attention templates from calibration prompts "
                f"(source groups={src_template_groups}, target groups={tgt_template_groups}, bins={args.transport_template_bins}, mode={template_mode})..."
            )
            translator._transport_src_group_templates = collect_group_attention_templates(
                src,
                tok_s,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.src_num_heads,
                group_count=src_template_groups,
                bins=args.transport_template_bins,
                template_mode=template_mode,
                reasoning_mode=args.source_reasoning_mode,
            )
            translator._transport_tgt_group_templates = collect_group_attention_templates(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=tgt_template_groups,
                bins=args.transport_template_bins,
                template_mode=template_mode,
                reasoning_mode="plain",
            )
            print(
                "Built grouped attention templates: "
                f"source layers={len(translator._transport_src_group_templates)}, "
                f"target layers={len(translator._transport_tgt_group_templates)}"
            )

    del src
    del tgt
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("\nFitting closed-form alignments (Procrustes / ridge)...")
    diagnostics = translator.fit_from_pairs(src_kvs, tgt_kvs, verbose=args.verbose)

    avg_cos_K = sum(d["K"]["mean_cosine_similarity"] for d in diagnostics.values()) / len(diagnostics)
    avg_cos_V = sum(d["V"]["mean_cosine_similarity"] for d in diagnostics.values()) / len(diagnostics)
    avg_err_K = sum(d["K"]["relative_frobenius_error"] for d in diagnostics.values()) / len(diagnostics)
    avg_err_V = sum(d["V"]["relative_frobenius_error"] for d in diagnostics.values()) / len(diagnostics)
    print(
        f"\nAverage alignment quality across {len(diagnostics)} layers:\n"
        f"    K: cos_sim = {avg_cos_K:.3f}  rel_frobenius_err = {avg_err_K:.3f}\n"
        f"    V: cos_sim = {avg_cos_V:.3f}  rel_frobenius_err = {avg_err_V:.3f}"
    )
    print(f"Selected target layers: {translator.selected_layer_indices()}")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    translator.save(str(out_path))
    print(f"\nSaved fitted translator to {out_path}")


if __name__ == "__main__":
    main()
