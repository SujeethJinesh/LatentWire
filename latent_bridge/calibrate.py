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
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Iterable, Sequence

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
            "grouped_adaptive_canonical_transport",
            "grouped_rotational_transport",
            "grouped_fitted_rotation_transport",
            "grouped_shared_basis_transport",
            "grouped_covariance_transport",
            "grouped_template_transport",
            "grouped_qk_retrieval_transport",
            "grouped_contrastive_template_transport",
            "grouped_template_subspace_transport",
            "broadcast_template_transport",
            "broadcast_template_ot_transport",
            "broadcast_peak_template_ot_transport",
            "broadcast_retrieval_spectrum_ot_transport",
            "broadcast_qk_template_ot_transport",
        ],
        default="auto",
    )
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument(
        "--fit-ridge-override-lambda",
        type=float,
        default=None,
        help="Override source->target fit ridge lambda for selected target layers/streams",
    )
    p.add_argument(
        "--fit-ridge-override-streams",
        choices=["kv", "k", "v"],
        default="kv",
        help="Streams receiving the fit ridge override",
    )
    p.add_argument(
        "--fit-ridge-override-layer",
        type=int,
        action="append",
        dest="fit_ridge_override_layers",
        default=None,
        help="Limit the fit ridge override to a target layer; repeat to select multiple layers",
    )
    p.add_argument(
        "--fit-ridge-protected-rank",
        type=int,
        default=None,
        help="Keep this many high-innovation output channels at the base ridge while the tail uses the fit ridge override",
    )
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
        "--whitening-streams",
        choices=["kv", "k", "v"],
        default="kv",
        help="Which source streams to whiten: both (`kv`), keys only (`k`), or values only (`v`)",
    )
    p.add_argument(
        "--target-whitening",
        action="store_true",
        help="Canonicalize target rotated coordinates too, then dewhiten after projection",
    )
    p.add_argument(
        "--target-whitening-streams",
        choices=["kv", "k", "v"],
        default="kv",
        help="Which target streams to canonicalize/dewhiten: both (`kv`), keys only (`k`), or values only (`v`)",
    )
    p.add_argument(
        "--conditioning-target-layer",
        type=int,
        action="append",
        dest="conditioning_target_layers",
        default=None,
        help="Limit conditioning to specific target layers; repeat the flag to select multiple layers",
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
        choices=[
            "none",
            "affine",
            "bridge_affine",
            "bridge_ridge",
            "bridge_ridge_query",
            "bridge_low_rank_bank",
            "bridge_ridge_residual_bank",
            "bridge_ridge_qk_residual_bank",
            "bridge_ridge_qk_cab_bank",
            "bridge_ridge_qk_weighted",
            "bridge_ridge_qk_projector",
            "bridge_ridge_qk_adapter",
            "bridge_ridge_qk_affinity_adapter",
            "bridge_ridge_qk_attnkl_adapter",
            "bridge_ridge_qk_cab_adapter",
            "bridge_ridge_qk_emkd_adapter",
            "bridge_ridge_qk_readout_adapter",
            "bridge_ridge_qk_predkl_adapter",
            "bridge_ridge_qk_asym_adapter",
            "bridge_ridge_qk_asym_projector",
            "bridge_ridge_qk_asym_predkl_adapter",
            "bridge_ridge_qk_asym_dynmap_adapter",
            "bridge_ridge_qk_xattn_adapter",
            "bridge_ridge_qk_xattn_dynmap_adapter",
            "bridge_ridge_qk_module_adapter",
            "bridge_ridge_qk_module_replace",
            "bridge_ridge_qk_bytespan_module_replace",
            "bridge_ridge_qk_spanalign_module_replace",
            "bridge_ridge_qk_ctxalign_module_replace",
            "bridge_ridge_qk_dynalign_ctxonly_module_replace",
            "bridge_ridge_qk_dynalign_module_replace",
            "bridge_ridge_qk_dynalign_preserve_module_replace",
            "bridge_ridge_qk_dynalign_eigenspace_module_replace",
            "bridge_ridge_qk_dynalign_saliency_module_replace",
            "bridge_ridge_qk_dynalign_saliency_preserve_module_replace",
            "bridge_ridge_qk_dynalign_anchor_tail_module_replace",
            "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace",
            "bridge_ridge_qk_dynalign_routed_module_replace",
            "bridge_ridge_qk_dynalign_value_routed_module_replace",
            "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
            "bridge_ridge_qk_dynalign_value_bank_module_replace",
            "bridge_ridge_qk_dynalign_value_query_bank_module_replace",
            "bridge_ridge_qk_dynalign_value_routed_bank_module_replace",
            "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace",
            "bridge_ridge_qk_dynalign_dwakd_module_replace",
            "bridge_ridge_qk_dynalign_likelihood_module_replace",
            "bridge_ridge_qk_dynalign_spanalm_module_replace",
            "bridge_ridge_qk_dynalign_prefdist_module_replace",
            "bridge_ridge_qk_dynalign_dwainteract_module_replace",
            "bridge_ridge_qk_dynalign_interact_module_replace",
            "bridge_ridge_qk_dpalign_module_replace",
            "bridge_ridge_qk_tokenbasis_replace",
            "bridge_ridge_qk_sae_adapter",
            "bridge_ridge_qk_generated_adapter",
            "bridge_ridge_qk_predkl_bank",
            "ridge",
            "low_rank",
        ],
        default="none",
        help="Optional decoder-side correction applied after quantize/dequantize",
    )
    p.add_argument(
        "--quantization-correction-rank",
        type=int,
        default=None,
        help="Optional rank for low-rank decoder-side correction after quantize/dequantize",
    )
    p.add_argument(
        "--bridge-bank-size",
        type=int,
        default=4,
        help="Number of bridge experts to fit for query-conditioned bridge-bank correction",
    )
    p.add_argument(
        "--learned-fusion-dropout",
        type=float,
        default=0.0,
        help="Dropout rate used when fitting the tiny coordinatewise learned fusion layer",
    )
    p.add_argument(
        "--innovation-target-set-json",
        default=None,
        help=(
            "SVAMP32 innovation target-set JSON. When used with "
            "bridge_ridge_qk_dynalign_query_innovation_resampler_replace, clean residual "
            "target IDs receive higher module-fit weight."
        ),
    )
    p.add_argument(
        "--innovation-positive-weight",
        type=float,
        default=16.0,
        help="Prompt-level sample weight for IDs in ids.clean_residual_targets.",
    )
    p.add_argument(
        "--innovation-default-weight",
        type=float,
        default=1.0,
        help="Prompt-level sample weight for non-target IDs when innovation target weighting is enabled.",
    )
    p.add_argument(
        "--source-reasoning-mode",
        choices=["plain", "brief_analysis", "cot", "scratchpad"],
        default="plain",
        help="Prompt template used on the source side during calibration",
    )
    p.add_argument(
        "--source-use-chat-template",
        action="store_true",
        help="Format source calibration prompts through tokenizer.apply_chat_template when available.",
    )
    p.add_argument(
        "--target-use-chat-template",
        action="store_true",
        help="Format target calibration prompts through tokenizer.apply_chat_template when available.",
    )
    p.add_argument(
        "--source-enable-thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help="Thinking flag passed into the source tokenizer chat template when supported.",
    )
    p.add_argument(
        "--target-enable-thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help="Thinking flag passed into the target tokenizer chat template when supported.",
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


def _stable_example_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _prompt_record_ids(obj: dict[str, Any], prompt: str) -> set[str]:
    ids: set[str] = set()
    for key in ("example_id", "id"):
        value = obj.get(key)
        if value is not None:
            ids.add(str(value))
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for key in ("example_id", "id"):
            value = metadata.get(key)
            if value is not None:
                ids.add(str(value))
    raw_answer = obj.get("answer_text", obj.get("answer", obj.get("target")))
    if raw_answer is not None:
        aliases = obj.get("aliases", [])
        answers = [str(raw_answer), *[str(alias) for alias in aliases]]
        ids.add(_stable_example_id({"prompt": str(prompt), "answers": answers}))
    return ids


def load_prompt_records(path: str) -> list[tuple[str, set[str]]]:
    records: list[tuple[str, set[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("question")
                if prompt is None:
                    raise ValueError("JSONL calibration records require `prompt` or `question`")
                records.append((str(prompt).strip(), _prompt_record_ids(obj, str(prompt))))
            else:
                records.append((line, set()))
    return records


def load_prompts(path: str) -> list[str]:
    return [prompt for prompt, _ in load_prompt_records(path)]


def load_innovation_target_ids(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ids_payload = payload.get("ids", {}) if isinstance(payload, dict) else {}
    target_ids = ids_payload.get("clean_residual_targets", []) if isinstance(ids_payload, dict) else []
    target_set = {str(item) for item in target_ids}
    if not target_set:
        raise ValueError(
            f"{path} does not contain any ids.clean_residual_targets entries"
        )
    return target_set


def build_innovation_prompt_weights(
    prompt_example_ids: Sequence[set[str]],
    target_ids: set[str],
    *,
    positive_weight: float,
    default_weight: float,
) -> tuple[torch.Tensor, int]:
    if positive_weight <= 0.0 or default_weight <= 0.0:
        raise ValueError("innovation prompt weights must be positive")
    weights: list[float] = []
    matched = 0
    for ids in prompt_example_ids:
        if ids & target_ids:
            weights.append(float(positive_weight))
            matched += 1
        else:
            weights.append(float(default_weight))
    return torch.tensor(weights, dtype=torch.float32), matched


def _optional_bool_from_arg(value: str) -> bool | None:
    lowered = value.lower()
    if lowered == "auto":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Unknown tri-state bool value: {value}")


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


def _format_prompt_for_tokenizer(
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
    enable_thinking: bool | None,
) -> str:
    prompt = prompt.strip()
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _prepare_prompt_text(
    prompt: str,
    *,
    reasoning_mode: str,
    tokenizer: AutoTokenizer,
    use_chat_template: bool,
    enable_thinking: bool | None,
) -> str:
    return _format_prompt_for_tokenizer(
        tokenizer,
        _source_reasoning_prompt(prompt, reasoning_mode),
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )


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
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
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
        batch = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
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


def _token_offsets_for_prompt_text(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    max_length: int,
) -> list[tuple[int, int]] | None:
    try:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return None
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _prompt_valid_length_for_text(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    max_length: int,
) -> int:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc.get("input_ids")
    if input_ids is None:
        raise ValueError("Tokenizer did not return input_ids while computing valid length")
    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return int(len(input_ids))


def _collect_prompt_content_token_spans(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    raw_prompt: str,
    *,
    max_length: int,
) -> list[tuple[int, int, int]] | None:
    raw_prompt = raw_prompt.strip()
    if not raw_prompt:
        return []
    prompt_start = prompt_text.find(raw_prompt)
    if prompt_start < 0:
        return None
    offsets = _token_offsets_for_prompt_text(tokenizer, prompt_text, max_length=max_length)
    if offsets is None:
        return None
    prompt_end = prompt_start + len(raw_prompt)
    spans: list[tuple[int, int, int]] = []
    for token_pos, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        span_start = max(start, prompt_start)
        span_end = min(end, prompt_end)
        if span_end <= span_start:
            continue
        spans.append((token_pos, span_start - prompt_start, span_end - prompt_start))
    return spans


def _byte_prefix_lengths(text: str) -> list[int]:
    prefix = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        prefix.append(total)
    return prefix


def _collect_prompt_content_token_byte_spans(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    raw_prompt: str,
    *,
    max_length: int,
) -> list[tuple[int, int, int]] | None:
    raw_prompt = raw_prompt.strip()
    if not raw_prompt:
        return []
    prompt_start = prompt_text.find(raw_prompt)
    if prompt_start < 0:
        return None
    offsets = _token_offsets_for_prompt_text(tokenizer, prompt_text, max_length=max_length)
    if offsets is None:
        return None
    prompt_end = prompt_start + len(raw_prompt)
    byte_prefix = _byte_prefix_lengths(raw_prompt)
    spans: list[tuple[int, int, int]] = []
    for token_pos, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        span_start = max(start, prompt_start)
        span_end = min(end, prompt_end)
        if span_end <= span_start:
            continue
        rel_start = span_start - prompt_start
        rel_end = span_end - prompt_start
        spans.append((token_pos, byte_prefix[rel_start], byte_prefix[rel_end]))
    return spans


def _align_token_spans_monotone(
    src_spans: Sequence[tuple[int, int, int]],
    tgt_spans: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    src_idx = 0
    tgt_idx = 0
    while src_idx < len(src_spans) and tgt_idx < len(tgt_spans):
        src_pos, src_start, src_end = src_spans[src_idx]
        tgt_pos, tgt_start, tgt_end = tgt_spans[tgt_idx]
        overlap_start = max(src_start, tgt_start)
        overlap_end = min(src_end, tgt_end)
        if overlap_end > overlap_start:
            pairs.append((src_pos, tgt_pos))
            if src_end <= tgt_end:
                src_idx += 1
            if tgt_end <= src_end:
                tgt_idx += 1
            continue
        if src_end <= tgt_start:
            src_idx += 1
            continue
        if tgt_end <= src_start:
            tgt_idx += 1
            continue
        if src_end < tgt_end:
            src_idx += 1
        else:
            tgt_idx += 1
    return pairs


def _align_token_spans_byte_dominant(
    src_spans: Sequence[tuple[int, int, int]],
    tgt_spans: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    tgt_scan_start = 0
    for src_pos, src_start, src_end in src_spans:
        best: tuple[int, float, int, int] | None = None
        best_tgt_pos: int | None = None
        tgt_idx = tgt_scan_start
        while tgt_idx < len(tgt_spans):
            tgt_pos, tgt_start, tgt_end = tgt_spans[tgt_idx]
            if tgt_end <= src_start:
                tgt_scan_start = tgt_idx + 1
                tgt_idx += 1
                continue
            if tgt_start >= src_end:
                break
            overlap = max(0, min(src_end, tgt_end) - max(src_start, tgt_start))
            if overlap > 0:
                src_len = max(src_end - src_start, 1)
                tgt_len = max(tgt_end - tgt_start, 1)
                union = max(src_end, tgt_end) - min(src_start, tgt_start)
                center_gap = abs((src_start + src_end) - (tgt_start + tgt_end))
                score = (overlap, overlap / float(max(union, 1)), -center_gap, -max(src_len, tgt_len))
                if best is None or score > best:
                    best = score
                    best_tgt_pos = int(tgt_pos)
            tgt_idx += 1
        if best_tgt_pos is not None:
            pairs.append((int(src_pos), best_tgt_pos))
    return pairs


def _span_char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = text.lower()
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[idx : idx + n] for idx in range(len(normalized) - n + 1)}


def _span_kind(text: str) -> str:
    if not text:
        return "empty"
    if text.isspace():
        return "space"
    if text.isdigit():
        return "digit"
    if text.isalpha():
        return "alpha"
    if all(not ch.isalnum() and not ch.isspace() for ch in text):
        return "punct"
    return "mixed"


def _contextual_alignment_score(
    raw_prompt: str,
    src_span: tuple[int, int, int],
    tgt_span: tuple[int, int, int],
    *,
    context_window: int = 4,
) -> float:
    _, src_start, src_end = src_span
    _, tgt_start, tgt_end = tgt_span
    src_len = max(src_end - src_start, 1)
    tgt_len = max(tgt_end - tgt_start, 1)
    overlap = max(0, min(src_end, tgt_end) - max(src_start, tgt_start))
    union = max(src_end, tgt_end) - min(src_start, tgt_start)
    overlap_ratio = float(overlap) / float(max(union, 1))
    src_mid = 0.5 * float(src_start + src_end)
    tgt_mid = 0.5 * float(tgt_start + tgt_end)
    prompt_scale = float(max(len(raw_prompt), 1))
    center_penalty = abs(src_mid - tgt_mid) / prompt_scale
    length_ratio = float(min(src_len, tgt_len)) / float(max(src_len, tgt_len))

    src_text = raw_prompt[src_start:src_end]
    tgt_text = raw_prompt[tgt_start:tgt_end]
    src_ngrams = _span_char_ngrams(src_text)
    tgt_ngrams = _span_char_ngrams(tgt_text)
    if src_ngrams or tgt_ngrams:
        token_jaccard = float(len(src_ngrams & tgt_ngrams)) / float(max(len(src_ngrams | tgt_ngrams), 1))
    else:
        token_jaccard = 0.0

    src_ctx = raw_prompt[max(0, src_start - context_window) : min(len(raw_prompt), src_end + context_window)]
    tgt_ctx = raw_prompt[max(0, tgt_start - context_window) : min(len(raw_prompt), tgt_end + context_window)]
    src_ctx_ngrams = _span_char_ngrams(src_ctx)
    tgt_ctx_ngrams = _span_char_ngrams(tgt_ctx)
    if src_ctx_ngrams or tgt_ctx_ngrams:
        context_jaccard = float(len(src_ctx_ngrams & tgt_ctx_ngrams)) / float(max(len(src_ctx_ngrams | tgt_ctx_ngrams), 1))
    else:
        context_jaccard = 0.0

    kind_bonus = 1.0 if _span_kind(src_text) == _span_kind(tgt_text) else 0.0
    return (
        2.5 * overlap_ratio
        + 1.25 * token_jaccard
        + 0.75 * context_jaccard
        + 0.5 * length_ratio
        + 0.25 * kind_bonus
        - 1.25 * center_penalty
    )


def _decode_token_piece(
    tokenizer: AutoTokenizer,
    token_id: int,
    cache: dict[int, str],
) -> str:
    token_id = int(token_id)
    cached = cache.get(token_id)
    if cached is not None:
        return cached
    text = ""
    try:
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([token_id])
    except Exception:
        text = ""
    if not text:
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            converted = convert(token_id)
            if isinstance(converted, list):
                converted = converted[0] if converted else ""
            text = str(converted)
    cache[token_id] = text
    return text


def _token_text_similarity(src_text: str, tgt_text: str) -> float:
    src_norm = src_text.strip().lower()
    tgt_norm = tgt_text.strip().lower()
    if not src_norm or not tgt_norm:
        return 0.0
    if src_norm == tgt_norm:
        return 1.0
    exact_containment = 1.0 if src_norm in tgt_norm or tgt_norm in src_norm else 0.0
    src_ngrams = _span_char_ngrams(src_norm)
    tgt_ngrams = _span_char_ngrams(tgt_norm)
    if src_ngrams or tgt_ngrams:
        jaccard = float(len(src_ngrams & tgt_ngrams)) / float(max(len(src_ngrams | tgt_ngrams), 1))
    else:
        jaccard = 0.0
    kind_bonus = 1.0 if _span_kind(src_norm) == _span_kind(tgt_norm) else 0.0
    return min(1.0, 0.7 * jaccard + 0.2 * exact_containment + 0.1 * kind_bonus)


def _prediction_token_overlap_score(
    src_texts: Sequence[str],
    src_log_probs: Sequence[float],
    tgt_texts: Sequence[str],
    tgt_log_probs: Sequence[float],
) -> float:
    if not src_texts or not tgt_texts:
        return 0.0
    src_log = torch.tensor(list(src_log_probs), dtype=torch.float32)
    tgt_log = torch.tensor(list(tgt_log_probs), dtype=torch.float32)
    src_probs = torch.softmax(src_log, dim=0)
    tgt_probs = torch.softmax(tgt_log, dim=0)
    score = 0.0
    for src_idx, src_text in enumerate(src_texts):
        for tgt_idx, tgt_text in enumerate(tgt_texts):
            score += float(src_probs[src_idx].item() * tgt_probs[tgt_idx].item()) * _token_text_similarity(src_text, tgt_text)
    return score


@torch.no_grad()
def _collect_prompt_prediction_signatures(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    *,
    max_length: int,
    device: str,
    topk: int = 8,
) -> list[tuple[tuple[str, ...], tuple[float, ...]]]:
    if topk <= 0:
        raise ValueError("topk must be positive")
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    out = model(**enc, use_cache=True, return_dict=True)
    logits = out.logits[0].detach().to("cpu", dtype=torch.float32)
    valid_len = int(enc["attention_mask"][0].sum().item())
    decode_cache: dict[int, str] = {}
    signatures: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for pos in range(valid_len):
        token_logits = logits[pos]
        k = min(int(topk), int(token_logits.shape[-1]))
        top_logits, top_ids = torch.topk(token_logits, k=k, dim=-1)
        top_log_probs = top_logits - torch.logsumexp(token_logits, dim=-1)
        texts = tuple(_decode_token_piece(tokenizer, int(token_id), decode_cache) for token_id in top_ids.tolist())
        signatures.append((texts, tuple(float(value) for value in top_log_probs.tolist())))
    return signatures


def collect_contextual_prompt_position_mixtures(
    src_tokenizer: AutoTokenizer,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    source_reasoning_mode: str = "plain",
    source_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_use_chat_template: bool = False,
    target_enable_thinking: bool | None = None,
    max_targets_per_source: int = 3,
    score_temperature: float = 0.35,
) -> list[list[tuple[int, tuple[int, ...], tuple[float, ...]]]]:
    del batch_size  # Alignment is prompt-local and cheap enough to build one prompt at a time.
    mixtures: list[list[tuple[int, tuple[int, ...], tuple[float, ...]]]] = []
    max_targets_per_source = max(1, int(max_targets_per_source))
    temperature = max(float(score_temperature), 1e-4)
    for prompt in prompts:
        src_text = _prepare_prompt_text(
            prompt,
            reasoning_mode=source_reasoning_mode,
            tokenizer=src_tokenizer,
            use_chat_template=source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        tgt_text = _format_prompt_for_tokenizer(
            tgt_tokenizer,
            prompt,
            use_chat_template=target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        src_spans = _collect_prompt_content_token_spans(
            src_tokenizer,
            src_text,
            prompt,
            max_length=max_length,
        )
        tgt_spans = _collect_prompt_content_token_spans(
            tgt_tokenizer,
            tgt_text,
            prompt,
            max_length=max_length,
        )
        prompt_mixtures: list[tuple[int, tuple[int, ...], tuple[float, ...]]] = []
        if src_spans is not None and tgt_spans is not None and src_spans and tgt_spans:
            anchor_pairs = _align_token_spans_monotone(src_spans, tgt_spans)
            anchor_map: dict[int, list[int]] = {}
            for src_pos, tgt_pos in anchor_pairs:
                anchor_map.setdefault(int(src_pos), []).append(int(tgt_pos))
            tgt_pos_to_idx = {int(token_pos): idx for idx, (token_pos, _, _) in enumerate(tgt_spans)}
            for src_token_pos, src_start, src_end in src_spans:
                anchor_positions = anchor_map.get(int(src_token_pos))
                if anchor_positions:
                    anchor_center = int(round(sum(anchor_positions) / float(len(anchor_positions))))
                else:
                    src_mid = 0.5 * float(src_start + src_end)
                    anchor_center = min(
                        tgt_spans,
                        key=lambda span: abs(0.5 * float(span[1] + span[2]) - src_mid),
                    )[0]
                anchor_idx = tgt_pos_to_idx.get(int(anchor_center), 0)
                candidate_indices: set[int] = {anchor_idx}
                for tgt_idx, (_, tgt_start, tgt_end) in enumerate(tgt_spans):
                    overlap = max(0, min(src_end, tgt_end) - max(src_start, tgt_start))
                    if overlap > 0:
                        candidate_indices.add(tgt_idx)
                for delta in (-1, 1):
                    candidate_idx = anchor_idx + delta
                    if 0 <= candidate_idx < len(tgt_spans):
                        candidate_indices.add(candidate_idx)
                scored_candidates: list[tuple[float, int]] = []
                for tgt_idx in sorted(candidate_indices):
                    tgt_span = tgt_spans[tgt_idx]
                    score = _contextual_alignment_score(prompt, (src_token_pos, src_start, src_end), tgt_span)
                    scored_candidates.append((score, tgt_idx))
                if not scored_candidates:
                    continue
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                kept = scored_candidates[:max_targets_per_source]
                score_tensor = torch.tensor([score for score, _ in kept], dtype=torch.float32)
                probs = torch.softmax(score_tensor / temperature, dim=0)
                tgt_positions = tuple(int(tgt_spans[tgt_idx][0]) for _, tgt_idx in kept)
                tgt_weights = tuple(float(weight) for weight in probs.tolist())
                prompt_mixtures.append((int(src_token_pos), tgt_positions, tgt_weights))
        if not prompt_mixtures:
            src_len = _prompt_valid_length_for_text(src_tokenizer, src_text, max_length=max_length)
            tgt_len = _prompt_valid_length_for_text(tgt_tokenizer, tgt_text, max_length=max_length)
            prompt_mixtures = [
                (int(pos), (int(pos),), (1.0,))
                for pos in range(min(src_len, tgt_len))
            ]
        mixtures.append(prompt_mixtures)
    return mixtures


def collect_dynamic_prompt_position_mixtures(
    src_model: AutoModelForCausalLM,
    src_tokenizer: AutoTokenizer,
    tgt_model: AutoModelForCausalLM,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: str,
    source_reasoning_mode: str = "plain",
    source_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_use_chat_template: bool = False,
    target_enable_thinking: bool | None = None,
    max_targets_per_source: int = 3,
    score_temperature: float = 0.35,
    prediction_topk: int = 8,
    prediction_weight: float = 1.5,
) -> list[list[tuple[int, tuple[int, ...], tuple[float, ...]]]]:
    del batch_size  # Alignment is prompt-local.
    mixtures: list[list[tuple[int, tuple[int, ...], tuple[float, ...]]]] = []
    max_targets_per_source = max(1, int(max_targets_per_source))
    temperature = max(float(score_temperature), 1e-4)
    pred_weight = float(prediction_weight)
    use_prediction_score = abs(pred_weight) > 1e-12
    for prompt in prompts:
        src_text = _prepare_prompt_text(
            prompt,
            reasoning_mode=source_reasoning_mode,
            tokenizer=src_tokenizer,
            use_chat_template=source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        tgt_text = _format_prompt_for_tokenizer(
            tgt_tokenizer,
            prompt,
            use_chat_template=target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        src_spans = _collect_prompt_content_token_spans(
            src_tokenizer,
            src_text,
            prompt,
            max_length=max_length,
        )
        tgt_spans = _collect_prompt_content_token_spans(
            tgt_tokenizer,
            tgt_text,
            prompt,
            max_length=max_length,
        )
        prompt_mixtures: list[tuple[int, tuple[int, ...], tuple[float, ...]]] = []
        if src_spans is not None and tgt_spans is not None and src_spans and tgt_spans:
            src_signatures = None
            tgt_signatures = None
            if use_prediction_score:
                src_signatures = _collect_prompt_prediction_signatures(
                    src_model,
                    src_tokenizer,
                    src_text,
                    max_length=max_length,
                    device=device,
                    topk=prediction_topk,
                )
                tgt_signatures = _collect_prompt_prediction_signatures(
                    tgt_model,
                    tgt_tokenizer,
                    tgt_text,
                    max_length=max_length,
                    device=device,
                    topk=prediction_topk,
                )
            anchor_pairs = _align_token_spans_monotone(src_spans, tgt_spans)
            anchor_map: dict[int, list[int]] = {}
            for src_pos, tgt_pos in anchor_pairs:
                anchor_map.setdefault(int(src_pos), []).append(int(tgt_pos))
            tgt_pos_to_idx = {int(token_pos): idx for idx, (token_pos, _, _) in enumerate(tgt_spans)}
            for src_token_pos, src_start, src_end in src_spans:
                if use_prediction_score and src_signatures is not None and int(src_token_pos) >= len(src_signatures):
                    continue
                anchor_positions = anchor_map.get(int(src_token_pos))
                if anchor_positions:
                    anchor_center = int(round(sum(anchor_positions) / float(len(anchor_positions))))
                else:
                    src_mid = 0.5 * float(src_start + src_end)
                    anchor_center = min(
                        tgt_spans,
                        key=lambda span: abs(0.5 * float(span[1] + span[2]) - src_mid),
                    )[0]
                anchor_idx = tgt_pos_to_idx.get(int(anchor_center), 0)
                candidate_indices: set[int] = {anchor_idx}
                for tgt_idx, (_, tgt_start, tgt_end) in enumerate(tgt_spans):
                    overlap = max(0, min(src_end, tgt_end) - max(src_start, tgt_start))
                    if overlap > 0:
                        candidate_indices.add(tgt_idx)
                for delta in (-2, -1, 1, 2):
                    candidate_idx = anchor_idx + delta
                    if 0 <= candidate_idx < len(tgt_spans):
                        candidate_indices.add(candidate_idx)
                scored_candidates: list[tuple[float, int]] = []
                for tgt_idx in sorted(candidate_indices):
                    tgt_span = tgt_spans[tgt_idx]
                    tgt_pos = int(tgt_span[0])
                    if use_prediction_score and tgt_signatures is not None and tgt_pos >= len(tgt_signatures):
                        continue
                    ctx_score = _contextual_alignment_score(prompt, (src_token_pos, src_start, src_end), tgt_span)
                    pred_score = 0.0
                    if use_prediction_score and src_signatures is not None and tgt_signatures is not None:
                        pred_score = _prediction_token_overlap_score(
                            src_signatures[int(src_token_pos)][0],
                            src_signatures[int(src_token_pos)][1],
                            tgt_signatures[tgt_pos][0],
                            tgt_signatures[tgt_pos][1],
                        )
                    score = ctx_score + pred_weight * pred_score
                    scored_candidates.append((score, tgt_idx))
                if not scored_candidates:
                    continue
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                kept = scored_candidates[:max_targets_per_source]
                score_tensor = torch.tensor([score for score, _ in kept], dtype=torch.float32)
                probs = torch.softmax(score_tensor / temperature, dim=0)
                tgt_positions = tuple(int(tgt_spans[tgt_idx][0]) for _, tgt_idx in kept)
                tgt_weights = tuple(float(weight) for weight in probs.tolist())
                prompt_mixtures.append((int(src_token_pos), tgt_positions, tgt_weights))
        if not prompt_mixtures:
            src_len = _prompt_valid_length_for_text(src_tokenizer, src_text, max_length=max_length)
            tgt_len = _prompt_valid_length_for_text(tgt_tokenizer, tgt_text, max_length=max_length)
            prompt_mixtures = [
                (int(pos), (int(pos),), (1.0,))
                for pos in range(min(src_len, tgt_len))
            ]
        mixtures.append(prompt_mixtures)
    return mixtures


def _normalized_entropy_confidence(prob_vector: torch.Tensor) -> float:
    probs = prob_vector.float().view(-1)
    if probs.numel() <= 1:
        return 1.0
    probs = probs / probs.sum().clamp_min(1e-8)
    entropy = float((-(probs * probs.clamp_min(1e-30).log()).sum()).item())
    max_entropy = math.log(float(probs.numel()))
    if max_entropy <= 1e-8:
        return 1.0
    return max(0.0, 1.0 - entropy / max_entropy)


def collect_alignment_confidence_weights(
    aligned_position_mixtures: Sequence[Sequence[tuple[int, Sequence[int], Sequence[float]]]],
) -> torch.Tensor:
    weights: list[float] = []
    for prompt_mixtures in aligned_position_mixtures:
        for _, _, tgt_weights in prompt_mixtures:
            probs = torch.tensor(list(tgt_weights), dtype=torch.float32)
            concentration = _normalized_entropy_confidence(probs)
            max_prob = float(probs.max().item()) if probs.numel() > 0 else 0.0
            weights.append(0.5 * concentration + 0.5 * max_prob)
    if not weights:
        raise ValueError("Could not build alignment confidence weights from empty mixtures")
    out = torch.tensor(weights, dtype=torch.float32)
    return out / out.mean().clamp_min(1e-8)


def collect_prediction_confidence_weights(
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    if teacher_log_probs.ndim != 2:
        raise ValueError(f"teacher_log_probs must be [samples, topk], got {tuple(teacher_log_probs.shape)}")
    probs = torch.softmax(teacher_log_probs.float(), dim=-1)
    confidences = []
    for row in probs:
        concentration = _normalized_entropy_confidence(row)
        max_prob = float(row.max().item()) if row.numel() > 0 else 0.0
        confidences.append(0.5 * concentration + 0.5 * max_prob)
    out = torch.tensor(confidences, dtype=torch.float32)
    return out / out.mean().clamp_min(1e-8)


def _align_score_matrix_monotone(
    score_matrix: torch.Tensor,
    *,
    skip_penalty: float = 0.2,
) -> list[tuple[int, int]]:
    if score_matrix.ndim != 2:
        raise ValueError(f"score_matrix must be 2-D, got shape {tuple(score_matrix.shape)}")
    src_count, tgt_count = score_matrix.shape
    if src_count <= 0 or tgt_count <= 0:
        return []
    penalty = float(skip_penalty)
    dp = torch.empty(src_count + 1, tgt_count + 1, dtype=torch.float32)
    back = torch.zeros(src_count + 1, tgt_count + 1, dtype=torch.int64)
    dp[0, 0] = 0.0
    for src_idx in range(1, src_count + 1):
        dp[src_idx, 0] = dp[src_idx - 1, 0] - penalty
        back[src_idx, 0] = 1
    for tgt_idx in range(1, tgt_count + 1):
        dp[0, tgt_idx] = dp[0, tgt_idx - 1] - penalty
        back[0, tgt_idx] = 2
    for src_idx in range(1, src_count + 1):
        for tgt_idx in range(1, tgt_count + 1):
            match_score = dp[src_idx - 1, tgt_idx - 1] + float(score_matrix[src_idx - 1, tgt_idx - 1].item())
            skip_src_score = dp[src_idx - 1, tgt_idx] - penalty
            skip_tgt_score = dp[src_idx, tgt_idx - 1] - penalty
            best_score = match_score
            best_move = 0
            if skip_src_score > best_score:
                best_score = skip_src_score
                best_move = 1
            if skip_tgt_score > best_score:
                best_score = skip_tgt_score
                best_move = 2
            dp[src_idx, tgt_idx] = best_score
            back[src_idx, tgt_idx] = best_move

    pairs: list[tuple[int, int]] = []
    src_idx = src_count
    tgt_idx = tgt_count
    while src_idx > 0 or tgt_idx > 0:
        move = int(back[src_idx, tgt_idx].item())
        if move == 0:
            pairs.append((src_idx - 1, tgt_idx - 1))
            src_idx -= 1
            tgt_idx -= 1
        elif move == 1:
            src_idx -= 1
        else:
            tgt_idx -= 1
    pairs.reverse()
    return pairs


def collect_dynamic_program_prompt_position_pairs(
    src_model: AutoModelForCausalLM,
    src_tokenizer: AutoTokenizer,
    tgt_model: AutoModelForCausalLM,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: str,
    source_reasoning_mode: str = "plain",
    source_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_use_chat_template: bool = False,
    target_enable_thinking: bool | None = None,
    prediction_topk: int = 8,
    prediction_weight: float = 1.5,
    skip_penalty: float = 0.2,
    max_target_offset: int = 8,
) -> list[list[tuple[int, int]]]:
    del batch_size  # Alignment is prompt-local.
    all_pairs: list[list[tuple[int, int]]] = []
    pred_weight = float(prediction_weight)
    for prompt in prompts:
        src_text = _prepare_prompt_text(
            prompt,
            reasoning_mode=source_reasoning_mode,
            tokenizer=src_tokenizer,
            use_chat_template=source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        tgt_text = _format_prompt_for_tokenizer(
            tgt_tokenizer,
            prompt,
            use_chat_template=target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        src_spans = _collect_prompt_content_token_spans(
            src_tokenizer,
            src_text,
            prompt,
            max_length=max_length,
        )
        tgt_spans = _collect_prompt_content_token_spans(
            tgt_tokenizer,
            tgt_text,
            prompt,
            max_length=max_length,
        )
        prompt_pairs: list[tuple[int, int]] = []
        if src_spans is not None and tgt_spans is not None and src_spans and tgt_spans:
            src_signatures = _collect_prompt_prediction_signatures(
                src_model,
                src_tokenizer,
                src_text,
                max_length=max_length,
                device=device,
                topk=prediction_topk,
            )
            tgt_signatures = _collect_prompt_prediction_signatures(
                tgt_model,
                tgt_tokenizer,
                tgt_text,
                max_length=max_length,
                device=device,
                topk=prediction_topk,
            )
            anchor_pairs = _align_token_spans_monotone(src_spans, tgt_spans)
            anchor_map: dict[int, list[int]] = {}
            tgt_pos_to_idx = {int(token_pos): idx for idx, (token_pos, _, _) in enumerate(tgt_spans)}
            for src_pos, tgt_pos in anchor_pairs:
                tgt_idx = tgt_pos_to_idx.get(int(tgt_pos))
                if tgt_idx is not None:
                    anchor_map.setdefault(int(src_pos), []).append(int(tgt_idx))
            score_matrix = torch.empty(len(src_spans), len(tgt_spans), dtype=torch.float32)
            for src_idx, src_span in enumerate(src_spans):
                src_pos = int(src_span[0])
                if src_pos >= len(src_signatures):
                    score_matrix[src_idx].fill_(-1e4)
                    continue
                anchor_indices = anchor_map.get(src_pos)
                if anchor_indices:
                    anchor_idx = int(round(sum(anchor_indices) / float(len(anchor_indices))))
                else:
                    src_mid = 0.5 * float(src_span[1] + src_span[2])
                    anchor_idx = min(
                        range(len(tgt_spans)),
                        key=lambda idx: abs(0.5 * float(tgt_spans[idx][1] + tgt_spans[idx][2]) - src_mid),
                    )
                candidate_indices: set[int] = {
                    tgt_idx
                    for tgt_idx, (_, tgt_start, tgt_end) in enumerate(tgt_spans)
                    if max(0, min(src_span[2], tgt_end) - max(src_span[1], tgt_start)) > 0
                }
                for delta in range(-max_target_offset, max_target_offset + 1):
                    candidate_idx = anchor_idx + delta
                    if 0 <= candidate_idx < len(tgt_spans):
                        candidate_indices.add(candidate_idx)
                score_matrix[src_idx].fill_(-1e4)
                for tgt_idx in sorted(candidate_indices):
                    tgt_span = tgt_spans[tgt_idx]
                    tgt_pos = int(tgt_span[0])
                    if tgt_pos >= len(tgt_signatures):
                        score_matrix[src_idx, tgt_idx] = -1e4
                        continue
                    ctx_score = _contextual_alignment_score(prompt, src_span, tgt_span)
                    pred_score = _prediction_token_overlap_score(
                        src_signatures[src_pos][0],
                        src_signatures[src_pos][1],
                        tgt_signatures[tgt_pos][0],
                        tgt_signatures[tgt_pos][1],
                    )
                    score_matrix[src_idx, tgt_idx] = float(ctx_score + pred_weight * pred_score)
            index_pairs = _align_score_matrix_monotone(score_matrix, skip_penalty=skip_penalty)
            prompt_pairs = [
                (int(src_spans[src_idx][0]), int(tgt_spans[tgt_idx][0]))
                for src_idx, tgt_idx in index_pairs
                if 0 <= src_idx < len(src_spans) and 0 <= tgt_idx < len(tgt_spans)
            ]
        if not prompt_pairs:
            src_len = _prompt_valid_length_for_text(src_tokenizer, src_text, max_length=max_length)
            tgt_len = _prompt_valid_length_for_text(tgt_tokenizer, tgt_text, max_length=max_length)
            prompt_pairs = [(int(pos), int(pos)) for pos in range(min(src_len, tgt_len))]
        all_pairs.append(prompt_pairs)
    return all_pairs


def collect_aligned_prompt_position_pairs(
    src_tokenizer: AutoTokenizer,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    source_reasoning_mode: str = "plain",
    source_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_use_chat_template: bool = False,
    target_enable_thinking: bool | None = None,
) -> list[list[tuple[int, int]]]:
    del batch_size  # Alignment is prompt-local and cheap enough to build one prompt at a time.
    position_pairs: list[list[tuple[int, int]]] = []
    for prompt in prompts:
        src_text = _prepare_prompt_text(
            prompt,
            reasoning_mode=source_reasoning_mode,
            tokenizer=src_tokenizer,
            use_chat_template=source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        tgt_text = _format_prompt_for_tokenizer(
            tgt_tokenizer,
            prompt,
            use_chat_template=target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        src_spans = _collect_prompt_content_token_spans(
            src_tokenizer,
            src_text,
            prompt,
            max_length=max_length,
        )
        tgt_spans = _collect_prompt_content_token_spans(
            tgt_tokenizer,
            tgt_text,
            prompt,
            max_length=max_length,
        )
        pairs: list[tuple[int, int]] = []
        if src_spans is not None and tgt_spans is not None:
            pairs = _align_token_spans_monotone(src_spans, tgt_spans)
        if not pairs:
            src_len = _prompt_valid_length_for_text(src_tokenizer, src_text, max_length=max_length)
            tgt_len = _prompt_valid_length_for_text(tgt_tokenizer, tgt_text, max_length=max_length)
            pairs = [(pos, pos) for pos in range(min(src_len, tgt_len))]
        position_pairs.append(pairs)
    return position_pairs


def collect_byte_aligned_prompt_position_pairs(
    src_tokenizer: AutoTokenizer,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    source_reasoning_mode: str = "plain",
    source_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_use_chat_template: bool = False,
    target_enable_thinking: bool | None = None,
) -> list[list[tuple[int, int]]]:
    del batch_size  # Alignment is prompt-local and cheap enough to build one prompt at a time.
    position_pairs: list[list[tuple[int, int]]] = []
    for prompt in prompts:
        src_text = _prepare_prompt_text(
            prompt,
            reasoning_mode=source_reasoning_mode,
            tokenizer=src_tokenizer,
            use_chat_template=source_use_chat_template,
            enable_thinking=source_enable_thinking,
        )
        tgt_text = _format_prompt_for_tokenizer(
            tgt_tokenizer,
            prompt,
            use_chat_template=target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        src_spans = _collect_prompt_content_token_byte_spans(
            src_tokenizer,
            src_text,
            prompt,
            max_length=max_length,
        )
        tgt_spans = _collect_prompt_content_token_byte_spans(
            tgt_tokenizer,
            tgt_text,
            prompt,
            max_length=max_length,
        )
        pairs: list[tuple[int, int]] = []
        if src_spans is not None and tgt_spans is not None:
            pairs = _align_token_spans_byte_dominant(src_spans, tgt_spans)
        if not pairs:
            src_len = _prompt_valid_length_for_text(src_tokenizer, src_text, max_length=max_length)
            tgt_len = _prompt_valid_length_for_text(tgt_tokenizer, tgt_text, max_length=max_length)
            pairs = [(pos, pos) for pos in range(min(src_len, tgt_len))]
        position_pairs.append(pairs)
    return position_pairs


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
    source_use_chat_template: bool = False,
    target_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_enable_thinking: bool | None = None,
    aligned_position_pairs: Sequence[Sequence[tuple[int, int]]] | None = None,
    aligned_position_mixtures: Sequence[Sequence[tuple[int, Sequence[int], Sequence[float]]]] | None = None,
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

    if aligned_position_pairs is not None and aligned_position_mixtures is not None:
        raise ValueError("Only one of aligned_position_pairs or aligned_position_mixtures may be provided")

    src_per_layer_K: list[list[torch.Tensor]] = []
    src_per_layer_V: list[list[torch.Tensor]] = []
    tgt_per_layer_K: list[list[torch.Tensor]] = []
    tgt_per_layer_V: list[list[torch.Tensor]] = []

    prompt_offset = 0
    for batch in batched(prompts, batch_size):
        src_batch = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=source_reasoning_mode,
                tokenizer=src_tokenizer,
                use_chat_template=source_use_chat_template,
                enable_thinking=source_enable_thinking,
            )
            for prompt in batch
        ]
        tgt_batch = [
            _format_prompt_for_tokenizer(
                tgt_tokenizer,
                prompt,
                use_chat_template=target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            for prompt in batch
        ]
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
            prompt_idx = prompt_offset + batch_idx
            if aligned_position_mixtures is not None:
                position_mixtures = list(aligned_position_mixtures[prompt_idx])
            elif aligned_position_pairs is None:
                valid_len = min(
                    int(src_mask[batch_idx].sum().item()),
                    int(tgt_mask[batch_idx].sum().item()),
                )
                position_pairs = [(pos, pos) for pos in range(valid_len)]
            else:
                position_pairs = list(aligned_position_pairs[prompt_idx])
            if aligned_position_mixtures is not None and not position_mixtures:
                continue
            if aligned_position_mixtures is None and not position_pairs:
                continue
            src_valid_len = int(src_mask[batch_idx].sum().item())
            tgt_valid_len = int(tgt_mask[batch_idx].sum().item())
            if aligned_position_mixtures is not None:
                for src_pos, tgt_positions, tgt_weights in position_mixtures:
                    if src_pos >= src_valid_len:
                        continue
                    valid_targets = [
                        (int(tgt_pos), float(weight))
                        for tgt_pos, weight in zip(tgt_positions, tgt_weights)
                        if int(tgt_pos) < tgt_valid_len and float(weight) > 0.0
                    ]
                    if not valid_targets:
                        continue
                    weight_sum = sum(weight for _, weight in valid_targets)
                    if weight_sum <= 0.0:
                        continue
                    normalized_targets = [(pos, weight / weight_sum) for pos, weight in valid_targets]
                    for layer_idx, (K, V) in enumerate(src_pkv_cpu):
                        src_per_layer_K[layer_idx].append(
                            K[batch_idx : batch_idx + 1, :, src_pos : src_pos + 1, :]
                        )
                        src_per_layer_V[layer_idx].append(
                            V[batch_idx : batch_idx + 1, :, src_pos : src_pos + 1, :]
                        )
                    for layer_idx, (K, V) in enumerate(tgt_pkv_cpu):
                        weighted_k = None
                        weighted_v = None
                        for tgt_pos, weight in normalized_targets:
                            k_sample = K[batch_idx : batch_idx + 1, :, tgt_pos : tgt_pos + 1, :] * float(weight)
                            v_sample = V[batch_idx : batch_idx + 1, :, tgt_pos : tgt_pos + 1, :] * float(weight)
                            weighted_k = k_sample if weighted_k is None else weighted_k + k_sample
                            weighted_v = v_sample if weighted_v is None else weighted_v + v_sample
                        assert weighted_k is not None and weighted_v is not None
                        tgt_per_layer_K[layer_idx].append(weighted_k)
                        tgt_per_layer_V[layer_idx].append(weighted_v)
            else:
                for src_pos, tgt_pos in position_pairs:
                    if src_pos >= src_valid_len or tgt_pos >= tgt_valid_len:
                        continue
                    for layer_idx, (K, V) in enumerate(src_pkv_cpu):
                        src_per_layer_K[layer_idx].append(
                            K[batch_idx : batch_idx + 1, :, src_pos : src_pos + 1, :]
                        )
                        src_per_layer_V[layer_idx].append(
                            V[batch_idx : batch_idx + 1, :, src_pos : src_pos + 1, :]
                        )
                    for layer_idx, (K, V) in enumerate(tgt_pkv_cpu):
                        tgt_per_layer_K[layer_idx].append(
                            K[batch_idx : batch_idx + 1, :, tgt_pos : tgt_pos + 1, :]
                        )
                        tgt_per_layer_V[layer_idx].append(
                            V[batch_idx : batch_idx + 1, :, tgt_pos : tgt_pos + 1, :]
                        )
        prompt_offset += len(batch)

    src_kvs = [
        (torch.cat(src_per_layer_K[l], dim=0), torch.cat(src_per_layer_V[l], dim=0))
        for l in range(len(src_per_layer_K))
    ]
    tgt_kvs = [
        (torch.cat(tgt_per_layer_K[l], dim=0), torch.cat(tgt_per_layer_V[l], dim=0))
        for l in range(len(tgt_per_layer_K))
    ]
    return src_kvs, tgt_kvs


def collect_aligned_prompt_valid_lengths(
    src_tokenizer: AutoTokenizer,
    tgt_tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_length: int,
    batch_size: int,
    source_reasoning_mode: str = "plain",
    source_use_chat_template: bool = False,
    source_enable_thinking: bool | None = None,
    target_use_chat_template: bool = False,
    target_enable_thinking: bool | None = None,
) -> list[int]:
    lengths: list[int] = []
    for batch in batched(prompts, batch_size):
        src_batch = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=source_reasoning_mode,
                tokenizer=src_tokenizer,
                use_chat_template=source_use_chat_template,
                enable_thinking=source_enable_thinking,
            )
            for prompt in batch
        ]
        tgt_batch = [
            _format_prompt_for_tokenizer(
                tgt_tokenizer,
                prompt,
                use_chat_template=target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            for prompt in batch
        ]
        src_enc = src_tokenizer(
            src_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tgt_enc = tgt_tokenizer(
            tgt_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        src_mask = src_enc["attention_mask"]
        tgt_mask = tgt_enc["attention_mask"]
        for batch_idx in range(src_mask.shape[0]):
            lengths.append(
                min(
                    int(src_mask[batch_idx].sum().item()),
                    int(tgt_mask[batch_idx].sum().item()),
                )
            )
    return lengths


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
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
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
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
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


@torch.no_grad()
def collect_group_attention_template_bank(
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
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
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
    layer_templates: list[list[torch.Tensor]] | None = None
    used_prompts = 0
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
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
            raise ValueError("Model did not return attentions while building grouped template bank")
        if layer_templates is None:
            layer_templates = [[] for _ in range(len(attentions))]
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
                layer_templates[layer_idx].append(grouped)
            used_prompts += 1
    if layer_templates is None or used_prompts == 0:
        raise ValueError("Could not build grouped attention template bank from calibration prompts")
    return [torch.stack(templates, dim=0).cpu() for templates in layer_templates]


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
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
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
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
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


def _model_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not locate transformer layers for QK template extraction")


def _resample_signed_profile(profile: torch.Tensor, bins: int) -> torch.Tensor:
    if bins <= 0:
        raise ValueError("bins must be positive")
    values = profile.float().view(1, 1, -1)
    if values.shape[-1] == bins:
        return values.view(-1)
    return F.interpolate(values, size=bins, mode="linear", align_corners=False).view(-1)


@torch.no_grad()
def collect_group_qk_templates(
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
    reasoning_mode: str = "plain",
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if kv_heads % group_count != 0:
        raise ValueError(f"kv_heads={kv_heads} must be divisible by group_count={group_count}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _model_layers(model)
    group_heads = kv_heads // group_count
    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states while building grouped QK templates")
        pkv = _normalize_past_key_values(out.past_key_values)
        if layer_sums is None:
            layer_sums = [torch.zeros(group_count, bins, dtype=torch.float32) for _ in range(len(pkv))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            valid_len = int(mask_cpu[batch_idx].sum().item())
            if valid_len <= 1:
                continue
            for layer_idx, layer_cache in enumerate(pkv):
                attn_mod = layers[layer_idx].self_attn
                q_proj = getattr(attn_mod, "q_proj", None)
                if q_proj is None:
                    raise ValueError("self_attn.q_proj is required for grouped QK templates")
                hidden = hidden_states[layer_idx][batch_idx, valid_len - 1 : valid_len].to(device)
                q_flat = q_proj(hidden).squeeze(0).squeeze(0).detach().to("cpu", dtype=torch.float32)
                keys = layer_cache[0][batch_idx, :, : valid_len - 1, :].detach().to("cpu", dtype=torch.float32)
                num_q_heads = q_flat.numel() // keys.shape[-1]
                if num_q_heads % kv_heads != 0:
                    raise ValueError(
                        f"num_q_heads={num_q_heads} must be divisible by kv_heads={kv_heads} for grouped QK templates"
                    )
                q = q_flat.view(num_q_heads, keys.shape[-1])
                q_per_kv = num_q_heads // kv_heads
                group_templates = []
                for group_idx in range(group_count):
                    start = group_idx * group_heads
                    stop = start + group_heads
                    head_profiles = []
                    for kv_head in range(start, stop):
                        q_start = kv_head * q_per_kv
                        q_stop = q_start + q_per_kv
                        logits = torch.einsum("hd,td->ht", q[q_start:q_stop], keys[kv_head]) / math.sqrt(keys.shape[-1])
                        logit_profile = logits.mean(dim=0)
                        logit_profile = (logit_profile - logit_profile.mean()) / logit_profile.std(unbiased=False).clamp_min(1e-6)
                        head_profiles.append(_resample_signed_profile(logit_profile, bins))
                    group_templates.append(torch.stack(head_profiles, dim=0).mean(dim=0))
                layer_sums[layer_idx] += torch.stack(group_templates, dim=0)
            used_prompts += 1
    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build grouped QK templates from calibration prompts")
    return [(templates / float(used_prompts)).cpu() for templates in layer_sums]


@torch.no_grad()
def collect_group_qk_template_bank(
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
    reasoning_mode: str = "plain",
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if kv_heads % group_count != 0:
        raise ValueError(f"kv_heads={kv_heads} must be divisible by group_count={group_count}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _model_layers(model)
    group_heads = kv_heads // group_count
    layer_bank: list[list[torch.Tensor]] | None = None
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states while building grouped QK template bank")
        pkv = _normalize_past_key_values(out.past_key_values)
        if layer_bank is None:
            layer_bank = [[] for _ in range(len(pkv))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            valid_len = int(mask_cpu[batch_idx].sum().item())
            if valid_len <= 1:
                continue
            for layer_idx, layer_cache in enumerate(pkv):
                attn_mod = layers[layer_idx].self_attn
                q_proj = getattr(attn_mod, "q_proj", None)
                if q_proj is None:
                    raise ValueError("self_attn.q_proj is required for grouped QK template bank")
                hidden = hidden_states[layer_idx][batch_idx, valid_len - 1 : valid_len].to(device)
                q_flat = q_proj(hidden).squeeze(0).squeeze(0).detach().to("cpu", dtype=torch.float32)
                keys = layer_cache[0][batch_idx, :, : valid_len - 1, :].detach().to("cpu", dtype=torch.float32)
                num_q_heads = q_flat.numel() // keys.shape[-1]
                if num_q_heads % kv_heads != 0:
                    raise ValueError(
                        f"num_q_heads={num_q_heads} must be divisible by kv_heads={kv_heads} for grouped QK template bank"
                    )
                q = q_flat.view(num_q_heads, keys.shape[-1])
                q_per_kv = num_q_heads // kv_heads
                group_templates = []
                for group_idx in range(group_count):
                    start = group_idx * group_heads
                    stop = start + group_heads
                    head_profiles = []
                    for kv_head in range(start, stop):
                        q_start = kv_head * q_per_kv
                        q_stop = q_start + q_per_kv
                        logits = torch.einsum("hd,td->ht", q[q_start:q_stop], keys[kv_head]) / math.sqrt(keys.shape[-1])
                        logit_profile = logits.mean(dim=0)
                        logit_profile = (logit_profile - logit_profile.mean()) / logit_profile.std(unbiased=False).clamp_min(1e-6)
                        head_profiles.append(_resample_signed_profile(logit_profile, bins))
                    group_templates.append(torch.stack(head_profiles, dim=0).mean(dim=0))
                layer_bank[layer_idx].append(torch.stack(group_templates, dim=0))
    if layer_bank is None or not layer_bank:
        raise ValueError("Could not build grouped QK template bank from calibration prompts")
    return [torch.stack(templates, dim=0).cpu() for templates in layer_bank]


@torch.no_grad()
def collect_aligned_qk_position_weights(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    aligned_lengths: Sequence[int],
    max_length: int,
    batch_size: int,
    device: str,
    kv_heads: int,
    reasoning_mode: str = "plain",
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
    floor: float = 0.25,
) -> list[torch.Tensor]:
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if len(aligned_lengths) != len(prompts):
        raise ValueError(
            f"aligned_lengths must match prompts, got {len(aligned_lengths)} lengths for {len(prompts)} prompts"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _model_layers(model)
    layer_weights: list[list[torch.Tensor]] | None = None
    prompt_offset = 0
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states while building aligned QK position weights")
        pkv = _normalize_past_key_values(out.past_key_values)
        if layer_weights is None:
            layer_weights = [[] for _ in range(len(pkv))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            prompt_idx = prompt_offset + batch_idx
            valid_len = int(mask_cpu[batch_idx].sum().item())
            aligned_len = min(valid_len, int(aligned_lengths[prompt_idx]))
            if aligned_len <= 0:
                continue
            for layer_idx, layer_cache in enumerate(pkv):
                if aligned_len <= 1:
                    layer_weights[layer_idx].append(torch.ones(aligned_len, dtype=torch.float32))
                    continue
                attn_mod = layers[layer_idx].self_attn
                q_proj = getattr(attn_mod, "q_proj", None)
                if q_proj is None:
                    raise ValueError("self_attn.q_proj is required for aligned QK position weights")
                hidden = hidden_states[layer_idx][batch_idx, valid_len - 1 : valid_len].to(device)
                q_flat = q_proj(hidden).squeeze(0).squeeze(0).detach().to("cpu", dtype=torch.float32)
                keys = layer_cache[0][batch_idx, :, : aligned_len - 1, :].detach().to("cpu", dtype=torch.float32)
                num_q_heads = q_flat.numel() // keys.shape[-1]
                if num_q_heads % kv_heads != 0:
                    raise ValueError(
                        f"num_q_heads={num_q_heads} must be divisible by kv_heads={kv_heads} for aligned QK weights"
                    )
                q = q_flat.view(num_q_heads, keys.shape[-1])
                q_per_kv = num_q_heads // kv_heads
                per_head_probs = []
                for kv_head in range(kv_heads):
                    q_start = kv_head * q_per_kv
                    q_stop = q_start + q_per_kv
                    logits = torch.einsum("hd,td->ht", q[q_start:q_stop], keys[kv_head]) / math.sqrt(keys.shape[-1])
                    probs = torch.softmax(logits, dim=-1).mean(dim=0)
                    per_head_probs.append(probs)
                position_probs = torch.stack(per_head_probs, dim=0).mean(dim=0)
                scaled = torch.ones(aligned_len, dtype=torch.float32)
                scaled[:-1] = position_probs * float(max(aligned_len - 1, 1))
                scaled = scaled / scaled.mean().clamp_min(1e-8)
                weights = floor + (1.0 - floor) * scaled
                layer_weights[layer_idx].append(weights / weights.mean().clamp_min(1e-8))
        prompt_offset += len(batch)
    if layer_weights is None or not layer_weights:
        raise ValueError("Could not build aligned QK position weights from calibration prompts")
    return [torch.cat(weights, dim=0).cpu() for weights in layer_weights]


@torch.no_grad()
def collect_aligned_query_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    aligned_lengths: Sequence[int],
    max_length: int,
    batch_size: int,
    device: str,
    kv_heads: int,
    head_dim: int,
    reasoning_mode: str = "plain",
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
    aligned_position_pairs: Sequence[Sequence[tuple[int, int]]] | None = None,
    aligned_position_mixtures: Sequence[Sequence[tuple[int, Sequence[int], Sequence[float]]]] | None = None,
) -> list[torch.Tensor]:
    if aligned_position_pairs is not None and aligned_position_mixtures is not None:
        raise ValueError("Only one of aligned_position_pairs or aligned_position_mixtures may be provided")
    if aligned_position_pairs is not None:
        if len(aligned_position_pairs) != len(prompts):
            raise ValueError(
                f"aligned_position_pairs must match prompts, got {len(aligned_position_pairs)} prompt pair sets for {len(prompts)} prompts"
            )
    elif aligned_position_mixtures is not None:
        if len(aligned_position_mixtures) != len(prompts):
            raise ValueError(
                f"aligned_position_mixtures must match prompts, got {len(aligned_position_mixtures)} prompt mixture sets for {len(prompts)} prompts"
            )
    elif len(aligned_lengths) != len(prompts):
        raise ValueError(
            f"aligned_lengths must match prompts, got {len(aligned_lengths)} lengths for {len(prompts)} prompts"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _model_layers(model)
    layer_features: list[list[torch.Tensor]] | None = None
    prompt_offset = 0
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states while building aligned query features")
        if layer_features is None:
            layer_features = [[] for _ in range(min(len(layers), len(hidden_states) - 1))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            prompt_idx = prompt_offset + batch_idx
            if aligned_position_mixtures is not None:
                position_mixtures = list(aligned_position_mixtures[prompt_idx])
            elif aligned_position_pairs is None:
                valid_len = min(int(mask_cpu[batch_idx].sum().item()), int(aligned_lengths[prompt_idx]))
                positions_cpu = torch.arange(valid_len, dtype=torch.long)
            else:
                positions_cpu = torch.tensor(
                    [tgt_pos for _, tgt_pos in aligned_position_pairs[prompt_idx]],
                    dtype=torch.long,
                )
            if aligned_position_mixtures is not None and not position_mixtures:
                continue
            if aligned_position_mixtures is None and positions_cpu.numel() <= 0:
                continue
            valid_len = int(mask_cpu[batch_idx].sum().item())
            for layer_idx in range(len(layer_features)):
                attn_mod = layers[layer_idx].self_attn
                q_proj = getattr(attn_mod, "q_proj", None)
                if q_proj is None:
                    raise ValueError("self_attn.q_proj is required for aligned query features")
                hidden = hidden_states[layer_idx][batch_idx, :valid_len].to(device)
                if aligned_position_mixtures is not None:
                    for _, tgt_positions, tgt_weights in position_mixtures:
                        valid_targets = [
                            (int(tgt_pos), float(weight))
                            for tgt_pos, weight in zip(tgt_positions, tgt_weights)
                            if int(tgt_pos) < valid_len and float(weight) > 0.0
                        ]
                        if not valid_targets:
                            continue
                        positions = torch.tensor([pos for pos, _ in valid_targets], dtype=torch.long, device=hidden.device)
                        weights = torch.tensor([weight for _, weight in valid_targets], dtype=torch.float32, device=hidden.device)
                        weights = weights / weights.sum().clamp_min(1e-8)
                        q_flat = q_proj(hidden.index_select(0, positions)).detach().to("cpu", dtype=torch.float32)
                        num_query_heads = q_flat.shape[-1] // head_dim
                        if num_query_heads % kv_heads != 0:
                            raise ValueError(
                                f"num_query_heads={num_query_heads} must be divisible by kv_heads={kv_heads} for aligned query features"
                            )
                        q = q_flat.view(q_flat.shape[0], num_query_heads, head_dim)
                        q_per_kv = num_query_heads // kv_heads
                        q = q.view(q_flat.shape[0], kv_heads, q_per_kv, head_dim).mean(dim=2)
                        q_weighted = (q * weights.to("cpu").view(-1, 1, 1)).sum(dim=0, keepdim=True)
                        layer_features[layer_idx].append(q_weighted.reshape(1, kv_heads * head_dim))
                else:
                    positions = positions_cpu.to(device=hidden.device)
                    q_flat = q_proj(hidden.index_select(0, positions)).detach().to("cpu", dtype=torch.float32)
                    sample_len = int(q_flat.shape[0])
                    num_query_heads = q_flat.shape[-1] // head_dim
                    if num_query_heads % kv_heads != 0:
                        raise ValueError(
                            f"num_query_heads={num_query_heads} must be divisible by kv_heads={kv_heads} for aligned query features"
                        )
                    q = q_flat.view(sample_len, num_query_heads, head_dim)
                    q_per_kv = num_query_heads // kv_heads
                    q = q.view(sample_len, kv_heads, q_per_kv, head_dim).mean(dim=2)
                    layer_features[layer_idx].append(q.reshape(sample_len, kv_heads * head_dim))
        prompt_offset += len(batch)
    if layer_features is None or not layer_features:
        raise ValueError("Could not build aligned query features from calibration prompts")
    return [torch.cat(features, dim=0).cpu() for features in layer_features]


@torch.no_grad()
def collect_aligned_prediction_teacher(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    aligned_lengths: Sequence[int],
    max_length: int,
    batch_size: int,
    device: str,
    topk: int = 8,
    reasoning_mode: str = "plain",
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
    aligned_position_pairs: Sequence[Sequence[tuple[int, int]]] | None = None,
    aligned_position_mixtures: Sequence[Sequence[tuple[int, Sequence[int], Sequence[float]]]] | None = None,
    include_next_token_targets: bool = False,
    next_token_target_weight: float = 0.5,
    span_likelihood_window: int = 0,
    span_likelihood_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if aligned_position_pairs is not None and aligned_position_mixtures is not None:
        raise ValueError("Only one of aligned_position_pairs or aligned_position_mixtures may be provided")
    if aligned_position_pairs is not None:
        if len(aligned_position_pairs) != len(prompts):
            raise ValueError(
                f"aligned_position_pairs must match prompts, got {len(aligned_position_pairs)} prompt pair sets for {len(prompts)} prompts"
            )
    elif aligned_position_mixtures is not None:
        if len(aligned_position_mixtures) != len(prompts):
            raise ValueError(
                f"aligned_position_mixtures must match prompts, got {len(aligned_position_mixtures)} prompt mixture sets for {len(prompts)} prompts"
            )
    elif len(aligned_lengths) != len(prompts):
        raise ValueError(
            f"aligned_lengths must match prompts, got {len(aligned_lengths)} lengths for {len(prompts)} prompts"
        )
    if topk <= 0:
        raise ValueError("topk must be positive")
    if next_token_target_weight < 0.0:
        raise ValueError("next_token_target_weight must be non-negative")
    if span_likelihood_window < 0:
        raise ValueError("span_likelihood_window must be non-negative")
    if span_likelihood_weight < 0.0:
        raise ValueError("span_likelihood_weight must be non-negative")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or getattr(output_embeddings, "weight", None) is None:
        raise ValueError("Model must expose output embeddings for prediction-teacher collection")
    output_weight = output_embeddings.weight.detach().to("cpu", dtype=torch.float32)

    sample_log_probs: list[torch.Tensor] = []
    sample_output_rows: list[torch.Tensor] = []

    def blend_sparse_likelihood_scores(
        candidate_ids: torch.Tensor,
        candidate_probs: torch.Tensor,
        base_probs: torch.Tensor,
        sparse_scores: dict[int, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sparse_scores or float(span_likelihood_weight) <= 0.0:
            return candidate_ids, candidate_probs
        blend = min(float(span_likelihood_weight), 1.0)
        total_score = sum(max(float(score), 0.0) for score in sparse_scores.values())
        if total_score <= 0.0:
            return candidate_ids, candidate_probs
        candidate_probs = candidate_probs * (1.0 - blend)
        for token_id, score in sparse_scores.items():
            if score <= 0.0:
                continue
            token_prob = blend * float(score) / total_score
            match = candidate_ids == int(token_id)
            if bool(match.any()):
                candidate_probs[match] += token_prob
            else:
                token_tensor = torch.tensor([int(token_id)], dtype=torch.long)
                candidate_ids = torch.cat([candidate_ids, token_tensor], dim=0)
                base_prob = base_probs[int(token_id)].view(1) * (1.0 - blend)
                candidate_probs = torch.cat(
                    [candidate_probs, base_prob + candidate_probs.new_tensor([token_prob])],
                    dim=0,
                )
        return candidate_ids, candidate_probs

    prompt_offset = 0
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, return_dict=True)
        logits = out.logits.detach().to("cpu", dtype=torch.float32)
        mask_cpu = enc["attention_mask"].to("cpu")
        input_ids_cpu = enc["input_ids"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            prompt_idx = prompt_offset + batch_idx
            if aligned_position_mixtures is not None:
                position_mixtures = list(aligned_position_mixtures[prompt_idx])
            elif aligned_position_pairs is None:
                valid_len = min(int(mask_cpu[batch_idx].sum().item()), int(aligned_lengths[prompt_idx]))
                positions = torch.arange(valid_len, dtype=torch.long)
            else:
                positions = torch.tensor(
                    [tgt_pos for _, tgt_pos in aligned_position_pairs[prompt_idx]],
                    dtype=torch.long,
                )
            if aligned_position_mixtures is not None and not position_mixtures:
                continue
            if aligned_position_mixtures is None and positions.numel() <= 0:
                continue
            prompt_logits = logits[batch_idx]
            if aligned_position_mixtures is not None:
                valid_len = int(mask_cpu[batch_idx].sum().item())
                prompt_probs = torch.softmax(prompt_logits[:valid_len], dim=-1)
                for _, tgt_positions, tgt_weights in position_mixtures:
                    valid_targets = [
                        (int(tgt_pos), float(weight))
                        for tgt_pos, weight in zip(tgt_positions, tgt_weights)
                        if int(tgt_pos) < valid_len and float(weight) > 0.0
                    ]
                    if not valid_targets:
                        continue
                    positions = torch.tensor([pos for pos, _ in valid_targets], dtype=torch.long)
                    weights = torch.tensor([weight for _, weight in valid_targets], dtype=torch.float32)
                    weights = weights / weights.sum().clamp_min(1e-8)
                    mixture_probs = (prompt_probs.index_select(0, positions) * weights.view(-1, 1)).sum(dim=0)
                    mixture_probs = mixture_probs / mixture_probs.sum().clamp_min(1e-8)
                    k = min(int(topk), int(mixture_probs.shape[-1]))
                    top_probs, top_ids = torch.topk(mixture_probs, k=k, dim=-1)
                    candidate_ids = top_ids.clone()
                    candidate_probs = top_probs.clone()
                    if include_next_token_targets and float(next_token_target_weight) > 0.0:
                        gold_mass: dict[int, float] = {}
                        for tgt_pos, weight in valid_targets:
                            next_pos = int(tgt_pos) + 1
                            if next_pos >= valid_len:
                                continue
                            token_id = int(input_ids_cpu[batch_idx, next_pos].item())
                            gold_mass[token_id] = gold_mass.get(token_id, 0.0) + float(weight)
                        for token_id, mass in gold_mass.items():
                            token_tensor = torch.tensor([token_id], dtype=torch.long)
                            match = candidate_ids == int(token_id)
                            if bool(match.any()):
                                candidate_probs[match] += float(next_token_target_weight) * float(mass)
                            else:
                                candidate_ids = torch.cat([candidate_ids, token_tensor], dim=0)
                                base_prob = mixture_probs[int(token_id)].view(1)
                                candidate_probs = torch.cat(
                                    [candidate_probs, base_prob + float(next_token_target_weight) * float(mass)],
                                    dim=0,
                                )
                    if int(span_likelihood_window) > 0 and float(span_likelihood_weight) > 0.0:
                        span_scores: dict[int, float] = {}
                        for tgt_pos, weight in valid_targets:
                            for offset in range(1, int(span_likelihood_window) + 1):
                                pred_pos = int(tgt_pos) + offset - 1
                                token_pos = int(tgt_pos) + offset
                                if pred_pos >= valid_len or token_pos >= valid_len:
                                    continue
                                token_id = int(input_ids_cpu[batch_idx, token_pos].item())
                                token_prob = float(prompt_probs[pred_pos, token_id].item())
                                # Later span tokens are useful but less directly aligned.
                                span_scores[token_id] = span_scores.get(token_id, 0.0) + (
                                    float(weight) * token_prob / float(offset)
                                )
                        candidate_ids, candidate_probs = blend_sparse_likelihood_scores(
                            candidate_ids,
                            candidate_probs,
                            mixture_probs,
                            span_scores,
                        )
                    top_candidate_probs, order = torch.topk(
                        candidate_probs,
                        k=min(int(topk), int(candidate_probs.numel())),
                        dim=-1,
                    )
                    top_candidate_ids = candidate_ids.index_select(0, order)
                    sample_log_probs.append(torch.log(top_candidate_probs.clamp_min(1e-30)).unsqueeze(0))
                    sample_output_rows.append(output_weight[top_candidate_ids].unsqueeze(0))
            else:
                if include_next_token_targets and float(next_token_target_weight) > 0.0:
                    valid_len = int(mask_cpu[batch_idx].sum().item())
                    for pos in positions.tolist():
                        pos = int(pos)
                        row_logits = prompt_logits[pos]
                        row_log_z = torch.logsumexp(row_logits, dim=-1)
                        k = min(int(topk), int(row_logits.shape[-1]))
                        top_logits, top_ids = torch.topk(row_logits, k=k, dim=-1)
                        candidate_ids = top_ids.clone()
                        candidate_probs = torch.exp(top_logits - row_log_z)
                        next_pos = pos + 1
                        if next_pos < valid_len:
                            token_id = int(input_ids_cpu[batch_idx, next_pos].item())
                            match = candidate_ids == int(token_id)
                            if bool(match.any()):
                                candidate_probs[match] += float(next_token_target_weight)
                            else:
                                token_tensor = torch.tensor([token_id], dtype=torch.long)
                                candidate_ids = torch.cat([candidate_ids, token_tensor], dim=0)
                                base_prob = torch.exp(row_logits[int(token_id)] - row_log_z).view(1)
                                candidate_probs = torch.cat(
                                    [candidate_probs, base_prob + float(next_token_target_weight)],
                                    dim=0,
                                )
                        top_candidate_probs, order = torch.topk(
                            candidate_probs,
                            k=min(int(topk), int(candidate_probs.numel())),
                            dim=-1,
                        )
                        top_candidate_ids = candidate_ids.index_select(0, order)
                        sample_log_probs.append(torch.log(top_candidate_probs.clamp_min(1e-30)).unsqueeze(0))
                        sample_output_rows.append(output_weight[top_candidate_ids].unsqueeze(0))
                else:
                    if int(span_likelihood_window) > 0 and float(span_likelihood_weight) > 0.0:
                        valid_len = int(mask_cpu[batch_idx].sum().item())
                        prompt_probs = torch.softmax(prompt_logits[:valid_len], dim=-1)
                        for pos in positions.tolist():
                            pos = int(pos)
                            if pos >= valid_len:
                                continue
                            row_probs = prompt_probs[pos]
                            k = min(int(topk), int(row_probs.shape[-1]))
                            top_probs, top_ids = torch.topk(row_probs, k=k, dim=-1)
                            span_scores: dict[int, float] = {}
                            for offset in range(1, int(span_likelihood_window) + 1):
                                pred_pos = pos + offset - 1
                                token_pos = pos + offset
                                if pred_pos >= valid_len or token_pos >= valid_len:
                                    continue
                                token_id = int(input_ids_cpu[batch_idx, token_pos].item())
                                token_prob = float(prompt_probs[pred_pos, token_id].item())
                                span_scores[token_id] = span_scores.get(token_id, 0.0) + token_prob / float(offset)
                            candidate_ids, candidate_probs = blend_sparse_likelihood_scores(
                                top_ids.clone(),
                                top_probs.clone(),
                                row_probs,
                                span_scores,
                            )
                            top_candidate_probs, order = torch.topk(
                                candidate_probs,
                                k=min(int(topk), int(candidate_probs.numel())),
                                dim=-1,
                            )
                            top_candidate_ids = candidate_ids.index_select(0, order)
                            sample_log_probs.append(torch.log(top_candidate_probs.clamp_min(1e-30)).unsqueeze(0))
                            sample_output_rows.append(output_weight[top_candidate_ids].unsqueeze(0))
                    else:
                        prompt_logits = prompt_logits.index_select(0, positions)
                        prompt_log_z = torch.logsumexp(prompt_logits, dim=-1, keepdim=True)
                        k = min(int(topk), int(prompt_logits.shape[-1]))
                        top_logits, top_ids = torch.topk(prompt_logits, k=k, dim=-1)
                        sample_log_probs.append(top_logits - prompt_log_z)
                        sample_output_rows.append(output_weight[top_ids])
        prompt_offset += len(batch)
    if not sample_log_probs:
        raise ValueError("Could not build aligned prediction teachers from calibration prompts")
    return torch.cat(sample_log_probs, dim=0).cpu(), torch.cat(sample_output_rows, dim=0).cpu()


@torch.no_grad()
def collect_group_qk_retrieval_templates(
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
    reasoning_mode: str = "plain",
    use_chat_template: bool = False,
    enable_thinking: bool | None = None,
) -> list[torch.Tensor]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if kv_heads % group_count != 0:
        raise ValueError(f"kv_heads={kv_heads} must be divisible by group_count={group_count}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = _model_layers(model)
    group_heads = kv_heads // group_count
    layer_sums: list[torch.Tensor] | None = None
    used_prompts = 0
    for batch in batched(prompts, batch_size):
        batch_text = [
            _prepare_prompt_text(
                prompt,
                reasoning_mode=reasoning_mode,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            for prompt in batch
        ]
        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, use_cache=True, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Model did not return hidden_states while building grouped QK retrieval templates")
        pkv = _normalize_past_key_values(out.past_key_values)
        if layer_sums is None:
            layer_sums = [torch.zeros(group_count, bins, dtype=torch.float32) for _ in range(len(pkv))]
        mask_cpu = enc["attention_mask"].to("cpu")
        for batch_idx in range(mask_cpu.shape[0]):
            valid_len = int(mask_cpu[batch_idx].sum().item())
            if valid_len <= 1:
                continue
            for layer_idx, layer_cache in enumerate(pkv):
                attn_mod = layers[layer_idx].self_attn
                q_proj = getattr(attn_mod, "q_proj", None)
                if q_proj is None:
                    raise ValueError("self_attn.q_proj is required for grouped QK retrieval templates")
                hidden = hidden_states[layer_idx][batch_idx, valid_len - 1 : valid_len].to(device)
                q_flat = q_proj(hidden).squeeze(0).squeeze(0).detach().to("cpu", dtype=torch.float32)
                keys = layer_cache[0][batch_idx, :, : valid_len - 1, :].detach().to("cpu", dtype=torch.float32)
                num_q_heads = q_flat.numel() // keys.shape[-1]
                if num_q_heads % kv_heads != 0:
                    raise ValueError(
                        f"num_q_heads={num_q_heads} must be divisible by kv_heads={kv_heads} for grouped QK retrieval templates"
                    )
                q = q_flat.view(num_q_heads, keys.shape[-1])
                q_per_kv = num_q_heads // kv_heads
                group_templates = []
                for group_idx in range(group_count):
                    start = group_idx * group_heads
                    stop = start + group_heads
                    head_profiles = []
                    for kv_head in range(start, stop):
                        q_start = kv_head * q_per_kv
                        q_stop = q_start + q_per_kv
                        logits = torch.einsum("hd,td->ht", q[q_start:q_stop], keys[kv_head]) / math.sqrt(keys.shape[-1])
                        retrieval_logits = logits.max(dim=0).values
                        weights = torch.softmax(retrieval_logits, dim=-1)
                        head_profiles.append(_resample_position_profile(weights, bins))
                    grouped = torch.stack(head_profiles, dim=0).mean(dim=0)
                    group_templates.append(grouped / grouped.sum().clamp_min(1e-8))
                layer_sums[layer_idx] += torch.stack(group_templates, dim=0)
            used_prompts += 1
    if layer_sums is None or used_prompts == 0:
        raise ValueError("Could not build grouped QK retrieval templates from calibration prompts")
    return [(templates / float(used_prompts)).cpu() for templates in layer_sums]


def main() -> None:
    args = parse_args()
    dtype = torch_dtype(args.dtype)
    source_enable_thinking = _optional_bool_from_arg(args.source_enable_thinking)
    target_enable_thinking = _optional_bool_from_arg(args.target_enable_thinking)

    prompt_records = load_prompt_records(args.calibration_file)
    prompts = [prompt for prompt, _ in prompt_records]
    prompt_example_ids = [ids for _, ids in prompt_records]
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

    aligned_position_pairs: list[list[tuple[int, int]]] | None = None
    aligned_position_mixtures: list[list[tuple[int, tuple[int, ...], tuple[float, ...]]]] | None = None
    if args.quantization_correction == "bridge_ridge_qk_spanalign_module_replace":
        print(
            "\nBuilding raw-prompt span-aligned source/target token pairs "
            "(content-only monotone overlap alignment)..."
        )
        aligned_position_pairs = collect_aligned_prompt_position_pairs(
            tok_s,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=source_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=target_enable_thinking,
        )
        total_pairs = sum(len(pairs) for pairs in aligned_position_pairs)
        print(
            "Built aligned token pairs: "
            f"prompts={len(aligned_position_pairs)}, total_pairs={total_pairs}, "
            f"mean_pairs_per_prompt={total_pairs / max(len(aligned_position_pairs), 1):.2f}"
        )
    elif args.quantization_correction == "bridge_ridge_qk_bytespan_module_replace":
        print(
            "\nBuilding raw-prompt byte-span-aligned source/target token pairs "
            "(content-only monotone byte-overlap alignment)..."
        )
        aligned_position_pairs = collect_byte_aligned_prompt_position_pairs(
            tok_s,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=source_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=target_enable_thinking,
        )
        total_pairs = sum(len(pairs) for pairs in aligned_position_pairs)
        span_position_pairs = collect_aligned_prompt_position_pairs(
            tok_s,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=source_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=target_enable_thinking,
        )
        changed_prompts = sum(
            int(byte_pairs != span_pairs)
            for byte_pairs, span_pairs in zip(aligned_position_pairs, span_position_pairs)
        )
        print(
            "Built byte-aligned token pairs: "
            f"prompts={len(aligned_position_pairs)}, total_pairs={total_pairs}, "
            f"mean_pairs_per_prompt={total_pairs / max(len(aligned_position_pairs), 1):.2f}, "
            f"changed_vs_spanalign_prompts={changed_prompts}"
        )
    elif args.quantization_correction == "bridge_ridge_qk_ctxalign_module_replace":
        print(
            "\nBuilding contextual source->target token mixtures "
            "(span-anchor plus local context-weighted remapping)..."
        )
        aligned_position_mixtures = collect_contextual_prompt_position_mixtures(
            tok_s,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=source_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=target_enable_thinking,
        )
        total_samples = sum(len(mixture) for mixture in aligned_position_mixtures)
        total_targets = sum(
            len(tgt_positions)
            for mixture in aligned_position_mixtures
            for _, tgt_positions, _ in mixture
        )
        print(
            "Built contextual token mixtures: "
            f"prompts={len(aligned_position_mixtures)}, samples={total_samples}, "
            f"mean_targets_per_sample={total_targets / max(total_samples, 1):.2f}"
        )
    elif args.quantization_correction in {
        "bridge_ridge_qk_dynalign_ctxonly_module_replace",
        "bridge_ridge_qk_dynalign_module_replace",
        "bridge_ridge_qk_dynalign_preserve_module_replace",
        "bridge_ridge_qk_dynalign_eigenspace_module_replace",
        "bridge_ridge_qk_dynalign_saliency_module_replace",
        "bridge_ridge_qk_dynalign_saliency_preserve_module_replace",
        "bridge_ridge_qk_dynalign_anchor_tail_module_replace",
        "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace",
        "bridge_ridge_qk_dynalign_routed_module_replace",
        "bridge_ridge_qk_dynalign_value_routed_module_replace",
        "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
        "bridge_ridge_qk_dynalign_dwakd_module_replace",
        "bridge_ridge_qk_dynalign_likelihood_module_replace",
        "bridge_ridge_qk_dynalign_spanalm_module_replace",
        "bridge_ridge_qk_dynalign_prefdist_module_replace",
        "bridge_ridge_qk_dynalign_dwainteract_module_replace",
        "bridge_ridge_qk_dynalign_interact_module_replace",
    }:
        ctxonly = args.quantization_correction == "bridge_ridge_qk_dynalign_ctxonly_module_replace"
        print(
            "\nBuilding dynamic source->target token mixtures "
            + (
                "(matched context-only null: span-anchor plus context remapping)..."
                if ctxonly
                else "(span-anchor plus context and output-overlap remapping)..."
            )
        )
        aligned_position_mixtures = collect_dynamic_prompt_position_mixtures(
            src,
            tok_s,
            tgt,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=source_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=target_enable_thinking,
            prediction_weight=0.0 if ctxonly else 1.5,
        )
        total_samples = sum(len(mixture) for mixture in aligned_position_mixtures)
        total_targets = sum(
            len(tgt_positions)
            for mixture in aligned_position_mixtures
            for _, tgt_positions, _ in mixture
        )
        mixture_label = (
            "Built dynamic context-only token mixtures: "
            if ctxonly
            else "Built dynamic token mixtures: "
        )
        print(
            mixture_label
            + f"prompts={len(aligned_position_mixtures)}, samples={total_samples}, "
            + f"mean_targets_per_sample={total_targets / max(total_samples, 1):.2f}"
        )
    elif args.quantization_correction == "bridge_ridge_qk_dpalign_module_replace":
        print(
            "\nBuilding dynamic-program source->target token pairs "
            "(global monotone alignment with context and output-overlap scores)..."
        )
        aligned_position_pairs = collect_dynamic_program_prompt_position_pairs(
            src,
            tok_s,
            tgt,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            source_reasoning_mode=args.source_reasoning_mode,
            source_use_chat_template=args.source_use_chat_template,
            source_enable_thinking=source_enable_thinking,
            target_use_chat_template=args.target_use_chat_template,
            target_enable_thinking=target_enable_thinking,
        )
        total_pairs = sum(len(pairs) for pairs in aligned_position_pairs)
        print(
            "Built dynamic-program token pairs: "
            f"prompts={len(aligned_position_pairs)}, pairs={total_pairs}, "
            f"mean_pairs_per_prompt={total_pairs / max(len(aligned_position_pairs), 1):.2f}"
        )

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
        source_use_chat_template=args.source_use_chat_template,
        target_use_chat_template=args.target_use_chat_template,
        source_enable_thinking=source_enable_thinking,
        target_enable_thinking=target_enable_thinking,
        aligned_position_pairs=aligned_position_pairs,
        aligned_position_mixtures=aligned_position_mixtures,
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
        whitening_streams=args.whitening_streams,
        target_whitening_streams=args.target_whitening_streams,
        conditioning_target_layers=(
            tuple(args.conditioning_target_layers)
            if args.conditioning_target_layers
            else None
        ),
        alignment_method=args.alignment,
        ridge_lambda=args.ridge_lambda,
        fit_ridge_override_lambda=args.fit_ridge_override_lambda,
        fit_ridge_override_streams=args.fit_ridge_override_streams,
        fit_ridge_override_layers=(
            tuple(args.fit_ridge_override_layers)
            if args.fit_ridge_override_layers
            else None
        ),
        fit_ridge_protected_rank=args.fit_ridge_protected_rank,
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
        quantization_correction_rank=args.quantization_correction_rank,
        bridge_bank_size=args.bridge_bank_size,
        learned_fusion_dropout=args.learned_fusion_dropout,
        seed=args.seed,
    )
    print(f"\nBuilding translator with config:\n  {config}")
    translator = RotAlignKVTranslator(config)

    if args.alignment in {
        "grouped_template_transport",
        "grouped_qk_retrieval_transport",
        "grouped_contrastive_template_transport",
        "grouped_template_subspace_transport",
        "broadcast_template_transport",
        "broadcast_template_ot_transport",
        "broadcast_peak_template_ot_transport",
        "broadcast_retrieval_spectrum_ot_transport",
        "broadcast_qk_template_ot_transport",
    }:
        group_count = math.gcd(config.src_num_heads, config.tgt_num_heads)
        is_broadcast = args.alignment in {
            "broadcast_template_transport",
            "broadcast_template_ot_transport",
            "broadcast_peak_template_ot_transport",
            "broadcast_retrieval_spectrum_ot_transport",
            "broadcast_qk_template_ot_transport",
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
                use_chat_template=args.source_use_chat_template,
                enable_thinking=source_enable_thinking,
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
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            print(
                "Built grouped key signatures: "
                f"source layers={len(translator._transport_src_group_templates)}, "
                f"target layers={len(translator._transport_tgt_group_templates)}"
            )
        elif args.alignment == "broadcast_qk_template_ot_transport":
            print(
                "\nBuilding grouped QK logit templates from calibration prompts "
                f"(source groups={src_template_groups}, target groups={tgt_template_groups}, bins={args.transport_template_bins})..."
            )
            translator._transport_src_group_templates = collect_group_qk_templates(
                src,
                tok_s,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.src_num_heads,
                group_count=src_template_groups,
                bins=args.transport_template_bins,
                reasoning_mode=args.source_reasoning_mode,
                use_chat_template=args.source_use_chat_template,
                enable_thinking=source_enable_thinking,
            )
            translator._transport_tgt_group_templates = collect_group_qk_templates(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=tgt_template_groups,
                bins=args.transport_template_bins,
                reasoning_mode="plain",
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            print(
                "Built grouped QK templates: "
                f"source layers={len(translator._transport_src_group_templates)}, "
                f"target layers={len(translator._transport_tgt_group_templates)}"
            )
        elif args.alignment == "grouped_qk_retrieval_transport":
            print(
                "\nBuilding grouped QK retrieval templates from calibration prompts "
                f"(source groups={src_template_groups}, target groups={tgt_template_groups}, bins={args.transport_template_bins})..."
            )
            translator._transport_src_group_templates = collect_group_qk_retrieval_templates(
                src,
                tok_s,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.src_num_heads,
                group_count=src_template_groups,
                bins=args.transport_template_bins,
                reasoning_mode=args.source_reasoning_mode,
                use_chat_template=args.source_use_chat_template,
                enable_thinking=source_enable_thinking,
            )
            translator._transport_tgt_group_templates = collect_group_qk_retrieval_templates(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=tgt_template_groups,
                bins=args.transport_template_bins,
                reasoning_mode="plain",
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            print(
                "Built grouped QK retrieval templates: "
                f"source layers={len(translator._transport_src_group_templates)}, "
                f"target layers={len(translator._transport_tgt_group_templates)}"
            )
        elif args.alignment == "grouped_contrastive_template_transport":
            print(
                "\nBuilding grouped attention-template banks from calibration prompts "
                f"(source groups={src_template_groups}, target groups={tgt_template_groups}, bins={args.transport_template_bins})..."
            )
            translator._transport_src_group_template_banks = collect_group_attention_template_bank(
                src,
                tok_s,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.src_num_heads,
                group_count=src_template_groups,
                bins=args.transport_template_bins,
                template_mode="mean",
                reasoning_mode=args.source_reasoning_mode,
                use_chat_template=args.source_use_chat_template,
                enable_thinking=source_enable_thinking,
            )
            translator._transport_tgt_group_template_banks = collect_group_attention_template_bank(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=tgt_template_groups,
                bins=args.transport_template_bins,
                template_mode="mean",
                reasoning_mode="plain",
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            translator._transport_src_group_templates = [
                bank.mean(dim=0) / bank.mean(dim=0).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                for bank in translator._transport_src_group_template_banks
            ]
            translator._transport_tgt_group_templates = [
                bank.mean(dim=0) / bank.mean(dim=0).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                for bank in translator._transport_tgt_group_template_banks
            ]
            print(
                "Built grouped template banks: "
                f"source layers={len(translator._transport_src_group_template_banks)}, "
                f"target layers={len(translator._transport_tgt_group_template_banks)}"
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
                use_chat_template=args.source_use_chat_template,
                enable_thinking=source_enable_thinking,
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
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
            print(
                "Built grouped attention templates: "
                f"source layers={len(translator._transport_src_group_templates)}, "
                f"target layers={len(translator._transport_tgt_group_templates)}"
            )

    aligned_lengths: list[int] | None = None
    if args.quantization_correction in {"bridge_low_rank_bank", "bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank", "bridge_ridge_qk_weighted", "bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter"}:
        if aligned_position_mixtures is not None:
            aligned_lengths = [len(mixture) for mixture in aligned_position_mixtures]
        elif aligned_position_pairs is not None:
            aligned_lengths = [len(pairs) for pairs in aligned_position_pairs]
        else:
            aligned_lengths = collect_aligned_prompt_valid_lengths(
                tok_s,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                source_reasoning_mode=args.source_reasoning_mode,
                source_use_chat_template=args.source_use_chat_template,
                source_enable_thinking=source_enable_thinking,
                target_use_chat_template=args.target_use_chat_template,
                target_enable_thinking=target_enable_thinking,
            )

    sample_prompt_ids = None
    if aligned_lengths is not None:
        if any(length <= 0 for length in aligned_lengths):
            print("Skipping zero-length calibration prompts while building bridge prompt ids")
        sample_prompt_ids = torch.cat(
            [
                torch.full((int(length),), prompt_idx, dtype=torch.long)
                for prompt_idx, length in enumerate(aligned_lengths)
                if int(length) > 0
            ],
            dim=0,
        )

    if args.innovation_target_set_json is not None:
        if (
            args.quantization_correction
            != "bridge_ridge_qk_dynalign_query_innovation_resampler_replace"
        ):
            raise ValueError(
                "--innovation-target-set-json is currently only supported for "
                "bridge_ridge_qk_dynalign_query_innovation_resampler_replace"
            )
        if sample_prompt_ids is None:
            raise ValueError(
                "--innovation-target-set-json requires aligned bridge sample prompt IDs"
            )
        target_ids = load_innovation_target_ids(args.innovation_target_set_json)
        prompt_weights, matched_prompts = build_innovation_prompt_weights(
            prompt_example_ids,
            target_ids,
            positive_weight=args.innovation_positive_weight,
            default_weight=args.innovation_default_weight,
        )
        if matched_prompts == 0:
            raise ValueError(
                "No calibration prompts matched ids.clean_residual_targets from "
                f"{args.innovation_target_set_json}; use a JSONL calibration file "
                "with stable example IDs."
            )
        sample_weights = prompt_weights[sample_prompt_ids]
        translator.set_bridge_sample_weights(
            [sample_weights.clone() for _ in range(config.num_tgt_layers)]
        )
        print(
            "Built innovation target sample weights: "
            f"matched_prompts={matched_prompts}, samples={int(sample_weights.numel())}, "
            f"default={args.innovation_default_weight:.3f}, "
            f"positive={args.innovation_positive_weight:.3f}"
        )

    if args.quantization_correction in {"bridge_low_rank_bank", "bridge_ridge_residual_bank", "bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace"}:
        if args.quantization_correction in {"bridge_ridge_qk_residual_bank", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace"}:
            print(
                "\nBuilding target QK template bank for query-conditioned bridge experts "
                f"(experts={args.bridge_bank_size}, bins={args.transport_template_bins})..."
            )
            bridge_template_bank = collect_group_qk_template_bank(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=1,
                bins=args.transport_template_bins,
                reasoning_mode="plain",
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
        else:
            print(
                "\nBuilding target attention template bank for query-conditioned bridge experts "
                f"(experts={args.bridge_bank_size}, bins={args.transport_template_bins})..."
            )
            bridge_template_bank = collect_group_attention_template_bank(
                tgt,
                tok_t,
                prompts,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                kv_heads=config.tgt_num_heads,
                group_count=1,
                bins=args.transport_template_bins,
                template_mode="mean",
                reasoning_mode="plain",
                use_chat_template=args.target_use_chat_template,
                enable_thinking=target_enable_thinking,
            )
        assert aligned_lengths is not None
        assert sample_prompt_ids is not None
        translator.set_bridge_runtime_template_bank(bridge_template_bank, sample_prompt_ids)
        print(
            "Built bridge template bank: "
            f"layers={len(bridge_template_bank)}, prompts={len(aligned_lengths)}, "
            f"samples={int(sample_prompt_ids.numel())}"
        )

    if args.quantization_correction in {"bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter"}:
        assert sample_prompt_ids is not None
        translator.set_bridge_sample_prompt_ids(sample_prompt_ids)
        print(
            "Built bridge sample prompt ids for local bridge distillation: "
            f"samples={int(sample_prompt_ids.numel())}"
        )

    if args.quantization_correction == "bridge_ridge_qk_weighted":
        assert aligned_lengths is not None
        print(
            "\nBuilding aligned target QK position weights for retrieval-weighted bridge fit "
            f"(bins={args.transport_template_bins})..."
        )
        bridge_sample_weights = collect_aligned_qk_position_weights(
            tgt,
            tok_t,
            prompts,
            aligned_lengths=aligned_lengths,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            kv_heads=config.tgt_num_heads,
            reasoning_mode="plain",
            use_chat_template=args.target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        translator.set_bridge_sample_weights(bridge_sample_weights)
        print(
            "Built aligned QK position weights: "
            f"layers={len(bridge_sample_weights)}, samples={int(bridge_sample_weights[0].numel())}"
        )

    if args.quantization_correction in {"bridge_ridge_qk_projector", "bridge_ridge_qk_adapter", "bridge_ridge_qk_affinity_adapter", "bridge_ridge_qk_attnkl_adapter", "bridge_ridge_qk_cab_adapter", "bridge_ridge_qk_emkd_adapter", "bridge_ridge_qk_readout_adapter", "bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_adapter", "bridge_ridge_qk_asym_projector", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_sae_adapter", "bridge_ridge_qk_generated_adapter", "bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"}:
        assert aligned_lengths is not None
        print(
            "\nBuilding aligned target query features for query-conditioned bridge projector/adapter "
            f"(width={config.tgt_num_heads * config.tgt_head_dim})..."
        )
        bridge_query_features = collect_aligned_query_features(
            tgt,
            tok_t,
            prompts,
            aligned_lengths=aligned_lengths,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            kv_heads=config.tgt_num_heads,
            head_dim=config.tgt_head_dim,
            reasoning_mode="plain",
            use_chat_template=args.target_use_chat_template,
            enable_thinking=target_enable_thinking,
            aligned_position_pairs=aligned_position_pairs,
            aligned_position_mixtures=aligned_position_mixtures,
        )
        translator.set_bridge_sample_query_features(bridge_query_features)
        print(
            "Built aligned query features: "
            f"layers={len(bridge_query_features)}, samples={int(bridge_query_features[0].shape[0])}"
        )

    if args.quantization_correction in {"bridge_ridge_qk_predkl_adapter", "bridge_ridge_qk_asym_predkl_adapter", "bridge_ridge_qk_asym_dynmap_adapter", "bridge_ridge_qk_xattn_dynmap_adapter", "bridge_ridge_qk_module_adapter", "bridge_ridge_qk_module_replace", "bridge_ridge_qk_bytespan_module_replace", "bridge_ridge_qk_spanalign_module_replace", "bridge_ridge_qk_ctxalign_module_replace", "bridge_ridge_qk_dynalign_ctxonly_module_replace", "bridge_ridge_qk_dynalign_module_replace", "bridge_ridge_qk_dynalign_preserve_module_replace", "bridge_ridge_qk_dynalign_eigenspace_module_replace", "bridge_ridge_qk_dynalign_saliency_module_replace", "bridge_ridge_qk_dynalign_saliency_preserve_module_replace", "bridge_ridge_qk_dynalign_anchor_tail_module_replace", "bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace", "bridge_ridge_qk_dynalign_routed_module_replace", "bridge_ridge_qk_dynalign_value_routed_module_replace", "bridge_ridge_qk_dynalign_query_resampler_replace", "bridge_ridge_qk_dynalign_query_innovation_resampler_replace", "bridge_ridge_qk_dynalign_value_bank_module_replace", "bridge_ridge_qk_dynalign_value_query_bank_module_replace", "bridge_ridge_qk_dynalign_value_routed_bank_module_replace", "bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace", "bridge_ridge_qk_dynalign_dwakd_module_replace", "bridge_ridge_qk_dynalign_likelihood_module_replace", "bridge_ridge_qk_dynalign_spanalm_module_replace", "bridge_ridge_qk_dynalign_prefdist_module_replace", "bridge_ridge_qk_dynalign_dwainteract_module_replace", "bridge_ridge_qk_dynalign_interact_module_replace", "bridge_ridge_qk_dpalign_module_replace", "bridge_ridge_qk_tokenbasis_replace", "bridge_ridge_qk_predkl_bank"}:
        assert aligned_lengths is not None
        print(
            "\nBuilding aligned target next-token teacher for prediction-level bridge distillation "
            "(topk=8)..."
        )
        teacher_log_probs, teacher_output_rows = collect_aligned_prediction_teacher(
            tgt,
            tok_t,
            prompts,
            aligned_lengths=aligned_lengths,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            topk=8,
            reasoning_mode="plain",
            use_chat_template=args.target_use_chat_template,
            enable_thinking=target_enable_thinking,
            aligned_position_pairs=aligned_position_pairs,
            aligned_position_mixtures=aligned_position_mixtures,
            include_next_token_targets=args.quantization_correction == "bridge_ridge_qk_dynalign_likelihood_module_replace",
            next_token_target_weight=0.5,
            span_likelihood_window=3 if args.quantization_correction == "bridge_ridge_qk_dynalign_spanalm_module_replace" else 0,
            span_likelihood_weight=0.20 if args.quantization_correction == "bridge_ridge_qk_dynalign_spanalm_module_replace" else 0.0,
        )
        translator.set_bridge_prediction_teacher(teacher_log_probs, teacher_output_rows)
        print(
            "Built aligned prediction teacher: "
            f"samples={int(teacher_log_probs.shape[0])}, topk={int(teacher_log_probs.shape[1])}"
        )
        if args.quantization_correction in {
            "bridge_ridge_qk_dynalign_dwakd_module_replace",
            "bridge_ridge_qk_dynalign_likelihood_module_replace",
            "bridge_ridge_qk_dynalign_spanalm_module_replace",
            "bridge_ridge_qk_dynalign_prefdist_module_replace",
            "bridge_ridge_qk_dynalign_dwainteract_module_replace",
        }:
            if aligned_position_mixtures is None:
                raise ValueError(f"{args.quantization_correction} requires aligned_position_mixtures")
            alignment_weights = collect_alignment_confidence_weights(aligned_position_mixtures)
            prediction_weights = collect_prediction_confidence_weights(teacher_log_probs)
            if alignment_weights.shape[0] != prediction_weights.shape[0]:
                raise ValueError(
                    "alignment and prediction confidence weights must align, "
                    f"got {alignment_weights.shape[0]} vs {prediction_weights.shape[0]}"
                )
            combined = 0.5 * alignment_weights + 0.5 * prediction_weights
            combined = combined / combined.mean().clamp_min(1e-8)
            combined = combined.clamp(0.25, 4.0)
            translator.set_bridge_sample_weights([combined.clone() for _ in range(config.num_tgt_layers)])
            print(
                "Built dynamic bridge sample weights: "
                f"samples={int(combined.numel())}, mean={float(combined.mean().item()):.3f}, "
                f"min={float(combined.min().item()):.3f}, max={float(combined.max().item()):.3f}"
            )

    if args.quantization_correction in {
        "bridge_ridge_qk_dynalign_dwainteract_module_replace",
        "bridge_ridge_qk_dynalign_interact_module_replace",
    }:
        assert sample_prompt_ids is not None
        translator.set_bridge_sample_prompt_ids(sample_prompt_ids)
        print(
            "Built bridge sample prompt ids for dynalign prompt-local interaction distillation: "
            f"samples={int(sample_prompt_ids.numel())}"
        )

    print("\nFitting closed-form alignments (Procrustes / ridge)...")
    diagnostics = translator.fit_from_pairs(src_kvs, tgt_kvs, verbose=args.verbose)

    if args.quantization_correction == "bridge_ridge_query":
        print(
            "\nBuilding target runtime attention templates for query-conditioned bridge gating "
            f"(bins={args.transport_template_bins})..."
        )
        bridge_templates = collect_group_attention_templates(
            tgt,
            tok_t,
            prompts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            kv_heads=config.tgt_num_heads,
            group_count=1,
            bins=args.transport_template_bins,
            template_mode="mean",
            reasoning_mode="plain",
            use_chat_template=args.target_use_chat_template,
            enable_thinking=target_enable_thinking,
        )
        translator.set_bridge_runtime_templates([template.squeeze(0) for template in bridge_templates])
        print(f"Built {len(bridge_templates)} bridge runtime templates")

    del src
    del tgt
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

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
