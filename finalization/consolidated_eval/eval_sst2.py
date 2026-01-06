#!/usr/bin/env python
"""
SST-2 Sentiment Classification Evaluation for LatentWire

Complete evaluation module for SST-2 (Stanford Sentiment Treebank v2).
Supports both text baselines and latent soft-token evaluation with:
- Full 872 sample validation set
- Per-class accuracy tracking
- Confidence intervals via bootstrap
- Batch evaluation for efficiency
- Integration with frozen LLM + soft token inputs_embeds interface

Dataset:
    - Source: GLUE SST-2 validation split
    - Size: 872 samples
    - Labels: 0=negative, 1=positive
    - Metric: Accuracy

Usage:
    # Text baseline (upper bound)
    python latentwire/eval_sst2.py --mode text --model_id meta-llama/Meta-Llama-3.1-8B-Instruct

    # Latent evaluation with checkpoint
    python latentwire/eval_sst2.py --mode latent --ckpt runs/exp/epoch10 --model llama

    # Batch evaluation for efficiency
    python latentwire/eval_sst2.py --mode latent --ckpt runs/exp/epoch10 --batch_size 8
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from latentwire.models import (
    InterlinguaEncoder,
    Adapter,
    LMWrapper,
    LMConfig,
    ByteTokenizer,
)
from latentwire.core_utils import (
    clean_pred,
    tensor_rms,
    collate_bytes,
)


# ========================================================================
# Configuration & Data Loading
# ========================================================================

@dataclass
class SST2Config:
    """Configuration for SST-2 evaluation."""
    mode: str = "text"  # text, latent, noise
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model: str = "llama"  # llama or qwen
    ckpt: Optional[str] = None
    batch_size: int = 1
    num_samples: int = 872  # Full validation set
    max_new_tokens: int = 10
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    output_dir: str = "runs/sst2_eval"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    seed: int = 42
    # Latent configuration
    latent_len: int = 32
    d_z: int = 256
    calibration: str = "embed_rms"
    encoder_type: str = "byte"
    # Prompt configuration
    prompt_template: str = "standard"  # standard, minimal, verbose


LABEL_NAMES = {0: "negative", 1: "positive"}


def load_sst2_validation(num_samples: int = 872) -> List[Dict[str, Any]]:
    """
    Load SST-2 validation set from GLUE.

    Returns:
        List of dicts with keys: sentence, label (0/1), label_name
    """
    ds = load_dataset("glue", "sst2", split="validation")

    # Take first num_samples (deterministic)
    samples = []
    for i in range(min(num_samples, len(ds))):
        item = ds[i]
        samples.append({
            "sentence": item["sentence"],
            "label": item["label"],
            "label_name": LABEL_NAMES[item["label"]],
        })

    return samples


def format_sst2_prompt(sentence: str, template: str = "standard") -> str:
    """
    Format SST-2 prompt for classification.

    Args:
        sentence: Review text
        template: Prompt style (standard, minimal, verbose)

    Returns:
        Formatted prompt string
    """
    if template == "minimal":
        return f"{sentence}\nSentiment:"
    elif template == "verbose":
        return f"Classify the sentiment of this review as 'positive' or 'negative':\n\nReview: {sentence}\n\nSentiment:"
    else:  # standard
        return f"Classify sentiment as 'positive' or 'negative': {sentence}\nSentiment: "


# ========================================================================
# Prediction Extraction
# ========================================================================

def extract_sentiment_prediction(output: str) -> str:
    """
    Extract sentiment prediction from model output.

    Handles various output formats:
        - "positive" or "negative" in text
        - Multi-word responses ("The sentiment is positive")
        - Edge cases (unknown, both, neither)

    Returns:
        "positive", "negative", or "unknown"
    """
    output_lower = output.lower().strip()

    # Clean common prefixes
    output_lower = clean_pred(output_lower)

    # Check for explicit labels
    has_positive = "positive" in output_lower
    has_negative = "negative" in output_lower

    if has_positive and not has_negative:
        return "positive"
    elif has_negative and not has_positive:
        return "negative"
    elif has_positive and has_negative:
        # Both present - take first occurrence
        pos_idx = output_lower.find("positive")
        neg_idx = output_lower.find("negative")
        return "positive" if pos_idx < neg_idx else "negative"
    else:
        return "unknown"


# ========================================================================
# Evaluation Functions
# ========================================================================

@torch.no_grad()
def eval_text_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    config: SST2Config,
) -> Dict[str, Any]:
    """
    Evaluate model with full text prompts (upper bound baseline).

    Args:
        model: LLM to evaluate
        tokenizer: Corresponding tokenizer
        samples: SST-2 samples
        config: Evaluation configuration

    Returns:
        Dictionary with accuracy metrics and per-example results
    """
    device = next(model.parameters()).device
    results = []
    correct = 0

    # Per-class tracking
    pos_correct, pos_total = 0, 0
    neg_correct, neg_total = 0, 0

    for sample in tqdm(samples, desc="Text baseline"):
        prompt = format_sst2_prompt(sample["sentence"], config.prompt_template)

        # Tokenize and generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Extract only generated tokens
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract prediction
        pred = extract_sentiment_prediction(output_text)
        is_correct = (pred == sample["label_name"])

        if is_correct:
            correct += 1

        # Track per-class
        if sample["label"] == 1:  # positive
            pos_total += 1
            if is_correct:
                pos_correct += 1
        else:  # negative
            neg_total += 1
            if is_correct:
                neg_correct += 1

        results.append({
            "sentence": sample["sentence"],
            "label": sample["label_name"],
            "prediction": pred,
            "output": output_text[:100],
            "correct": is_correct,
        })

    total = len(samples)
    accuracy = 100 * correct / total
    pos_acc = 100 * pos_correct / max(pos_total, 1)
    neg_acc = 100 * neg_correct / max(neg_total, 1)

    return {
        "mode": "text",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "positive_total": pos_total,
        "negative_total": neg_total,
        "per_class": {
            "positive": {"correct": pos_correct, "total": pos_total, "accuracy": pos_acc},
            "negative": {"correct": neg_correct, "total": neg_total, "accuracy": neg_acc},
        },
        "samples": results,
    }


@torch.no_grad()
def eval_latent_mode(
    wrapper: LMWrapper,
    encoder: InterlinguaEncoder,
    adapter: Adapter,
    samples: List[Dict[str, Any]],
    config: SST2Config,
) -> Dict[str, Any]:
    """
    Evaluate with latent soft tokens (LatentWire compression).

    Args:
        wrapper: LM wrapper (frozen LLM)
        encoder: Interlingua encoder
        adapter: Model-specific adapter
        samples: SST-2 samples
        config: Evaluation configuration

    Returns:
        Dictionary with accuracy metrics and per-example results
    """
    device = next(wrapper.model.parameters()).device
    results = []
    correct = 0

    # Per-class tracking
    pos_correct, pos_total = 0, 0
    neg_correct, neg_total = 0, 0

    # Compute target RMS for calibration
    if config.calibration == "embed_rms":
        target_rms = wrapper.input_embedding_rms()
    else:
        target_rms = 0.015  # Fixed default

    # Process in batches
    for i in tqdm(range(0, len(samples), config.batch_size), desc="Latent eval"):
        batch_samples = samples[i:i + config.batch_size]
        batch_size = len(batch_samples)

        # Format prompts
        prompts = [
            format_sst2_prompt(s["sentence"], config.prompt_template)
            for s in batch_samples
        ]

        # Encode to latent
        if config.encoder_type == "byte":
            byte_tok = ByteTokenizer(max_bytes=512)
            z_bytes = collate_bytes(prompts, byte_tok, device)
            Z = encoder(z_bytes)  # [B, M, d_z]
        else:
            # Simple encoder (uses wrapper's tokenizer)
            Z = encoder(prompts)  # [B, M, d_z]

        # Apply adapter and calibration
        prefix_embeds = adapter(Z)  # [B, M, d_model]

        # Per-example calibration
        if config.calibration != "none":
            cur_rms = prefix_embeds.float().pow(2).mean(dim=[1, 2], keepdim=True).sqrt().clamp_min(1e-8)
            gain = (target_rms / cur_rms).to(prefix_embeds.dtype)
            prefix_embeds = prefix_embeds * gain

        # Add anchor text embedding: "Sentiment: "
        anchor_text = "Sentiment: "
        anchor_tokens = wrapper.tokenizer(
            [anchor_text] * batch_size,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        anchor_embeds = wrapper.model.get_input_embeddings()(anchor_tokens.input_ids)

        # Combine: [prefix] + [anchor]
        combined_embeds = torch.cat([prefix_embeds, anchor_embeds], dim=1)
        attn_mask = torch.ones(
            combined_embeds.shape[:2],
            device=device,
            dtype=torch.long,
        )

        # Generate
        outputs = wrapper.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0,
            pad_token_id=wrapper.tokenizer.eos_token_id,
        )

        # Process batch outputs
        for j, sample in enumerate(batch_samples):
            output_text = wrapper.tokenizer.decode(outputs[j], skip_special_tokens=True)

            # Extract prediction
            pred = extract_sentiment_prediction(output_text)
            is_correct = (pred == sample["label_name"])

            if is_correct:
                correct += 1

            # Track per-class
            if sample["label"] == 1:  # positive
                pos_total += 1
                if is_correct:
                    pos_correct += 1
            else:  # negative
                neg_total += 1
                if is_correct:
                    neg_correct += 1

            results.append({
                "sentence": sample["sentence"],
                "label": sample["label_name"],
                "prediction": pred,
                "output": output_text[:100],
                "correct": is_correct,
            })

    total = len(samples)
    accuracy = 100 * correct / total
    pos_acc = 100 * pos_correct / max(pos_total, 1)
    neg_acc = 100 * neg_correct / max(neg_total, 1)

    return {
        "mode": "latent",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "positive_total": pos_total,
        "negative_total": neg_total,
        "per_class": {
            "positive": {"correct": pos_correct, "total": pos_total, "accuracy": pos_acc},
            "negative": {"correct": neg_correct, "total": neg_total, "accuracy": neg_acc},
        },
        "samples": results,
    }


@torch.no_grad()
def eval_noise_baseline(
    wrapper: LMWrapper,
    samples: List[Dict[str, Any]],
    config: SST2Config,
) -> Dict[str, Any]:
    """
    Evaluate with random noise soft tokens (should be ~50% accuracy).

    This baseline checks if the model has intrinsic bias or if soft tokens
    are actually meaningful.

    Args:
        wrapper: LM wrapper (frozen LLM)
        samples: SST-2 samples
        config: Evaluation configuration

    Returns:
        Dictionary with accuracy metrics
    """
    device = next(wrapper.model.parameters()).device
    results = []
    correct = 0

    # Per-class tracking
    pos_correct, pos_total = 0, 0
    neg_correct, neg_total = 0, 0

    # Target RMS for scaling noise
    target_rms = wrapper.input_embedding_rms()
    d_model = wrapper.model.config.hidden_size

    for sample in tqdm(samples, desc="Noise baseline"):
        # Random soft tokens scaled to typical embedding magnitude
        noise_tokens = torch.randn(
            1, config.latent_len, d_model,
            device=device,
            dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
        ) * target_rms

        # Add anchor
        anchor_text = "Sentiment: "
        anchor_tokens = wrapper.tokenizer(
            anchor_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        anchor_embeds = wrapper.model.get_input_embeddings()(anchor_tokens.input_ids)

        # Combine
        combined_embeds = torch.cat([noise_tokens, anchor_embeds], dim=1)
        attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

        # Generate
        outputs = wrapper.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=wrapper.tokenizer.eos_token_id,
        )

        output_text = wrapper.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract prediction
        pred = extract_sentiment_prediction(output_text)
        is_correct = (pred == sample["label_name"])

        if is_correct:
            correct += 1

        # Track per-class
        if sample["label"] == 1:
            pos_total += 1
            if is_correct:
                pos_correct += 1
        else:
            neg_total += 1
            if is_correct:
                neg_correct += 1

        results.append({
            "sentence": sample["sentence"],
            "label": sample["label_name"],
            "prediction": pred,
            "output": output_text[:100],
            "correct": is_correct,
        })

    total = len(samples)
    accuracy = 100 * correct / total
    pos_acc = 100 * pos_correct / max(pos_total, 1)
    neg_acc = 100 * neg_correct / max(neg_total, 1)

    return {
        "mode": "noise",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "per_class": {
            "positive": {"correct": pos_correct, "total": pos_total, "accuracy": pos_acc},
            "negative": {"correct": neg_correct, "total": neg_total, "accuracy": neg_acc},
        },
        "samples": results[:20],  # Save subset for inspection
    }


# ========================================================================
# Bootstrap Confidence Intervals
# ========================================================================

def bootstrap_confidence_interval(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for accuracy.

    Args:
        results: List of per-example results with "correct" key
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) as percentages
    """
    import random

    n = len(results)
    accuracies = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = random.choices(results, k=n)
        acc = 100 * sum(r["correct"] for r in resampled) / n
        accuracies.append(acc)

    accuracies.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return accuracies[lower_idx], accuracies[upper_idx]


# ========================================================================
# Main Evaluation Pipeline
# ========================================================================

def load_checkpoint_for_eval(
    ckpt_path: str,
    model_key: str,
    device_map: str,
    dtype: str,
) -> Tuple[LMWrapper, InterlinguaEncoder, Adapter, dict]:
    """
    Load checkpoint and construct models for evaluation.

    Args:
        ckpt_path: Path to checkpoint directory
        model_key: "llama" or "qwen"
        device_map: Device placement strategy
        dtype: Model dtype (bfloat16, float32, etc.)

    Returns:
        (wrapper, encoder, adapter, config)
    """
    # Load config
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path) as f:
        train_config = json.load(f)

    # Determine model ID
    model_id = train_config.get(
        f"{model_key}_id",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" if model_key == "llama" else "Qwen/Qwen2.5-7B-Instruct"
    )

    # Load LM wrapper
    print(f"Loading {model_key} model: {model_id}")
    lm_config = LMConfig(
        model_id=model_id,
        device_map=device_map,
        torch_dtype=dtype,
    )
    wrapper = LMWrapper(lm_config)
    wrapper.model.eval()

    device = next(wrapper.model.parameters()).device

    # Load encoder
    encoder_type = train_config.get("encoder_type", "byte")
    latent_len = train_config.get("latent_len", 32)
    d_z = train_config.get("d_z", 256)

    print(f"Loading encoder (type={encoder_type}, M={latent_len}, d_z={d_z})")
    if encoder_type == "byte":
        from latentwire.models import ByteTokenizer
        encoder = InterlinguaEncoder(
            vocab_size=256,
            d_model=512,
            latent_len=latent_len,
            d_z=d_z,
            n_layer=4,
            n_head=8,
        ).to(device)
    else:
        raise NotImplementedError(f"Encoder type {encoder_type} not yet supported in eval_sst2")

    encoder_path = os.path.join(ckpt_path, "encoder.pt")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.eval()

    # Load adapter
    d_model = wrapper.model.config.hidden_size
    adapter = Adapter(d_z, d_model).to(device)

    adapter_path = os.path.join(ckpt_path, f"adapter_{model_key}.pt")
    adapter.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True))
    adapter.eval()

    return wrapper, encoder, adapter, train_config


def main():
    parser = argparse.ArgumentParser(description="SST-2 Evaluation for LatentWire")

    # Mode selection
    parser.add_argument("--mode", choices=["text", "latent", "noise"], default="text",
                        help="Evaluation mode")

    # Model configuration
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model ID for text baseline")
    parser.add_argument("--model", choices=["llama", "qwen"], default="llama",
                        help="Model family for latent mode")
    parser.add_argument("--ckpt", type=str,
                        help="Checkpoint path for latent mode")

    # Dataset configuration
    parser.add_argument("--num_samples", type=int, default=872,
                        help="Number of samples (default: full validation set)")
    parser.add_argument("--prompt_template", choices=["standard", "minimal", "verbose"],
                        default="standard", help="Prompt format")

    # Generation configuration
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Max tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling threshold")

    # System configuration
    parser.add_argument("--device_map", default="auto",
                        help="Device map for model loading")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float16"],
                        default="bfloat16", help="Model dtype")
    parser.add_argument("--output_dir", default="runs/sst2_eval",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Create config
    config = SST2Config(**vars(args))

    # Set seed
    torch.manual_seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 80)
    print("SST-2 SENTIMENT CLASSIFICATION EVALUATION")
    print("=" * 80)
    print(f"Mode: {config.mode}")
    print(f"Samples: {config.num_samples}")
    print(f"Prompt: {config.prompt_template}")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading SST-2 validation set...")
    samples = load_sst2_validation(config.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Compute class distribution
    pos_count = sum(1 for s in samples if s["label"] == 1)
    neg_count = len(samples) - pos_count
    print(f"  Positive: {pos_count} ({100*pos_count/len(samples):.1f}%)")
    print(f"  Negative: {neg_count} ({100*neg_count/len(samples):.1f}%)")
    print(f"  Majority baseline: {100*max(pos_count, neg_count)/len(samples):.1f}%")
    print()

    # Run evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    if config.mode == "text":
        # Text baseline
        print(f"Loading model: {config.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
            device_map=config.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        results = eval_text_baseline(model, tokenizer, samples, config)

    elif config.mode == "latent":
        # Latent evaluation
        if not config.ckpt:
            raise ValueError("--ckpt required for latent mode")

        wrapper, encoder, adapter, train_config = load_checkpoint_for_eval(
            config.ckpt,
            config.model,
            config.device_map,
            config.dtype,
        )

        # Update config with checkpoint settings
        config.latent_len = train_config.get("latent_len", 32)
        config.d_z = train_config.get("d_z", 256)
        config.encoder_type = train_config.get("encoder_type", "byte")
        config.calibration = train_config.get("calibration", "embed_rms")

        results = eval_latent_mode(wrapper, encoder, adapter, samples, config)

    elif config.mode == "noise":
        # Noise baseline
        print(f"Loading model: {config.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
            device_map=config.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        lm_config = LMConfig(
            model_id=config.model_id,
            device_map=config.device_map,
            torch_dtype=config.dtype,
        )
        wrapper = LMWrapper(lm_config)
        wrapper.model = model
        wrapper.tokenizer = tokenizer

        results = eval_noise_baseline(wrapper, samples, config)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Compute confidence intervals
    print("\nComputing bootstrap confidence intervals...")
    ci_lower, ci_upper = bootstrap_confidence_interval(results["samples"])

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
    print(f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print()
    print(f"Positive class: {results['positive_accuracy']:.2f}% ({results['per_class']['positive']['correct']}/{results['per_class']['positive']['total']})")
    print(f"Negative class: {results['negative_accuracy']:.2f}% ({results['per_class']['negative']['correct']}/{results['per_class']['negative']['total']})")
    print()
    print(f"Time: {elapsed:.1f}s ({elapsed/len(samples):.3f}s/sample)")
    print("=" * 80)

    # Interpretation
    acc = results["accuracy"]
    print()
    if acc < 55:
        print("INTERPRETATION: Near random (50%) - model not learning sentiment")
    elif acc < 70:
        print("INTERPRETATION: Some signal but weak performance")
    elif acc < 85:
        print("INTERPRETATION: Good performance, model understands sentiment")
    else:
        print("INTERPRETATION: Excellent performance, near human-level")

    # Save results
    output_path = os.path.join(config.output_dir, f"sst2_{config.mode}_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "config": vars(config),
            "results": results,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "confidence": 0.95,
            },
            "elapsed_seconds": elapsed,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
