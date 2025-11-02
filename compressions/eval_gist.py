#!/usr/bin/env python3
"""
Evaluate gist tokens on Alpaca dataset with baselines.

Compares:
1. Full text (positive control - full prompt)
2. Gist tokens (our method - compressed prompt, NO instruction)
3. Truncated text (negative control - truncate to gist token count)

Measures:
- ROUGE scores (generation quality)
- Compression ratio (actual compression by removing instruction)
- Output quality vs baselines
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import GistLlama wrapper and mask functions from training script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from compressions.train_gist_faithful import GistLlama, make_gist_mask


def load_gist_model(checkpoint_dir: str, device: str):
    """Load trained gist model with GistLlama wrapper."""
    print(f"Loading gist model from {checkpoint_dir}...")

    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Load gist embedding data
    gist_path = Path(checkpoint_dir) / "gist_embedding.pt"
    if not gist_path.exists():
        raise FileNotFoundError(f"No gist embedding found at {gist_path}")

    gist_data = torch.load(gist_path, map_location=device, weights_only=False)
    num_gist_tokens = gist_data['num_gist_tokens']
    gist_token_id = gist_data['gist_token_id']

    # Wrap with GistLlama to enable gist attention masking
    model = GistLlama(
        base_model=base_model,
        num_gist_tokens=num_gist_tokens,
        gist_token_id=gist_token_id,
        hidden_dim=base_model.config.hidden_size
    )

    # Load trained gist embedding
    model.gist_embedding.data = gist_data['gist_embedding'].to(device)

    print(f"✓ Loaded GistLlama wrapper with {num_gist_tokens} gist tokens")
    return model, tokenizer, gist_data


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 128, device: str = "cuda", use_gist_mask: bool = False, gist_token_id: Optional[int] = None):
    """Generate outputs from batch of prompts with optional gist attention masking."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    input_lengths = inputs.attention_mask.sum(dim=1)

    # Prepare kwargs for generation
    gen_kwargs = {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'max_new_tokens': max_new_tokens,
        'do_sample': False,  # Deterministic for fair comparison
        'pad_token_id': tokenizer.pad_token_id,
    }

    # Add gist attention mask if requested
    if use_gist_mask and gist_token_id is not None:
        # Create gist attention mask for the batch
        attention_mask_gist = make_gist_mask(
            inputs.input_ids,
            gist_token=gist_token_id,
            pad_token=tokenizer.pad_token_id,
            dtype=torch.bool
        )
        gen_kwargs['attention_mask_gist'] = attention_mask_gist

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # Decode only the generated part (skip input) for each sample
    generated_texts = []
    for i, output in enumerate(outputs):
        generated = tokenizer.decode(output[input_lengths[i]:], skip_special_tokens=True)
        generated_texts.append(generated.strip())

    return generated_texts


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    # Average scores
    return {
        'rouge1': sum(scores['rouge1']) / len(scores['rouge1']),
        'rouge2': sum(scores['rouge2']) / len(scores['rouge2']),
        'rougeL': sum(scores['rougeL']) / len(scores['rougeL']),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gist tokens with baselines")
    parser.add_argument('--checkpoint', type=str, required=True, help='Gist model checkpoint directory')
    parser.add_argument('--samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (defaults to checkpoint dir)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GIST TOKENS EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {args.samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")

    # Load model
    model, tokenizer, gist_data = load_gist_model(args.checkpoint, args.device)
    model.eval()
    num_gist_tokens = gist_data['num_gist_tokens']
    gist_token_id = gist_data['gist_token_id']

    # Load test data (held-out from Alpaca)
    print("Loading Alpaca test data...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    # CRITICAL: Use same shuffle as training (seed=42) to ensure proper split
    dataset = dataset.shuffle(seed=42)

    # Read actual number of training samples from checkpoint metadata
    metrics_path = os.path.join(args.checkpoint, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    num_train_samples = metrics['num_samples']

    # Training uses first N samples, we use samples after that for test
    # This ensures zero overlap with training data
    total_size = len(dataset)
    test_start_idx = num_train_samples  # Start right after training data
    test_end_idx = min(test_start_idx + args.samples, total_size)

    if test_end_idx <= test_start_idx:
        raise ValueError(f"Not enough data for test set! Training used {num_train_samples} samples, "
                        f"dataset has {total_size} total. Need at least {num_train_samples + args.samples} samples.")

    test_data = dataset.select(range(test_start_idx, test_end_idx))
    print(f"✓ Loaded {len(test_data)} test samples (indices {test_start_idx}-{test_end_idx})")
    print(f"✓ Zero overlap with training data (training used indices 0-{num_train_samples} after shuffle)\n")

    # Evaluate all samples in batches
    print(f"Evaluating samples (batch_size={args.batch_size})...")
    full_outputs, gist_outputs, truncated_outputs, references = [], [], [], []
    all_prompts_full, all_prompts_gist, all_prompts_truncated = [], [], []

    start_time = time.time()

    # First, prepare all prompts
    print("Preparing prompts...")
    print("CRITICAL: Gist prompts have INSTRUCTION REMOVED for true compression!")
    for example in test_data:
        instruction = example['instruction']
        input_text = example.get('input', '')

        # Format using chat template for Llama 3.1 Instruct

        # Full prompt baseline (includes full instruction + input)
        if input_text:
            full_user_content = f"{instruction}\n\n{input_text}"
        else:
            full_user_content = instruction

        full_messages = [{"role": "user", "content": full_user_content}]
        full_prompt = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )

        # Gist prompt: REPLACE instruction with gist tokens, keep input
        # This achieves the actual compression that the paper demonstrates
        gist_str = " ".join(["<GIST>"] * num_gist_tokens)
        if input_text:
            gist_user_content = f"{gist_str}\n\n{input_text}"
        else:
            gist_user_content = gist_str

        gist_messages = [{"role": "user", "content": gist_user_content}]
        gist_prompt = tokenizer.apply_chat_template(
            gist_messages, tokenize=False, add_generation_prompt=True
        )

        # Truncated text baseline (truncate instruction to similar length as gist)
        # Truncate the user content, not the full formatted prompt
        full_content_tokens = tokenizer(full_user_content, add_special_tokens=False).input_ids
        truncate_len = min(len(full_content_tokens), num_gist_tokens * 10)
        truncated_content = tokenizer.decode(
            full_content_tokens[:truncate_len], skip_special_tokens=True
        )

        truncated_messages = [{"role": "user", "content": truncated_content}]
        truncated_prompt = tokenizer.apply_chat_template(
            truncated_messages, tokenize=False, add_generation_prompt=True
        )

        all_prompts_full.append(full_prompt)
        all_prompts_gist.append(gist_prompt)
        all_prompts_truncated.append(truncated_prompt)
        references.append(example['output'])

    # Process in batches
    print("Generating outputs...")
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Batches"):
        batch_end = min(i + args.batch_size, len(test_data))

        # Full text baseline (no gist masking)
        full_batch = generate_batch(
            model, tokenizer, all_prompts_full[i:batch_end],
            args.max_new_tokens, args.device,
            use_gist_mask=False
        )
        full_outputs.extend(full_batch)

        # Gist tokens (WITH gist masking - critical!)
        gist_batch = generate_batch(
            model, tokenizer, all_prompts_gist[i:batch_end],
            args.max_new_tokens, args.device,
            use_gist_mask=True,
            gist_token_id=gist_token_id
        )
        gist_outputs.extend(gist_batch)

        # Truncated baseline (no gist masking)
        truncated_batch = generate_batch(
            model, tokenizer, all_prompts_truncated[i:batch_end],
            args.max_new_tokens, args.device,
            use_gist_mask=False
        )
        truncated_outputs.extend(truncated_batch)

    elapsed = time.time() - start_time

    # Store results for saving
    results = []
    for i in range(len(test_data)):
        results.append({
            "full_text": full_outputs[i],
            "gist": gist_outputs[i],
            "truncated": truncated_outputs[i],
            "reference": references[i],
            "prompts": {
                "full": all_prompts_full[i],
                "gist": all_prompts_gist[i],
                "truncated": all_prompts_truncated[i],
            }
        })

    # Compute ROUGE scores
    print("\nComputing ROUGE scores...")
    rouge_full = compute_rouge(full_outputs, references)
    rouge_gist = compute_rouge(gist_outputs, references)
    rouge_truncated = compute_rouge(truncated_outputs, references)

    # Compute compression ratio
    def avg_tokens(prompts):
        return sum(len(tokenizer(p).input_ids) for p in prompts) / len(prompts)

    avg_full_tokens = avg_tokens([r['prompts']['full'] for r in results])
    avg_gist_tokens = avg_tokens([r['prompts']['gist'] for r in results])
    compression_ratio = avg_full_tokens / avg_gist_tokens

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nCompression:")
    print(f"  Full text avg tokens: {avg_full_tokens:.1f}")
    print(f"  Gist avg tokens: {avg_gist_tokens:.1f}")
    print(f"  Compression ratio: {compression_ratio:.2f}×")

    print(f"\nROUGE Scores:")
    print(f"\n  Full Text (positive control):")
    print(f"    ROUGE-1: {rouge_full['rouge1']:.4f}")
    print(f"    ROUGE-2: {rouge_full['rouge2']:.4f}")
    print(f"    ROUGE-L: {rouge_full['rougeL']:.4f}")

    print(f"\n  Gist Tokens (our method):")
    print(f"    ROUGE-1: {rouge_gist['rouge1']:.4f}")
    print(f"    ROUGE-2: {rouge_gist['rouge2']:.4f}")
    print(f"    ROUGE-L: {rouge_gist['rougeL']:.4f}")

    print(f"\n  Truncated Text (negative control):")
    print(f"    ROUGE-1: {rouge_truncated['rouge1']:.4f}")
    print(f"    ROUGE-2: {rouge_truncated['rouge2']:.4f}")
    print(f"    ROUGE-L: {rouge_truncated['rougeL']:.4f}")

    print(f"\n  Relative Performance (vs Full Text):")
    print(f"    Gist ROUGE-L: {rouge_gist['rougeL']/rouge_full['rougeL']*100:.1f}% of full text")
    print(f"    Truncated ROUGE-L: {rouge_truncated['rougeL']/rouge_full['rougeL']*100:.1f}% of full text")

    print(f"\n  Evaluation time: {elapsed:.1f}s ({elapsed/len(test_data):.2f}s per sample)")
    print("="*80 + "\n")

    # Save results
    eval_results = {
        'checkpoint': args.checkpoint,
        'num_samples': len(test_data),
        'num_gist_tokens': num_gist_tokens,
        'compression_ratio': compression_ratio,
        'avg_full_tokens': avg_full_tokens,
        'avg_gist_tokens': avg_gist_tokens,
        'rouge_scores': {
            'full_text': rouge_full,
            'gist': rouge_gist,
            'truncated': rouge_truncated,
        },
        'relative_performance': {
            'gist_vs_full': rouge_gist['rougeL'] / rouge_full['rougeL'],
            'truncated_vs_full': rouge_truncated['rougeL'] / rouge_full['rougeL'],
        },
        'eval_time_seconds': elapsed,
    }

    results_file = output_dir / 'eval_results.json'
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"✓ Saved results to {results_file}")

    # Save sample outputs for inspection
    sample_outputs_file = output_dir / 'sample_outputs.json'
    sample_outputs = results[:10]  # Save first 10 for inspection
    with open(sample_outputs_file, 'w') as f:
        json.dump(sample_outputs, f, indent=2)
    print(f"✓ Saved sample outputs to {sample_outputs_file}")


if __name__ == "__main__":
    main()
