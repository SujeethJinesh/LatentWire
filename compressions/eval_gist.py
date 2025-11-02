#!/usr/bin/env python3
"""
Evaluate gist tokens on Alpaca dataset with baselines.

Compares:
1. Full text (positive control - full prompt)
2. Gist tokens (our method - compressed prompt)
3. Truncated text (negative control - truncate to gist token count)

Measures:
- ROUGE scores (generation quality)
- Compression ratio
- Output quality vs baselines
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gist_model(checkpoint_dir: str, device: str):
    """Load trained gist model."""
    print(f"Loading gist model from {checkpoint_dir}...")

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Load gist embedding
    gist_path = Path(checkpoint_dir) / "gist_embedding.pt"
    if gist_path.exists():
        gist_data = torch.load(gist_path, map_location=device)
        print(f"✓ Loaded gist embedding: {gist_data['num_gist_tokens']} tokens")
        return model, tokenizer, gist_data
    else:
        raise FileNotFoundError(f"No gist embedding found at {gist_path}")


def generate_output(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "cuda"):
    """Generate output from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for fair comparison
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (skip input)
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate_sample(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    reference_output: str,
    num_gist_tokens: int,
    max_new_tokens: int,
    device: str,
) -> Dict:
    """Evaluate one sample with all baselines."""

    # Format prompts
    if input_text:
        full_prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        gist_str = " ".join(["<GIST>"] * num_gist_tokens)
        gist_prompt = f"Instruction: {instruction}\n{gist_str}\nInput: {input_text}\nOutput:"
    else:
        full_prompt = f"Instruction: {instruction}\nOutput:"
        gist_str = " ".join(["<GIST>"] * num_gist_tokens)
        gist_prompt = f"Instruction: {instruction}\n{gist_str}\nOutput:"

    # Baseline 1: Full text (positive control)
    full_output = generate_output(model, tokenizer, full_prompt, max_new_tokens, device)

    # Our method: Gist tokens
    gist_output = generate_output(model, tokenizer, gist_prompt, max_new_tokens, device)

    # Baseline 2: Truncated text (negative control - truncate to num_gist_tokens)
    full_tokens = tokenizer(full_prompt, return_tensors="pt").input_ids[0]
    # Truncate to approximately num_gist_tokens (keep instruction + num_gist_tokens worth of input)
    truncate_len = min(len(full_tokens), 50 + num_gist_tokens * 10)  # Heuristic
    truncated_ids = full_tokens[:truncate_len]
    truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True) + "\nOutput:"
    truncated_output = generate_output(model, tokenizer, truncated_prompt, max_new_tokens, device)

    return {
        "full_text": full_output,
        "gist": gist_output,
        "truncated": truncated_output,
        "reference": reference_output,
        "prompts": {
            "full": full_prompt,
            "gist": gist_prompt,
            "truncated": truncated_prompt,
        }
    }


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

    # Load test data (held-out from Alpaca)
    print("Loading Alpaca test data...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    # Use last samples as test (assuming training used first samples)
    test_data = dataset.select(range(len(dataset) - args.samples, len(dataset)))
    print(f"✓ Loaded {len(test_data)} test samples\n")

    # Evaluate all samples
    print("Evaluating samples...")
    results = []
    full_outputs, gist_outputs, truncated_outputs, references = [], [], [], []

    start_time = time.time()

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        result = evaluate_sample(
            model=model,
            tokenizer=tokenizer,
            instruction=example['instruction'],
            input_text=example.get('input', ''),
            reference_output=example['output'],
            num_gist_tokens=num_gist_tokens,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

        results.append(result)
        full_outputs.append(result['full_text'])
        gist_outputs.append(result['gist'])
        truncated_outputs.append(result['truncated'])
        references.append(result['reference'])

    elapsed = time.time() - start_time

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
