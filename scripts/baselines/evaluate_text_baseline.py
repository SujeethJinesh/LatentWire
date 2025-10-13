"""
Text Baseline Evaluation - No Checkpoint Required

Evaluates LLM with full text prompts to establish upper bound performance.

Usage:
  python scripts/baselines/evaluate_text_baseline.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --samples 10000 \
    --dataset squad
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from latentwire.data import load_examples
from latentwire.core_utils import batch_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--max_new_tokens', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation (256=safe, 384=aggressive, 512+=OOM risk)')
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    start_time = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Text Baseline Evaluation")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}\n")

    # Load model
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map='auto' if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Use left padding for decoder-only models
    tokenizer.padding_side = 'left'

    print(f"Model loaded!\n")

    # Load data
    print(f"Loading {args.samples} examples from {args.dataset}...")
    examples = load_examples(dataset=args.dataset, split='validation', samples=args.samples, seed=42)
    print(f"Loaded {len(examples)} examples\n")

    # Evaluate with batching for speed
    print(f"Generating answers with full text prompts (batch_size={args.batch_size})...")
    predictions = []
    references = []

    for batch_start in range(0, len(examples), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(examples))
        batch = examples[batch_start:batch_end]

        if batch_start % 100 == 0:
            print(f"  {batch_start}/{len(examples)}...")

        # Tokenize batch
        sources = [ex['source'] for ex in batch]
        encoded = tokenizer(sources, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Generate
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode each output in batch
        for i, output in enumerate(outputs):
            # Find where input ends
            input_len = input_ids[i].shape[0]
            pred_tokens = output[input_len:]
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(batch[i]['answer'])

        # Clear cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute metrics
    print("\nComputing metrics...")
    em_score, f1_score = batch_metrics(predictions, references)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"  Model: {args.model_id}")
    print(f"  Samples: {len(examples)}")
    print(f"  EM: {em_score:.2%}")
    print(f"  F1: {f1_score:.2%}")
    print("="*80)

    # Save results
    results_dict = {
        'config': vars(args),
        'em': float(em_score),
        'f1': float(f1_score),
        'exact_match': float(em_score),  # Alias
        'f1_score': float(f1_score),      # Alias
        'num_examples': len(examples),
        'n_examples': len(examples),           # Alias
        'timestamp': datetime.now().isoformat(),
        'total_time_sec': time.time() - start_time,
    }

    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - start_time:.1f}s\n")


if __name__ == '__main__':
    main()
