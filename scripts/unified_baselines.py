#!/usr/bin/env python3
"""
Unified Baseline Evaluation Script

Combines text baseline, token budget baseline, and PCA baseline evaluations
into a single script with a common interface.

Usage:
  # Text baseline (full prompt, upper bound)
  python scripts/unified_baselines.py --baseline text \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --samples 10000 --dataset squad --save_dir runs/baselines/text

  # Token budget baseline (truncated to M tokens)
  python scripts/unified_baselines.py --baseline token_budget \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --token_budget 32 --samples 10000 --dataset squad --save_dir runs/baselines/token_budget

  # PCA baseline (linear compression)
  python scripts/unified_baselines.py --baseline pca \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --latent_len 32 --samples 1000 --dataset squad --save_dir runs/baselines/pca
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data import load_examples
from latentwire.core_utils import batch_metrics


@torch.no_grad()
def evaluate_text_baseline(
    model,
    tokenizer,
    examples,
    max_new_tokens: int = 12,
    batch_size: int = 256,
    device: torch.device = None
) -> dict:
    """
    Evaluate with full text prompts (upper bound baseline).

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        examples: List of examples with 'source' and 'answer' keys
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for evaluation
        device: Device to use

    Returns:
        Dictionary with EM/F1 scores and predictions
    """
    if device is None:
        device = next(model.parameters()).device

    predictions = []
    references = []

    for batch_start in range(0, len(examples), batch_size):
        batch_end = min(batch_start + batch_size, len(examples))
        batch = examples[batch_start:batch_end]

        if batch_start % 500 == 0:
            print(f"  Processing {batch_start}/{len(examples)}...")

        sources = [ex['source'] for ex in batch]
        encoded = tokenizer(sources, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        for i, output in enumerate(outputs):
            input_len = input_ids[i].shape[0]
            pred_tokens = output[input_len:]
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(batch[i]['answer'])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    em_score, f1_score = batch_metrics(predictions, references)

    return {
        'em': float(em_score),
        'f1': float(f1_score),
        'predictions': predictions,
        'references': references
    }


@torch.no_grad()
def evaluate_token_budget_baseline(
    model,
    tokenizer,
    examples,
    token_budget: int,
    max_new_tokens: int = 12,
    batch_size: int = 256,
    device: torch.device = None
) -> dict:
    """
    Evaluate with text truncated to token_budget tokens (fair comparison baseline).

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        examples: List of examples with 'source' and 'answer' keys
        token_budget: Maximum number of tokens to keep (M)
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for evaluation
        device: Device to use

    Returns:
        Dictionary with EM/F1 scores and predictions
    """
    if device is None:
        device = next(model.parameters()).device

    predictions = []
    references = []

    for batch_start in range(0, len(examples), batch_size):
        batch_end = min(batch_start + batch_size, len(examples))
        batch = examples[batch_start:batch_end]

        if batch_start % 500 == 0:
            print(f"  Processing {batch_start}/{len(examples)}...")

        sources = [ex['source'] for ex in batch]
        # Truncate to token budget
        encoded = tokenizer(sources, return_tensors='pt', truncation=True, max_length=token_budget, padding=True)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        for i, output in enumerate(outputs):
            input_len = input_ids[i].shape[0]
            pred_tokens = output[input_len:]
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(batch[i]['answer'])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    em_score, f1_score = batch_metrics(predictions, references)

    return {
        'em': float(em_score),
        'f1': float(f1_score),
        'token_budget': token_budget,
        'predictions': predictions,
        'references': references
    }


@torch.no_grad()
def evaluate_pca_baseline(
    model,
    tokenizer,
    examples,
    latent_len: int = 32,
    max_new_tokens: int = 12,
    device: torch.device = None
) -> dict:
    """
    Evaluate with PCA-compressed embeddings (linear compression baseline).

    Tests whether linear compression (PCA) of text embeddings can achieve
    similar performance to learned non-linear compression.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        examples: List of examples with 'source' and 'answer' keys
        latent_len: Number of PCA components (M)
        max_new_tokens: Maximum tokens to generate
        device: Device to use

    Returns:
        Dictionary with EM/F1 scores, PCA statistics, and predictions
    """
    from sklearn.decomposition import IncrementalPCA

    if device is None:
        device = next(model.parameters()).device

    # Get embedding layer device
    embed_device = model.get_input_embeddings().weight.device
    print(f"Embedding layer device: {embed_device}")

    # Fit IncrementalPCA in batches
    print("Fitting IncrementalPCA (batched GPU extraction)...")
    pca = IncrementalPCA(n_components=latent_len, batch_size=1000)

    gpu_batch_size = 128
    pca_fit_every = 500
    total_tokens = 0
    embedding_dim = None
    accumulated_embeddings = []

    for batch_start in range(0, len(examples), gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, len(examples))
        batch = examples[batch_start:batch_end]

        if batch_start % 500 == 0:
            print(f"  Processing examples {batch_start}/{len(examples)} for PCA fitting...")

        sources = [ex['source'] for ex in batch]
        encoded = tokenizer(sources, return_tensors='pt', truncation=True, max_length=256, padding=True)
        input_ids = encoded['input_ids'].to(embed_device)

        embeddings = model.get_input_embeddings()(input_ids)
        embeddings_np = embeddings.cpu().float().numpy()

        for i in range(embeddings_np.shape[0]):
            seq_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
            accumulated_embeddings.append(embeddings_np[i, :seq_len, :])

        embedding_dim = embeddings_np.shape[2]

        if len(accumulated_embeddings) >= pca_fit_every or batch_end == len(examples):
            batch_embeddings_np = np.concatenate(accumulated_embeddings, axis=0)
            total_tokens += batch_embeddings_np.shape[0]
            print(f"    Fitting PCA on {batch_embeddings_np.shape[0]} tokens...")
            pca.partial_fit(batch_embeddings_np)
            accumulated_embeddings = []
            del batch_embeddings_np

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"  Total tokens: {total_tokens}")
    print(f"  Embedding dim: {embedding_dim}")

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA: {embedding_dim}D -> {latent_len}D")
    print(f"  Explained variance: {explained_var:.2%}")
    print(f"  First 5 components: {pca.explained_variance_ratio_[:5]}")

    # Evaluate with PCA-compressed embeddings
    print("\nEvaluating with PCA-compressed embeddings...")
    predictions = []
    references = []

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  Example {i}/{len(examples)}...")

        encoded = tokenizer(ex['source'], return_tensors='pt', truncation=True, max_length=256)
        input_ids = encoded['input_ids'].to(embed_device)

        text_embeds = model.get_input_embeddings()(input_ids)
        text_embeds_np = text_embeds[0].cpu().float().numpy()

        # Compress and decompress
        compressed = pca.transform(text_embeds_np)
        reconstructed = pca.inverse_transform(compressed)

        pca_embeds = torch.from_numpy(reconstructed).to(embed_device).to(text_embeds.dtype)
        pca_embeds = pca_embeds.unsqueeze(0)

        outputs = model.generate(
            inputs_embeds=pca_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred_text)
        references.append(ex['answer'])

    em_score, f1_score = batch_metrics(predictions, references)

    return {
        'em': float(em_score),
        'f1': float(f1_score),
        'latent_len': latent_len,
        'pca_explained_variance': float(explained_var),
        'pca_first_5_components': pca.explained_variance_ratio_[:5].tolist(),
        'predictions': predictions,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description='Unified Baseline Evaluation')
    parser.add_argument('--baseline', type=str, required=True,
                       choices=['text', 'token_budget', 'pca'],
                       help='Baseline type to evaluate')
    parser.add_argument('--model_id', type=str, required=True,
                       help='HuggingFace model ID')
    parser.add_argument('--dataset', type=str, default='squad',
                       help='Dataset to evaluate on')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=12,
                       help='Maximum tokens to generate')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save results')

    # Token budget specific
    parser.add_argument('--token_budget', type=int, default=32,
                       help='Token budget for token_budget baseline')

    # PCA specific
    parser.add_argument('--latent_len', type=int, default=32,
                       help='Latent length (M) for PCA baseline')

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*80}")
    print(f"Baseline Evaluation: {args.baseline.upper()}")
    print(f"{'='*80}")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}")
    print()

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
    tokenizer.padding_side = 'left'

    print(f"Model loaded!\n")

    # Load data
    print(f"Loading {args.samples} examples from {args.dataset}...")
    examples = load_examples(dataset=args.dataset, split='validation', samples=args.samples, seed=42)
    print(f"Loaded {len(examples)} examples\n")

    # Run evaluation based on baseline type
    if args.baseline == 'text':
        print("Evaluating with full text prompts...")
        results = evaluate_text_baseline(
            model, tokenizer, examples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=device
        )

    elif args.baseline == 'token_budget':
        print(f"Evaluating with text truncated to {args.token_budget} tokens...")
        results = evaluate_token_budget_baseline(
            model, tokenizer, examples,
            token_budget=args.token_budget,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=device
        )

    elif args.baseline == 'pca':
        print(f"Evaluating with PCA compression (M={args.latent_len})...")
        results = evaluate_pca_baseline(
            model, tokenizer, examples,
            latent_len=args.latent_len,
            max_new_tokens=args.max_new_tokens,
            device=device
        )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"  Baseline: {args.baseline}")
    print(f"  Model: {args.model_id}")
    print(f"  Samples: {len(examples)}")
    print(f"  EM: {results['em']:.2%}")
    print(f"  F1: {results['f1']:.2%}")
    if 'token_budget' in results:
        print(f"  Token Budget: {results['token_budget']}")
    if 'latent_len' in results:
        print(f"  Latent Length: {results['latent_len']}")
        print(f"  PCA Explained Variance: {results['pca_explained_variance']:.2%}")
    print("="*80)

    # Save results
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    # Prepare results dict (excluding large arrays)
    results_dict = {
        'config': vars(args),
        'baseline': args.baseline,
        'em': results['em'],
        'f1': results['f1'],
        'exact_match': results['em'],  # Alias
        'f1_score': results['f1'],      # Alias
        'num_examples': len(examples),
        'timestamp': datetime.now().isoformat(),
        'total_time_sec': total_time,
    }

    # Add baseline-specific fields
    if 'token_budget' in results:
        results_dict['token_budget'] = results['token_budget']
    if 'latent_len' in results:
        results_dict['latent_len'] = results['latent_len']
        results_dict['pca_explained_variance'] = results['pca_explained_variance']
        results_dict['pca_first_5_components'] = results['pca_first_5_components']

    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total time: {total_time:.1f}s\n")

    if args.baseline == 'pca':
        print("INTERPRETATION:")
        print("  If PCA F1 ~= Text baseline -> Linear compression is sufficient")
        print("  If PCA F1 << Text baseline -> Need learned non-linear encoder")
        print()


if __name__ == '__main__':
    main()
