"""
PCA Baseline - Linear Compression

Tests if linear compression (PCA) of text embeddings can achieve similar
performance to learned non-linear compression.

This answers: Do we need the learned encoder, or is PCA enough?

Usage:
  PYTHONPATH=. python scripts/baselines/pca_baseline.py \
    --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --samples 1000 \
    --latent_len 32 \
    --dataset squad
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from latentwire.data import load_examples
from latentwire.models import LMWrapper
from latentwire.metrics import compute_em_f1
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--latent_len', type=int, default=32, help='M: number of compressed tokens')
    parser.add_argument('--max_new_tokens', type=int, default=12)
    parser.add_argument('--save_dir', type=str, default='runs/baselines/pca')
    args = parser.parse_args()

    start_time = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}\n")

    # Load model
    print(f"Loading {args.llama_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map='auto' if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.llama_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded!\n")

    # Load data
    print(f"Loading {args.samples} examples from {args.dataset}...")
    examples = load_examples(dataset=args.dataset, split='validation', samples=args.samples, seed=42)
    print(f"Loaded {len(examples)} examples\n")

    # Extract text embeddings for PCA training
    print("Extracting text embeddings for PCA...")
    all_embeddings = []

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(examples)}...")

        encoded = tokenizer(ex['source'], return_tensors='pt', truncation=True, max_length=256)
        input_ids = encoded['input_ids'].to(device)

        # Get embeddings
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(input_ids)  # [1, seq_len, d_model]
            # Flatten to [seq_len, d_model] and move to CPU
            all_embeddings.append(embeddings[0].cpu().float().numpy())

    # Concatenate all embeddings
    all_embeddings_np = np.concatenate(all_embeddings, axis=0)  # [total_tokens, d_model]
    print(f"  Collected {all_embeddings_np.shape[0]} token embeddings")
    print(f"  Embedding dim: {all_embeddings_np.shape[1]}\n")

    # Fit PCA
    print(f"Fitting PCA to reduce {all_embeddings_np.shape[1]}D → {args.latent_len}D...")
    pca = PCA(n_components=args.latent_len)
    pca.fit(all_embeddings_np)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA fitted!")
    print(f"  Explained variance: {explained_var:.2%}")
    print(f"  First 5 components: {pca.explained_variance_ratio_[:5]}\n")

    # Save PCA
    pca_path = save_dir / f'pca_M{args.latent_len}.pkl'
    import pickle
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Saved PCA to {pca_path}\n")

    # Evaluate with PCA-compressed embeddings
    print("="*80)
    print("EVALUATION: Generating answers with PCA-compressed embeddings")
    print("="*80)
    print()

    predictions = []
    references = []

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  Example {i}/{len(examples)}...")

        # Tokenize source
        encoded = tokenizer(ex['source'], return_tensors='pt', truncation=True, max_length=256)
        input_ids = encoded['input_ids'].to(device)

        # Get text embeddings
        with torch.no_grad():
            text_embeds = model.get_input_embeddings()(input_ids)  # [1, seq_len, d_model]

            # Compress with PCA
            text_embeds_np = text_embeds[0].cpu().float().numpy()  # [seq_len, d_model]
            compressed = pca.transform(text_embeds_np)  # [seq_len, M]

            # Decompress back
            reconstructed = pca.inverse_transform(compressed)  # [seq_len, d_model]

            # Convert back to torch and to device
            pca_embeds = torch.from_numpy(reconstructed).to(device).to(text_embeds.dtype)
            pca_embeds = pca_embeds.unsqueeze(0)  # [1, seq_len, d_model]

            # Generate with reconstructed embeddings
            outputs = model.generate(
                inputs_embeds=pca_embeds,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode prediction
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(ex['target'])

    # Compute metrics
    print("\nComputing metrics...")
    results = compute_em_f1(predictions, references)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"  Samples: {len(examples)}")
    print(f"  Latent dim (M): {args.latent_len}")
    print(f"  PCA explained variance: {explained_var:.2%}")
    print(f"  EM: {results['em']:.2%}")
    print(f"  F1: {results['f1']:.2%}")
    print("="*80)

    # Save results
    results_dict = {
        'config': vars(args),
        'pca_explained_variance': float(explained_var),
        'pca_first_5_components': pca.explained_variance_ratio_[:5].tolist(),
        'em': float(results['em']),
        'f1': float(results['f1']),
        'num_examples': len(examples),
        'timestamp': datetime.now().isoformat(),
        'total_time_sec': time.time() - start_time,
    }

    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - start_time:.1f}s\n")

    print("INTERPRETATION:")
    print("  If PCA F1 ≈ Text baseline → Linear compression is sufficient")
    print("  If PCA F1 << Text baseline → Need learned non-linear encoder")
    print()


if __name__ == '__main__':
    main()
