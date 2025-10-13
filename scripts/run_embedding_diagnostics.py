"""
Comprehensive embedding diagnostics script.

Answers critical questions:
1. Where did "115-120× magnitude mismatch" come from?
2. What's actually different between text and learned embeddings?
3. Why does RMS scaling destroy everything?
4. What properties does the LLM actually need?

Usage:
  python scripts/run_embedding_diagnostics.py --checkpoint path/to/checkpoint

  Or without checkpoint (uses synthetic bad embeddings):
  python scripts/run_embedding_diagnostics.py --no-checkpoint
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data import load_examples
from latentwire.embed_experiments import get_vocab_embedding_stats


def compute_per_token_rms(embeddings):
    """Compute RMS per token. Returns [batch, seq_len]."""
    return embeddings.pow(2).mean(dim=-1).sqrt()


def compute_per_dim_stats(embeddings):
    """Compute mean/std per dimension. Returns dict."""
    B, S, D = embeddings.shape
    flat = embeddings.view(-1, D)  # [B*S, D]
    return {
        'per_dim_mean': flat.mean(dim=0),  # [D]
        'per_dim_std': flat.std(dim=0),    # [D]
        'per_dim_min': flat.min(dim=0).values,
        'per_dim_max': flat.max(dim=0).values,
    }


def compute_nearest_vocab_cosine(embeddings, vocab_embeddings):
    """Compute cosine similarity to nearest vocab token for each position."""
    B, S, D = embeddings.shape
    flat = embeddings.view(-1, D)  # [B*S, D]

    # Normalize
    flat_norm = F.normalize(flat, dim=-1)
    vocab_norm = F.normalize(vocab_embeddings, dim=-1)

    # Cosine similarity to all vocab
    sim = flat_norm @ vocab_norm.T  # [B*S, vocab_size]

    # Max similarity per position
    max_sim = sim.max(dim=-1).values  # [B*S]

    return max_sim.view(B, S)


def apply_rms_scaling(embeddings, target_rms):
    """Apply RMS scaling (what failed in our experiments)."""
    current_rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt()
    scale = target_rms / (current_rms + 1e-8)
    return embeddings * scale


def apply_batch_dist(embeddings, target_mean, target_std):
    """Apply batch distribution matching (what helped slightly)."""
    current_mean = embeddings.mean()
    current_std = embeddings.std()

    # Standardize
    normed = (embeddings - current_mean) / (current_std + 1e-8)

    # Rescale
    return normed * target_std + target_mean


def compute_covariance_spectrum(embeddings):
    """Compute eigenvalues of covariance matrix (measures structure)."""
    B, S, D = embeddings.shape
    flat = embeddings.view(-1, D)  # [B*S, D]

    # Center
    centered = flat - flat.mean(dim=0, keepdim=True)

    # Covariance
    cov = (centered.T @ centered) / (flat.size(0) - 1)

    # Eigenvalues
    try:
        eigenvalues = torch.linalg.eigvalsh(cov)
        return eigenvalues.cpu()
    except:
        return None


def analyze_embeddings(embeddings, name, vocab_stats, device):
    """Comprehensive analysis of embeddings."""
    B, S, D = embeddings.shape

    stats = {
        'name': name,
        'shape': [B, S, D],
    }

    # 1. Per-token RMS distribution
    per_token_rms = compute_per_token_rms(embeddings)  # [B, S]
    stats['per_token_rms'] = {
        'min': float(per_token_rms.min()),
        'max': float(per_token_rms.max()),
        'mean': float(per_token_rms.mean()),
        'std': float(per_token_rms.std()),
        'median': float(per_token_rms.median()),
        'q25': float(per_token_rms.quantile(0.25)),
        'q75': float(per_token_rms.quantile(0.75)),
    }

    # 2. Overall RMS
    stats['overall_rms'] = float(embeddings.pow(2).mean().sqrt())

    # 3. Per-dimension statistics
    dim_stats = compute_per_dim_stats(embeddings)
    stats['per_dim'] = {
        'mean_of_means': float(dim_stats['per_dim_mean'].mean()),
        'std_of_means': float(dim_stats['per_dim_mean'].std()),
        'mean_of_stds': float(dim_stats['per_dim_std'].mean()),
        'std_of_stds': float(dim_stats['per_dim_std'].std()),
    }

    # 4. Nearest vocab token cosine similarity
    nearest_cos = compute_nearest_vocab_cosine(embeddings, vocab_stats['embeddings'].to(device))
    stats['nearest_vocab_cosine'] = {
        'min': float(nearest_cos.min()),
        'max': float(nearest_cos.max()),
        'mean': float(nearest_cos.mean()),
        'std': float(nearest_cos.std()),
        'median': float(nearest_cos.median()),
    }

    # 5. Covariance spectrum (top 10 eigenvalues)
    spectrum = compute_covariance_spectrum(embeddings)
    if spectrum is not None:
        top10 = spectrum[-10:].numpy().tolist()  # Largest 10
        stats['covariance_spectrum_top10'] = [float(x) for x in top10]

    # 6. Mean and std (for batch_dist comparison)
    stats['global_mean'] = float(embeddings.mean())
    stats['global_std'] = float(embeddings.std())

    return stats


def test_transforms(embeddings, name, vocab_stats, device):
    """Test various transforms and measure their effect."""
    results = {}

    # 1. RMS scaling to vocab RMS
    target_rms = vocab_stats['rms']
    rms_scaled = apply_rms_scaling(embeddings, target_rms)

    results['rms_scaled'] = {
        'target_rms': target_rms,
        'per_token_rms_after': {
            'min': float(compute_per_token_rms(rms_scaled).min()),
            'max': float(compute_per_token_rms(rms_scaled).max()),
            'mean': float(compute_per_token_rms(rms_scaled).mean()),
            'std': float(compute_per_token_rms(rms_scaled).std()),
        },
        'overall_rms_after': float(rms_scaled.pow(2).mean().sqrt()),
        'nearest_vocab_cosine_after': {
            'mean': float(compute_nearest_vocab_cosine(rms_scaled, vocab_stats['embeddings'].to(device)).mean()),
        },
    }

    # 2. Batch distribution matching
    batch_dist_scaled = apply_batch_dist(embeddings, vocab_stats['mean'], vocab_stats['std'])

    results['batch_dist'] = {
        'target_mean': vocab_stats['mean'],
        'target_std': vocab_stats['std'],
        'mean_after': float(batch_dist_scaled.mean()),
        'std_after': float(batch_dist_scaled.std()),
        'per_token_rms_after': {
            'mean': float(compute_per_token_rms(batch_dist_scaled).mean()),
            'std': float(compute_per_token_rms(batch_dist_scaled).std()),
        },
        'nearest_vocab_cosine_after': {
            'mean': float(compute_nearest_vocab_cosine(batch_dist_scaled, vocab_stats['embeddings'].to(device)).mean()),
        },
    }

    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained checkpoint (encoder+adapter)')
    parser.add_argument('--no-checkpoint', action='store_true', help='Run without checkpoint (synthetic bad embeddings)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for embedding extraction')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--output_dir', type=str, default='runs/embed_diagnostics')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start timing
    start_time = time.time()
    experiment_log = {
        'start_time': datetime.now().isoformat(),
        'config': vars(args),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'num_gpus': torch.cuda.device_count(),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Samples: {args.samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_dir}\n")

    # Load model
    t0 = time.time()
    print(f"Loading model: {args.model_id}...")
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

    model_load_time = time.time() - t0
    print(f"Model loaded! ({model_load_time:.2f}s)")
    experiment_log['model_load_time_sec'] = model_load_time

    # Cache vocab stats
    t0 = time.time()
    print("\nCaching vocabulary statistics...")
    vocab_stats = get_vocab_embedding_stats(model)
    vocab_cache_time = time.time() - t0
    print(f"  Vocab RMS: {vocab_stats['rms']:.4f}")
    print(f"  Vocab mean: {vocab_stats['mean']:.6f}")
    print(f"  Vocab std: {vocab_stats['std']:.4f}")
    print(f"  Time: {vocab_cache_time:.2f}s")
    experiment_log['vocab_cache_time_sec'] = vocab_cache_time

    # Save vocab stats
    vocab_stats_save = {
        'rms': float(vocab_stats['rms']),
        'mean': float(vocab_stats['mean']),
        'std': float(vocab_stats['std']),
        'vocab_size': vocab_stats['embeddings'].shape[0],
        'embedding_dim': vocab_stats['embeddings'].shape[1],
    }
    with open(output_dir / 'vocab_stats.json', 'w') as f:
        json.dump(vocab_stats_save, f, indent=2)
    print(f"  Saved vocab stats to {output_dir / 'vocab_stats.json'}")

    # Load data
    t0 = time.time()
    print(f"\nLoading {args.samples} examples from {args.dataset}...")
    examples = load_examples(dataset=args.dataset, split='validation', samples=args.samples, seed=42)
    data_load_time = time.time() - t0
    print(f"  Loaded {len(examples)} examples ({data_load_time:.2f}s)")
    experiment_log['data_load_time_sec'] = data_load_time
    experiment_log['num_examples'] = len(examples)

    # Collect text embeddings (batch processing)
    print("\n" + "="*80)
    print("ANALYZING TEXT EMBEDDINGS (Ground Truth)")
    print("="*80)

    t0 = time.time()
    text_embeddings_list = []
    total_tokens = 0
    batch_size = args.batch_size
    num_batches = (len(examples) + batch_size - 1) // batch_size

    print(f"Processing {len(examples)} examples in {num_batches} batches of size {batch_size}...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(examples))
        batch = examples[start_idx:end_idx]

        if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
            print(f"  Batch {batch_idx+1}/{num_batches} (examples {start_idx}-{end_idx})...")

        # Extract source texts
        source_texts = [ex['source'] for ex in batch]

        # Tokenize with padding for batch processing
        encoded = tokenizer(
            source_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        input_ids = encoded['input_ids'].to(device)  # [batch_size, max_seq_len]
        attention_mask = encoded['attention_mask'].to(device)  # [batch_size, max_seq_len]

        # Get embeddings for entire batch
        batch_embeddings = model.get_input_embeddings()(input_ids)  # [batch_size, max_seq_len, d_model]

        # Extract only non-padding tokens
        for i in range(batch_embeddings.shape[0]):
            mask = attention_mask[i].bool()  # [max_seq_len]
            valid_embeddings = batch_embeddings[i][mask]  # [actual_seq_len, d_model]
            text_embeddings_list.append(valid_embeddings)
            total_tokens += valid_embeddings.shape[0]

    # Concatenate along sequence dimension
    text_embeddings = torch.cat(text_embeddings_list, dim=0)  # [total_tokens, d_model]
    # Add batch dimension back for analysis functions
    text_embeddings = text_embeddings.unsqueeze(0)  # [1, total_tokens, d_model]

    text_collect_time = time.time() - t0
    print(f"\nCollected text embeddings: {text_embeddings.shape}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/example: {total_tokens/len(examples):.1f}")
    print(f"  Time: {text_collect_time:.2f}s ({total_tokens/text_collect_time:.0f} tokens/sec)")
    print(f"  Throughput: {len(examples)/text_collect_time:.1f} examples/sec")
    experiment_log['text_collect_time_sec'] = text_collect_time
    experiment_log['total_tokens'] = total_tokens
    experiment_log['avg_tokens_per_example'] = total_tokens / len(examples)
    experiment_log['examples_per_sec'] = len(examples) / text_collect_time

    # Analyze text embeddings
    t0 = time.time()
    text_stats = analyze_embeddings(text_embeddings, "text_embeddings", vocab_stats, device)
    text_analysis_time = time.time() - t0

    print("\nText Embedding Statistics:")
    print(f"  Per-token RMS: min={text_stats['per_token_rms']['min']:.4f}, "
          f"max={text_stats['per_token_rms']['max']:.4f}, "
          f"mean={text_stats['per_token_rms']['mean']:.4f}, "
          f"std={text_stats['per_token_rms']['std']:.4f}")
    print(f"  Overall RMS: {text_stats['overall_rms']:.4f}")
    print(f"  Nearest vocab cosine: mean={text_stats['nearest_vocab_cosine']['mean']:.4f}")
    print(f"  Analysis time: {text_analysis_time:.2f}s")
    experiment_log['text_analysis_time_sec'] = text_analysis_time

    # Save text embeddings analysis
    with open(output_dir / 'text_embeddings_analysis.json', 'w') as f:
        json.dump(text_stats, f, indent=2)
    print(f"  Saved analysis to {output_dir / 'text_embeddings_analysis.json'}")

    # Test transforms on text embeddings
    print("\nTesting transforms on text embeddings...")
    t0 = time.time()
    text_transform_results = test_transforms(text_embeddings, "text_embeddings", vocab_stats, device)
    text_transform_time = time.time() - t0
    print(f"  Transform testing time: {text_transform_time:.2f}s")
    experiment_log['text_transform_time_sec'] = text_transform_time

    # Save transform results
    with open(output_dir / 'text_transforms.json', 'w') as f:
        json.dump(text_transform_results, f, indent=2)
    print(f"  Saved transforms to {output_dir / 'text_transforms.json'}")

    # Handle learned embeddings
    learned_stats = None
    learned_transform_results = None

    if args.checkpoint:
        print("\n" + "="*80)
        print("ANALYZING LEARNED EMBEDDINGS (from checkpoint)")
        print("="*80)
        print(f"\nCheckpoint: {args.checkpoint}")
        # TODO: Load encoder+adapter from checkpoint and generate learned embeddings
        print("WARNING: Checkpoint loading not implemented yet. Use --no-checkpoint for synthetic test.")

    elif args.no_checkpoint:
        print("\n" + "="*80)
        print("ANALYZING SYNTHETIC 'BAD' EMBEDDINGS")
        print("="*80)
        print("\nGenerating synthetic embeddings that simulate common failure modes...")

        t0 = time.time()
        # Create synthetic "bad" embeddings with known issues
        D = text_embeddings.size(-1)

        # Mode 1: 100× too large magnitude (simulates our original hypothesis)
        learned_embeddings = text_embeddings * 100.0
        learned_embeddings = learned_embeddings + torch.randn_like(learned_embeddings) * 10.0  # Add noise

        learned_gen_time = time.time() - t0
        print(f"Synthetic embeddings: {learned_embeddings.shape}")
        print(f"  Mode: 100× magnitude + noise")
        print(f"  Generation time: {learned_gen_time:.2f}s")
        experiment_log['learned_gen_time_sec'] = learned_gen_time

        # Analyze synthetic embeddings
        t0 = time.time()
        learned_stats = analyze_embeddings(learned_embeddings, "synthetic_100x", vocab_stats, device)
        learned_analysis_time = time.time() - t0

        print("\nSynthetic Embedding Statistics:")
        print(f"  Per-token RMS: min={learned_stats['per_token_rms']['min']:.4f}, "
              f"max={learned_stats['per_token_rms']['max']:.4f}, "
              f"mean={learned_stats['per_token_rms']['mean']:.4f}, "
              f"std={learned_stats['per_token_rms']['std']:.4f}")
        print(f"  Overall RMS: {learned_stats['overall_rms']:.4f}")
        print(f"  Nearest vocab cosine: mean={learned_stats['nearest_vocab_cosine']['mean']:.4f}")
        print(f"  Analysis time: {learned_analysis_time:.2f}s")
        experiment_log['learned_analysis_time_sec'] = learned_analysis_time

        # Save learned embeddings analysis
        with open(output_dir / 'learned_embeddings_analysis.json', 'w') as f:
            json.dump(learned_stats, f, indent=2)
        print(f"  Saved analysis to {output_dir / 'learned_embeddings_analysis.json'}")

        # Test transforms
        print("\nTesting transforms on synthetic embeddings...")
        t0 = time.time()
        learned_transform_results = test_transforms(learned_embeddings, "synthetic_100x", vocab_stats, device)
        learned_transform_time = time.time() - t0
        print(f"  Transform testing time: {learned_transform_time:.2f}s")
        experiment_log['learned_transform_time_sec'] = learned_transform_time

        # Save learned transform results
        with open(output_dir / 'learned_transforms.json', 'w') as f:
            json.dump(learned_transform_results, f, indent=2)
        print(f"  Saved transforms to {output_dir / 'learned_transforms.json'}")

    # Comparison
    if learned_stats:
        print("\n" + "="*80)
        print("COMPARISON: Text vs Learned")
        print("="*80)

        print("\n1. Per-token RMS:")
        print(f"  Text:    mean={text_stats['per_token_rms']['mean']:.4f}, std={text_stats['per_token_rms']['std']:.4f}")
        print(f"  Learned: mean={learned_stats['per_token_rms']['mean']:.4f}, std={learned_stats['per_token_rms']['std']:.4f}")
        print(f"  Ratio: {learned_stats['per_token_rms']['mean'] / text_stats['per_token_rms']['mean']:.2f}×")

        print("\n2. Overall RMS:")
        print(f"  Text:    {text_stats['overall_rms']:.4f}")
        print(f"  Learned: {learned_stats['overall_rms']:.4f}")
        print(f"  Ratio: {learned_stats['overall_rms'] / text_stats['overall_rms']:.2f}×")

        print("\n3. Nearest vocab cosine:")
        print(f"  Text:    {text_stats['nearest_vocab_cosine']['mean']:.4f}")
        print(f"  Learned: {learned_stats['nearest_vocab_cosine']['mean']:.4f}")
        print(f"  Diff: {text_stats['nearest_vocab_cosine']['mean'] - learned_stats['nearest_vocab_cosine']['mean']:.4f}")

        print("\n4. Per-token RMS variation:")
        text_rms_cv = text_stats['per_token_rms']['std'] / text_stats['per_token_rms']['mean']
        learned_rms_cv = learned_stats['per_token_rms']['std'] / learned_stats['per_token_rms']['mean']
        print(f"  Text CV (std/mean):    {text_rms_cv:.4f}")
        print(f"  Learned CV (std/mean): {learned_rms_cv:.4f}")
        print(f"  → {'PRESERVED' if abs(text_rms_cv - learned_rms_cv) < 0.1 else 'DESTROYED'}")

    # Finalize experiment log
    total_time = time.time() - start_time
    experiment_log['end_time'] = datetime.now().isoformat()
    experiment_log['total_time_sec'] = total_time

    # Save comprehensive results
    results = {
        'experiment_log': experiment_log,
        'vocab_stats': {
            'rms': float(vocab_stats['rms']),
            'mean': float(vocab_stats['mean']),
            'std': float(vocab_stats['std']),
        },
        'text_embeddings': text_stats,
        'text_transforms': text_transform_results,
    }

    if learned_stats:
        results['learned_embeddings'] = learned_stats
        results['learned_transforms'] = learned_transform_results

    output_path = output_dir / 'diagnostics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save experiment log separately for easy access
    with open(output_dir / 'experiment_log.json', 'w') as f:
        json.dump(experiment_log, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - diagnostics.json (comprehensive results)")
    print(f"  - experiment_log.json (timing and config)")
    print(f"  - vocab_stats.json")
    print(f"  - text_embeddings_analysis.json")
    print(f"  - text_transforms.json")
    if learned_stats:
        print(f"  - learned_embeddings_analysis.json")
        print(f"  - learned_transforms.json")
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"{'='*80}\n")

    # Key insights
    print("KEY INSIGHTS:")
    print()
    print("1. RMS Scaling Effect:")
    print(f"   Before: per-token RMS std = {text_stats['per_token_rms']['std']:.4f}")
    print(f"   After:  per-token RMS std = {text_transform_results['rms_scaled']['per_token_rms_after']['std']:.4f}")
    print(f"   → RMS scaling {'PRESERVES' if text_transform_results['rms_scaled']['per_token_rms_after']['std'] > 0.1 else 'DESTROYS'} per-token variation")
    print()

    print("2. Batch Distribution Effect:")
    print(f"   Per-token RMS variation after: {text_transform_results['batch_dist']['per_token_rms_after']['std']:.4f}")
    print(f"   → Batch dist {'PRESERVES' if text_transform_results['batch_dist']['per_token_rms_after']['std'] > text_stats['per_token_rms']['std'] * 0.5 else 'REDUCES'} variation")
    print()

    if learned_stats:
        print("3. Why Learned Embeddings Fail:")
        magnitude_ratio = learned_stats['overall_rms'] / text_stats['overall_rms']
        cosine_drop = text_stats['nearest_vocab_cosine']['mean'] - learned_stats['nearest_vocab_cosine']['mean']

        if magnitude_ratio > 10:
            print(f"   ⚠️  Magnitude is {magnitude_ratio:.1f}× too large")
        if cosine_drop > 0.2:
            print(f"   ⚠️  Direction drift: {cosine_drop:.3f} drop in vocab alignment")
        if learned_rms_cv > text_rms_cv * 2 or learned_rms_cv < text_rms_cv * 0.5:
            print(f"   ⚠️  Per-token variation is wrong: CV={learned_rms_cv:.4f} vs {text_rms_cv:.4f}")


if __name__ == '__main__':
    main()
