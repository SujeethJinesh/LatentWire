"""
Simplified embedding experiment sweep with REALISTIC performance.
Tests only lightweight transformations. Uses real text embeddings to isolate transform effects.

IMPORTANT: This script does sequential generation (no batching) so runtime is proportional
to num_samples × num_experiments. With 100 samples × ~10 configs on 8B model:
- GPU: ~30-60 minutes
- CPU: ~2-4 hours
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data import load_examples
from latentwire.core_utils import em, f1  # Use existing metrics
from latentwire.embed_experiments import (
    get_vocab_embedding_stats,
    StatisticalMatcher,
)


def apply_rms_transform(embeddings, target_rms, device):
    """Apply per-example RMS scaling. Lightweight, no vocab lookups."""
    B, S, D = embeddings.shape
    current_rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt()
    target = torch.ones(B, S, 1, device=device) * target_rms
    scale = target / (current_rms + 1e-8)
    return embeddings * scale


def apply_batch_dist_transform(embeddings, vocab_mean, vocab_std):
    """Apply batch-level standardization. Lightweight."""
    current_mean = embeddings.mean()
    current_std = embeddings.std()
    embeddings = (embeddings - current_mean) / (current_std + 1e-8)
    embeddings = embeddings * vocab_std + vocab_mean
    return embeddings


@torch.no_grad()
def run_experiment(experiment_name, transform_config, model, tokenizer, examples, device, vocab_stats, max_samples=100):
    """
    Run experiment: Get text embeddings, apply transform, generate, evaluate.

    NOTE: Sequential generation, no batching. Runtime = num_samples × generation_time.
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Config: {transform_config}")
    print(f"{'='*80}")

    model.eval()

    predictions = []
    references = []
    empty_count = 0
    start_time = time.time()

    for idx, example in enumerate(examples[:max_samples]):
        if idx > 0 and idx % 25 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / idx
            remaining = rate * (max_samples - idx)
            print(f"  Progress: {idx}/{max_samples} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

        source_text = example['source']
        answer = example['answer']

        # Tokenize source
        encoded = tokenizer(
            source_text,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=256
        )
        input_ids = encoded['input_ids'].to(device)

        # Get embeddings
        source_embeddings = model.get_input_embeddings()(input_ids)

        # Apply transformation (using cached vocab_stats)
        transform_type = transform_config['type']

        if transform_type == 'baseline':
            transformed_embeddings = source_embeddings

        elif transform_type == 'rms_matching':
            target_scale = transform_config.get('target_scale', 1.0)
            target_rms = vocab_stats['rms'] * target_scale
            transformed_embeddings = apply_rms_transform(source_embeddings, target_rms, device)

        elif transform_type == 'batch_distribution':
            transformed_embeddings = apply_batch_dist_transform(
                source_embeddings,
                vocab_stats['mean'],
                vocab_stats['std']
            )
        else:
            transformed_embeddings = source_embeddings

        # Add answer anchor
        anchor_text = "Answer: "
        anchor_encoded = tokenizer(anchor_text, return_tensors='pt', add_special_tokens=False)
        anchor_ids = anchor_encoded['input_ids'].to(device)
        anchor_embeddings = model.get_input_embeddings()(anchor_ids)

        # Concatenate
        full_embeddings = torch.cat([transformed_embeddings, anchor_embeddings], dim=1)
        attention_mask = torch.ones(full_embeddings.size()[:2], dtype=torch.long, device=device)

        # Generate
        try:
            outputs = model.generate(
                inputs_embeds=full_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer part
            if "Answer:" in generated_text:
                generated_text = generated_text.split("Answer:")[-1].strip()

        except Exception as e:
            print(f"    Generation error at idx {idx}: {e}")
            generated_text = ""

        if not generated_text:
            empty_count += 1

        predictions.append(generated_text)
        references.append(answer)

    elapsed = time.time() - start_time

    # Compute metrics using core_utils
    f1_scores = [f1(pred, ref) for pred, ref in zip(predictions, references)]
    em_scores = [em(pred, ref) for pred, ref in zip(predictions, references)]

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    empty_rate = empty_count / len(predictions) if predictions else 0.0

    results = {
        'experiment': experiment_name,
        'config': transform_config,
        'f1': avg_f1,
        'em': avg_em,
        'empty_rate': empty_rate,
        'num_samples': len(predictions),
        'time_sec': elapsed,
    }

    print(f"\n  Results: F1={avg_f1:.4f}, EM={avg_em:.4f}, Empty={empty_rate:.2%}, Time={elapsed:.1f}s")

    return results


def generate_experiment_configs():
    """Generate lightweight experiment configurations only."""
    configs = []

    # Baseline
    configs.append({
        'name': 'baseline',
        'config': {'type': 'baseline'},
        'description': 'No transform (upper bound)',
    })

    # RMS Matching - sweep target scale (LIGHTWEIGHT)
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        configs.append({
            'name': f'rms_scale{scale}',
            'config': {'type': 'rms_matching', 'target_scale': scale},
            'description': f'RMS matching (scale={scale})',
        })

    # Batch distribution (LIGHTWEIGHT)
    configs.append({
        'name': 'batch_dist',
        'config': {'type': 'batch_distribution'},
        'description': 'Batch distribution matching',
    })

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to evaluate. Runtime ≈ samples × num_experiments × 0.5-2s')
    parser.add_argument('--output_dir', type=str, default='runs/embed_sweep_simple')
    parser.add_argument('--experiment', type=str, default='all',
                       help='Run specific experiment or "all"')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model: {args.model_id}...")
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

    print("Model loaded!")

    # Cache vocab stats ONCE
    print("\nCaching vocabulary statistics...")
    vocab_stats = get_vocab_embedding_stats(model)
    print(f"  Vocab RMS: {vocab_stats['rms']:.2f}")
    print(f"  Vocab mean: {vocab_stats['mean']:.4f}")
    print(f"  Vocab std: {vocab_stats['std']:.2f}")

    # Load data
    print(f"\nLoading dataset: {args.dataset} ({args.samples} samples)...")
    examples = load_examples(dataset=args.dataset, split='validation', samples=args.samples, seed=42)
    print(f"Loaded {len(examples)} examples")

    # Generate experiment configs
    all_configs = generate_experiment_configs()

    # Filter if specific experiment requested
    if args.experiment != 'all':
        all_configs = [c for c in all_configs if c['name'] == args.experiment]
        if not all_configs:
            print(f"ERROR: Experiment '{args.experiment}' not found")
            print(f"Available: {[c['name'] for c in generate_experiment_configs()]}")
            sys.exit(1)

    num_experiments = len(all_configs)
    estimated_time = args.samples * num_experiments * (2 if device.type == 'cuda' else 4) / 60

    print(f"\n{'='*80}")
    print(f"Running {num_experiments} LIGHTWEIGHT experiments")
    print(f"Estimated runtime: ~{estimated_time:.0f} minutes ({args.samples} samples × {num_experiments} configs)")
    print(f"{'='*80}\n")

    # Run all experiments
    results = []
    for idx, exp_config in enumerate(all_configs, 1):
        print(f"\n[{idx}/{num_experiments}]")

        try:
            result = run_experiment(
                exp_config['name'],
                exp_config['config'],
                model,
                tokenizer,
                examples,
                device,
                vocab_stats,  # Pass cached stats
                max_samples=args.samples
            )
            result['description'] = exp_config['description']
            results.append(result)

            # Save intermediate results
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"{exp_config['name']}.json", 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"\n❌ ERROR in {exp_config['name']}: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({
                'experiment': exp_config['name'],
                'error': str(e),
                'description': exp_config['description'],
            })

    # Print summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    # Sort by F1 score
    results_sorted = sorted(
        [r for r in results if 'error' not in r],
        key=lambda x: x.get('f1', 0.0),
        reverse=True
    )

    print(f"{'Rank':<6} {'Experiment':<30} {'F1':>8} {'EM':>8} {'Empty%':>8} {'Time(s)':>8}")
    print(f"{'-'*6} {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for rank, result in enumerate(results_sorted, 1):
        exp_name = result['experiment']
        f1_val = result['f1']
        em_val = result['em']
        empty = result.get('empty_rate', 0.0) * 100
        time_sec = result.get('time_sec', 0.0)
        print(f"{rank:<6} {exp_name:<30} {f1_val:>8.4f} {em_val:>8.4f} {empty:>7.1f}% {time_sec:>8.1f}")

    # Show errors
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\n{'-'*80}")
        print("ERRORS:")
        for result in errors:
            print(f"  {result['experiment']}: {result['error']}")

    print(f"\n{'='*80}\n")

    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'total_experiments': len(all_configs),
            'successful': len(results_sorted),
            'failed': len(errors),
            'results': results,
            'sorted_by_f1': [
                {
                    'rank': i+1,
                    'experiment': r['experiment'],
                    'f1': r['f1'],
                    'em': r['em'],
                    'empty_rate': r['empty_rate'],
                    'config': r['config'],
                }
                for i, r in enumerate(results_sorted[:10])
            ]
        }, f, indent=2)

    print(f"Summary saved to: {output_dir}/summary.json")

    # Print top 3 winners
    if results_sorted:
        print(f"\n{'='*80}")
        print("TOP 3 WINNERS:")
        print(f"{'='*80}\n")
        for i, result in enumerate(results_sorted[:3], 1):
            print(f"{i}. {result['experiment']}")
            print(f"   F1: {result['f1']:.4f}, EM: {result['em']:.4f}, Empty: {result['empty_rate']:.1%}")
            print(f"   Config: {result['config']}")
            print(f"   Description: {result['description']}")
            print()


if __name__ == '__main__':
    main()
