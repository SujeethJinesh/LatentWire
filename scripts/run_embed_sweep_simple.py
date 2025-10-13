"""
Simplified embedding experiment sweep with hyperparameter sweeps.
Tests transformations only, no training. Uses real text embeddings to isolate transform effects.
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
from latentwire.embed_experiments import (
    get_vocab_embedding_stats,
    StatisticalMatcher,
    NearestKProjector,
    AnchorPlusOffset,
    SoftCodebook,
)
from latentwire.metrics import compute_f1, compute_em


def apply_transform(embeddings, transform_config, model, device, modules_cache):
    """Apply embedding transformation based on config."""
    transform_type = transform_config['type']

    if transform_type == 'baseline':
        return embeddings

    vocab_stats = get_vocab_embedding_stats(model)

    if transform_type == 'rms_matching':
        B, S, D = embeddings.shape
        target_scale = transform_config.get('target_scale', 1.0)
        target_rms = torch.ones(B, S, 1, device=device) * vocab_stats['rms'] * target_scale
        matcher = StatisticalMatcher(mode='per_example_rms')
        return matcher(embeddings, {'rms': target_rms})

    elif transform_type == 'batch_distribution':
        matcher = StatisticalMatcher(mode='batch_distribution')
        return matcher(embeddings, vocab_stats)

    elif transform_type == 'nearest_k':
        k = transform_config.get('k', 5)
        alpha = transform_config.get('alpha', 0.5)
        cache_key = f'nearest_k_{k}_{alpha}'

        if cache_key not in modules_cache:
            modules_cache[cache_key] = NearestKProjector(
                vocab_stats['embeddings'], k=k, alpha=alpha
            ).to(device)
        return modules_cache[cache_key](embeddings)

    elif transform_type == 'anchor_offset':
        epsilon = transform_config.get('epsilon', 0.1)
        cache_key = f'anchor_offset_{epsilon}'

        if cache_key not in modules_cache:
            modules_cache[cache_key] = AnchorPlusOffset(
                vocab_stats['embeddings'], epsilon=epsilon
            ).to(device)
        result, _ = modules_cache[cache_key](embeddings)
        return result

    elif transform_type == 'soft_codebook':
        codebook_size = transform_config.get('codebook_size', 512)
        temperature = transform_config.get('temperature', 1.0)
        cache_key = f'soft_codebook_{codebook_size}_{temperature}'

        if cache_key not in modules_cache:
            D = embeddings.size(-1)
            modules_cache[cache_key] = SoftCodebook(
                d_model=D, codebook_size=codebook_size, temperature=temperature
            ).to(device)
        result, _ = modules_cache[cache_key](embeddings, hard=False)
        return result

    return embeddings


@torch.no_grad()
def run_experiment(experiment_name, transform_config, model, tokenizer, examples, device, max_samples=100):
    """
    Run experiment: Get text embeddings, apply transform, generate, evaluate.
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Config: {transform_config}")
    print(f"{'='*80}")

    model.eval()
    modules_cache = {}

    predictions = []
    references = []
    empty_count = 0
    start_time = time.time()

    for idx, example in enumerate(examples[:max_samples]):
        if idx > 0 and idx % 25 == 0:
            print(f"  Progress: {idx}/{max_samples}")

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

        # Apply transformation
        transformed_embeddings = apply_transform(
            source_embeddings,
            transform_config,
            model,
            device,
            modules_cache
        )

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

    # Compute metrics
    f1_scores = [compute_f1(pred, ref) for pred, ref in zip(predictions, references)]
    em_scores = [compute_em(pred, ref) for pred, ref in zip(predictions, references)]

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
    """Generate all experiment configurations including hyperparameter sweeps."""
    configs = []

    # Baseline
    configs.append({
        'name': 'baseline',
        'config': {'type': 'baseline'},
        'description': 'No transform (upper bound)',
    })

    # RMS Matching - sweep target scale
    for scale in [0.8, 1.0, 1.2]:
        configs.append({
            'name': f'rms_scale{scale}',
            'config': {'type': 'rms_matching', 'target_scale': scale},
            'description': f'RMS matching (scale={scale})',
        })

    # Batch distribution
    configs.append({
        'name': 'batch_dist',
        'config': {'type': 'batch_distribution'},
        'description': 'Batch distribution matching',
    })

    # K-Nearest Projection - sweep k and alpha
    k_values = [3, 5, 7, 10]
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Sample key combinations (not all 20, too many)
    k_alpha_pairs = [
        (3, 0.2), (3, 0.5),
        (5, 0.3), (5, 0.5), (5, 0.7),
        (7, 0.5), (7, 0.7),
        (10, 0.5), (10, 0.7), (10, 0.9),
    ]

    for k, alpha in k_alpha_pairs:
        configs.append({
            'name': f'nearest_k{k}_a{alpha}',
            'config': {'type': 'nearest_k', 'k': k, 'alpha': alpha},
            'description': f'K-nearest (k={k}, α={alpha})',
        })

    # Anchor + Offset - sweep epsilon
    epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    for eps in epsilon_values:
        configs.append({
            'name': f'anchor_eps{eps}',
            'config': {'type': 'anchor_offset', 'epsilon': eps},
            'description': f'Anchor+offset (ε={eps})',
        })

    # Soft Codebook - sweep size and temperature
    codebook_configs = [
        (128, 1.0),
        (256, 0.7),
        (256, 1.0),
        (512, 0.7),
        (512, 1.0),
        (1024, 1.0),
    ]

    for size, temp in codebook_configs:
        configs.append({
            'name': f'codebook_{size}_t{temp}',
            'config': {'type': 'soft_codebook', 'codebook_size': size, 'temperature': temp},
            'description': f'Soft codebook (size={size}, τ={temp})',
        })

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--samples', type=int, default=100)
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

    print(f"\n{'='*80}")
    print(f"Running {len(all_configs)} experiments (including hyperparameter sweeps)")
    print(f"{'='*80}\n")

    # Run all experiments
    results = []
    for idx, exp_config in enumerate(all_configs, 1):
        print(f"\n[{idx}/{len(all_configs)}]")

        try:
            result = run_experiment(
                exp_config['name'],
                exp_config['config'],
                model,
                tokenizer,
                examples,
                device,
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

    print(f"{'Rank':<6} {'Experiment':<30} {'F1':>8} {'EM':>8} {'Empty%':>8} {'Description'}")
    print(f"{'-'*6} {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*40}")

    for rank, result in enumerate(results_sorted, 1):
        exp_name = result['experiment']
        desc = result.get('description', '')
        f1 = result['f1']
        em = result['em']
        empty = result.get('empty_rate', 0.0) * 100
        print(f"{rank:<6} {exp_name:<30} {f1:>8.4f} {em:>8.4f} {empty:>7.1f}% {desc}")

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
                }
                for i, r in enumerate(results_sorted[:10])  # Top 10
            ]
        }, f, indent=2)

    print(f"Summary saved to: {output_dir}/summary.json")

    # Print top 3 winners
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
