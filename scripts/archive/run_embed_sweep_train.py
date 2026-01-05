"""
Lightweight training script for embedding experiment sweep.
Quickly trains and evaluates each experiment for 1 epoch.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data import get_dataset
from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper, apply_lora_if_requested
from latentwire.embed_experiments import (
    get_vocab_embedding_stats,
    StatisticalMatcher,
    NearestKProjector,
    AnchorPlusOffset,
    SoftCodebook,
    RandomInterpolationBaseline,
)
from latentwire.metrics import compute_f1, compute_em
from latentwire.prefix_utils import calibrate_latent_scale


def train_one_epoch(encoder, adapters, models, dataloader, optimizer, experiment_config, device, epoch):
    """Train for one epoch with the specified experiment configuration."""
    encoder.train()
    for adapter in adapters.values():
        adapter.train()

    total_loss = 0.0
    total_cos_sim = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Get text and convert to bytes
        texts = batch['question']  # Assuming dataset has 'question' field

        # Encode with byte encoder
        byte_sequences = [text.encode('utf-8') for text in texts]
        max_len = max(len(seq) for seq in byte_sequences)
        byte_ids = torch.zeros(len(byte_sequences), max_len, dtype=torch.long, device=device)

        for i, seq in enumerate(byte_sequences):
            byte_ids[i, :len(seq)] = torch.tensor(list(seq), dtype=torch.long)

        # Encoder forward
        Z = encoder(byte_ids, attn_mask=None)  # [B, M, d_z]

        # Get target embeddings from tokenizer
        model_key = list(models.keys())[0]  # Use first model
        model = models[model_key]
        tokenizer = model.tokenizer

        encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoded['input_ids'].to(device)

        with torch.no_grad():
            target_embeddings = model.model.get_input_embeddings()(input_ids)  # [B, seq_len, d_model]

        # Adapter forward
        adapter = adapters[model_key]
        reconstructed = adapter(Z)  # [B, M, d_model]

        # Apply experiment transformation
        reconstructed = apply_experiment_transform(
            reconstructed,
            experiment_config,
            models[model_key].model,
            device
        )

        # Reconstruction loss (average over all positions)
        target_trunc = target_embeddings[:, :reconstructed.size(1), :]  # Match sequence length

        mse_loss = F.mse_loss(reconstructed, target_trunc)
        cos_sim = F.cosine_similarity(
            reconstructed.flatten(0, 1),
            target_trunc.flatten(0, 1),
            dim=-1
        ).mean()

        loss = mse_loss + (1.0 - cos_sim)  # Combined loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        for adapter in adapters.values():
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cos_sim += cos_sim.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}, cos_sim={cos_sim.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_cos_sim = total_cos_sim / max(num_batches, 1)

    return {
        'epoch': epoch,
        'loss': avg_loss,
        'cos_sim': avg_cos_sim,
    }


def apply_experiment_transform(embeddings, experiment_config, model, device):
    """Apply experiment-specific transformation to embeddings."""
    exp_type = experiment_config['type']

    if exp_type == 'baseline':
        return embeddings

    elif exp_type == 'rms_matching':
        # Per-example RMS matching
        vocab_stats = get_vocab_embedding_stats(model)
        matcher = StatisticalMatcher(mode='per_example_rms').to(device)

        B, S, D = embeddings.shape
        target_rms = torch.ones(B, S, 1, device=device) * vocab_stats['rms']
        return matcher(embeddings, {'rms': target_rms})

    elif exp_type == 'batch_distribution':
        vocab_stats = get_vocab_embedding_stats(model)
        matcher = StatisticalMatcher(mode='batch_distribution').to(device)
        return matcher(embeddings, vocab_stats)

    elif exp_type == 'nearest_k':
        vocab_stats = get_vocab_embedding_stats(model)
        projector = NearestKProjector(
            vocab_stats['embeddings'],
            k=experiment_config.get('k', 5),
            alpha=experiment_config.get('alpha', 0.5)
        ).to(device)
        return projector(embeddings)

    elif exp_type == 'anchor_offset':
        vocab_stats = get_vocab_embedding_stats(model)
        anchor = AnchorPlusOffset(
            vocab_stats['embeddings'],
            epsilon=experiment_config.get('epsilon', 0.1)
        ).to(device)
        result, _ = anchor(embeddings)
        return result

    elif exp_type == 'soft_codebook':
        if not hasattr(experiment_config, '_codebook_module'):
            experiment_config._codebook_module = SoftCodebook(
                d_model=embeddings.size(-1),
                codebook_size=experiment_config.get('codebook_size', 512),
                temperature=experiment_config.get('temperature', 1.0)
            ).to(device)
        result, _ = experiment_config._codebook_module(embeddings, hard=False)
        return result

    return embeddings


@torch.no_grad()
def evaluate(encoder, adapters, models, dataloader, experiment_config, device):
    """Evaluate on the dataset."""
    encoder.eval()
    for adapter in adapters.values():
        adapter.eval()

    predictions = []
    references = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 20:  # Quick eval, just 20 batches
            break

        texts = batch['question']
        answers = batch['answer']

        # Encode
        byte_sequences = [text.encode('utf-8') for text in texts]
        max_len = max(len(seq) for seq in byte_sequences)
        byte_ids = torch.zeros(len(byte_sequences), max_len, dtype=torch.long, device=device)

        for i, seq in enumerate(byte_sequences):
            byte_ids[i, :len(seq)] = torch.tensor(list(seq), dtype=torch.long)

        Z = encoder(byte_ids, attn_mask=None)

        # Use first model for generation
        model_key = list(models.keys())[0]
        model = models[model_key]
        adapter = adapters[model_key]

        reconstructed = adapter(Z)
        reconstructed = apply_experiment_transform(
            reconstructed,
            experiment_config,
            model.model,
            device
        )

        # Generate
        outputs = model.model.generate(
            inputs_embeds=reconstructed,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )

        # Decode
        for i, output_ids in enumerate(outputs):
            generated_text = model.tokenizer.decode(output_ids, skip_special_tokens=True)
            predictions.append(generated_text)
            references.append(answers[i])

    # Compute metrics
    f1_scores = [compute_f1(pred, ref) for pred, ref in zip(predictions, references)]
    em_scores = [compute_em(pred, ref) for pred, ref in zip(predictions, references)]

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

    return {
        'f1': avg_f1,
        'em': avg_em,
        'num_samples': len(predictions),
    }


def run_experiment(experiment_name, experiment_config, args, device):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config: {experiment_config}")
    print(f"{'='*80}\n")

    # Create output dir
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(args.dataset, split='train', samples=args.samples)
    subset_size = min(args.samples, len(dataset))
    dataset = Subset(dataset, range(subset_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    print(f"Loading model: {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map='auto' if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if requested
    if experiment_config.get('use_lora'):
        lora_config = experiment_config.get('lora_config', {})
        apply_lora_if_requested(model, lora_config, args.model_id)
        print(f"Applied LoRA: {lora_config}")

    # Wrap model
    models = {
        'model': LMWrapper(
            model=model,
            tokenizer=tokenizer,
            model_key='model',
            device=device,
        )
    }

    # Create encoder
    print("Creating encoder...")
    encoder = InterlinguaEncoder(
        d_z=args.d_z,
        n_layers=args.encoder_layers,
        n_heads=args.encoder_heads,
        latent_len=args.latent_len,
        model_keys=['model'],
    ).to(device)

    # Create adapter
    d_model = model.config.hidden_size
    adapters = {
        'model': Adapter(
            d_z=args.d_z,
            d_model=d_model,
            latent_length=args.latent_len,
            enable_metadata=False,
            hidden_mult=2,
        ).to(device)
    }

    # Optimizer
    params = list(encoder.parameters())
    for adapter in adapters.values():
        params.extend(adapter.parameters())

    if experiment_config.get('use_lora'):
        # Add LoRA parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # Training
    print(f"\nTraining for {args.epochs} epoch(s)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            encoder, adapters, models, dataloader,
            optimizer, experiment_config, device, epoch
        )

        print(f"  Train metrics: {train_metrics}")

        # Save metrics
        with open(output_dir / 'diagnostics.jsonl', 'a') as f:
            f.write(json.dumps({**train_metrics, 'type': 'train'}) + '\n')

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.time() - start_time

    # Evaluation
    print("\nEvaluating...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_start = time.time()
    eval_metrics = evaluate(
        encoder, adapters, models, dataloader,
        experiment_config, device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_time = time.time() - eval_start

    print(f"  Eval metrics: {eval_metrics}")

    # Save final results
    final_results = {
        'experiment': experiment_name,
        'config': experiment_config,
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'train_time_sec': train_time,
        'eval_time_sec': eval_time,
        'type': 'final_eval',
    }

    with open(output_dir / 'diagnostics.jsonl', 'a') as f:
        f.write(json.dumps(final_results) + '\n')

    print(f"\n{experiment_name} complete: F1={eval_metrics['f1']:.4f}, EM={eval_metrics['em']:.4f}")

    # Cleanup to save memory
    del model, encoder, adapters, optimizer
    torch.cuda.empty_cache()

    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name or "all" to run all')
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_len', type=int, default=32)
    parser.add_argument('--d_z', type=int, default=256)
    parser.add_argument('--encoder_layers', type=int, default=4)
    parser.add_argument('--encoder_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--output_dir', type=str, default='runs')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define all experiments
    experiments = {
        'baseline': {
            'type': 'baseline',
            'description': 'No transformation (current state)',
        },
        'rms_matching': {
            'type': 'rms_matching',
            'description': 'Per-example RMS matching to vocab embeddings',
        },
        'batch_distribution': {
            'type': 'batch_distribution',
            'description': 'Batch-level mean+std matching',
        },
        'nearest_k': {
            'type': 'nearest_k',
            'k': 5,
            'alpha': 0.5,
            'description': 'Project onto K nearest vocab embeddings',
        },
        'nearest_k_strict': {
            'type': 'nearest_k',
            'k': 3,
            'alpha': 0.2,
            'description': 'Stricter projection (K=3, alpha=0.2)',
        },
        'anchor_offset_small': {
            'type': 'anchor_offset',
            'epsilon': 0.05,
            'description': 'Anchor + 5% offset',
        },
        'anchor_offset_medium': {
            'type': 'anchor_offset',
            'epsilon': 0.15,
            'description': 'Anchor + 15% offset',
        },
        'soft_codebook_512': {
            'type': 'soft_codebook',
            'codebook_size': 512,
            'temperature': 1.0,
            'description': 'Soft codebook with 512 entries',
        },
        'soft_codebook_1024': {
            'type': 'soft_codebook',
            'codebook_size': 1024,
            'temperature': 0.7,
            'description': 'Soft codebook with 1024 entries, lower temp',
        },
        'lora_half': {
            'type': 'baseline',
            'use_lora': True,
            'lora_config': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.05,
                'target_modules': 'auto',
                'first_n': 16,  # First 50% of layers (assuming 32 layers)
            },
            'description': 'LoRA on first 50% of layers',
        },
        'lora_full': {
            'type': 'baseline',
            'use_lora': True,
            'lora_config': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.05,
                'target_modules': 'auto',
                'first_n': None,  # All layers
            },
            'description': 'LoRA on all layers',
        },
    }

    # Run experiments
    if args.experiment == 'all':
        results = {}
        for exp_name, exp_config in experiments.items():
            try:
                result = run_experiment(exp_name, exp_config, args, device)
                results[exp_name] = result
            except Exception as e:
                print(f"\n ERROR in {exp_name}: {e}\n")
                import traceback
                traceback.print_exc()
                results[exp_name] = {'error': str(e)}

        # Print summary
        print(f"\n\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}\n")
        print(f"{'Experiment':<30} {'F1':>10} {'EM':>10} {'Description':<40}")
        print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*40}")

        for exp_name, result in results.items():
            if 'error' in result:
                print(f"{exp_name:<30} {'ERROR':>10} {'ERROR':>10} {experiments[exp_name].get('description', ''):<40}")
            else:
                f1 = result['eval_metrics']['f1']
                em = result['eval_metrics']['em']
                desc = experiments[exp_name].get('description', '')
                print(f"{exp_name:<30} {f1:>10.4f} {em:>10.4f} {desc:<40}")

        print(f"\n{'='*80}\n")

        # Save summary
        summary_path = Path(args.output_dir) / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    else:
        # Run single experiment
        if args.experiment not in experiments:
            print(f"ERROR: Unknown experiment '{args.experiment}'")
            print(f"Available: {list(experiments.keys())}")
            sys.exit(1)

        exp_config = experiments[args.experiment]
        run_experiment(args.experiment, exp_config, args, device)


if __name__ == '__main__':
    main()
