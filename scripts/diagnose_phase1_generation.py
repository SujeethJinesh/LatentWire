#!/usr/bin/env python3
"""
Diagnostic script to inspect Phase 1a generation outputs.

Usage:
    python scripts/diagnose_phase1_generation.py \
        --checkpoint runs/phase1a_cluster/baseline/adapter_phase1_best.pt \
        --samples 10
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add latentwire to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data import load_examples
from train_adapter_only_phase1 import EmbeddingCompressor
from latentwire.models import Adapter
from latentwire.core_utils import batch_metrics


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load adapter and compressor from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract config
    config = ckpt['config']
    print(f"Config: {config}")

    # Initialize compressor
    compressor = EmbeddingCompressor(
        input_dim=config['input_dim'],
        output_dim=config['compress_dim'],
        method=config['compress_method']
    )
    compressor.initialize_from_pca(
        components=ckpt['compressor']['projection'].T.numpy(),
        mean=ckpt['compressor']['mean'].numpy(),
        explained_variance_ratio=ckpt['compressor']['explained_variance_ratio'],
    )
    print(f"Compressor initialized: {config['input_dim']}D → {config['compress_dim']}D")
    print(f"PCA explained variance: {compressor.explained_variance_ratio:.2%}\n")

    # Initialize adapter
    adapter = Adapter(
        d_z=config['compress_dim'],
        d_model=config['input_dim'],
        latent_length=32,  # Not used
        hidden_mult=config['adapter_hidden_mult'],
        dropout=0.0,  # Eval mode
        enable_metadata=False,
        colorize=False,
    ).to(device)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    adapter.eval()
    print(f"Adapter loaded: {sum(p.numel() for p in adapter.parameters()):,} parameters\n")

    return adapter, compressor


def diagnose_generation(
    adapter,
    compressor,
    model,
    tokenizer,
    examples,
    device: torch.device,
    max_new_tokens: int = 12,
):
    """Run generation and print detailed diagnostics."""
    adapter.eval()
    model.eval()

    embed_device = model.get_input_embeddings().weight.device
    model_dtype = model.get_input_embeddings().weight.dtype
    adapter_dtype = next(adapter.parameters()).dtype

    predictions = []
    references = []

    print("="*80)
    print("GENERATION DIAGNOSTICS")
    print("="*80)
    print()

    for idx, ex in enumerate(examples):
        print(f"Example {idx + 1}/{len(examples)}")
        print("-" * 80)
        print(f"Source: {ex['source'][:200]}...")
        print(f"Gold answer: {ex['answer']}")
        print()

        # Tokenize
        encoded = tokenizer(
            ex['source'],
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=False
        )
        input_ids = encoded['input_ids'].to(embed_device)
        print(f"Input tokens: {input_ids.shape[1]} tokens")

        # Extract embeddings
        with torch.no_grad():
            text_embeds = model.get_input_embeddings()(input_ids)
            print(f"Original embeddings: {text_embeds.shape}")
            print(f"  Mean: {text_embeds.mean().item():.4f}, Std: {text_embeds.std().item():.4f}")
            print(f"  RMS: {text_embeds.norm(p=2, dim=-1).mean().item():.4f}")

            # Compress
            compressed = compressor.compress(text_embeds)
            print(f"Compressed: {compressed.shape}")
            print(f"  Mean: {compressed.mean().item():.4f}, Std: {compressed.std().item():.4f}")
            print(f"  RMS: {compressed.norm(p=2, dim=-1).mean().item():.4f}")

            # Reconstruct
            compressed = compressed.to(device, dtype=adapter_dtype)
            reconstructed = adapter(compressed)
            reconstructed = reconstructed.to(embed_device, dtype=model_dtype)
            print(f"Reconstructed: {reconstructed.shape}")
            print(f"  Mean: {reconstructed.mean().item():.4f}, Std: {reconstructed.std().item():.4f}")
            print(f"  RMS: {reconstructed.norm(p=2, dim=-1).mean().item():.4f}")

            # Compute reconstruction quality
            cos_sim = torch.nn.functional.cosine_similarity(
                reconstructed.view(-1, reconstructed.shape[-1]),
                text_embeds.view(-1, text_embeds.shape[-1]),
                dim=-1
            ).mean()
            mse = torch.nn.functional.mse_loss(reconstructed, text_embeds)
            print(f"Reconstruction quality:")
            print(f"  Cosine sim: {cos_sim.item():.4f}")
            print(f"  MSE: {mse.item():.4f}")
            print()

            # Generate
            outputs = model.generate(
                inputs_embeds=reconstructed,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated (full): {pred_text}")
            print()

            # Try to extract just the answer portion
            # The generated text should start with the source, so let's see what was added
            source_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if pred_text.startswith(source_text):
                answer_only = pred_text[len(source_text):].strip()
                print(f"Generated (answer only): {answer_only}")
            else:
                print(f"WARNING: Generated text doesn't start with source!")
                print(f"Source length: {len(source_text)}")
                print(f"Generated length: {len(pred_text)}")
                answer_only = pred_text

            predictions.append(pred_text)
            references.append(ex['answer'])

        print()
        print()

    # Compute overall metrics
    em, f1 = batch_metrics(predictions, references)
    print("="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"EM: {em:.2%}")
    print(f"F1: {f1:.2%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose Phase 1a generation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='squad', choices=['squad', 'hotpot', 'squad_v2'])
    parser.add_argument('--samples', type=int, default=10, help='Number of examples to diagnose')
    parser.add_argument('--max_new_tokens', type=int, default=12)

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model and tokenizer
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print("Model loaded!\n")

    # Load checkpoint
    adapter, compressor = load_checkpoint(checkpoint_path, device)

    # Load examples
    print(f"Loading {args.samples} examples from {args.dataset}...")
    examples = load_examples(
        dataset=args.dataset,
        split='validation',
        samples=args.samples,
        seed=42
    )
    print(f"Loaded {len(examples)} examples\n")

    # Run diagnostics
    diagnose_generation(
        adapter=adapter,
        compressor=compressor,
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == '__main__':
    main()
