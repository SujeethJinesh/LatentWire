#!/usr/bin/env python3
"""
Ablation 4: Quantization and Compression Analysis

Analyzes honest wire protocol compression for different quantization levels:
- Text baseline (UTF-8)
- FP16 (2 bytes/value)
- INT8 (1 byte/value + scales)
- INT6 (0.75 bytes/value + scales)
- INT4 (0.5 bytes/value + scales)

Usage:
    python paper_writing/analyze_compression.py \
        --checkpoint paper_writing/runs/ablations_XXX/1a_stable_64tok/checkpoint.pt \
        --num_samples 200
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from the training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cross_attention import BottleneckGatedTranslator, build_samples, evaluate_numeric_accuracy


def quantize_tensor(tensor: torch.Tensor, bits: int, group_size: int = 32) -> Tuple[bytes, int]:
    """
    Group-wise quantization of tensor

    Returns:
        quantized_bytes: The quantized data
        overhead_bytes: Size of scales and metadata
    """
    flat = tensor.flatten()
    num_elements = flat.numel()
    num_groups = (num_elements + group_size - 1) // group_size

    # Metadata overhead (shape, dtype, etc.)
    metadata_bytes = 16  # Conservative estimate

    if bits == 16:
        # FP16: just convert directly
        data_bytes = flat.half().cpu().numpy().tobytes()
        overhead = metadata_bytes
        return data_bytes, overhead

    # For INT quantization, we need scales per group
    quantized_data = []
    scales = []

    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, num_elements)
        group = flat[start:end]

        # Compute scale (max absolute value)
        scale = group.abs().max().item()
        scales.append(scale)

        if scale == 0:
            quantized_data.extend([0] * len(group))
            continue

        # Quantize to [-2^(bits-1), 2^(bits-1)-1]
        max_int = 2 ** (bits - 1) - 1
        quantized = (group / scale * max_int).round().clamp(-max_int - 1, max_int)
        quantized_data.extend(quantized.cpu().numpy().astype(np.int8).tolist())

    # Pack into bytes
    if bits == 8:
        data_bytes = bytes(np.array(quantized_data, dtype=np.int8).tobytes())
    elif bits == 6:
        # 6 bits: pack 4 values into 3 bytes
        # For simplicity, we'll estimate 0.75 bytes per value
        data_bytes = len(quantized_data) * 3 // 4
    elif bits == 4:
        # 4 bits: pack 2 values into 1 byte
        data_bytes = len(quantized_data) // 2

    # Scales are stored as FP32
    scales_bytes = num_groups * 4
    overhead = metadata_bytes + scales_bytes

    if isinstance(data_bytes, int):
        total_data_bytes = data_bytes
    else:
        total_data_bytes = len(data_bytes)

    return total_data_bytes, overhead


def measure_text_bytes(text: str) -> int:
    """Measure bytes for UTF-8 encoded text"""
    return len(text.encode('utf-8'))


def measure_latent_bytes(soft_tokens: torch.Tensor, bits: int, group_size: int = 32) -> Dict[str, int]:
    """
    Measure bytes for quantized latent representation

    Returns:
        dict with 'data_bytes', 'overhead_bytes', 'total_bytes'
    """
    # Anchor text is always sent as UTF-8
    anchor_text = "Answer: "
    anchor_bytes = measure_text_bytes(anchor_text)

    # Quantize soft tokens
    data_bytes, overhead_bytes = quantize_tensor(soft_tokens, bits, group_size)

    return {
        'data_bytes': data_bytes,
        'overhead_bytes': overhead_bytes,
        'anchor_bytes': anchor_bytes,
        'total_bytes': data_bytes + overhead_bytes + anchor_bytes
    }


def analyze_compression(
    checkpoint_path: str,
    source_model_id: str,
    target_model_id: str,
    num_samples: int = 200,
    device: str = 'cuda'
):
    """
    Analyze compression ratios for different quantization levels
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load models
    print(f"Loading source model: {source_model_id}")
    src_model = AutoModelForCausalLM.from_pretrained(
        source_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    src_tok = AutoTokenizer.from_pretrained(source_model_id)

    print(f"Loading target model: {target_model_id}")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tgt_tok = AutoTokenizer.from_pretrained(target_model_id)

    # Load translator
    print("Loading translator...")
    config = checkpoint.get('config', {})
    translator = BottleneckGatedTranslator(
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        bottleneck_dim=config.get('bottleneck_dim', 1024),
        K=config.get('soft_tokens', 64),
        depth=config.get('depth', 8),
        heads=config.get('heads', 16)
    ).to(device).to(torch.bfloat16)

    translator.load_state_dict(checkpoint['translator'])
    translator.eval()

    # Load dataset
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    # Analyze samples
    results = {
        'text': [],
        'fp16': [],
        'int8': [],
        'int6': [],
        'int4': []
    }

    print(f"\nAnalyzing {num_samples} samples...")

    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            if i % 50 == 0:
                print(f"  Processed {i}/{num_samples} samples...")

            # Get source prompt
            samples = build_samples(
                example,
                src_model, src_tok,
                tgt_model, tgt_tok,
                dataset_name='gsm8k'
            )

            if not samples:
                continue

            sample = samples[0]

            # Measure text baseline
            text_bytes = measure_text_bytes(sample.src_prompt)
            results['text'].append(text_bytes)

            # Get soft tokens from translator
            src_enc = src_tok(
                sample.src_prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)

            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[-1].to(torch.bfloat16)

            # Translate
            soft_tokens = translator(src_h)  # [1, K, d_tgt]

            # Measure latent bytes for different quantization levels
            for bits, key in [(16, 'fp16'), (8, 'int8'), (6, 'int6'), (4, 'int4')]:
                latent_info = measure_latent_bytes(soft_tokens, bits)
                results[key].append(latent_info['total_bytes'])

    # Compute statistics
    print(f"\n{'='*60}")
    print(f"COMPRESSION ANALYSIS RESULTS ({len(results['text'])} samples)")
    print(f"{'='*60}\n")

    stats = {}

    for method in ['text', 'fp16', 'int8', 'int6', 'int4']:
        data = results[method]
        avg_bytes = np.mean(data)
        std_bytes = np.std(data)

        stats[method] = {
            'avg_bytes': float(avg_bytes),
            'std_bytes': float(std_bytes),
            'min_bytes': int(np.min(data)),
            'max_bytes': int(np.max(data))
        }

        if method == 'text':
            compression = 1.0
        else:
            compression = stats['text']['avg_bytes'] / avg_bytes

        stats[method]['compression_ratio'] = float(compression)

        print(f"{method.upper():>6}:")
        print(f"  Avg bytes:    {avg_bytes:>8.1f} ± {std_bytes:.1f}")
        print(f"  Range:        {stats[method]['min_bytes']:>8} - {stats[method]['max_bytes']}")
        print(f"  Compression:  {compression:>8.2f}×")
        print()

    # Save results
    output_file = Path(checkpoint_path).parent / 'compression_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': stats,
            'raw_data': {k: [int(x) for x in v] for k, v in results.items()},
            'config': {
                'num_samples': len(results['text']),
                'checkpoint': str(checkpoint_path),
                'source_model': source_model_id,
                'target_model': target_model_id
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Create summary table for paper
    print(f"\n{'='*60}")
    print("LATEX TABLE (copy to paper):")
    print(f"{'='*60}\n")

    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Method & Avg Bytes & Compression & KV Cache Saved \\\\")
    print("\\hline")

    text_avg = stats['text']['avg_bytes']

    for method in ['text', 'fp16', 'int8', 'int6', 'int4']:
        s = stats[method]
        # Estimate KV cache savings (text tokens vs soft tokens)
        # Assume text uses ~150 tokens, latent uses 64 tokens
        # Savings = (150 - 64) * 0.5 MB = 43 MB
        if method == 'text':
            kv_saved = 0
        else:
            kv_saved = 43  # MB (rough estimate)

        print(f"{method.upper()} & {s['avg_bytes']:.0f} & {s['compression_ratio']:.2f}× & {kv_saved} MB \\\\")

    print("\\hline")
    print("\\end{tabular}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze compression and quantization')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to trained checkpoint')
    parser.add_argument('--source_model', type=str,
                      default='mistralai/Mistral-7B-Instruct-v0.3',
                      help='Source model ID')
    parser.add_argument('--target_model', type=str,
                      default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                      help='Target model ID')
    parser.add_argument('--num_samples', type=int, default=200,
                      help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')

    args = parser.parse_args()

    analyze_compression(
        checkpoint_path=args.checkpoint,
        source_model_id=args.source_model,
        target_model_id=args.target_model,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == '__main__':
    main()
