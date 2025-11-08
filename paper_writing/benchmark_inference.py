#!/usr/bin/env python3
"""
Ablation 4: Inference Metrics - KV Cache and Latency Benchmarking

Measures per-sample metrics for all baselines:
- KV cache memory usage during generation
- End-to-end latency
- Peak GPU memory
- Quality (EM/F1)

Stores raw per-sample data for flexible post-hoc analysis.

Usage:
    python paper_writing/benchmark_inference.py \
        --checkpoint paper_writing/runs/ablations_XXX/1a_stable_64tok/checkpoint.pt \
        --num_samples 1319  # Full test set
"""

import argparse
import json
import time
import torch
import gc
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
from cross_attention import (
    BottleneckGatedTranslator, build_samples,
    extract_answer_number, compute_em_f1
)


def get_kv_cache_memory(past_key_values) -> float:
    """
    Calculate KV cache memory in MB

    past_key_values is a tuple of tuples:
    - Outer tuple: one per layer
    - Inner tuple: (key, value) tensors
    Each tensor is [batch, num_heads, seq_len, head_dim]
    """
    if past_key_values is None:
        return 0.0

    total_bytes = 0
    for layer_kv in past_key_values:
        key, value = layer_kv
        # Each tensor: batch × num_heads × seq_len × head_dim
        # BF16 = 2 bytes per element
        total_bytes += key.numel() * 2  # bf16 = 2 bytes
        total_bytes += value.numel() * 2

    return total_bytes / (1024 ** 2)  # Convert to MB


def get_gpu_memory() -> float:
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def benchmark_text_baseline(
    sample: Dict,
    tgt_model,
    tgt_tok,
    max_new_tokens: int,
    device: str
) -> Dict:
    """
    Benchmark text-only baseline
    Full prompt → target model
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = get_gpu_memory()
    start_time = time.time()

    # Tokenize input
    inputs = tgt_tok(
        sample['tgt_prompt'],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    input_len = inputs['input_ids'].shape[1]

    # Generate
    with torch.no_grad():
        outputs = tgt_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True
        )

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Decode
    generated_ids = outputs.sequences[0, input_len:]
    generated_text = tgt_tok.decode(generated_ids, skip_special_tokens=True)

    # Estimate KV cache memory
    # KV cache = (key + value) × num_layers × (input_len + output_len)
    # For Llama 8B: 32 layers, 32 heads, head_dim=128, hidden=4096
    # Each position: 2 × 32 × 32 × 128 × 2 bytes (bf16) = 524,288 bytes = 0.5 MB
    output_len = len(generated_ids)
    total_seq_len = input_len + output_len
    kv_cache_mb = total_seq_len * 0.5  # Rule of thumb: 0.5 MB per token

    return {
        'method': 'text_baseline',
        'generated_text': generated_text,
        'input_len': input_len,
        'output_len': output_len,
        'total_seq_len': total_seq_len,
        'kv_cache_mb': kv_cache_mb,
        'latency_sec': end_time - start_time,
        'peak_mem_mb': peak_mem,
        'mem_delta_mb': peak_mem - start_mem
    }


def benchmark_latent(
    sample: Dict,
    src_model,
    src_tok,
    tgt_model,
    tgt_tok,
    translator,
    max_new_tokens: int,
    device: str,
    dtype
) -> Dict:
    """
    Benchmark latent baseline
    Question → Source → Translator → Target
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = get_gpu_memory()
    start_time = time.time()

    # 1. Encode with source model
    src_inputs = src_tok(
        sample['src_prompt'],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        src_outputs = src_model(**src_inputs, output_hidden_states=True)
        src_hidden = src_outputs.hidden_states[-1].to(dtype)

        # 2. Translate to soft tokens
        soft_tokens = translator(src_hidden)  # [1, K, d_tgt]

        # 3. Prepare target embeddings with anchor
        anchor_text = "Answer: "
        anchor_ids = tgt_tok(anchor_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        anchor_embeds = tgt_model.get_input_embeddings()(anchor_ids)

        # Concatenate: [soft_tokens, anchor_embeds]
        combined_embeds = torch.cat([soft_tokens, anchor_embeds], dim=1)
        attention_mask = torch.ones(1, combined_embeds.shape[1], device=device)

        # 4. Generate from soft tokens
        outputs = tgt_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True
        )

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Decode (skip the anchor tokens)
    input_len = combined_embeds.shape[1]
    generated_ids = outputs.sequences[0, input_len:]
    generated_text = tgt_tok.decode(generated_ids, skip_special_tokens=True)

    # KV cache calculation
    # For latent: K soft tokens + anchor tokens + generated tokens
    K = soft_tokens.shape[1]
    anchor_len = anchor_embeds.shape[1]
    output_len = len(generated_ids)
    total_seq_len = K + anchor_len + output_len
    kv_cache_mb = total_seq_len * 0.5  # 0.5 MB per token

    return {
        'method': 'latent',
        'generated_text': generated_text,
        'soft_tokens': K,
        'anchor_len': anchor_len,
        'output_len': output_len,
        'total_seq_len': total_seq_len,
        'kv_cache_mb': kv_cache_mb,
        'latency_sec': end_time - start_time,
        'peak_mem_mb': peak_mem,
        'mem_delta_mb': peak_mem - start_mem
    }


def benchmark_token_budget(
    sample: Dict,
    tgt_model,
    tgt_tok,
    max_new_tokens: int,
    budget_tokens: int,
    device: str
) -> Dict:
    """
    Benchmark token-budget baseline
    Truncate prompt to M tokens, send to target
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = get_gpu_memory()
    start_time = time.time()

    # Tokenize and truncate to budget
    inputs = tgt_tok(
        sample['tgt_prompt'],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=budget_tokens
    ).to(device)

    input_len = inputs['input_ids'].shape[1]

    # Generate
    with torch.no_grad():
        outputs = tgt_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True
        )

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Decode
    generated_ids = outputs.sequences[0, input_len:]
    generated_text = tgt_tok.decode(generated_ids, skip_special_tokens=True)

    # KV cache
    output_len = len(generated_ids)
    total_seq_len = input_len + output_len
    kv_cache_mb = total_seq_len * 0.5

    return {
        'method': f'token_budget_{budget_tokens}',
        'generated_text': generated_text,
        'input_len': input_len,
        'output_len': output_len,
        'total_seq_len': total_seq_len,
        'kv_cache_mb': kv_cache_mb,
        'latency_sec': end_time - start_time,
        'peak_mem_mb': peak_mem,
        'mem_delta_mb': peak_mem - start_mem
    }


def run_benchmark(
    checkpoint_path: str,
    source_model_id: str,
    target_model_id: str,
    dataset_name: str = 'gsm8k',
    num_samples: int = 1319,
    max_new_tokens: int = 256,
    device: str = 'cuda'
):
    """
    Run complete benchmark across all baselines
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Load models
    print(f"Loading models...")
    dtype = torch.bfloat16

    src_model = AutoModelForCausalLM.from_pretrained(
        source_model_id,
        torch_dtype=dtype,
        device_map=device
    )
    src_tok = AutoTokenizer.from_pretrained(source_model_id)
    src_model.eval()

    tgt_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=dtype,
        device_map=device
    )
    tgt_tok = AutoTokenizer.from_pretrained(target_model_id)
    tgt_model.eval()

    # Load translator
    print("Loading translator...")
    K = config.get('soft_tokens', 64)
    translator = BottleneckGatedTranslator(
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        bottleneck_dim=config.get('bottleneck_dim', 1024),
        K=K,
        depth=config.get('depth', 8),
        heads=config.get('heads', 16)
    ).to(device).to(dtype)

    translator.load_state_dict(checkpoint['translator'])
    translator.eval()

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'gsm8k':
        dataset = load_dataset("gsm8k", "main", split="test")
    elif dataset_name == 'hotpotqa':
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Run benchmarks
    results = []

    print(f"\nBenchmarking {num_samples} samples...")
    print(f"Methods: text_baseline, latent, token_budget_{K}")
    print("-" * 60)

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        if i % 50 == 0:
            print(f"Sample {i}/{num_samples}...")

        # Build sample
        samples = build_samples(
            example,
            src_model, src_tok,
            tgt_model, tgt_tok,
            dataset_name=dataset_name
        )

        if not samples:
            continue

        sample = samples[0]
        gold_answer = sample.answer

        # Benchmark each method
        sample_results = {
            'sample_id': i,
            'question': sample.src_prompt[:200] + '...',  # Truncate for storage
            'gold_answer': gold_answer
        }

        # 1. Text baseline
        text_result = benchmark_text_baseline(
            sample, tgt_model, tgt_tok, max_new_tokens, device
        )
        pred_text = extract_answer_number(text_result['generated_text'])
        em, f1 = compute_em_f1(pred_text, gold_answer)
        text_result['em'] = em
        text_result['f1'] = f1
        sample_results['text_baseline'] = text_result

        # 2. Latent
        latent_result = benchmark_latent(
            sample, src_model, src_tok, tgt_model, tgt_tok,
            translator, max_new_tokens, device, dtype
        )
        pred_latent = extract_answer_number(latent_result['generated_text'])
        em, f1 = compute_em_f1(pred_latent, gold_answer)
        latent_result['em'] = em
        latent_result['f1'] = f1
        sample_results['latent'] = latent_result

        # 3. Token budget (truncate to K tokens)
        budget_result = benchmark_token_budget(
            sample, tgt_model, tgt_tok, max_new_tokens, K, device
        )
        pred_budget = extract_answer_number(budget_result['generated_text'])
        em, f1 = compute_em_f1(pred_budget, gold_answer)
        budget_result['em'] = em
        budget_result['f1'] = f1
        sample_results['token_budget'] = budget_result

        results.append(sample_results)

    # Save per-sample results
    output_dir = Path(checkpoint_path).parent
    per_sample_file = output_dir / 'inference_per_sample.jsonl'

    print(f"\nSaving per-sample results to {per_sample_file}...")
    with open(per_sample_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")

    methods = ['text_baseline', 'latent', 'token_budget']
    aggregate = {}

    for method in methods:
        kv_caches = [r[method]['kv_cache_mb'] for r in results]
        latencies = [r[method]['latency_sec'] for r in results]
        ems = [r[method]['em'] for r in results]
        f1s = [r[method]['f1'] for r in results]

        aggregate[method] = {
            'num_samples': len(results),
            'kv_cache_mb': {
                'mean': float(sum(kv_caches) / len(kv_caches)),
                'min': float(min(kv_caches)),
                'max': float(max(kv_caches))
            },
            'latency_sec': {
                'mean': float(sum(latencies) / len(latencies)),
                'min': float(min(latencies)),
                'max': float(max(latencies))
            },
            'accuracy': {
                'em': float(sum(ems) / len(ems)),
                'f1': float(sum(f1s) / len(f1s))
            }
        }

    # Compute savings
    text_kv = aggregate['text_baseline']['kv_cache_mb']['mean']
    latent_kv = aggregate['latent']['kv_cache_mb']['mean']

    aggregate['kv_cache_savings'] = {
        'absolute_mb': text_kv - latent_kv,
        'relative_pct': (text_kv - latent_kv) / text_kv * 100
    }

    # Save aggregate results
    aggregate_file = output_dir / 'inference_aggregate.json'
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nSaved aggregate results to {aggregate_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 60)

    for method in methods:
        stats = aggregate[method]
        print(f"\n{method.upper()}:")
        print(f"  KV Cache:  {stats['kv_cache_mb']['mean']:>6.1f} MB (avg)")
        print(f"  Latency:   {stats['latency_sec']['mean']:>6.2f} sec (avg)")
        print(f"  Accuracy:  EM={stats['accuracy']['em']:.1%}, F1={stats['accuracy']['f1']:.1%}")

    print(f"\nKV CACHE SAVINGS (latent vs text):")
    print(f"  Absolute:  {aggregate['kv_cache_savings']['absolute_mb']:.1f} MB")
    print(f"  Relative:  {aggregate['kv_cache_savings']['relative_pct']:.1f}%")

    return aggregate


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference metrics')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to trained checkpoint')
    parser.add_argument('--source_model', type=str,
                      default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--target_model', type=str,
                      default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                      choices=['gsm8k', 'hotpotqa'])
    parser.add_argument('--num_samples', type=int, default=1319,
                      help='Number of samples to benchmark (use full test set)')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        source_model_id=args.source_model,
        target_model_id=args.target_model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )


if __name__ == '__main__':
    main()
