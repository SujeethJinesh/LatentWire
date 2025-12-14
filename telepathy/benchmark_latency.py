#!/usr/bin/env python
# telepathy/benchmark_latency.py
"""
Latency Benchmark: Compare inference times for Bridge vs Text-Relay vs Direct Text

This script answers the question: "Is Bridge actually faster than alternatives?"

Methods compared:
1. Bridge: Llama hidden state → Bridge (8 soft tokens) → Mistral classify
2. Text-Relay: Llama generate summary (~50 tokens) → Mistral classify
3. Direct Text: Mistral reads full text → classify

Also captures qualitative examples for paper figures.
"""
import os
import torch
import json
import time
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F

from latent_bridge_v15 import LatentBridgeV15


class Args:
    """Args object for LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=8, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def load_models(device):
    """Load Llama and Mistral models."""
    print("Loading Mistral...")
    mistral_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    print("Loading Llama...")
    llama_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    return llama_model, llama_tok, mistral_model, mistral_tok


def load_bridge(checkpoint_path, device, soft_tokens=8):
    """Load a trained bridge checkpoint."""
    args = Args(soft_tokens=soft_tokens)
    bridge = LatentBridgeV15(args, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    bridge.load_state_dict(ckpt['bridge_state_dict'])
    bridge.eval()

    return bridge


def measure_bridge_latency(bridge, llama_model, mistral_model, llama_tok, mistral_tok,
                           texts, device, num_warmup=5, num_trials=50):
    """Measure Bridge inference latency."""
    print("\n" + "=" * 60)
    print("BRIDGE LATENCY MEASUREMENT")
    print("=" * 60)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for text in texts[:num_warmup]:
        src_inputs = llama_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            src_out = llama_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]
            latents, _, _, _ = bridge(src_hidden, src_inputs.attention_mask)

            # Prepare for Mistral
            tgt_embed = mistral_model.get_input_embeddings()
            prefix = mistral_tok.encode("Classify:", add_special_tokens=False)
            prefix_embeds = tgt_embed(torch.tensor([prefix], device=device))
            combined = torch.cat([latents, prefix_embeds], dim=1)
            _ = mistral_model(inputs_embeds=combined, use_cache=False)

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Timed runs
    encode_times = []
    bridge_times = []
    decode_times = []
    total_times = []

    print(f"Running {num_trials} timed trials...")
    for text in tqdm(texts[:num_trials], desc="Bridge"):
        src_inputs = llama_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)

        # Phase 1: Llama encode
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        with torch.no_grad():
            src_out = llama_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()

        # Phase 2: Bridge transform
        with torch.no_grad():
            latents, _, _, _ = bridge(src_hidden, src_inputs.attention_mask)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t2 = time.perf_counter()

        # Phase 3: Mistral decode (single forward pass, no generation)
        with torch.no_grad():
            tgt_embed = mistral_model.get_input_embeddings()
            prefix = mistral_tok.encode("Classify:", add_special_tokens=False)
            prefix_embeds = tgt_embed(torch.tensor([prefix], device=device))
            combined = torch.cat([latents, prefix_embeds], dim=1)
            _ = mistral_model(inputs_embeds=combined, use_cache=False)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t3 = time.perf_counter()

        encode_times.append(t1 - t0)
        bridge_times.append(t2 - t1)
        decode_times.append(t3 - t2)
        total_times.append(t3 - t0)

    return {
        "method": "bridge",
        "soft_tokens": bridge.num_soft_tokens if hasattr(bridge, 'num_soft_tokens') else 8,
        "encode_ms": sum(encode_times) / len(encode_times) * 1000,
        "bridge_ms": sum(bridge_times) / len(bridge_times) * 1000,
        "decode_ms": sum(decode_times) / len(decode_times) * 1000,
        "total_ms": sum(total_times) / len(total_times) * 1000,
        "num_trials": num_trials,
    }


def measure_text_relay_latency(llama_model, mistral_model, llama_tok, mistral_tok,
                               texts, device, num_warmup=5, num_trials=50, max_summary_tokens=50):
    """Measure Text-Relay inference latency."""
    print("\n" + "=" * 60)
    print("TEXT-RELAY LATENCY MEASUREMENT")
    print("=" * 60)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for text in texts[:num_warmup]:
        prompt = f"Summarize in one sentence:\n\n{text[:256]}\n\nSummary:"
        inputs = llama_tok(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs, max_new_tokens=max_summary_tokens, do_sample=False,
                pad_token_id=llama_tok.eos_token_id
            )
            summary = llama_tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            classify_prompt = f"Classify the sentiment: {summary}\n\nAnswer:"
            classify_inputs = mistral_tok(classify_prompt, return_tensors="pt").to(device)
            _ = mistral_model.generate(
                **classify_inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=mistral_tok.eos_token_id
            )

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Timed runs
    summarize_times = []
    classify_times = []
    total_times = []
    summary_lengths = []

    print(f"Running {num_trials} timed trials...")
    for text in tqdm(texts[:num_trials], desc="Text-Relay"):
        prompt = f"Summarize in one sentence:\n\n{text[:256]}\n\nSummary:"
        inputs = llama_tok(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)

        # Phase 1: Llama generate summary
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs, max_new_tokens=max_summary_tokens, do_sample=False,
                pad_token_id=llama_tok.eos_token_id
            )
            summary = llama_tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()

        summary_lengths.append(len(llama_tok.encode(summary)))

        # Phase 2: Mistral classify from summary
        classify_prompt = f"Classify the sentiment: {summary}\n\nAnswer:"
        classify_inputs = mistral_tok(classify_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = mistral_model.generate(
                **classify_inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=mistral_tok.eos_token_id
            )
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t2 = time.perf_counter()

        summarize_times.append(t1 - t0)
        classify_times.append(t2 - t1)
        total_times.append(t2 - t0)

    return {
        "method": "text_relay",
        "avg_summary_tokens": sum(summary_lengths) / len(summary_lengths),
        "summarize_ms": sum(summarize_times) / len(summarize_times) * 1000,
        "classify_ms": sum(classify_times) / len(classify_times) * 1000,
        "total_ms": sum(total_times) / len(total_times) * 1000,
        "num_trials": num_trials,
    }


def measure_direct_text_latency(mistral_model, mistral_tok, texts, device,
                                 num_warmup=5, num_trials=50):
    """Measure Direct Text classification latency (Mistral only)."""
    print("\n" + "=" * 60)
    print("DIRECT TEXT LATENCY MEASUREMENT")
    print("=" * 60)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for text in texts[:num_warmup]:
        prompt = f"Classify the sentiment of this text as positive or negative:\n\n{text[:256]}\n\nAnswer:"
        inputs = mistral_tok(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        with torch.no_grad():
            _ = mistral_model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=mistral_tok.eos_token_id
            )

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Timed runs
    total_times = []
    input_lengths = []

    print(f"Running {num_trials} timed trials...")
    for text in tqdm(texts[:num_trials], desc="Direct Text"):
        prompt = f"Classify the sentiment of this text as positive or negative:\n\n{text[:256]}\n\nAnswer:"
        inputs = mistral_tok(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        input_lengths.append(inputs['input_ids'].shape[1])

        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = mistral_model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=mistral_tok.eos_token_id
            )
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()

        total_times.append(t1 - t0)

    return {
        "method": "direct_text",
        "avg_input_tokens": sum(input_lengths) / len(input_lengths),
        "total_ms": sum(total_times) / len(total_times) * 1000,
        "num_trials": num_trials,
    }


def capture_qualitative_examples(llama_model, llama_tok, texts, labels, device, num_examples=5):
    """Capture text-relay summaries for qualitative analysis."""
    print("\n" + "=" * 60)
    print("QUALITATIVE EXAMPLES: Text-Relay Summaries")
    print("=" * 60)

    examples = []
    for i, (text, label) in enumerate(zip(texts[:num_examples], labels[:num_examples])):
        prompt = f"Summarize in one sentence:\n\n{text[:256]}\n\nSummary:"
        inputs = llama_tok(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)

        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=llama_tok.eos_token_id
            )
            summary = llama_tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        example = {
            "original": text[:200],
            "summary": summary.strip(),
            "label": label,
            "original_tokens": len(llama_tok.encode(text[:256])),
            "summary_tokens": len(llama_tok.encode(summary)),
        }
        examples.append(example)

        print(f"\n[{i+1}] Label: {label}")
        print(f"    Original ({example['original_tokens']} tokens): \"{text[:80]}...\"")
        print(f"    Summary ({example['summary_tokens']} tokens): \"{summary[:80]}...\"")

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to bridge checkpoint (optional)")
    parser.add_argument("--num_trials", type=int, default=50, help="Number of trials for timing")
    parser.add_argument("--output_dir", type=str, default="runs/latency_benchmark")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    llama_model, llama_tok, mistral_model, mistral_tok = load_models(device)

    # Load SST-2 dataset for consistent testing
    print("\nLoading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2", split="validation")
    texts = [item['sentence'] for item in dataset]
    labels = ["positive" if item['label'] == 1 else "negative" for item in dataset]

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "num_trials": args.num_trials,
    }

    # Measure Direct Text latency
    direct_results = measure_direct_text_latency(
        mistral_model, mistral_tok, texts, device,
        num_trials=args.num_trials
    )
    results["direct_text"] = direct_results
    print(f"\nDirect Text: {direct_results['total_ms']:.1f}ms avg")

    # Measure Text-Relay latency
    relay_results = measure_text_relay_latency(
        llama_model, mistral_model, llama_tok, mistral_tok, texts, device,
        num_trials=args.num_trials
    )
    results["text_relay"] = relay_results
    print(f"Text-Relay: {relay_results['total_ms']:.1f}ms avg ({relay_results['avg_summary_tokens']:.0f} summary tokens)")

    # Measure Bridge latency (if checkpoint provided)
    if args.checkpoint and os.path.exists(args.checkpoint):
        bridge = load_bridge(args.checkpoint, device)
        bridge_results = measure_bridge_latency(
            bridge, llama_model, mistral_model, llama_tok, mistral_tok, texts, device,
            num_trials=args.num_trials
        )
        results["bridge"] = bridge_results
        print(f"Bridge: {bridge_results['total_ms']:.1f}ms avg (8 soft tokens)")

    # Capture qualitative examples
    examples = capture_qualitative_examples(llama_model, llama_tok, texts, labels, device)
    results["qualitative_examples"] = examples

    # Print summary
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Direct Text (Mistral): {results['direct_text']['total_ms']:.1f}ms")
    print(f"Text-Relay (Llama→text→Mistral): {results['text_relay']['total_ms']:.1f}ms")
    if 'bridge' in results:
        print(f"Bridge (Llama→soft tokens→Mistral): {results['bridge']['total_ms']:.1f}ms")
        speedup_vs_relay = results['text_relay']['total_ms'] / results['bridge']['total_ms']
        print(f"\nBridge is {speedup_vs_relay:.1f}x faster than Text-Relay")

    # Save results
    output_file = os.path.join(args.output_dir, "latency_benchmark.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
