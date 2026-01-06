#!/usr/bin/env python3
"""
Integration patch for adding proper GPU timing to run_unified_comparison.py

This script shows how to integrate the fixed latency utilities into the existing
unified comparison script to get accurate GPU timing measurements.

Key changes needed:
1. Import latency_utils_fixed
2. Replace time.time() with measure_gpu_latency()
3. Add proper CUDA synchronization
4. Use LatencyProfiler for comprehensive benchmarking
"""

import torch
import time
import numpy as np
from typing import Callable, List, Tuple


def patch_eval_bridge(original_eval_bridge):
    """
    Patch the eval_bridge function to use proper GPU timing.

    Replace lines like:
        start = time.time()
        # ... inference code ...
        latencies.append(time.time() - start)

    With:
        from telepathy.latency_utils_fixed import measure_gpu_latency

        def inference_fn():
            # ... inference code ...
            return outputs

        latency_ms, _ = measure_gpu_latency(inference_fn, num_iterations=1, warmup=0)
        latencies.append(latency_ms / 1000)  # Convert to seconds
    """

    def patched_eval_bridge(bridge, sender, sender_tok, receiver, receiver_tok,
                           eval_ds, dataset_name, source_layer, device):
        """Patched version with proper GPU timing."""
        from telepathy.latency_utils_fixed import measure_gpu_latency

        config = DATASET_CONFIGS[dataset_name]
        label_tokens = get_label_tokens(receiver_tok, dataset_name)

        correct = 0
        total = 0
        latencies = []

        for item in tqdm(eval_ds, desc="Bridge eval"):
            text = item["text"]
            label = item["label"]

            # Create inference function for timing
            def inference_fn():
                # Sender
                sender_inputs = sender_tok(
                    text, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(device)

                with torch.no_grad():
                    sender_out = sender(**sender_inputs, output_hidden_states=True)
                    sender_hidden = sender_out.hidden_states[source_layer]

                    # Bridge
                    soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])

                    # Receiver
                    prompt = f"\n{config['task_prompt']}\nAnswer:"
                    prompt_inputs = receiver_tok(prompt, return_tensors="pt", add_special_tokens=False)
                    prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))
                    inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

                    # Create attention mask
                    soft_mask = torch.ones(1, soft_tokens.shape[1], device=device)
                    full_mask = torch.cat([soft_mask, prompt_inputs["attention_mask"].to(device)], dim=1)

                    outputs = receiver(inputs_embeds=inputs_embeds, attention_mask=full_mask)
                    return outputs.logits[0, -1]

            # Measure with proper GPU timing
            latency_ms, _ = measure_gpu_latency(inference_fn, num_iterations=1, warmup=0)
            latencies.append(latency_ms / 1000)  # Convert to seconds for compatibility

            # Get prediction (run once more for the actual result)
            logits = inference_fn()
            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            if pred == label:
                correct += 1
            total += 1

        return {
            "accuracy": 100.0 * correct / total,
            "correct": correct,
            "total": total,
            "latency_ms": np.mean(latencies) * 1000,  # Convert back to ms
            "latency_std_ms": np.std(latencies) * 1000,
            "latency_p95_ms": np.percentile(latencies, 95) * 1000,
            "latency_p99_ms": np.percentile(latencies, 99) * 1000,
        }

    return patched_eval_bridge


def add_comprehensive_benchmarking(script_path: str):
    """
    Add comprehensive latency benchmarking to unified comparison.

    This would add a new --benchmark flag that runs detailed latency analysis.
    """

    benchmark_code = '''
def run_latency_benchmark(args, sender, receiver, sender_tok, receiver_tok, eval_ds, bridge):
    """Run comprehensive latency benchmark using LatencyProfiler."""
    from telepathy.latency_utils_fixed import ComparativeLatencyBenchmark

    print("\\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE LATENCY BENCHMARK")
    print("=" * 80)

    # Create benchmark
    benchmark = ComparativeLatencyBenchmark(
        warmup_iterations=10,
        measurement_iterations=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path(args.output_dir) / "latency_benchmark"
    )

    # Sample a batch for testing
    test_batch = list(eval_ds.take(args.benchmark_batch_size))

    # Create inference functions for each approach
    def bridge_inference():
        for item in test_batch:
            # ... bridge inference code ...
            pass

    def text_relay_inference():
        for item in test_batch:
            # ... text relay inference code ...
            pass

    def zero_shot_inference():
        for item in test_batch:
            # ... zero shot inference code ...
            pass

    # Run benchmark
    results = benchmark.compare_approaches(
        bridge_fn=bridge_inference,
        text_relay_fn=text_relay_inference,
        zero_shot_fn=zero_shot_inference,
        batch_size=args.benchmark_batch_size,
        num_tokens=20  # Approximate tokens per response
    )

    return results

# Add to argparse
parser.add_argument("--benchmark", action="store_true",
                    help="Run comprehensive latency benchmark")
parser.add_argument("--benchmark_batch_size", type=int, default=8,
                    help="Batch size for latency benchmark")

# Add to main execution
if args.benchmark:
    benchmark_results = run_latency_benchmark(
        args, sender, receiver, sender_tok, receiver_tok,
        eval_ds, bridge
    )
    all_seeds_results["latency_benchmark"] = benchmark_results
'''

    return benchmark_code


def create_minimal_fix():
    """
    Create minimal fix that can be applied immediately to get accurate timing.

    This is the simplest integration that requires minimal code changes.
    """

    fix_code = '''
# Add this import at the top of run_unified_comparison.py
from telepathy.latency_utils_fixed import measure_gpu_latency

# Then in eval_bridge, eval_text_relay, etc., replace:
#     start = time.time()
#     ... model inference ...
#     latencies.append(time.time() - start)

# With:
#     def inference():
#         ... model inference ...
#         return result
#
#     latency_ms, _ = measure_gpu_latency(inference, num_iterations=1)
#     latencies.append(latency_ms / 1000)  # Keep in seconds for compatibility

# Or even simpler, just add synchronization:
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start = time.time()
#     ... model inference ...
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     latencies.append(time.time() - start)
'''

    return fix_code


def main():
    """Display integration instructions."""

    print("=" * 80)
    print("GPU TIMING INTEGRATION GUIDE")
    print("=" * 80)

    print("\nCURRENT ISSUES in run_unified_comparison.py:")
    print("1. Uses time.time() without GPU synchronization")
    print("2. Measures Python overhead, not actual GPU compute time")
    print("3. No warmup iterations for stable measurements")
    print("4. No percentile metrics (p95, p99)")

    print("\nQUICK FIX (minimal changes):")
    print("-" * 40)
    print(create_minimal_fix())

    print("\nFULL INTEGRATION:")
    print("-" * 40)
    print("1. Import latency_utils_fixed")
    print("2. Use measure_gpu_latency() for all timing")
    print("3. Add --benchmark flag for detailed analysis")
    print("4. Report p50, p95, p99 latencies")

    print("\nEXPECTED IMPACT:")
    print("-" * 40)
    print("• More accurate latency measurements")
    print("• Stable results (less variance)")
    print("• Fair comparison between approaches")
    print("• Publication-ready metrics")

    print("\nTO APPLY:")
    print("-" * 40)
    print("1. Copy latency_utils_fixed.py to telepathy/")
    print("2. Update run_unified_comparison.py with fixes")
    print("3. Run with --benchmark flag for detailed analysis")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()