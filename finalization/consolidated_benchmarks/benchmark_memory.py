#!/usr/bin/env python3
"""
Memory Analysis Benchmark.

Addresses reviewer concern: "What's the memory overhead of your approach?"

Compares peak memory usage across methods:
- Bridge (Llama + Bridge + Mistral)
- LoRA (Mistral + LoRA adapter)
- Full Fine-tune (Mistral with gradient checkpointing)
- Direct inference (just Mistral)

Usage:
    python benchmark_memory.py --output_dir runs/memory_analysis
"""

import argparse
import json
import os
import gc
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Import bridge
from latent_bridge_v15 import LatentBridgeV15


class Args:
    """Args object for LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=16, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def count_parameters(model, trainable_only=True):
    """Count parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_peak_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def measure_direct_inference(device):
    """Measure memory for direct Mistral inference."""
    print("\n" + "=" * 60)
    print("MEASURING: Direct Mistral Inference")
    print("=" * 60)

    reset_memory_stats()
    mem_before = get_gpu_memory_mb()

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    mem_after_load = get_gpu_memory_mb()

    # Run inference
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    mem_after_inference = get_peak_gpu_memory_mb()

    result = {
        "method": "direct_inference",
        "model": "Mistral-7B",
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "mem_after_inference_mb": mem_after_inference,
        "model_memory_mb": mem_after_load - mem_before,
        "peak_memory_mb": mem_after_inference,
        "total_params": count_parameters(model, trainable_only=False),
        "trainable_params": 0,  # No training
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def measure_lora(device, rank=8):
    """Measure memory for LoRA training."""
    print("\n" + "=" * 60)
    print(f"MEASURING: LoRA Training (rank={rank})")
    print("=" * 60)

    reset_memory_stats()
    mem_before = get_gpu_memory_mb()

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    mem_after_load = get_gpu_memory_mb()
    trainable = count_parameters(model, trainable_only=True)

    # Simulate training forward pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    labels = inputs.input_ids.clone()

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    mem_after_train = get_peak_gpu_memory_mb()

    result = {
        "method": f"lora_r{rank}",
        "model": "Mistral-7B + LoRA",
        "lora_rank": rank,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "mem_after_train_mb": mem_after_train,
        "model_memory_mb": mem_after_load - mem_before,
        "peak_memory_mb": mem_after_train,
        "total_params": count_parameters(model, trainable_only=False),
        "trainable_params": trainable,
    }

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def measure_bridge(device, soft_tokens=16):
    """Measure memory for Bridge training."""
    print("\n" + "=" * 60)
    print(f"MEASURING: Bridge Training ({soft_tokens} tokens)")
    print("=" * 60)

    reset_memory_stats()
    mem_before = get_gpu_memory_mb()

    # Load Llama (sender)
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    mem_after_llama = get_gpu_memory_mb()

    # Load Mistral (receiver)
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    mem_after_both = get_gpu_memory_mb()

    # Freeze LLMs
    for p in llama.parameters():
        p.requires_grad = False
    for p in mistral.parameters():
        p.requires_grad = False

    # Create bridge
    bridge_args = Args(soft_tokens=soft_tokens)
    bridge = LatentBridgeV15(bridge_args, src_dim=4096, tgt_dim=4096, target_rms=0.03)
    bridge = bridge.to(device).to(torch.bfloat16)

    mem_after_bridge = get_gpu_memory_mb()
    bridge_params = count_parameters(bridge, trainable_only=True)

    # Simulate training
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    inputs = llama_tok("Hello, how are you?", return_tensors="pt").to(device)

    with torch.no_grad():
        llama_out = llama(**inputs, output_hidden_states=True)
        llama_hidden = llama_out.hidden_states[31]

    latents, aux_loss, _, _ = bridge(llama_hidden, inputs.attention_mask)

    # Forward through Mistral
    outputs = mistral(inputs_embeds=latents, labels=inputs.input_ids)
    loss = outputs.loss + aux_loss
    loss.backward()

    mem_after_train = get_peak_gpu_memory_mb()

    result = {
        "method": f"bridge_{soft_tokens}tok",
        "model": "Llama + Bridge + Mistral",
        "soft_tokens": soft_tokens,
        "mem_before_mb": mem_before,
        "mem_after_llama_mb": mem_after_llama,
        "mem_after_both_mb": mem_after_both,
        "mem_after_bridge_mb": mem_after_bridge,
        "mem_after_train_mb": mem_after_train,
        "llama_memory_mb": mem_after_llama - mem_before,
        "mistral_memory_mb": mem_after_both - mem_after_llama,
        "bridge_memory_mb": mem_after_bridge - mem_after_both,
        "peak_memory_mb": mem_after_train,
        "total_params": (count_parameters(llama, trainable_only=False) +
                        count_parameters(mistral, trainable_only=False) +
                        bridge_params),
        "trainable_params": bridge_params,
    }

    del llama, mistral, bridge, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def measure_full_finetune(device, num_layers=4):
    """Measure memory for full fine-tuning (last N layers)."""
    print("\n" + "=" * 60)
    print(f"MEASURING: Full Fine-tune (last {num_layers} layers)")
    print("=" * 60)

    reset_memory_stats()
    mem_before = get_gpu_memory_mb()

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    # Freeze most layers
    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.model.layers)
    start_layer = total_layers - num_layers
    for i in range(start_layer, total_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    mem_after_load = get_gpu_memory_mb()
    trainable = count_parameters(model, trainable_only=True)

    # Simulate training
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5
    )
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    labels = inputs.input_ids.clone()

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    mem_after_train = get_peak_gpu_memory_mb()

    result = {
        "method": f"full_ft_{num_layers}layers",
        "model": f"Mistral-7B (last {num_layers} layers)",
        "finetune_layers": num_layers,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "mem_after_train_mb": mem_after_train,
        "model_memory_mb": mem_after_load - mem_before,
        "peak_memory_mb": mem_after_train,
        "total_params": count_parameters(model, trainable_only=False),
        "trainable_params": trainable,
    }

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/memory_analysis")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "experiment": "memory_analysis",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "results": [],
    }

    # Run all measurements
    results["results"].append(measure_direct_inference(device))
    results["results"].append(measure_lora(device, rank=8))
    results["results"].append(measure_lora(device, rank=64))
    results["results"].append(measure_full_finetune(device, num_layers=4))
    results["results"].append(measure_full_finetune(device, num_layers=8))
    results["results"].append(measure_bridge(device, soft_tokens=8))
    results["results"].append(measure_bridge(device, soft_tokens=16))
    results["results"].append(measure_bridge(device, soft_tokens=32))

    # Summary table
    print("\n" + "=" * 80)
    print("MEMORY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<30} {'Peak Memory (MB)':<18} {'Trainable Params':<18}")
    print("-" * 80)

    for r in results["results"]:
        print(f"{r['method']:<30} {r['peak_memory_mb']:>12,.0f} MB    {r['trainable_params']:>15,}")

    # Save results
    output_file = f"{args.output_dir}/memory_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
