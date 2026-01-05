#!/usr/bin/env python
"""Test script to verify memory configurations are correct."""

from memory_configs import get_memory_safe_config

def test_llama8b_mistral7b():
    """Test that Llama-8B + Mistral-7B gives batch_size=2."""
    config = get_memory_safe_config(
        source_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        target_model="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=1536,
        soft_tokens=128,
    )

    assert config["batch_size"] == 2, f"Expected batch_size=2, got {config['batch_size']}"
    assert config["gradient_accumulation_steps"] == 4, f"Expected grad_accum=4, got {config['gradient_accumulation_steps']}"
    assert config["estimated_memory_gb"] < 80.0, f"Memory estimate {config['estimated_memory_gb']} exceeds 80GB!"

    print("✅ Llama-8B + Mistral-7B config verified:")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"   Estimated memory: {config['estimated_memory_gb']:.1f} GB")
    print(f"   Memory breakdown:")
    breakdown = config["memory_breakdown"]
    print(f"     - Models: {breakdown['source_model_gb'] + breakdown['target_model_gb']:.1f} GB")
    print(f"     - Model optimizers: {breakdown['source_optimizer_gb'] + breakdown['target_optimizer_gb']:.1f} GB")
    print(f"     - Bridge: {breakdown['bridge_gb']:.1f} GB")
    print(f"     - Activations: {breakdown['total_activation_gb']:.1f} GB")
    print(f"     - Safety margin: {breakdown['safety_margin_gb']:.1f} GB")

def test_conservative_batch_sizes():
    """Test that all configs are conservative enough."""
    test_cases = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", None),  # Single 8B
        ("meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),  # 3B + 3B
        ("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),  # Small models
    ]

    for source, target in test_cases:
        config = get_memory_safe_config(source, target)
        model_desc = f"{source.split('/')[-1]} → {target.split('/')[-1]}" if target else source.split('/')[-1]

        assert config["estimated_memory_gb"] < 75.0, f"{model_desc}: Memory {config['estimated_memory_gb']:.1f}GB too close to 80GB limit!"
        assert config["batch_size"] >= 1, f"{model_desc}: Invalid batch_size {config['batch_size']}"

        print(f"✅ {model_desc}: batch_size={config['batch_size']}, memory={config['estimated_memory_gb']:.1f}GB")

if __name__ == "__main__":
    print("Testing memory configurations...\n")
    test_llama8b_mistral7b()
    print("\nTesting conservative batch sizes...")
    test_conservative_batch_sizes()
    print("\n✅ All tests passed!")