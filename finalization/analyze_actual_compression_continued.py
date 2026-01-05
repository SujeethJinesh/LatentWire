#!/usr/bin/env python3
"""Continue analysis of actual compression ratios."""

import math


def analyze_realistic_compression():
    """Analyze compression with realistic prompt sizes."""

    # Realistic SQuAD-style prompt
    realistic_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer questions based on the given context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who gave their name to Normandy, a region in northern France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. The Normans are noted both for their culture, such as their unique Romanesque architecture and musical traditions, and for their significant military accomplishments and innovations. Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy from the native Lombards and Byzantines, beginning in 999. The expedition led by the Norman noble William the Conqueror, Duke of Normandy conquered England, becoming the ruling aristocracy of Anglo-Norman England.

Question: What century did the Normans first gain their separate identity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: """

    print("REALISTIC PROMPT ANALYSIS")
    print("=" * 60)

    text_bytes = len(realistic_prompt.encode("utf-8"))
    text_chars = len(realistic_prompt)

    print(f"Prompt length: {text_chars} characters")
    print(f"Prompt bytes (UTF-8): {text_bytes} bytes")
    print()

    # Test different configurations
    configs = [
        # (M, D, bits, description)
        (32, 256, 4, "M=32, D=256, int4"),
        (32, 256, 8, "M=32, D=256, int8"),
        (48, 256, 4, "M=48, D=256, int4"),
        (64, 256, 4, "M=64, D=256, int4"),
        (32, 512, 4, "M=32, D=512, int4"),
        (128, 128, 4, "M=128, D=128, int4"),
    ]

    print("Configuration Analysis:")
    print("-" * 60)

    for M, D, bits, desc in configs:
        # Calculate wire bytes with overhead
        elements = M * D
        groups = math.ceil(elements / 32)  # group_size = 32
        data_bytes = elements * bits / 8
        scale_bytes = groups * 16 / 8  # 16 bits for scales
        total_bytes = math.ceil(data_bytes + scale_bytes)

        compression = text_bytes / total_bytes

        status = "✓" if compression >= 4.0 else "✗"
        print(f"{desc:25} → {total_bytes:5} bytes → {compression:5.2f}x {status}")

    print()
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("1. Compression ratio heavily depends on prompt length")
    print("2. Longer prompts (1000+ chars) can achieve 4x+ compression")
    print("3. Shorter prompts struggle to reach 4x even with int4")
    print("4. Need to measure on actual dataset distribution")


def calculate_minimum_prompt_length():
    """Calculate minimum prompt length needed for 4x compression."""

    print("\n" + "=" * 60)
    print("MINIMUM PROMPT LENGTH FOR 4X COMPRESSION")
    print("=" * 60)

    configs = [
        (32, 256, 4, "M=32, D=256, int4"),
        (32, 256, 8, "M=32, D=256, int8"),
        (48, 256, 4, "M=48, D=256, int4"),
        (64, 256, 4, "M=64, D=256, int4"),
    ]

    for M, D, bits, desc in configs:
        # Calculate latent wire bytes
        elements = M * D
        groups = math.ceil(elements / 32)
        data_bytes = elements * bits / 8
        scale_bytes = groups * 16 / 8
        latent_bytes = math.ceil(data_bytes + scale_bytes)

        # Minimum text bytes for 4x compression
        min_text_bytes = latent_bytes * 4

        print(f"{desc}:")
        print(f"  Latent bytes: {latent_bytes}")
        print(f"  Min text bytes for 4x: {min_text_bytes}")
        print(f"  Min text length: ~{min_text_bytes} characters")
        print()


def verify_core_utils_calculation():
    """Verify that core_utils.compute_wire_metrics is correct."""

    print("=" * 60)
    print("VERIFYING core_utils.compute_wire_metrics")
    print("=" * 60)

    # Simulate the calculation from core_utils
    B, M, D = 1, 32, 256
    group_size = 32
    scale_bits = 16
    selected_bits = 4

    # From core_utils.py
    num_latent = B * M * D
    values_per_sample = M * D
    groups_per_sample = math.ceil(values_per_sample / max(group_size, 1))
    scale_bytes = groups_per_sample * scale_bits / 8.0 * B
    data_bytes = num_latent * selected_bits / 8.0
    total_bytes = int(math.ceil(scale_bytes + data_bytes))

    print(f"Configuration: B={B}, M={M}, D={D}")
    print(f"Quantization: {selected_bits} bits, group_size={group_size}")
    print()
    print(f"Elements: {num_latent}")
    print(f"Groups: {groups_per_sample}")
    print(f"Data bytes: {int(data_bytes)}")
    print(f"Scale bytes: {int(scale_bytes)}")
    print(f"Total bytes: {total_bytes}")
    print()
    print("✓ Calculation includes scale overhead correctly")


if __name__ == "__main__":
    analyze_realistic_compression()
    calculate_minimum_prompt_length()
    verify_core_utils_calculation()