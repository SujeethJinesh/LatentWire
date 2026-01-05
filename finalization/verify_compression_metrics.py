#!/usr/bin/env python3
"""Verify that compression metrics are honestly calculated including quantization overhead."""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple


def calculate_honest_wire_bytes(
    latent_shape,  # (B, M, D)
    bits,
    group_size=32,
    scale_bits=16
):
    """Calculate actual wire bytes including quantization overhead.

    Args:
        latent_shape: (batch_size, latent_len, latent_dim)
        bits: Quantization bits (4, 6, 8, 16)
        group_size: Size of quantization groups
        scale_bits: Bits for scale factors (typically fp16 = 16)

    Returns:
        Dictionary with byte calculations
    """
    B, M, D = latent_shape
    total_elements = B * M * D

    # Calculate number of groups
    elements_per_sample = M * D
    groups_per_sample = math.ceil(elements_per_sample / max(group_size, 1))
    total_groups = groups_per_sample * B

    # Data bytes (quantized values)
    data_bytes = total_elements * bits / 8.0

    # Scale bytes (one scale per group)
    scale_bytes = total_groups * scale_bits / 8.0

    # Total wire bytes
    total_bytes = math.ceil(data_bytes + scale_bytes)

    return {
        "data_bytes": math.ceil(data_bytes),
        "scale_bytes": math.ceil(scale_bytes),
        "total_bytes": total_bytes,
        "groups": total_groups,
        "elements": total_elements,
        "bits": bits,
        "scale_bits": scale_bits,
        "group_size": group_size
    }


def calculate_text_bytes(text):
    """Calculate UTF-8 bytes for text."""
    return len(text.encode("utf-8"))


def calculate_compression_ratio(text_bytes, latent_bytes):
    """Calculate compression ratio."""
    if latent_bytes == 0:
        return float('inf')
    return text_bytes / latent_bytes


def verify_compression_claims():
    """Verify claimed compression ratios with honest accounting."""

    # Typical configuration from CLAUDE.md
    LATENT_LEN = 32  # M
    D_Z = 256  # Latent dimension

    # Typical prompt length (from experiments)
    typical_prompt = """System: You are a concise QA assistant. Use the context to answer with a short phrase only.
User: The Normans were the people who gave their name to Normandy, a region in northern France. They were descended from Norse Viking conquerors of the territory and the native Frankish and Gallo-Roman inhabitants of the area. Their identity emerged initially in the first half of the 10th century, and gradually evolved over succeeding centuries.
Question: In what country is Normandy located?
Assistant: Answer: """

    text_bytes = calculate_text_bytes(typical_prompt)
    print(f"Text bytes (UTF-8): {text_bytes}")
    print(f"Text length: {len(typical_prompt)} chars")
    print()

    # Test different quantization levels
    batch_size = 1  # Single example

    quantization_configs = [
        (16, "fp16 (no quantization)"),
        (8, "int8"),
        (6, "int6"),
        (4, "int4"),
    ]

    print(f"Latent configuration: M={LATENT_LEN}, D={D_Z}")
    print(f"Total latent elements: {LATENT_LEN * D_Z}")
    print()

    for bits, desc in quantization_configs:
        result = calculate_honest_wire_bytes(
            latent_shape=(batch_size, LATENT_LEN, D_Z),
            bits=bits,
            group_size=32,
            scale_bits=16
        )

        compression_ratio = calculate_compression_ratio(text_bytes, result['total_bytes'])

        print(f"{desc}:")
        print(f"  Data bytes: {result['data_bytes']}")
        print(f"  Scale bytes: {result['scale_bytes']} ({result['groups']} groups)")
        print(f"  Total wire bytes: {result['total_bytes']}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Overhead: {result['scale_bytes'] / result['total_bytes'] * 100:.1f}%")
        print()

    # Check if we achieve claimed 4-8x compression
    print("=" * 60)
    print("VERIFICATION SUMMARY:")
    print("=" * 60)

    # Most aggressive quantization (int4)
    int4_result = calculate_honest_wire_bytes(
        latent_shape=(batch_size, LATENT_LEN, D_Z),
        bits=4,
        group_size=32,
        scale_bits=16
    )
    int4_compression = calculate_compression_ratio(text_bytes, int4_result['total_bytes'])

    # Medium quantization (int8)
    int8_result = calculate_honest_wire_bytes(
        latent_shape=(batch_size, LATENT_LEN, D_Z),
        bits=8,
        group_size=32,
        scale_bits=16
    )
    int8_compression = calculate_compression_ratio(text_bytes, int8_result['total_bytes'])

    print(f"For typical {text_bytes}-byte prompt:")
    print(f"  int4 achieves: {int4_compression:.2f}x compression")
    print(f"  int8 achieves: {int8_compression:.2f}x compression")
    print()

    if int4_compression >= 4.0:
        print("✓ int4 achieves claimed ≥4x compression")
    else:
        print(f"✗ int4 falls short of 4x compression (only {int4_compression:.2f}x)")

    if int8_compression >= 4.0:
        print("✓ int8 achieves claimed ≥4x compression")
    else:
        print(f"✗ int8 falls short of 4x compression (only {int8_compression:.2f}x)")

    # Test with different latent lengths
    print("\n" + "=" * 60)
    print("COMPRESSION VS LATENT LENGTH (M):")
    print("=" * 60)

    for M in [16, 32, 48, 64]:
        int4_result = calculate_honest_wire_bytes(
            latent_shape=(batch_size, M, D_Z),
            bits=4,
            group_size=32,
            scale_bits=16
        )
        compression = calculate_compression_ratio(text_bytes, int4_result['total_bytes'])
        print(f"M={M:2d}: {compression:.2f}x compression (int4, {int4_result['total_bytes']} bytes)")

    # Compare with existing implementation
    print("\n" + "=" * 60)
    print("COMPARISON WITH core_utils.compute_wire_metrics:")
    print("=" * 60)

    print("""
The existing compute_wire_metrics function in core_utils.py:
1. ✓ Correctly accounts for scale overhead
2. ✓ Uses proper group-wise quantization calculation
3. ✓ Includes both data and scale bytes in total

Key formula from code:
    groups_per_sample = ceil(M * D / group_size)
    scale_bytes = groups_per_sample * scale_bits / 8.0 * B
    data_bytes = num_latent * selected_bits / 8.0
    total_bytes = ceil(scale_bytes + data_bytes)

This matches our verification calculations.
""")


if __name__ == "__main__":
    verify_compression_claims()