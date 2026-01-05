#!/usr/bin/env python3
"""Fix compression metric calculation to properly report byte-level compression."""

import sys
from pathlib import Path


def create_patch():
    """Create a patch to fix compression metrics in eval.py."""

    print("COMPRESSION METRIC FIX")
    print("=" * 60)
    print()

    print("ISSUE IDENTIFIED:")
    print("-" * 40)
    print("The current code calculates TOKEN compression ratio:")
    print("  compression = avg_prompt_tokens / latent_len")
    print()
    print("This gives ~10-20x 'compression' but is misleading because:")
    print("1. It compares tokens to latent vectors (apples to oranges)")
    print("2. Doesn't account for actual wire bytes")
    print("3. Doesn't include quantization overhead")
    print()

    print("PROPER CALCULATION:")
    print("-" * 40)
    print("Compression should be calculated as:")
    print("  compression = text_bytes / latent_wire_bytes")
    print()
    print("Where:")
    print("  text_bytes = len(prompt.encode('utf-8'))")
    print("  latent_wire_bytes = quantized_data_bytes + scale_bytes")
    print()

    print("RECOMMENDED FIX:")
    print("-" * 40)
    print("""
In latentwire/eval.py, around line 1572, change:

OLD:
    "compression": {name: text_results[name]["avg_prompt_tokens"] / max(latent_len, 1) for name in model_contexts},

NEW:
    "byte_compression": _calculate_byte_compression(model_outputs, wire),
    "token_compression": {name: text_results[name]["avg_prompt_tokens"] / max(latent_len, 1) for name in model_contexts},

And add this helper function:

def _calculate_byte_compression(model_outputs, wire):
    '''Calculate actual byte-level compression ratio.'''
    compressions = {}

    # Get latent wire bytes (with quantization overhead)
    latent_bytes = wire.get("selected_latent_bytes")
    if latent_bytes is None:
        # Fallback to fp16 if no quantization
        latent_bytes = wire["latent_bytes"].get("fp16", wire["latent_bytes"]["fp32"])

    # Calculate for each model
    for name in ["llama", "qwen"]:
        if name in model_outputs:
            prompts = model_outputs[name]["chat_prompts"]
            # Calculate average text bytes
            text_bytes = sum(len(p.encode("utf-8")) for p in prompts) / len(prompts)
            compressions[name] = text_bytes / latent_bytes if latent_bytes > 0 else 0

    return compressions
""")

    print()
    print("IMPACT ON REPORTED RESULTS:")
    print("-" * 40)
    print("With M=32, D=256, int4 quantization:")
    print("  Latent wire bytes: 4,608")
    print("  Typical prompt: ~1,500 bytes")
    print("  ACTUAL compression: 0.33x (expansion, not compression!)")
    print()
    print("To achieve 4x compression, need prompts of 18,432+ bytes")
    print("This is ~18K characters, much longer than typical QA prompts")
    print()

    print("HONEST REPORTING:")
    print("-" * 40)
    print("The paper should report:")
    print("1. Byte-level compression (may be <1x for short prompts)")
    print("2. Token-level compression (current metric)")
    print("3. Clarify that compression depends heavily on prompt length")
    print("4. Show compression curves vs prompt length")
    print()


if __name__ == "__main__":
    create_patch()