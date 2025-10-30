#!/usr/bin/env python3
"""
Diagnostic script to test hypotheses about Procrustes and adapter failures.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

print("=" * 80)
print("HYPOTHESIS TESTING: Cross-Model Alignment Issues")
print("=" * 80)

# ============================================================================
# Hypothesis 1: Procrustes numerical overflow
# ============================================================================
print("\n1. PROCRUSTES NUMERICAL STABILITY TEST")
print("-" * 40)

# Simulate hidden states from layer 32
torch.manual_seed(42)
num_tokens = 6057  # From log
hidden_dim = 4096  # Model dimension

# Test with realistic values (from log: mean norm ~150)
for scale in [1, 10, 100, 150, 200]:
    hidden_states = torch.randn(num_tokens, hidden_dim) * scale

    # Center
    centered = hidden_states - hidden_states.mean(dim=0, keepdim=True)

    # Try original method
    try:
        norm_original = torch.sqrt((centered ** 2).sum())
        print(f"Scale {scale:3d}: Original method = {norm_original.item():.2e}")
    except:
        print(f"Scale {scale:3d}: Original method = OVERFLOW")

    # Try stable method
    norm_stable = torch.norm(centered, 'fro')
    print(f"Scale {scale:3d}: Stable method   = {norm_stable.item():.2e}")

    # Check if they match (when not overflow)
    if not torch.isinf(norm_original):
        diff = abs(norm_original - norm_stable) / norm_stable
        print(f"Scale {scale:3d}: Relative diff   = {diff.item():.2e}")
    print()

# ============================================================================
# Hypothesis 2: Tokenizer length mismatch
# ============================================================================
print("\n2. TOKENIZER LENGTH MISMATCH TEST")
print("-" * 40)

# Load tokenizers
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# Test texts
test_texts = [
    "The capital of France is Paris.",
    "To solve this problem, we need to understand the underlying mathematics.",
    "Artificial intelligence is transforming how we interact with technology.",
]

print(f"{'Text':<50} {'Llama':<10} {'Mistral':<10} {'Diff':<10}")
print("-" * 80)

for text in test_texts:
    llama_tokens = llama_tokenizer(text, return_tensors="pt")["input_ids"]
    mistral_tokens = mistral_tokenizer(text, return_tensors="pt")["input_ids"]

    llama_len = llama_tokens.shape[1]
    mistral_len = mistral_tokens.shape[1]
    diff = llama_len - mistral_len

    print(f"{text[:47]+'...' if len(text) > 47 else text:<50} {llama_len:<10} {mistral_len:<10} {diff:<10}")

# ============================================================================
# Hypothesis 3: Hidden state scale across layers
# ============================================================================
print("\n3. HIDDEN STATE SCALE ANALYSIS")
print("-" * 40)
print("Loading models to check actual hidden state scales...")

# Load just the models (not for generation, just to check scales)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create a sample input
sample_text = "The future of artificial intelligence is"
llama_inputs = llama_tokenizer(sample_text, return_tensors="pt").to(device)

# Load model in inference mode
print("Loading Llama model...")
llama_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    torch_dtype=torch.float16
)

# Get hidden states at different layers
with torch.no_grad():
    outputs = llama_model(**llama_inputs, output_hidden_states=True)

    print(f"\n{'Layer':<10} {'Mean Norm':<15} {'Max Value':<15} {'Std Dev':<15}")
    print("-" * 55)

    for layer_idx in [0, 8, 16, 24, 32]:
        hidden = outputs.hidden_states[layer_idx]
        mean_norm = hidden.mean(dim=(0,1)).norm().item()
        max_val = hidden.abs().max().item()
        std_dev = hidden.std().item()

        print(f"Layer {layer_idx:<4} {mean_norm:<15.4f} {max_val:<15.4f} {std_dev:<15.4f}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

# ============================================================================
# Proposed Fixes
# ============================================================================
print("\nPROPOSED FIXES:")
print("-" * 40)
print("1. Procrustes: Use torch.norm(x, 'fro') instead of manual computation")
print("2. Procrustes: Normalize hidden states before computing alignment")
print("3. Adapters: Ensure sequence lengths match by padding/truncating")
print("4. Adapters: Use position_ids explicitly to handle length mismatch")
print("5. Both: Add input validation and better error messages")