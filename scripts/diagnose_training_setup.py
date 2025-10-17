#!/usr/bin/env python3
"""
Diagnostic script to analyze training/eval setup issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data import load_examples
from transformers import AutoTokenizer
import numpy as np

def analyze_dataset(dataset='squad', samples=1000, seed=42):
    """Analyze answer lengths and characteristics."""

    print("="*80)
    print(f"DATASET ANALYSIS: {dataset}")
    print("="*80)

    # Load data
    examples = load_examples(dataset=dataset, split='train', samples=samples, seed=seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Analyze answer lengths
    answer_lengths = []
    source_lengths = []

    print(f"\nAnalyzing {len(examples)} examples...")

    for ex in examples:
        # Tokenize answer
        answer_toks = tokenizer(ex['answer'], add_special_tokens=False)['input_ids']
        source_toks = tokenizer(ex['source'], add_special_tokens=False)['input_ids']

        answer_lengths.append(len(answer_toks))
        source_lengths.append(len(source_toks))

    # Statistics
    answer_lengths = np.array(answer_lengths)
    source_lengths = np.array(source_lengths)

    print("\n" + "-"*80)
    print("ANSWER TOKEN LENGTHS")
    print("-"*80)
    print(f"Mean: {answer_lengths.mean():.2f} tokens")
    print(f"Median: {np.median(answer_lengths):.0f} tokens")
    print(f"Min: {answer_lengths.min()} tokens")
    print(f"Max: {answer_lengths.max()} tokens")
    print(f"95th percentile: {np.percentile(answer_lengths, 95):.0f} tokens")
    print(f"99th percentile: {np.percentile(answer_lengths, 99):.0f} tokens")

    print("\nPercentage of answers exceeding max_new_tokens:")
    for threshold in [12, 20, 32, 64]:
        pct = (answer_lengths > threshold).mean() * 100
        print(f"  > {threshold} tokens: {pct:.1f}%")

    print("\n" + "-"*80)
    print("SOURCE (CONTEXT + QUESTION) TOKEN LENGTHS")
    print("-"*80)
    print(f"Mean: {source_lengths.mean():.2f} tokens")
    print(f"Median: {np.median(source_lengths):.0f} tokens")
    print(f"Min: {source_lengths.min()} tokens")
    print(f"Max: {source_lengths.max()} tokens")

    # Sample examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES (first 10)")
    print("="*80)

    for i in range(min(10, len(examples))):
        ex = examples[i]
        answer_len = answer_lengths[i]

        print(f"\n[{i+1}] Answer length: {answer_len} tokens")
        print(f"  Answer: {ex['answer'][:100]}")
        print(f"  Question: {ex['source'].split('Question:')[1][:80] if 'Question:' in ex['source'] else 'N/A'}")

    print("\n" + "="*80)
    print("TRAINING/EVAL MISMATCH ANALYSIS")
    print("="*80)

    print("\n1. ANSWER LENGTH MISMATCH:")
    print(f"   - Training: Answers truncated to 64 tokens (line 272 of train_sequence_compression.py)")
    print(f"   - Eval: Generation limited to 12 tokens (max_new_tokens=12)")
    print(f"   - Problem: {(answer_lengths > 12).mean()*100:.1f}% of answers exceed 12 tokens!")
    print(f"   - Impact: Evaluation artificially truncates answers, lowering F1 scores")

    print("\n2. NO ANCHOR TEXT:")
    print("   - Training input: [compressed_256_tokens] [answer_embed_1, answer_embed_2, ...]")
    print("   - Eval input:     [compressed_256_tokens] → generate")
    print("   - Problem: No signal telling model to start generating (no 'Answer:', no BOS)")
    print("   - Compare to LatentWire: [compressed] [BOS] 'Answer: ' → generate")

    print("\n3. TEACHER FORCING MISMATCH:")
    print("   - Training: Model sees gold answer embeddings at each step")
    print("   - Eval: Model generates autoregressively from its own predictions")
    print("   - Problem: Model never practices generating from compressed tokens alone")

    print("\n4. LOSS OBJECTIVE:")
    print("   - Loss: Cross-entropy on answer tokens only")
    print("   - Compressed tokens NOT supervised (labels=-100)")
    print("   - Problem: No direct signal for compression quality")
    print("   - Compressor learns to minimize CE on teacher-forced answers")
    print("   - No guarantee these representations work for autoregressive generation")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\n1. IMMEDIATE FIXES:")
    print("   a) Increase max_new_tokens from 12 → 32 (covers 95% of answers)")
    print("   b) Add anchor text: tokenize 'Answer: ' and prepend to generation")
    print("   c) Prepend BOS token after compressed sequence")

    print("\n2. ARCHITECTURE FIXES:")
    print("   a) Add anchor embeddings between compressed tokens and answer")
    print("   b) Train with anchor: [compressed] [anchor='Answer: '] [answer]")
    print("   c) Eval with anchor: [compressed] [anchor='Answer: '] → generate")

    print("\n3. TRAINING FIXES:")
    print("   a) Add KL divergence loss on compressed→answer transition")
    print("   b) Scheduled sampling: mix teacher forcing with autoregressive")
    print("   c) Add auxiliary loss on first generated token")

    print()

if __name__ == '__main__':
    analyze_dataset(dataset='squad', samples=1000, seed=42)
