#!/usr/bin/env python3
"""
Test script for TREC dataset loader.
Run with: python scripts/test_trec_loader.py
"""

import sys
sys.path.insert(0, '.')

from latentwire.data import load_trec_subset, load_examples
from collections import Counter

print("="*60)
print("Testing TREC Dataset Loader")
print("="*60)

# Test 1: Load full test set
print("\n[Test 1] Loading full test set (500 examples)...")
test_data = load_trec_subset(split="test", samples=None, seed=0)
print(f"Loaded {len(test_data)} examples")

# Check label distribution
labels = [ex["answer"] for ex in test_data]
label_counts = Counter(labels)
print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    pct = 100 * count / len(test_data)
    print(f"  {label}: {count:3d} ({pct:5.2f}%)")

# Show a few examples
print("\nSample examples:")
for i in range(min(5, len(test_data))):
    ex = test_data[i]
    source_lines = ex["source"].strip().split("\n")
    question = source_lines[0].replace("Question: ", "")
    print(f"  [{ex['answer']}] {question}")

# Test 2: Load subset
print("\n[Test 2] Loading 50-example subset...")
subset_data = load_trec_subset(split="test", samples=50, seed=42)
print(f"Loaded {len(subset_data)} examples")

# Test 3: Load via unified interface
print("\n[Test 3] Loading via load_examples()...")
unified_data = load_examples(dataset="trec", split="test", samples=10)
print(f"Loaded {len(unified_data)} examples")
print("First example:")
print(f"  Source: {unified_data[0]['source'].strip()}")
print(f"  Answer: {unified_data[0]['answer']}")

# Test 4: Load training set
print("\n[Test 4] Loading 100 examples from training set...")
train_data = load_trec_subset(split="train", samples=100, seed=0)
print(f"Loaded {len(train_data)} examples")

print("\n" + "="*60)
print("All tests passed successfully!")
print("="*60)
