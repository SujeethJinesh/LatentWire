#!/usr/bin/env python3
"""
Quick script to check TREC dataset label distribution.
Run with: python scripts/check_trec_distribution.py
"""

from datasets import load_dataset
from collections import Counter
import json

# Load dataset
print("Loading TREC dataset...")
# Use SetFit/TREC-QC - modern parquet format, no loading scripts
dataset = load_dataset('SetFit/TREC-QC')

# Define coarse label names
COARSE_LABELS = {
    0: "ABBR",  # Abbreviation
    1: "ENTY",  # Entity
    2: "DESC",  # Description
    3: "HUM",   # Human
    4: "LOC",   # Location
    5: "NUM"    # Numeric
}

print("\n" + "="*60)
print("TREC Question Classification Dataset Statistics")
print("="*60)

for split_name in ['train', 'test']:
    split_data = dataset[split_name]

    print(f"\n{split_name.upper()} Split:")
    print(f"  Total examples: {len(split_data)}")

    # Count coarse labels (SetFit uses 'label_coarse' field)
    coarse_counts = Counter(split_data['label_coarse'])

    print(f"\n  Coarse Label Distribution:")
    for label_id in sorted(coarse_counts.keys()):
        count = coarse_counts[label_id]
        pct = 100 * count / len(split_data)
        print(f"    {COARSE_LABELS[label_id]:4s} ({label_id}): {count:4d} ({pct:5.2f}%)")

    # Show a few examples
    print(f"\n  Sample questions:")
    for i in range(min(3, len(split_data))):
        example = split_data[i]
        label_name = COARSE_LABELS[example['label_coarse']]
        print(f"    [{label_name}] {example['text']}")

print("\n" + "="*60)
print("Dataset Features (SetFit/TREC-QC format):")
print("  - text: question string")
print("  - label_coarse: 0-5 (ABBR, ENTY, DESC, HUM, LOC, NUM)")
print("  - label_coarse_text: coarse label name")
print("  - label: 0-49 (fine-grained class ID)")
print("  - label_text: fine-grained label name")
print("="*60)
