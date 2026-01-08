#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to diagnose and fix SQuAD dataset loading issues.
This tests multiple approaches to loading the SQuAD dataset from HuggingFace.

Current year: 2025
Last checked: January 2025
"""

import sys
import json
from typing import List, Dict, Any

# Test 1: Check datasets library version
print("=" * 60)
print("STEP 1: Check environment and library versions")
print("=" * 60)

try:
    import datasets
    print(f"✓ datasets library version: {datasets.__version__}")
except ImportError:
    print("✗ datasets library not installed!")
    sys.exit(1)

try:
    import transformers
    print(f"✓ transformers library version: {transformers.__version__}")
except ImportError:
    print("  transformers library not installed (optional)")

print()

# Test 2: Try different loading approaches
print("=" * 60)
print("STEP 2: Test different SQuAD loading approaches")
print("=" * 60)

from datasets import load_dataset

def test_squad_loading(identifier: str, **kwargs):
    """Test loading a specific SQuAD dataset identifier."""
    print(f"\nTesting: {identifier}")
    print(f"  kwargs: {kwargs}")

    try:
        ds = load_dataset(identifier, **kwargs)

        # Check structure
        splits = list(ds.keys()) if hasattr(ds, 'keys') else []
        print(f"  ✓ Successfully loaded! Splits: {splits}")

        # Check train split
        if 'train' in splits or 'train' in str(splits):
            train_data = ds['train'] if 'train' in splits else ds.get('train')
            if train_data:
                print(f"    Train size: {len(train_data)}")

                # Check first example structure
                if len(train_data) > 0:
                    first = train_data[0]
                    print(f"    Keys in first example: {list(first.keys())}")

                    # Check answers structure
                    if 'answers' in first:
                        answers = first['answers']
                        print(f"    Answers type: {type(answers)}")
                        if hasattr(answers, 'keys'):
                            print(f"    Answers keys: {list(answers.keys())}")
                        if 'text' in answers or hasattr(answers, 'text'):
                            try:
                                text_val = answers.get('text') if hasattr(answers, 'get') else answers['text']
                                print(f"    Answer text type: {type(text_val)}")
                                if text_val:
                                    print(f"    First answer: {text_val[0] if isinstance(text_val, list) else text_val}")
                            except Exception as e:
                                print(f"    Could not access answer text: {e}")

        return True

    except Exception as e:
        print(f"  ✗ Failed with error: {type(e).__name__}: {e}")
        return False

# Test various identifiers and configurations
test_configurations = [
    # Option 1: Original rajpurkar paths
    ("rajpurkar/squad", {"split": "train"}),
    ("rajpurkar/squad", {"split": "train", "trust_remote_code": True}),
    ("rajpurkar/squad_v2", {"split": "train"}),
    ("rajpurkar/squad_v2", {"split": "train", "trust_remote_code": True}),

    # Option 2: Direct squad identifiers (common in newer datasets versions)
    ("squad", {"split": "train"}),
    ("squad", {"split": "train", "trust_remote_code": True}),
    ("squad_v2", {"split": "train"}),
    ("squad_v2", {"split": "train", "trust_remote_code": True}),

    # Option 3: Alternative sources
    ("lhoestq/squad", {"split": "train"}),
    ("SetFit/squad", {"split": "train"}),
]

successful_configs = []
for identifier, kwargs in test_configurations:
    if test_squad_loading(identifier, **kwargs):
        successful_configs.append((identifier, kwargs))

print("\n" + "=" * 60)
print("STEP 3: Summary of successful configurations")
print("=" * 60)

if successful_configs:
    print("\n✓ The following configurations worked:")
    for identifier, kwargs in successful_configs:
        print(f"  - load_dataset('{identifier}', {', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())})")
else:
    print("\n✗ No configurations worked! This might be a network or authentication issue.")

# Test 3: Test the actual function from data.py with fixes
print("\n" + "=" * 60)
print("STEP 4: Test improved load_squad_subset function")
print("=" * 60)

def load_squad_subset_fixed(split: str = "train", samples: int = 512, seed: int = 0, v2: bool = False) -> List[Dict[str, Any]]:
    """Fixed version of load_squad_subset that handles multiple loading approaches."""
    import random

    # Try different loading approaches in order of preference
    loading_approaches = [
        # Approach 1: Try direct squad/squad_v2 (works with newer datasets versions)
        (lambda: load_dataset("squad_v2" if v2 else "squad", split=split), "direct identifier"),

        # Approach 2: Try with rajpurkar prefix
        (lambda: load_dataset(f"rajpurkar/{'squad_v2' if v2 else 'squad'}", split=split), "rajpurkar prefix"),

        # Approach 3: Try with trust_remote_code for older versions
        (lambda: load_dataset("squad_v2" if v2 else "squad", split=split, trust_remote_code=True), "direct with trust_remote_code"),

        # Approach 4: rajpurkar with trust_remote_code
        (lambda: load_dataset(f"rajpurkar/{'squad_v2' if v2 else 'squad'}", split=split, trust_remote_code=True), "rajpurkar with trust_remote_code"),
    ]

    ds = None
    successful_approach = None

    for loader_func, approach_name in loading_approaches:
        try:
            ds = loader_func()
            successful_approach = approach_name
            print(f"  ✓ Successfully loaded using: {approach_name}")
            break
        except Exception as e:
            continue

    if ds is None:
        raise RuntimeError(f"Could not load SQuAD dataset using any approach")

    # Sample and process examples
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:samples]

    out = []
    for i in idxs[:3]:  # Just test first 3 examples
        ex = ds[i]

        # Robust extraction of fields
        context = str(ex.get("context", ex.get("passage", "")))
        question = str(ex.get("question", ""))

        # Handle answers field - it can be in various formats
        answer = ""
        if "answers" in ex:
            answers_obj = ex["answers"]

            # Try different ways to extract answer text
            if isinstance(answers_obj, dict):
                if "text" in answers_obj:
                    text_val = answers_obj["text"]
                    answer = str(text_val[0]) if isinstance(text_val, list) and text_val else str(text_val)
            elif hasattr(answers_obj, "text"):
                text_val = answers_obj.text
                answer = str(text_val[0]) if isinstance(text_val, list) and text_val else str(text_val)
            elif isinstance(answers_obj, list) and answers_obj:
                # Sometimes answers is directly a list of strings
                answer = str(answers_obj[0])

        source = f"Context: {context}\nQuestion: {question}\n"
        out.append({"source": source, "answer": answer})

    return out

# Test the fixed function
print("\nTesting fixed load_squad_subset function:")
try:
    examples = load_squad_subset_fixed(split="train", samples=10, seed=0, v2=False)
    print(f"  ✓ Successfully loaded {len(examples)} examples")
    if examples:
        print(f"\n  First example:")
        print(f"    Source (truncated): {examples[0]['source'][:200]}...")
        print(f"    Answer: {examples[0]['answer']}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Generate the recommended fix for data.py
print("\n" + "=" * 60)
print("STEP 5: Recommended fix for latentwire/data.py")
print("=" * 60)

print("""
Replace the load_squad_subset function (lines 176-210) with this improved version:

def load_squad_subset(split: str = "train", samples: int = 512, seed: int = 0, v2: bool = False) -> List[Dict[str, Any]]:
    # Try different loading approaches for better compatibility
    loading_approaches = [
        # Try direct identifier first (works with newer datasets versions)
        (lambda: load_dataset("squad_v2" if v2 else "squad", split=split), "direct"),
        # Then try with rajpurkar prefix
        (lambda: load_dataset(f"rajpurkar/{'squad_v2' if v2 else 'squad'}", split=split), "rajpurkar"),
        # Fallback with trust_remote_code for older versions
        (lambda: load_dataset("squad_v2" if v2 else "squad", split=split, trust_remote_code=True), "trust"),
    ]

    ds = None
    for loader_func, _ in loading_approaches:
        try:
            ds = loader_func()
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError(f"Could not load SQuAD{'v2' if v2 else ''} dataset")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:samples]
    out = []

    for i in idxs:
        ex = ds[i]
        context = str(ex.get("context", ""))
        question = str(ex.get("question", ""))

        # Robust answer extraction
        answer = ""
        if "answers" in ex:
            answers_obj = ex["answers"]
            if isinstance(answers_obj, dict) and "text" in answers_obj:
                text_val = answers_obj["text"]
                answer = str(text_val[0]) if isinstance(text_val, list) and text_val else str(text_val)
            elif hasattr(answers_obj, "text"):
                text_val = answers_obj.text
                answer = str(text_val[0]) if isinstance(text_val, list) and text_val else str(text_val)

        source = f"Context: {context}\\nQuestion: {question}\\n"
        out.append({"source": source, "answer": answer})

    return out
""")

print("\nThis fix:")
print("1. Tries multiple loading approaches for compatibility")
print("2. Handles different dataset library versions")
print("3. Uses robust field extraction to handle various data formats")
print("4. Falls back gracefully if one approach fails")