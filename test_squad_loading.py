#!/usr/bin/env python3
"""Test script to diagnose SQuAD dataset loading issue."""

from datasets import load_dataset
import traceback

print("Testing SQuAD dataset loading...")
print("=" * 60)

# Test 1: Load without config
print("\nTest 1: Loading rajpurkar/squad without config")
try:
    ds = load_dataset("rajpurkar/squad", split="train")
    print(f"OK - Success! Loaded {len(ds)} examples")

    # Inspect first example
    ex = ds[0]
    print(f"\nFirst example keys: {ex.keys()}")
    print(f"Type of example: {type(ex)}")

    # Try to access fields
    print("\nTrying to access fields:")
    try:
        context = ex["context"]
        print(f"OK - context (bracket): {context[:100]}...")
    except Exception as e:
        print(f"FAIL - context (bracket): {e}")

    try:
        question = ex["question"]
        print(f"OK - question (bracket): {question}")
    except Exception as e:
        print(f"FAIL - question (bracket): {e}")

    try:
        answers = ex["answers"]
        print(f"OK - answers (bracket): {answers}")
        print(f"  Type: {type(answers)}")
        if isinstance(answers, dict):
            print(f"  Keys: {answers.keys()}")
            if "text" in answers:
                print(f"  Text field: {answers['text']}")
    except Exception as e:
        print(f"FAIL - answers (bracket): {e}")

except Exception as e:
    print(f"FAIL - Failed: {e}")
    traceback.print_exc()

# Test 2: Load with plain_text config
print("\n" + "=" * 60)
print("\nTest 2: Loading rajpurkar/squad with 'plain_text' config")
try:
    ds = load_dataset("rajpurkar/squad", "plain_text", split="train")
    print(f"OK - Success! Loaded {len(ds)} examples")

    # Inspect first example
    ex = ds[0]
    print(f"\nFirst example keys: {ex.keys()}")
    print(f"Type of example: {type(ex)}")

except Exception as e:
    print(f"FAIL - Failed: {e}")
    traceback.print_exc()

# Test 3: Check what configs are available
print("\n" + "=" * 60)
print("\nTest 3: Checking available configs")
try:
    from datasets import get_dataset_config_names
    configs = get_dataset_config_names("rajpurkar/squad")
    print(f"Available configs: {configs}")
except Exception as e:
    print(f"Could not get configs: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
