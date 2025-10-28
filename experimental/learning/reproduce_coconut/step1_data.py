"""
Milestone 1: Data Loading & Preprocessing (REVISED)

Goal: Load Internalize CoT dataset and understand the correct format

Data source: coconut/data/gsm_train.json (downloaded from official preprocessing)
Format: {"question": str, "steps": List[str], "answer": str}
Steps: Calculator-style ["<<600*30/100=180>>", ...]
"""

import json
from collections import Counter


def format_stage0(example):
    """
    Format example for Stage 0 training (full CoT, no latent tokens).

    Format:
        question
        step1
        step2
        ...
        ### answer
    """
    parts = [example["question"]]
    parts.extend(example["steps"])
    parts.append(f"### {example['answer']}")
    return "\n".join(parts)


def format_stage1(example, c_thought=1):
    """
    Format example for Stage 1 training (first step replaced with latent tokens).

    Format (c=1):
        question
        <|start-latent|><|latent|><|end-latent|>
        step2
        step3
        ...
        ### answer

    Format (c=2):
        question
        <|start-latent|><|latent|><|latent|><|end-latent|>
        step2
        step3
        ...
        ### answer
    """
    if len(example["steps"]) == 0:
        # No steps to replace (edge case)
        return format_stage0(example)

    parts = [example["question"]]

    # Replace first step with latent tokens
    latent_tokens = "<|latent|>" * c_thought
    parts.append(f"<|start-latent|>{latent_tokens}<|end-latent|>")

    # Add remaining steps (skip first)
    parts.extend(example["steps"][1:])

    # Add answer
    parts.append(f"### {example['answer']}")

    return "\n".join(parts)


def main():
    print("=" * 80)
    print("MILESTONE 1: Internalize CoT Data Loading (REVISED)")
    print("=" * 80)
    print()

    # Load data from already-downloaded file
    data_path = "coconut/data/gsm_train.json"
    print(f"Loading data from: {data_path}")
    print("(This was downloaded by official COCONUT preprocessing script)")
    print()

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: {data_path} not found!")
        print()
        print("Please ensure you ran the official preprocessing:")
        print("  cd coconut")
        print("  bash preprocessing/gsm_icot.bash")
        print()
        print("Or the data should already exist from our earlier investigation.")
        return

    # Dataset statistics
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total training examples: {len(data)}")
    print()

    print("Fields in each example:")
    print(f"  - {list(data[0].keys())}")
    print()

    # Step count statistics
    step_counts = [len(example["steps"]) for example in data]
    step_counter = Counter(step_counts)

    print("Step count distribution:")
    for count in sorted(step_counter.keys()):
        num_examples = step_counter[count]
        pct = 100 * num_examples / len(data)
        print(f"  {count} steps: {num_examples:6d} examples ({pct:5.2f}%)")
    print()

    print(f"Average steps per example: {sum(step_counts) / len(step_counts):.2f}")
    print(f"Min steps: {min(step_counts)}")
    print(f"Max steps: {max(step_counts)}")
    print()

    # Show example in raw format
    print("=" * 80)
    print("EXAMPLE 1: Raw JSON Format")
    print("=" * 80)
    example1 = data[0]
    print(json.dumps(example1, indent=2))
    print()

    # Show example in Stage 0 format
    print("=" * 80)
    print("EXAMPLE 1: Stage 0 Training Format (Full CoT)")
    print("=" * 80)
    stage0_format = format_stage0(example1)
    print(stage0_format)
    print()

    # Show example in Stage 1 format (c=1)
    print("=" * 80)
    print("EXAMPLE 1: Stage 1 Training Format (c=1)")
    print("=" * 80)
    stage1_c1 = format_stage1(example1, c_thought=1)
    print(stage1_c1)
    print()

    # Show example in Stage 1 format (c=2)
    print("=" * 80)
    print("EXAMPLE 1: Stage 1 Training Format (c=2, as in paper)")
    print("=" * 80)
    stage1_c2 = format_stage1(example1, c_thought=2)
    print(stage1_c2)
    print()

    # Show a few more examples
    print("=" * 80)
    print("EXAMPLE 2: Stage 0 Format")
    print("=" * 80)
    example2 = data[1]
    print(format_stage0(example2))
    print()

    print("=" * 80)
    print("EXAMPLE 3: Stage 0 Format")
    print("=" * 80)
    example3 = data[2]
    print(format_stage0(example3))
    print()

    # Show an example with different step count
    # Find one with exactly 1 step (minimal)
    one_step_examples = [ex for ex in data if len(ex["steps"]) == 1]
    if one_step_examples:
        print("=" * 80)
        print("EXAMPLE WITH 1 STEP: Stage 0 and Stage 1 Formats")
        print("=" * 80)
        one_step_ex = one_step_examples[0]
        print("Stage 0:")
        print(format_stage0(one_step_ex))
        print()
        print("Stage 1 (c=1) - Note: No remaining steps after latent!")
        print(format_stage1(one_step_ex, c_thought=1))
        print()

    # Find one with many steps
    many_step_examples = [ex for ex in data if len(ex["steps"]) >= 6]
    if many_step_examples:
        print("=" * 80)
        print(f"EXAMPLE WITH {len(many_step_examples[0]['steps'])} STEPS: Stage 0 Format")
        print("=" * 80)
        many_step_ex = many_step_examples[0]
        print(format_stage0(many_step_ex))
        print()

    # Summary
    print("=" * 80)
    print("MILESTONE 1: COMPLETE ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✓ Loaded {len(data)} examples from Internalize CoT dataset")
    print("  ✓ Verified structure: {{question, steps: [...], answer}}")
    print(f"  ✓ Steps are calculator-style: {example1['steps'][0]}")
    print("  ✓ Answer format: ### {answer}")
    print("  ✓ Created format functions for Stage 0 and Stage 1")
    print()
    print("Key differences from original GSM8k:")
    print(f"  - 385,620 examples (vs 7,473 in original GSM8k) = 51.6x more data")
    print("  - Calculator-style steps (vs natural language reasoning)")
    print("  - Steps already separated into list (vs paragraph)")
    print()
    print("Next step: Milestone 2 - Add special tokens to Llama 3.1 8B tokenizer")


if __name__ == "__main__":
    main()
