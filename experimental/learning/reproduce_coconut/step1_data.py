"""
Milestone 1: Data Loading & Exploration

Goal: Load GSM8k, understand format, verify we can use it

Research sources:
- HuggingFace GSM8k dataset page: https://huggingface.co/datasets/openai/gsm8k
- Dataset has 7,473 train and 1,319 test examples
- Fields: "question" and "answer"
- Answer format: reasoning steps with <<calculations>> and #### final_answer
"""

from datasets import load_dataset
import re


def parse_gsm8k_answer(answer: str) -> dict:
    """
    Parse GSM8k answer format to extract reasoning steps and final answer.

    Format example:
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72"

    Returns:
        dict with "reasoning" (str) and "final_answer" (str)
    """
    # Split by "####" to separate reasoning from final answer
    parts = answer.split("####")

    if len(parts) == 2:
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
    elif len(parts) > 2:
        # Multiple #### markers (rare edge case)
        reasoning = parts[0].strip()
        final_answer = parts[-1].strip()
    else:
        # No #### marker (edge case)
        reasoning = answer.strip()
        final_answer = ""

    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
        "has_final_answer": bool(final_answer)
    }


def format_as_cot(question: str, answer: str) -> str:
    """
    Format GSM8k example in standard CoT format for training.

    Format: Question: {q}\nAnswer: Let's think step by step. {reasoning}\nThe answer is {final_answer}
    """
    parsed = parse_gsm8k_answer(answer)

    # Clean reasoning (remove << >> calculation annotations for cleaner CoT)
    reasoning_clean = re.sub(r'<<[^>]+>>', '', parsed["reasoning"])
    reasoning_clean = reasoning_clean.strip()

    cot_format = (
        f"Question: {question}\n"
        f"Answer: Let's think step by step. {reasoning_clean}\n"
        f"The answer is {parsed['final_answer']}"
    )

    return cot_format


def validate_dataset(dataset):
    """
    Validate that all examples can be parsed correctly.

    Returns: dict with validation statistics
    """
    train_valid = 0
    train_invalid = 0
    test_valid = 0
    test_invalid = 0

    # Check train set
    for example in dataset["train"]:
        parsed = parse_gsm8k_answer(example["answer"])
        if parsed["has_final_answer"]:
            train_valid += 1
        else:
            train_invalid += 1

    # Check test set
    for example in dataset["test"]:
        parsed = parse_gsm8k_answer(example["answer"])
        if parsed["has_final_answer"]:
            test_valid += 1
        else:
            test_invalid += 1

    return {
        "train_valid": train_valid,
        "train_invalid": train_invalid,
        "test_valid": test_valid,
        "test_invalid": test_invalid
    }


def main():
    print("=" * 80)
    print("MILESTONE 1: GSM8k Data Loading & Exploration")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading GSM8k dataset from HuggingFace...")
    print("Dataset: openai/gsm8k, config: main")
    print()

    dataset = load_dataset("openai/gsm8k", "main")

    # Show dataset statistics
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Total examples: {len(dataset['train']) + len(dataset['test'])}")
    print()

    print("Fields in each example:")
    print(f"  - {list(dataset['train'][0].keys())}")
    print()

    # Validate dataset
    print("Validating dataset format...")
    validation = validate_dataset(dataset)
    print(f"  Train: {validation['train_valid']} valid, {validation['train_invalid']} invalid")
    print(f"  Test: {validation['test_valid']} valid, {validation['test_invalid']} invalid")
    if validation['train_invalid'] > 0 or validation['test_invalid'] > 0:
        print("  ⚠️  Warning: Some examples missing final answer (####)")
    else:
        print("  ✓ All examples have final answer marker (####)")
    print()

    # Compute reasoning length statistics (for training planning)
    print("Computing reasoning length statistics...")
    reasoning_lengths = []
    train_sample = dataset["train"].select(range(min(1000, len(dataset["train"]))))
    for example in train_sample:
        parsed = parse_gsm8k_answer(example["answer"])
        # Count tokens approximately (split by whitespace)
        tokens = len(parsed["reasoning"].split())
        reasoning_lengths.append(tokens)

    avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
    max_length = max(reasoning_lengths)
    min_length = min(reasoning_lengths)
    print(f"  Average reasoning length: {avg_length:.1f} tokens")
    print(f"  Min/Max: {min_length} / {max_length} tokens")
    print(f"  (Based on {len(reasoning_lengths)} train examples)")
    print()

    # Show 3 example problems
    print("=" * 80)
    print("EXAMPLE 1: Raw Format (from dataset)")
    print("=" * 80)
    example1 = dataset["train"][0]
    print(f"Question: {example1['question']}")
    print()
    print(f"Answer (raw):\n{example1['answer']}")
    print()

    # Parse and show components
    parsed1 = parse_gsm8k_answer(example1['answer'])
    print("Parsed components:")
    print(f"  Reasoning: {parsed1['reasoning'][:100]}...")
    print(f"  Final answer: {parsed1['final_answer']}")
    print()

    # Show CoT format
    print("=" * 80)
    print("EXAMPLE 1: Chain-of-Thought Format (for training)")
    print("=" * 80)
    cot1 = format_as_cot(example1['question'], example1['answer'])
    print(cot1)
    print()

    # Show 2 more examples in CoT format
    print("=" * 80)
    print("EXAMPLE 2: Chain-of-Thought Format")
    print("=" * 80)
    example2 = dataset["train"][1]
    cot2 = format_as_cot(example2['question'], example2['answer'])
    print(cot2)
    print()

    print("=" * 80)
    print("EXAMPLE 3: Chain-of-Thought Format")
    print("=" * 80)
    example3 = dataset["train"][2]
    cot3 = format_as_cot(example3['question'], example3['answer'])
    print(cot3)
    print()

    # Test set example
    print("=" * 80)
    print("TEST SET EXAMPLE")
    print("=" * 80)
    test_example = dataset["test"][0]
    cot_test = format_as_cot(test_example['question'], test_example['answer'])
    print(cot_test)
    print()

    print("=" * 80)
    print("MILESTONE 1: COMPLETE ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Successfully loaded GSM8k dataset")
    print("  ✓ Verified train/test splits")
    print("  ✓ Parsed answer format (reasoning + final answer)")
    print("  ✓ Created CoT formatting function for training")
    print()
    print("Next step: Milestone 2 - Add special tokens to Llama 3.1 tokenizer")


if __name__ == "__main__":
    main()
