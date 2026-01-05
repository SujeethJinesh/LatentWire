"""
GSM8K Mathematical Reasoning Evaluation Module

Complete evaluation framework for GSM8K (Grade School Math 8K) dataset.

Dataset:
    - Source: openai/gsm8k (main configuration)
    - Train: 7,473 examples
    - Test: 1,319 examples
    - Format: Grade-school level math word problems with chain-of-thought solutions

Evaluation Protocol:
    - 8-shot chain-of-thought prompting (standard from lm-evaluation-harness)
    - Answer extraction via regex: #### {numerical_answer}
    - Metric: Exact match on final numerical answer
    - Generation: max_new_tokens=256 for reasoning chains

Answer Format:
    - Dataset format: "explanation #### numerical_answer"
    - CoT format: "The answer is {number}"
    - Normalization: Remove commas, handle decimals and negatives

Usage:
    # Load dataset
    train_data = load_gsm8k_dataset(split="train", samples=1000)
    test_data = load_gsm8k_dataset(split="test", samples=200)

    # Format prompts
    prompt = format_gsm8k_prompt(question, few_shot=True)

    # Extract answer
    gold_answer = extract_numerical_answer(dataset_answer)
    pred_answer = extract_numerical_answer(model_output)

    # Compute metrics
    accuracy = compute_gsm8k_accuracy(predictions, gold_answers)

References:
    - Original paper: https://arxiv.org/abs/2110.14168
    - lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
    - Standard 8-shot CoT from official evaluation protocol
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from datasets import load_dataset

# =============================================================================
# 8-Shot CoT Examples (Standard from lm-evaluation-harness)
# =============================================================================

# These examples are the canonical 8-shot demonstrations used in GSM8K evaluation
# Format: "Q: {question}\nA: {chain_of_thought}. The answer is {number}."
GSM8K_FEW_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.

"""


# =============================================================================
# Data Loading
# =============================================================================

def load_gsm8k_dataset(
    split: str = "test",
    samples: Optional[int] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split ("train" or "test")
        samples: Number of samples to load (None = all)
        seed: Random seed for shuffling

    Returns:
        List of dicts with keys:
            - question: Math word problem
            - answer: Full answer with explanation and #### marker
            - numerical_answer: Extracted numerical answer

    Note:
        - Train set: 7,473 examples
        - Test set: 1,319 examples
        - Standard evaluation uses full test set
    """
    import random

    # Load from HuggingFace
    ds = load_dataset("openai/gsm8k", "main", split=split)

    # Sample if requested
    if samples is not None and samples < len(ds):
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[:samples]
    else:
        indices = list(range(len(ds)))

    # Process examples
    examples = []
    for idx in indices:
        ex = ds[idx]
        question = ex["question"]
        answer = ex["answer"]

        # Pre-extract numerical answer for convenience
        numerical_answer = extract_numerical_answer(answer)

        examples.append({
            "question": question,
            "answer": answer,
            "numerical_answer": numerical_answer,
        })

    return examples


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_numerical_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from GSM8K format.

    Handles multiple formats:
        1. Dataset format: "explanation #### NUMBER"
        2. CoT format: "The answer is NUMBER"
        3. Fallback: Last number in text

    Normalization:
        - Removes commas from numbers (e.g., "1,234" -> "1234")
        - Preserves decimals (e.g., "3.14")
        - Preserves negative signs (e.g., "-5")

    Args:
        text: Raw text containing answer

    Returns:
        Normalized numerical answer as string, or None if no number found

    Examples:
        >>> extract_numerical_answer("She has 42 apples. #### 42")
        '42'
        >>> extract_numerical_answer("The answer is 1,234")
        '1234'
        >>> extract_numerical_answer("So the total is -5.5 dollars.")
        '-5.5'
    """
    if not text:
        return None

    # Pattern for numbers: optional negative, digits with optional commas, optional decimal
    number_pattern = r'-?\d+(?:,\d+)*(?:\.\d+)?'

    # Priority 1: Look for #### marker (dataset format)
    match = re.search(r'####\s*(' + number_pattern + r')', text)
    if match:
        return match.group(1).replace(',', '')

    # Priority 2: Look for "The answer is X" (CoT format)
    match = re.search(r'[Tt]he answer is\s*(' + number_pattern + r')', text)
    if match:
        return match.group(1).replace(',', '')

    # Priority 3: Look for "answer: X" or "Answer: X"
    match = re.search(r'[Aa]nswer\s*:?\s*(' + number_pattern + r')', text)
    if match:
        return match.group(1).replace(',', '')

    # Fallback: Last number in text
    numbers = re.findall(number_pattern, text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def normalize_number(num_str: str) -> str:
    """
    Normalize a number string for comparison.

    Args:
        num_str: Number as string

    Returns:
        Normalized string (commas removed, whitespace stripped)

    Examples:
        >>> normalize_number("1,234")
        '1234'
        >>> normalize_number(" -5.5 ")
        '-5.5'
    """
    if not num_str:
        return ""
    return num_str.replace(',', '').strip()


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_gsm8k_prompt(
    question: str,
    few_shot: bool = True,
    format_style: str = "standard",
) -> str:
    """
    Format a GSM8K prompt for evaluation.

    Args:
        question: Math word problem
        few_shot: Whether to include 8-shot demonstrations
        format_style: Prompt format ("standard" or "instruct")
            - "standard": "Q: {question}\nA:" (lm-evaluation-harness format)
            - "instruct": "Solve this math problem:\n{question}\nSolution:"

    Returns:
        Formatted prompt string

    Note:
        Standard GSM8K evaluation uses few_shot=True with "standard" format.
    """
    if format_style == "standard":
        # Standard lm-evaluation-harness format
        if few_shot:
            prompt = GSM8K_FEW_SHOT_EXAMPLES + f"Q: {question}\nA:"
        else:
            prompt = f"Q: {question}\nA:"

    elif format_style == "instruct":
        # Alternative instruction-following format
        if few_shot:
            # Convert few-shot to instruction format
            examples = []
            for ex in GSM8K_FEW_SHOT_EXAMPLES.strip().split('\n\n'):
                if not ex.strip():
                    continue
                # Parse Q: ... A: ...
                parts = ex.split('\nA: ')
                if len(parts) == 2:
                    q = parts[0].replace('Q: ', '')
                    a = parts[1]
                    examples.append(f"Question: {q}\nSolution: {a}")

            few_shot_text = '\n\n'.join(examples) + '\n\n'
            prompt = few_shot_text + f"Question: {question}\nSolution:"
        else:
            prompt = f"Question: {question}\nSolution:"

    else:
        raise ValueError(f"Unknown format_style: {format_style}")

    return prompt


# =============================================================================
# Metrics
# =============================================================================

def compute_gsm8k_accuracy(
    predictions: List[str],
    gold_answers: List[str],
) -> Tuple[float, int, int]:
    """
    Compute exact match accuracy for GSM8K.

    Args:
        predictions: Model predictions (can be full text or just numbers)
        gold_answers: Gold standard answers (can be full text or just numbers)

    Returns:
        Tuple of (accuracy, num_correct, total)

    Note:
        Both predictions and gold_answers are automatically normalized
        and have numerical answers extracted.
    """
    if len(predictions) != len(gold_answers):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(gold_answers)} gold"
        )

    correct = 0
    total = 0

    for pred, gold in zip(predictions, gold_answers):
        # Extract numerical answers
        pred_num = extract_numerical_answer(pred)
        gold_num = extract_numerical_answer(gold)

        # Skip if either is None (no number found)
        if pred_num is None or gold_num is None:
            total += 1
            continue

        # Normalize and compare
        pred_normalized = normalize_number(pred_num)
        gold_normalized = normalize_number(gold_num)

        if pred_normalized == gold_normalized:
            correct += 1

        total += 1

    accuracy = 100.0 * correct / max(total, 1)
    return accuracy, correct, total


def evaluate_gsm8k_batch(
    predictions: List[str],
    gold_examples: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a batch of GSM8K predictions with detailed metrics.

    Args:
        predictions: Model predictions (raw text)
        gold_examples: Gold examples from load_gsm8k_dataset()
        verbose: Whether to print per-example results

    Returns:
        Dict containing:
            - accuracy: Overall accuracy (0-100)
            - correct: Number of correct predictions
            - total: Total number of predictions
            - details: List of per-example results (if verbose=True)
    """
    gold_answers = [ex["answer"] for ex in gold_examples]
    accuracy, correct, total = compute_gsm8k_accuracy(predictions, gold_answers)

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    if verbose:
        details = []
        for i, (pred, gold_ex) in enumerate(zip(predictions, gold_examples)):
            pred_num = extract_numerical_answer(pred)
            gold_num = gold_ex["numerical_answer"]

            is_correct = (
                pred_num is not None and
                gold_num is not None and
                normalize_number(pred_num) == normalize_number(gold_num)
            )

            details.append({
                "index": i,
                "question": gold_ex["question"],
                "gold_answer": gold_num,
                "predicted_answer": pred_num,
                "prediction_text": pred[:200],  # Truncate for readability
                "correct": is_correct,
            })

        results["details"] = details

    return results


# =============================================================================
# Evaluation Helpers
# =============================================================================

def print_gsm8k_examples(
    predictions: List[str],
    gold_examples: List[Dict[str, Any]],
    num_examples: int = 5,
    show_correct: bool = True,
    show_incorrect: bool = True,
):
    """
    Print sample predictions for inspection.

    Args:
        predictions: Model predictions
        gold_examples: Gold examples
        num_examples: Maximum examples to print
        show_correct: Whether to show correct examples
        show_incorrect: Whether to show incorrect examples
    """
    shown = 0

    for i, (pred, gold_ex) in enumerate(zip(predictions, gold_examples)):
        if shown >= num_examples:
            break

        pred_num = extract_numerical_answer(pred)
        gold_num = gold_ex["numerical_answer"]

        is_correct = (
            pred_num is not None and
            gold_num is not None and
            normalize_number(pred_num) == normalize_number(gold_num)
        )

        # Filter based on correctness
        if is_correct and not show_correct:
            continue
        if not is_correct and not show_incorrect:
            continue

        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"

        print(f"\n{'='*80}")
        print(f"Example {i} - {status}")
        print(f"{'='*80}")
        print(f"Question: {gold_ex['question']}")
        print(f"\nGold answer: {gold_num}")
        print(f"Predicted answer: {pred_num}")
        print(f"\nFull prediction:\n{pred[:300]}...")

        shown += 1


# =============================================================================
# Dataset Statistics
# =============================================================================

def analyze_gsm8k_dataset(split: str = "test") -> Dict[str, Any]:
    """
    Analyze GSM8K dataset statistics.

    Args:
        split: Dataset split to analyze

    Returns:
        Dict with statistics:
            - total_examples: Number of examples
            - answer_range: (min, max) of numerical answers
            - answer_median: Median answer value
            - answer_distribution: Histogram of answer ranges
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    # Extract all numerical answers
    answers = []
    for ex in ds:
        answer_text = ex["answer"]
        num = extract_numerical_answer(answer_text)
        if num is not None:
            try:
                # Convert to float for statistics
                answers.append(float(num))
            except ValueError:
                pass

    answers.sort()

    stats = {
        "total_examples": len(ds),
        "valid_numerical_answers": len(answers),
        "answer_range": (min(answers), max(answers)) if answers else (0, 0),
        "answer_median": answers[len(answers) // 2] if answers else 0,
        "answer_mean": sum(answers) / len(answers) if answers else 0,
    }

    # Create histogram bins
    if answers:
        bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        histogram = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins)-1)}

        for ans in answers:
            for i in range(len(bins)-1):
                if bins[i] <= ans < bins[i+1]:
                    bin_label = f"{bins[i]}-{bins[i+1]}" if bins[i+1] != float('inf') else f"{bins[i]}+"
                    histogram[bin_label] = histogram.get(bin_label, 0) + 1
                    break

        stats["answer_histogram"] = histogram

    return stats


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the GSM8K evaluation pipeline.
    """
    print("=" * 80)
    print("GSM8K Evaluation Module - Example Usage")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading GSM8K test set (200 samples)...")
    test_data = load_gsm8k_dataset(split="test", samples=200, seed=42)
    print(f"   Loaded {len(test_data)} examples")

    # Show example
    print("\n2. Example question:")
    ex = test_data[0]
    print(f"   Q: {ex['question']}")
    print(f"   A: {ex['numerical_answer']}")

    # Format prompt
    print("\n3. Formatted prompt (with 8-shot):")
    prompt = format_gsm8k_prompt(ex['question'], few_shot=True)
    print(f"   Length: {len(prompt)} chars")
    print(f"   Preview: {prompt[:200]}...")

    # Test answer extraction
    print("\n4. Testing answer extraction:")
    test_texts = [
        "The answer is 42",
        "So we get 1,234 total. #### 1234",
        "The final answer is -5.5",
        "No numbers here!",
    ]
    for text in test_texts:
        extracted = extract_numerical_answer(text)
        print(f"   '{text}' -> {extracted}")

    # Simulate evaluation
    print("\n5. Simulating evaluation (using gold as predictions):")
    mock_predictions = [ex["answer"] for ex in test_data[:10]]
    mock_gold = test_data[:10]
    results = evaluate_gsm8k_batch(mock_predictions, mock_gold, verbose=False)
    print(f"   Accuracy: {results['accuracy']:.1f}%")
    print(f"   Correct: {results['correct']}/{results['total']}")

    # Dataset statistics
    print("\n6. Dataset statistics:")
    stats = analyze_gsm8k_dataset(split="test")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Answer range: {stats['answer_range'][0]:.0f} to {stats['answer_range'][1]:.0f}")
    print(f"   Median answer: {stats['answer_median']:.0f}")

    print("\n" + "=" * 80)
    print("Module ready for use!")
    print("=" * 80)
