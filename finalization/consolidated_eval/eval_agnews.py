#!/usr/bin/env python
# latentwire/eval_agnews.py
"""
AG News Classification Evaluation Module for LatentWire

This module provides evaluation capabilities for AG News topic classification:
- Dataset: AG News test set (7,600 samples, 1,900 per class)
- Task: 4-way classification (World, Sports, Business, Science/Tech)
- Metric: Accuracy (overall and per-class)

The module supports:
1. Loading full 7600 test samples
2. Memory-efficient batch evaluation
3. Permissive matching for science/tech variants
4. Comprehensive accuracy metrics (overall + per-class)
5. Progress tracking with tqdm

Usage:
    from latentwire.eval_agnews import AGNewsEvaluator

    evaluator = AGNewsEvaluator()
    results = evaluator.evaluate_model(
        model=model,
        tokenizer=tokenizer,
        num_samples=7600,  # Full test set
        batch_size=32,
        device=device
    )
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json


# AG News class labels (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
AGNEWS_LABELS = ["world", "sports", "business", "science"]

# Permissive matching for science/tech (AG News uses "Sci/Tech")
SCIENCE_SYNONYMS = ["science", "technology", "tech", "sci/tech", "scitech", "sci-tech"]


def check_label_match(label: str, output: str) -> bool:
    """
    Check if label matches output, with permissive matching for science.

    Args:
        label: Ground truth label (one of AGNEWS_LABELS)
        output: Model's generated output (lowercased)

    Returns:
        True if label is found in output (with special handling for science)
    """
    output_lower = output.lower()

    if label == "science":
        return any(syn in output_lower for syn in SCIENCE_SYNONYMS)

    return label in output_lower


def format_agnews_prompt(text: str, max_chars: int = 256) -> str:
    """
    Format AG News article into evaluation prompt.

    Args:
        text: Article text
        max_chars: Maximum characters to include from article

    Returns:
        Formatted prompt string
    """
    truncated_text = text[:max_chars]
    return f"Article: {truncated_text}\nTopic (world, sports, business, or science):"


class AGNewsEvaluator:
    """
    Evaluator for AG News topic classification.

    Handles loading the dataset, formatting prompts, running inference,
    and computing accuracy metrics.
    """

    def __init__(self, max_article_chars: int = 256):
        """
        Initialize AG News evaluator.

        Args:
            max_article_chars: Maximum characters from article to include in prompt
        """
        self.max_article_chars = max_article_chars
        self.labels = AGNEWS_LABELS
        self.dataset_cache = None

    def load_dataset(self, split: str = "test") -> Any:
        """
        Load AG News dataset.

        Args:
            split: Dataset split to load (default: "test")

        Returns:
            Loaded dataset
        """
        if self.dataset_cache is None:
            self.dataset_cache = load_dataset("ag_news", split=split)
        return self.dataset_cache

    def evaluate_text_baseline(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        num_samples: int = 7600,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        max_new_tokens: int = 10,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate model with full text prompts (baseline).

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            num_samples: Number of samples to evaluate (default: 7600 = full test set)
            batch_size: Batch size for evaluation
            device: Device to run on (if None, uses model's device)
            max_new_tokens: Maximum tokens to generate
            model_name: Name for progress bar

        Returns:
            Dictionary with accuracy metrics and per-class results
        """
        if device is None:
            device = next(model.parameters()).device

        ds = self.load_dataset()
        num_samples = min(num_samples, len(ds))

        # Track overall and per-class accuracy
        correct = 0
        total = 0
        class_correct = {label: 0 for label in self.labels}
        class_total = {label: 0 for label in self.labels}

        # Process in batches for memory efficiency
        for start_idx in tqdm(
            range(0, num_samples, batch_size),
            desc=f"{model_name} text baseline"
        ):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_items = [ds[i] for i in range(start_idx, end_idx)]

            for item in batch_items:
                text = item['text']
                label = self.labels[item['label']]

                # Format prompt
                prompt = format_agnews_prompt(text, self.max_article_chars)

                # Generate
                with torch.no_grad():
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=300
                    ).to(device)

                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    # Extract only generated portion
                    gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
                    output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

                # Check accuracy
                is_correct = check_label_match(label, output)
                if is_correct:
                    correct += 1
                    class_correct[label] += 1

                total += 1
                class_total[label] += 1

        # Compute metrics
        overall_acc = 100 * correct / total if total > 0 else 0.0

        per_class_metrics = {}
        for label in self.labels:
            if class_total[label] > 0:
                acc = 100 * class_correct[label] / class_total[label]
                per_class_metrics[label] = {
                    "correct": class_correct[label],
                    "total": class_total[label],
                    "accuracy": acc
                }
            else:
                per_class_metrics[label] = {
                    "correct": 0,
                    "total": 0,
                    "accuracy": 0.0
                }

        return {
            "overall_accuracy": overall_acc,
            "correct": correct,
            "total": total,
            "per_class": per_class_metrics,
            "num_samples": num_samples,
            "model_name": model_name
        }

    def evaluate_with_embeddings(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        get_embeddings_fn,
        num_samples: int = 7600,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        max_new_tokens: int = 10,
        mode_name: str = "Latent"
    ) -> Dict[str, Any]:
        """
        Evaluate model with soft token embeddings (e.g., from LatentWire encoder).

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            get_embeddings_fn: Function that takes (text, device) and returns soft embeddings
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
            mode_name: Name for this evaluation mode (e.g., "Latent", "Compressed")

        Returns:
            Dictionary with accuracy metrics and per-class results
        """
        if device is None:
            device = next(model.parameters()).device

        ds = self.load_dataset()
        num_samples = min(num_samples, len(ds))

        correct = 0
        total = 0
        class_correct = {label: 0 for label in self.labels}
        class_total = {label: 0 for label in self.labels}

        for start_idx in tqdm(
            range(0, num_samples, batch_size),
            desc=f"{mode_name} evaluation"
        ):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_items = [ds[i] for i in range(start_idx, end_idx)]

            for item in batch_items:
                text = item['text']
                label = self.labels[item['label']]

                # Get soft embeddings from encoder
                prompt = format_agnews_prompt(text, self.max_article_chars)
                inputs_embeds = get_embeddings_fn(prompt, device)

                # Create attention mask
                attn_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    device=device,
                    dtype=torch.long
                )

                # Generate
                with torch.no_grad():
                    out_ids = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()

                # Check accuracy
                is_correct = check_label_match(label, output)
                if is_correct:
                    correct += 1
                    class_correct[label] += 1

                total += 1
                class_total[label] += 1

        # Compute metrics
        overall_acc = 100 * correct / total if total > 0 else 0.0

        per_class_metrics = {}
        for label in self.labels:
            if class_total[label] > 0:
                acc = 100 * class_correct[label] / class_total[label]
                per_class_metrics[label] = {
                    "correct": class_correct[label],
                    "total": class_total[label],
                    "accuracy": acc
                }
            else:
                per_class_metrics[label] = {
                    "correct": 0,
                    "total": 0,
                    "accuracy": 0.0
                }

        return {
            "overall_accuracy": overall_acc,
            "correct": correct,
            "total": total,
            "per_class": per_class_metrics,
            "num_samples": num_samples,
            "mode_name": mode_name
        }

    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted way.

        Args:
            results: Results dictionary from evaluate_text_baseline or evaluate_with_embeddings
        """
        print("\n" + "=" * 60)
        print(f"AG NEWS EVALUATION RESULTS - {results.get('model_name', results.get('mode_name', 'Model'))}")
        print("=" * 60)
        print(f"Overall Accuracy: {results['correct']}/{results['total']} ({results['overall_accuracy']:.2f}%)")
        print("\nPer-Class Accuracy:")

        for label in self.labels:
            metrics = results['per_class'][label]
            print(f"  {label.capitalize():10s}: {metrics['correct']:4d}/{metrics['total']:4d} ({metrics['accuracy']:5.2f}%)")

        print("=" * 60)

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Results dictionary to save
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def get_class_distribution(split: str = "test") -> Dict[str, Any]:
    """
    Get class distribution statistics for AG News dataset.

    Args:
        split: Dataset split to analyze

    Returns:
        Dictionary with class counts and percentages
    """
    ds = load_dataset("ag_news", split=split)
    labels = [item['label'] for item in ds]

    class_counts = {AGNEWS_LABELS[i]: labels.count(i) for i in range(4)}
    total = len(labels)

    class_percentages = {
        label: 100 * count / total
        for label, count in class_counts.items()
    }

    majority_class = max(class_counts, key=class_counts.get)

    return {
        "total_samples": total,
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "majority_class": majority_class,
        "majority_percentage": class_percentages[majority_class]
    }


# Convenience function for quick evaluation
def evaluate_model_on_agnews(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples: int = 7600,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model on AG News.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        num_samples: Number of samples to evaluate (default: 7600 = full test set)
        batch_size: Batch size for evaluation
        device: Device to run on
        model_name: Name for the model
        save_path: Optional path to save results JSON

    Returns:
        Dictionary with evaluation results
    """
    evaluator = AGNewsEvaluator()
    results = evaluator.evaluate_text_baseline(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        model_name=model_name
    )

    evaluator.print_results(results)

    if save_path:
        evaluator.save_results(results, save_path)

    return results


if __name__ == "__main__":
    """
    Example usage and dataset statistics.
    """
    import argparse

    parser = argparse.ArgumentParser(description="AG News Evaluation Module")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    if args.stats:
        print("\n" + "=" * 60)
        print("AG NEWS DATASET STATISTICS")
        print("=" * 60)

        dist = get_class_distribution("test")
        print(f"Total samples: {dist['total_samples']}")
        print("\nClass distribution:")
        for label in AGNEWS_LABELS:
            count = dist['class_counts'][label]
            pct = dist['class_percentages'][label]
            print(f"  {label.capitalize():10s}: {count:4d} ({pct:5.2f}%)")

        print(f"\nMajority class: {dist['majority_class'].capitalize()} ({dist['majority_percentage']:.2f}%)")
        print(f"Random baseline: 25.0%")
        print(f"Majority baseline: {dist['majority_percentage']:.2f}%")
        print("=" * 60)
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python latentwire/eval_agnews.py --stats    # Show dataset statistics")
        print("\nFor evaluation, import this module in your script:")
        print("  from latentwire.eval_agnews import AGNewsEvaluator, evaluate_model_on_agnews")
