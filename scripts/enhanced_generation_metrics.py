#!/usr/bin/env python3
"""
Enhanced generation metrics for LatentWire experiments.

This module provides comprehensive generation quality metrics including:
- ROUGE scores (1, 2, L) with confidence intervals
- BLEU scores (1-4 gram)
- BERTScore for semantic similarity
- Perplexity and entropy metrics
- Length statistics and diversity measures
- Factuality/correctness for math problems
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import re
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationMetrics:
    """Container for comprehensive generation metrics."""

    # ROUGE metrics
    rouge1_f1: float
    rouge1_precision: float
    rouge1_recall: float
    rouge2_f1: float
    rouge2_precision: float
    rouge2_recall: float
    rougeL_f1: float
    rougeL_precision: float
    rougeL_recall: float

    # BLEU scores
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    bleu_avg: float

    # Length statistics
    avg_pred_length: float
    avg_ref_length: float
    length_ratio: float
    std_pred_length: float

    # Diversity metrics
    distinct_1: float  # Unique unigrams / total unigrams
    distinct_2: float  # Unique bigrams / total bigrams
    distinct_3: float  # Unique trigrams / total trigrams
    vocab_size: int
    repetition_rate: float

    # Task-specific metrics
    exact_match: Optional[float] = None
    correctness: Optional[float] = None  # For math/reasoning
    bertscore_f1: Optional[float] = None
    perplexity: Optional[float] = None

    # Statistical measures
    rouge1_f1_ci: Optional[Tuple[float, float]] = None
    rouge2_f1_ci: Optional[Tuple[float, float]] = None
    rougeL_f1_ci: Optional[Tuple[float, float]] = None

    # Per-sample scores for analysis
    per_sample_scores: Optional[List[Dict]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'rouge': {
                'rouge1': {
                    'f1': self.rouge1_f1,
                    'precision': self.rouge1_precision,
                    'recall': self.rouge1_recall
                },
                'rouge2': {
                    'f1': self.rouge2_f1,
                    'precision': self.rouge2_precision,
                    'recall': self.rouge2_recall
                },
                'rougeL': {
                    'f1': self.rougeL_f1,
                    'precision': self.rougeL_precision,
                    'recall': self.rougeL_recall
                }
            },
            'bleu': {
                'bleu1': self.bleu1,
                'bleu2': self.bleu2,
                'bleu3': self.bleu3,
                'bleu4': self.bleu4,
                'average': self.bleu_avg
            },
            'length': {
                'avg_prediction': self.avg_pred_length,
                'avg_reference': self.avg_ref_length,
                'ratio': self.length_ratio,
                'std_prediction': self.std_pred_length
            },
            'diversity': {
                'distinct_1': self.distinct_1,
                'distinct_2': self.distinct_2,
                'distinct_3': self.distinct_3,
                'vocab_size': self.vocab_size,
                'repetition_rate': self.repetition_rate
            }
        }

        # Add optional metrics
        if self.exact_match is not None:
            result['exact_match'] = self.exact_match
        if self.correctness is not None:
            result['correctness'] = self.correctness
        if self.bertscore_f1 is not None:
            result['bertscore_f1'] = self.bertscore_f1
        if self.perplexity is not None:
            result['perplexity'] = self.perplexity

        # Add confidence intervals
        if self.rouge1_f1_ci is not None:
            result['rouge']['rouge1']['f1_ci_95'] = list(self.rouge1_f1_ci)
        if self.rouge2_f1_ci is not None:
            result['rouge']['rouge2']['f1_ci_95'] = list(self.rouge2_f1_ci)
        if self.rougeL_f1_ci is not None:
            result['rouge']['rougeL']['f1_ci_95'] = list(self.rougeL_f1_ci)

        # Add per-sample scores if available
        if self.per_sample_scores is not None:
            result['per_sample'] = self.per_sample_scores

        return result

    def summary_string(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Generation Metrics Summary:",
            f"  ROUGE-1 F1: {self.rouge1_f1:.4f}",
            f"  ROUGE-2 F1: {self.rouge2_f1:.4f}",
            f"  ROUGE-L F1: {self.rougeL_f1:.4f}",
            f"  BLEU-4: {self.bleu4:.4f}",
            f"  Avg Length: {self.avg_pred_length:.1f} (ref: {self.avg_ref_length:.1f})",
            f"  Distinct-1: {self.distinct_1:.3f}",
            f"  Distinct-2: {self.distinct_2:.3f}"
        ]

        if self.exact_match is not None:
            lines.append(f"  Exact Match: {self.exact_match:.3f}")
        if self.correctness is not None:
            lines.append(f"  Correctness: {self.correctness:.3f}")
        if self.bertscore_f1 is not None:
            lines.append(f"  BERTScore F1: {self.bertscore_f1:.4f}")
        if self.perplexity is not None:
            lines.append(f"  Perplexity: {self.perplexity:.2f}")

        return "\n".join(lines)


def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
    compute_bertscore: bool = False,
    compute_perplexity: bool = False,
    perplexity_model_name: Optional[str] = None,
    task_type: str = "general",  # "general", "math", "code"
    compute_confidence_intervals: bool = True,
    n_bootstrap: int = 1000,
    save_path: Optional[Path] = None,
    return_per_sample: bool = False
) -> GenerationMetrics:
    """
    Compute comprehensive generation metrics.

    Args:
        predictions: List of generated texts
        references: List of reference texts
        compute_bertscore: Whether to compute BERTScore (slower)
        compute_perplexity: Whether to compute perplexity
        perplexity_model_name: Model to use for perplexity
        task_type: Type of generation task for specialized metrics
        compute_confidence_intervals: Whether to compute bootstrap CIs
        n_bootstrap: Number of bootstrap iterations
        save_path: Path to save detailed results
        return_per_sample: Whether to include per-sample scores

    Returns:
        GenerationMetrics object with all computed metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Compute per-sample scores
    rouge_scores = []
    bleu_scores = []
    exact_matches = []
    correctness_scores = []

    for pred, ref in zip(predictions, references):
        # ROUGE
        rouge = scorer.score(ref, pred)
        rouge_scores.append({
            'rouge1_f1': rouge['rouge1'].fmeasure,
            'rouge1_p': rouge['rouge1'].precision,
            'rouge1_r': rouge['rouge1'].recall,
            'rouge2_f1': rouge['rouge2'].fmeasure,
            'rouge2_p': rouge['rouge2'].precision,
            'rouge2_r': rouge['rouge2'].recall,
            'rougeL_f1': rouge['rougeL'].fmeasure,
            'rougeL_p': rouge['rougeL'].precision,
            'rougeL_r': rouge['rougeL'].recall,
        })

        # BLEU (with smoothing for short sequences)
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        smoothing = SmoothingFunction().method1

        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1,0,0,0), smoothing_function=smoothing)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5,0.5,0,0), smoothing_function=smoothing)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33,0.33,0.34,0), smoothing_function=smoothing)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothing)

        bleu_scores.append({
            'bleu1': bleu1, 'bleu2': bleu2,
            'bleu3': bleu3, 'bleu4': bleu4
        })

        # Exact match
        exact_matches.append(1.0 if pred.strip() == ref.strip() else 0.0)

        # Task-specific correctness
        if task_type == "math":
            correct = _check_math_correctness(pred, ref)
            correctness_scores.append(1.0 if correct else 0.0)

    # Aggregate ROUGE scores
    rouge1_f1 = np.mean([s['rouge1_f1'] for s in rouge_scores])
    rouge1_p = np.mean([s['rouge1_p'] for s in rouge_scores])
    rouge1_r = np.mean([s['rouge1_r'] for s in rouge_scores])
    rouge2_f1 = np.mean([s['rouge2_f1'] for s in rouge_scores])
    rouge2_p = np.mean([s['rouge2_p'] for s in rouge_scores])
    rouge2_r = np.mean([s['rouge2_r'] for s in rouge_scores])
    rougeL_f1 = np.mean([s['rougeL_f1'] for s in rouge_scores])
    rougeL_p = np.mean([s['rougeL_p'] for s in rouge_scores])
    rougeL_r = np.mean([s['rougeL_r'] for s in rouge_scores])

    # Aggregate BLEU scores
    bleu1 = np.mean([s['bleu1'] for s in bleu_scores])
    bleu2 = np.mean([s['bleu2'] for s in bleu_scores])
    bleu3 = np.mean([s['bleu3'] for s in bleu_scores])
    bleu4 = np.mean([s['bleu4'] for s in bleu_scores])
    bleu_avg = np.mean([bleu1, bleu2, bleu3, bleu4])

    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    avg_pred_length = np.mean(pred_lengths)
    avg_ref_length = np.mean(ref_lengths)
    length_ratio = avg_pred_length / max(avg_ref_length, 1.0)
    std_pred_length = np.std(pred_lengths)

    # Diversity metrics
    distinct_1, distinct_2, distinct_3 = _compute_distinct_ngrams(predictions)
    vocab_size = len(set(' '.join(predictions).split()))
    repetition_rate = _compute_repetition_rate(predictions)

    # Exact match and correctness
    exact_match = np.mean(exact_matches) if exact_matches else None
    correctness = np.mean(correctness_scores) if correctness_scores else None

    # BERTScore (optional, requires bert-score package)
    bertscore_f1 = None
    if compute_bertscore:
        try:
            from bert_score import score
            P, R, F1 = score(predictions, references, lang='en', verbose=False)
            bertscore_f1 = float(F1.mean())
        except ImportError:
            print("Warning: bert-score not installed, skipping BERTScore")

    # Perplexity (optional)
    perplexity = None
    if compute_perplexity:
        perplexity = _compute_perplexity(predictions, perplexity_model_name)

    # Bootstrap confidence intervals
    rouge1_ci = None
    rouge2_ci = None
    rougeL_ci = None
    if compute_confidence_intervals and len(predictions) >= 30:
        rouge1_ci = _bootstrap_metric_ci(
            [s['rouge1_f1'] for s in rouge_scores], n_bootstrap
        )
        rouge2_ci = _bootstrap_metric_ci(
            [s['rouge2_f1'] for s in rouge_scores], n_bootstrap
        )
        rougeL_ci = _bootstrap_metric_ci(
            [s['rougeL_f1'] for s in rouge_scores], n_bootstrap
        )

    # Per-sample scores
    per_sample = None
    if return_per_sample:
        per_sample = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            per_sample.append({
                'prediction': pred[:200],  # Truncate for storage
                'reference': ref[:200],
                'rouge1_f1': rouge_scores[i]['rouge1_f1'],
                'rouge2_f1': rouge_scores[i]['rouge2_f1'],
                'rougeL_f1': rouge_scores[i]['rougeL_f1'],
                'bleu4': bleu_scores[i]['bleu4'],
                'exact_match': exact_matches[i]
            })

    # Create metrics object
    metrics = GenerationMetrics(
        rouge1_f1=rouge1_f1,
        rouge1_precision=rouge1_p,
        rouge1_recall=rouge1_r,
        rouge2_f1=rouge2_f1,
        rouge2_precision=rouge2_p,
        rouge2_recall=rouge2_r,
        rougeL_f1=rougeL_f1,
        rougeL_precision=rougeL_p,
        rougeL_recall=rougeL_r,
        bleu1=bleu1,
        bleu2=bleu2,
        bleu3=bleu3,
        bleu4=bleu4,
        bleu_avg=bleu_avg,
        avg_pred_length=avg_pred_length,
        avg_ref_length=avg_ref_length,
        length_ratio=length_ratio,
        std_pred_length=std_pred_length,
        distinct_1=distinct_1,
        distinct_2=distinct_2,
        distinct_3=distinct_3,
        vocab_size=vocab_size,
        repetition_rate=repetition_rate,
        exact_match=exact_match,
        correctness=correctness,
        bertscore_f1=bertscore_f1,
        perplexity=perplexity,
        rouge1_f1_ci=rouge1_ci,
        rouge2_f1_ci=rouge2_ci,
        rougeL_f1_ci=rougeL_ci,
        per_sample_scores=per_sample
    )

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'generation_metrics.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save detailed per-sample analysis
        if return_per_sample:
            with open(save_path / 'per_sample_analysis.json', 'w') as f:
                json.dump(per_sample, f, indent=2)

    return metrics


def _compute_distinct_ngrams(texts: List[str]) -> Tuple[float, float, float]:
    """Compute distinct n-gram ratios for diversity measurement."""
    unigrams = []
    bigrams = []
    trigrams = []

    for text in texts:
        tokens = text.split()
        unigrams.extend(tokens)
        bigrams.extend(zip(tokens, tokens[1:]))
        trigrams.extend(zip(tokens, tokens[1:], tokens[2:]))

    distinct_1 = len(set(unigrams)) / max(len(unigrams), 1)
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)
    distinct_3 = len(set(trigrams)) / max(len(trigrams), 1)

    return distinct_1, distinct_2, distinct_3


def _compute_repetition_rate(texts: List[str]) -> float:
    """Compute rate of repeated phrases (3+ grams)."""
    all_phrases = []
    for text in texts:
        tokens = text.split()
        phrases = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        all_phrases.extend(phrases)

    if not all_phrases:
        return 0.0

    phrase_counts = Counter(all_phrases)
    repeated = sum(1 for count in phrase_counts.values() if count > 1)
    return repeated / len(phrase_counts)


def _check_math_correctness(prediction: str, reference: str) -> bool:
    """Check if math answer is correct (extracts final number)."""
    # Extract numbers from both strings
    pred_nums = re.findall(r'-?\d+\.?\d*', prediction)
    ref_nums = re.findall(r'-?\d+\.?\d*', reference)

    if not pred_nums or not ref_nums:
        return False

    # Check if final number matches
    try:
        pred_final = float(pred_nums[-1])
        ref_final = float(ref_nums[-1])
        return abs(pred_final - ref_final) < 1e-5
    except:
        return False


def _compute_perplexity(
    texts: List[str],
    model_name: Optional[str] = None
) -> float:
    """Compute average perplexity using a language model."""
    if model_name is None:
        model_name = "gpt2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        perplexities = []
        with torch.no_grad():
            for text in texts[:100]:  # Limit to 100 for speed
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs, labels=inputs.input_ids)
                perplexity = torch.exp(outputs.loss).item()
                perplexities.append(perplexity)

        return np.mean(perplexities)
    except Exception as e:
        print(f"Warning: Could not compute perplexity: {e}")
        return None


def _bootstrap_metric_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric."""
    scores = np.array(scores)
    bootstrapped = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(len(scores), len(scores), replace=True)
        bootstrapped.append(np.mean(scores[indices]))

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))

    return (lower, upper)


def main():
    """Example usage."""
    # Sample data
    predictions = [
        "The movie was great and I enjoyed it a lot.",
        "This film is terrible and boring.",
        "An average movie with some good moments."
    ]
    references = [
        "The film was excellent and very enjoyable.",
        "This movie is awful and not interesting.",
        "A mediocre film with occasional highlights."
    ]

    print("Computing generation metrics...")
    metrics = compute_generation_metrics(
        predictions, references,
        compute_bertscore=False,
        task_type="general",
        return_per_sample=True,
        save_path=Path("test_generation_metrics")
    )

    print(metrics.summary_string())


if __name__ == "__main__":
    main()