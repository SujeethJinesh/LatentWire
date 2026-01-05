"""
Linear Probe Baselines on LLM Hidden States

Standard methodology for extracting hidden states from specific layers and training
linear probes for classification tasks. Follows best practices from recent research
(2025-2026) including proper normalization, multi-seed evaluation, and statistical
significance testing.

References:
- Calibrating LLM Judges (Dec 2025): https://arxiv.org/abs/2512.22245
- PING Framework (Sep 2025): https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v2.full
- HuggingFace Transformers documentation: output_hidden_states
- scikit-learn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# Dataset Loading
# ============================================================================

def load_sst2(split: str = "train", max_samples: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load SST-2 sentiment classification dataset."""
    dataset = load_dataset("glue", "sst2", split=split)

    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        examples.append({
            "text": item["sentence"],
            "label": item["label"],  # 0 = negative, 1 = positive
        })

    return examples


def load_agnews(split: str = "train", max_samples: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load AG News topic classification dataset."""
    dataset = load_dataset("fancyzhx/ag_news", split=split)

    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        examples.append({
            "text": item["text"],
            "label": item["label"],  # 0-3: World, Sports, Business, Sci/Tech
        })

    return examples


def load_trec(split: str = "train", max_samples: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
    """Load TREC question classification dataset."""
    dataset = load_dataset("trec", split=split)

    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        examples.append({
            "text": item["text"],
            "label": item["coarse_label"],  # 0-5: question types
        })

    return examples


def load_dataset_by_name(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load dataset by name."""
    loaders = {
        "sst2": load_sst2,
        "agnews": load_agnews,
        "trec": load_trec,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")

    return loaders[dataset_name](split=split, max_samples=max_samples, seed=seed)


# ============================================================================
# Hidden State Extraction
# ============================================================================

def extract_hidden_states(
    texts: List[str],
    model,
    tokenizer,
    layer_idx: int = 16,
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
    pooling: str = "last_token",
) -> np.ndarray:
    """
    Extract hidden states from a specific layer of an LLM.

    Args:
        texts: List of input texts
        model: HuggingFace model (AutoModelForCausalLM)
        tokenizer: HuggingFace tokenizer
        layer_idx: Which layer to extract from (0-indexed, 0=embeddings, N=last layer)
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        device: Device to run on
        pooling: How to pool hidden states ("last_token", "mean", "first_token", "cls")

    Returns:
        Array of shape [num_examples, hidden_dim]
    """
    model.eval()
    model.to(device)

    all_hidden_states = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting layer {layer_idx} hidden states"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract hidden states from specified layer
        # outputs.hidden_states is a tuple: (embeddings, layer1, layer2, ..., layerN)
        # So layer_idx=0 is embeddings, layer_idx=1 is first transformer layer, etc.
        hidden = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]

        # Pool across sequence dimension
        if pooling == "last_token":
            # Use the last non-padding token
            attention_mask = inputs["attention_mask"]  # [batch, seq_len]
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(hidden.size(0), device=device)
            pooled = hidden[batch_indices, seq_lengths]  # [batch, hidden_dim]
        elif pooling == "first_token" or pooling == "cls":
            pooled = hidden[:, 0, :]  # [batch, hidden_dim]
        elif pooling == "mean":
            # Mean over non-padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch, seq_len, 1]
            masked_hidden = hidden * attention_mask
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        all_hidden_states.append(pooled.cpu().numpy())

    return np.vstack(all_hidden_states)


# ============================================================================
# Linear Probe Training
# ============================================================================

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalize: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a linear probe (logistic regression) on hidden states.

    Args:
        X_train: Training features [n_train, hidden_dim]
        y_train: Training labels [n_train]
        X_test: Test features [n_test, hidden_dim]
        y_test: Test labels [n_test]
        normalize: Normalization method ("l2", "standardize", "none")
        C: Inverse regularization strength (higher = less regularization)
        max_iter: Maximum iterations for solver
        random_state: Random seed

    Returns:
        Dictionary with results and trained models
    """
    # Normalize features
    if normalize == "l2":
        # L2 normalization: ||x|| = 1 for each example
        from sklearn.preprocessing import normalize as sk_normalize
        X_train_norm = sk_normalize(X_train, norm='l2', axis=1)
        X_test_norm = sk_normalize(X_test, norm='l2', axis=1)
        scaler = None
    elif normalize == "standardize":
        # Standardization: zero mean, unit variance per feature
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
    elif normalize == "none":
        X_train_norm = X_train
        X_test_norm = X_test
        scaler = None
    else:
        raise ValueError(f"Unknown normalization: {normalize}")

    # Train logistic regression
    # Use lbfgs solver (default, robust for most cases)
    # L2 regularization by default (controlled by C)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        multi_class='auto',
        n_jobs=-1,  # Use all CPU cores
    )

    start_time = time.time()
    clf.fit(X_train_norm, y_train)
    train_time = time.time() - start_time

    # Evaluate
    y_train_pred = clf.predict(X_train_norm)
    y_test_pred = clf.predict(X_test_norm)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # F1 scores (macro average for multi-class)
    num_classes = len(np.unique(y_train))
    average = 'binary' if num_classes == 2 else 'macro'
    train_f1 = f1_score(y_train, y_train_pred, average=average)
    test_f1 = f1_score(y_test, y_test_pred, average=average)

    # Cross-validation on training set (for robustness check)
    cv_scores = cross_val_score(clf, X_train_norm, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "train_time": float(train_time),
        "num_train": len(y_train),
        "num_test": len(y_test),
        "num_classes": int(num_classes),
        "model": clf,
        "scaler": scaler,
    }


# ============================================================================
# Multi-Seed Evaluation Protocol
# ============================================================================

def multi_seed_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    layer_idx: int,
    num_seeds: int = 5,
    test_size: float = 0.2,
    normalize: str = "l2",
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Run linear probe evaluation with multiple random seeds for statistical reliability.

    Following best practices from:
    - Paired Seed Evaluation (2025): https://www.arxiv.org/pdf/2512.24145
    - Cross-validation literature for model evaluation

    Args:
        X: Features [n_examples, hidden_dim]
        y: Labels [n_examples]
        layer_idx: Layer index for reporting
        num_seeds: Number of random seeds to evaluate
        test_size: Fraction of data to use for test set
        normalize: Normalization method
        C: Regularization parameter

    Returns:
        Dictionary with aggregated results across seeds
    """
    results_per_seed = []

    for seed in range(num_seeds):
        # Split data with different random seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=y,  # Ensure balanced splits
        )

        # Train and evaluate
        result = train_linear_probe(
            X_train, y_train,
            X_test, y_test,
            normalize=normalize,
            C=C,
            random_state=seed,
        )

        results_per_seed.append({
            "seed": seed,
            "test_accuracy": result["test_accuracy"],
            "test_f1": result["test_f1"],
            "train_accuracy": result["train_accuracy"],
            "train_f1": result["train_f1"],
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
        })

    # Aggregate statistics
    test_accs = [r["test_accuracy"] for r in results_per_seed]
    test_f1s = [r["test_f1"] for r in results_per_seed]

    return {
        "layer_idx": layer_idx,
        "num_seeds": num_seeds,
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "test_accuracy_min": float(np.min(test_accs)),
        "test_accuracy_max": float(np.max(test_accs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "test_f1_min": float(np.min(test_f1s)),
        "test_f1_max": float(np.max(test_f1s)),
        "per_seed_results": results_per_seed,
    }


# ============================================================================
# Layer Sweep Experiments
# ============================================================================

def layer_sweep_experiment(
    texts: List[str],
    labels: List[int],
    model,
    tokenizer,
    layer_indices: List[int],
    device: str = "cuda",
    num_seeds: int = 5,
    test_size: float = 0.2,
    normalize: str = "l2",
    pooling: str = "last_token",
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate linear probes across multiple layers to find optimal layer.

    Args:
        texts: Input texts
        labels: Classification labels
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        layer_indices: List of layer indices to evaluate
        device: Device to run on
        num_seeds: Number of random seeds per layer
        test_size: Test set fraction
        normalize: Normalization method
        pooling: Pooling method for hidden states
        batch_size: Batch size for extraction

    Returns:
        Dictionary with results for each layer
    """
    y = np.array(labels)
    results = {}

    for layer_idx in layer_indices:
        print(f"\n{'='*80}")
        print(f"Evaluating Layer {layer_idx}")
        print(f"{'='*80}")

        # Extract hidden states
        X = extract_hidden_states(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            batch_size=batch_size,
            device=device,
            pooling=pooling,
        )

        print(f"Extracted hidden states: {X.shape}")

        # Multi-seed evaluation
        layer_results = multi_seed_evaluation(
            X=X,
            y=y,
            layer_idx=layer_idx,
            num_seeds=num_seeds,
            test_size=test_size,
            normalize=normalize,
        )

        results[f"layer_{layer_idx}"] = layer_results

        print(f"Test Accuracy: {layer_results['test_accuracy_mean']:.4f} ± {layer_results['test_accuracy_std']:.4f}")
        print(f"Test F1: {layer_results['test_f1_mean']:.4f} ± {layer_results['test_f1_std']:.4f}")

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Linear Probe Hidden States Baseline")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "agnews", "trec"],
                        help="Dataset to use")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum training samples (None = use all)")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum test samples (None = use all)")

    # Model args
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--layer_idx", type=int, default=16,
                        help="Layer index to extract (0=embeddings, N=last layer)")
    parser.add_argument("--layer_sweep", action="store_true",
                        help="Evaluate across multiple layers")
    parser.add_argument("--layer_start", type=int, default=0,
                        help="Start layer for sweep")
    parser.add_argument("--layer_end", type=int, default=32,
                        help="End layer for sweep (exclusive)")
    parser.add_argument("--layer_step", type=int, default=4,
                        help="Layer step for sweep")

    # Extraction args
    parser.add_argument("--pooling", type=str, default="last_token",
                        choices=["last_token", "mean", "first_token", "cls"],
                        help="How to pool hidden states across sequence")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for extraction")

    # Probe args
    parser.add_argument("--normalize", type=str, default="l2",
                        choices=["l2", "standardize", "none"],
                        help="Feature normalization method")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of random seeds for evaluation")

    # Infrastructure
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="runs/linear_probe",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("="*80)
    print("Linear Probe Hidden States Baseline")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    train_data = load_dataset_by_name(
        args.dataset,
        split="train",
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    test_data = load_dataset_by_name(
        args.dataset,
        split="test" if args.dataset != "trec" else "test",
        max_samples=args.max_test_samples,
        seed=args.seed,
    )

    # Combine for train/test split
    all_texts = [ex["text"] for ex in train_data] + [ex["text"] for ex in test_data]
    all_labels = [ex["label"] for ex in train_data] + [ex["label"] for ex in test_data]

    print(f"Total examples: {len(all_texts)}")
    print(f"Num classes: {len(set(all_labels))}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    # Run experiment
    if args.layer_sweep:
        print("\nRunning layer sweep experiment...")
        layer_indices = list(range(args.layer_start, args.layer_end, args.layer_step))
        results = layer_sweep_experiment(
            texts=all_texts,
            labels=all_labels,
            model=model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            device=args.device,
            num_seeds=args.num_seeds,
            test_size=args.test_size,
            normalize=args.normalize,
            pooling=args.pooling,
            batch_size=args.batch_size,
        )
    else:
        print(f"\nExtracting layer {args.layer_idx} hidden states...")
        X = extract_hidden_states(
            texts=all_texts,
            model=model,
            tokenizer=tokenizer,
            layer_idx=args.layer_idx,
            batch_size=args.batch_size,
            device=args.device,
            pooling=args.pooling,
        )

        print(f"Extracted hidden states: {X.shape}")

        print(f"\nRunning multi-seed evaluation ({args.num_seeds} seeds)...")
        y = np.array(all_labels)
        results = multi_seed_evaluation(
            X=X,
            y=y,
            layer_idx=args.layer_idx,
            num_seeds=args.num_seeds,
            test_size=args.test_size,
            normalize=args.normalize,
            C=args.C,
        )

        results = {f"layer_{args.layer_idx}": results}

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        # Remove non-serializable objects
        serializable_results = {}
        for layer_key, layer_data in results.items():
            serializable_results[layer_key] = {
                k: v for k, v in layer_data.items()
                if k not in ["model", "scaler"]
            }
        json.dump(serializable_results, f, indent=2)

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)

    for layer_key, layer_data in results.items():
        print(f"\n{layer_key}:")
        print(f"  Test Accuracy: {layer_data['test_accuracy_mean']:.4f} ± {layer_data['test_accuracy_std']:.4f}")
        print(f"  Test F1:       {layer_data['test_f1_mean']:.4f} ± {layer_data['test_f1_std']:.4f}")

    print(f"\nResults saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
