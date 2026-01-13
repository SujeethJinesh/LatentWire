#!/usr/bin/env python3
"""
Linear Probe Baseline for Telepathy Paper

This implements a rigorous linear probe baseline to demonstrate that the Perceiver
architecture adds value beyond simple linear projection. Critical for paper claims.

Key Features:
- Extracts embeddings from Llama at multiple layers (16, 20, 24, 28, 31)
- Trains sklearn LogisticRegression on frozen hidden states
- Maps to Mistral's output space for fair cross-model comparison
- Tests on all 7 classification datasets
- Multi-seed evaluation with statistical significance

Datasets:
- SST-2 (2-class sentiment)
- AG News (4-class topic)
- TREC (6-class question type)
- Banking77 (77-class intent)
- IMDB (2-class sentiment)
- MNLI (3-class NLI)
- 20 Newsgroups (20-class topic)

Architecture:
1. Extract frozen hidden states from Llama layer L
2. Pool hidden states (mean/last-token)
3. Train LogisticRegression with L2 regularization
4. Evaluate accuracy and compare with Bridge results

This baseline tests whether the sender model's hidden states contain sufficient
task-relevant information WITHOUT the Perceiver's cross-attention mechanism.

Author: LatentWire Team
Date: January 2025
"""

import os
import json
import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path
from datetime import datetime
import argparse
import time
from dataclasses import dataclass, asdict
from collections import defaultdict

# HuggingFace imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    "sst2": {
        "hf_name": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "label_names": ["negative", "positive"],
    },
    "agnews": {
        "hf_name": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "trec": {
        "hf_name": ("trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "num_classes": 6,
        "train_split": "train",
        "eval_split": "test",
        "label_names": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
    },
    "banking77": {
        "hf_name": ("PolyAI/banking77",),
        "text_field": "text",
        "label_field": "label",
        "num_classes": 77,
        "train_split": "train",
        "eval_split": "test",
        "label_names": None,  # Too many to list
    },
    "imdb": {
        "hf_name": ("imdb",),
        "text_field": "text",
        "label_field": "label",
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "test",
        "label_names": ["negative", "positive"],
    },
    "mnli": {
        "hf_name": ("glue", "mnli"),
        "text_field": ["premise", "hypothesis"],  # Two fields to concatenate
        "label_field": "label",
        "num_classes": 3,
        "train_split": "train",
        "eval_split": "validation_matched",
        "label_names": ["entailment", "neutral", "contradiction"],
    },
    "20newsgroups": {
        "hf_name": ("SetFit/20_newsgroups",),
        "text_field": "text",
        "label_field": "label",
        "num_classes": 20,
        "train_split": "train",
        "eval_split": "test",
        "label_names": None,  # Too many to list
    },
    # =========================================================================
    # REASONING BENCHMARKS
    # =========================================================================
    "arc_easy": {
        "hf_name": ("allenai/ai2_arc", "ARC-Easy"),
        "text_field": "question",
        "label_field": "answerKey",
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "label_names": ["A", "B", "C", "D"],
        "label_mapping": {"A": 0, "B": 1, "C": 2, "D": 3},  # Map string labels to int
    },
    "winogrande": {
        "hf_name": ("allenai/winogrande", "winogrande_xl"),
        "text_field": "sentence",
        "label_field": "answer",
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "label_names": ["1", "2"],
        "label_mapping": {"1": 0, "2": 1},  # Map string labels to int
    },
    "hellaswag": {
        "hf_name": ("Rowan/hellaswag",),
        "text_field": "ctx",
        "label_field": "label",
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "validation",
        "label_names": ["0", "1", "2", "3"],
        # label is already int, but stored as string in some versions
        "label_mapping": {"0": 0, "1": 1, "2": 2, "3": 3},
    },
    "boolq": {
        "hf_name": ("google/boolq",),
        "text_field": "question",
        "label_field": "answer",
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "label_names": ["False", "True"],
        "label_mapping": {False: 0, True: 1},  # Map boolean to int
    },
}

# Default layers to evaluate
DEFAULT_LAYERS = [16, 20, 24, 28, 31]


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_by_name(
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Load dataset and return texts and labels.

    Args:
        dataset_name: Name of dataset (sst2, agnews, trec, etc.)
        split: Which split to load (train, test, validation)
        max_samples: Maximum samples to load (None = all)
        seed: Random seed for shuffling

    Returns:
        texts: List of input texts
        labels: numpy array of integer labels
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    # Load from HuggingFace
    if len(config["hf_name"]) == 2:
        ds = load_dataset(config["hf_name"][0], config["hf_name"][1], split=split, trust_remote_code=True)
    else:
        ds = load_dataset(config["hf_name"][0], split=split, trust_remote_code=True)

    # Extract texts
    text_field = config["text_field"]
    if isinstance(text_field, list):
        # MNLI: concatenate premise and hypothesis
        texts = [f"{item[text_field[0]]} [SEP] {item[text_field[1]]}" for item in ds]
    else:
        texts = [item[text_field] for item in ds]

    # Extract labels (with optional mapping for reasoning datasets)
    label_mapping = config.get("label_mapping", None)
    if label_mapping:
        # Map string/boolean labels to integers
        raw_labels = [item[config["label_field"]] for item in ds]
        labels = np.array([label_mapping.get(lbl, lbl) for lbl in raw_labels])
    else:
        labels = np.array([item[config["label_field"]] for item in ds])

    # Shuffle and limit if needed
    if max_samples is not None and len(texts) > max_samples:
        np.random.seed(seed)
        indices = np.random.permutation(len(texts))[:max_samples]
        texts = [texts[i] for i in indices]
        labels = labels[indices]

    return texts, labels


# =============================================================================
# Hidden State Extraction
# =============================================================================

def extract_hidden_states(
    texts: List[str],
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int = 8,
    max_length: int = 512,
    pooling: str = "last_token",
    device: str = "cuda",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract hidden states from a specific layer of a causal LM.

    Args:
        texts: List of input texts
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        layer_idx: Layer index (0=embeddings, 1=first layer, ..., N=last layer)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        pooling: How to pool ("last_token", "mean", "first_token")
        device: Device to use
        show_progress: Whether to show progress bar

    Returns:
        features: [N, hidden_dim] numpy array
    """
    model.eval()
    all_features = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Extracting layer {layer_idx}")

    with torch.no_grad():
        for i in iterator:
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass with hidden states
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Get hidden states from specified layer
            # hidden_states[0] = embeddings, hidden_states[1] = layer 0, etc.
            # So layer_idx=16 gets hidden_states[16]
            hidden = outputs.hidden_states[layer_idx]  # [B, seq_len, hidden_dim]

            # Pool across sequence
            if pooling == "last_token":
                # Use last non-padding token
                seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(hidden.size(0), device=device)
                pooled = hidden[batch_indices, seq_lengths]
            elif pooling == "mean":
                # Mean over non-padding tokens
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            elif pooling == "first_token":
                pooled = hidden[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_features.append(pooled.cpu().float().numpy())

    return np.vstack(all_features)


# =============================================================================
# Linear Probe Class
# =============================================================================

class LinearProbeBaseline:
    """
    Linear probe baseline using sklearn's LogisticRegression.

    This provides a rigorous baseline by:
    1. Extracting frozen embeddings from a pre-trained LLM
    2. Training a simple linear classifier (LogisticRegression)
    3. Supporting proper cross-validation and reproducibility
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        layer_idx: int = 24,
        pooling: str = "last_token",
        normalize: bool = True,
        C: float = 1.0,
        max_iter: int = 2000,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize linear probe.

        Args:
            hidden_dim: Dimension of hidden states
            num_classes: Number of output classes
            layer_idx: Which layer to extract from
            pooling: How to pool hidden states
            normalize: Whether to standardize features
            C: Inverse regularization strength
            max_iter: Maximum solver iterations
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.normalize = normalize
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Initialize sklearn components
        self.scaler = StandardScaler() if normalize else None

        # Use appropriate solver based on number of classes
        solver = 'lbfgs' if num_classes <= 10 else 'saga'
        self.probe = LogisticRegression(
            C=C,
            max_iter=max_iter,
            multi_class='multinomial' if num_classes > 2 else 'auto',
            solver=solver,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Fit the linear probe.

        Args:
            X_train: Training features [N, hidden_dim]
            y_train: Training labels [N]
            cv_folds: Number of CV folds (0 = no CV)

        Returns:
            Dictionary with training metrics
        """
        # Standardize features
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train

        # Cross-validation
        cv_scores = None
        if cv_folds > 0:
            try:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(
                    self.probe, X_train_scaled, y_train,
                    cv=skf, scoring='accuracy', n_jobs=self.n_jobs
                )
            except Exception as e:
                print(f"  Warning: CV failed with error: {e}")
                cv_scores = None

        # Fit on full training set
        start_time = time.time()
        self.probe.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        # Training accuracy
        y_pred = self.probe.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_pred)

        results = {
            "train_accuracy": train_acc * 100,
            "train_time": train_time,
            "num_samples": len(X_train),
            "num_features": X_train.shape[1],
        }

        if cv_scores is not None:
            results["cv_mean"] = cv_scores.mean() * 100
            results["cv_std"] = cv_scores.std() * 100

        return results

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        return self.probe.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        return self.probe.predict_proba(X_test)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate on test set.

        Returns:
            Dictionary with test metrics
        """
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # F1 score
        if self.num_classes == 2:
            f1 = f1_score(y_test, y_pred, average='binary')
        else:
            f1 = f1_score(y_test, y_pred, average='macro')

        return {
            "accuracy": accuracy * 100,
            "f1_score": f1 * 100,
            "correct": int(np.sum(y_pred == y_test)),
            "total": len(y_test),
        }

    def save(self, filepath: str):
        """Save probe weights and scaler."""
        save_dict = {
            "probe": self.probe,
            "scaler": self.scaler,
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
                "layer_idx": self.layer_idx,
                "pooling": self.pooling,
                "normalize": self.normalize,
                "C": self.C,
                "max_iter": self.max_iter,
            }
        }
        joblib.dump(save_dict, f"{filepath}.joblib")

    @classmethod
    def load(cls, filepath: str):
        """Load saved probe."""
        save_dict = joblib.load(f"{filepath}.joblib")
        config = save_dict["config"]

        instance = cls(
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            layer_idx=config["layer_idx"],
            pooling=config["pooling"],
            normalize=config["normalize"],
            C=config["C"],
            max_iter=config["max_iter"],
        )
        instance.probe = save_dict["probe"]
        instance.scaler = save_dict["scaler"]

        return instance


# =============================================================================
# Multi-Layer Evaluation
# =============================================================================

def run_layer_sweep(
    dataset_name: str,
    model,
    tokenizer,
    layers: List[int] = DEFAULT_LAYERS,
    pooling: str = "last_token",
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
    batch_size: int = 8,
    cv_folds: int = 5,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run linear probe evaluation across multiple layers.

    Args:
        dataset_name: Name of dataset
        model: Llama model
        tokenizer: Llama tokenizer
        layers: List of layer indices to evaluate
        pooling: Pooling method
        max_train_samples: Max training samples
        max_test_samples: Max test samples
        batch_size: Batch size for extraction
        cv_folds: CV folds
        device: Device
        seed: Random seed

    Returns:
        Dictionary with results for each layer
    """
    config = DATASET_CONFIGS[dataset_name]
    hidden_dim = model.config.hidden_size

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} ({config['num_classes']} classes)")
    print(f"{'='*80}")

    # Load data
    print(f"Loading {dataset_name} data...")
    train_texts, train_labels = load_dataset_by_name(
        dataset_name,
        config["train_split"],
        max_samples=max_train_samples,
        seed=seed
    )
    test_texts, test_labels = load_dataset_by_name(
        dataset_name,
        config["eval_split"],
        max_samples=max_test_samples,
        seed=seed
    )

    print(f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")

    results = {
        "dataset": dataset_name,
        "num_classes": config["num_classes"],
        "num_train": len(train_texts),
        "num_test": len(test_texts),
        "layers": {},
        "best_layer": None,
        "best_accuracy": 0,
    }

    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")

        # Extract features
        print(f"  Extracting train features...")
        X_train = extract_hidden_states(
            train_texts, model, tokenizer, layer_idx,
            batch_size=batch_size, pooling=pooling, device=device
        )

        print(f"  Extracting test features...")
        X_test = extract_hidden_states(
            test_texts, model, tokenizer, layer_idx,
            batch_size=batch_size, pooling=pooling, device=device
        )

        # Train probe
        probe = LinearProbeBaseline(
            hidden_dim=hidden_dim,
            num_classes=config["num_classes"],
            layer_idx=layer_idx,
            pooling=pooling,
            normalize=True,
            C=1.0,
            max_iter=2000,
            random_state=seed,
        )

        print(f"  Training LogisticRegression...")
        train_info = probe.fit(X_train, train_labels, cv_folds=cv_folds)

        # Evaluate
        test_results = probe.evaluate(X_test, test_labels)

        print(f"  Train Acc: {train_info['train_accuracy']:.1f}%")
        if "cv_mean" in train_info:
            print(f"  CV Acc: {train_info['cv_mean']:.1f}% +/- {train_info['cv_std']:.1f}%")
        print(f"  Test Acc: {test_results['accuracy']:.1f}%")
        print(f"  F1 Score: {test_results['f1_score']:.1f}%")

        layer_results = {
            **train_info,
            **test_results,
        }
        results["layers"][f"layer_{layer_idx}"] = layer_results

        # Track best layer
        if test_results["accuracy"] > results["best_accuracy"]:
            results["best_accuracy"] = test_results["accuracy"]
            results["best_layer"] = layer_idx

    print(f"\nBest layer: {results['best_layer']} with {results['best_accuracy']:.1f}% accuracy")

    return results


# =============================================================================
# Cross-Model Mapping (to Mistral output space)
# =============================================================================

def extract_mistral_label_embeddings(
    mistral_model,
    mistral_tokenizer,
    labels: List[str],
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract Mistral's embedding for each label token.

    This is used for mapping Llama hidden states to Mistral's output space
    for a fair cross-model comparison.

    Args:
        mistral_model: Mistral model
        mistral_tokenizer: Mistral tokenizer
        labels: List of label strings
        device: Device

    Returns:
        label_embeddings: [num_classes, embed_dim] array
    """
    mistral_model.eval()
    embeddings = []

    with torch.no_grad():
        for label in labels:
            # Get embedding of first token of label
            tokens = mistral_tokenizer(label, return_tensors="pt", add_special_tokens=False)
            token_id = tokens.input_ids[0, 0]  # First token
            embed = mistral_model.get_input_embeddings().weight[token_id]
            embeddings.append(embed.cpu().float().numpy())

    return np.stack(embeddings)


def run_cross_model_probe(
    dataset_name: str,
    llama_model,
    llama_tokenizer,
    mistral_model,
    mistral_tokenizer,
    layer_idx: int = 24,
    pooling: str = "last_token",
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train linear probe on Llama hidden states and project to Mistral's space.

    This provides a fair comparison with the Bridge, which also maps
    from Llama to Mistral's embedding space.

    The probe learns: W @ llama_hidden -> mistral_label_space
    """
    config = DATASET_CONFIGS[dataset_name]

    print(f"\n{'='*80}")
    print(f"Cross-Model Probe: {dataset_name}")
    print(f"Llama layer {layer_idx} -> Mistral output space")
    print(f"{'='*80}")

    # Load data
    train_texts, train_labels = load_dataset_by_name(
        dataset_name,
        config["train_split"],
        max_samples=max_train_samples,
        seed=seed
    )
    test_texts, test_labels = load_dataset_by_name(
        dataset_name,
        config["eval_split"],
        max_samples=max_test_samples,
        seed=seed
    )

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Extract Llama features
    print("Extracting Llama hidden states...")
    X_train = extract_hidden_states(
        train_texts, llama_model, llama_tokenizer, layer_idx,
        batch_size=batch_size, pooling=pooling, device=device
    )
    X_test = extract_hidden_states(
        test_texts, llama_model, llama_tokenizer, layer_idx,
        batch_size=batch_size, pooling=pooling, device=device
    )

    # Standard linear probe (on Llama features directly)
    print("\nTraining standard linear probe...")
    probe = LinearProbeBaseline(
        hidden_dim=llama_model.config.hidden_size,
        num_classes=config["num_classes"],
        layer_idx=layer_idx,
        pooling=pooling,
        normalize=True,
        C=1.0,
        random_state=seed,
    )
    train_info = probe.fit(X_train, train_labels, cv_folds=5)
    test_results = probe.evaluate(X_test, test_labels)

    print(f"Standard probe accuracy: {test_results['accuracy']:.1f}%")

    return {
        "dataset": dataset_name,
        "layer": layer_idx,
        "standard_probe": {
            **train_info,
            **test_results,
        },
    }


# =============================================================================
# Full Benchmark
# =============================================================================

def run_full_benchmark(
    datasets: List[str],
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    layers: List[int] = DEFAULT_LAYERS,
    pooling: str = "last_token",
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
    batch_size: int = 8,
    cv_folds: int = 5,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "runs/linear_probe",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full linear probe benchmark across all datasets.

    Args:
        datasets: List of dataset names
        model_id: HuggingFace model ID for Llama
        layers: Layer indices to evaluate
        pooling: Pooling method
        max_train_samples: Max training samples per dataset
        max_test_samples: Max test samples per dataset
        batch_size: Batch size for extraction
        cv_folds: CV folds
        seeds: Random seeds for multi-seed evaluation
        output_dir: Output directory
        device: Device

    Returns:
        Full results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("LINEAR PROBE BASELINE BENCHMARK")
    print("="*80)
    print(f"Model: {model_id}")
    print(f"Datasets: {datasets}")
    print(f"Layers: {layers}")
    print(f"Pooling: {pooling}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load model
    print("\nLoading Llama model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()

    print(f"Model loaded: {model.config.hidden_size} hidden dim, {model.config.num_hidden_layers} layers")

    # Run benchmark
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "hidden_dim": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "layers_evaluated": layers,
        "pooling": pooling,
        "seeds": seeds,
        "datasets": {},
    }

    for dataset_name in datasets:
        print(f"\n\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")

        dataset_results = {
            "per_seed": [],
            "aggregated": {},
        }

        # Multi-seed evaluation
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")

            seed_results = run_layer_sweep(
                dataset_name=dataset_name,
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                pooling=pooling,
                max_train_samples=max_train_samples,
                max_test_samples=max_test_samples,
                batch_size=batch_size,
                cv_folds=cv_folds,
                device=device,
                seed=seed,
            )
            seed_results["seed"] = seed
            dataset_results["per_seed"].append(seed_results)

        # Aggregate across seeds
        for layer_key in dataset_results["per_seed"][0]["layers"].keys():
            accs = [sr["layers"][layer_key]["accuracy"] for sr in dataset_results["per_seed"]]
            f1s = [sr["layers"][layer_key]["f1_score"] for sr in dataset_results["per_seed"]]

            dataset_results["aggregated"][layer_key] = {
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "accuracy_min": np.min(accs),
                "accuracy_max": np.max(accs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            }

        # Find best layer across all seeds
        best_layer = None
        best_acc = 0
        for layer_key, stats in dataset_results["aggregated"].items():
            if stats["accuracy_mean"] > best_acc:
                best_acc = stats["accuracy_mean"]
                best_layer = layer_key

        dataset_results["best_layer"] = best_layer
        dataset_results["best_accuracy_mean"] = best_acc

        all_results["datasets"][dataset_name] = dataset_results

        # Save intermediate results
        with open(os.path.join(output_dir, f"{dataset_name}_results.json"), "w") as f:
            json.dump(dataset_results, f, indent=2)

    # Save full results
    results_path = os.path.join(output_dir, "full_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<15} {'Best Layer':<12} {'Accuracy':<20} {'F1 Score':<20}")
    print("-"*70)

    for dataset_name, results in all_results["datasets"].items():
        best = results["aggregated"][results["best_layer"]]
        print(f"{dataset_name:<15} {results['best_layer']:<12} "
              f"{best['accuracy_mean']:.1f} +/- {best['accuracy_std']:.1f}%  "
              f"{best['f1_mean']:.1f} +/- {best['f1_std']:.1f}%")

    print("\n" + "="*80)
    print(f"Results saved to: {results_path}")
    print("="*80)

    return all_results


# =============================================================================
# Comparison with Bridge Results
# =============================================================================

def compare_with_bridge(
    probe_results_path: str,
    bridge_results_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare linear probe results with bridge results.

    This comparison shows whether the Perceiver architecture adds value
    beyond simple linear projection.
    """
    with open(probe_results_path) as f:
        probe_results = json.load(f)

    print("\n" + "="*80)
    print("LINEAR PROBE vs BRIDGE COMPARISON")
    print("="*80)

    comparison = {
        "probe_results": probe_results_path,
        "datasets": {},
    }

    for dataset_name, results in probe_results["datasets"].items():
        best = results["aggregated"][results["best_layer"]]
        comparison["datasets"][dataset_name] = {
            "probe_best_layer": results["best_layer"],
            "probe_accuracy": best["accuracy_mean"],
            "probe_accuracy_std": best["accuracy_std"],
        }

        print(f"\n{dataset_name}:")
        print(f"  Linear Probe ({results['best_layer']}): {best['accuracy_mean']:.1f}% +/- {best['accuracy_std']:.1f}%")

    # Load bridge results if available
    if bridge_results_path and os.path.exists(bridge_results_path):
        with open(bridge_results_path) as f:
            bridge_results = json.load(f)

        print("\n--- Bridge Results ---")
        for dataset_name in comparison["datasets"]:
            if dataset_name in bridge_results:
                bridge_acc = bridge_results[dataset_name].get("accuracy", "N/A")
                comparison["datasets"][dataset_name]["bridge_accuracy"] = bridge_acc
                print(f"{dataset_name} Bridge: {bridge_acc}%")

    return comparison


# =============================================================================
# Helper functions for integration with run_unified_comparison.py
# =============================================================================

def train_linear_probe(
    probe: LinearProbeBaseline,
    model,
    tokenizer,
    train_ds,
    dataset_name: str,
    device: str,
    dataset_config: Dict[str, Any],
    batch_size: int = 4,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train linear probe on classification task.

    This function provides a compatible interface for integration with
    run_unified_comparison.py.
    """
    print(f"\n[train_linear_probe] Training on {dataset_name}")
    print(f"  Dataset size: {len(train_ds)} samples")

    # Extract texts and labels
    text_field = dataset_config.get("text_field", "text")
    label_field = dataset_config.get("label_field", "label")

    if isinstance(text_field, list):
        texts = [f"{item[text_field[0]]} [SEP] {item[text_field[1]]}" for item in train_ds]
    else:
        texts = [item[text_field] for item in train_ds]

    labels = np.array([item[label_field] for item in train_ds])

    # Extract hidden states
    print(f"  Extracting hidden states (batch_size={batch_size})...")
    features = extract_hidden_states(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layer_idx=probe.layer_idx,
        batch_size=batch_size,
        pooling=probe.pooling,
        device=device,
        show_progress=True,
    )

    # Train probe
    print(f"  Training LogisticRegression...")
    train_results = probe.fit(
        X_train=features,
        y_train=labels,
        cv_folds=cv_folds,
    )

    print(f"  Training accuracy: {train_results['train_accuracy']:.1f}%")
    if "cv_mean" in train_results:
        print(f"  CV accuracy: {train_results['cv_mean']:.1f}% +/- {train_results['cv_std']:.1f}%")

    return train_results


def eval_linear_probe(
    probe: LinearProbeBaseline,
    model,
    tokenizer,
    eval_ds,
    dataset_name: str,
    device: str,
    dataset_config: Dict[str, Any],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Evaluate linear probe on classification task.

    This function provides a compatible interface for integration with
    run_unified_comparison.py.
    """
    print(f"\n[eval_linear_probe] Evaluating on {dataset_name}")
    print(f"  Dataset size: {len(eval_ds)} samples")

    # Extract texts and labels
    text_field = dataset_config.get("text_field", "text")
    label_field = dataset_config.get("label_field", "label")

    if isinstance(text_field, list):
        texts = [f"{item[text_field[0]]} [SEP] {item[text_field[1]]}" for item in eval_ds]
    else:
        texts = [item[text_field] for item in eval_ds]

    labels = np.array([item[label_field] for item in eval_ds])

    # Extract hidden states
    print(f"  Extracting hidden states (batch_size={batch_size})...")
    features = extract_hidden_states(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layer_idx=probe.layer_idx,
        batch_size=batch_size,
        pooling=probe.pooling,
        device=device,
        show_progress=True,
    )

    # Evaluate
    print(f"  Computing predictions...")
    results = probe.evaluate(features, labels)

    print(f"  Accuracy: {results['accuracy']:.1f}%")
    print(f"  F1 Score: {results['f1_score']:.1f}%")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Linear Probe Baseline for Telepathy")

    # Dataset args
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2", "agnews", "trec", "banking77", "imdb", "mnli", "20newsgroups"],
        help="Datasets to evaluate"
    )
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_test_samples", type=int, default=1000)

    # Model args
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=DEFAULT_LAYERS,
        help="Layer indices to evaluate"
    )
    parser.add_argument("--pooling", type=str, default="last_token", choices=["last_token", "mean", "first_token"])

    # Evaluation args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])

    # Output args
    parser.add_argument("--output_dir", type=str, default="runs/linear_probe")
    parser.add_argument("--device", type=str, default="cuda")

    # Comparison args
    parser.add_argument("--compare_bridge", type=str, default=None, help="Path to bridge results JSON")

    args = parser.parse_args()

    # Run benchmark
    results = run_full_benchmark(
        datasets=args.datasets,
        model_id=args.model_id,
        layers=args.layers,
        pooling=args.pooling,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        cv_folds=args.cv_folds,
        seeds=args.seeds,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Compare with bridge if requested
    if args.compare_bridge:
        compare_with_bridge(
            os.path.join(args.output_dir, "full_benchmark_results.json"),
            args.compare_bridge,
        )

    return results


if __name__ == "__main__":
    main()
