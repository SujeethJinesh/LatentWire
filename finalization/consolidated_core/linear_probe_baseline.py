#!/usr/bin/env python3
"""
Linear Probe Baseline for LatentWire/Telepathy Project

This implements a scientifically rigorous linear probe baseline using sklearn's
LogisticRegression. This is a CRITICAL baseline demanded by reviewers for comparing
against the Bridge method.

Key Features:
- Uses sklearn LogisticRegression (not neural networks) for scientific rigor
- Memory-efficient batch processing to handle large datasets
- Proper cross-validation with stratified splits
- Save/load probe weights for reproducibility
- Integration with run_unified_comparison.py

Architecture:
1. Extract frozen embeddings from source model (e.g., Llama)
2. Pool hidden states (mean/last-token/first-token)
3. Train LogisticRegression with L2 regularization
4. Evaluate on classification tasks (SST-2, AG News, TREC)

This baseline tests whether the sender model's hidden states already contain
sufficient information for the task, without needing cross-model transfer.

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path


class LinearProbeBaseline:
    """
    Linear probe baseline using sklearn's LogisticRegression.

    This class provides a scientifically rigorous baseline by:
    1. Extracting frozen embeddings from a pre-trained LLM
    2. Training a simple linear classifier (LogisticRegression)
    3. Supporting proper cross-validation and reproducibility

    Args:
        hidden_dim: Dimension of hidden states (e.g., 4096 for Llama-8B)
        num_classes: Number of output classes
        layer_idx: Which layer to extract from (0=embeddings, -1=last layer)
        pooling: How to pool hidden states ("mean", "last_token", "first_token")
        normalize: Whether to standardize features before classification
        C: Inverse of regularization strength (smaller = stronger regularization)
        max_iter: Maximum iterations for LogisticRegression solver
        n_jobs: Number of parallel jobs for cross-validation
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        layer_idx: int = 24,
        pooling: str = "mean",
        normalize: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
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
        self.probe = LogisticRegression(
            C=C,
            max_iter=max_iter,
            multi_class='multinomial' if num_classes > 2 else 'ovr',
            solver='lbfgs' if num_classes <= 10 else 'saga',
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0,
        )

        # Storage for extracted features (memory-efficient)
        self._feature_cache = {}

    def extract_hidden_states_batch(
        self,
        model,
        tokenizer,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "cuda",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract hidden states from model in batches (memory-efficient).

        Args:
            model: Frozen LLM to extract hidden states from
            tokenizer: Tokenizer for the model
            texts: List of input texts
            batch_size: Batch size for processing (small to avoid OOM)
            max_length: Maximum sequence length
            device: Device to run extraction on
            show_progress: Whether to show progress bar

        Returns:
            features: [N, hidden_dim] numpy array of pooled hidden states
        """
        model.eval()
        all_features = []

        # Process in batches to avoid OOM
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting layer {self.layer_idx} embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]

                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Extract hidden states
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get hidden states from specific layer
                # Note: outputs.hidden_states[0] = embeddings, [1] = first layer, etc.
                hidden = outputs.hidden_states[self.layer_idx]  # [B, seq_len, hidden_dim]

                # Pool across sequence dimension
                if self.pooling == "mean":
                    # Mean pooling over non-padding tokens
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    masked_hidden = hidden * mask
                    pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                elif self.pooling == "last_token":
                    # Use last non-padding token
                    seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
                    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                    pooled = hidden[batch_indices, seq_lengths]
                elif self.pooling == "first_token":
                    # Use first token
                    pooled = hidden[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")

                # Move to CPU and convert to numpy
                pooled_np = pooled.cpu().numpy().astype(np.float32)
                all_features.append(pooled_np)

        # Concatenate all batches
        features = np.vstack(all_features)
        return features

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
        return_cv_scores: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit the linear probe with optional cross-validation.

        Args:
            X_train: Training features [N, hidden_dim]
            y_train: Training labels [N]
            cv_folds: Number of cross-validation folds (0 = no CV)
            return_cv_scores: Whether to compute and return CV scores

        Returns:
            Dictionary with training info (train_acc, cv_scores if requested)
        """
        # Standardize features if requested
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)

        # Optional cross-validation for hyperparameter selection
        cv_scores = None
        if cv_folds > 0 and return_cv_scores:
            print(f"  Running {cv_folds}-fold cross-validation...")
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.probe, X_train, y_train,
                cv=skf, scoring='accuracy', n_jobs=self.n_jobs
            )
            print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Fit on full training set
        print(f"  Fitting LogisticRegression (C={self.C}, max_iter={self.max_iter})...")
        self.probe.fit(X_train, y_train)

        # Compute training accuracy
        y_pred = self.probe.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)

        results = {
            "train_accuracy": train_acc * 100,
            "num_samples": len(X_train),
            "num_features": X_train.shape[1],
        }

        if cv_scores is not None:
            results["cv_mean"] = cv_scores.mean() * 100
            results["cv_std"] = cv_scores.std() * 100
            results["cv_scores"] = (cv_scores * 100).tolist()

        return results

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test features.

        Args:
            X_test: Test features [N, hidden_dim]

        Returns:
            Predicted labels [N]
        """
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        return self.probe.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test features.

        Args:
            X_test: Test features [N, hidden_dim]

        Returns:
            Class probabilities [N, num_classes]
        """
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        return self.probe.predict_proba(X_test)

    def save(self, filepath: str):
        """
        Save probe weights and scaler for reproducibility.

        Args:
            filepath: Path to save the probe (without extension)
        """
        save_dict = {
            "probe": self.probe,
            "scaler": self.scaler,
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
                "layer_idx": self.layer_idx,
                "pooling": self.pooling,
                "normalize": self.scaler is not None,
                "C": self.C,
                "max_iter": self.max_iter,
            }
        }
        joblib.dump(save_dict, f"{filepath}.joblib")
        print(f"  Saved probe to {filepath}.joblib")

    @classmethod
    def load(cls, filepath: str):
        """
        Load saved probe weights and configuration.

        Args:
            filepath: Path to saved probe (without extension)

        Returns:
            Loaded LinearProbeBaseline instance
        """
        save_dict = joblib.load(f"{filepath}.joblib")

        # Create new instance with saved config
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

        # Load trained components
        instance.probe = save_dict["probe"]
        instance.scaler = save_dict["scaler"]

        return instance


# =============================================================================
# Training and Evaluation Functions
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

    This function:
    1. Extracts hidden states from the frozen model
    2. Trains LogisticRegression with cross-validation
    3. Returns training metrics

    Args:
        probe: LinearProbeBaseline instance
        model: Frozen LLM to extract hidden states from
        tokenizer: Tokenizer for the model
        train_ds: Training dataset
        dataset_name: Name of dataset (for logging)
        device: Device to run extraction on
        dataset_config: Dataset configuration dict
        batch_size: Batch size for feature extraction
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with training metrics
    """
    print(f"\n[train_linear_probe] Training on {dataset_name}")
    print(f"  Dataset size: {len(train_ds)} samples")
    print(f"  Extracting from layer {probe.layer_idx} with {probe.pooling} pooling")

    # Extract texts and labels
    texts = [item[dataset_config["text_field"]] for item in train_ds]
    labels = np.array([item[dataset_config["label_field"]] for item in train_ds])

    # Extract hidden states (memory-efficient batching)
    print(f"  Extracting hidden states (batch_size={batch_size})...")
    features = probe.extract_hidden_states_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
        device=device,
        show_progress=True,
    )

    # Train probe with cross-validation
    print(f"  Training LogisticRegression...")
    train_results = probe.fit(
        X_train=features,
        y_train=labels,
        cv_folds=cv_folds,
        return_cv_scores=True,
    )

    print(f"  Training accuracy: {train_results['train_accuracy']:.1f}%")
    if "cv_mean" in train_results:
        print(f"  Cross-validation: {train_results['cv_mean']:.1f}% ± {train_results['cv_std']:.1f}%")

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

    Args:
        probe: Trained LinearProbeBaseline instance
        model: Frozen LLM to extract hidden states from
        tokenizer: Tokenizer for the model
        eval_ds: Evaluation dataset
        dataset_name: Name of dataset (for logging)
        device: Device to run extraction on
        dataset_config: Dataset configuration dict
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n[eval_linear_probe] Evaluating on {dataset_name}")
    print(f"  Dataset size: {len(eval_ds)} samples")

    # Extract texts and labels
    texts = [item[dataset_config["text_field"]] for item in eval_ds]
    labels = np.array([item[dataset_config["label_field"]] for item in eval_ds])

    # Extract hidden states
    print(f"  Extracting hidden states (batch_size={batch_size})...")
    features = probe.extract_hidden_states_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
        device=device,
        show_progress=True,
    )

    # Predict
    print(f"  Computing predictions...")
    predictions = probe.predict(features)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)

    # Compute per-class F1 scores
    if dataset_config["num_classes"] == 2:
        f1 = f1_score(labels, predictions, average='binary')
    else:
        f1 = f1_score(labels, predictions, average='macro')

    results = {
        "accuracy": accuracy * 100,
        "f1_score": f1 * 100,
        "correct": int(np.sum(predictions == labels)),
        "total": len(labels),
    }

    print(f"  Accuracy: {results['accuracy']:.1f}%")
    print(f"  F1 Score: {results['f1_score']:.1f}%")

    return results


# =============================================================================
# Integration Example for run_unified_comparison.py
# =============================================================================

def integration_example():
    """
    Example showing how to integrate LinearProbeBaseline into run_unified_comparison.py.
    """
    code = '''
    # In run_unified_comparison.py, add after loading models:

    from linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe

    # Initialize results tracking
    seed_results_list = {
        "bridge": [],
        "prompt_tuning": [],
        "linear_probe": [],  # ADD THIS
        "text_relay": [],
        "fewshot": [],
        "zeroshot_sender": [],
        "zeroshot_receiver": [],
    }

    # For each dataset and seed:
    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]

        # ... existing code ...

        # LINEAR PROBE BASELINE (using sklearn)
        print(f"\\n[LINEAR PROBE] Training sklearn LogisticRegression (seed={seed})...")

        linear_probe = LinearProbeBaseline(
            hidden_dim=sender_dim,  # e.g., 4096 for Llama-8B
            num_classes=config["num_classes"],
            layer_idx=24,  # Layer 24 often works well
            pooling="mean",  # or "last_token"
            normalize=True,  # Standardize features
            C=1.0,  # Regularization strength
            max_iter=1000,
            n_jobs=-1,  # Use all CPU cores
            random_state=seed,
        )

        # Train with cross-validation
        train_info = train_linear_probe(
            probe=linear_probe,
            model=sender,
            tokenizer=sender_tok,
            train_ds=train_ds,
            dataset_name=dataset_name,
            device=device,
            dataset_config=config,
            batch_size=4,  # Small batch to avoid OOM
            cv_folds=5,  # 5-fold cross-validation
        )

        # Evaluate
        linear_probe_results = eval_linear_probe(
            probe=linear_probe,
            model=sender,
            tokenizer=sender_tok,
            eval_ds=eval_ds,
            dataset_name=dataset_name,
            device=device,
            dataset_config=config,
            batch_size=4,
        )

        # Add training info to results
        linear_probe_results["train_info"] = train_info

        # Save results
        dataset_results["linear_probe"] = linear_probe_results
        seed_results_list["linear_probe"].append(linear_probe_results)

        # Save probe weights for reproducibility
        probe_path = f"{args.output_dir}/linear_probe_{dataset_name}_seed{seed}"
        linear_probe.save(probe_path)

        print(f"  Linear probe accuracy: {linear_probe_results['accuracy']:.1f}%")
        if "cv_mean" in train_info:
            print(f"  CV accuracy: {train_info['cv_mean']:.1f}% ± {train_info['cv_std']:.1f}%")
    '''
    return code


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Linear Probe Baseline - sklearn Implementation")
    print("="*80)

    print("""
    This implementation provides a scientifically rigorous baseline using:

    1. sklearn's LogisticRegression (not neural networks)
    2. Memory-efficient batch processing for large datasets
    3. Proper cross-validation with stratified k-fold
    4. Feature standardization (optional but recommended)
    5. Save/load functionality for reproducibility

    Key Advantages:
    - Uses well-established sklearn library (reviewer-friendly)
    - Proper statistical methodology with cross-validation
    - Memory-efficient processing (won't OOM on large datasets)
    - Deterministic and reproducible results
    - Fast training on CPU (no GPU needed for LogisticRegression)

    Integration:
    - Drop-in replacement for the PyTorch version
    - Works with run_unified_comparison.py
    - Supports multi-seed evaluation
    - Saves probe weights for later analysis
    """)

    print("\n" + "="*80)
    print("Example Integration Code:")
    print("="*80)
    print(integration_example())

    print("\n" + "="*80)
    print("Quick Test Example:")
    print("="*80)
    print("""
    # Quick test to verify it works:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from linear_probe_baseline import LinearProbeBaseline

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Create probe
    probe = LinearProbeBaseline(
        hidden_dim=4096,
        num_classes=2,
        layer_idx=24,
        pooling="mean",
    )

    # Extract features for some texts
    texts = ["This movie is great!", "This movie is terrible."]
    features = probe.extract_hidden_states_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=2,
        device="cuda",
    )

    print(f"Extracted features shape: {features.shape}")
    """)