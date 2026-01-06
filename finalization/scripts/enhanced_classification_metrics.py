#!/usr/bin/env python3
"""
Enhanced classification metrics capture for LatentWire experiments.

This module provides comprehensive classification metrics including:
- Confusion matrices with visualization
- Per-class precision, recall, F1
- Classification reports in multiple formats
- ROC curves and AUC scores for binary classification
- Statistical significance testing between classifiers
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict


@dataclass
class ClassificationMetrics:
    """Container for comprehensive classification metrics."""

    # Basic metrics
    accuracy: float
    macro_f1: float
    weighted_f1: float

    # Per-class metrics
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]

    # Confusion matrix
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray

    # Additional metrics
    cohen_kappa: float
    matthews_corrcoef: float

    # For binary classification
    roc_auc: Optional[float] = None
    roc_curve_data: Optional[Dict] = None

    # Statistical measures
    confidence_interval_95: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'accuracy': float(self.accuracy),
            'macro_f1': float(self.macro_f1),
            'weighted_f1': float(self.weighted_f1),
            'cohen_kappa': float(self.cohen_kappa),
            'matthews_corrcoef': float(self.matthews_corrcoef),
            'per_class': {
                'precision': {k: float(v) for k, v in self.per_class_precision.items()},
                'recall': {k: float(v) for k, v in self.per_class_recall.items()},
                'f1': {k: float(v) for k, v in self.per_class_f1.items()},
                'support': self.per_class_support
            },
            'confusion_matrix': self.confusion_matrix.tolist(),
            'normalized_confusion_matrix': self.normalized_confusion_matrix.tolist()
        }

        if self.roc_auc is not None:
            result['roc_auc'] = float(self.roc_auc)
        if self.roc_curve_data is not None:
            result['roc_curve'] = self.roc_curve_data
        if self.confidence_interval_95 is not None:
            result['confidence_interval_95'] = list(self.confidence_interval_95)

        return result

    def summary_string(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Classification Metrics Summary:",
            f"  Accuracy: {self.accuracy:.4f}",
            f"  Macro F1: {self.macro_f1:.4f}",
            f"  Weighted F1: {self.weighted_f1:.4f}",
            f"  Cohen's Kappa: {self.cohen_kappa:.4f}",
            f"  Matthews Correlation: {self.matthews_corrcoef:.4f}"
        ]

        if self.roc_auc is not None:
            lines.append(f"  ROC-AUC: {self.roc_auc:.4f}")

        if self.confidence_interval_95 is not None:
            lines.append(f"  95% CI: [{self.confidence_interval_95[0]:.4f}, {self.confidence_interval_95[1]:.4f}]")

        lines.append("\nPer-Class Metrics:")
        for class_name in self.per_class_precision.keys():
            lines.append(
                f"  {class_name}: "
                f"P={self.per_class_precision[class_name]:.3f} "
                f"R={self.per_class_recall[class_name]:.3f} "
                f"F1={self.per_class_f1[class_name]:.3f} "
                f"N={self.per_class_support[class_name]}"
            )

        return "\n".join(lines)


def compute_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    y_proba: Optional[np.ndarray] = None,
    compute_confidence_interval: bool = True,
    n_bootstrap: int = 1000,
    save_path: Optional[Path] = None,
    plot_confusion_matrix: bool = True
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels for display (if None, inferred from data)
        y_proba: Probability predictions for ROC-AUC (binary only)
        compute_confidence_interval: Whether to compute bootstrap CI
        n_bootstrap: Number of bootstrap iterations
        save_path: Path to save results and plots
        plot_confusion_matrix: Whether to generate confusion matrix plot

    Returns:
        ClassificationMetrics object with all computed metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Infer labels if not provided
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Create per-class dictionaries
    label_names = [str(l) for l in labels]
    per_class_precision = dict(zip(label_names, precision))
    per_class_recall = dict(zip(label_names, recall))
    per_class_f1 = dict(zip(label_names, f1))
    per_class_support = dict(zip(label_names, support.astype(int)))

    # Aggregate F1 scores
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.nan_to_num(cm_normalized, copy=False, nan=0.0)

    # Additional metrics
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # ROC-AUC for binary classification
    roc_auc = None
    roc_curve_data = None
    if len(labels) == 2 and y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
            roc_curve_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")

    # Bootstrap confidence interval
    ci = None
    if compute_confidence_interval and len(y_true) >= 30:
        ci = _bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap)

    # Create metrics object
    metrics = ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=cm,
        normalized_confusion_matrix=cm_normalized,
        cohen_kappa=cohen_kappa,
        matthews_corrcoef=mcc,
        roc_auc=roc_auc,
        roc_curve_data=roc_curve_data,
        confidence_interval_95=ci
    )

    # Save results if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        with open(save_path / 'classification_metrics.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save detailed classification report
        report = classification_report(
            y_true, y_pred, labels=labels,
            target_names=label_names, output_dict=True
        )
        with open(save_path / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Plot confusion matrix
        if plot_confusion_matrix:
            _plot_confusion_matrix(
                cm_normalized, label_names,
                save_path / 'confusion_matrix.png'
            )

            # Also save raw confusion matrix plot
            _plot_confusion_matrix(
                cm, label_names,
                save_path / 'confusion_matrix_raw.png',
                normalize=False
            )

    return metrics


def _bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for accuracy."""
    n_samples = len(y_true)
    accuracies = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, n_samples, replace=True)
        acc = accuracy_score(y_true[indices], y_pred[indices])
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    alpha = 1 - confidence_level
    lower = np.percentile(accuracies, 100 * alpha / 2)
    upper = np.percentile(accuracies, 100 * (1 - alpha / 2))

    return (lower, upper)


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: Path,
    normalize: bool = True
) -> None:
    """Generate and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))

    # Use appropriate format string
    fmt = '.2f' if normalize else 'd'
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'

    # Create heatmap
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_classifiers(
    results_dict: Dict[str, Dict[str, Union[List, np.ndarray]]],
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compare multiple classifiers with statistical tests.

    Args:
        results_dict: Dictionary mapping classifier names to
                     {'y_true': [...], 'y_pred': [...]}
        save_path: Path to save comparison results

    Returns:
        Dictionary with comparison results and significance tests
    """
    comparison = {}

    # Compute metrics for each classifier
    for name, data in results_dict.items():
        metrics = compute_classification_metrics(
            data['y_true'], data['y_pred'],
            compute_confidence_interval=True
        )
        comparison[name] = {
            'accuracy': metrics.accuracy,
            'macro_f1': metrics.macro_f1,
            'confidence_interval': metrics.confidence_interval_95
        }

    # Pairwise significance tests
    from scipy.stats import mcnemar
    classifiers = list(results_dict.keys())
    significance_tests = {}

    for i, clf1 in enumerate(classifiers):
        for clf2 in classifiers[i+1:]:
            y_true = results_dict[clf1]['y_true']
            pred1 = results_dict[clf1]['y_pred']
            pred2 = results_dict[clf2]['y_pred']

            # McNemar's test
            correct1 = (pred1 == y_true)
            correct2 = (pred2 == y_true)

            # Build contingency table
            n00 = np.sum(~correct1 & ~correct2)  # Both wrong
            n01 = np.sum(~correct1 & correct2)   # 1 wrong, 2 right
            n10 = np.sum(correct1 & ~correct2)   # 1 right, 2 wrong
            n11 = np.sum(correct1 & correct2)    # Both right

            table = [[n11, n10], [n01, n00]]
            result = mcnemar(table, exact=False, correction=True)

            significance_tests[f"{clf1}_vs_{clf2}"] = {
                'statistic': float(result.statistic),
                'p_value': float(result.pvalue),
                'significant_at_0.05': result.pvalue < 0.05
            }

    # Build final comparison
    final_comparison = {
        'classifiers': comparison,
        'significance_tests': significance_tests
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / 'classifier_comparison.json', 'w') as f:
            json.dump(final_comparison, f, indent=2)

    return final_comparison


def main():
    """Example usage and testing."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3

    # Simulate predictions from two classifiers
    y_true = np.random.randint(0, n_classes, n_samples)

    # Classifier 1: 85% accuracy
    y_pred1 = y_true.copy()
    mask = np.random.random(n_samples) < 0.15
    y_pred1[mask] = np.random.randint(0, n_classes, np.sum(mask))

    # Classifier 2: 80% accuracy
    y_pred2 = y_true.copy()
    mask = np.random.random(n_samples) < 0.20
    y_pred2[mask] = np.random.randint(0, n_classes, np.sum(mask))

    # Compute metrics for classifier 1
    print("Computing metrics for Classifier 1...")
    metrics1 = compute_classification_metrics(
        y_true, y_pred1,
        labels=['Class A', 'Class B', 'Class C'],
        save_path=Path('test_metrics/classifier1')
    )
    print(metrics1.summary_string())
    print()

    # Compare classifiers
    print("Comparing classifiers...")
    comparison = compare_classifiers(
        {
            'Classifier1': {'y_true': y_true, 'y_pred': y_pred1},
            'Classifier2': {'y_true': y_true, 'y_pred': y_pred2}
        },
        save_path=Path('test_metrics/comparison')
    )

    print("Comparison Results:")
    for clf, metrics in comparison['classifiers'].items():
        print(f"  {clf}: Accuracy={metrics['accuracy']:.3f}, "
              f"CI={metrics['confidence_interval']}")

    print("\nSignificance Tests:")
    for test, result in comparison['significance_tests'].items():
        print(f"  {test}: p-value={result['p_value']:.4f}, "
              f"significant={result['significant_at_0.05']}")


if __name__ == "__main__":
    main()