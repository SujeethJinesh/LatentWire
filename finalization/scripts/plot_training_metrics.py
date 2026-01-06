#!/usr/bin/env python3
"""
Plot training metrics from JSON files saved during training.
Reads results_epoch_*.json files and creates visualization graphs.
"""

import json
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def load_metrics_from_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file (one JSON object per line)."""
    metrics = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def load_metrics_from_json_files(directory: Path) -> List[Dict[str, Any]]:
    """Load metrics from individual results_epoch_*.json files."""
    metrics = []
    json_files = sorted(directory.glob('results_epoch_*.json'))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            metrics.append(data)

    return sorted(metrics, key=lambda x: x.get('epoch', 0))


def plot_losses(metrics: List[Dict[str, Any]], output_dir: Path):
    """Plot training and validation losses over epochs."""
    epochs = [m['epoch'] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot losses
    train_losses = [m.get('train_loss', np.nan) for m in metrics]
    val_losses = [m.get('validation_loss_proxy', np.nan) for m in metrics]

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
    if not all(np.isnan(val_losses)):
        ax1.plot(epochs, val_losses, 'r--', label='Validation Loss (proxy)', marker='s')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot perplexity
    train_perp = [m.get('train_perplexity', np.nan) for m in metrics]
    val_perp = [m.get('validation_perplexity_proxy', np.nan) for m in metrics]

    ax2.plot(epochs, train_perp, 'b-', label='Training Perplexity', marker='o')
    if not all(np.isnan(val_perp)):
        ax2.plot(epochs, val_perp, 'r--', label='Validation Perplexity (proxy)', marker='s')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'losses_and_perplexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'losses_and_perplexity.png'}")


def plot_first_token_accuracy(metrics: List[Dict[str, Any]], output_dir: Path):
    """Plot first token accuracy metrics."""
    epochs = [m['epoch'] for m in metrics]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot first token accuracy for each model
    for model_name in ['llama', 'qwen']:
        model_metrics_key = f'{model_name}_metrics'
        first_acc = []
        first_acc_max = []

        for m in metrics:
            if model_metrics_key in m:
                model_data = m[model_metrics_key]
                first_acc.append(model_data.get('first_token_accuracy', np.nan))
                first_acc_max.append(model_data.get('first_token_accuracy_max', np.nan))

        if first_acc and not all(np.isnan(first_acc)):
            ax.plot(epochs, first_acc, label=f'{model_name.capitalize()} First Token Acc', marker='o')
            if not all(np.isnan(first_acc_max)):
                ax.plot(epochs, first_acc_max, '--', alpha=0.5,
                       label=f'{model_name.capitalize()} First Token Max')

    # Plot EMA and best accuracy if available
    first_acc_ema = [m.get('first_acc_ema', np.nan) for m in metrics]
    best_first_acc = [m.get('best_first_acc', np.nan) for m in metrics]

    if not all(np.isnan(first_acc_ema)):
        ax.plot(epochs, first_acc_ema, 'g-', label='First Acc EMA', linewidth=2)
    if not all(np.isnan(best_first_acc)):
        ax.plot(epochs, best_first_acc, 'k:', label='Best First Acc', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('First Token Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'first_token_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'first_token_accuracy.png'}")


def plot_evaluation_metrics(metrics: List[Dict[str, Any]], output_dir: Path):
    """Plot F1, exact match, and other evaluation metrics if available."""
    epochs = [m['epoch'] for m in metrics]

    # Check if evaluation metrics are available
    has_f1 = any(m.get('f1_score') is not None for m in metrics)
    has_em = any(m.get('exact_match') is not None for m in metrics)

    if not has_f1 and not has_em:
        print("No evaluation metrics (F1/EM) available yet.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if has_f1:
        f1_scores = [m.get('f1_score', np.nan) for m in metrics]
        ax.plot(epochs, f1_scores, 'b-', label='F1 Score', marker='o')

    if has_em:
        em_scores = [m.get('exact_match', np.nan) for m in metrics]
        ax.plot(epochs, em_scores, 'r-', label='Exact Match', marker='s')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'evaluation_metrics.png'}")


def create_summary_table(metrics: List[Dict[str, Any]], output_dir: Path):
    """Create a summary table of key metrics."""
    if not metrics:
        print("No metrics to summarize")
        return

    latest = metrics[-1]

    summary = []
    summary.append("=" * 60)
    summary.append("TRAINING METRICS SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Total Epochs: {latest['epoch']}")
    summary.append(f"Global Steps: {latest.get('global_step', 'N/A')}")
    summary.append("")

    summary.append("Final Epoch Metrics:")
    summary.append(f"  Train Loss: {latest.get('train_loss', 'N/A'):.4f}" if 'train_loss' in latest else "  Train Loss: N/A")
    summary.append(f"  Train Perplexity: {latest.get('train_perplexity', 'N/A'):.2f}" if 'train_perplexity' in latest else "  Train Perplexity: N/A")
    summary.append(f"  Validation Loss (proxy): {latest.get('validation_loss_proxy', 'N/A'):.4f}" if 'validation_loss_proxy' in latest else "  Validation Loss: N/A")
    summary.append(f"  First Token Acc (EMA): {latest.get('first_acc_ema', 'N/A'):.4f}" if 'first_acc_ema' in latest else "  First Token Acc: N/A")
    summary.append("")

    if 'config' in latest:
        summary.append("Configuration:")
        for key, value in latest['config'].items():
            summary.append(f"  {key}: {value}")

    summary.append("=" * 60)

    summary_text = "\n".join(summary)

    # Save to file
    summary_file = output_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)

    # Print to console
    print(summary_text)
    print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from JSON files')
    parser.add_argument('run_dir', type=str, help='Directory containing training run results')
    parser.add_argument('--output-dir', type=str, help='Output directory for plots (default: same as run_dir)')
    parser.add_argument('--use-jsonl', action='store_true', help='Read from training_metrics.jsonl instead of individual files')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        return

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    if args.use_jsonl:
        jsonl_file = run_dir / 'training_metrics.jsonl'
        if not jsonl_file.exists():
            print(f"Error: {jsonl_file} does not exist")
            return
        metrics = load_metrics_from_jsonl(jsonl_file)
        print(f"Loaded {len(metrics)} epochs from {jsonl_file}")
    else:
        metrics = load_metrics_from_json_files(run_dir)
        print(f"Loaded {len(metrics)} epoch files from {run_dir}")

    if not metrics:
        print("No metrics found to plot")
        return

    # Create plots
    plot_losses(metrics, output_dir)
    plot_first_token_accuracy(metrics, output_dir)
    plot_evaluation_metrics(metrics, output_dir)

    # Create summary
    create_summary_table(metrics, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()