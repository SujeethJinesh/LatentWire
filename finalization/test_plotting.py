#!/usr/bin/env python3
"""
Test script for LatentWire plotting functions.

This script demonstrates how to use the comprehensive plotting function
to generate all required visualizations:
1. Learning curves (loss/F1 vs epoch)
2. Compression vs quality Pareto frontier
3. Latency comparison bar chart
4. Per-dataset performance heatmap

Usage:
    python test_plotting.py
"""

import sys
import numpy as np
from pathlib import Path

# Import the Visualizer from LATENTWIRE.py
sys.path.insert(0, str(Path(__file__).parent))
from LATENTWIRE import Visualizer

def generate_sample_results():
    """Generate sample experiment results for plotting."""

    # Sample training history
    num_epochs = 20
    epochs = list(range(1, num_epochs + 1))

    # Simulated training curves (loss decreases, F1 increases)
    train_loss = [3.5 * np.exp(-0.15 * i) + 0.5 + np.random.normal(0, 0.1) for i in range(num_epochs)]
    val_loss = [3.5 * np.exp(-0.12 * i) + 0.6 + np.random.normal(0, 0.12) for i in range(num_epochs)]
    train_f1 = [min(0.95, 0.1 + 0.8 * (1 - np.exp(-0.2 * i)) + np.random.normal(0, 0.02)) for i in range(num_epochs)]
    val_f1 = [min(0.92, 0.08 + 0.75 * (1 - np.exp(-0.18 * i)) + np.random.normal(0, 0.03)) for i in range(num_epochs)]

    # Compression results for different configurations
    compression_results = {
        "Baseline": {"compression_ratio": 1, "f1_score": 0.92},
        "LatentWire-32": {"compression_ratio": 4.2, "f1_score": 0.82},
        "LatentWire-16": {"compression_ratio": 8.5, "f1_score": 0.71},
        "LatentWire-8": {"compression_ratio": 16.8, "f1_score": 0.58},
        "LLMLingua": {"compression_ratio": 3.8, "f1_score": 0.78},
        "LinearProbe": {"compression_ratio": 5.2, "f1_score": 0.65},
        "Aggressive": {"compression_ratio": 32, "f1_score": 0.42},
    }

    # Latency results
    latency_results = {
        "Baseline": {"inference_ms": 145, "throughput": 6.9},
        "LatentWire": {"inference_ms": 42, "throughput": 23.8},
        "LLMLingua": {"inference_ms": 78, "throughput": 12.8},
        "Linear Probe": {"inference_ms": 28, "throughput": 35.7},
        "Cached": {"inference_ms": 12, "throughput": 83.3},
    }

    # Per-dataset results
    dataset_results = {
        "SQuAD": {"F1": 0.82, "EM": 0.75, "BLEU": 0.78, "Accuracy": 0.80, "Compression": 0.85},
        "HotpotQA": {"F1": 0.68, "EM": 0.62, "BLEU": 0.65, "Accuracy": 0.70, "Compression": 0.82},
        "SST-2": {"F1": 0.88, "EM": 0.85, "BLEU": 0.80, "Accuracy": 0.92, "Compression": 0.78},
        "AG News": {"F1": 0.85, "EM": 0.82, "BLEU": 0.76, "Accuracy": 0.88, "Compression": 0.80},
        "TREC": {"F1": 0.75, "EM": 0.70, "BLEU": 0.72, "Accuracy": 0.78, "Compression": 0.75},
        "MRPC": {"F1": 0.79, "EM": 0.76, "BLEU": 0.74, "Accuracy": 0.82, "Compression": 0.77},
    }

    # Summary metrics for the combined plot
    summary_metrics = {
        "Best F1": 0.82,
        "Compression": "4.2x",
        "Latency": "42ms",
        "Throughput": "23.8/s",
        "Parameters": "12.5M",
    }

    return {
        "training_history": {
            "epochs": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1": train_f1,
            "val_f1": val_f1,
        },
        "compression_results": compression_results,
        "latency_results": latency_results,
        "dataset_results": dataset_results,
        "summary_metrics": summary_metrics,
    }

def main():
    """Main function to test plotting capabilities."""

    print("LatentWire Plotting Test")
    print("=" * 80)

    # Initialize visualizer with custom save directory
    save_dir = "latex/figures"
    viz = Visualizer(save_dir=save_dir)

    # Generate sample results
    print("\nGenerating sample experiment results...")
    experiment_results = generate_sample_results()

    # Create comprehensive plots
    print("\nGenerating comprehensive visualizations...")
    viz.plot_comprehensive_results(
        experiment_results=experiment_results,
        save_dir=save_dir
    )

    print("\n" + "=" * 80)
    print("Plotting complete! Generated files:")
    print(f"  1. {save_dir}/learning_curves.pdf/.png - Training loss and F1 curves")
    print(f"  2. {save_dir}/pareto_frontier.pdf/.png - Compression vs quality trade-off")
    print(f"  3. {save_dir}/latency_comparison.pdf/.png - Inference speed comparison")
    print(f"  4. {save_dir}/performance_heatmap.pdf/.png - Per-dataset performance matrix")
    print(f"  5. {save_dir}/experiment_summary.pdf/.png - Combined summary figure")
    print("\nAll figures are saved in both PDF (for LaTeX) and PNG (for viewing) formats.")
    print("=" * 80)

if __name__ == "__main__":
    main()