#!/usr/bin/env python3
"""
Analyze results from 8-epoch training runs.

This script processes the outputs from 8-epoch training experiments and
generates publication-ready figures and tables.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_epoch_results(output_dir: Path):
    """Load evaluation results from all epochs."""
    epoch_dir = output_dir / "epoch_evals"
    results = {}

    for epoch in range(1, 9):
        eval_file = epoch_dir / f"epoch{epoch}_eval.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results[epoch] = json.load(f)

    return results


def load_training_metrics(output_dir: Path):
    """Load training metrics from diagnostics files."""
    metrics = []

    for epoch in range(1, 9):
        diag_file = output_dir / f"epoch{epoch}" / "diagnostics.jsonl"
        if diag_file.exists():
            epoch_metrics = []
            with open(diag_file, 'r') as f:
                for line in f:
                    epoch_metrics.append(json.loads(line))
            if epoch_metrics:
                metrics.append({
                    'epoch': epoch,
                    'metrics': epoch_metrics
                })

    return metrics


def create_training_curves(results, output_dir):
    """Create comprehensive training curve plots."""
    epochs = sorted(results.keys())

    # Extract metrics
    metrics_data = {
        'epoch': [],
        'latent_f1': [],
        'text_f1': [],
        'token_budget_f1': [],
        'first_tok_acc': [],
        'latent_nll': [],
        'text_nll': [],
        'compression_ratio': []
    }

    for epoch in epochs:
        if 'aggregate_stats' in results[epoch]:
            stats = results[epoch]['aggregate_stats']
            metrics_data['epoch'].append(epoch)
            metrics_data['latent_f1'].append(stats.get('latent_f1_mean', 0))
            metrics_data['text_f1'].append(stats.get('text_f1_mean', 0))
            metrics_data['token_budget_f1'].append(stats.get('token_budget_f1_mean', 0))
            metrics_data['first_tok_acc'].append(stats.get('first_tok_acc', 0))
            metrics_data['latent_nll'].append(stats.get('latent_nll_mean', 0))
            metrics_data['text_nll'].append(stats.get('text_nll_mean', 0))

            # Calculate compression ratio if available
            if 'compression_stats' in results[epoch]:
                comp = results[epoch]['compression_stats']
                ratio = comp.get('text_bytes', 1) / comp.get('latent_bytes', 1)
                metrics_data['compression_ratio'].append(ratio)
            else:
                metrics_data['compression_ratio'].append(4.0)  # Default estimate

    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. F1 Score Comparison
    axes[0, 0].plot(metrics_data['epoch'], metrics_data['latent_f1'],
                    'b-o', label='Latent', linewidth=2, markersize=8)
    axes[0, 0].plot(metrics_data['epoch'], metrics_data['text_f1'],
                    'g--s', label='Text Baseline', linewidth=2, markersize=8)
    axes[0, 0].plot(metrics_data['epoch'], metrics_data['token_budget_f1'],
                    'r-.^', label='Token Budget', linewidth=1.5, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('F1 Score', fontsize=12)
    axes[0, 0].set_title('F1 Score Progression', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # 2. First Token Accuracy
    axes[0, 1].plot(metrics_data['epoch'], metrics_data['first_tok_acc'],
                    'purple', marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('First Token Accuracy', fontsize=12)
    axes[0, 1].set_title('First Token Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Add target line
    axes[0, 1].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5,
                       label='Target (20%)')
    axes[0, 1].legend()

    # 3. NLL Comparison
    axes[1, 0].plot(metrics_data['epoch'], metrics_data['latent_nll'],
                    'b-o', label='Latent', linewidth=2, markersize=8)
    axes[1, 0].plot(metrics_data['epoch'], metrics_data['text_nll'],
                    'g--s', label='Text', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('NLL per Token', fontsize=12)
    axes[1, 0].set_title('Negative Log Likelihood', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Compression Ratio
    axes[1, 1].plot(metrics_data['epoch'], metrics_data['compression_ratio'],
                    'orange', marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Compression Ratio', fontsize=12)
    axes[1, 1].set_title('Compression Efficiency', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=4.0, color='gray', linestyle='--', alpha=0.5,
                       label='Target (4×)')
    axes[1, 1].legend()

    # 5. Relative Improvement
    if len(metrics_data['latent_f1']) > 0 and metrics_data['latent_f1'][0] > 0:
        baseline = metrics_data['latent_f1'][0]
        relative_improvement = [(f / baseline - 1) * 100 for f in metrics_data['latent_f1']]
        axes[2, 0].plot(metrics_data['epoch'], relative_improvement,
                        'teal', marker='o', linewidth=2, markersize=8)
        axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2, 0].set_xlabel('Epoch', fontsize=12)
        axes[2, 0].set_ylabel('Relative Improvement (%)', fontsize=12)
        axes[2, 0].set_title('F1 Improvement from Epoch 1', fontsize=14, fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)

    # 6. Performance Gap Analysis
    latent_gap = [t - l for t, l in zip(metrics_data['text_f1'], metrics_data['latent_f1'])]
    budget_gap = [t - b for t, b in zip(metrics_data['text_f1'], metrics_data['token_budget_f1'])]

    axes[2, 1].plot(metrics_data['epoch'], latent_gap,
                    'b-o', label='Text - Latent', linewidth=2, markersize=8)
    axes[2, 1].plot(metrics_data['epoch'], budget_gap,
                    'r-.^', label='Text - Token Budget', linewidth=2, markersize=6)
    axes[2, 1].set_xlabel('Epoch', fontsize=12)
    axes[2, 1].set_ylabel('Performance Gap', fontsize=12)
    axes[2, 1].set_title('Performance Gap to Text Baseline', fontsize=14, fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.suptitle('8-Epoch Training Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.savefig(fig_dir / "training_analysis.png", dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / "training_analysis.pdf", bbox_inches='tight')
    plt.close()

    print(f"Training analysis plots saved to {fig_dir}")

    return metrics_data


def generate_latex_tables(metrics_data, output_dir):
    """Generate LaTeX tables for the paper."""

    # Main results table
    latex_lines = []
    latex_lines.append("% Main Results Table")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Performance Progression over 8 Epochs}")
    latex_lines.append("\\label{tab:main_results}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Epoch & Latent F1 & Text F1 & Token-Budget F1 & First-Tok Acc & Compression & NLL/tok \\\\")
    latex_lines.append("\\midrule")

    for i, epoch in enumerate(metrics_data['epoch']):
        latex_lines.append(
            f"{epoch} & "
            f"{metrics_data['latent_f1'][i]:.3f} & "
            f"{metrics_data['text_f1'][i]:.3f} & "
            f"{metrics_data['token_budget_f1'][i]:.3f} & "
            f"{metrics_data['first_tok_acc'][i]:.3f} & "
            f"{metrics_data['compression_ratio'][i]:.1f}× & "
            f"{metrics_data['latent_nll'][i]:.2f} \\\\"
        )

    # Add summary statistics
    latex_lines.append("\\midrule")
    latex_lines.append(
        f"Mean & "
        f"{np.mean(metrics_data['latent_f1']):.3f} & "
        f"{np.mean(metrics_data['text_f1']):.3f} & "
        f"{np.mean(metrics_data['token_budget_f1']):.3f} & "
        f"{np.mean(metrics_data['first_tok_acc']):.3f} & "
        f"{np.mean(metrics_data['compression_ratio']):.1f}× & "
        f"{np.mean(metrics_data['latent_nll']):.2f} \\\\"
    )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Save table
    table_file = output_dir / "main_results_table.tex"
    with open(table_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"LaTeX table saved to {table_file}")

    # Convergence analysis table
    if len(metrics_data['latent_f1']) >= 3:
        convergence_lines = []
        convergence_lines.append("% Convergence Analysis Table")
        convergence_lines.append("\\begin{table}[h]")
        convergence_lines.append("\\centering")
        convergence_lines.append("\\caption{Convergence Analysis}")
        convergence_lines.append("\\label{tab:convergence}")
        convergence_lines.append("\\begin{tabular}{lc}")
        convergence_lines.append("\\toprule")
        convergence_lines.append("Metric & Value \\\\")
        convergence_lines.append("\\midrule")

        # Calculate convergence metrics
        last_3_f1 = metrics_data['latent_f1'][-3:]
        f1_std = np.std(last_3_f1)
        f1_mean = np.mean(last_3_f1)

        # Best epoch
        best_epoch = metrics_data['epoch'][np.argmax(metrics_data['latent_f1'])]
        best_f1 = max(metrics_data['latent_f1'])

        # Improvement
        if metrics_data['latent_f1'][0] > 0:
            total_improvement = (metrics_data['latent_f1'][-1] / metrics_data['latent_f1'][0] - 1) * 100
        else:
            total_improvement = 0

        convergence_lines.append(f"Best Epoch & {best_epoch} \\\\")
        convergence_lines.append(f"Best F1 Score & {best_f1:.4f} \\\\")
        convergence_lines.append(f"Final F1 Score & {metrics_data['latent_f1'][-1]:.4f} \\\\")
        convergence_lines.append(f"Last 3 Epochs Mean & {f1_mean:.4f} \\\\")
        convergence_lines.append(f"Last 3 Epochs Std & {f1_std:.4f} \\\\")
        convergence_lines.append(f"Total Improvement & {total_improvement:.1f}\\% \\\\")
        convergence_lines.append(f"Converged & {'Yes' if f1_std < 0.01 else 'No'} \\\\")

        convergence_lines.append("\\bottomrule")
        convergence_lines.append("\\end{tabular}")
        convergence_lines.append("\\end{table}")

        conv_file = output_dir / "convergence_table.tex"
        with open(conv_file, 'w') as f:
            f.write('\n'.join(convergence_lines))

        print(f"Convergence table saved to {conv_file}")


def generate_summary_json(metrics_data, results, output_dir):
    """Generate comprehensive summary JSON."""

    summary = {
        'experiment': '8-epoch training with per-epoch evaluation',
        'total_epochs': len(metrics_data['epoch']),
        'metrics': {
            'final': {
                'latent_f1': metrics_data['latent_f1'][-1] if metrics_data['latent_f1'] else 0,
                'text_f1': metrics_data['text_f1'][-1] if metrics_data['text_f1'] else 0,
                'token_budget_f1': metrics_data['token_budget_f1'][-1] if metrics_data['token_budget_f1'] else 0,
                'first_tok_acc': metrics_data['first_tok_acc'][-1] if metrics_data['first_tok_acc'] else 0,
                'compression_ratio': metrics_data['compression_ratio'][-1] if metrics_data['compression_ratio'] else 0,
                'latent_nll': metrics_data['latent_nll'][-1] if metrics_data['latent_nll'] else 0,
            },
            'best': {
                'latent_f1': max(metrics_data['latent_f1']) if metrics_data['latent_f1'] else 0,
                'latent_f1_epoch': metrics_data['epoch'][np.argmax(metrics_data['latent_f1'])] if metrics_data['latent_f1'] else 0,
                'first_tok_acc': max(metrics_data['first_tok_acc']) if metrics_data['first_tok_acc'] else 0,
                'first_tok_acc_epoch': metrics_data['epoch'][np.argmax(metrics_data['first_tok_acc'])] if metrics_data['first_tok_acc'] else 0,
            },
            'progression': metrics_data
        }
    }

    # Add convergence analysis
    if len(metrics_data['latent_f1']) >= 3:
        last_3 = metrics_data['latent_f1'][-3:]
        summary['convergence'] = {
            'last_3_mean': float(np.mean(last_3)),
            'last_3_std': float(np.std(last_3)),
            'converged': bool(np.std(last_3) < 0.01),
            'epochs_to_convergence': None  # Could be calculated with more sophisticated analysis
        }

    # Add statistical tests if available
    if len(metrics_data['latent_f1']) > 1:
        # Simple linear regression for trend
        epochs_arr = np.array(metrics_data['epoch'])
        f1_arr = np.array(metrics_data['latent_f1'])
        slope, intercept = np.polyfit(epochs_arr, f1_arr, 1)

        summary['trend'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'improving': bool(slope > 0),
            'improvement_per_epoch': float(slope)
        }

    # Save summary
    summary_file = output_dir / "comprehensive_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Comprehensive summary saved to {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze 8-epoch training results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to training output directory')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='Optional: Path to another run for comparison')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        return

    print(f"Analyzing results from: {output_dir}")
    print("=" * 60)

    # Load results
    results = load_epoch_results(output_dir)
    if not results:
        print("No epoch evaluation results found!")
        return

    print(f"Found results for {len(results)} epochs")

    # Create visualizations
    metrics_data = create_training_curves(results, output_dir)

    # Generate tables
    generate_latex_tables(metrics_data, output_dir)

    # Generate summary
    summary = generate_summary_json(metrics_data, results, output_dir)

    # Print summary to console
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Final latent F1: {summary['metrics']['final']['latent_f1']:.4f}")
    print(f"Final text F1: {summary['metrics']['final']['text_f1']:.4f}")
    print(f"Best latent F1: {summary['metrics']['best']['latent_f1']:.4f} (epoch {summary['metrics']['best']['latent_f1_epoch']})")
    print(f"Final first-token acc: {summary['metrics']['final']['first_tok_acc']:.4f}")
    print(f"Compression ratio: {summary['metrics']['final']['compression_ratio']:.1f}×")

    if 'convergence' in summary:
        print(f"\nConvergence: {'Yes' if summary['convergence']['converged'] else 'No'}")
        print(f"Last 3 epochs std: {summary['convergence']['last_3_std']:.4f}")

    if 'trend' in summary:
        print(f"\nTrend: {'Improving' if summary['trend']['improving'] else 'Not improving'}")
        print(f"Improvement per epoch: {summary['trend']['improvement_per_epoch']:.4f}")

    print("=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()