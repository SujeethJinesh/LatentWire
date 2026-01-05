#!/usr/bin/env python3
"""
Test script for visualization pipeline components.
Generates sample data and creates plots to verify visualization works.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

def generate_sample_results(output_dir: Path):
    """Generate sample results data for testing visualization."""

    # Sample training metrics over time
    epochs = 10
    steps_per_epoch = 50

    training_metrics = {
        'epochs': list(range(epochs)),
        'steps': list(range(epochs * steps_per_epoch)),
        'loss': [3.5 * np.exp(-0.3 * i) + 0.5 + 0.1 * np.random.randn() for i in range(epochs)],
        'nll': [2.5 * np.exp(-0.25 * i) + 0.3 + 0.08 * np.random.randn() for i in range(epochs)],
        'first_token_acc': [0.05 + 0.08 * i + 0.02 * np.random.randn() for i in range(epochs)],
        'learning_rate': [1e-3 * (0.95 ** i) for i in range(epochs)]
    }

    # Sample evaluation results
    eval_results = {
        'llama': {
            'text_baseline': {
                'f1': 0.823 + 0.02 * np.random.randn(),
                'em': 0.756 + 0.02 * np.random.randn(),
                'nll': 1.234,
                'first_token_acc': 0.912
            },
            'latent': {
                'f1': 0.156 + 0.05 * np.random.randn(),
                'em': 0.089 + 0.03 * np.random.randn(),
                'nll': 3.456,
                'first_token_acc': 0.234
            },
            'token_budget': {
                'f1': 0.234 + 0.04 * np.random.randn(),
                'em': 0.156 + 0.04 * np.random.randn(),
                'nll': 2.890,
                'first_token_acc': 0.345
            }
        }
    }

    # Compression results
    compression_results = {
        'methods': ['Text', 'Latent-fp16', 'Latent-int8', 'Latent-int4'],
        'bytes': [512, 128, 64, 32],
        'quality': [0.85, 0.65, 0.55, 0.35]
    }

    # Save sample data
    (output_dir / 'training_metrics.json').write_text(json.dumps(training_metrics, indent=2))
    (output_dir / 'eval_results.json').write_text(json.dumps(eval_results, indent=2))
    (output_dir / 'compression_results.json').write_text(json.dumps(compression_results, indent=2))

    return training_metrics, eval_results, compression_results


def create_training_plots(metrics, output_dir: Path):
    """Create training progress plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # Loss plot
    ax = axes[0, 0]
    ax.plot(metrics['epochs'], metrics['loss'], 'b-', linewidth=2, label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # NLL plot
    ax = axes[0, 1]
    ax.plot(metrics['epochs'], metrics['nll'], 'r-', linewidth=2, label='NLL')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL')
    ax.set_title('Negative Log-Likelihood')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # First token accuracy
    ax = axes[1, 0]
    ax.plot(metrics['epochs'], metrics['first_token_acc'], 'g-', linewidth=2, label='FirstTok@1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('First Token Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.legend()

    # Learning rate
    ax = axes[1, 1]
    ax.semilogy(metrics['epochs'], metrics['learning_rate'], 'orange', linewidth=2, label='LR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / 'training_progress.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created training plot: {plot_path}")
    return plot_path


def create_evaluation_plots(results, output_dir: Path):
    """Create evaluation comparison plots."""

    # Extract data for plotting
    methods = ['Text Baseline', 'Latent', 'Token Budget']
    model_results = results.get('llama', {})

    f1_scores = [
        model_results.get('text_baseline', {}).get('f1', 0),
        model_results.get('latent', {}).get('f1', 0),
        model_results.get('token_budget', {}).get('f1', 0)
    ]

    em_scores = [
        model_results.get('text_baseline', {}).get('em', 0),
        model_results.get('latent', {}).get('em', 0),
        model_results.get('token_budget', {}).get('em', 0)
    ]

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Evaluation Results Comparison', fontsize=16, fontweight='bold')

    # F1 scores
    ax = axes[0]
    bars = ax.bar(methods, f1_scores, color=['green', 'blue', 'orange'])
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Method')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')

    # EM scores
    ax = axes[1]
    bars = ax.bar(methods, em_scores, color=['green', 'blue', 'orange'])
    ax.set_ylabel('Exact Match Score')
    ax.set_title('Exact Match by Method')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, em_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = output_dir / 'evaluation_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created evaluation plot: {plot_path}")
    return plot_path


def create_compression_plots(compression, output_dir: Path):
    """Create compression analysis plots."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Compression Analysis', fontsize=16, fontweight='bold')

    methods = compression['methods']
    bytes_data = compression['bytes']
    quality = compression['quality']

    # Compression ratio plot
    ax = axes[0]
    compression_ratios = [bytes_data[0] / b for b in bytes_data]
    bars = ax.bar(methods, compression_ratios, color=['gray', 'blue', 'green', 'red'])
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio by Method')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}x', ha='center', va='bottom')

    # Quality vs Compression tradeoff
    ax = axes[1]
    ax.scatter(bytes_data, quality, s=100, c=['gray', 'blue', 'green', 'red'])
    for i, method in enumerate(methods):
        ax.annotate(method, (bytes_data[i], quality[i]),
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Wire Bytes')
    ax.set_ylabel('Quality (F1 Score)')
    ax.set_title('Quality vs Compression Tradeoff')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plot_path = output_dir / 'compression_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created compression plot: {plot_path}")
    return plot_path


def create_summary_dashboard(output_dir: Path):
    """Create a comprehensive summary dashboard."""

    fig = plt.figure(figsize=(16, 10))

    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('LatentWire Integration Test Results Dashboard', fontsize=18, fontweight='bold')

    # Load all data
    training = json.loads((output_dir / 'training_metrics.json').read_text())
    eval_data = json.loads((output_dir / 'eval_results.json').read_text())
    compression = json.loads((output_dir / 'compression_results.json').read_text())

    # 1. Training loss curve (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(training['epochs'], training['loss'], 'b-', linewidth=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # 2. Key metrics table (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    llama_results = eval_data.get('llama', {})
    text_baseline = llama_results.get('text_baseline', {})
    latent_results = llama_results.get('latent', {})

    metrics_text = f"""Key Metrics:

Text Baseline:
  F1: {text_baseline.get('f1', 0):.3f}
  EM: {text_baseline.get('em', 0):.3f}

Latent:
  F1: {latent_results.get('f1', 0):.3f}
  EM: {latent_results.get('em', 0):.3f}

Compression: {compression['bytes'][0]/compression['bytes'][1]:.1f}x"""

    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')

    # 3. F1 comparison bar chart (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    methods = ['Text', 'Latent', 'Token']
    f1_scores = [
        text_baseline.get('f1', 0),
        latent_results.get('f1', 0),
        llama_results.get('token_budget', {}).get('f1', 0)
    ]
    bars = ax3.bar(methods, f1_scores, color=['green', 'blue', 'orange'])
    ax3.set_title('F1 Score Comparison', fontweight='bold')
    ax3.set_ylabel('F1 Score')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. First token accuracy over time (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(training['epochs'], training['first_token_acc'], 'g-', linewidth=2)
    ax4.set_title('First Token Accuracy', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim([0, max(training['first_token_acc']) * 1.2])
    ax4.grid(True, alpha=0.3)

    # 5. Compression tradeoff (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(compression['bytes'], compression['quality'],
               s=100, c=['gray', 'blue', 'green', 'red'])
    ax5.set_title('Quality vs Compression', fontweight='bold')
    ax5.set_xlabel('Wire Bytes')
    ax5.set_ylabel('Quality')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)

    # 6. Learning curves comparison (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    epochs = training['epochs']
    ax6.plot(epochs, training['loss'], 'b-', label='Loss', linewidth=2)
    ax6.plot(epochs, training['nll'], 'r-', label='NLL', linewidth=2)
    ax6.set_title('Training Curves', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Value')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()
    dashboard_path = output_dir / 'summary_dashboard.png'
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Created summary dashboard: {dashboard_path}")
    return dashboard_path


def main():
    """Main test function for visualization pipeline."""

    # Parse arguments
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('runs/visualization_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Testing Visualization Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    try:
        # Step 1: Generate sample data
        print("Step 1: Generating sample data...")
        training_metrics, eval_results, compression_results = generate_sample_results(output_dir)
        print("✓ Sample data generated")
        print()

        # Step 2: Create training plots
        print("Step 2: Creating training plots...")
        training_plot = create_training_plots(training_metrics, output_dir)
        print()

        # Step 3: Create evaluation plots
        print("Step 3: Creating evaluation plots...")
        eval_plot = create_evaluation_plots(eval_results, output_dir)
        print()

        # Step 4: Create compression plots
        print("Step 4: Creating compression plots...")
        compression_plot = create_compression_plots(compression_results, output_dir)
        print()

        # Step 5: Create summary dashboard
        print("Step 5: Creating summary dashboard...")
        dashboard = create_summary_dashboard(output_dir)
        print()

        # Summary
        print("=" * 60)
        print("Visualization Pipeline Test Complete!")
        print("=" * 60)
        print(f"All plots saved to: {output_dir}")
        print()
        print("Generated files:")
        for file in sorted(output_dir.glob('*.png')):
            print(f"  - {file.name}")
        for file in sorted(output_dir.glob('*.json')):
            print(f"  - {file.name}")

        return 0

    except Exception as e:
        print(f"[ERROR] Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())