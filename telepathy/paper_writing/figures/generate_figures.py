#!/usr/bin/env python3
"""
Generate figures for LatentWire paper.

Usage:
    cd telepathy/paper_writing/figures
    python generate_figures.py

Outputs:
    - results_comparison.pdf: Main results bar chart
    - latency_comparison.pdf: Latency breakdown
    - token_scaling.pdf: Inverse token scaling
    - training_curves.pdf: Training convergence (if data available)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use a clean style suitable for papers
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')  # Fallback
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
COLORS = {
    'bridge': '#2ecc71',       # Green - our method
    'bridge_reverse': '#27ae60', # Darker green
    'llama': '#3498db',        # Blue
    'mistral': '#9b59b6',      # Purple
    'text_relay': '#e74c3c',   # Red
    'prompt_tuning': '#95a5a6', # Gray
    'random': '#bdc3c7',       # Light gray
}


def fig1_results_comparison():
    """
    Figure 2: Main results comparison across datasets.
    Bar chart showing Bridge vs baselines.
    """
    datasets = ['SST-2', 'AG News', 'TREC', 'Banking77']

    # Data from paper (accuracy %)
    bridge = [96.7, 90.7, 95.3, 21.5]
    llama_direct = [92.0, 79.0, 53.5, 22.0]
    mistral_direct = [88.5, 79.0, 43.0, 19.5]
    text_relay = [71.0, 64.5, 58.0, 1.0]
    prompt_tuning = [49.5, 19.8, 19.0, None]  # No Banking77 data
    random_chance = [50.0, 25.0, 16.7, 1.3]

    x = np.arange(len(datasets))
    width = 0.14

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Bars
    bars1 = ax.bar(x - 2.5*width, bridge, width, label='Bridge (Ours)',
                   color=COLORS['bridge'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 1.5*width, llama_direct, width, label='Llama Direct',
                   color=COLORS['llama'], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x - 0.5*width, mistral_direct, width, label='Mistral Direct',
                   color=COLORS['mistral'], edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 0.5*width, text_relay, width, label='Text-Relay',
                   color=COLORS['text_relay'], edgecolor='black', linewidth=0.5)

    # Prompt tuning (handle None for Banking77)
    prompt_tuning_plot = [p if p is not None else 0 for p in prompt_tuning]
    bars5 = ax.bar(x + 1.5*width, prompt_tuning_plot, width, label='Prompt-Tuning',
                   color=COLORS['prompt_tuning'], edgecolor='black', linewidth=0.5)

    # Random chance line
    for i, (dataset, chance) in enumerate(zip(datasets, random_chance)):
        ax.hlines(chance, i - 3*width, i + 2.5*width, colors='gray',
                  linestyles='dashed', linewidth=1, alpha=0.7)

    # Labels and formatting
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.9)

    # Add value labels on Bridge bars
    for bar, val in zip(bars1, bridge):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    # Add "Random" annotation
    ax.annotate('Random chance', xy=(0.5, 52), fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('results_comparison.pdf')
    plt.savefig('results_comparison.png')
    print("Saved: results_comparison.pdf")
    plt.close()


def fig2_latency_comparison():
    """
    Figure 3: Latency comparison showing speedup.
    Stacked bar showing where time is spent.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: Total latency comparison (Bridge vs Text-Relay only - fair comparison)
    methods = ['Text-Relay', 'Bridge (Ours)']
    latencies = [795.2, 170.6]  # Updated to match Table 2
    colors = [COLORS['text_relay'], COLORS['bridge']]

    bars = ax1.barh(methods, latencies, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Latency (ms)')
    ax1.set_title('Total Inference Latency')
    ax1.set_xlim(0, 900)

    # Add speedup annotations
    ax1.annotate('1.0×', xy=(795.2 + 10, 0), va='center', fontsize=9)
    ax1.annotate('4.66×', xy=(170.6 + 10, 1), va='center', fontsize=9, fontweight='bold')

    # Right: Bridge breakdown (matches Table 2 and Section 3.5)
    # Total Bridge: 170.6ms = 14ms overhead + 156.2ms Mistral
    components = ['Llama\nEncode', 'Bridge\nTransform', 'Mistral\nDecode']
    times = [10.0, 4.4, 156.2]  # Sum = 170.6ms
    percentages = [6, 3, 91]

    bars2 = ax2.bar(components, times, color=['#3498db', '#2ecc71', '#9b59b6'],
                    edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Bridge Latency Breakdown')
    ax2.set_ylim(0, 180)

    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        ax2.annotate(f'{pct}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig('latency_comparison.pdf')
    plt.savefig('latency_comparison.png')
    print("Saved: latency_comparison.pdf")
    plt.close()


def fig3_token_scaling():
    """
    Figure 4: Inverse token scaling on Banking77.
    Shows that fewer tokens = better performance.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Data from Table 3
    tokens = [16, 32, 64, 128]
    accuracy = [21.5, 13.5, 7.5, 1.0]
    random_chance = 1.3  # 1/77

    ax.plot(tokens, accuracy, 'o-', color=COLORS['bridge'], linewidth=2,
            markersize=8, label='Bridge Accuracy')
    ax.axhline(y=random_chance, color='gray', linestyle='--', linewidth=1,
               label=f'Random Chance ({random_chance}%)')

    ax.set_xlabel('Number of Soft Tokens')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Inverse Token Scaling (Banking77)')
    ax.set_xticks(tokens)
    ax.set_xscale('log', base=2)
    ax.set_xlim(12, 160)
    ax.set_ylim(0, 25)
    ax.legend(loc='upper right')

    # Add annotation about the trend
    ax.annotate('Fewer tokens\n= Better', xy=(20, 20), fontsize=9,
                style='italic', color='gray')
    ax.annotate('', xy=(16, 21.5), xytext=(40, 18),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.tight_layout()
    plt.savefig('token_scaling.pdf')
    plt.savefig('token_scaling.png')
    print("Saved: token_scaling.pdf")
    plt.close()


def fig4_bidirectional():
    """
    Figure 5: Bidirectional transfer comparison.
    Simple bar chart comparing L→M vs M→L.
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))

    directions = ['Llama→Mistral', 'Mistral→Llama']
    accuracy = [96.7, 97.2]
    errors = [0.6, 0.6]

    bars = ax.bar(directions, accuracy, yerr=errors, capsize=5,
                  color=[COLORS['bridge'], COLORS['bridge_reverse']],
                  edgecolor='black', linewidth=0.5)

    # Individual model baselines
    ax.axhline(y=92.0, color=COLORS['llama'], linestyle='--', linewidth=1,
               label='Llama Direct (92.0%)')
    ax.axhline(y=88.5, color=COLORS['mistral'], linestyle='--', linewidth=1,
               label='Mistral Direct (88.5%)')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Bidirectional Transfer (SST-2)')
    ax.set_ylim(85, 100)
    ax.legend(loc='lower right', fontsize=8)

    # Add value labels
    for bar, val, err in zip(bars, accuracy, errors):
        ax.annotate(f'{val}±{err}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('bidirectional.pdf')
    plt.savefig('bidirectional.png')
    print("Saved: bidirectional.pdf")
    plt.close()


def fig5_training_curves():
    """
    Figure 6: Training convergence curves (optional).
    Uses data from JSON logs if available.
    """
    # Training data from SST-2 experiments (steps vs accuracy)
    # From the JSON logs
    steps = [400, 800, 1200, 1600, 2000]

    # Seed 42, 123, 456 for Llama→Mistral
    seed42 = [92.0, 98.0, 96.0, 98.0, 98.0]
    seed123 = [94.0, 98.0, 96.0, 100.0, 98.0]
    seed456 = [100.0, 96.0, 98.0, 96.0, 96.0]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(steps, seed42, 'o-', label='Seed 42', alpha=0.7, markersize=5)
    ax.plot(steps, seed123, 's-', label='Seed 123', alpha=0.7, markersize=5)
    ax.plot(steps, seed456, '^-', label='Seed 456', alpha=0.7, markersize=5)

    # Mean line
    mean_acc = [(s1+s2+s3)/3 for s1, s2, s3 in zip(seed42, seed123, seed456)]
    ax.plot(steps, mean_acc, 'k-', linewidth=2, label='Mean', alpha=0.9)

    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Convergence (SST-2)')
    ax.set_xlim(0, 2200)
    ax.set_ylim(45, 105)
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig('training_curves.pdf')
    plt.savefig('training_curves.png')
    print("Saved: training_curves.pdf")
    plt.close()


def fig6_prompt_tuning_comparison():
    """
    Key figure: Bridge vs Prompt-Tuning showing sender is essential.
    This is the "killer" visualization.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    datasets = ['SST-2', 'AG News', 'TREC']
    bridge = [96.7, 90.7, 95.3]
    prompt_tuning = [49.5, 19.8, 19.0]
    random_chance = [50.0, 25.0, 16.7]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, bridge, width, label='Bridge (with Llama)',
                   color=COLORS['bridge'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, prompt_tuning, width, label='Prompt-Tuning (no Llama)',
                   color=COLORS['prompt_tuning'], edgecolor='black', linewidth=0.5)

    # Random chance markers
    for i, chance in enumerate(random_chance):
        ax.scatter([i - width/2, i + width/2], [chance, chance],
                   marker='_', s=100, color='red', zorder=5, linewidths=2)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')

    # Add delta annotations
    for i, (b, pt) in enumerate(zip(bridge, prompt_tuning)):
        delta = b - pt
        mid = (b + pt) / 2
        ax.annotate(f'+{delta:.1f}pp', xy=(i, mid), ha='center', fontsize=10,
                    fontweight='bold', color='darkgreen')

    # Add "Random Chance" label
    ax.annotate('— Random', xy=(2.3, 20), fontsize=8, color='red')

    ax.set_title('Sender Model is Essential: Bridge vs Prompt-Tuning')

    plt.tight_layout()
    plt.savefig('sender_essential.pdf')
    plt.savefig('sender_essential.png')
    print("Saved: sender_essential.pdf")
    plt.close()


def fig7_cross_vs_same_model():
    """
    Figure 7: Cross-model vs Same-model transfer.
    Key finding: heterogeneity helps!
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Data from experiments
    configs = ['Llama→Llama\n(same)', 'Mistral→Mistral\n(same)', 'Llama→Mistral\n(cross)']
    sst2_acc = [84.5, 95.5, 96.7]

    colors = [COLORS['llama'], COLORS['mistral'], COLORS['bridge']]

    bars = ax.bar(configs, sst2_acc, color=colors, edgecolor='black', linewidth=0.5)

    # Highlight the cross-model bar
    bars[2].set_hatch('//')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cross-Model vs Same-Model Transfer (SST-2)')
    ax.set_ylim(80, 100)

    # Add value labels
    for bar, val in zip(bars, sst2_acc):
        ax.annotate(f'{val}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    # Add delta annotations
    ax.annotate('', xy=(2, 96.7), xytext=(0, 84.5),
                arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
    ax.annotate('+12.2pp', xy=(1.0, 90.5), fontsize=10, fontweight='bold',
                color='darkgreen', ha='center')

    # Add insight annotation
    ax.text(1, 82.5, 'Heterogeneity is a feature,\nnot a bug!',
            ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('cross_vs_same.pdf')
    plt.savefig('cross_vs_same.png')
    print("Saved: cross_vs_same.pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating figures for LatentWire paper...")
    print("=" * 50)

    fig1_results_comparison()
    fig2_latency_comparison()
    fig3_token_scaling()
    fig4_bidirectional()
    fig5_training_curves()
    fig6_prompt_tuning_comparison()
    fig7_cross_vs_same_model()

    print("=" * 50)
    print("All figures generated!")
    print("\nTo compile architecture diagram:")
    print("  dot -Tpdf architecture.dot -o architecture.pdf")
    print("\nFigures ready for LaTeX:")
    print("  - architecture.pdf (compile from .dot)")
    print("  - results_comparison.pdf")
    print("  - latency_comparison.pdf")
    print("  - token_scaling.pdf")
    print("  - bidirectional.pdf")
    print("  - training_curves.pdf")
    print("  - sender_essential.pdf")
