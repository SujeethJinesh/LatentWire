#!/usr/bin/env python3
"""
Quick analysis script for embedding experiment sweep results.
Generates human-readable summary and insights.
"""

import argparse
import json
from pathlib import Path


def analyze_results(summary_path):
    """Analyze sweep results and generate insights."""

    with open(summary_path) as f:
        data = json.load(f)

    results = [r for r in data['results'] if 'error' not in r]

    print("="*80)
    print("EMBEDDING EXPERIMENT SWEEP - ANALYSIS")
    print("="*80)
    print()

    print(f"Total experiments: {data['total_experiments']}")
    print(f"Successful: {data['successful']}")
    print(f"Failed: {data['failed']}")
    print()

    # Sort by F1
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)

    # Top performers
    print("="*80)
    print("TOP 5 PERFORMERS (by F1)")
    print("="*80)
    print()
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i}. {r['experiment']}")
        print(f"   F1: {r['f1']:.4f}  |  EM: {r['em']:.4f}  |  Empty: {r['empty_rate']:.2%}")
        print(f"   Config: {r['config']}")
        print()

    # Best by category
    print("="*80)
    print("BEST IN EACH CATEGORY")
    print("="*80)
    print()

    categories = {
        'baseline': 'Baseline',
        'rms': 'RMS Matching',
        'batch_dist': 'Batch Distribution',
        'nearest_k': 'K-Nearest Projection',
        'anchor': 'Anchor+Offset',
        'codebook': 'Soft Codebook',
    }

    for prefix, cat_name in categories.items():
        cat_results = [r for r in results if r['experiment'].startswith(prefix)]
        if not cat_results:
            continue

        best = max(cat_results, key=lambda x: x['f1'])
        print(f"{cat_name}:")
        print(f"  Best: {best['experiment']}")
        print(f"  F1: {best['f1']:.4f}  |  EM: {best['em']:.4f}  |  Empty: {best['empty_rate']:.2%}")
        print(f"  Config: {best['config']}")
        print()

    # Hyperparameter insights
    print("="*80)
    print("HYPERPARAMETER INSIGHTS")
    print("="*80)
    print()

    # K-nearest: analyze k and alpha
    knearest_results = [r for r in results if r['experiment'].startswith('nearest_k')]
    if knearest_results:
        print("K-Nearest Projection:")
        k_values = {}
        alpha_values = {}

        for r in knearest_results:
            k = r['config']['k']
            alpha = r['config']['alpha']
            f1 = r['f1']

            if k not in k_values:
                k_values[k] = []
            k_values[k].append(f1)

            if alpha not in alpha_values:
                alpha_values[alpha] = []
            alpha_values[alpha].append(f1)

        print(f"  Best k: {max(k_values.items(), key=lambda x: sum(x[1])/len(x[1]))[0]} (avg F1: {max(sum(v)/len(v) for v in k_values.values()):.4f})")
        print(f"  Best α: {max(alpha_values.items(), key=lambda x: sum(x[1])/len(x[1]))[0]} (avg F1: {max(sum(v)/len(v) for v in alpha_values.values()):.4f})")
        print()

    # Anchor+offset: analyze epsilon
    anchor_results = [r for r in results if r['experiment'].startswith('anchor_eps')]
    if anchor_results:
        print("Anchor+Offset:")
        eps_values = {}

        for r in anchor_results:
            eps = r['config']['epsilon']
            f1 = r['f1']

            if eps not in eps_values:
                eps_values[eps] = []
            eps_values[eps].append(f1)

        best_eps = max(eps_values.items(), key=lambda x: sum(x[1])/len(x[1]))
        print(f"  Best ε: {best_eps[0]} (F1: {best_eps[1][0]:.4f})")
        print(f"  Trend: ", end="")
        sorted_eps = sorted(eps_values.items())
        for eps, f1s in sorted_eps:
            print(f"ε={eps:.2f}→{sum(f1s)/len(f1s):.3f}  ", end="")
        print()
        print()

    # RMS matching: analyze scale
    rms_results = [r for r in results if r['experiment'].startswith('rms_scale')]
    if rms_results:
        print("RMS Matching:")
        for r in sorted(rms_results, key=lambda x: x['config']['target_scale']):
            scale = r['config']['target_scale']
            print(f"  scale={scale}: F1={r['f1']:.4f}, EM={r['em']:.4f}, Empty={r['empty_rate']:.2%}")
        print()

    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()

    baseline = next((r for r in results if r['experiment'] == 'baseline'), None)
    if baseline:
        print(f"✓ Baseline F1: {baseline['f1']:.4f}")
        print(f"  → This is the upper bound (text embeddings work perfectly)")
        print()

        # Find improvements
        improvements = [r for r in results if r['f1'] > baseline['f1'] * 0.8 and r['experiment'] != 'baseline']
        if improvements:
            print(f"✓ {len(improvements)} experiments achieved >80% of baseline:")
            for r in sorted(improvements, key=lambda x: x['f1'], reverse=True)[:3]:
                ratio = r['f1'] / baseline['f1'] * 100
                print(f"  - {r['experiment']}: {r['f1']:.4f} ({ratio:.1f}% of baseline)")
            print()

        # Check for disasters
        disasters = [r for r in results if r['empty_rate'] > 0.5]
        if disasters:
            print(f"⚠ {len(disasters)} experiments had >50% empty generations:")
            for r in disasters[:3]:
                print(f"  - {r['experiment']}: {r['empty_rate']:.1%} empty")
            print()

    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    top3 = sorted_results[:3]
    print("Based on these results, integrate the following into training:")
    print()
    for i, r in enumerate(top3, 1):
        print(f"{i}. {r['experiment']}")
        print(f"   → Config: {r['config']}")
        print(f"   → Expected improvement: F1={r['f1']:.4f} (vs baseline {baseline['f1']:.4f})")
        print()

    print("Next steps:")
    print("  1. Integrate winner into latentwire/train.py")
    print("  2. Add hyperparameters to train.py argparse")
    print("  3. Run full training: 10k samples, 10 epochs")
    print("  4. Monitor convergence and tune")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=str, default='runs/embed_sweep_simple/summary.json')
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: Summary file not found: {summary_path}")
        print("Run the sweep first: bash scripts/run_embed_sweep_simple.sh")
        exit(1)

    analyze_results(summary_path)
