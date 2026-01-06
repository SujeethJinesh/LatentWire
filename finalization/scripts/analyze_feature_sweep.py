#!/usr/bin/env python3
"""
Analyze Feature Sweep Results

Reads diagnostics.jsonl files from all sweep experiments and creates
a comparison table showing which features helped fix mode collapse.

Usage:
    python3 scripts/analyze_feature_sweep.py runs/feature_sweep
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_diagnostics(diagnostics_file: Path) -> Dict:
    """Parse diagnostics.jsonl and extract final metrics."""
    if not diagnostics_file.exists():
        return {'error': 'File not found'}

    try:
        with open(diagnostics_file) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Get all eval metrics
        eval_lines = [l for l in lines if l.get('type') == 'full_eval']

        if not eval_lines:
            return {'error': 'No eval metrics found'}

        # Get final eval
        final_eval = eval_lines[-1]

        # Get best eval (highest F1)
        best_eval = max(eval_lines, key=lambda x: x.get('f1', 0))

        # Get training loss progression
        epoch_lines = [l for l in lines if l.get('type') == 'train_epoch']
        final_loss = epoch_lines[-1].get('avg_loss', 0) if epoch_lines else 0

        return {
            'final_f1': final_eval.get('f1', 0) * 100,
            'final_em': final_eval.get('em', 0) * 100,
            'final_diversity': final_eval.get('diversity', 0) * 100,
            'best_f1': best_eval.get('f1', 0) * 100,
            'best_diversity': best_eval.get('diversity', 0) * 100,
            'final_loss': final_loss,
            'num_epochs': len(epoch_lines),
        }
    except Exception as e:
        return {'error': str(e)}


def parse_config_name(config_name: str) -> Tuple[str, str]:
    """Parse configuration name to extract feature and hyperparameters."""
    if config_name == 'baseline':
        return 'Baseline', 'None'

    if config_name.startswith('contrastive_'):
        parts = config_name.replace('contrastive_', '').split('_')
        return 'Contrastive', f"weight={parts[0][1:]}, temp={parts[1][1:]}"

    if config_name.startswith('k_token_'):
        k = config_name.replace('k_token_k', '')
        return 'K-token CE', f"K={k}"

    if config_name.startswith('reconstruction_'):
        parts = config_name.replace('reconstruction_', '').split('_')
        return 'Reconstruction', f"weight={parts[0][1:]}, layers={parts[1][1:]}"

    if config_name.startswith('kd_'):
        parts = config_name.replace('kd_', '').split('_')
        return 'Knowledge Distillation', f"weight={parts[0][1:]}, tau={parts[1][3:]}"

    return 'Unknown', config_name


def main():
    parser = argparse.ArgumentParser(description="Analyze feature sweep results")
    parser.add_argument('sweep_dir', type=str, help='Directory containing sweep results')
    parser.add_argument('--sort_by', type=str, default='final_f1',
                       choices=['final_f1', 'best_f1', 'final_diversity'],
                       help='Metric to sort by')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    # Find all config directories
    config_dirs = [d for d in sweep_dir.iterdir() if d.is_dir()]

    if not config_dirs:
        print(f"Error: No configuration directories found in {sweep_dir}")
        sys.exit(1)

    print(f"\nAnalyzing {len(config_dirs)} configurations in {sweep_dir}")
    print("="*120)

    # Parse results
    results = []
    for config_dir in config_dirs:
        diagnostics_file = config_dir / 'diagnostics.jsonl'
        config_name = config_dir.name

        metrics = parse_diagnostics(diagnostics_file)
        feature, hyperparams = parse_config_name(config_name)

        results.append({
            'config': config_name,
            'feature': feature,
            'hyperparams': hyperparams,
            **metrics
        })

    # Sort results
    results.sort(key=lambda x: x.get(args.sort_by, 0), reverse=True)

    # Print table header
    print(f"\n{'Rank':<6} {'Feature':<24} {'Hyperparams':<28} {'Final F1':<10} {'Best F1':<10} {'Final Div':<12} {'Final EM':<10} {'Loss':<8}")
    print("="*120)

    # Print results
    for rank, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{rank:<6} {result['feature']:<24} {result['hyperparams']:<28} ERROR: {result['error']}")
        else:
            print(f"{rank:<6} {result['feature']:<24} {result['hyperparams']:<28} "
                  f"{result['final_f1']:>8.2f}%  {result['best_f1']:>8.2f}%  "
                  f"{result['final_diversity']:>10.0f}%  {result['final_em']:>8.2f}%  "
                  f"{result['final_loss']:>6.3f}")

    # Print statistics by feature
    print("\n" + "="*120)
    print("SUMMARY BY FEATURE")
    print("="*120)

    features = {}
    for result in results:
        if 'error' not in result:
            feature = result['feature']
            if feature not in features:
                features[feature] = []
            features[feature].append(result)

    for feature, configs in sorted(features.items()):
        avg_f1 = sum(c['final_f1'] for c in configs) / len(configs)
        avg_diversity = sum(c['final_diversity'] for c in configs) / len(configs)
        best_config = max(configs, key=lambda x: x['final_f1'])

        print(f"\n{feature}:")
        print(f"  Configs tested: {len(configs)}")
        print(f"  Average F1: {avg_f1:.2f}%")
        print(f"  Average Diversity: {avg_diversity:.0f}%")
        print(f"  Best config: {best_config['hyperparams']}")
        print(f"    → F1={best_config['final_f1']:.2f}%, Diversity={best_config['final_diversity']:.0f}%")

    # Compare to baseline
    baseline = next((r for r in results if r['feature'] == 'Baseline'), None)
    if baseline and 'error' not in baseline:
        print("\n" + "="*120)
        print("IMPROVEMENT OVER BASELINE")
        print("="*120)
        print(f"\nBaseline: F1={baseline['final_f1']:.2f}%, Diversity={baseline['final_diversity']:.0f}%\n")

        for feature, configs in sorted(features.items()):
            if feature == 'Baseline':
                continue

            best_config = max(configs, key=lambda x: x['final_f1'])
            f1_improvement = best_config['final_f1'] - baseline['final_f1']
            div_improvement = best_config['final_diversity'] - baseline['final_diversity']

            status = "✅ IMPROVED" if f1_improvement > 5.0 or div_improvement > 20 else "⚠️ MARGINAL" if f1_improvement > 0 else "❌ WORSE"

            print(f"{feature}:")
            print(f"  Best F1: {best_config['final_f1']:.2f}% ({f1_improvement:+.2f}%)")
            print(f"  Best Diversity: {best_config['final_diversity']:.0f}% ({div_improvement:+.0f}%)")
            print(f"  Status: {status}")
            print()

    print("="*120)
    print("\nAnalysis complete!")
    print(f"To view detailed logs, check: {sweep_dir}/<config>/train_*.log")
    print()


if __name__ == '__main__':
    main()
