#!/usr/bin/env python3
"""
Simplified aggregation script without scipy dependency.

Collects results and generates basic statistics for the paper.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def compute_statistics(values):
    """Compute basic statistics without scipy."""
    values = np.array(values)
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'n': len(values)
    }

    # Simple 95% CI approximation (1.96 * SE for normal distribution)
    if len(values) > 1:
        se = stats['std'] / np.sqrt(len(values))
        ci_margin = 1.96 * se
        stats['ci95_lower'] = stats['mean'] - ci_margin
        stats['ci95_upper'] = stats['mean'] + ci_margin
    else:
        stats['ci95_lower'] = stats['mean']
        stats['ci95_upper'] = stats['mean']

    return stats


def main():
    parser = argparse.ArgumentParser(description="Simple result aggregation")
    parser.add_argument('--input_dir', type=str, default='runs/final_experiments')
    parser.add_argument('--output_dir', type=str, default='finalization/results')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SIMPLE RESULT AGGREGATION")
    print("=" * 80)

    # Collect all JSON files
    results_by_experiment = defaultdict(list)
    json_files = list(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract experiment name
            exp_name = data.get('experiment_name', json_file.stem)
            results_by_experiment[exp_name].append(data)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    # Aggregate metrics
    aggregated = {}
    for exp_name, runs in results_by_experiment.items():
        print(f"\nProcessing {exp_name}: {len(runs)} runs")

        # Extract metrics
        metrics_lists = defaultdict(list)
        for run in runs:
            # Handle different formats
            if 'metrics' in run:
                metrics = run['metrics']
            elif 'results' in run:
                metrics = run['results']
            else:
                metrics = run

            # Collect numeric values
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metrics_lists[key].append(value)

        # Compute statistics
        exp_stats = {}
        for metric_name, values in metrics_lists.items():
            if values:
                exp_stats[metric_name] = compute_statistics(values)

        aggregated[exp_name] = exp_stats

        # Print summary
        if 'f1' in exp_stats:
            f1_stats = exp_stats['f1']
            print(f"  F1: {f1_stats['mean']:.4f} ± {f1_stats['std']:.4f}")

    # Save results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'aggregated_results': aggregated,
        'summary': {
            'total_experiments': len(aggregated),
            'best_f1': max((stats.get('f1', {}).get('mean', 0)
                          for stats in aggregated.values()), default=0),
            'best_compression': max((stats.get('compression_ratio', {}).get('mean', 1)
                                   for stats in aggregated.values()), default=1)
        }
    }

    # Save JSON
    output_file = output_dir / "FINAL_RESULTS.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    # Generate LaTeX tables
    latex_tables = []

    # Main results table
    latex_tables.append("% Main Results Table")
    latex_tables.append("\\begin{table}[h]")
    latex_tables.append("\\centering")
    latex_tables.append("\\caption{Main Results}")
    latex_tables.append("\\label{tab:main_results}")
    latex_tables.append("\\begin{tabular}{lccc}")
    latex_tables.append("\\toprule")
    latex_tables.append("Method & F1 & EM & Compression \\\\")
    latex_tables.append("\\midrule")

    for method in ['text_baseline', 'latentwire', 'llmlingua']:
        if method in aggregated:
            stats = aggregated[method]
            f1 = stats.get('f1', {}).get('mean', 0)
            f1_std = stats.get('f1', {}).get('std', 0)
            em = stats.get('exact_match', {}).get('mean', 0)
            comp = stats.get('compression_ratio', {}).get('mean', 1)

            method_display = method.replace('_', ' ').title()
            if method == 'latentwire':
                method_display = '\\textbf{LatentWire}'

            row = f"{method_display} & ${f1:.3f} \\pm {f1_std:.3f}$ & ${em:.3f}$ & ${comp:.1f}\\times$ \\\\"
            latex_tables.append(row)

    latex_tables.append("\\bottomrule")
    latex_tables.append("\\end{tabular}")
    latex_tables.append("\\end{table}")

    # Save LaTeX tables
    latex_file = output_dir / "paper_tables.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_tables))

    print(f"LaTeX tables saved to {latex_file}")

    # Save text summary
    summary_file = output_dir / "statistical_report.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LATENTWIRE RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {final_results['timestamp']}\n\n")

        f.write("AGGREGATED RESULTS\n")
        f.write("-" * 40 + "\n")

        for exp_name, stats in aggregated.items():
            f.write(f"\n{exp_name}:\n")
            for metric, values in stats.items():
                if 'mean' in values:
                    f.write(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary saved to {summary_file}")
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()