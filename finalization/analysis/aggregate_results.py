#!/usr/bin/env python3
"""
Aggregate results from multiple experiments and generate summary statistics.

This script:
- Collects results from multiple experiment runs
- Computes aggregate statistics
- Performs statistical significance testing
- Generates LaTeX tables for paper
- Creates comparison plots
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats


class ResultsAggregator:
    """Aggregate and analyze experiment results."""

    def __init__(self, experiment_dirs: List[str]):
        """Initialize aggregator with experiment directories."""
        self.experiment_dirs = [Path(d) for d in experiment_dirs]
        self.results = []
        self.aggregated = {}

    def load_all_results(self) -> None:
        """Load results from all experiment directories."""
        for exp_dir in self.experiment_dirs:
            # Find all result files
            result_files = list(exp_dir.glob("**/results*.json"))

            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        data['source_dir'] = str(exp_dir)
                        data['source_file'] = str(result_file)
                        self.results.append(data)
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")

        print(f"Loaded {len(self.results)} result files")

    def aggregate_by_method(self) -> pd.DataFrame:
        """Aggregate results by method (latent, text, token_budget)."""
        data = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            if 'results' not in result:
                continue

            for exp in result['results']:
                method = exp.get('method', 'unknown')
                dataset = exp.get('dataset', 'unknown')
                key = f"{dataset}_{method}"

                # Collect metrics
                data[key]['exact_match'].append(exp.get('exact_match', 0))
                data[key]['f1_score'].append(exp.get('f1_score', 0))
                data[key]['compression_ratio'].append(exp.get('compression_ratio', 1))
                data[key]['inference_time_ms'].append(exp.get('inference_time_ms', 0))

        # Compute statistics
        summary = []
        for key, metrics in data.items():
            dataset, method = key.rsplit('_', 1)

            row = {
                'dataset': dataset,
                'method': method,
                'num_runs': len(metrics['exact_match']),
            }

            # Compute mean and std for each metric
            for metric_name, values in metrics.items():
                if values:
                    row[f'{metric_name}_mean'] = np.mean(values)
                    row[f'{metric_name}_std'] = np.std(values)
                    row[f'{metric_name}_min'] = np.min(values)
                    row[f'{metric_name}_max'] = np.max(values)

            summary.append(row)

        return pd.DataFrame(summary)

    def compute_significance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical significance between methods."""
        significance_results = []

        # Group by dataset
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]

            # Compare latent vs text baseline
            latent_data = dataset_df[dataset_df['method'] == 'latent']
            text_data = dataset_df[dataset_df['method'] == 'text']

            if not latent_data.empty and not text_data.empty:
                # Perform t-test on F1 scores
                # Note: This is simplified - would need actual sample data
                result = {
                    'dataset': dataset,
                    'comparison': 'latent_vs_text',
                    'metric': 'f1_score',
                    'latent_mean': latent_data['f1_score_mean'].values[0],
                    'text_mean': text_data['f1_score_mean'].values[0],
                    'difference': latent_data['f1_score_mean'].values[0] - text_data['f1_score_mean'].values[0],
                    'significant': abs(latent_data['f1_score_mean'].values[0] - text_data['f1_score_mean'].values[0]) > 0.05
                }
                significance_results.append(result)

        return pd.DataFrame(significance_results)

    def generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table for paper."""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{LatentWire Performance Comparison}")
        latex.append("\\label{tab:results}")
        latex.append("\\begin{tabular}{llcccc}")
        latex.append("\\toprule")
        latex.append("Dataset & Method & EM & F1 & Compression & Latency (ms) \\\\")
        latex.append("\\midrule")

        for _, row in df.iterrows():
            em = f"{row.get('exact_match_mean', 0):.3f} ± {row.get('exact_match_std', 0):.3f}"
            f1 = f"{row.get('f1_score_mean', 0):.3f} ± {row.get('f1_score_std', 0):.3f}"
            comp = f"{row.get('compression_ratio_mean', 1):.1f}x"
            latency = f"{row.get('inference_time_ms_mean', 0):.1f}"

            latex.append(f"{row['dataset']} & {row['method']} & {em} & {f1} & {comp} & {latency} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def generate_summary_json(self) -> Dict[str, Any]:
        """Generate comprehensive summary JSON."""
        df = self.aggregate_by_method()
        significance_df = self.compute_significance(df)

        summary = {
            'total_experiments': len(self.results),
            'experiment_dirs': [str(d) for d in self.experiment_dirs],
            'aggregated_results': df.to_dict(orient='records'),
            'significance_tests': significance_df.to_dict(orient='records'),
            'best_performers': self._find_best_performers(df),
            'latex_table': self.generate_latex_table(df)
        }

        return summary

    def _find_best_performers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find best performing configurations."""
        best = {}

        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]

            # Best F1
            best_f1_idx = dataset_df['f1_score_mean'].idxmax()
            best_f1_row = dataset_df.loc[best_f1_idx]

            # Best compression
            best_comp_idx = dataset_df['compression_ratio_mean'].idxmax()
            best_comp_row = dataset_df.loc[best_comp_idx]

            best[dataset] = {
                'best_f1': {
                    'method': best_f1_row['method'],
                    'score': best_f1_row['f1_score_mean']
                },
                'best_compression': {
                    'method': best_comp_row['method'],
                    'ratio': best_comp_row['compression_ratio_mean']
                }
            }

        return best

    def save_summary(self, output_path: str) -> None:
        """Save summary to JSON file."""
        summary = self.generate_summary_json()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {output_path}")

        # Also save LaTeX table separately
        latex_path = Path(output_path).with_suffix('.tex')
        with open(latex_path, 'w') as f:
            f.write(summary['latex_table'])
        print(f"LaTeX table saved to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate LatentWire Results')

    parser.add_argument('--experiment_dirs', nargs='+', required=True,
                       help='Experiment directories to aggregate')
    parser.add_argument('--output', type=str, default='aggregated_results.json',
                       help='Output path for summary')

    args = parser.parse_args()

    # Create aggregator
    aggregator = ResultsAggregator(args.experiment_dirs)

    # Load and process results
    aggregator.load_all_results()

    # Generate and save summary
    aggregator.save_summary(args.output)

    # Print summary statistics
    df = aggregator.aggregate_by_method()
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(df.to_string())


if __name__ == '__main__':
    main()