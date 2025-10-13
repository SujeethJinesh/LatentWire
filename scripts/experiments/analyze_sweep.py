"""
Analyze Hyperparameter Sweep Results

Aggregates results from all sweep runs and identifies best configurations.

Usage:
  python scripts/experiments/analyze_sweep.py --sweep_dir runs/sweeps/hyperparam
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def extract_metrics_from_log(log_path):
    """Extract key metrics from training log."""
    metrics = {
        'first_acc_best': 0.0,
        'final_loss': None,
        'completed': False,
    }

    if not log_path.exists():
        return metrics

    with open(log_path) as f:
        for line in f:
            # Look for best first token accuracy
            if 'NEW PEAK: first_acc_ema=' in line:
                try:
                    parts = line.split('first_acc_ema=')[1].split('%')[0]
                    acc = float(parts)
                    metrics['first_acc_best'] = max(metrics['first_acc_best'], acc)
                except:
                    pass

            # Look for final step loss
            if 'step ' in line and '/' in line:
                try:
                    if 'llama(L): tf=' in line:
                        loss_str = line.split('tf=')[1].split()[0]
                        metrics['final_loss'] = float(loss_str)
                except:
                    pass

            # Check if training completed
            if 'Training complete' in line or 'Saved latest checkpoint' in line:
                metrics['completed'] = True

    return metrics


def parse_run_name(run_name):
    """Extract hyperparameters from run name like M32_dz256_ftce0.5_K4."""
    params = {}
    parts = run_name.split('_')

    for part in parts:
        if part.startswith('M'):
            params['latent_len'] = int(part[1:])
        elif part.startswith('dz'):
            params['d_z'] = int(part[2:])
        elif part.startswith('ftce'):
            params['ftce_weight'] = float(part[4:])
        elif part.startswith('K'):
            params['K'] = int(part[1:])

    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', type=str, required=True)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        return

    # Collect results from all runs
    results = []

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        if run_name.startswith('sweep_') or run_name == 'summary':
            continue

        # Parse hyperparameters
        params = parse_run_name(run_name)
        if not params:
            continue

        # Extract metrics from log
        log_path = run_dir / 'train.log'
        metrics = extract_metrics_from_log(log_path)

        # Combine
        result = {
            'run_name': run_name,
            **params,
            **metrics,
        }
        results.append(result)

    if not results:
        print("No results found in sweep directory")
        return

    # Convert to dataframe
    df = pd.DataFrame(results)

    # Sort by best first token accuracy
    df = df.sort_values('first_acc_best', ascending=False)

    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print()

    print(f"Total runs: {len(df)}")
    print(f"Completed: {df['completed'].sum()}")
    print(f"Failed: {(~df['completed']).sum()}")
    print()

    # Top 10 configurations
    print("Top 10 Configurations by First Token Accuracy:")
    print("-" * 80)
    top10 = df.head(10)
    for i, row in enumerate(top10.itertuples(), 1):
        print(f"{i:2d}. {row.run_name:30s} | FirstTok: {row.first_acc_best:5.1f}% | "
              f"M={row.latent_len:2d}, d_z={row.d_z:3d}, ftce={row.ftce_weight:.1f}, K={row.K:d}")

    print()

    # Statistics by hyperparameter
    print("Average First Token Acc by Hyperparameter:")
    print("-" * 80)

    for param in ['latent_len', 'd_z', 'ftce_weight', 'K']:
        if param in df.columns:
            grouped = df.groupby(param)['first_acc_best'].mean().sort_values(ascending=False)
            print(f"\n{param}:")
            for val, acc in grouped.items():
                print(f"  {val}: {acc:.2f}%")

    print()

    # Save summary
    summary = {
        'total_runs': len(df),
        'completed': int(df['completed'].sum()),
        'failed': int((~df['completed']).sum()),
        'best_run': {
            'run_name': df.iloc[0]['run_name'],
            'first_acc_best': float(df.iloc[0]['first_acc_best']),
            'latent_len': int(df.iloc[0]['latent_len']),
            'd_z': int(df.iloc[0]['d_z']),
            'ftce_weight': float(df.iloc[0]['ftce_weight']),
            'K': int(df.iloc[0]['K']),
        },
        'top10': [
            {
                'run_name': row.run_name,
                'first_acc_best': float(row.first_acc_best),
                'latent_len': int(row.latent_len),
                'd_z': int(row.d_z),
                'ftce_weight': float(row.ftce_weight),
                'K': int(row.K),
            }
            for row in top10.itertuples()
        ],
    }

    summary_path = sweep_dir / 'sweep_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")
    print()

    # Save full results CSV
    csv_path = sweep_dir / 'sweep_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to: {csv_path}")
    print()


if __name__ == '__main__':
    main()
