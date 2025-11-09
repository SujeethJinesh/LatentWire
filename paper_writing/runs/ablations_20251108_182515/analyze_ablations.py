#!/usr/bin/env python3
"""
Analyze ablation results for paper
Generates plots and tables for:
1. Stability: with vs without fixes
2. Sequence length: 32 vs 48 vs 64 tokens
3. Dataset: GSM8K vs HotpotQA
"""

import os
import re
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file):
    """Extract evaluation metrics from training log"""
    results = {
        'evals': [],
        'peak_bridged': 0,
        'peak_step': 0,
        'final_bridged': 0,
        'final_target': 0
    }

    with open(log_file) as f:
        for line in f:
            # Extract eval lines
            if '[Eval] Step' in line and 'Bridged acc:' in line:
                parts = line.split('|')
                step = int(re.search(r'Step (\d+)', parts[0]).group(1))
                target = float(re.search(r'Target-alone acc: ([0-9.]+)', parts[1]).group(1))
                bridged = float(re.search(r'Bridged acc: ([0-9.]+)', parts[2]).group(1))

                results['evals'].append({
                    'step': step,
                    'target': target,
                    'bridged': bridged
                })

                if bridged > results['peak_bridged']:
                    results['peak_bridged'] = bridged
                    results['peak_step'] = step

            # Extract final results
            if '[Final Eval]' in line and 'Bridged acc:' in line:
                results['final_target'] = float(re.search(r'Target-alone acc: ([0-9.]+)', line).group(1))
                results['final_bridged'] = float(re.search(r'Bridged acc: ([0-9.]+)', line).group(1))

    return results

def main():
    script_dir = Path(__file__).parent
    results = {}

    # Parse all experiment logs
    for exp_dir in glob.glob(str(script_dir / '*/')):
        exp_name = Path(exp_dir).name
        log_file = Path(exp_dir) / 'train.log'

        if log_file.exists() and exp_name != script_dir.name:
            print(f"Parsing {exp_name}...")
            results[exp_name] = parse_log(log_file)

    # Add baseline from existing experiment (for comparison)
    results['1b_baseline_64tok'] = {
        'evals': [
            {'step': 250, 'target': 0.730, 'bridged': 0.290},
            {'step': 500, 'target': 0.730, 'bridged': 0.655},
            {'step': 750, 'target': 0.730, 'bridged': 0.535},
            {'step': 1000, 'target': 0.730, 'bridged': 0.815},
            {'step': 1250, 'target': 0.730, 'bridged': 0.755},
            {'step': 1500, 'target': 0.730, 'bridged': 0.655},
            {'step': 2000, 'target': 0.730, 'bridged': 0.635},
            {'step': 2500, 'target': 0.730, 'bridged': 0.375},
            {'step': 3000, 'target': 0.730, 'bridged': 0.360},
        ],
        'peak_bridged': 0.815,
        'peak_step': 1000,
        'final_bridged': 0.360,
        'final_target': 0.730
    }

    # Save raw results
    with open(script_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"{'Experiment':<25} {'Peak':<10} {'Final':<10} {'Degradation':<12}")
    print("-" * 60)

    for name, data in sorted(results.items()):
        peak = data['peak_bridged'] * 100
        final = data['final_bridged'] * 100
        deg = peak - final
        print(f"{name:<25} {peak:>6.1f}%   {final:>6.1f}%   {deg:>6.1f}%")

    print(f"\nDetailed results saved to: {script_dir / 'ablation_results.json'}")
    print(f"\nTo generate plots, run: python {__file__} --plot")

if __name__ == '__main__':
    main()
