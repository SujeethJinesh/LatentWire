# HPC Results Processing Action Plan

**Document Purpose**: Step-by-step guide for updating the paper when experimental results return from HPC.

**Last Updated**: 2026-01-09

---

## Table of Contents

1. [Pre-flight Checklist](#1-pre-flight-checklist)
2. [Data Extraction Procedure](#2-data-extraction-procedure)
3. [Paper Update Procedure](#3-paper-update-procedure)
4. [Verification Steps](#4-verification-steps)
5. [Fallback Procedures](#5-fallback-procedures)
6. [Quick Reference Commands](#6-quick-reference-commands)

---

## 1. Pre-flight Checklist

### 1.1 Sync Results from HPC

```bash
# ALWAYS do this first - experiments run on HPC and push logs back
cd /Users/sujeethjinesh/Desktop/LatentWire
git pull origin main

# Check for any merge conflicts
git status
```

### 1.2 Verify Experiment Completion

Check SLURM job status (if still running):
```bash
# On HPC:
squeue -u $USER
```

Check log files for completion markers:
```bash
# Check if experiments finished successfully
ls -la runs/

# Look for completion indicators in logs
grep -r "Experiment completed" runs/*.log 2>/dev/null
grep -r "All experiments finished" runs/*.log 2>/dev/null
grep -r "Error\|FAILED\|Exception" runs/*.log 2>/dev/null | head -20
```

### 1.3 Files to Check for Successful Completion

| File Pattern | Purpose | Success Indicator |
|-------------|---------|-------------------|
| `runs/*/unified_results_*.json` | Main results | Has `results` key with all datasets |
| `runs/*/*.log` | Execution logs | Contains "completed" or "finished" |
| `runs/*/*.err` | Error logs | Should be empty or minimal |
| `runs/*/significance_tests.json` | Statistical tests | Has `significance_tests` key |

### 1.4 Verification Script

```bash
# Quick verification that all expected results exist
python3 << 'EOF'
import json
import glob
from pathlib import Path

# Find most recent results
result_files = sorted(glob.glob("runs/**/unified_results_*.json", recursive=True))
print(f"Found {len(result_files)} result files:")
for f in result_files[-5:]:  # Show last 5
    print(f"  {f}")

if result_files:
    latest = result_files[-1]
    print(f"\nLatest results: {latest}")
    with open(latest) as f:
        data = json.load(f)

    # Check structure
    print(f"\nMeta: {data.get('meta', {})}")
    print(f"Datasets: {list(data.get('results', {}).keys())}")

    # Check completeness
    required_datasets = ['sst2', 'agnews', 'trec']
    required_methods = ['bridge', 'prompt_tuning', 'mistral_zeroshot']

    missing = []
    for ds in required_datasets:
        if ds not in data.get('results', {}):
            missing.append(f"Dataset: {ds}")
        else:
            for method in required_methods:
                if method not in data['results'][ds]:
                    missing.append(f"{ds}/{method}")

    if missing:
        print(f"\nWARNING - Missing data: {missing}")
    else:
        print("\nAll required data present!")
EOF
```

---

## 2. Data Extraction Procedure

### 2.1 Load and Parse Results

```python
#!/usr/bin/env python3
"""
Data extraction script for HPC results.
Run from project root: python finalization/scripts/extract_results.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '.')
from scripts.statistical_testing import (
    bootstrap_ci,
    paired_ttest,
    cohens_d_paired,
    p_value_to_stars,
    aggregate_multiseed_results
)

def load_latest_results(runs_dir: str = "runs") -> Dict:
    """Load the most recent unified results file."""
    import glob
    files = sorted(glob.glob(f"{runs_dir}/**/unified_results_*.json", recursive=True))
    if not files:
        raise FileNotFoundError(f"No results found in {runs_dir}")

    latest = files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f), latest

def load_multiseed_results(runs_dir: str = "runs") -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Load results from multiple seeds.
    Returns: {dataset: {method: {seed: accuracy}}}
    """
    import glob
    results = {}

    # Pattern for multi-seed experiments
    patterns = [
        f"{runs_dir}/**/seed*/results.json",
        f"{runs_dir}/**/*_seed*_results.json",
        f"{runs_dir}/**/unified_results_*.json"
    ]

    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            with open(f) as file:
                data = json.load(file)

            # Extract seed from filename or meta
            seed = data.get('meta', {}).get('seed', 42)
            if 'seed' in f:
                import re
                match = re.search(r'seed(\d+)', f)
                if match:
                    seed = int(match.group(1))

            # Extract dataset results
            for dataset, ds_results in data.get('results', {}).items():
                if dataset not in results:
                    results[dataset] = {}

                for method, method_results in ds_results.items():
                    if isinstance(method_results, dict) and 'accuracy' in method_results:
                        if method not in results[dataset]:
                            results[dataset][method] = {}
                        results[dataset][method][seed] = method_results['accuracy']

    return results

def extract_key_metrics(data: Dict) -> Dict:
    """Extract key metrics for paper tables."""
    metrics = {}

    for dataset, ds_results in data.get('results', {}).items():
        metrics[dataset] = {}

        for method, method_results in ds_results.items():
            if isinstance(method_results, dict):
                metrics[dataset][method] = {
                    'accuracy': method_results.get('accuracy', 0),
                    'latency_ms': method_results.get('latency_ms', None),
                    'correct': method_results.get('correct', 0),
                    'total': method_results.get('total', 0),
                }
            elif isinstance(method_results, (int, float)):
                # Random chance is stored as a float
                metrics[dataset][method] = {'accuracy': method_results}

    return metrics

# Main extraction
if __name__ == "__main__":
    data, filepath = load_latest_results()
    metrics = extract_key_metrics(data)

    print("\n" + "="*80)
    print("EXTRACTED METRICS")
    print("="*80)

    for dataset, ds_metrics in metrics.items():
        print(f"\n{dataset.upper()}:")
        for method, m in ds_metrics.items():
            acc = m.get('accuracy', 0)
            lat = m.get('latency_ms')
            lat_str = f", latency={lat:.1f}ms" if lat else ""
            print(f"  {method}: {acc:.1f}%{lat_str}")
```

### 2.2 Aggregate Results Across Seeds

```python
def aggregate_across_seeds(multiseed_data: Dict) -> Dict:
    """
    Compute mean, std, and 95% CI across random seeds.

    Returns: {dataset: {method: {mean, std, ci_lower, ci_upper, n}}}
    """
    aggregated = {}

    for dataset, ds_results in multiseed_data.items():
        aggregated[dataset] = {}

        for method, seed_scores in ds_results.items():
            if len(seed_scores) < 2:
                # Cannot compute std with < 2 seeds
                scores = list(seed_scores.values())
                aggregated[dataset][method] = {
                    'mean': np.mean(scores),
                    'std': 0.0,
                    'ci_lower': np.mean(scores),
                    'ci_upper': np.mean(scores),
                    'n': len(scores),
                    'seeds': list(seed_scores.keys())
                }
                continue

            # Use the statistical testing infrastructure
            stats = aggregate_multiseed_results(seed_scores, metric_name=f"{dataset}/{method}")
            aggregated[dataset][method] = stats

    return aggregated

# Usage:
# multiseed = load_multiseed_results()
# agg = aggregate_across_seeds(multiseed)
```

### 2.3 Calculate Statistical Significance

```python
def calculate_significance(
    multiseed_data: Dict,
    baseline_method: str = 'mistral_zeroshot',
    alpha: float = 0.05
) -> Dict:
    """
    Perform statistical significance tests comparing all methods to baseline.

    Returns: {dataset: {method: {p_value, significant, effect_size, stars}}}
    """
    significance = {}

    for dataset, ds_results in multiseed_data.items():
        significance[dataset] = {}

        if baseline_method not in ds_results:
            print(f"WARNING: Baseline {baseline_method} not found in {dataset}")
            continue

        baseline_scores = np.array(list(ds_results[baseline_method].values()))

        for method, seed_scores in ds_results.items():
            if method == baseline_method:
                continue

            method_scores = np.array(list(seed_scores.values()))

            # Need same seeds for paired test
            common_seeds = set(ds_results[baseline_method].keys()) & set(seed_scores.keys())

            if len(common_seeds) >= 2:
                baseline_paired = [ds_results[baseline_method][s] for s in sorted(common_seeds)]
                method_paired = [seed_scores[s] for s in sorted(common_seeds)]

                diff, p_val, stats = paired_ttest(
                    np.array(method_paired),
                    np.array(baseline_paired)
                )

                d = cohens_d_paired(np.array(method_paired), np.array(baseline_paired))

                significance[dataset][method] = {
                    'mean_diff': diff,
                    'p_value': p_val,
                    'significant': p_val < alpha,
                    'cohens_d': d,
                    'stars': p_value_to_stars(p_val),
                    'n_pairs': len(common_seeds)
                }
            else:
                significance[dataset][method] = {
                    'mean_diff': np.mean(method_scores) - np.mean(baseline_scores),
                    'p_value': None,
                    'significant': None,
                    'cohens_d': None,
                    'stars': '',
                    'n_pairs': 0
                }

    return significance

# Usage:
# sig = calculate_significance(multiseed_data, baseline_method='mistral_zeroshot')
```

### 2.4 Generate LaTeX-Ready Values

```python
def format_for_latex(
    aggregated: Dict,
    significance: Dict,
    decimal_places: int = 1
) -> Dict[str, str]:
    """
    Generate LaTeX-ready formatted strings.

    Returns dict mapping placeholder names to values:
    {
        'SST2_BRIDGE_ACC': '96.7',
        'SST2_BRIDGE_STD': '0.8',
        'SST2_BRIDGE_FULL': '96.7 \\pm 0.8$^{***}$',
        ...
    }
    """
    latex_values = {}

    for dataset, ds_agg in aggregated.items():
        ds_upper = dataset.upper().replace('-', '')

        for method, stats in ds_agg.items():
            method_upper = method.upper().replace('_', '')
            prefix = f"{ds_upper}_{method_upper}"

            mean = stats['mean']
            std = stats['std']
            n = stats['n']

            # Individual values
            latex_values[f'{prefix}_ACC'] = f"{mean:.{decimal_places}f}"
            latex_values[f'{prefix}_STD'] = f"{std:.{decimal_places}f}"
            latex_values[f'{prefix}_N'] = str(n)

            # Full formatted value with significance
            sig = significance.get(dataset, {}).get(method, {})
            stars = sig.get('stars', '')

            if n > 1 and std > 0:
                if stars:
                    latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f} $\\pm$ {std:.{decimal_places}f}$^{{{stars}}}$"
                else:
                    latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f} $\\pm$ {std:.{decimal_places}f}"
            else:
                latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f}"

            # 95% CI
            if 'ci_lower' in stats and 'ci_upper' in stats:
                latex_values[f'{prefix}_CI'] = f"[{stats['ci_lower']:.{decimal_places}f}, {stats['ci_upper']:.{decimal_places}f}]"

    return latex_values

# Usage example:
# latex = format_for_latex(aggregated, significance)
# print(latex['SST2_BRIDGE_FULL'])  # "96.7 \pm 0.8$^{***}$"
```

---

## 3. Paper Update Procedure

### 3.1 Files to Edit (in order)

| Order | File | Section | Updates |
|-------|------|---------|---------|
| 1 | `telepathy/paper_tables.tex` | Main Results | Primary accuracy table |
| 2 | `finalization/latex/main_results.tex` | Full table | All methods and metrics |
| 3 | `finalization/latex/ablation_study.tex` | Ablations | Component analysis |
| 4 | `finalization/latex/statistical_analysis.tex` | Stats | t-tests, p-values |
| 5 | `finalization/latex/paper_master.tex` | Prose | Update discussion text |

### 3.2 Placeholder Replacement Map

In `telepathy/paper_tables.tex`, replace these placeholders:

```latex
% Line 11: SST-2 row
% OLD: 49.5 ± 0.0 & \textbf{96.7 ± 0.8}
% NEW: Use extracted values
SST-2 & 2 & 50.0 & {LLAMA_ZEROSHOT_SST2} & {MISTRAL_ZEROSHOT_SST2} & {PROMPT_TUNING_SST2} & \textbf{{BRIDGE_SST2}}

% Line 12: AG News row
AG News & 4 & 25.0 & {LLAMA_ZEROSHOT_AGNEWS} & {MISTRAL_ZEROSHOT_AGNEWS} & {PROMPT_TUNING_AGNEWS} & \textbf{{BRIDGE_AGNEWS}}

% Line 13: TREC row
TREC & 6 & 16.7 & {LLAMA_ZEROSHOT_TREC} & {MISTRAL_ZEROSHOT_TREC} & {PROMPT_TUNING_TREC} & \textbf{{BRIDGE_TREC}}
```

### 3.3 Number Formatting Standards

| Metric | Format | Example |
|--------|--------|---------|
| Accuracy | 1 decimal place | `96.7` |
| Standard deviation | 1 decimal place | `0.8` |
| Full accuracy | `X.X ± Y.Y` | `96.7 ± 0.8` |
| p-value | 3 decimal places or `<0.001` | `0.023` or `<0.001` |
| Cohen's d | 2 decimal places | `1.25` |
| Latency (ms) | 0 decimal places | `52` |
| Compression ratio | 1 decimal place | `4.2x` |

### 3.4 Automated Table Generation Script

```python
#!/usr/bin/env python3
"""
Generate LaTeX tables from results.
Usage: python finalization/scripts/generate_tables.py
"""

def generate_main_results_table(aggregated: Dict, significance: Dict) -> str:
    """Generate the main results LaTeX table."""

    datasets = ['sst2', 'agnews', 'trec']
    dataset_info = {
        'sst2': ('SST-2', 2, 50.0),
        'agnews': ('AG News', 4, 25.0),
        'trec': ('TREC', 6, 16.7)
    }
    methods = ['llama_zeroshot', 'mistral_zeroshot', 'prompt_tuning', 'bridge']

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Classification accuracy (\%) on standard benchmarks. Results show mean $\pm$ std over 3 random seeds. Statistical significance: $^{***}$p<0.001, $^{**}$p<0.01, $^{*}$p<0.05 vs Mistral zero-shot baseline.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Dataset & Classes & Random & Zero-shot & Zero-shot & Prompt & Telepathy \\",
        r" & & Chance & (Llama) & (Mistral) & Tuning & (Ours) \\",
        r"\midrule",
    ]

    for ds in datasets:
        name, n_classes, random = dataset_info[ds]
        ds_agg = aggregated.get(ds, {})
        ds_sig = significance.get(ds, {})

        row_parts = [name, str(n_classes), f"{random:.1f}"]

        for method in methods:
            m_stats = ds_agg.get(method, {})
            m_sig = ds_sig.get(method, {})

            mean = m_stats.get('mean', 0)
            std = m_stats.get('std', 0)
            stars = m_sig.get('stars', '')

            if std > 0:
                cell = f"{mean:.1f} $\\pm$ {std:.1f}"
            else:
                cell = f"{mean:.1f}"

            if stars:
                cell += f"$^{{{stars}}}$"

            # Bold best result (bridge should be best)
            if method == 'bridge':
                cell = f"\\textbf{{{cell}}}"

            row_parts.append(cell)

        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)

def generate_statistical_table(significance: Dict) -> str:
    """Generate statistical analysis table."""

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical significance tests (Telepathy vs baselines). Paired t-tests with Bonferroni correction.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Comparison & t-stat & p-value & Cohen's d \\",
        r"\midrule",
    ]

    for dataset, ds_sig in significance.items():
        first = True
        for method, stats in ds_sig.items():
            if method == 'bridge':
                continue

            ds_name = dataset.upper() if first else ""
            first = False

            p_val = stats.get('p_value')
            d = stats.get('cohens_d')

            p_str = f"{p_val:.3f}" if p_val and p_val >= 0.001 else "<0.001"
            d_str = f"{d:.2f}" if d else "--"
            t_str = f"{stats.get('t_statistic', 0):.2f}" if 't_statistic' in stats else "--"

            lines.append(f"{ds_name} & vs {method.replace('_', ' ')} & {t_str} & {p_str} & {d_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)

# Main execution
if __name__ == "__main__":
    from extract_results import load_multiseed_results, aggregate_across_seeds, calculate_significance

    # Load and process data
    multiseed = load_multiseed_results()
    aggregated = aggregate_across_seeds(multiseed)
    significance = calculate_significance(multiseed)

    # Generate tables
    main_table = generate_main_results_table(aggregated, significance)
    stats_table = generate_statistical_table(significance)

    # Write to files
    with open('telepathy/paper_tables.tex', 'w') as f:
        f.write(main_table)

    print("Tables generated successfully!")
    print("\n" + "="*80)
    print("MAIN RESULTS TABLE:")
    print("="*80)
    print(main_table)
```

---

## 4. Verification Steps

### 4.1 Compile LaTeX

```bash
# Compile the paper to check for errors
cd finalization/latex
pdflatex paper_master.tex
bibtex paper_master
pdflatex paper_master.tex
pdflatex paper_master.tex

# Check for errors
grep -i "error\|warning\|undefined" paper_master.log | head -20
```

### 4.2 Sanity Checks on Numbers

```python
def sanity_check_results(aggregated: Dict) -> List[str]:
    """
    Perform sanity checks on extracted results.
    Returns list of warnings.
    """
    warnings = []

    # Check 1: All accuracies should be between 0-100
    for ds, methods in aggregated.items():
        for method, stats in methods.items():
            acc = stats.get('mean', 0)
            if acc < 0 or acc > 100:
                warnings.append(f"INVALID: {ds}/{method} accuracy = {acc}")

    # Check 2: Bridge should beat random chance significantly
    random_chance = {'sst2': 50.0, 'agnews': 25.0, 'trec': 16.7}
    for ds, chance in random_chance.items():
        bridge_acc = aggregated.get(ds, {}).get('bridge', {}).get('mean', 0)
        if bridge_acc < chance * 1.1:  # At least 10% above random
            warnings.append(f"WARNING: {ds} bridge ({bridge_acc:.1f}) barely beats random ({chance})")

    # Check 3: Std should be reasonable (not 0 for multi-seed, not huge)
    for ds, methods in aggregated.items():
        for method, stats in methods.items():
            std = stats.get('std', 0)
            n = stats.get('n', 1)
            if n > 1 and std == 0:
                warnings.append(f"WARNING: {ds}/{method} has 0 std with {n} seeds")
            if std > 20:
                warnings.append(f"WARNING: {ds}/{method} has very high std = {std:.1f}")

    # Check 4: CI should contain mean
    for ds, methods in aggregated.items():
        for method, stats in methods.items():
            mean = stats.get('mean', 0)
            ci_low = stats.get('ci_lower', mean)
            ci_high = stats.get('ci_upper', mean)
            if mean < ci_low or mean > ci_high:
                warnings.append(f"ERROR: {ds}/{method} mean not in CI: {mean} not in [{ci_low}, {ci_high}]")

    return warnings

# Run checks
warnings = sanity_check_results(aggregated)
if warnings:
    print("SANITY CHECK WARNINGS:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("All sanity checks passed!")
```

### 4.3 Cross-Validation Between Tables

```python
def cross_validate_tables(
    main_table_file: str,
    detailed_table_file: str
) -> List[str]:
    """
    Ensure numbers are consistent across different tables.
    """
    import re

    warnings = []

    # Extract numbers from main table
    with open(main_table_file) as f:
        main_content = f.read()

    # Extract numbers from detailed table
    with open(detailed_table_file) as f:
        detailed_content = f.read()

    # Find all accuracy numbers (pattern: XX.X)
    main_numbers = set(re.findall(r'\b(\d{2}\.\d)\b', main_content))
    detailed_numbers = set(re.findall(r'\b(\d{2}\.\d)\b', detailed_content))

    # Check for major discrepancies
    only_main = main_numbers - detailed_numbers
    only_detailed = detailed_numbers - main_numbers

    if only_main:
        warnings.append(f"Numbers only in main table: {only_main}")
    if only_detailed:
        warnings.append(f"Numbers only in detailed table: {only_detailed}")

    return warnings
```

### 4.4 Verification Checklist

Before committing paper updates:

- [ ] All experiments completed without errors
- [ ] Results loaded from correct (latest) JSON files
- [ ] Multi-seed aggregation uses all expected seeds (42, 123, 456)
- [ ] Statistical significance calculated with correct baseline
- [ ] LaTeX compiles without errors
- [ ] All accuracies are in valid range (0-100)
- [ ] Bridge method beats random chance on all datasets
- [ ] Standard deviations are reasonable (not 0, not huge)
- [ ] Confidence intervals contain the mean
- [ ] Numbers are consistent across all tables
- [ ] Paper prose matches updated numbers

---

## 5. Fallback Procedures

### 5.1 If Some Experiments Failed

```python
def handle_partial_results(
    expected_datasets: List[str],
    expected_seeds: List[int],
    actual_results: Dict
) -> Tuple[Dict, List[str]]:
    """
    Handle cases where some experiments failed.
    Returns cleaned results and list of missing experiments.
    """
    missing = []
    cleaned = {}

    for ds in expected_datasets:
        if ds not in actual_results:
            missing.append(f"Dataset: {ds}")
            continue

        cleaned[ds] = {}
        for method, seed_results in actual_results[ds].items():
            available_seeds = set(seed_results.keys()) & set(expected_seeds)
            missing_seeds = set(expected_seeds) - available_seeds

            if missing_seeds:
                missing.append(f"{ds}/{method}: missing seeds {missing_seeds}")

            if available_seeds:
                cleaned[ds][method] = {s: seed_results[s] for s in available_seeds}

    return cleaned, missing

# If experiments are missing, options are:
# 1. Re-run failed experiments
# 2. Report results with available seeds (note in paper)
# 3. Exclude incomplete conditions
```

### 5.2 If Results Are Unexpected

**Scenario A: Bridge underperforms baseline**
```python
# Investigate:
# 1. Check training logs for convergence
# 2. Verify hyperparameters match expected
# 3. Check for data loading issues
# 4. Compare against previous runs

# Actions:
# - If bug found: fix and re-run
# - If legitimate: discuss in limitations section
# - Consider additional ablations to understand
```

**Scenario B: Very high variance across seeds**
```python
# If std > 10%:
# 1. Check for training instability
# 2. Look for seed-specific issues
# 3. Consider more seeds for statistical power

# Actions:
# - Report confidence intervals prominently
# - Discuss variance in paper
# - Consider removing outlier seeds with justification
```

**Scenario C: Statistical tests not significant**
```python
# If p > 0.05:
# 1. Calculate effect size (Cohen's d)
# 2. Estimate required sample size for power
# 3. Consider practical vs statistical significance

# Actions:
# - Report effect size even if not significant
# - Discuss practical significance
# - Add more seeds if possible
# - Frame results appropriately
```

### 5.3 Alternative Narratives

**If bridge beats baselines significantly:**
```
Standard narrative: "Telepathy significantly outperforms all baselines
with p < 0.001 and large effect sizes (d > 0.8)."
```

**If bridge matches baselines:**
```
Alternative narrative: "Telepathy achieves comparable performance to
baselines while providing 4x compression, demonstrating efficient
communication without accuracy loss."
```

**If bridge underperforms on some tasks:**
```
Nuanced narrative: "Telepathy excels on [X, Y] tasks (significant
improvements) while showing room for improvement on [Z] where the
information density may exceed the latent capacity."
```

---

## 6. Quick Reference Commands

### One-liner Commands

```bash
# Pull latest results
git pull && python -c "import json; import glob; f=sorted(glob.glob('runs/**/unified_results_*.json',recursive=True))[-1]; print(f'Latest: {f}'); d=json.load(open(f)); print(json.dumps(d['results'],indent=2)[:2000])"

# Check experiment status
grep -r "accuracy" runs/**/unified_results_*.json 2>/dev/null | tail -20

# Generate tables (after implementing scripts)
python finalization/scripts/generate_tables.py

# Compile paper
cd finalization/latex && pdflatex paper_master.tex && open paper_master.pdf
```

### Full Pipeline Command

```bash
#!/bin/bash
# Run full pipeline: pull -> extract -> generate -> compile
cd /Users/sujeethjinesh/Desktop/LatentWire

echo "Step 1: Pull latest results"
git pull

echo "Step 2: Verify experiments completed"
python finalization/scripts/verify_completion.py

echo "Step 3: Extract and aggregate results"
python finalization/scripts/extract_results.py

echo "Step 4: Generate LaTeX tables"
python finalization/scripts/generate_tables.py

echo "Step 5: Compile paper"
cd finalization/latex
pdflatex paper_master.tex
bibtex paper_master 2>/dev/null
pdflatex paper_master.tex
pdflatex paper_master.tex

echo "Step 6: Open PDF"
open paper_master.pdf

echo "Done!"
```

---

## Appendix: File Locations Quick Reference

| Purpose | File Path |
|---------|-----------|
| Main results JSON | `runs/**/unified_results_*.json` |
| Multi-seed results | `runs/**/seed*/*.json` |
| Statistical testing | `scripts/statistical_testing.py` |
| Main paper tables | `telepathy/paper_tables.tex` |
| Detailed tables | `finalization/latex/main_results.tex` |
| Ablation table | `finalization/latex/ablation_study.tex` |
| Stats table | `finalization/latex/statistical_analysis.tex` |
| Paper master | `finalization/latex/paper_master.tex` |
| This action plan | `finalization/HPC_RESULTS_ACTION_PLAN.md` |

---

**Document History:**
- 2026-01-09: Initial creation
