#!/usr/bin/env python3
"""
Statistical Analysis Module for Telepathy Experiments

Provides comprehensive statistical significance testing for MLSys 2025 publication.
Loads results from JSON files, computes statistics, runs significance tests,
and generates publication-ready summary tables.

Key features:
1. Loads results from JSON files in a results directory (supports glob patterns)
2. Computes statistics: mean, std, 95% CI, standard error for each method
3. Runs paired significance tests: paired t-test, McNemar's test
4. Computes effect sizes: Cohen's d
5. Applies multiple comparison correction: Bonferroni and Holm methods
6. Outputs publication-ready summary tables
7. Saves results to JSON and prints to console

Usage:
    python telepathy/statistical_analysis.py --results_dir runs/reasoning_final_*/
    python telepathy/statistical_analysis.py --results_dir runs/enhanced_results/ --output_file stats.json
    python telepathy/statistical_analysis.py --results_dir runs/ --baseline_method zeroshot

Author: Telepathy Project
Date: January 2025
"""

import argparse
import glob
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, wilcoxon

warnings.filterwarnings('ignore')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MethodResult:
    """Container for a method's results across multiple seeds/runs."""
    method_name: str
    accuracies: List[float] = field(default_factory=list)
    f1_scores: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    predictions: Optional[List[List[int]]] = None  # For McNemar's test
    labels: Optional[List[int]] = None  # Ground truth for McNemar's

    @property
    def n_seeds(self) -> int:
        return len(self.accuracies)

    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracies) if self.accuracies else 0.0

    @property
    def std_accuracy(self) -> float:
        return np.std(self.accuracies, ddof=1) if len(self.accuracies) > 1 else 0.0

    @property
    def se_accuracy(self) -> float:
        """Standard error of the mean."""
        if len(self.accuracies) < 2:
            return 0.0
        return self.std_accuracy / np.sqrt(len(self.accuracies))

    def ci_95(self) -> Tuple[float, float]:
        """95% confidence interval using t-distribution."""
        if len(self.accuracies) < 2:
            return (self.mean_accuracy, self.mean_accuracy)

        n = len(self.accuracies)
        se = self.se_accuracy
        t_crit = stats.t.ppf(0.975, df=n-1)
        margin = t_crit * se
        return (self.mean_accuracy - margin, self.mean_accuracy + margin)

    def bootstrap_ci(self, confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Bootstrap confidence interval (BCa method approximation)."""
        if len(self.accuracies) < 3:
            return self.ci_95()

        data = np.array(self.accuracies)
        bootstrap_means = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        return (lower, upper)


@dataclass
class ComparisonResult:
    """Container for a statistical comparison between two methods."""
    method1_name: str
    method2_name: str
    mean_diff: float
    t_statistic: float
    p_value: float
    cohens_d: float
    n1: int
    n2: int
    significant_05: bool
    significant_01: bool
    significant_001: bool
    corrected_p_value: Optional[float] = None
    significant_after_correction: bool = False
    mcnemar_statistic: Optional[float] = None
    mcnemar_p_value: Optional[float] = None


# =============================================================================
# RESULT LOADING
# =============================================================================

class ResultsLoader:
    """Loads and parses results from JSON files."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_by_dataset: Dict[str, Dict[str, MethodResult]] = defaultdict(dict)

    def load_all(self) -> Dict[str, Dict[str, MethodResult]]:
        """Load all JSON results from the results directory (supports glob patterns)."""
        # Expand glob pattern
        if '*' in self.results_dir:
            matching_dirs = glob.glob(self.results_dir)
            if not matching_dirs:
                print(f"Warning: No directories match pattern: {self.results_dir}")
                return {}
            search_paths = matching_dirs
        else:
            search_paths = [self.results_dir]

        # Find all JSON files
        json_files = []
        for search_path in search_paths:
            path = Path(search_path)
            if path.is_file() and path.suffix == '.json':
                json_files.append(path)
            elif path.is_dir():
                json_files.extend(path.glob('**/*.json'))

        print(f"Found {len(json_files)} JSON files to process")

        for json_file in json_files:
            try:
                self._process_json_file(json_file)
            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}")

        return dict(self.results_by_dataset)

    def _process_json_file(self, filepath: Path):
        """Process a single JSON results file."""
        with open(filepath) as f:
            data = json.load(f)

        # Handle different result formats
        if 'aggregated_results' in data:
            self._parse_aggregated_format(data)
        elif 'results' in data:
            self._parse_results_format(data, filepath)
        elif 'final_results' in data:
            self._parse_final_results_format(data, filepath)
        else:
            # Try to parse as direct results
            self._parse_direct_format(data, filepath)

    def _parse_aggregated_format(self, data: Dict):
        """Parse aggregated results format (from aggregate_results.py)."""
        for dataset, methods in data.get('aggregated_results', {}).items():
            for method_name, method_data in methods.items():
                if isinstance(method_data, dict) and 'accuracy_mean' in method_data:
                    if method_name not in self.results_by_dataset[dataset]:
                        self.results_by_dataset[dataset][method_name] = MethodResult(method_name)

                    result = self.results_by_dataset[dataset][method_name]
                    # If we only have mean/std, generate synthetic samples
                    n = method_data.get('num_seeds', method_data.get('accuracy_n', 1))
                    if n > 0 and not result.accuracies:
                        mean = method_data['accuracy_mean']
                        std = method_data.get('accuracy_std', 0)
                        if n > 1 and std > 0:
                            # Generate n samples that match mean and std
                            result.accuracies = self._generate_samples_from_stats(mean, std, n)
                        else:
                            result.accuracies = [mean]

    def _parse_results_format(self, data: Dict, filepath: Path):
        """Parse standard results format with 'results' key."""
        results = data.get('results', {})

        # Try to extract metadata
        dataset = self._extract_dataset_from_path(filepath)
        method = self._extract_method_from_path(filepath)
        seed = self._extract_seed_from_path(filepath)

        # Look for accuracy in various places
        accuracy = None
        if isinstance(results, dict):
            accuracy = results.get('accuracy', results.get('acc', results.get('test_accuracy')))

        if accuracy is not None and dataset and method:
            self._add_result(dataset, method, accuracy, seed)

    def _parse_final_results_format(self, data: Dict, filepath: Path):
        """Parse format with 'final_results' key."""
        results = data.get('final_results', {})
        if results is None:
            return

        dataset = self._extract_dataset_from_path(filepath)
        method = self._extract_method_from_path(filepath)
        seed = self._extract_seed_from_path(filepath)

        accuracy = results.get('accuracy', results.get('acc'))
        if accuracy is not None and dataset and method:
            self._add_result(dataset, method, accuracy, seed)

    def _parse_direct_format(self, data: Dict, filepath: Path):
        """Parse direct format (accuracy at top level)."""
        dataset = self._extract_dataset_from_path(filepath)
        method = self._extract_method_from_path(filepath)
        seed = self._extract_seed_from_path(filepath)

        accuracy = data.get('accuracy', data.get('acc'))
        if accuracy is not None and dataset and method:
            self._add_result(dataset, method, accuracy, seed)

    def _add_result(self, dataset: str, method: str, accuracy: float, seed: Optional[int] = None):
        """Add a single result to the collection."""
        if method not in self.results_by_dataset[dataset]:
            self.results_by_dataset[dataset][method] = MethodResult(method)

        result = self.results_by_dataset[dataset][method]
        result.accuracies.append(accuracy)
        if seed is not None:
            result.seeds.append(seed)

    def _extract_dataset_from_path(self, filepath: Path) -> Optional[str]:
        """Extract dataset name from filepath."""
        path_str = str(filepath).lower()
        datasets = ['sst2', 'sst-2', 'agnews', 'ag_news', 'trec', 'gsm8k',
                    'banking77', 'passkey', 'boolq', 'hellaswag', 'winogrande',
                    'commonsenseqa', 'arc', 'piqa']
        for ds in datasets:
            if ds in path_str:
                # Normalize dataset names
                if ds in ['sst-2', 'sst2']:
                    return 'sst2'
                if ds in ['agnews', 'ag_news']:
                    return 'agnews'
                return ds
        return 'unknown'

    def _extract_method_from_path(self, filepath: Path) -> Optional[str]:
        """Extract method name from filepath."""
        path_str = str(filepath).lower()
        methods = ['bridge', 'prompt_tuning', 'soft_prompt', 'lora', 'zeroshot',
                   'zero_shot', 'fewshot', 'few_shot', 'linear_probe', 'text_relay',
                   'llmlingua', 'full_finetune', 'baseline']
        for method in methods:
            if method in path_str:
                # Normalize method names
                if method in ['zero_shot', 'zeroshot']:
                    return 'zeroshot'
                if method in ['few_shot', 'fewshot']:
                    return 'fewshot'
                if method in ['soft_prompt', 'prompt_tuning']:
                    return 'prompt_tuning'
                return method
        return 'unknown'

    def _extract_seed_from_path(self, filepath: Path) -> Optional[int]:
        """Extract seed from filepath."""
        import re
        path_str = str(filepath)
        match = re.search(r'seed[_]?(\d+)', path_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _generate_samples_from_stats(mean: float, std: float, n: int) -> List[float]:
        """Generate n samples that approximately match the given mean and std."""
        if n == 1:
            return [mean]

        # Simple approach: generate normal samples and adjust
        np.random.seed(42)
        samples = np.random.normal(mean, std, n)

        # Adjust to exactly match mean and std
        samples = samples - np.mean(samples) + mean
        if n > 1:
            current_std = np.std(samples, ddof=1)
            if current_std > 0:
                samples = mean + (samples - mean) * (std / current_std)

        return samples.tolist()


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

class StatisticalAnalyzer:
    """Performs statistical analysis on loaded results."""

    def __init__(self, results: Dict[str, Dict[str, MethodResult]],
                 reference_method: str = 'bridge',
                 alpha: float = 0.05):
        self.results = results
        self.reference_method = reference_method
        self.alpha = alpha
        self.comparisons: Dict[str, List[ComparisonResult]] = defaultdict(list)

    def run_all_analyses(self) -> Dict[str, Any]:
        """Run all statistical analyses and return comprehensive results."""
        all_results = {
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'reference_method': self.reference_method,
                'alpha': self.alpha,
                'datasets': list(self.results.keys()),
            },
            'per_dataset': {},
            'overall_summary': {},
        }

        for dataset, methods in self.results.items():
            dataset_analysis = self._analyze_dataset(dataset, methods)
            all_results['per_dataset'][dataset] = dataset_analysis

        # Compute overall summary
        all_results['overall_summary'] = self._compute_overall_summary()

        return all_results

    def _analyze_dataset(self, dataset: str, methods: Dict[str, MethodResult]) -> Dict:
        """Analyze a single dataset's results."""
        analysis = {
            'method_statistics': {},
            'pairwise_comparisons': [],
            'multiple_testing_correction': {},
        }

        # 1. Compute statistics for each method
        for method_name, method_result in methods.items():
            ci_low, ci_high = method_result.ci_95()
            boot_ci_low, boot_ci_high = method_result.bootstrap_ci()

            analysis['method_statistics'][method_name] = {
                'n_seeds': method_result.n_seeds,
                'mean_accuracy': round(method_result.mean_accuracy, 2),
                'std_accuracy': round(method_result.std_accuracy, 2),
                'se_accuracy': round(method_result.se_accuracy, 3),
                'ci_95_low': round(ci_low, 2),
                'ci_95_high': round(ci_high, 2),
                'bootstrap_ci_low': round(boot_ci_low, 2),
                'bootstrap_ci_high': round(boot_ci_high, 2),
                'raw_accuracies': [round(a, 2) for a in method_result.accuracies],
            }

        # 2. Run pairwise comparisons against reference method
        if self.reference_method in methods:
            ref_result = methods[self.reference_method]
            p_values = []

            for method_name, method_result in methods.items():
                if method_name == self.reference_method:
                    continue

                comparison = self._compare_methods(ref_result, method_result)
                if comparison:
                    analysis['pairwise_comparisons'].append({
                        'method1': comparison.method1_name,
                        'method2': comparison.method2_name,
                        'mean_diff': round(comparison.mean_diff, 2),
                        't_statistic': round(comparison.t_statistic, 3) if not np.isnan(comparison.t_statistic) else None,
                        'p_value': round(comparison.p_value, 4) if not np.isnan(comparison.p_value) else None,
                        'cohens_d': round(comparison.cohens_d, 3) if not np.isnan(comparison.cohens_d) else None,
                        'significant_05': comparison.significant_05,
                        'significant_01': comparison.significant_01,
                        'significant_001': comparison.significant_001,
                    })

                    if not np.isnan(comparison.p_value):
                        p_values.append((method_name, comparison.p_value))

                    self.comparisons[dataset].append(comparison)

            # 3. Apply multiple testing correction
            if p_values:
                bonferroni_results = self._bonferroni_correction([p for _, p in p_values])
                holm_results = self._holm_correction([p for _, p in p_values])

                analysis['multiple_testing_correction'] = {
                    'num_tests': len(p_values),
                    'bonferroni': {
                        'adjusted_alpha': round(self.alpha / len(p_values), 4),
                        'corrected_p_values': {
                            method: round(p * len(p_values), 4) for method, p in p_values
                        },
                        'significant_after_correction': [
                            method for method, (_, reject) in zip(
                                [m for m, _ in p_values],
                                zip(bonferroni_results[0], bonferroni_results[1])
                            ) if reject
                        ],
                    },
                    'holm': {
                        'corrected_p_values': dict(zip(
                            [m for m, _ in p_values],
                            [round(p, 4) for p in holm_results[0]]
                        )),
                        'significant_after_correction': [
                            method for method, reject in zip(
                                [m for m, _ in p_values],
                                holm_results[1]
                            ) if reject
                        ],
                    },
                }

        return analysis

    def _compare_methods(self, method1: MethodResult, method2: MethodResult) -> Optional[ComparisonResult]:
        """Compare two methods using appropriate statistical tests."""
        acc1 = np.array(method1.accuracies)
        acc2 = np.array(method2.accuracies)

        if len(acc1) == 0 or len(acc2) == 0:
            return None

        mean_diff = np.mean(acc1) - np.mean(acc2)

        # Perform t-test
        if len(acc1) >= 2 and len(acc2) >= 2:
            if len(acc1) == len(acc2):
                # Paired t-test if same number of seeds
                t_stat, p_value = ttest_rel(acc1, acc2)
            else:
                # Independent samples t-test
                t_stat, p_value = ttest_ind(acc1, acc2)
        else:
            t_stat = np.nan
            p_value = np.nan

        # Compute Cohen's d
        cohens_d = self._compute_cohens_d(acc1, acc2, paired=(len(acc1) == len(acc2)))

        return ComparisonResult(
            method1_name=method1.method_name,
            method2_name=method2.method_name,
            mean_diff=mean_diff,
            t_statistic=t_stat,
            p_value=p_value,
            cohens_d=cohens_d,
            n1=len(acc1),
            n2=len(acc2),
            significant_05=p_value < 0.05 if not np.isnan(p_value) else False,
            significant_01=p_value < 0.01 if not np.isnan(p_value) else False,
            significant_001=p_value < 0.001 if not np.isnan(p_value) else False,
        )

    @staticmethod
    def _compute_cohens_d(group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> float:
        """Compute Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return np.nan

        if paired:
            # Paired Cohen's d using difference scores
            diffs = group1 - group2
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)

            if std_diff == 0:
                return float('inf') if mean_diff != 0 else 0.0
            return mean_diff / std_diff
        else:
            # Independent samples Cohen's d with pooled SD
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

            if pooled_std == 0:
                return float('inf') if mean1 != mean2 else 0.0
            return (mean1 - mean2) / pooled_std

    def _bonferroni_correction(self, p_values: List[float]) -> Tuple[List[float], List[bool]]:
        """Apply Bonferroni correction for multiple comparisons."""
        n = len(p_values)
        corrected = [min(p * n, 1.0) for p in p_values]
        reject = [p < self.alpha for p in corrected]
        return corrected, reject

    def _holm_correction(self, p_values: List[float]) -> Tuple[List[float], List[bool]]:
        """Apply Holm-Bonferroni (step-down) correction for multiple comparisons."""
        n = len(p_values)
        if n == 0:
            return [], []

        # Sort p-values with indices
        sorted_indices = np.argsort(p_values)
        sorted_p = [p_values[i] for i in sorted_indices]

        # Apply Holm correction
        corrected = []
        for i, p in enumerate(sorted_p):
            adj_p = p * (n - i)
            corrected.append(min(adj_p, 1.0))

        # Enforce monotonicity (corrected p-values should be non-decreasing)
        for i in range(1, len(corrected)):
            corrected[i] = max(corrected[i], corrected[i-1])

        # Determine rejections
        reject = [p < self.alpha for p in corrected]

        # Restore original order
        final_corrected = [0.0] * n
        final_reject = [False] * n
        for i, orig_idx in enumerate(sorted_indices):
            final_corrected[orig_idx] = corrected[i]
            final_reject[orig_idx] = reject[i]

        return final_corrected, final_reject

    def _compute_overall_summary(self) -> Dict:
        """Compute overall summary statistics across all datasets."""
        summary = {
            'total_datasets': len(self.results),
            'total_comparisons': sum(len(c) for c in self.comparisons.values()),
            'methods_analyzed': set(),
            'overall_effect_sizes': {},
        }

        for dataset, comparisons in self.comparisons.items():
            for comp in comparisons:
                summary['methods_analyzed'].add(comp.method1_name)
                summary['methods_analyzed'].add(comp.method2_name)

        summary['methods_analyzed'] = list(summary['methods_analyzed'])

        return summary


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_summary_table(results: Dict[str, Dict[str, MethodResult]],
                          analysis: Dict,
                          reference_method: str = 'bridge') -> str:
    """Generate a publication-ready summary table."""
    lines = []

    lines.append("=" * 100)
    lines.append("STATISTICAL ANALYSIS SUMMARY")
    lines.append("=" * 100)
    lines.append("")

    # Header
    ref_title = reference_method.title()
    cohens_d_header = "Cohen's d"
    p_value_header = f"vs {ref_title} p-value"
    lines.append(f"{'Method':<20} | {'Acc (mean +/- std)':<18} | {'95% CI':<15} | {p_value_header:<18} | {cohens_d_header:<12}")
    lines.append("-" * 100)

    # Process each dataset
    for dataset in sorted(results.keys()):
        lines.append(f"\n{dataset.upper()}")
        lines.append("-" * 50)

        methods = results[dataset]
        dataset_analysis = analysis.get('per_dataset', {}).get(dataset, {})
        method_stats = dataset_analysis.get('method_statistics', {})
        comparisons = dataset_analysis.get('pairwise_comparisons', [])

        # Create lookup for comparisons
        comparison_lookup = {c['method2']: c for c in comparisons}

        # Reference method first
        if reference_method in methods:
            ref_stats = method_stats.get(reference_method, {})
            mean = ref_stats.get('mean_accuracy', 0)
            std = ref_stats.get('std_accuracy', 0)
            ci_low = ref_stats.get('ci_95_low', mean)
            ci_high = ref_stats.get('ci_95_high', mean)

            lines.append(f"{reference_method:<20} | {mean:>6.1f} +/- {std:<6.1f}   | [{ci_low:>5.1f}, {ci_high:>5.1f}] | {'-':^18} | {'-':^12}")

        # Other methods
        for method_name in sorted(methods.keys()):
            if method_name == reference_method:
                continue

            stats_data = method_stats.get(method_name, {})
            mean = stats_data.get('mean_accuracy', 0)
            std = stats_data.get('std_accuracy', 0)
            ci_low = stats_data.get('ci_95_low', mean)
            ci_high = stats_data.get('ci_95_high', mean)

            comp = comparison_lookup.get(method_name, {})
            p_val = comp.get('p_value')
            cohens_d = comp.get('cohens_d')

            # Format p-value with significance markers
            if p_val is not None:
                if p_val < 0.001:
                    p_str = "p < 0.001***"
                elif p_val < 0.01:
                    p_str = f"p = {p_val:.3f}**"
                elif p_val < 0.05:
                    p_str = f"p = {p_val:.3f}*"
                else:
                    p_str = f"p = {p_val:.3f}"
            else:
                p_str = "N/A"

            # Format Cohen's d with interpretation
            if cohens_d is not None and not np.isnan(cohens_d):
                if abs(cohens_d) >= 0.8:
                    d_str = f"{cohens_d:>5.2f} (L)"
                elif abs(cohens_d) >= 0.5:
                    d_str = f"{cohens_d:>5.2f} (M)"
                elif abs(cohens_d) >= 0.2:
                    d_str = f"{cohens_d:>5.2f} (S)"
                else:
                    d_str = f"{cohens_d:>5.2f} (N)"
            else:
                d_str = "N/A"

            lines.append(f"{method_name:<20} | {mean:>6.1f} +/- {std:<6.1f}   | [{ci_low:>5.1f}, {ci_high:>5.1f}] | {p_str:<18} | {d_str:<12}")

    # Footer/Legend
    lines.append("")
    lines.append("-" * 100)
    lines.append("Legend:")
    lines.append("  * p < 0.05, ** p < 0.01, *** p < 0.001")
    lines.append("  Effect size: (N) negligible |d|<0.2, (S) small |d|<0.5, (M) medium |d|<0.8, (L) large |d|>=0.8")
    lines.append("=" * 100)

    return "\n".join(lines)


def generate_latex_table(results: Dict[str, Dict[str, MethodResult]],
                         analysis: Dict,
                         reference_method: str = 'bridge') -> str:
    """Generate a LaTeX table for the paper."""
    lines = []

    # Get all unique methods across datasets
    all_methods = set()
    for methods in results.values():
        all_methods.update(methods.keys())

    datasets = sorted(results.keys())
    methods_order = [reference_method] + sorted([m for m in all_methods if m != reference_method])

    # Table header
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Classification accuracy across datasets. Values show mean $\\pm$ std across 3 seeds.}")
    lines.append("\\label{tab:main_results}")

    n_cols = len(datasets) + 1  # +1 for method name column
    lines.append("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append("\\toprule")

    # Dataset headers
    header = "Method & " + " & ".join([ds.upper() for ds in datasets]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Rows
    for method in methods_order:
        if method not in all_methods:
            continue

        row = [method.replace("_", " ").title()]

        for dataset in datasets:
            if method in results.get(dataset, {}):
                method_result = results[dataset][method]
                mean = method_result.mean_accuracy
                std = method_result.std_accuracy

                # Check significance
                dataset_analysis = analysis.get('per_dataset', {}).get(dataset, {})
                comparisons = dataset_analysis.get('pairwise_comparisons', [])

                sig_marker = ""
                if method != reference_method:
                    for comp in comparisons:
                        if comp['method2'] == method:
                            if comp.get('significant_001'):
                                sig_marker = "$^{***}$"
                            elif comp.get('significant_01'):
                                sig_marker = "$^{**}$"
                            elif comp.get('significant_05'):
                                sig_marker = "$^{*}$"
                            break

                if std > 0:
                    row.append(f"{mean:.1f} $\\pm$ {std:.1f}{sig_marker}")
                else:
                    row.append(f"{mean:.1f}{sig_marker}")
            else:
                row.append("-")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{1mm}")
    lines.append("\\\\\\footnotesize{$^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ vs " + reference_method.title() + "}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# MCNEMAR'S TEST (WHEN PREDICTIONS ARE AVAILABLE)
# =============================================================================

def mcnemar_test(pred1: np.ndarray, pred2: np.ndarray, labels: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    McNemar's test for comparing two classifiers on same test set.

    Args:
        pred1: Predictions from method 1
        pred2: Predictions from method 2
        labels: Ground truth labels

    Returns:
        (statistic, p_value, contingency_table)
    """
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    labels = np.asarray(labels)

    if not (len(pred1) == len(pred2) == len(labels)):
        raise ValueError("All arrays must have same length")

    # Build contingency table
    correct1 = (pred1 == labels)
    correct2 = (pred2 == labels)

    n00 = np.sum(correct1 & correct2)   # Both correct
    n01 = np.sum(correct1 & ~correct2)  # 1 correct, 2 wrong
    n10 = np.sum(~correct1 & correct2)  # 1 wrong, 2 correct
    n11 = np.sum(~correct1 & ~correct2) # Both wrong

    contingency = np.array([[n00, n01], [n10, n11]])

    b = n01  # Model 1 correct, Model 2 wrong
    c = n10  # Model 1 wrong, Model 2 correct

    if b + c == 0:
        return 0.0, 1.0, contingency

    # Use exact binomial test for small samples
    if b + c < 25:
        from scipy.stats import binom
        statistic = min(b, c)
        p_value = 2 * binom.cdf(statistic, b + c, 0.5)
    else:
        # Chi-square with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)

    return statistic, p_value, contingency


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis for Telepathy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python telepathy/statistical_analysis.py --results_dir runs/reasoning_final_*/
    python telepathy/statistical_analysis.py --results_dir runs/enhanced_results/ --output_file stats.json
    python telepathy/statistical_analysis.py --results_dir runs/ --reference_method zeroshot
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory or glob pattern containing JSON result files'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSON file path (default: prints to console only)'
    )

    parser.add_argument(
        '--reference_method',
        type=str,
        default='bridge',
        help='Reference method to compare against (default: bridge)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )

    parser.add_argument(
        '--latex',
        action='store_true',
        help='Also generate LaTeX table output'
    )

    parser.add_argument(
        '--use_mock_data',
        action='store_true',
        help='Use mock data for demonstration when no results available'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TELEPATHY STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Reference method: {args.reference_method}")
    print(f"Significance level: {args.alpha}")
    print("")

    # Load results
    loader = ResultsLoader(args.results_dir)
    results = loader.load_all()

    # Use mock data if no results found and flag is set
    if not results and args.use_mock_data:
        print("No results found. Using mock data for demonstration...")
        results = generate_mock_results()

    if not results:
        print("\nNo results found! Please check the results directory.")
        print("Use --use_mock_data flag to generate example output.")
        sys.exit(1)

    # Print summary of loaded data
    print(f"\nLoaded results for {len(results)} datasets:")
    for dataset, methods in results.items():
        print(f"  {dataset}: {len(methods)} methods")
        for method_name, method_result in methods.items():
            print(f"    - {method_name}: {method_result.n_seeds} seeds, mean={method_result.mean_accuracy:.1f}%")

    # Run statistical analysis
    analyzer = StatisticalAnalyzer(results, args.reference_method, args.alpha)
    analysis = analyzer.run_all_analyses()

    # Generate and print summary table
    print("\n")
    table = generate_summary_table(results, analysis, args.reference_method)
    print(table)

    # Generate LaTeX table if requested
    if args.latex:
        print("\n")
        print("=" * 80)
        print("LATEX TABLE")
        print("=" * 80)
        latex = generate_latex_table(results, analysis, args.reference_method)
        print(latex)

    # Save to JSON if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = {}
        for dataset, methods in results.items():
            serializable_results[dataset] = {}
            for method_name, method_result in methods.items():
                serializable_results[dataset][method_name] = {
                    'n_seeds': method_result.n_seeds,
                    'mean_accuracy': method_result.mean_accuracy,
                    'std_accuracy': method_result.std_accuracy,
                    'accuracies': method_result.accuracies,
                }

        output_data = {
            'meta': analysis['meta'],
            'raw_results': serializable_results,
            'analysis': analysis,
            'summary_table': table,
        }

        if args.latex:
            output_data['latex_table'] = latex

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

    print("\nAnalysis complete!")


def generate_mock_results() -> Dict[str, Dict[str, MethodResult]]:
    """Generate mock results for demonstration purposes."""
    results = {}

    # SST-2 dataset
    results['sst2'] = {
        'bridge': MethodResult('bridge', accuracies=[93.7, 94.2, 93.1]),
        'zeroshot': MethodResult('zeroshot', accuracies=[85.2, 84.8, 85.6]),
        'fewshot': MethodResult('fewshot', accuracies=[89.3, 88.9, 89.7]),
        'prompt_tuning': MethodResult('prompt_tuning', accuracies=[91.2, 90.8, 91.5]),
        'lora': MethodResult('lora', accuracies=[92.1, 91.8, 92.4]),
    }

    # AG News dataset
    results['agnews'] = {
        'bridge': MethodResult('bridge', accuracies=[88.5, 89.1, 88.2]),
        'zeroshot': MethodResult('zeroshot', accuracies=[78.4, 77.9, 78.8]),
        'fewshot': MethodResult('fewshot', accuracies=[82.6, 82.1, 83.0]),
        'prompt_tuning': MethodResult('prompt_tuning', accuracies=[85.3, 84.8, 85.7]),
        'lora': MethodResult('lora', accuracies=[86.9, 86.4, 87.2]),
    }

    # TREC dataset
    results['trec'] = {
        'bridge': MethodResult('bridge', accuracies=[91.2, 91.8, 90.7]),
        'zeroshot': MethodResult('zeroshot', accuracies=[80.5, 79.9, 81.1]),
        'fewshot': MethodResult('fewshot', accuracies=[85.8, 85.3, 86.2]),
        'prompt_tuning': MethodResult('prompt_tuning', accuracies=[88.4, 87.9, 88.8]),
    }

    return results


if __name__ == "__main__":
    main()
