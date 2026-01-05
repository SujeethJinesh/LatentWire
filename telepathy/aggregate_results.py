#!/usr/bin/env python
"""
Comprehensive Results Aggregation for LatentWire Paper

This script:
1. Collects all JSON results from runs/
2. Aggregates across 3 seeds (42, 123, 456)
3. Creates publication-ready tables (Markdown + LaTeX)
4. Generates statistical significance annotations
5. Creates execution gate decisions
6. Produces final RESULTS_SUMMARY.md

Author: LatentWire Team
Date: January 2025
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ResultsAggregator:
    """Main class for aggregating experimental results."""

    def __init__(self, base_dir: Path, output_dir: Path):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard seeds used in experiments
        self.seeds = [42, 123, 456]

        # Datasets we evaluate on
        self.datasets = ["sst2", "agnews", "trec", "gsm8k", "banking77", "passkey"]

        # Models evaluated
        self.models = ["llama3.1-8b", "mistral-7b", "qwen2.5-7b", "gemma-2b"]

        # Experiment types
        self.exp_types = [
            "bridge",
            "prompt_tuning",
            "lora",
            "full_finetune",
            "linear_probe",
            "llmlingua",
            "reverse",
            "same_model",
            "zeroshot",
            "fewshot"
        ]

        # Initialize results storage
        self.raw_results = defaultdict(lambda: defaultdict(list))
        self.aggregated_results = {}
        self.significance_tests = {}

    def collect_results(self) -> None:
        """Collect all JSON results from runs directory."""
        print("=" * 80)
        print("COLLECTING EXPERIMENTAL RESULTS")
        print("=" * 80)

        # Pattern matching for different result files
        patterns = [
            "**/*_results.json",
            "**/results.json",
            "**/eval_results.json",
            "**/final_results.json"
        ]

        result_files = []
        for pattern in patterns:
            result_files.extend(self.base_dir.glob(pattern))

        print(f"Found {len(result_files)} result files")

        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)

                # Extract experiment metadata from path and content
                exp_info = self._parse_experiment_info(result_file, data)

                if exp_info:
                    key = (exp_info["exp_type"], exp_info["dataset"], exp_info.get("model", "default"))
                    self.raw_results[key]["seed"].append(exp_info.get("seed", 42))
                    self.raw_results[key]["accuracy"].append(exp_info.get("accuracy", 0))
                    self.raw_results[key]["f1"].append(exp_info.get("f1", 0))
                    self.raw_results[key]["latency_ms"].append(exp_info.get("latency_ms", 0))
                    self.raw_results[key]["memory_mb"].append(exp_info.get("memory_mb", 0))
                    self.raw_results[key]["compression_ratio"].append(exp_info.get("compression_ratio", 1.0))

            except Exception as e:
                print(f"  Warning: Could not parse {result_file}: {e}")

        print(f"\nCollected results for {len(self.raw_results)} experimental conditions")

    def _parse_experiment_info(self, filepath: Path, data: dict) -> Optional[dict]:
        """Extract experiment information from filepath and data."""
        info = {}

        # Parse from filepath
        path_parts = filepath.parts

        # Determine experiment type
        for exp_type in self.exp_types:
            if exp_type in str(filepath).lower():
                info["exp_type"] = exp_type
                break

        # Determine dataset
        for dataset in self.datasets:
            if dataset in str(filepath).lower():
                info["dataset"] = dataset
                break

        # Extract seed
        if "_seed" in str(filepath):
            try:
                seed_part = str(filepath).split("_seed")[1].split("/")[0].split("_")[0]
                info["seed"] = int(seed_part)
            except:
                info["seed"] = 42

        # Extract metrics from data
        if "final_results" in data:
            results = data["final_results"]
        elif "results" in data:
            results = data["results"]
        else:
            results = data

        info["accuracy"] = results.get("accuracy", results.get("acc", 0))
        info["f1"] = results.get("f1_score", results.get("f1", 0))
        info["latency_ms"] = results.get("latency_ms", results.get("inference_time_ms", 0))
        info["memory_mb"] = results.get("memory_mb", results.get("peak_memory_mb", 0))
        info["compression_ratio"] = results.get("compression_ratio", 1.0)

        # Extract model information
        for model in self.models:
            if model.replace(".", "").replace("-", "") in str(filepath).lower():
                info["model"] = model
                break

        return info if "exp_type" in info and "dataset" in info else None

    def aggregate_across_seeds(self) -> None:
        """Compute mean and std across seeds."""
        print("\n" + "=" * 80)
        print("AGGREGATING ACROSS SEEDS")
        print("=" * 80)

        for key, metrics in self.raw_results.items():
            exp_type, dataset, model = key

            agg_key = f"{exp_type}_{dataset}_{model}"
            self.aggregated_results[agg_key] = {}

            for metric_name, values in metrics.items():
                if metric_name == "seed":
                    continue

                values = np.array([v for v in values if v is not None and v > 0])

                if len(values) > 0:
                    self.aggregated_results[agg_key][f"{metric_name}_mean"] = float(np.mean(values))
                    self.aggregated_results[agg_key][f"{metric_name}_std"] = float(np.std(values))
                    self.aggregated_results[agg_key][f"{metric_name}_n"] = int(len(values))

                    # Compute confidence intervals
                    if len(values) >= 2:
                        ci = self._compute_confidence_interval(values)
                        self.aggregated_results[agg_key][f"{metric_name}_ci_low"] = float(ci[0])
                        self.aggregated_results[agg_key][f"{metric_name}_ci_high"] = float(ci[1])

        print(f"Aggregated {len(self.aggregated_results)} experimental conditions")

    def _compute_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval using bootstrap."""
        n_bootstrap = 10000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return lower, upper

    def compute_statistical_significance(self) -> None:
        """Compute statistical significance between key comparisons."""
        print("\n" + "=" * 80)
        print("COMPUTING STATISTICAL SIGNIFICANCE")
        print("=" * 80)

        comparisons = [
            # Bridge vs baselines
            ("bridge", "prompt_tuning", "Bridge vs Prompt-Tuning"),
            ("bridge", "lora", "Bridge vs LoRA"),
            ("bridge", "linear_probe", "Bridge vs Linear Probe"),
            ("bridge", "llmlingua", "Bridge vs LLMLingua"),

            # Cross-model vs same-model
            ("bridge", "same_model", "Cross-Model vs Same-Model"),

            # Direction comparison
            ("bridge", "reverse", "Forward vs Reverse Direction"),

            # Few-shot vs zero-shot
            ("fewshot", "zeroshot", "Few-Shot vs Zero-Shot")
        ]

        for exp1, exp2, name in comparisons:
            for dataset in self.datasets:
                key1_pattern = f"{exp1}_{dataset}_"
                key2_pattern = f"{exp2}_{dataset}_"

                # Find matching keys
                keys1 = [k for k in self.aggregated_results if k.startswith(key1_pattern)]
                keys2 = [k for k in self.aggregated_results if k.startswith(key2_pattern)]

                if keys1 and keys2:
                    # Get accuracy values
                    acc1 = []
                    acc2 = []

                    for key in keys1:
                        if "accuracy_mean" in self.aggregated_results[key]:
                            acc1.append(self.aggregated_results[key]["accuracy_mean"])

                    for key in keys2:
                        if "accuracy_mean" in self.aggregated_results[key]:
                            acc2.append(self.aggregated_results[key]["accuracy_mean"])

                    if acc1 and acc2:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(acc1, acc2)

                        test_key = f"{name}_{dataset}"
                        self.significance_tests[test_key] = {
                            "mean1": float(np.mean(acc1)),
                            "std1": float(np.std(acc1)),
                            "mean2": float(np.mean(acc2)),
                            "std2": float(np.std(acc2)),
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant_0.05": p_value < 0.05,
                            "significant_0.01": p_value < 0.01,
                            "significant_0.001": p_value < 0.001
                        }

        print(f"Computed {len(self.significance_tests)} significance tests")

    def generate_execution_gates(self) -> Dict[str, dict]:
        """Generate execution gate decisions based on results."""
        gates = {}

        # Gate 1: Bridge significantly beats prompt-tuning baseline
        gate1_pass = True
        for dataset in ["sst2", "agnews", "trec"]:
            test_key = f"Bridge vs Prompt-Tuning_{dataset}"
            if test_key in self.significance_tests:
                if not self.significance_tests[test_key]["significant_0.05"]:
                    gate1_pass = False

        gates["gate1_sender_necessary"] = {
            "passed": gate1_pass,
            "description": "Bridge must significantly outperform prompt-tuning baseline",
            "recommendation": "PROCEED" if gate1_pass else "INVESTIGATE"
        }

        # Gate 2: Cross-model transfer works
        gate2_pass = False
        for dataset in self.datasets:
            bridge_key = f"bridge_{dataset}_default"
            if bridge_key in self.aggregated_results:
                acc = self.aggregated_results[bridge_key].get("accuracy_mean", 0)
                if acc > 80:  # 80% threshold for cross-model success
                    gate2_pass = True
                    break

        gates["gate2_cross_model_transfer"] = {
            "passed": gate2_pass,
            "description": "Cross-model transfer achieves >80% accuracy on at least one dataset",
            "recommendation": "PROCEED" if gate2_pass else "REFINE"
        }

        # Gate 3: Compression is meaningful
        gate3_pass = False
        compression_ratios = []
        for key in self.aggregated_results:
            if "compression_ratio_mean" in self.aggregated_results[key]:
                compression_ratios.append(self.aggregated_results[key]["compression_ratio_mean"])

        if compression_ratios:
            avg_compression = np.mean(compression_ratios)
            gate3_pass = avg_compression >= 4.0  # 4x compression target

        gates["gate3_compression_achieved"] = {
            "passed": gate3_pass,
            "description": "Average compression ratio ≥ 4x",
            "recommendation": "PROCEED" if gate3_pass else "OPTIMIZE",
            "current_ratio": avg_compression if compression_ratios else 1.0
        }

        # Gate 4: Latency improvement
        gate4_pass = False
        latency_improvements = []
        for dataset in self.datasets:
            bridge_key = f"bridge_{dataset}_default"
            baseline_key = f"zeroshot_{dataset}_default"

            if bridge_key in self.aggregated_results and baseline_key in self.aggregated_results:
                bridge_latency = self.aggregated_results[bridge_key].get("latency_ms_mean", float('inf'))
                baseline_latency = self.aggregated_results[baseline_key].get("latency_ms_mean", float('inf'))

                if baseline_latency > 0:
                    improvement = (baseline_latency - bridge_latency) / baseline_latency
                    latency_improvements.append(improvement)

        if latency_improvements:
            avg_improvement = np.mean(latency_improvements)
            gate4_pass = avg_improvement > 0.2  # 20% latency reduction

        gates["gate4_latency_improved"] = {
            "passed": gate4_pass,
            "description": "Average latency reduction > 20%",
            "recommendation": "PROCEED" if gate4_pass else "ACCEPTABLE",
            "current_improvement": avg_improvement if latency_improvements else 0
        }

        return gates

    def create_markdown_table(self, subset: str = "main") -> str:
        """Create markdown table for results."""
        rows = []

        # Header
        if subset == "main":
            rows.append("| Method | SST-2 | AG News | TREC | GSM8K | Avg |")
            rows.append("|--------|-------|---------|------|-------|-----|")

            datasets = ["sst2", "agnews", "trec", "gsm8k"]
        else:
            rows.append("| Method | Banking77 | PassKey | SST-2 | AG News | TREC |")
            rows.append("|--------|-----------|---------|-------|---------|------|")

            datasets = ["banking77", "passkey", "sst2", "agnews", "trec"]

        # Methods to include
        methods = [
            ("bridge", "Telepathy Bridge"),
            ("prompt_tuning", "Prompt-Tuning"),
            ("lora", "LoRA"),
            ("linear_probe", "Linear Probe"),
            ("llmlingua", "LLMLingua"),
            ("same_model", "Same-Model"),
            ("zeroshot", "Zero-Shot"),
            ("fewshot", "Few-Shot (3)")
        ]

        for method_key, method_name in methods:
            row = [method_name]
            accuracies = []

            for dataset in datasets:
                key = f"{method_key}_{dataset}_default"

                if key in self.aggregated_results:
                    acc_mean = self.aggregated_results[key].get("accuracy_mean", 0)
                    acc_std = self.aggregated_results[key].get("accuracy_std", 0)
                    n = self.aggregated_results[key].get("accuracy_n", 0)

                    # Add significance markers
                    sig_marker = ""
                    test_key = f"Bridge vs {method_name.replace(' ', '-')}_{dataset}"
                    if test_key in self.significance_tests:
                        if self.significance_tests[test_key]["significant_0.001"]:
                            sig_marker = "***"
                        elif self.significance_tests[test_key]["significant_0.01"]:
                            sig_marker = "**"
                        elif self.significance_tests[test_key]["significant_0.05"]:
                            sig_marker = "*"

                    if n >= 3:
                        row.append(f"{acc_mean:.1f}±{acc_std:.1f}{sig_marker}")
                    else:
                        row.append(f"{acc_mean:.1f}{sig_marker}")

                    accuracies.append(acc_mean)
                else:
                    row.append("-")

            # Average
            if accuracies:
                avg = np.mean(accuracies)
                row.append(f"{avg:.1f}")
            else:
                row.append("-")

            rows.append(f"| {' | '.join(row)} |")

        return "\n".join(rows)

    def create_latex_table(self, subset: str = "main") -> str:
        """Create LaTeX table for paper."""
        lines = []

        # Table setup
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")

        if subset == "main":
            lines.append("\\begin{tabular}{lccccc}")
            lines.append("\\toprule")
            lines.append("Method & SST-2 & AG News & TREC & GSM8K & Avg \\\\")
            datasets = ["sst2", "agnews", "trec", "gsm8k"]
        else:
            lines.append("\\begin{tabular}{lcccccc}")
            lines.append("\\toprule")
            lines.append("Method & Banking77 & PassKey & SST-2 & AG News & TREC \\\\")
            datasets = ["banking77", "passkey", "sst2", "agnews", "trec"]

        lines.append("\\midrule")

        # Methods
        methods = [
            ("bridge", "\\textbf{Telepathy Bridge}"),
            ("prompt_tuning", "Prompt-Tuning"),
            ("lora", "LoRA"),
            ("linear_probe", "Linear Probe"),
            ("llmlingua", "LLMLingua"),
            ("same_model", "Same-Model"),
            ("zeroshot", "Zero-Shot"),
            ("fewshot", "Few-Shot (3)")
        ]

        best_per_dataset = {dataset: 0 for dataset in datasets}

        # First pass to find best scores
        for method_key, _ in methods:
            for dataset in datasets:
                key = f"{method_key}_{dataset}_default"
                if key in self.aggregated_results:
                    acc = self.aggregated_results[key].get("accuracy_mean", 0)
                    best_per_dataset[dataset] = max(best_per_dataset[dataset], acc)

        # Second pass to create rows
        for method_key, method_name in methods:
            row = [method_name]
            accuracies = []

            for dataset in datasets:
                key = f"{method_key}_{dataset}_default"

                if key in self.aggregated_results:
                    acc_mean = self.aggregated_results[key].get("accuracy_mean", 0)
                    acc_std = self.aggregated_results[key].get("accuracy_std", 0)
                    n = self.aggregated_results[key].get("accuracy_n", 0)

                    # Bold if best
                    is_best = abs(acc_mean - best_per_dataset[dataset]) < 0.1

                    # Significance markers
                    sig_marker = ""
                    if method_key != "bridge":
                        test_key = f"Bridge vs {method_name.replace('\\textbf{', '').replace('}', '')}_{dataset}"
                        if test_key in self.significance_tests:
                            p = self.significance_tests[test_key]["p_value"]
                            if p < 0.001:
                                sig_marker = "^{***}"
                            elif p < 0.01:
                                sig_marker = "^{**}"
                            elif p < 0.05:
                                sig_marker = "^{*}"

                    if n >= 3:
                        val = f"{acc_mean:.1f}$\\pm${acc_std:.1f}{sig_marker}"
                    else:
                        val = f"{acc_mean:.1f}{sig_marker}"

                    if is_best:
                        val = f"\\textbf{{{val}}}"

                    row.append(val)
                    accuracies.append(acc_mean)
                else:
                    row.append("-")

            # Average
            if accuracies:
                avg = np.mean(accuracies)
                row.append(f"{avg:.1f}")
            else:
                row.append("-")

            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Performance comparison across methods and datasets. ")
        lines.append("* p<0.05, ** p<0.01, *** p<0.001 vs Telepathy Bridge.}")
        lines.append("\\label{tab:main_results}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def create_summary_report(self) -> str:
        """Create comprehensive summary report."""
        report = []

        # Header
        report.append("# TELEPATHY BRIDGE - RESULTS SUMMARY")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "=" * 80)

        # Executive Summary
        report.append("\n## Executive Summary\n")

        # Find best performing configuration
        best_acc = 0
        best_config = None
        for key, results in self.aggregated_results.items():
            if "accuracy_mean" in results and results["accuracy_mean"] > best_acc:
                best_acc = results["accuracy_mean"]
                best_config = key

        report.append(f"**Best Configuration**: {best_config}")
        report.append(f"**Best Accuracy**: {best_acc:.1f}%")

        # Execution Gates
        report.append("\n## Execution Gate Decisions\n")
        gates = self.generate_execution_gates()

        for gate_name, gate_info in gates.items():
            status = "✅ PASSED" if gate_info["passed"] else "❌ FAILED"
            report.append(f"\n### {gate_name.replace('_', ' ').title()}")
            report.append(f"- **Status**: {status}")
            report.append(f"- **Description**: {gate_info['description']}")
            report.append(f"- **Recommendation**: {gate_info['recommendation']}")
            if "current_ratio" in gate_info:
                report.append(f"- **Current Value**: {gate_info['current_ratio']:.2f}x")
            if "current_improvement" in gate_info:
                report.append(f"- **Current Improvement**: {gate_info['current_improvement']:.1%}")

        # Main Results Table
        report.append("\n## Main Results\n")
        report.append(self.create_markdown_table("main"))

        # Statistical Significance
        report.append("\n## Statistical Significance Tests\n")

        for test_name, test_results in sorted(self.significance_tests.items()):
            report.append(f"\n### {test_name}")
            report.append(f"- Mean 1: {test_results['mean1']:.1f}% ± {test_results['std1']:.1f}%")
            report.append(f"- Mean 2: {test_results['mean2']:.1f}% ± {test_results['std2']:.1f}%")
            report.append(f"- t-statistic: {test_results['t_statistic']:.2f}")
            report.append(f"- p-value: {test_results['p_value']:.2e}")

            sig = ""
            if test_results['significant_0.001']:
                sig = "*** (p < 0.001)"
            elif test_results['significant_0.01']:
                sig = "** (p < 0.01)"
            elif test_results['significant_0.05']:
                sig = "* (p < 0.05)"
            else:
                sig = "n.s."
            report.append(f"- **Significance**: {sig}")

        # Performance Metrics
        report.append("\n## Performance Metrics\n")

        # Compression ratios
        report.append("\n### Compression Ratios")
        compression_data = []
        for key, results in self.aggregated_results.items():
            if "compression_ratio_mean" in results:
                compression_data.append((key, results["compression_ratio_mean"]))

        if compression_data:
            compression_data.sort(key=lambda x: x[1], reverse=True)
            for config, ratio in compression_data[:5]:
                report.append(f"- {config}: {ratio:.2f}x")

        # Latency
        report.append("\n### Inference Latency (ms)")
        latency_data = []
        for key, results in self.aggregated_results.items():
            if "latency_ms_mean" in results:
                latency_data.append((key, results["latency_ms_mean"]))

        if latency_data:
            latency_data.sort(key=lambda x: x[1])
            for config, latency in latency_data[:5]:
                report.append(f"- {config}: {latency:.1f}ms")

        # Memory usage
        report.append("\n### Peak Memory Usage (MB)")
        memory_data = []
        for key, results in self.aggregated_results.items():
            if "memory_mb_mean" in results:
                memory_data.append((key, results["memory_mb_mean"]))

        if memory_data:
            memory_data.sort(key=lambda x: x[1])
            for config, memory in memory_data[:5]:
                report.append(f"- {config}: {memory:.0f}MB")

        # Recommendations
        report.append("\n## Recommendations\n")

        all_gates_passed = all(g["passed"] for g in gates.values())

        if all_gates_passed:
            report.append("✅ **All execution gates passed!**")
            report.append("\nThe Telepathy Bridge demonstrates:")
            report.append("1. Significant improvement over baselines")
            report.append("2. Successful cross-model transfer")
            report.append("3. Meaningful compression ratios")
            report.append("4. Improved inference latency")
            report.append("\n**Next Steps**: Proceed to production deployment planning")
        else:
            report.append("⚠️ **Some execution gates require attention**")
            report.append("\nPriority improvements needed:")

            if not gates["gate1_sender_necessary"]["passed"]:
                report.append("1. **Sender Model Impact**: Bridge not significantly better than prompt-tuning")
                report.append("   - Investigate encoder architecture")
                report.append("   - Increase training data diversity")

            if not gates["gate2_cross_model_transfer"]["passed"]:
                report.append("2. **Cross-Model Transfer**: Accuracy below 80% threshold")
                report.append("   - Fine-tune adapter layers")
                report.append("   - Experiment with different latent dimensions")

            if not gates["gate3_compression_achieved"]["passed"]:
                report.append("3. **Compression Ratio**: Below 4x target")
                report.append("   - Reduce latent sequence length")
                report.append("   - Implement quantization techniques")

            if not gates["gate4_latency_improved"]["passed"]:
                report.append("4. **Latency**: Insufficient improvement")
                report.append("   - Optimize inference pipeline")
                report.append("   - Implement caching strategies")

        # LaTeX table for paper
        report.append("\n## LaTeX Table for Paper\n")
        report.append("```latex")
        report.append(self.create_latex_table("main"))
        report.append("```")

        return "\n".join(report)

    def save_all_outputs(self) -> None:
        """Save all aggregated outputs."""
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)

        # Save raw JSON data
        json_output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "n_experiments": len(self.raw_results),
                "n_aggregated": len(self.aggregated_results),
                "seeds": self.seeds,
                "datasets": self.datasets,
                "models": self.models
            },
            "raw_results": dict(self.raw_results),
            "aggregated_results": self.aggregated_results,
            "significance_tests": self.significance_tests,
            "execution_gates": self.generate_execution_gates()
        }

        json_path = self.output_dir / "aggregated_results.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2, default=str)
        print(f"  ✓ Saved JSON: {json_path}")

        # Save summary report
        report = self.create_summary_report()
        report_path = self.output_dir / "RESULTS_SUMMARY.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  ✓ Saved Report: {report_path}")

        # Save LaTeX tables
        latex_main = self.create_latex_table("main")
        latex_path = self.output_dir / "results_table_main.tex"
        with open(latex_path, "w") as f:
            f.write(latex_main)
        print(f"  ✓ Saved LaTeX: {latex_path}")

        # Save significance tests separately
        sig_path = self.output_dir / "significance_tests.json"
        with open(sig_path, "w") as f:
            json.dump(self.significance_tests, f, indent=2)
        print(f"  ✓ Saved Significance: {sig_path}")

        # Create CSV for easy analysis
        if self.aggregated_results:
            rows = []
            for key, metrics in self.aggregated_results.items():
                row = {"configuration": key}
                row.update(metrics)
                rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = self.output_dir / "results_table.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experimental results for Telepathy Bridge paper"
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("runs"),
        help="Base directory containing experimental results"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("telepathy/results"),
        help="Output directory for aggregated results"
    )
    parser.add_argument(
        "--use_mock_data",
        action="store_true",
        help="Use mock data for testing (when no real results available)"
    )

    args = parser.parse_args()

    # Initialize aggregator
    aggregator = ResultsAggregator(args.base_dir, args.output_dir)

    if args.use_mock_data:
        # Generate mock data for testing
        print("Using mock data for demonstration...")
        aggregator._generate_mock_data()
    else:
        # Collect real results
        aggregator.collect_results()

    # Process results
    aggregator.aggregate_across_seeds()
    aggregator.compute_statistical_significance()

    # Save outputs
    aggregator.save_all_outputs()

    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - RESULTS_SUMMARY.md : Main summary report")
    print(f"  - aggregated_results.json : Complete JSON data")
    print(f"  - results_table_main.tex : LaTeX table for paper")
    print(f"  - significance_tests.json : Statistical test results")
    print(f"  - results_table.csv : CSV for further analysis")

    # Print execution gate summary
    gates = aggregator.generate_execution_gates()
    print("\n" + "=" * 80)
    print("EXECUTION GATES SUMMARY")
    print("=" * 80)

    for gate_name, gate_info in gates.items():
        status = "✅" if gate_info["passed"] else "❌"
        print(f"{status} {gate_name}: {gate_info['recommendation']}")

    all_passed = all(g["passed"] for g in gates.values())
    if all_passed:
        print("\n🎉 All gates passed! Ready for production deployment.")
    else:
        print("\n⚠️  Some gates need attention. See RESULTS_SUMMARY.md for details.")


def _generate_mock_data(self):
    """Generate mock data for testing when no real results available."""
    # Mock data based on expected results from paper
    mock_configs = [
        # Bridge results (3 seeds each)
        ("bridge", "sst2", 42, 96.5, 0.95, 45, 1200, 4.2),
        ("bridge", "sst2", 123, 96.0, 0.94, 46, 1210, 4.1),
        ("bridge", "sst2", 456, 97.5, 0.96, 44, 1195, 4.3),
        ("bridge", "agnews", 42, 90.0, 0.89, 52, 1400, 3.8),
        ("bridge", "agnews", 123, 91.0, 0.90, 51, 1390, 3.9),
        ("bridge", "agnews", 456, 91.0, 0.90, 53, 1410, 3.7),
        ("bridge", "trec", 42, 95.0, 0.94, 38, 1100, 4.5),
        ("bridge", "trec", 123, 95.5, 0.95, 37, 1090, 4.6),
        ("bridge", "trec", 456, 95.5, 0.95, 39, 1110, 4.4),

        # Prompt-tuning baseline (3 seeds each)
        ("prompt_tuning", "sst2", 42, 49.5, 0.48, 55, 1500, 1.0),
        ("prompt_tuning", "sst2", 123, 49.5, 0.48, 56, 1510, 1.0),
        ("prompt_tuning", "sst2", 456, 49.5, 0.48, 54, 1490, 1.0),
        ("prompt_tuning", "agnews", 42, 30.5, 0.29, 62, 1700, 1.0),
        ("prompt_tuning", "agnews", 123, 14.5, 0.13, 63, 1710, 1.0),
        ("prompt_tuning", "agnews", 456, 14.5, 0.13, 61, 1690, 1.0),

        # LoRA baseline
        ("lora", "sst2", 42, 92.0, 0.91, 48, 1300, 1.0),
        ("lora", "sst2", 123, 91.5, 0.90, 49, 1310, 1.0),
        ("lora", "sst2", 456, 92.5, 0.91, 47, 1290, 1.0),

        # Linear probe
        ("linear_probe", "sst2", 42, 84.5, 0.83, 35, 900, 6.0),
        ("linear_probe", "sst2", 123, 85.0, 0.84, 34, 890, 6.1),
        ("linear_probe", "sst2", 456, 84.0, 0.82, 36, 910, 5.9),

        # Zero-shot
        ("zeroshot", "sst2", 42, 88.0, 0.87, 65, 1800, 1.0),
        ("zeroshot", "agnews", 42, 75.0, 0.74, 72, 2000, 1.0),
        ("zeroshot", "trec", 42, 82.0, 0.81, 58, 1600, 1.0),

        # Few-shot
        ("fewshot", "sst2", 42, 91.0, 0.90, 85, 2200, 1.0),
        ("fewshot", "agnews", 42, 82.0, 0.81, 92, 2400, 1.0),
        ("fewshot", "trec", 42, 88.0, 0.87, 78, 2000, 1.0),
    ]

    for exp_type, dataset, seed, acc, f1, latency, memory, compression in mock_configs:
        key = (exp_type, dataset, "default")
        self.raw_results[key]["seed"].append(seed)
        self.raw_results[key]["accuracy"].append(acc)
        self.raw_results[key]["f1"].append(f1)
        self.raw_results[key]["latency_ms"].append(latency)
        self.raw_results[key]["memory_mb"].append(memory)
        self.raw_results[key]["compression_ratio"].append(compression)


# Add the mock data generation method to the class
ResultsAggregator._generate_mock_data = _generate_mock_data


if __name__ == "__main__":
    main()