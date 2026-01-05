#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Results Aggregation for LatentWire Paper
Compatible with Python 3.5+

This script:
1. Collects all JSON results from runs/
2. Aggregates across 3 seeds
3. Creates publication-ready tables
4. Generates statistical significance annotations
5. Creates execution gate decisions
6. Produces final RESULTS_SUMMARY.md
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try importing pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, CSV export disabled")


class ResultsAggregator:
    """Main class for aggregating experimental results."""

    def __init__(self, base_dir, output_dir):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard seeds used in experiments
        self.seeds = [42, 123, 456]

        # Datasets we evaluate on
        self.datasets = ["sst2", "agnews", "trec", "gsm8k", "banking77", "passkey"]

        # Models evaluated
        self.models = ["llama3.1-8b", "mistral-7b", "qwen2.5-7b", "gemma-2b"]

        # Experiment types
        self.exp_types = [
            "bridge", "prompt_tuning", "lora", "full_finetune",
            "linear_probe", "llmlingua", "reverse", "same_model",
            "zeroshot", "fewshot"
        ]

        # Initialize results storage
        self.raw_results = defaultdict(lambda: defaultdict(list))
        self.aggregated_results = {}
        self.significance_tests = {}

    def collect_results(self):
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

        print("Found {} result files".format(len(result_files)))

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
                print("  Warning: Could not parse {}: {}".format(result_file, e))

        print("\nCollected results for {} experimental conditions".format(len(self.raw_results)))

    def _parse_experiment_info(self, filepath, data):
        """Extract experiment information from filepath and data."""
        info = {}

        # Parse from filepath
        path_str = str(filepath).lower()

        # Determine experiment type
        for exp_type in self.exp_types:
            if exp_type in path_str:
                info["exp_type"] = exp_type
                break

        # Determine dataset
        for dataset in self.datasets:
            if dataset in path_str:
                info["dataset"] = dataset
                break

        # Extract seed
        if "_seed" in path_str:
            try:
                seed_part = path_str.split("_seed")[1].split("/")[0].split("_")[0]
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

        # Handle None/null results gracefully
        if results is None:
            results = {}

        info["accuracy"] = results.get("accuracy", results.get("acc", 0)) if results else 0
        info["f1"] = results.get("f1_score", results.get("f1", 0)) if results else 0
        info["latency_ms"] = results.get("latency_ms", results.get("inference_time_ms", 0)) if results else 0
        info["memory_mb"] = results.get("memory_mb", results.get("peak_memory_mb", 0)) if results else 0
        info["compression_ratio"] = results.get("compression_ratio", 1.0) if results else 1.0

        # Extract model information
        for model in self.models:
            model_clean = model.replace(".", "").replace("-", "")
            if model_clean in path_str:
                info["model"] = model
                break

        return info if "exp_type" in info and "dataset" in info else None

    def aggregate_across_seeds(self):
        """Compute mean and std across seeds."""
        print("\n" + "=" * 80)
        print("AGGREGATING ACROSS SEEDS")
        print("=" * 80)

        for key, metrics in self.raw_results.items():
            exp_type, dataset, model = key

            agg_key = "{}__{}_{}".format(exp_type, dataset, model)
            self.aggregated_results[agg_key] = {}

            for metric_name, values in metrics.items():
                if metric_name == "seed":
                    continue

                values = np.array([v for v in values if v is not None and v > 0])

                if len(values) > 0:
                    self.aggregated_results[agg_key]["{}_mean".format(metric_name)] = float(np.mean(values))
                    self.aggregated_results[agg_key]["{}_std".format(metric_name)] = float(np.std(values))
                    self.aggregated_results[agg_key]["{}_n".format(metric_name)] = int(len(values))

                    # Compute confidence intervals
                    if len(values) >= 2:
                        ci = self._compute_confidence_interval(values)
                        self.aggregated_results[agg_key]["{}_ci_low".format(metric_name)] = float(ci[0])
                        self.aggregated_results[agg_key]["{}_ci_high".format(metric_name)] = float(ci[1])

        print("Aggregated {} experimental conditions".format(len(self.aggregated_results)))

    def _compute_confidence_interval(self, data, confidence=0.95):
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

    def compute_statistical_significance(self):
        """Compute statistical significance between key comparisons."""
        print("\n" + "=" * 80)
        print("COMPUTING STATISTICAL SIGNIFICANCE")
        print("=" * 80)

        comparisons = [
            ("bridge", "prompt_tuning", "Bridge vs Prompt-Tuning"),
            ("bridge", "lora", "Bridge vs LoRA"),
            ("bridge", "linear_probe", "Bridge vs Linear Probe"),
            ("bridge", "llmlingua", "Bridge vs LLMLingua"),
            ("bridge", "same_model", "Cross-Model vs Same-Model"),
            ("bridge", "reverse", "Forward vs Reverse Direction"),
            ("fewshot", "zeroshot", "Few-Shot vs Zero-Shot")
        ]

        for exp1, exp2, name in comparisons:
            for dataset in self.datasets:
                key1_pattern = "{}__{}".format(exp1, dataset)
                key2_pattern = "{}__{}".format(exp2, dataset)

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

                        test_key = "{}_{}".format(name, dataset)
                        self.significance_tests[test_key] = {
                            "mean1": float(np.mean(acc1)),
                            "std1": float(np.std(acc1)),
                            "mean2": float(np.mean(acc2)),
                            "std2": float(np.std(acc2)),
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant_0.05": bool(p_value < 0.05),
                            "significant_0.01": bool(p_value < 0.01),
                            "significant_0.001": bool(p_value < 0.001)
                        }

        print("Computed {} significance tests".format(len(self.significance_tests)))

    def generate_execution_gates(self):
        """Generate execution gate decisions based on results."""
        gates = {}

        # Gate 1: Bridge significantly beats prompt-tuning baseline
        gate1_pass = True
        for dataset in ["sst2", "agnews", "trec"]:
            test_key = "Bridge vs Prompt-Tuning_{}".format(dataset)
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
            bridge_key = "bridge__{}_default".format(dataset)
            if bridge_key in self.aggregated_results:
                acc = self.aggregated_results[bridge_key].get("accuracy_mean", 0)
                if acc > 80:
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

        avg_compression = 1.0
        if compression_ratios:
            avg_compression = np.mean(compression_ratios)
            gate3_pass = avg_compression >= 4.0

        gates["gate3_compression_achieved"] = {
            "passed": gate3_pass,
            "description": "Average compression ratio >= 4x",
            "recommendation": "PROCEED" if gate3_pass else "OPTIMIZE",
            "current_ratio": avg_compression
        }

        # Gate 4: Latency improvement
        gate4_pass = False
        latency_improvements = []
        for dataset in self.datasets:
            bridge_key = "bridge__{}_default".format(dataset)
            baseline_key = "zeroshot__{}_default".format(dataset)

            if bridge_key in self.aggregated_results and baseline_key in self.aggregated_results:
                bridge_latency = self.aggregated_results[bridge_key].get("latency_ms_mean", float('inf'))
                baseline_latency = self.aggregated_results[baseline_key].get("latency_ms_mean", float('inf'))

                if baseline_latency > 0:
                    improvement = (baseline_latency - bridge_latency) / baseline_latency
                    latency_improvements.append(improvement)

        avg_improvement = 0
        if latency_improvements:
            avg_improvement = np.mean(latency_improvements)
            gate4_pass = avg_improvement > 0.2

        gates["gate4_latency_improved"] = {
            "passed": gate4_pass,
            "description": "Average latency reduction > 20%",
            "recommendation": "PROCEED" if gate4_pass else "ACCEPTABLE",
            "current_improvement": avg_improvement
        }

        return gates

    def create_markdown_table(self, subset="main"):
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
                key = "{}__{}_default".format(method_key, dataset)

                if key in self.aggregated_results:
                    acc_mean = self.aggregated_results[key].get("accuracy_mean", 0)
                    acc_std = self.aggregated_results[key].get("accuracy_std", 0)
                    n = self.aggregated_results[key].get("accuracy_n", 0)

                    # Add significance markers
                    sig_marker = ""
                    test_key = "Bridge vs {}_{}".format(method_name.replace(' ', '-'), dataset)
                    if test_key in self.significance_tests:
                        if self.significance_tests[test_key]["significant_0.001"]:
                            sig_marker = "***"
                        elif self.significance_tests[test_key]["significant_0.01"]:
                            sig_marker = "**"
                        elif self.significance_tests[test_key]["significant_0.05"]:
                            sig_marker = "*"

                    if n >= 3:
                        cell = "{:.1f}±{:.1f}{}".format(acc_mean, acc_std, sig_marker)
                    else:
                        cell = "{:.1f}{}".format(acc_mean, sig_marker)

                    row.append(cell)
                    accuracies.append(acc_mean)
                else:
                    row.append("-")

            # Average
            if accuracies:
                avg = np.mean(accuracies)
                row.append("{:.1f}".format(avg))
            else:
                row.append("-")

            rows.append("| {} |".format(" | ".join(row)))

        return "\n".join(rows)

    def create_summary_report(self):
        """Create comprehensive summary report."""
        report = []

        # Header
        report.append("# TELEPATHY BRIDGE - RESULTS SUMMARY")
        report.append("\nGenerated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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

        if best_config:
            report.append("**Best Configuration**: {}".format(best_config))
            report.append("**Best Accuracy**: {:.1f}%".format(best_acc))

        # Execution Gates
        report.append("\n## Execution Gate Decisions\n")
        gates = self.generate_execution_gates()

        for gate_name, gate_info in gates.items():
            status = "✅ PASSED" if gate_info["passed"] else "❌ FAILED"
            report.append("\n### {}".format(gate_name.replace('_', ' ').title()))
            report.append("- **Status**: {}".format(status))
            report.append("- **Description**: {}".format(gate_info['description']))
            report.append("- **Recommendation**: {}".format(gate_info['recommendation']))
            if "current_ratio" in gate_info:
                report.append("- **Current Value**: {:.2f}x".format(gate_info['current_ratio']))
            if "current_improvement" in gate_info:
                report.append("- **Current Improvement**: {:.1%}".format(gate_info['current_improvement']))

        # Main Results Table
        report.append("\n## Main Results\n")
        report.append(self.create_markdown_table("main"))

        # Statistical Significance
        report.append("\n## Statistical Significance Tests\n")

        for test_name, test_results in sorted(self.significance_tests.items()):
            report.append("\n### {}".format(test_name))
            report.append("- Mean 1: {:.1f}% ± {:.1f}%".format(
                test_results['mean1'], test_results['std1']))
            report.append("- Mean 2: {:.1f}% ± {:.1f}%".format(
                test_results['mean2'], test_results['std2']))
            report.append("- t-statistic: {:.2f}".format(test_results['t_statistic']))
            report.append("- p-value: {:.2e}".format(test_results['p_value']))

            sig = ""
            if test_results['significant_0.001']:
                sig = "*** (p < 0.001)"
            elif test_results['significant_0.01']:
                sig = "** (p < 0.01)"
            elif test_results['significant_0.05']:
                sig = "* (p < 0.05)"
            else:
                sig = "n.s."
            report.append("- **Significance**: {}".format(sig))

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

            if not gates.get("gate1_sender_necessary", {}).get("passed", False):
                report.append("1. **Sender Model Impact**: Bridge not significantly better than prompt-tuning")
                report.append("   - Investigate encoder architecture")
                report.append("   - Increase training data diversity")

            if not gates.get("gate2_cross_model_transfer", {}).get("passed", False):
                report.append("2. **Cross-Model Transfer**: Accuracy below 80% threshold")
                report.append("   - Fine-tune adapter layers")
                report.append("   - Experiment with different latent dimensions")

            if not gates.get("gate3_compression_achieved", {}).get("passed", False):
                report.append("3. **Compression Ratio**: Below 4x target")
                report.append("   - Reduce latent sequence length")
                report.append("   - Implement quantization techniques")

            if not gates.get("gate4_latency_improved", {}).get("passed", False):
                report.append("4. **Latency**: Insufficient improvement")
                report.append("   - Optimize inference pipeline")
                report.append("   - Implement caching strategies")

        return "\n".join(report)

    def save_all_outputs(self):
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
            "raw_results": {str(k): v for k, v in self.raw_results.items()},
            "aggregated_results": self.aggregated_results,
            "significance_tests": self.significance_tests,
            "execution_gates": self.generate_execution_gates()
        }

        json_path = self.output_dir / "aggregated_results.json"
        with open(str(json_path), "w") as f:
            json.dump(json_output, f, indent=2, default=str)
        print("  ✓ Saved JSON: {}".format(json_path))

        # Save summary report
        report = self.create_summary_report()
        report_path = self.output_dir / "RESULTS_SUMMARY.md"
        with open(str(report_path), "w") as f:
            f.write(report)
        print("  ✓ Saved Report: {}".format(report_path))

        # Save significance tests separately
        sig_path = self.output_dir / "significance_tests.json"
        with open(str(sig_path), "w") as f:
            json.dump(self.significance_tests, f, indent=2)
        print("  ✓ Saved Significance: {}".format(sig_path))

        # Create CSV for easy analysis (if pandas available)
        if HAS_PANDAS and self.aggregated_results:
            rows = []
            for key, metrics in self.aggregated_results.items():
                row = {"configuration": key}
                row.update(metrics)
                rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = self.output_dir / "results_table.csv"
            df.to_csv(str(csv_path), index=False)
            print("  ✓ Saved CSV: {}".format(csv_path))

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


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experimental results for Telepathy Bridge paper"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="runs",
        help="Base directory containing experimental results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="telepathy/results",
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
    print("\nResults saved to: {}".format(args.output_dir))
    print("  - RESULTS_SUMMARY.md : Main summary report")
    print("  - aggregated_results.json : Complete JSON data")
    print("  - significance_tests.json : Statistical test results")
    if HAS_PANDAS:
        print("  - results_table.csv : CSV for further analysis")

    # Print execution gate summary
    gates = aggregator.generate_execution_gates()
    print("\n" + "=" * 80)
    print("EXECUTION GATES SUMMARY")
    print("=" * 80)

    for gate_name, gate_info in gates.items():
        status = "✅" if gate_info["passed"] else "❌"
        print("{} {}: {}".format(status, gate_name, gate_info['recommendation']))

    all_passed = all(g["passed"] for g in gates.values())
    if all_passed:
        print("\n🎉 All gates passed! Ready for production deployment.")
    else:
        print("\n⚠️  Some gates need attention. See RESULTS_SUMMARY.md for details.")


if __name__ == "__main__":
    main()