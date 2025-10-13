"""
Comprehensive Analysis - Compare All Experiment Results

Aggregates and compares results from all experiments:
- Text baseline (upper bound)
- Token budget baseline (fair comparison)
- PCA baseline (linear compression)
- LatentWire (learned compression)

Generates comparison report showing:
- Performance hierarchy
- Compression effectiveness
- Key insights and conclusions

Usage:
  python scripts/analyze_all_results.py --results_dir runs/full_suite
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_eval_results(results_dir):
    """Load results.json from evaluation directory."""
    results_path = results_dir / 'results.json'
    if not results_path.exists():
        return None

    with open(results_path) as f:
        return json.load(f)


def extract_metrics(results):
    """Extract key metrics from results dict."""
    if results is None:
        return {'em': None, 'f1': None, 'status': 'missing'}

    metrics = {
        'em': results.get('em', results.get('exact_match')),
        'f1': results.get('f1', results.get('f1_score')),
        'num_examples': results.get('num_examples', results.get('n_examples')),
        'status': 'success',
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("Loading results...")
    print()

    # Load all results
    all_results = {}

    # Text baselines
    all_results['text_llama'] = load_eval_results(results_dir / 'baselines' / 'text_llama')
    all_results['text_qwen'] = load_eval_results(results_dir / 'baselines' / 'text_qwen')

    # Token budget baselines (find M value from directory name)
    token_budget_dirs = list((results_dir / 'baselines').glob('token_budget_*_m*'))
    for tb_dir in token_budget_dirs:
        if 'llama' in tb_dir.name:
            all_results['token_budget_llama'] = load_eval_results(tb_dir)
        elif 'qwen' in tb_dir.name:
            all_results['token_budget_qwen'] = load_eval_results(tb_dir)

    # PCA baseline
    pca_dirs = list((results_dir / 'baselines').glob('pca_m*'))
    if pca_dirs:
        all_results['pca'] = load_eval_results(pca_dirs[0])

    # LatentWire
    all_results['latentwire_llama'] = load_eval_results(results_dir / 'latentwire' / 'eval_llama')
    all_results['latentwire_qwen'] = load_eval_results(results_dir / 'latentwire' / 'eval_qwen')

    # Extract metrics
    metrics = {name: extract_metrics(res) for name, res in all_results.items()}

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE EXPERIMENT RESULTS")
    report.append("=" * 80)
    report.append("")
    report.append(f"Analysis timestamp: {datetime.now().isoformat()}")
    report.append(f"Results directory: {results_dir}")
    report.append("")

    # =============================================================================
    # RESULTS TABLE
    # =============================================================================
    report.append("=" * 80)
    report.append("RESULTS SUMMARY")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'Experiment':<35} {'EM':>8} {'F1':>8} {'Status':<10}")
    report.append("-" * 80)

    # Text baselines (upper bound)
    report.append("TEXT BASELINE (Upper Bound):")
    for name in ['text_llama', 'text_qwen']:
        if name in metrics and metrics[name]['status'] == 'success':
            m = metrics[name]
            model_name = "Llama" if 'llama' in name else "Qwen"
            em_str = f"{m['em']*100:6.2f}%" if m['em'] is not None else "    N/A"
            f1_str = f"{m['f1']*100:6.2f}%" if m['f1'] is not None else "    N/A"
            report.append(f"  {model_name:<33} {em_str:>8} {f1_str:>8} {m['status']:<10}")

    report.append("")

    # Token budget baselines (fair comparison)
    report.append("TOKEN BUDGET BASELINE (Fair Comparison):")
    for name in ['token_budget_llama', 'token_budget_qwen']:
        if name in metrics and metrics[name]['status'] == 'success':
            m = metrics[name]
            model_name = "Llama" if 'llama' in name else "Qwen"
            em_str = f"{m['em']*100:6.2f}%" if m['em'] is not None else "    N/A"
            f1_str = f"{m['f1']*100:6.2f}%" if m['f1'] is not None else "    N/A"
            report.append(f"  {model_name:<33} {em_str:>8} {f1_str:>8} {m['status']:<10}")

    report.append("")

    # PCA baseline (linear compression)
    report.append("PCA BASELINE (Linear Compression):")
    if 'pca' in metrics and metrics['pca']['status'] == 'success':
        m = metrics['pca']
        em_str = f"{m['em']*100:6.2f}%" if m['em'] is not None else "    N/A"
        f1_str = f"{m['f1']*100:6.2f}%" if m['f1'] is not None else "    N/A"
        report.append(f"  Llama (PCA-compressed){'':>10} {em_str:>8} {f1_str:>8} {m['status']:<10}")

    report.append("")

    # LatentWire (learned compression)
    report.append("LATENTWIRE (Learned Compression):")
    for name in ['latentwire_llama', 'latentwire_qwen']:
        if name in metrics and metrics[name]['status'] == 'success':
            m = metrics[name]
            model_name = "Llama" if 'llama' in name else "Qwen"
            em_str = f"{m['em']*100:6.2f}%" if m['em'] is not None else "    N/A"
            f1_str = f"{m['f1']*100:6.2f}%" if m['f1'] is not None else "    N/A"
            report.append(f"  {model_name:<33} {em_str:>8} {f1_str:>8} {m['status']:<10}")

    report.append("")

    # =============================================================================
    # KEY COMPARISONS
    # =============================================================================
    report.append("=" * 80)
    report.append("KEY COMPARISONS")
    report.append("=" * 80)
    report.append("")

    # Function to get F1 safely
    def get_f1(name):
        if name in metrics and metrics[name]['status'] == 'success':
            return metrics[name]['f1']
        return None

    # Comparison 1: LatentWire vs Token Budget (CRITICAL)
    report.append("1. LatentWire vs Token Budget (same M) - CRITICAL TEST")
    report.append("   Does learned compression beat simple truncation?")
    report.append("")

    lw_llama_f1 = get_f1('latentwire_llama')
    tb_llama_f1 = get_f1('token_budget_llama')

    if lw_llama_f1 is not None and tb_llama_f1 is not None:
        diff = lw_llama_f1 - tb_llama_f1
        pct_improvement = (diff / tb_llama_f1) * 100 if tb_llama_f1 > 0 else 0

        report.append(f"   Llama:")
        report.append(f"     LatentWire:   F1 = {lw_llama_f1*100:.2f}%")
        report.append(f"     Token Budget: F1 = {tb_llama_f1*100:.2f}%")
        report.append(f"     Difference:   {diff*100:+.2f}% ({pct_improvement:+.1f}% relative)")

        if diff > 0.05:  # 5% absolute improvement
            report.append(f"     ✓ SUCCESS: Learned compression beats simple truncation!")
        elif diff > 0:
            report.append(f"     ⚠ MARGINAL: Small improvement, may need better training")
        else:
            report.append(f"     ✗ FAILURE: Not learning to compress effectively")
    else:
        report.append("   Data not available for comparison")

    report.append("")

    # Comparison 2: LatentWire vs PCA
    report.append("2. LatentWire vs PCA")
    report.append("   Is non-linear encoding necessary?")
    report.append("")

    pca_f1 = get_f1('pca')

    if lw_llama_f1 is not None and pca_f1 is not None:
        diff = lw_llama_f1 - pca_f1
        pct_improvement = (diff / pca_f1) * 100 if pca_f1 > 0 else 0

        report.append(f"   LatentWire: F1 = {lw_llama_f1*100:.2f}%")
        report.append(f"   PCA:        F1 = {pca_f1*100:.2f}%")
        report.append(f"   Difference: {diff*100:+.2f}% ({pct_improvement:+.1f}% relative)")

        if diff > 0.10:  # 10% absolute improvement
            report.append(f"   ✓ Non-linear encoder is clearly beneficial")
        elif diff > 0:
            report.append(f"   ~ Modest benefit from non-linear encoding")
        else:
            report.append(f"   ✗ Linear compression (PCA) is competitive")
    else:
        report.append("   Data not available for comparison")

    report.append("")

    # Comparison 3: LatentWire vs Text (Information Loss)
    report.append("3. LatentWire vs Text Baseline")
    report.append("   How much information loss from compression?")
    report.append("")

    text_llama_f1 = get_f1('text_llama')

    if lw_llama_f1 is not None and text_llama_f1 is not None:
        retention = (lw_llama_f1 / text_llama_f1) * 100 if text_llama_f1 > 0 else 0

        report.append(f"   Text (full):  F1 = {text_llama_f1*100:.2f}%")
        report.append(f"   LatentWire:   F1 = {lw_llama_f1*100:.2f}%")
        report.append(f"   Retention:    {retention:.1f}% of full-text performance")

        if retention > 80:
            report.append(f"   ✓ Excellent: Minimal information loss")
        elif retention > 60:
            report.append(f"   ~ Good: Moderate information retention")
        elif retention > 40:
            report.append(f"   ⚠ Fair: Significant information loss")
        else:
            report.append(f"   ✗ Poor: Severe information loss")
    else:
        report.append("   Data not available for comparison")

    report.append("")

    # =============================================================================
    # CROSS-MODEL COMPARISON
    # =============================================================================
    report.append("=" * 80)
    report.append("CROSS-MODEL GENERALIZATION")
    report.append("=" * 80)
    report.append("")
    report.append("Does the learned interlingua work for both models?")
    report.append("")

    lw_qwen_f1 = get_f1('latentwire_qwen')

    if lw_llama_f1 is not None and lw_qwen_f1 is not None:
        avg_f1 = (lw_llama_f1 + lw_qwen_f1) / 2
        diff = abs(lw_llama_f1 - lw_qwen_f1)

        report.append(f"   Llama: F1 = {lw_llama_f1*100:.2f}%")
        report.append(f"   Qwen:  F1 = {lw_qwen_f1*100:.2f}%")
        report.append(f"   Average:   {avg_f1*100:.2f}%")
        report.append(f"   Gap:       {diff*100:.2f}%")

        if diff < 0.05:  # Less than 5% gap
            report.append(f"   ✓ Excellent: Interlingua generalizes well across models")
        elif diff < 0.10:
            report.append(f"   ~ Good: Reasonable cross-model generalization")
        else:
            report.append(f"   ⚠ Warning: Large performance gap between models")
    else:
        report.append("   Data not available for cross-model comparison")

    report.append("")

    # =============================================================================
    # OVERALL CONCLUSION
    # =============================================================================
    report.append("=" * 80)
    report.append("OVERALL CONCLUSION")
    report.append("=" * 80)
    report.append("")

    # Determine overall success
    success_criteria = []

    if lw_llama_f1 is not None and tb_llama_f1 is not None:
        if lw_llama_f1 > tb_llama_f1:
            success_criteria.append("✓ Beats token budget baseline")
        else:
            success_criteria.append("✗ Does not beat token budget baseline")

    if lw_llama_f1 is not None and pca_f1 is not None:
        if lw_llama_f1 > pca_f1 + 0.05:  # 5% margin
            success_criteria.append("✓ Beats PCA (non-linear needed)")
        else:
            success_criteria.append("~ PCA is competitive (linear may suffice)")

    if lw_llama_f1 is not None and text_llama_f1 is not None:
        retention = (lw_llama_f1 / text_llama_f1) * 100 if text_llama_f1 > 0 else 0
        if retention > 60:
            success_criteria.append(f"✓ Retains {retention:.0f}% of full-text performance")
        else:
            success_criteria.append(f"⚠ Only retains {retention:.0f}% of full-text performance")

    for criterion in success_criteria:
        report.append(f"   {criterion}")

    report.append("")

    if all('✓' in c for c in success_criteria):
        report.append("   🎉 SUCCESS: Compressed interlingua is working!")
        report.append("      Learned compression beats baselines and retains quality.")
    elif any('✓' in c for c in success_criteria):
        report.append("   ⚠ PARTIAL SUCCESS: Some objectives met, but needs improvement.")
        report.append("     Consider hyperparameter tuning or architecture changes.")
    else:
        report.append("   ✗ NEEDS WORK: Compressed interlingua not yet effective.")
        report.append("     Debug training, check embeddings, tune hyperparameters.")

    report.append("")
    report.append("=" * 80)

    # Print report
    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_txt_path = results_dir / 'comparison_report.txt'
    with open(report_txt_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_txt_path}")

    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'results_dir': str(results_dir),
        'metrics': metrics,
        'success_criteria': success_criteria,
    }

    summary_path = results_dir / 'comparison_report.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"JSON summary saved to: {summary_path}")
    print()


if __name__ == '__main__':
    main()
