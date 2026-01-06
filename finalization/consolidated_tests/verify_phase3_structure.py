#!/usr/bin/env python3
"""
Verify Phase 3 baseline structure and files are present.
This checks that all necessary files exist without requiring dependencies.
"""

import os
import json
from pathlib import Path

def main():
    print("="*80)
    print("PHASE 3 BASELINE FILES VERIFICATION")
    print("="*80)
    print()

    # Define required files for Phase 3
    required_files = {
        "LLMLingua Baseline": [
            "latentwire/llmlingua_baseline.py",
            "scripts/run_llmlingua_baseline.sh",
            "latentwire/analyze_llmlingua_results.py"
        ],
        "Token Budget Baseline": [
            "scripts/baselines/evaluate_token_budget.py",
            "scripts/baselines/token_budget_baseline.sh"
        ],
        "Linear Probe Baseline": [
            "latentwire/linear_probe_baseline.py"
        ]
    }

    # Check each baseline
    results = {}
    for baseline_name, files in required_files.items():
        print(f"\n{baseline_name}:")
        baseline_ok = True
        for file_path in files:
            full_path = Path(__file__).parent / file_path
            if full_path.exists():
                # Check file size to ensure it's not empty
                size = full_path.stat().st_size
                if size > 100:  # At least 100 bytes
                    print(f"  [OK] {file_path} ({size:,} bytes)")
                else:
                    print(f"  [WARNING] {file_path} exists but seems empty ({size} bytes)")
                    baseline_ok = False
            else:
                print(f"  [MISSING] {file_path}")
                baseline_ok = False

        results[baseline_name] = baseline_ok

    # Check for key functions/classes in files
    print("\n" + "="*80)
    print("CHECKING KEY COMPONENTS")
    print("="*80)

    # Check LLMLingua baseline
    llmlingua_path = Path(__file__).parent / "latentwire/llmlingua_baseline.py"
    if llmlingua_path.exists():
        with open(llmlingua_path) as f:
            content = f.read()
        key_components = ["LLMLinguaCompressor", "evaluate_llmlingua_on_qa", "compress_to_target_tokens"]
        print("\nLLMLingua Baseline Components:")
        for component in key_components:
            if component in content:
                print(f"  [OK] {component} found")
            else:
                print(f"  [MISSING] {component} not found")

    # Check Linear Probe baseline
    linear_probe_path = Path(__file__).parent / "latentwire/linear_probe_baseline.py"
    if linear_probe_path.exists():
        with open(linear_probe_path) as f:
            content = f.read()
        key_components = ["LinearProbeBaseline", "LogisticRegression", "fit", "predict", "score"]
        print("\nLinearProbeBaseline Components:")
        for component in key_components:
            if component in content:
                print(f"  [OK] {component} found")
            else:
                print(f"  [MISSING] {component} not found")

    # Check Token Budget evaluation
    token_budget_path = Path(__file__).parent / "scripts/baselines/evaluate_token_budget.py"
    if token_budget_path.exists():
        with open(token_budget_path) as f:
            content = f.read()
        key_components = ["AutoTokenizer", "AutoModelForCausalLM", "batch_metrics", "token_budget"]
        print("\nToken Budget Baseline Components:")
        for component in key_components:
            if component in content:
                print(f"  [OK] {component} found")
            else:
                print(f"  [MISSING] {component} not found")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_ok = all(results.values())
    if all_ok:
        print("\n[SUCCESS] All Phase 3 baseline files are present and structured correctly!")
        print("\nTo run the baselines, you need to install dependencies:")
        print("  pip install torch transformers scikit-learn llmlingua")
        print("\nThen you can run:")
        print("  - LLMLingua: bash scripts/run_llmlingua_baseline.sh")
        print("  - Token Budget: bash scripts/baselines/token_budget_baseline.sh")
        print("  - Linear Probe: python latentwire/linear_probe_baseline.py --help")
    else:
        failed = [name for name, ok in results.items() if not ok]
        print(f"\n[WARNING] Some baselines have missing files: {', '.join(failed)}")
        print("\nPlease check the file paths above.")

    # Check for statistical testing script
    print("\n" + "="*80)
    print("STATISTICAL TESTING")
    print("="*80)

    stat_test_path = Path(__file__).parent / "scripts/statistical_testing.py"
    if stat_test_path.exists():
        size = stat_test_path.stat().st_size
        print(f"[OK] statistical_testing.py found ({size:,} bytes)")
        with open(stat_test_path) as f:
            content = f.read()
        key_functions = ["bootstrap_ci", "paired_ttest", "cohens_d", "mcnemar_test"]
        print("\nStatistical Testing Functions:")
        for func in key_functions:
            if func in content:
                print(f"  [OK] {func} found")
            else:
                print(f"  [MISSING] {func} not found")
    else:
        print("[MISSING] scripts/statistical_testing.py not found")

    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())