#!/usr/bin/env bash
set -e

# =============================================================================
# COMPREHENSIVE ARXIV EXPERIMENT SUITE
# =============================================================================
# Complete suite for arXiv submission with statistical robustness:
#
# PHASE 1: Llama text baselines for SST-2 and AG News (quick upper bounds)
# PHASE 2: Full unified comparison on all 3 datasets (sst2, agnews, trec)
# PHASE 3: Multi-seed runs (42, 123, 456) for statistical robustness
# PHASE 4: Generate comprehensive summary with mean/std across seeds
#
# Hardware: Optimized for 4× H100 GPUs on HPC
# Total time: ~3-4 hours for complete suite
# =============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/full_arxiv_suite_$(date +%Y%m%d_%H%M%S)}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/full_arxiv_suite_${TIMESTAMP}.log"

echo "=============================================================="
echo "COMPREHENSIVE ARXIV EXPERIMENT SUITE"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Experiments:"
echo "  Phase 1: Llama text baselines (SST-2, AG News)"
echo "  Phase 2: Unified comparison (SST-2, AG News, TREC)"
echo "  Phase 3: Multi-seed robustness (seeds: 42, 123, 456)"
echo "  Phase 4: Statistical summary generation"
echo ""
echo "Hardware: 4× H100 GPUs"
echo "Estimated time: 3-4 hours"
echo "=============================================================="
echo ""

# Run full suite with tee to capture ALL output
{
    echo "Starting full arXiv experiment suite at $(date)"
    echo ""

    # =========================================================================
    # PHASE 1: LLAMA TEXT BASELINES
    # =========================================================================
    # Quick baselines to establish upper bounds before full experiments
    # These help validate that Bridge results are meaningful
    echo "=========================================================================="
    echo "PHASE 1: LLAMA TEXT BASELINES"
    echo "=========================================================================="
    echo "Running Llama text baselines for SST-2 and AG News..."
    echo "These provide upper bounds for what direct text access achieves."
    echo ""

    mkdir -p "$OUTPUT_DIR/phase1_baselines"

    # SST-2 Llama baseline
    echo "[1/2] SST-2 Llama text baseline..."
    PYTHONPATH=. python telepathy/eval_sst2_baselines.py \
        --num_samples 200 \
        --output_dir "$OUTPUT_DIR/phase1_baselines" \
        --bf16

    echo ""
    echo "[2/2] AG News Llama text baseline..."
    PYTHONPATH=. python telepathy/eval_agnews_baselines.py \
        --num_samples 200 \
        --output_dir "$OUTPUT_DIR/phase1_baselines" \
        --bf16

    echo ""
    echo "Phase 1 complete! Baselines saved to: $OUTPUT_DIR/phase1_baselines/"
    echo ""

    # =========================================================================
    # PHASE 2: UNIFIED COMPARISON (SINGLE SEED)
    # =========================================================================
    # Main experiment with all baselines on all datasets
    # Seed 42 for initial results
    echo "=========================================================================="
    echo "PHASE 2: UNIFIED COMPARISON (SEED 42)"
    echo "=========================================================================="
    echo "Running complete baseline comparison on all 3 datasets..."
    echo "  - Bridge (Llama→Mistral with cross-attention compression)"
    echo "  - Prompt-Tuning (receiver-only, no sender)"
    echo "  - Text-Relay (Llama summarizes → Mistral classifies)"
    echo "  - Zero-shot (Llama and Mistral direct)"
    echo "  - Few-shot (5-shot in-context learning)"
    echo ""

    mkdir -p "$OUTPUT_DIR/phase2_unified"

    PYTHONPATH=. python telepathy/run_unified_comparison.py \
        --datasets sst2 agnews trec \
        --output_dir "$OUTPUT_DIR/phase2_unified/seed_42" \
        --train_steps 2000 \
        --eval_samples 200 \
        --soft_tokens 8 \
        --seed 42

    echo ""
    echo "Phase 2 complete! Results saved to: $OUTPUT_DIR/phase2_unified/seed_42/"
    echo ""

    # =========================================================================
    # PHASE 3: MULTI-SEED ROBUSTNESS
    # =========================================================================
    # Run with additional seeds for statistical significance
    # Seeds: 42 (already done), 123, 456
    echo "=========================================================================="
    echo "PHASE 3: MULTI-SEED ROBUSTNESS"
    echo "=========================================================================="
    echo "Running experiments with additional seeds for statistical robustness..."
    echo "Seeds: 123, 456 (seed 42 already complete)"
    echo ""

    mkdir -p "$OUTPUT_DIR/phase3_multiseed"

    # Seed 123
    echo "[1/2] Running with seed 123..."
    PYTHONPATH=. python telepathy/run_unified_comparison.py \
        --datasets sst2 agnews trec \
        --output_dir "$OUTPUT_DIR/phase3_multiseed/seed_123" \
        --train_steps 2000 \
        --eval_samples 200 \
        --soft_tokens 8 \
        --seed 123

    echo ""
    echo "[2/2] Running with seed 456..."
    PYTHONPATH=. python telepathy/run_unified_comparison.py \
        --datasets sst2 agnews trec \
        --output_dir "$OUTPUT_DIR/phase3_multiseed/seed_456" \
        --train_steps 2000 \
        --eval_samples 200 \
        --soft_tokens 8 \
        --seed 456

    echo ""
    echo "Phase 3 complete! Multi-seed results saved to: $OUTPUT_DIR/phase3_multiseed/"
    echo ""

    # =========================================================================
    # PHASE 4: STATISTICAL SUMMARY
    # =========================================================================
    # Aggregate results across all seeds and generate summary
    echo "=========================================================================="
    echo "PHASE 4: STATISTICAL SUMMARY"
    echo "=========================================================================="
    echo "Aggregating results across all seeds (42, 123, 456)..."
    echo ""

    # Create summary script inline to avoid new file creation
    python - "$OUTPUT_DIR" <<'SUMMARY_SCRIPT'
import json
import numpy as np
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])

# Collect results from all seeds
seed_dirs = [
    output_dir / "phase2_unified" / "seed_42",
    output_dir / "phase3_multiseed" / "seed_123",
    output_dir / "phase3_multiseed" / "seed_456",
]

datasets = ["sst2", "agnews", "trec"]
methods = ["bridge", "prompt_tuning", "text_relay", "llama_zeroshot",
           "mistral_zeroshot", "mistral_fewshot"]

# Aggregate results
aggregated = {}
for dataset in datasets:
    aggregated[dataset] = {}
    for method in methods:
        accuracies = []
        for seed_dir in seed_dirs:
            results_files = list(seed_dir.glob("unified_results_*.json"))
            if results_files:
                with open(results_files[0]) as f:
                    data = json.load(f)
                    if dataset in data.get("results", {}):
                        result = data["results"][dataset].get(method, {})
                        acc = result.get("accuracy")
                        if acc is not None:
                            accuracies.append(acc)

        if accuracies:
            aggregated[dataset][method] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
                "n_seeds": len(accuracies),
            }

# Print summary table
print("=" * 100)
print("STATISTICAL SUMMARY ACROSS SEEDS (42, 123, 456)")
print("=" * 100)
print()

for dataset in datasets:
    print(f"\n{dataset.upper()}")
    print("-" * 100)
    print(f"{'Method':<25} {'Mean Acc':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'N':<5}")
    print("-" * 100)

    method_names = {
        "bridge": "Bridge (ours)",
        "prompt_tuning": "Prompt-Tuning",
        "text_relay": "Text-Relay",
        "llama_zeroshot": "Llama 0-shot",
        "mistral_zeroshot": "Mistral 0-shot",
        "mistral_fewshot": "Mistral 5-shot",
    }

    for method in methods:
        if method in aggregated[dataset]:
            stats = aggregated[dataset][method]
            name = method_names.get(method, method)
            print(f"{name:<25} {stats['mean']:>10.2f}%  {stats['std']:>8.2f}  "
                  f"{stats['min']:>8.2f}  {stats['max']:>8.2f}  {stats['n_seeds']:>3}")

# Save summary JSON
summary_file = output_dir / "statistical_summary.json"
with open(summary_file, "w") as f:
    json.dump(aggregated, f, indent=2)

print()
print("=" * 100)
print(f"Summary saved to: {summary_file}")
print("=" * 100)
SUMMARY_SCRIPT

    echo ""
    echo "Phase 4 complete! Statistical summary saved."
    echo ""

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    echo "=========================================================================="
    echo "ALL EXPERIMENTS COMPLETE"
    echo "=========================================================================="
    echo "Finished at $(date)"
    echo ""
    echo "Results directory structure:"
    echo "  $OUTPUT_DIR/"
    echo "    ├── phase1_baselines/          # Llama text baselines"
    echo "    ├── phase2_unified/seed_42/    # Main experiments (seed 42)"
    echo "    ├── phase3_multiseed/"
    echo "    │   ├── seed_123/              # Robustness check"
    echo "    │   └── seed_456/              # Robustness check"
    echo "    └── statistical_summary.json   # Aggregated stats"
    echo ""

    # Print key results if available
    SUMMARY_FILE="$OUTPUT_DIR/statistical_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo "=========================================================================="
        echo "KEY RESULTS (mean ± std across 3 seeds)"
        echo "=========================================================================="
        python -c "
import json
with open('$SUMMARY_FILE') as f:
    data = json.load(f)

for ds in ['sst2', 'agnews', 'trec']:
    if ds in data:
        print(f'\n{ds.upper()}:')
        if 'bridge' in data[ds]:
            b = data[ds]['bridge']
            print(f'  Bridge:        {b[\"mean\"]:.1f}% ± {b[\"std\"]:.1f}%')
        if 'prompt_tuning' in data[ds]:
            p = data[ds]['prompt_tuning']
            print(f'  Prompt-Tuning: {p[\"mean\"]:.1f}% ± {p[\"std\"]:.1f}%')
        if 'llama_zeroshot' in data[ds]:
            l = data[ds]['llama_zeroshot']
            print(f'  Llama 0-shot:  {l[\"mean\"]:.1f}% ± {l[\"std\"]:.1f}%')
"
        echo ""
    fi

    echo ""
    echo "=========================================================================="
    echo "NEXT STEPS FOR PAPER"
    echo "=========================================================================="
    echo "1. Review results in: $OUTPUT_DIR/statistical_summary.json"
    echo "2. Check individual seed variance in phase3_multiseed/"
    echo "3. Extract latency numbers from phase2_unified/seed_42/"
    echo "4. Generate plots from JSON results"
    echo "5. Compute statistical significance (t-tests between methods)"
    echo ""
    echo "For significance testing, use:"
    echo "  PYTHONPATH=. python telepathy/compute_significance.py \\"
    echo "    --results_dir $OUTPUT_DIR"
    echo ""

} 2>&1 | tee "$LOG_FILE"

echo "=========================================================================="
echo "COMPLETE! Full log saved to:"
echo "  $LOG_FILE"
echo "=========================================================================="
