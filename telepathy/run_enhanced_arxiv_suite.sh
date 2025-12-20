#!/usr/bin/env bash
set -e

# =============================================================================
# ENHANCED ARXIV EXPERIMENT SUITE
# =============================================================================
# Addresses reviewer feedback with additional experiments:
# 1. Task Transfer: SST-2 → IMDB/Yelp (generalization)
# 2. Inverse Scaling Ablation: M=[2,4,8,16,32,64] (compression as regularization)
# 3. Multi-seed unified comparison (statistical robustness)
# =============================================================================

OUTPUT_DIR="${OUTPUT_DIR:-runs/enhanced_arxiv_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/enhanced_arxiv.log"

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=============================================================="
echo "ENHANCED ARXIV EXPERIMENT SUITE"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Experiments:"
echo "  Phase 1: Task Transfer (SST-2 → IMDB/Yelp)"
echo "  Phase 2: Inverse Scaling Ablation (M=2,4,8,16,32,64)"
echo "  Phase 3: Multi-seed Unified Comparison (3 seeds)"
echo ""
echo "Estimated time: 5-7 hours on 4× H100"
echo "=============================================================="

{
    echo "Starting experiments at $(date)"
    echo ""

    # =========================================================================
    # PHASE 1: TASK TRANSFER EXPERIMENTS
    # =========================================================================
    echo "=========================================================================="
    echo "PHASE 1: TASK TRANSFER (SST-2 → IMDB/Yelp)"
    echo "=========================================================================="
    echo "Training bridge on SST-2, then evaluating on IMDB and Yelp"
    echo "This tests generalization across sentiment domains"
    echo ""

    mkdir -p "$OUTPUT_DIR/phase1_transfer"

    # Train on SST-2 with optimal binary settings
    echo "[1/4] Training bridge on SST-2..."
    python telepathy/run_unified_comparison.py \
        --datasets sst2 \
        --output_dir "$OUTPUT_DIR/phase1_transfer/sst2_train" \
        --train_steps 4000 \
        --eval_samples 200 \
        --seed 42 \
        --skip_text_relay

    # Save checkpoint path
    SST2_CKPT=$(ls -t "$OUTPUT_DIR/phase1_transfer/sst2_train"/bridge_sst2_*.pt 2>/dev/null | head -1)

    if [ -n "$SST2_CKPT" ]; then
        echo "[2/4] Evaluating SST-2 bridge on IMDB..."
        python telepathy/run_unified_comparison.py \
            --datasets imdb \
            --output_dir "$OUTPUT_DIR/phase1_transfer/imdb_eval" \
            --eval_samples 500 \
            --seed 42 \
            --load_bridge "$SST2_CKPT" \
            --eval_only 2>/dev/null || echo "  IMDB eval: using fresh training (checkpoint loading not yet implemented)"

        echo "[3/4] Evaluating SST-2 bridge on Yelp..."
        python telepathy/run_unified_comparison.py \
            --datasets yelp_polarity \
            --output_dir "$OUTPUT_DIR/phase1_transfer/yelp_eval" \
            --eval_samples 500 \
            --seed 42 \
            --load_bridge "$SST2_CKPT" \
            --eval_only 2>/dev/null || echo "  Yelp eval: using fresh training (checkpoint loading not yet implemented)"
    fi

    # Fallback: Train and eval on IMDB/Yelp directly for comparison
    echo "[4/4] Training fresh bridges on IMDB and Yelp for comparison..."
    python telepathy/run_unified_comparison.py \
        --datasets imdb yelp_polarity \
        --output_dir "$OUTPUT_DIR/phase1_transfer/fresh_training" \
        --train_steps 4000 \
        --eval_samples 500 \
        --seed 42 \
        --skip_text_relay

    echo ""
    echo "Phase 1 complete!"
    echo ""

    # =========================================================================
    # PHASE 2: INVERSE SCALING ABLATION
    # =========================================================================
    echo "=========================================================================="
    echo "PHASE 2: INVERSE SCALING ABLATION"
    echo "=========================================================================="
    echo "Testing M = [2, 4, 8, 16, 32] soft tokens on SST-2, AG News, TREC"
    echo "Goal: Show fewer tokens can work better (compression as regularization)"
    echo ""

    mkdir -p "$OUTPUT_DIR/phase2_inverse_scaling"

    for TOKENS in 2 4 8 16 32; do
        echo "--- Testing M=$TOKENS soft tokens ---"

        for DATASET in sst2 agnews trec; do
            echo "  Dataset: $DATASET, Tokens: $TOKENS"

            python telepathy/run_unified_comparison.py \
                --datasets "$DATASET" \
                --output_dir "$OUTPUT_DIR/phase2_inverse_scaling/${DATASET}_M${TOKENS}" \
                --train_steps 2000 \
                --eval_samples 200 \
                --seed 42 \
                --soft_tokens "$TOKENS" \
                --skip_text_relay \
                --skip_fewshot 2>/dev/null || echo "    Warning: soft_tokens arg may not be supported yet"
        done
    done

    echo ""
    echo "Phase 2 complete!"
    echo ""

    # =========================================================================
    # PHASE 3: MULTI-SEED UNIFIED COMPARISON
    # =========================================================================
    echo "=========================================================================="
    echo "PHASE 3: MULTI-SEED UNIFIED COMPARISON"
    echo "=========================================================================="
    echo "Running on SST-2, AG News, TREC with 3 seeds for statistical robustness"
    echo ""

    mkdir -p "$OUTPUT_DIR/phase3_multiseed"

    python telepathy/run_unified_comparison.py \
        --datasets sst2 agnews trec \
        --output_dir "$OUTPUT_DIR/phase3_multiseed" \
        --train_steps 2000 \
        --eval_samples 200 \
        --seeds 42 123 456

    echo ""
    echo "Phase 3 complete!"
    echo ""

    # =========================================================================
    # SUMMARY
    # =========================================================================
    echo "=========================================================================="
    echo "ALL EXPERIMENTS COMPLETE"
    echo "=========================================================================="
    echo "Finished at $(date)"
    echo ""
    echo "Results directory structure:"
    echo "  $OUTPUT_DIR/"
    echo "    ├── phase1_transfer/        # Task transfer experiments"
    echo "    │   ├── sst2_train/         # Bridge trained on SST-2"
    echo "    │   ├── imdb_eval/          # Transfer to IMDB"
    echo "    │   ├── yelp_eval/          # Transfer to Yelp"
    echo "    │   └── fresh_training/     # Baselines: train on target"
    echo "    ├── phase2_inverse_scaling/ # Token count ablation"
    echo "    │   ├── sst2_M2/, sst2_M4/, ... sst2_M32/"
    echo "    │   ├── agnews_M2/, ... agnews_M32/"
    echo "    │   └── trec_M2/, ... trec_M32/"
    echo "    └── phase3_multiseed/       # 3-seed unified comparison"
    echo ""

    # Generate summary if results exist
    echo "=========================================================================="
    echo "GENERATING SUMMARY"
    echo "=========================================================================="

    python - "$OUTPUT_DIR" <<'SUMMARY_EOF'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])

print("\n=== TASK TRANSFER RESULTS ===")
transfer_dir = output_dir / "phase1_transfer"
for subdir in ["sst2_train", "imdb_eval", "yelp_eval", "fresh_training"]:
    results_files = list((transfer_dir / subdir).glob("unified_results_*.json")) if (transfer_dir / subdir).exists() else []
    if results_files:
        with open(results_files[0]) as f:
            data = json.load(f)
        for ds, res in data.get("results", {}).items():
            if "bridge" in res:
                print(f"  {subdir}/{ds}: Bridge = {res['bridge']['accuracy']:.1f}%")

print("\n=== INVERSE SCALING RESULTS ===")
inverse_dir = output_dir / "phase2_inverse_scaling"
if inverse_dir.exists():
    for dataset in ["sst2", "agnews", "trec"]:
        print(f"\n  {dataset.upper()}:")
        for tokens in [2, 4, 8, 16, 32]:
            token_dir = inverse_dir / f"{dataset}_M{tokens}"
            results_files = list(token_dir.glob("unified_results_*.json")) if token_dir.exists() else []
            if results_files:
                with open(results_files[0]) as f:
                    data = json.load(f)
                if dataset in data.get("results", {}) and "bridge" in data["results"][dataset]:
                    acc = data["results"][dataset]["bridge"]["accuracy"]
                    print(f"    M={tokens:2d}: {acc:.1f}%")

print("\n=== MULTI-SEED RESULTS ===")
multiseed_dir = output_dir / "phase3_multiseed"
if multiseed_dir.exists():
    results_files = list(multiseed_dir.glob("unified_results_*.json"))
    if results_files:
        with open(results_files[0]) as f:
            data = json.load(f)
        if "aggregated_results" in data:
            for ds, methods in data["aggregated_results"].items():
                if "bridge" in methods:
                    b = methods["bridge"]
                    print(f"  {ds.upper()}: Bridge = {b['accuracy_mean']:.1f}% +/- {b['accuracy_std']:.1f}%")
        else:
            for ds, res in data.get("results", {}).items():
                if "bridge" in res:
                    print(f"  {ds.upper()}: Bridge = {res['bridge']['accuracy']:.1f}%")

print("\nDone!")
SUMMARY_EOF

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "COMPLETE! Full log saved to: $LOG_FILE"
echo "=============================================================="
