#!/bin/bash
# Comprehensive Sequence Compression Test Suite
#
# Tests multiple strategies to find what works for sequence compression
# Each experiment ~1-2 hours, ~8-10 hours total
#
# Run with: git pull && rm -rf runs && PYTHONPATH=. bash scripts/test_compression_strategies.sh

set -e
set -x

echo "===================================================================="
echo "COMPREHENSIVE SEQUENCE COMPRESSION TEST SUITE"
echo "===================================================================="
echo "Testing 7 different approaches:"
echo "  1. Baseline (no compression) - F1 24% expected"
echo "  2. Pooling 4× (pure reconstruction) - FAILED (F1 0.7%)"
echo "  3. Pooling 4× with generation loss - Can proper objective fix it?"
echo "  4. Pooling 2× (less aggressive) - Does less compression work?"
echo "  5. Hierarchical pooling 4× - Preserve structure better?"
echo "  6. Convolutional downsampling 4× - Local structure preservation"
echo "  7. Hybrid: Pool + expand for reconstruction - Best of both?"
echo ""
echo "Timeline: ~8-10 hours total"
echo "===================================================================="
echo ""

# Shared settings
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
SAMPLES=10000
PCA_SAMPLES=5000
EPOCHS=1
BATCH_SIZE=8
COMPRESS_DIM=1024
ADAPTER_LR=5e-4
EVAL_SAMPLES=200

mkdir -p runs

echo ""
echo "===================================================================="
echo "EXPERIMENT 1: Baseline (Replication)"
echo "===================================================================="
echo "No sequence compression - dimension compression only"
echo "Expected: F1 ≈ 24%"
echo "===================================================================="

python train_adapter_only_phase1.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --adapter_lr $ADAPTER_LR \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp1_baseline" \
  --diagnostic_log "runs/exp1_baseline/diagnostics.jsonl"

echo "✓ Experiment 1 complete"
echo ""

echo "===================================================================="
echo "EXPERIMENT 2: SKIP - Already know it fails (F1 0.7%)"
echo "===================================================================="
echo "Pooling 4× with reconstruction loss on averaged embeddings"
echo "Result: F1 0.7% (averaging destroys sequential information)"
echo "Skipping to save time"
echo ""

echo "===================================================================="
echo "EXPERIMENT 3: Pooling 4× with Generation Loss"
echo "===================================================================="
echo "Architecture: Embed → PCA → Pooler [75] → Adapter → LLM"
echo "Loss: Direct generation loss (no reconstruction)"
echo "Hypothesis: Proper objective might make pooling work"
echo "===================================================================="

python train_adapter_only_phase1_pooling_v2.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --sequence_pooling_target 75 \
  --pooling_mode "generation_loss" \
  --adapter_lr $ADAPTER_LR \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp3_pooling_genloss" \
  --diagnostic_log "runs/exp3_pooling_genloss/diagnostics.jsonl"

echo "✓ Experiment 3 complete"
echo ""

echo "===================================================================="
echo "EXPERIMENT 4: Pooling 2× (Less Aggressive)"
echo "===================================================================="
echo "Architecture: Embed → PCA → Pooler [150] → Adapter → LLM"
echo "Loss: Generation loss"
echo "Hypothesis: Less compression = easier to preserve info"
echo "===================================================================="

python train_adapter_only_phase1_pooling_v2.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --sequence_pooling_target 150 \
  --pooling_mode "generation_loss" \
  --adapter_lr $ADAPTER_LR \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp4_pooling_2x" \
  --diagnostic_log "runs/exp4_pooling_2x/diagnostics.jsonl"

echo "✓ Experiment 4 complete"
echo ""

echo "===================================================================="
echo "EXPERIMENT 5: Hierarchical Pooling 4×"
echo "===================================================================="
echo "Architecture: Embed → PCA → Hierarchical Pooler [75] → Adapter"
echo "Pooling stages: 300 → 225 → 150 → 75 (3 stages, 1.33× each)"
echo "Hypothesis: Gradual compression preserves structure better"
echo "===================================================================="

python train_adapter_only_phase1_pooling_v2.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --sequence_pooling_target 75 \
  --pooling_mode "hierarchical" \
  --adapter_lr $ADAPTER_LR \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp5_hierarchical" \
  --diagnostic_log "runs/exp5_hierarchical/diagnostics.jsonl"

echo "✓ Experiment 5 complete"
echo ""

echo "===================================================================="
echo "EXPERIMENT 6: Convolutional Downsampling 4×"
echo "===================================================================="
echo "Architecture: Embed → PCA → Conv1D(stride=4) [75] → Adapter"
echo "Hypothesis: Conv preserves local context better than global pooling"
echo "===================================================================="

python train_adapter_only_phase1_pooling_v2.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --sequence_pooling_target 75 \
  --pooling_mode "convolutional" \
  --adapter_lr $ADAPTER_LR \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp6_convolutional" \
  --diagnostic_log "runs/exp6_convolutional/diagnostics.jsonl"

echo "✓ Experiment 6 complete"
echo ""

echo "===================================================================="
echo "EXPERIMENT 7: Hybrid Pool-Expand-Reconstruct"
echo "===================================================================="
echo "Architecture: Embed → PCA → Pooler [75] → Expand [300] → Adapter"
echo "Training: Reconstruction loss on expanded sequence"
echo "Inference: Use compressed [75] tokens directly"
echo "Hypothesis: Train with reconstruction, use compressed at test time"
echo "===================================================================="

python train_adapter_only_phase1_pooling_v2.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --sequence_pooling_target 75 \
  --pooling_mode "hybrid_expand" \
  --adapter_lr $ADAPTER_LR \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp7_hybrid" \
  --diagnostic_log "runs/exp7_hybrid/diagnostics.jsonl"

echo "✓ Experiment 7 complete"
echo ""

echo "===================================================================="
echo "ALL EXPERIMENTS COMPLETE - ANALYZING RESULTS"
echo "===================================================================="
echo ""

# Extract results
python3 << 'EOF'
import json
import os

experiments = [
    ("exp1_baseline", "Baseline (no seq compression)"),
    ("exp3_pooling_genloss", "Pooling 4× + generation loss"),
    ("exp4_pooling_2x", "Pooling 2× + generation loss"),
    ("exp5_hierarchical", "Hierarchical pooling 4×"),
    ("exp6_convolutional", "Convolutional 4×"),
    ("exp7_hybrid", "Hybrid pool-expand"),
]

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"{'Experiment':<40} {'F1':<10} {'EM':<10} {'Status':<15}")
print("-"*80)

for exp_id, exp_name in experiments:
    diag_path = f"runs/{exp_id}/diagnostics.jsonl"
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            lines = f.readlines()
            # Find last full_eval
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    if data.get("type") == "full_eval":
                        f1 = data["f1"]
                        em = data["em"]

                        # Determine status
                        if f1 >= 0.30:
                            status = "✅ SUCCESS"
                        elif f1 >= 0.15:
                            status = "⚠️  MARGINAL"
                        elif f1 >= 0.05:
                            status = "❌ POOR"
                        else:
                            status = "❌ FAILURE"

                        print(f"{exp_name:<40} {f1:<10.3f} {em:<10.3f} {status:<15}")
                        break
                except:
                    continue
    else:
        print(f"{exp_name:<40} {'N/A':<10} {'N/A':<10} {'Not run':<15}")

print("="*80)
print()
print("SUCCESS CRITERIA:")
print("  F1 ≥ 30%: Sequence compression works - proceed to more compression")
print("  F1 = 15-30%: Marginal - may work with tuning or LoRA")
print("  F1 < 15%: Approach doesn't work - try different method")
print()
print("BASELINE COMPARISON:")
print("  Baseline (no compression): F1 ≈ 24%")
print("  Target: Match or exceed baseline while compressing sequence")
print("="*80)
EOF

echo ""
echo "===================================================================="
echo "NEXT STEPS BASED ON RESULTS"
echo "===================================================================="
echo ""
echo "IF any experiment ≥ 30% F1:"
echo "  → Success! Use that method"
echo "  → Try more compression (4×, 8×, 10×)"
echo "  → Add second model (Qwen) for shared interlingua"
echo ""
echo "IF best result is 15-30% F1:"
echo "  → Marginal - try improvements:"
echo "  → Add LoRA to help LLM adapt"
echo "  → Increase training time (more epochs)"
echo "  → Try combined approaches"
echo ""
echo "IF all results < 15% F1:"
echo "  → Sequence compression fundamentally difficult"
echo "  → Consider alternatives:"
echo "  → Modest compression only (2×)"
echo "  → ByteEncoder end-to-end (different paradigm)"
echo "  → Document limits for research contribution"
echo "=="=================================================================="
