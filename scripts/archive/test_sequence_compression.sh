#!/bin/bash
# Test Sequence Compression: Phase 1a Baseline vs Pooling vs Pooling+LoRA
#
# This script runs 3 experiments to test if sequence compression (300→75 tokens) works:
# 1. Phase 1a baseline: Dimension compression only (should replicate F1=24%)
# 2. Phase 1a + pooling: Add learned sequence pooling (300→75 tokens)
# 3. Phase 1a + pooling + LoRA: Add LoRA adaptation to help LLM process compressed sequences
#
# Run with: git pull && rm -rf runs && PYTHONPATH=. bash scripts/test_sequence_compression.sh

set -e  # Exit on error
set -x  # Print commands

echo "===================================================================="
echo "SEQUENCE COMPRESSION TEST SUITE"
echo "===================================================================="
echo "Running 3 experiments:"
echo "  1. Phase 1a baseline (dimension compression only)"
echo "  2. Phase 1a + sequence pooling (300→75 tokens)"
echo "  3. Phase 1a + pooling + LoRA (add adapter to LLM)"
echo ""
echo "Expected timeline: ~3-4 hours total (1-1.5hr each)"
echo "===================================================================="
echo ""

# Shared hyperparameters
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
SAMPLES=10000
PCA_SAMPLES=5000
EPOCHS=1
BATCH_SIZE=8
COMPRESS_DIM=1024
ADAPTER_LR=5e-4
EVAL_SAMPLES=200

# Create runs directory
mkdir -p runs

echo ""
echo "===================================================================="
echo "EXPERIMENT 1: Phase 1a Baseline (Dimension Compression Only)"
echo "===================================================================="
echo "Architecture:"
echo "  Text → Embed [300, 4096]"
echo "       → PCA [300, 1024]"
echo "       → Adapter [300, 4096]"
echo "       → LLM generation"
echo ""
echo "No sequence compression - still processing 300 tokens"
echo "Expected F1: ~24% (replicating previous result)"
echo "===================================================================="

python train_adapter_only_phase1.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --compress_method pca \
  --adapter_lr $ADAPTER_LR \
  --eval_every 1 \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp1_baseline" \
  --diagnostic_log "runs/exp1_baseline/diagnostics.jsonl"

echo ""
echo "✓ Experiment 1 complete!"
echo ""

# Extract F1 from diagnostics
if [ -f "runs/exp1_baseline/diagnostics.jsonl" ]; then
    F1_BASELINE=$(grep '"type": "full_eval"' runs/exp1_baseline/diagnostics.jsonl | tail -1 | python -c "import sys, json; print(f\"{json.load(sys.stdin)['f1']:.1%}\")" 2>/dev/null || echo "N/A")
    echo "Baseline F1: $F1_BASELINE"
fi

echo ""
echo "===================================================================="
echo "EXPERIMENT 2: Phase 1a + Sequence Pooling"
echo "===================================================================="
echo "Architecture:"
echo "  Text → Embed [300, 4096]"
echo "       → PCA [300, 1024]"
echo "       → Sequence Pooler [75, 1024]  ← NEW: 4× compression"
echo "       → Adapter [75, 4096]"
echo "       → LLM generation"
echo ""
echo "Tests: Can learned pooling compress 300→75 tokens?"
echo "Expected F1: >30% (modest compression should work)"
echo "           : <30% (compression too lossy, need to adjust)"
echo "===================================================================="

python train_adapter_only_phase1_pooling.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --compress_method pca \
  --sequence_pooling_target 75 \
  --adapter_lr $ADAPTER_LR \
  --eval_every 1 \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp2_pooling" \
  --diagnostic_log "runs/exp2_pooling/diagnostics.jsonl"

echo ""
echo "✓ Experiment 2 complete!"
echo ""

if [ -f "runs/exp2_pooling/diagnostics.jsonl" ]; then
    F1_POOLING=$(grep '"type": "full_eval"' runs/exp2_pooling/diagnostics.jsonl | tail -1 | python -c "import sys, json; print(f\"{json.load(sys.stdin)['f1']:.1%}\")" 2>/dev/null || echo "N/A")
    echo "Pooling F1: $F1_POOLING"
fi

echo ""
echo "===================================================================="
echo "EXPERIMENT 3: Phase 1a + Pooling + LoRA"
echo "===================================================================="
echo "Architecture:"
echo "  Text → Embed [300, 4096]"
echo "       → PCA [300, 1024]"
echo "       → Sequence Pooler [75, 1024]"
echo "       → Adapter [75, 4096]"
echo "       → LLM (with LoRA on first 4 layers) ← NEW"
echo ""
echo "Tests: Can LoRA help LLM adapt to compressed sequences?"
echo "Expected F1: If Exp2 was 30-40%, this should push to 40-50%"
echo "           : If Exp2 was <30%, LoRA might not help enough"
echo "===================================================================="

python train_adapter_only_phase1_pooling.py \
  --model_id "$MODEL" \
  --samples $SAMPLES \
  --pca_samples $PCA_SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --compress_dim $COMPRESS_DIM \
  --compress_method pca \
  --sequence_pooling_target 75 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_layers 4 \
  --adapter_lr $ADAPTER_LR \
  --eval_every 1 \
  --eval_samples $EVAL_SAMPLES \
  --save_dir "runs/exp3_pooling_lora" \
  --diagnostic_log "runs/exp3_pooling_lora/diagnostics.jsonl"

echo ""
echo "✓ Experiment 3 complete!"
echo ""

if [ -f "runs/exp3_pooling_lora/diagnostics.jsonl" ]; then
    F1_LORA=$(grep '"type": "full_eval"' runs/exp3_pooling_lora/diagnostics.jsonl | tail -1 | python -c "import sys, json; print(f\"{json.load(sys.stdin)['f1']:.1%}\")" 2>/dev/null || echo "N/A")
    echo "Pooling+LoRA F1: $F1_LORA"
fi

echo ""
echo "===================================================================="
echo "SUMMARY OF RESULTS"
echo "===================================================================="
echo ""
echo "Experiment 1 (Baseline):      F1 = ${F1_BASELINE:-N/A}"
echo "Experiment 2 (+ Pooling):     F1 = ${F1_POOLING:-N/A}"
echo "Experiment 3 (+ Pool + LoRA): F1 = ${F1_LORA:-N/A}"
echo ""
echo "===================================================================="
echo "INTERPRETATION GUIDE"
echo "===================================================================="
echo ""
echo "IF Exp1 ≈ 24%:"
echo "  ✓ Baseline replicated successfully"
echo ""
echo "IF Exp2 ≥ 30%:"
echo "  ✓ Sequence compression (4×) works!"
echo "  → Next: Try more compression (300→50 or 300→32 tokens)"
echo "  → Next: Add second model (Qwen) for shared interlingua"
echo ""
echo "IF Exp2 = 15-30%:"
echo "  ⚠ Marginal - compression somewhat lossy"
echo "  → Check if Exp3 (LoRA) improves it"
echo "  → May need to reduce compression (300→100 tokens)"
echo ""
echo "IF Exp2 < 15%:"
echo "  ✗ Sequence compression too aggressive"
echo "  → Try less compression (300→150 tokens, 2×)"
echo "  → Or try different pooling method (hierarchical, conv)"
echo ""
echo "IF Exp3 ≫ Exp2 (e.g., 40% vs 30%):"
echo "  ✓ LoRA helps significantly!"
echo "  → Use LoRA in full LatentWire system"
echo ""
echo "IF Exp3 ≈ Exp2 (no improvement):"
echo "  ⚠ LoRA doesn't help"
echo "  → Bottleneck is compression quality, not LLM adaptation"
echo "  → Focus on improving pooler architecture"
echo ""
echo "===================================================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "===================================================================="
