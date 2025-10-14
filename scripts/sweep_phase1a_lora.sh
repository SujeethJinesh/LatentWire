#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Phase 1a + LoRA Sweep
# Tests if LoRA adaptation helps pure reconstruction (24% F1) without K-token CE collapse (0% F1)
#
# Hypothesis: Phase 1a (pure reconstruction) achieved 24% F1 with correct semantics.
#             Adding LoRA might help LLM "listen better" to compressed embeddings.
#             Avoids Phase 1b K-token CE conflict that caused 0% F1 collapse.
#
# Evidence from LOG.md:
# - Phase 1a (reconstruction only): 24% F1, answer present but wrong format
# - Phase 1b (reconstruction + K-token CE): 0% F1, catastrophic collapse
# - Section 2.8 (K-token CE + LoRA): 4% F1, still collapsed
# - Gap: Never tested reconstruction + LoRA (no K-token CE)

echo "=================================================="
echo "PHASE 1A + LORA SWEEP"
echo "=================================================="
echo ""
echo "Testing: Pure reconstruction + LoRA adaptation"
echo "Baseline: Phase 1a (24% F1, no LoRA)"
echo "Goal: Improve Phase 1a by helping LLM process compressed embeddings"
echo ""

# Configuration
SAMPLES="${SAMPLES:-5000}"     # Reduced from 10k for faster iteration
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"  # Conservative for multi-GPU
DATASET="${DATASET:-squad}"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Phase 1a Architecture (PCA + Adapter)
COMPRESS_DIM=1024               # PCA: 4096 → 1024
PCA_SAMPLES=5000                # Samples to fit PCA
ADAPTER_HIDDEN_MULT=4           # Adapter hidden dimension multiplier
ADAPTER_DROPOUT=0.1             # Adapter dropout
ADAPTER_LR=5e-4                 # Adapter learning rate

# Reconstruction loss weights
COSINE_WEIGHT=1.0               # Cosine similarity (direction)
MSE_WEIGHT=0.1                  # MSE (magnitude)

# LoRA sweep configurations
# Format: name:rank:alpha:layers
declare -a CONFIGS=(
    # Baseline (no LoRA)
    "baseline:0:0:0"

    # Small LoRA (minimal adaptation)
    "r4_a8_layers8:4:8:8"
    "r4_a16_layers8:4:16:8"

    # Medium LoRA (balanced)
    "r8_a8_layers12:8:8:12"
    "r8_a16_layers12:8:16:12"
    "r8_a32_layers12:8:32:12"

    # Large LoRA (more capacity)
    "r16_a16_layers16:16:16:16"
    "r16_a32_layers16:16:32:16"

    # Full model LoRA (all layers)
    "r8_a16_full:8:16:0"
    "r16_a32_full:16:32:0"

    # High rank (maximum adaptation)
    "r32_a32_layers16:32:32:16"
)

echo "Sweep Configuration:"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo ""
echo "  Phase 1a Architecture (Pure Reconstruction):"
echo "    PCA compression: 4096 → $COMPRESS_DIM"
echo "    PCA samples: $PCA_SAMPLES"
echo "    Adapter hidden mult: $ADAPTER_HIDDEN_MULT"
echo "    Adapter LR: $ADAPTER_LR"
echo "    Loss: Cosine (${COSINE_WEIGHT}×) + MSE (${MSE_WEIGHT}×)"
echo ""
echo "  LoRA Configurations: ${#CONFIGS[@]}"
echo ""

# Create output directory
OUTPUT_BASE="runs/phase1a_lora_sweep"
mkdir -p "$OUTPUT_BASE"

# Create summary file
SUMMARY_FILE="$OUTPUT_BASE/sweep_summary.txt"
cat > "$SUMMARY_FILE" <<EOF
Phase 1a + LoRA Sweep
Started: $(date)
================================================

Configuration:
  Samples: $SAMPLES
  Epochs: $EPOCHS
  Batch Size: $BATCH_SIZE
  PCA compression: 4096 → $COMPRESS_DIM
  Adapter LR: $ADAPTER_LR
  Loss: Cosine (${COSINE_WEIGHT}×) + MSE (${MSE_WEIGHT}×)

Hypothesis:
  Pure reconstruction (24% F1) + LoRA adaptation
  → Better than Phase 1a (no LoRA)
  → Avoids Phase 1b collapse (K-token CE conflict)

Configurations tested:
EOF

for config in "${CONFIGS[@]}"; do
    echo "  - $config" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "Results:" >> "$SUMMARY_FILE"
echo "========" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run each configuration
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name r alpha layers <<< "$config"

    echo ""
    echo "=================================================="
    echo "Configuration: $name"
    echo "=================================================="
    echo "  LoRA rank (r): $r"
    echo "  LoRA alpha: $alpha"
    echo "  LoRA layers: $layers"
    echo ""

    RUN_DIR="$OUTPUT_BASE/$name"
    mkdir -p "$RUN_DIR"
    LOG_FILE="$RUN_DIR/training_$(date +"%Y%m%d_%H%M%S").log"

    # Build command for Phase 1a training
    CMD="python train_adapter_only_phase1.py \
        --model_id \"$MODEL\" \
        --dataset $DATASET \
        --samples $SAMPLES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --compress_dim $COMPRESS_DIM \
        --compress_method pca \
        --pca_samples $PCA_SAMPLES \
        --adapter_hidden_mult $ADAPTER_HIDDEN_MULT \
        --adapter_dropout $ADAPTER_DROPOUT \
        --adapter_lr $ADAPTER_LR \
        --cosine_weight $COSINE_WEIGHT \
        --mse_weight $MSE_WEIGHT \
        --eval_every 1 \
        --eval_samples 100 \
        --max_new_tokens 12 \
        --save_dir \"$RUN_DIR\" \
        --diagnostic_log \"$RUN_DIR/diagnostics.jsonl\""

    # Add LoRA flags if not baseline
    if [ "$r" != "0" ]; then
        CMD="$CMD \
        --use_lora \
        --lora_r $r \
        --lora_alpha $alpha \
        --lora_dropout 0.05"

        # Add layers if specified (0 = full model)
        if [ "$layers" != "0" ]; then
            CMD="$CMD --lora_layers $layers"
        fi
    fi

    echo "Command: $CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Run training with logging
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

    # Extract results from diagnostics
    if [ -f "$RUN_DIR/diagnostics.jsonl" ]; then
        echo "" | tee -a "$SUMMARY_FILE"
        echo "Configuration: $name" | tee -a "$SUMMARY_FILE"
        echo "  LoRA: r=$r, alpha=$alpha, layers=$layers" | tee -a "$SUMMARY_FILE"

        # Get best F1 from diagnostics
        BEST_F1=$(python -c "
import json
import sys
try:
    best_f1 = 0
    with open('$RUN_DIR/diagnostics.jsonl') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'f1' in data:
                    best_f1 = max(best_f1, data.get('f1', 0))
            except:
                pass
    print(f'{best_f1:.3f}')
except Exception as e:
    print('N/A')
" 2>/dev/null || echo "N/A")

        echo "  Best F1: $BEST_F1" | tee -a "$SUMMARY_FILE"
        echo "  Log: $LOG_FILE" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"

        echo ""
        echo "  ✓ Best F1: $BEST_F1"
    else
        echo "  ✗ No diagnostics found - check log for errors"
        echo "Configuration: $name - FAILED (no diagnostics)" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"
    fi

    echo ""
done

echo ""
echo "=================================================="
echo "SWEEP COMPLETE"
echo "=================================================="
echo ""
echo "Results summary: $SUMMARY_FILE"
echo ""
echo "To analyze results:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "To compare configurations:"
echo "  python -c \""
echo "import json"
echo "import glob"
echo "for f in sorted(glob.glob('$OUTPUT_BASE/*/diagnostics.jsonl')):"
echo "    config = f.split('/')[-2]"
echo "    best_f1 = 0"
echo "    with open(f) as fp:"
echo "        for line in fp:"
echo "            try:"
echo "                data = json.loads(line)"
echo "                if 'f1' in data:"
echo "                    best_f1 = max(best_f1, data.get('f1', 0))"
echo "            except: pass"
echo "    print(f'{config:30} F1={best_f1:.3f}')"
echo "\""
echo ""

# Create comparison script
cat > "$OUTPUT_BASE/compare_results.py" <<'COMPARE_SCRIPT'
#!/usr/bin/env python3
import json
import glob
import sys
from pathlib import Path

# Collect results
results = []
for diag_file in sorted(glob.glob('runs/phase1a_lora_sweep/*/diagnostics.jsonl')):
    config = Path(diag_file).parent.name

    best_f1 = 0.0
    best_em = 0.0
    final_loss = None

    with open(diag_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'f1' in data:
                    f1 = data.get('f1', 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_em = data.get('em', 0.0)
                if 'loss' in data:
                    final_loss = data['loss']
            except:
                pass

    results.append({
        'config': config,
        'best_f1': best_f1,
        'best_em': best_em,
        'final_loss': final_loss if final_loss is not None else float('inf')
    })

# Sort by F1 descending
results.sort(key=lambda x: x['best_f1'], reverse=True)

# Print table
print("\nPhase 1a + LoRA Sweep Results")
print("="*80)
print(f"{'Configuration':<30} {'F1':>8} {'EM':>8} {'Final Loss':>12}")
print("-"*80)

for r in results:
    loss_str = f"{r['final_loss']:.3f}" if r['final_loss'] != float('inf') else "N/A"
    print(f"{r['config']:<30} {r['best_f1']:>7.1%} {r['best_em']:>7.1%} {loss_str:>12}")

print("="*80)
print()

# Find baseline
baseline = next((r for r in results if r['config'] == 'baseline'), None)
if baseline:
    print(f"Baseline (no LoRA): F1={baseline['best_f1']:.1%}")

    # Find best LoRA config
    lora_results = [r for r in results if r['config'] != 'baseline']
    if lora_results:
        best_lora = lora_results[0]
        improvement = best_lora['best_f1'] - baseline['best_f1']
        print(f"Best LoRA config: {best_lora['config']}")
        print(f"  F1={best_lora['best_f1']:.1%} (Δ={improvement:+.1%} vs baseline)")

        if improvement > 0.02:  # 2% improvement
            print()
            print("✅ LoRA provides significant improvement!")
        elif improvement > 0:
            print()
            print("⚠️  LoRA provides marginal improvement")
        else:
            print()
            print("❌ LoRA does not improve over baseline")
else:
    print("Baseline not found - cannot compute improvements")

print()
COMPARE_SCRIPT

chmod +x "$OUTPUT_BASE/compare_results.py"

echo "Comparison script created: $OUTPUT_BASE/compare_results.py"
echo "Run with: python $OUTPUT_BASE/compare_results.py"
echo ""
