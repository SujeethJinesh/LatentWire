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

# Minimal supervision (pure reconstruction style)
FIRST_TOKEN_CE_WEIGHT="0.01"    # Tiny CE weight (not 0.0 to avoid division by zero)
KD_WEIGHT="0.0"                 # No knowledge distillation
K_TOKENS="4"                    # Minimal supervision window

# LoRA sweep configurations
# Format: name:rank:alpha:firstN
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
    "r8_a16_full:8:16:32"
    "r16_a32_full:16:32:32"

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
echo "  Supervision (minimal, like Phase 1a):"
echo "    First-token CE weight: $FIRST_TOKEN_CE_WEIGHT (vs 0.5 in Phase 1b)"
echo "    KD weight: $KD_WEIGHT (disabled)"
echo "    K tokens: $K_TOKENS"
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
  First-token CE: $FIRST_TOKEN_CE_WEIGHT (minimal supervision)
  KD weight: $KD_WEIGHT (disabled)
  K tokens: $K_TOKENS

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
    IFS=':' read -r name r alpha firstN <<< "$config"

    echo ""
    echo "=================================================="
    echo "Configuration: $name"
    echo "=================================================="
    echo "  LoRA rank (r): $r"
    echo "  LoRA alpha: $alpha"
    echo "  LoRA layers (firstN): $firstN"
    echo ""

    RUN_DIR="$OUTPUT_BASE/$name"
    mkdir -p "$RUN_DIR"
    LOG_FILE="$RUN_DIR/training_$(date +"%Y%m%d_%H%M%S").log"

    # Build command
    CMD="python latentwire/train.py \
        --llama_id \"$MODEL\" \
        --models llama \
        --dataset $DATASET \
        --samples $SAMPLES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --encoder_type byte \
        --latent_len 32 \
        --d_z 256 \
        --first_token_ce_weight $FIRST_TOKEN_CE_WEIGHT \
        --kd_first_k_weight $KD_WEIGHT \
        --K $K_TOKENS \
        --lr 1e-4 \
        --max_answer_tokens 32 \
        --diagnostic_log \"$RUN_DIR/diagnostics.jsonl\""

    # Add LoRA flags if not baseline
    if [ "$r" != "0" ]; then
        CMD="$CMD \
        --use_lora \
        --lora_r $r \
        --lora_alpha $alpha \
        --lora_dropout 0.05"

        # Add firstN if not full model
        if [ "$firstN" != "32" ]; then
            CMD="$CMD --lora_firstN $firstN"
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
        echo "  LoRA: r=$r, alpha=$alpha, firstN=$firstN" | tee -a "$SUMMARY_FILE"

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
                if 'text_f1' in data:
                    best_f1 = max(best_f1, data.get('text_f1', 0))
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
echo "                if 'text_f1' in data:"
echo "                    best_f1 = max(best_f1, data.get('text_f1', 0))"
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
                if 'text_f1' in data:
                    f1 = data.get('text_f1', 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_em = data.get('text_em', 0.0)
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
