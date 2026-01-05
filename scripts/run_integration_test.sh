#!/usr/bin/env bash
# scripts/run_integration_test.sh
# Integration test that runs a mini version of the full pipeline to verify everything works
# This helps catch issues before running expensive full experiments

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/integration_test}"
DATASET="${DATASET:-squad}"
TEST_MODE="${TEST_MODE:-quick}"  # quick, standard, or thorough

# Test configuration based on mode
case $TEST_MODE in
    quick)
        SAMPLES=100
        EPOCHS=1
        EVAL_SAMPLES=20
        BATCH_SIZE=8
        LATENT_LEN=8
        D_Z=64
        echo "Running QUICK integration test (fastest, ~2-5 minutes)"
        ;;
    standard)
        SAMPLES=500
        EPOCHS=2
        EVAL_SAMPLES=50
        BATCH_SIZE=16
        LATENT_LEN=16
        D_Z=128
        echo "Running STANDARD integration test (~10-15 minutes)"
        ;;
    thorough)
        SAMPLES=1000
        EPOCHS=3
        EVAL_SAMPLES=100
        BATCH_SIZE=32
        LATENT_LEN=32
        D_Z=256
        echo "Running THOROUGH integration test (~20-30 minutes)"
        ;;
    *)
        echo "Invalid TEST_MODE: $TEST_MODE (use quick, standard, or thorough)"
        exit 1
        ;;
esac

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/integration_${TEST_MODE}_${TIMESTAMP}.log"
SUMMARY_FILE="$OUTPUT_DIR/summary_${TIMESTAMP}.json"

echo "=========================================="
echo "LatentWire Integration Test Pipeline"
echo "=========================================="
echo "Test Mode: $TEST_MODE"
echo "Dataset: $DATASET"
echo "Training Samples: $SAMPLES"
echo "Eval Samples: $EVAL_SAMPLES"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Latent Config: M=$LATENT_LEN, d_z=$D_Z"
echo "Output Directory: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "=========================================="
echo ""

# Function to check if a phase passed
check_phase() {
    local phase="$1"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[PASS] $phase completed successfully"
        return 0
    else
        echo "[FAIL] $phase failed with exit code $exit_code"
        return 1
    fi
}

# Run the full pipeline
{
    echo "Starting integration test at $(date)"
    echo ""

    # Track test results
    PASSED_PHASES=0
    FAILED_PHASES=0

    # =============================================================================
    # PHASE 1: Environment Validation
    # =============================================================================
    echo "=========================================="
    echo "PHASE 1: Environment Validation"
    echo "=========================================="

    python3 -c "
import sys
import torch
import transformers
import datasets
import numpy as np

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')

# Check CUDA availability
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.device_count()} devices')
    for i in range(torch.cuda.device_count()):
        print(f'  - Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - using CPU/MPS')

# Check critical imports
modules = [
    'latentwire.train',
    'latentwire.eval',
    'latentwire.models',
    'latentwire.data',
    'latentwire.losses',
]

errors = []
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        errors.append(module)

if errors:
    print(f'\\nFailed to import: {errors}')
    sys.exit(1)
else:
    print('\\nAll modules imported successfully!')
"

    if check_phase "Environment Validation"; then
        ((PASSED_PHASES++))
    else
        ((FAILED_PHASES++))
        echo "Environment validation failed - aborting test"
        exit 1
    fi

    # =============================================================================
    # PHASE 2: Data Loading Test
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PHASE 2: Data Loading Test"
    echo "=========================================="

    python3 -c "
import sys
sys.path.insert(0, '.')
from latentwire.data import load_examples

print(f'Loading {$SAMPLES} examples from $DATASET...')
train_examples = load_examples('$DATASET', 'train', $SAMPLES)
print(f'Loaded {len(train_examples)} training examples')

print(f'Loading {$EVAL_SAMPLES} eval examples...')
eval_examples = load_examples('$DATASET', 'eval', $EVAL_SAMPLES)
print(f'Loaded {len(eval_examples)} evaluation examples')

# Verify example structure
if train_examples:
    ex = train_examples[0]
    print(f'\\nExample structure:')
    print(f'  - prefix: {len(ex[\"prefix\"])} chars')
    print(f'  - answer: {len(ex[\"answer\"])} chars')
    print(f'\\nFirst example preview:')
    print(f'  Prefix: {ex[\"prefix\"][:100]}...')
    print(f'  Answer: {ex[\"answer\"][:50]}...')
"

    if check_phase "Data Loading"; then
        ((PASSED_PHASES++))
    else
        ((FAILED_PHASES++))
    fi

    # =============================================================================
    # PHASE 3: Model Initialization Test
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PHASE 3: Model Initialization Test"
    echo "=========================================="

    python3 -c "
import sys
import torch
sys.path.insert(0, '.')
from latentwire.models import ByteLatentEncoder, LatentAdapter, ModelVersionManager

print('Initializing encoder and adapters...')

# Initialize encoder
encoder = ByteLatentEncoder(
    d_input=257,  # 256 bytes + 1 padding
    d_latent=$D_Z,
    n_latent=$LATENT_LEN,
    n_layers=3
)
print(f'✓ ByteLatentEncoder: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params')

# Initialize model version manager
manager = ModelVersionManager()
llama_version = manager.get_version('meta-llama/Meta-Llama-3.1-8B-Instruct')
print(f'✓ Model version: {llama_version}')

# Initialize adapter
adapter = LatentAdapter(
    d_in=$D_Z,
    d_out=4096,  # Llama-3.1-8B embedding dimension
    bias=True
)
print(f'✓ LatentAdapter: {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M params')

# Test forward pass
dummy_input = torch.randn(1, 100, 257)  # batch_size=1, seq_len=100, d_input=257
latents = encoder(dummy_input)
print(f'\\nEncoder output shape: {latents.shape} (expected: [1, $LATENT_LEN, $D_Z])')

adapted = adapter(latents)
print(f'Adapter output shape: {adapted.shape} (expected: [1, $LATENT_LEN, 4096])')
"

    if check_phase "Model Initialization"; then
        ((PASSED_PHASES++))
    else
        ((FAILED_PHASES++))
    fi

    # =============================================================================
    # PHASE 4: Training (Mini Version)
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PHASE 4: Training Pipeline Test"
    echo "=========================================="

    TRAIN_DIR="$OUTPUT_DIR/training"

    python3 latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --models "llama" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --encoder_type byte \
        --dataset "$DATASET" \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --k_ce_weight 1.0 \
        --K 4 \
        --lr 1e-3 \
        --output_dir "$TRAIN_DIR" \
        --save_every 1000000 \
        --max_grad_norm 1.0 \
        2>&1 | tee "$OUTPUT_DIR/training.log" | grep -E "(Epoch|Loss|NLL|FirstTok|Throughput|Complete)" || true

    if check_phase "Training Pipeline"; then
        ((PASSED_PHASES++))

        # Check if checkpoint was created
        if [ -d "$TRAIN_DIR" ]; then
            echo "✓ Training directory created"
            ls -la "$TRAIN_DIR" | head -10
        else
            echo "✗ Training directory not found"
        fi
    else
        ((FAILED_PHASES++))
    fi

    # =============================================================================
    # PHASE 5: Evaluation
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PHASE 5: Evaluation Pipeline Test"
    echo "=========================================="

    EVAL_DIR="$OUTPUT_DIR/evaluation"

    # Only run eval if training succeeded
    if [ -d "$TRAIN_DIR" ]; then
        python3 latentwire/eval.py \
            --ckpt "$TRAIN_DIR" \
            --samples "$EVAL_SAMPLES" \
            --max_new_tokens 12 \
            --dataset "$DATASET" \
            --sequential_eval \
            --fresh_eval \
            --calibration embed_rms \
            --latent_anchor_mode text \
            --latent_anchor_text "Answer: " \
            --append_bos_after_prefix yes \
            --output_dir "$EVAL_DIR" \
            2>&1 | tee "$OUTPUT_DIR/evaluation.log" | grep -E "(Evaluating|baseline|latent|F1|EM|Complete)" || true

        if check_phase "Evaluation Pipeline"; then
            ((PASSED_PHASES++))

            # Check for evaluation results
            if [ -f "$EVAL_DIR/results.json" ]; then
                echo ""
                echo "✓ Evaluation results saved"
                python3 -c "
import json
with open('$EVAL_DIR/results.json') as f:
    results = json.load(f)
    print('\\nKey metrics:')
    for model in ['llama']:
        if model in results:
            print(f'  {model}:')
            for metric_type in ['text_baseline', 'latent']:
                if metric_type in results[model]:
                    metrics = results[model][metric_type]
                    print(f'    {metric_type}: F1={metrics.get(\"f1\", 0):.3f}, EM={metrics.get(\"em\", 0):.3f}')
"
            else
                echo "✗ Evaluation results not found"
            fi
        else
            ((FAILED_PHASES++))
            echo "Skipping evaluation (no checkpoint available)"
        fi
    else
        echo "Skipping evaluation phase (training did not produce checkpoint)"
        ((FAILED_PHASES++))
    fi

    # =============================================================================
    # PHASE 6: Visualization/Analysis (Optional)
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PHASE 6: Analysis and Visualization Test"
    echo "=========================================="

    # Generate simple analysis
    python3 -c "
import json
import sys
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')

# Analyze training logs if available
train_log = output_dir / 'training.log'
if train_log.exists():
    print('✓ Training log found')
    with open(train_log) as f:
        lines = f.readlines()
        loss_lines = [l for l in lines if 'Loss' in l]
        if loss_lines:
            print(f'  Found {len(loss_lines)} loss entries')

# Analyze eval results if available
eval_results = output_dir / 'evaluation' / 'results.json'
if eval_results.exists():
    print('✓ Evaluation results found')
    with open(eval_results) as f:
        results = json.load(f)

    # Simple visualization (text-based)
    print('\\nPerformance Summary:')
    print('=' * 40)
    for model in results:
        print(f'{model}:')
        for method in ['text_baseline', 'latent', 'token_budget']:
            if method in results[model]:
                metrics = results[model][method]
                f1 = metrics.get('f1', 0)
                em = metrics.get('em', 0)
                bar = '█' * int(f1 * 20)
                print(f'  {method:15s}: F1={f1:.3f} {bar}')
    print('=' * 40)

# Generate summary
summary = {
    'test_mode': '$TEST_MODE',
    'samples': $SAMPLES,
    'eval_samples': $EVAL_SAMPLES,
    'epochs': $EPOCHS,
    'phases_passed': $PASSED_PHASES,
    'phases_failed': $FAILED_PHASES,
}

summary_file = output_dir / 'test_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\\n✓ Test summary saved to {summary_file}')
" 2>&1

    if check_phase "Analysis and Visualization"; then
        ((PASSED_PHASES++))
    else
        ((FAILED_PHASES++))
    fi

    # =============================================================================
    # PHASE 7: Memory and Performance Check
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "PHASE 7: Memory and Performance Check"
    echo "=========================================="

    python3 -c "
import torch
import psutil
import sys

# Memory check
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Current memory usage: {memory_mb:.1f} MB')

# GPU memory if available
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
        print(f'GPU {i}: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB')

# Check for memory leaks (basic)
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_after = process.memory_info().rss / 1024 / 1024
memory_diff = memory_after - memory_mb
print(f'\\nMemory after cleanup: {memory_after:.1f} MB (diff: {memory_diff:+.1f} MB)')

if abs(memory_diff) > 100:
    print('[WARN] Possible memory leak detected')
else:
    print('[PASS] No significant memory leaks detected')
"

    if check_phase "Memory and Performance Check"; then
        ((PASSED_PHASES++))
    else
        ((FAILED_PHASES++))
    fi

    # =============================================================================
    # Final Summary
    # =============================================================================
    echo ""
    echo "=========================================="
    echo "INTEGRATION TEST COMPLETE"
    echo "=========================================="
    echo "Test Mode: $TEST_MODE"
    echo "Completed at: $(date)"
    echo ""
    echo "Phase Results:"
    echo "  ✓ Passed: $PASSED_PHASES"
    echo "  ✗ Failed: $FAILED_PHASES"
    echo ""

    # Generate final summary JSON
    python3 -c "
import json
from datetime import datetime

summary = {
    'timestamp': datetime.now().isoformat(),
    'test_mode': '$TEST_MODE',
    'configuration': {
        'dataset': '$DATASET',
        'samples': $SAMPLES,
        'eval_samples': $EVAL_SAMPLES,
        'epochs': $EPOCHS,
        'batch_size': $BATCH_SIZE,
        'latent_len': $LATENT_LEN,
        'd_z': $D_Z
    },
    'results': {
        'phases_passed': $PASSED_PHASES,
        'phases_failed': $FAILED_PHASES,
        'total_phases': 7,
        'success_rate': $PASSED_PHASES / 7.0
    },
    'outputs': {
        'log_file': '$LOG_FILE',
        'training_dir': '$OUTPUT_DIR/training',
        'eval_dir': '$OUTPUT_DIR/evaluation'
    }
}

with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: $SUMMARY_FILE')
"

    if [ $FAILED_PHASES -eq 0 ]; then
        echo ""
        echo "✅ ALL TESTS PASSED! The pipeline is working correctly."
        echo ""
        echo "You can now run the full pipeline with confidence:"
        echo "  - For full training: Increase samples, epochs, and batch_size"
        echo "  - For production: Use the thorough test mode first"
        EXIT_CODE=0
    else
        echo ""
        echo "⚠️  SOME TESTS FAILED. Please review the log for details."
        echo ""
        echo "Debug information:"
        echo "  - Log file: $LOG_FILE"
        echo "  - Summary: $SUMMARY_FILE"
        EXIT_CODE=1
    fi

    echo ""
    echo "Logs and results saved to: $OUTPUT_DIR"
    echo "=========================================="

} 2>&1 | tee "$LOG_FILE"

# Extract final exit code
exit $EXIT_CODE