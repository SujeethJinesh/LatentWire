#!/bin/bash
#SBATCH --job-name=prod_ready_check
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/prod_ready_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/prod_ready_%j.err

# =============================================================================
# Production Readiness Check for LatentWire System
# =============================================================================
# This script performs comprehensive checks to ensure the system is ready
# for production deployment on HPC with SLURM
# =============================================================================
# Submit with: sbatch telepathy/submit_production_readiness.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create runs directory if needed
mkdir -p runs figures

# Configure git identity if not set (required for commits)
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

# Pull latest code
echo ""
echo "Pulling latest code..."
git pull || echo "Warning: Could not pull (may be offline or no changes)"

echo ""
echo "=============================================================="
echo "1. DEPENDENCY CHECK"
echo "=============================================================="
python3 -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.executable}')

# Check core dependencies
dependencies = {
    'torch': None,
    'transformers': None,
    'datasets': None,
    'accelerate': None,
    'numpy': None,
    'scipy': None,
    'sklearn': None,
    'rouge_score': None,
    'statsmodels': None,
    'pandas': None,
    'matplotlib': None,
    'seaborn': None,
    'peft': None,
    'tqdm': None,
    'sentence_transformers': None
}

failed = []
for module_name in dependencies:
    try:
        mod = __import__(module_name)
        if hasattr(mod, '__version__'):
            version = mod.__version__
        else:
            version = 'unknown'
        dependencies[module_name] = version
        print(f'✓ {module_name:25s} {version}')
    except ImportError as e:
        failed.append(module_name)
        print(f'✗ {module_name:25s} MISSING - {e}')

if failed:
    print(f'\n❌ Missing dependencies: {failed}')
    print('Install with: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('\n✅ All core dependencies installed')
" || { echo "Dependency check failed"; exit 1; }

echo ""
echo "=============================================================="
echo "2. GPU CHECK"
echo "=============================================================="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('WARNING: No GPUs detected!')
"

echo ""
echo "=============================================================="
echo "3. IMPORT CHECK"
echo "=============================================================="
echo "Testing imports from latentwire modules..."
python3 -c "
try:
    # Core modules
    from latentwire.train import ElasticGPUConfig
    from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper
    from latentwire.data import load_examples
    from latentwire.core_utils import em, f1, calibrate_to_embed_rms
    from latentwire.checkpointing import save_latest_checkpoint
    from latentwire.data_pipeline import prepare_training_data
    from latentwire.feature_registry import FeatureRegistry
    from latentwire.loss_bundles import loss_with_text_prompt_chunked
    print('✅ All core latentwire modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo ""
echo "=============================================================="
echo "4. QUICK TRAINING TEST (5 samples, 1 epoch)"
echo "=============================================================="
echo "Running minimal training to verify system integration..."

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_DIR="runs/prod_ready_test_${TIMESTAMP}"
LOG_FILE="${TEST_DIR}/training.log"

mkdir -p "$TEST_DIR"

{
    python3 latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --samples 5 \
        --epochs 1 \
        --batch_size 1 \
        --latent_len 16 \
        --d_z 128 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "$TEST_DIR" \
        --gradient_accumulation_steps 1 \
        --skip_eval \
        --disable_wandb \
        --auto_cpu_offload \
        --save_every 0
} 2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training test completed successfully"
    echo "Log saved to: $LOG_FILE"
else
    echo ""
    echo "❌ Training test failed - check log at: $LOG_FILE"
    exit 1
fi

echo ""
echo "=============================================================="
echo "5. FILE PERMISSIONS CHECK"
echo "=============================================================="
echo "Checking script permissions..."
for script in scripts/*.sh telepathy/*.slurm; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✓ $script is executable"
        else
            echo "✗ $script is NOT executable - fixing..."
            chmod +x "$script"
        fi
    fi
done

echo ""
echo "=============================================================="
echo "6. DATA ACCESS CHECK"
echo "=============================================================="
python3 -c "
from latentwire.data import load_examples

datasets_to_check = ['squad', 'hotpotqa', 'agnews', 'sst2', 'gsm8k']
for dataset in datasets_to_check:
    try:
        examples = load_examples(dataset, limit=2)
        print(f'✓ {dataset:15s} - Loaded {len(examples)} examples')
    except Exception as e:
        print(f'✗ {dataset:15s} - Failed: {e}')
"

echo ""
echo "=============================================================="
echo "7. MEMORY ESTIMATE"
echo "=============================================================="
python3 -c "
import torch

# Estimate memory for typical configuration
batch_size = 64
latent_len = 32
d_z = 256
seq_len = 512  # typical sequence length

# Model sizes (approximate)
llama_params = 8e9  # 8B parameters
qwen_params = 7e9   # 7B parameters
encoder_params = 50e6  # ~50M for encoder + adapters

# Memory per parameter (fp16)
bytes_per_param = 2

# Calculate memory requirements
model_memory = (llama_params + qwen_params + encoder_params) * bytes_per_param / 1e9
activation_memory = batch_size * seq_len * 4096 * 4 / 1e9  # rough estimate
optimizer_memory = model_memory * 2  # Adam optimizer states

total_memory = model_memory + activation_memory + optimizer_memory

print(f'Model memory: {model_memory:.1f} GB')
print(f'Activation memory (batch={batch_size}): {activation_memory:.1f} GB')
print(f'Optimizer memory: {optimizer_memory:.1f} GB')
print(f'Total estimated: {total_memory:.1f} GB')
print(f'')
print(f'Recommended SLURM memory: {int(total_memory * 1.5)} GB')
print(f'Recommended GPUs: 4 (for model parallel + data parallel)')
"

echo ""
echo "=============================================================="
echo "PRODUCTION READINESS SUMMARY"
echo "=============================================================="

# Check if we got this far
if [ $? -eq 0 ]; then
    echo "✅ System is READY for production deployment"
    echo ""
    echo "Recommended SLURM settings:"
    echo "  --gpus=4"
    echo "  --mem=256GB"
    echo "  --time=12:00:00"
    echo "  --partition=preempt"
    echo "  --account=marlowe-m000066"
    echo ""
    echo "Next steps:"
    echo "1. Submit training job: sbatch telepathy/submit_enhanced_arxiv.slurm"
    echo "2. Monitor progress: squeue -u \$USER"
    echo "3. Check logs: tail -f runs/slurm_*.log"
else
    echo "❌ System has issues - review output above"
    exit 1
fi

# Clean up test artifacts
rm -rf "$TEST_DIR"

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="