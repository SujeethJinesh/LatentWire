# DEPLOYMENT CHECKLIST

**Purpose**: Final go/no-go verification before launching full experiments
**Last Updated**: January 2026
**Status**: Ready for verification

---

## 🔴 CRITICAL PRE-FLIGHT CHECKS

### □ Python Environment Ready

**How to verify:**
```bash
# On HPC
module load python/3.10
python --version
which python
```

**Expected outcome:**
- Python 3.10+ installed
- Correct module loaded
- Path points to HPC Python, not system Python

**Fix if failed:**
```bash
module purge
module load python/3.10
# Add to ~/.bashrc: module load python/3.10
```

---

### □ All Packages Installed

**How to verify:**
```bash
# On HPC
cd /projects/m000066/sujinesh/LatentWire
python -c "
import torch
import transformers
import datasets
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('Transformers:', transformers.__version__)
print('All imports successful!')
"
```

**Expected outcome:**
- All imports succeed
- PyTorch 2.0+ with CUDA support
- 4 GPUs detected
- No import errors

**Fix if failed:**
```bash
pip install --user -r requirements.txt
# Or if requirements.txt missing:
pip install --user torch transformers datasets accelerate scikit-learn pandas matplotlib seaborn
```

---

### □ Models Accessible

**How to verify:**
```bash
# Test model downloads
cd /projects/m000066/sujinesh/LatentWire
python -c "
from transformers import AutoModel, AutoTokenizer
import torch

models = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct'
]

for model_id in models:
    print(f'Testing {model_id}...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Just load config, not full model (saves time/memory)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f'✓ {model_id} accessible')
"
```

**Expected outcome:**
- Both models download successfully
- No authentication errors
- Configs load without issues

**Fix if failed:**
```bash
# Set up Hugging Face token
export HF_TOKEN="your_token_here"
huggingface-cli login --token $HF_TOKEN

# Or use cache if models pre-downloaded:
export HF_HOME=/projects/m000066/sujinesh/.cache/huggingface
```

---

### □ Datasets Downloadable

**How to verify:**
```bash
cd /projects/m000066/sujinesh/LatentWire
python -c "
from datasets import load_dataset

datasets_to_test = [
    ('squad', None),
    ('hotpot_qa', 'distractor'),
    ('ag_news', None),
    ('stanfordnlp/sst2', None),
    ('gsm8k', 'main'),
    ('trec', None)
]

for name, config in datasets_to_test:
    print(f'Testing {name}...')
    ds = load_dataset(name, config, split='train', streaming=True)
    sample = next(iter(ds))
    print(f'✓ {name} accessible, keys: {list(sample.keys())}')
"
```

**Expected outcome:**
- All datasets load without errors
- Sample data retrievable
- Correct fields present

**Fix if failed:**
```bash
# Clear dataset cache if corrupted
rm -rf ~/.cache/huggingface/datasets

# Set custom cache location if needed
export HF_DATASETS_CACHE=/projects/m000066/sujinesh/.cache/datasets
```

---

### □ GPU Allocation Works

**How to verify:**
```bash
# Submit test job
cd /projects/m000066/sujinesh/LatentWire
cat << 'EOF' > test_gpu.slurm
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=00:05:00
#SBATCH --mem=64GB

nvidia-smi
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
EOF

sbatch test_gpu.slurm
# Wait and check output
sleep 30
tail -f slurm-*.out
```

**Expected outcome:**
- Job starts within 5 minutes
- 4 H100 GPUs detected
- CUDA available = True
- nvidia-smi shows all GPUs

**Fix if failed:**
```bash
# Check SLURM account/partition
sacctmgr show assoc where user=$USER

# Verify GPU partition access
sinfo -p preempt -o "%20N %10c %10m %25G %10t"

# Try different partition if needed
#SBATCH --partition=gpu  # fallback option
```

---

### □ Checkpoint System Tested

**How to verify:**
```bash
cd /projects/m000066/sujinesh/LatentWire
python -c "
import torch
import os
from pathlib import Path

# Test checkpoint save/load
test_dir = Path('runs/test_checkpoint')
test_dir.mkdir(parents=True, exist_ok=True)

# Save test checkpoint
checkpoint = {
    'epoch': 1,
    'model_state': {'test': torch.randn(10, 10)},
    'optimizer_state': {'lr': 0.001},
    'metrics': {'loss': 0.5}
}
torch.save(checkpoint, test_dir / 'checkpoint.pt')
print(f'✓ Saved checkpoint to {test_dir}')

# Load and verify
loaded = torch.load(test_dir / 'checkpoint.pt')
assert loaded['epoch'] == 1
assert loaded['metrics']['loss'] == 0.5
print('✓ Checkpoint load successful')

# Clean up
import shutil
shutil.rmtree(test_dir)
print('✓ Cleanup successful')
"
```

**Expected outcome:**
- Checkpoint saves without errors
- Can be loaded back correctly
- File I/O permissions work
- Cleanup succeeds

**Fix if failed:**
```bash
# Check disk space
df -h /projects/m000066/sujinesh

# Check permissions
ls -la /projects/m000066/sujinesh/LatentWire/runs/

# Create runs directory if missing
mkdir -p /projects/m000066/sujinesh/LatentWire/runs
chmod 755 /projects/m000066/sujinesh/LatentWire/runs
```

---

### □ Logging Verified

**How to verify:**
```bash
cd /projects/m000066/sujinesh/LatentWire
# Test logging with tee pattern
{
    echo "Test message 1"
    echo "Error message" >&2
    python -c "print('Python output')"
    python -c "import sys; sys.stderr.write('Python error\n')"
} 2>&1 | tee test_log.txt

# Verify capture
echo "--- Checking log contents ---"
cat test_log.txt
rm test_log.txt
```

**Expected outcome:**
- All messages appear on screen AND in log file
- Both stdout and stderr captured
- Tee command works correctly

**Fix if failed:**
```bash
# Ensure tee is available
which tee

# Test alternative logging
script -c "your_command" output.log  # alternative to tee
```

---

### □ Preemption Handling Tested

**How to verify:**
```bash
# Check if preemptible partition works
sinfo -p preempt

# Submit test job with checkpoint saving
cd /projects/m000066/sujinesh/LatentWire
cat << 'EOF' > test_preempt.py
import time
import torch
from pathlib import Path

checkpoint_dir = Path('runs/preempt_test')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for i in range(10):
    print(f"Iteration {i}")
    time.sleep(2)

    # Save checkpoint
    if i % 3 == 0:
        torch.save({'iteration': i}, checkpoint_dir / f'ckpt_{i}.pt')
        print(f"Saved checkpoint {i}")

print("Job completed normally")
EOF

# Submit with preemption test
sbatch --time=00:02:00 --wrap="python test_preempt.py"
```

**Expected outcome:**
- Job runs in preempt partition
- Checkpoints saved periodically
- Can resume from checkpoint if preempted

**Fix if failed:**
```bash
# Add signal handler for preemption
# In Python scripts, add:
import signal
def handle_preemption(signum, frame):
    save_checkpoint()
    sys.exit(0)
signal.signal(signal.SIGTERM, handle_preemption)
```

---

### □ Results Aggregation Works

**How to verify:**
```bash
cd /projects/m000066/sujinesh/LatentWire
python -c "
import json
from pathlib import Path

# Test result aggregation
test_results = {
    'experiment': 'test',
    'metrics': {
        'accuracy': 0.95,
        'f1': 0.93,
        'loss': 0.15
    },
    'config': {
        'model': 'llama',
        'epochs': 10
    }
}

# Save test results
results_dir = Path('runs/test_results')
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / 'results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print('✓ Saved results.json')

# Test aggregation across multiple files
for i in range(3):
    with open(results_dir / f'run_{i}.json', 'w') as f:
        json.dump({'run': i, 'score': 0.8 + i*0.05}, f)

# Aggregate
all_results = []
for json_file in results_dir.glob('run_*.json'):
    with open(json_file) as f:
        all_results.append(json.load(f))

avg_score = sum(r['score'] for r in all_results) / len(all_results)
print(f'✓ Aggregated {len(all_results)} results, avg score: {avg_score:.3f}')

# Cleanup
import shutil
shutil.rmtree(results_dir)
"
```

**Expected outcome:**
- JSON files save correctly
- Can read and aggregate multiple results
- Statistics computed correctly

**Fix if failed:**
```bash
# Ensure JSON module works
python -c "import json; print(json.__version__)"

# Check file permissions
umask 022  # Ensure files are readable
```

---

### □ Paper Generation Pipeline Tested

**How to verify:**
```bash
cd /projects/m000066/sujinesh/LatentWire/finalization

# Test LaTeX compilation (if available)
which pdflatex || echo "LaTeX not installed - will generate tables only"

# Test figure generation
python -c "
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Test plot
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='Test')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
fig.savefig('test_figure.pdf', dpi=150, bbox_inches='tight')
fig.savefig('test_figure.png', dpi=150, bbox_inches='tight')
print('✓ Figure generation works')

import os
os.remove('test_figure.pdf')
os.remove('test_figure.png')
"
```

**Expected outcome:**
- Matplotlib works with Agg backend
- Can save PDF and PNG figures
- No display errors on headless system

**Fix if failed:**
```bash
# Set matplotlib backend
export MPLBACKEND=Agg
echo 'export MPLBACKEND=Agg' >> ~/.bashrc

# Install if missing
pip install --user matplotlib seaborn
```

---

## 🟢 GO/NO-GO DECISION MATRIX

| Component | Status | Critical? | Notes |
|-----------|--------|-----------|-------|
| Python Environment | ⬜ | ✓ | Must be 3.10+ |
| Package Installation | ⬜ | ✓ | All deps required |
| Model Access | ⬜ | ✓ | Both models needed |
| Dataset Access | ⬜ | ✓ | All 6 datasets |
| GPU Allocation | ⬜ | ✓ | 4x H100 required |
| Checkpoint System | ⬜ | ✓ | Must save/load |
| Logging | ⬜ | ✓ | Debug impossible without |
| Preemption | ⬜ | ⚠️ | Important but not blocking |
| Results Aggregation | ⬜ | ⚠️ | Can manually aggregate |
| Paper Pipeline | ⬜ | ⚠️ | Can generate locally |

**Decision Rules:**
- **GO**: All ✓ items checked
- **CONDITIONAL GO**: All ✓ items checked, some ⚠️ items have workarounds
- **NO GO**: Any ✓ item fails

---

## 🚀 FINAL LAUNCH SEQUENCE

Once all checks pass:

```bash
# 1. Final git sync
cd /projects/m000066/sujinesh/LatentWire
git pull

# 2. Launch main experiments
sbatch finalization/slurm/final_main_experiments.slurm

# 3. Monitor progress
watch -n 60 'squeue -u $USER'
tail -f runs/final_main_*.log

# 4. Launch follow-ups after main completes
# (wait for main to finish first)
sbatch finalization/slurm/final_ablations.slurm
sbatch finalization/slurm/final_scaling.slurm

# 5. Generate final report
cd finalization
python aggregate_final_results.py
python generate_latex_tables.py
```

---

## 📊 SUCCESS CRITERIA

**Minimum viable results:**
- [ ] Main experiments complete for 3+ models
- [ ] Ablations show clear trends
- [ ] Scaling law visible (3+ points)
- [ ] Results JSON files generated
- [ ] Logs captured for all runs

**Full success:**
- [ ] All 6 datasets evaluated
- [ ] All architectural variants tested
- [ ] Compression ratios computed
- [ ] Statistical significance calculated
- [ ] LaTeX tables generated
- [ ] Figures created

---

## 🔧 EMERGENCY PROCEDURES

### If jobs won't start:
```bash
# Check queue
squeue -p preempt
# Try different partition
sed -i 's/preempt/gpu/g' *.slurm
```

### If GPU OOM:
```bash
# Reduce batch size
sed -i 's/--batch_size 64/--batch_size 32/g' scripts/*.sh
# Reduce model parallelism
export CUDA_VISIBLE_DEVICES=0,1  # Use only 2 GPUs
```

### If preempted frequently:
```bash
# Switch to dedicated partition (if available)
#SBATCH --partition=gpu
#SBATCH --qos=normal
```

### If results corrupted:
```bash
# Re-run specific experiment
sbatch --array=3 finalization/slurm/final_main_experiments.slurm
```

---

## ✅ FINAL SIGN-OFF

**Pre-launch checklist completed:**
- Date: _____________
- Verified by: _____________
- GO / NO-GO decision: _____________
- Notes: _____________

**Post-launch monitoring:**
- Jobs submitted: _____________
- Jobs completed: _____ / _____
- Failures requiring rerun: _____________
- Final results ready: _____________