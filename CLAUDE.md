# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles

### 0. FIX ROOT CAUSES - DON'T SKIP OR WORKAROUND

**When encountering bugs, OOMs, or performance issues:**

✅ **DO**:
- Fix the actual root cause (use IncrementalPCA for memory issues, add proper batching, optimize algorithms)
- Make features work correctly and efficiently
- Use proper data structures and algorithms for the problem
- Optimize memory usage through better design

❌ **DON'T**:
- Skip features with flags like `SKIP_PCA=yes` because they OOM
- Remove functionality because it's "hard to fix"
- Add workarounds that hide problems
- Give up when something is slow or memory-intensive

**Examples:**
- ❌ PCA OOMs → Don't skip it
- ✅ PCA OOMs → Use IncrementalPCA for batch processing
- ❌ Eval too slow → Don't reduce samples
- ✅ Eval too slow → Add proper batching and parallelization

**Deliberate design choices are different:**
- Skipping Qwen to focus on Llama first = valid optimization choice
- Skipping PCA because it OOMs = bug that needs fixing

## Critical Workflow Requirements

**ALWAYS follow these requirements when working on this codebase:**

### 1. End-to-End Execution Model (CRITICAL)
**Scripts must work completely from scratch with NO pre-existing checkpoints or data:**

- **Standard run pattern**: `git pull && rm -rf runs && PYTHONPATH=. bash <script.sh>`
- **Checkpoints are NOT saved between runs** - `rm -rf runs` deletes everything
- **Scripts MUST be fully self-contained and work end-to-end**
- **NO synthetic data or synthetic tests** - only real data from real training

**What this means for script design:**
1. **Diagnostic/analysis scripts must TRAIN first, then analyze**
   - Example: `run_embedding_diagnostics.sh` trains a quick checkpoint, then analyzes it
   - Cannot assume any pre-existing checkpoints exist
2. **Scripts cannot rely on previous runs** - each execution starts fresh
3. **All data must be generated from real training** - no synthetic or mocked data
4. **Training must be fast enough for iteration** - use small sample counts for diagnostics

**Example end-to-end script structure:**
```bash
# PHASE 1: Generate real data (train checkpoint)
python latentwire/train.py --samples 1000 --epochs 1 --output_dir runs/diagnostic/checkpoint

# PHASE 2: Analyze real data from checkpoint
python scripts/analyze.py --checkpoint runs/diagnostic/checkpoint
```

**This is NON-NEGOTIABLE:** Scripts that require pre-existing checkpoints will fail in the standard workflow.

### 1. Logging and Output Capture (MANDATORY)
- **ALL scripts MUST use `tee` to capture output to log files**
- Log files should be timestamped: `{script_name}_$(date +"%Y%m%d_%H%M%S").log`
- Both stdout and stderr must be captured: `{ command } 2>&1 | tee "$LOG_FILE"`
- Log files must be saved in the same directory as other results
- **Without logs, you are flying blind** - analysis is impossible
- **This is a NON-NEGOTIABLE requirement for ALL scripts**

**Mandatory Template for ALL Bash Scripts:**
```bash
#!/usr/bin/env bash
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/experiment_name}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/script_name_${TIMESTAMP}.log"

echo "Starting experiment..."
echo "Log file: $LOG_FILE"
echo ""

# Run command with tee to capture ALL output
{
    python scripts/your_script.py \
        --arg1 value1 \
        --arg2 value2 \
        --output_dir "$OUTPUT_DIR"
} 2>&1 | tee "$LOG_FILE"

echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/results.json"
echo "  - $LOG_FILE"
```

**When creating ANY new script:**
1. Start with this template
2. Replace `script_name` with actual script name
3. Add any configuration variables needed
4. Wrap the main command in `{ } 2>&1 | tee "$LOG_FILE"`
5. Never skip this - it's required for debugging and analysis

**For Python Scripts:**
All Python scripts should:
- Print progress updates to stdout (these will be captured by tee)
- Save structured results to JSON files
- Include timing information in output
- Log configuration at start of execution

Example Python script structure:
```python
import time
from datetime import datetime

def main():
    start_time = time.time()
    print(f"Starting {script_name} at {datetime.now().isoformat()}")
    print(f"Configuration: {config}")

    # ... do work with progress updates ...
    print(f"Processing batch {i}/{total}...")

    # Save results
    results = {...}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nComplete! Total time: {total_time:.2f}s")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
```

**Checklist for ALL Scripts:**
Before considering ANY script complete, verify:
- [ ] Bash wrapper uses `{ } 2>&1 | tee "$LOG_FILE"` pattern
- [ ] Log file is timestamped and saved in output directory
- [ ] Python script prints progress updates to stdout
- [ ] Results saved to JSON files with comprehensive data
- [ ] Timing information included in output
- [ ] Configuration logged at start
- [ ] Script committed and pushed to git

**If a script doesn't have tee logging, it is INCOMPLETE and must be fixed immediately.**

### 2. Version Control
- **ALWAYS commit after completing a task or fix**
- **ALWAYS push commits to remote immediately**
- Use descriptive commit messages that explain what was fixed/changed
- Never leave uncommitted changes unless explicitly asked to pause

### 3. Communication Style
- **Be matter of fact** - no superlatives, no fluff, no unnecessary praise
- Focus on technical accuracy and objectivity
- State what was done, what the results are, and what needs to happen next
- If something is uncertain, investigate first rather than speculating

### 4. Hardware Awareness
- This project has access to **4 H100 GPUs** on HPC
- Design for large-scale batch processing and high throughput
- Don't artificially limit sample sizes or batch sizes
- Use parallel processing and efficient memory management

### 5. Script Design
- Every experiment script must save comprehensive results
- JSON files for structured data, log files for execution traces
- Include timing, throughput metrics, and configuration in logs
- Make results easy to analyze programmatically

## Commands for Development

### Important Note on Training Infrastructure
**Training runs on a remote server** - checkpoints, logs, and diagnostics are saved there. When analyzing training progress locally, only the synced log files (diagnostics.jsonl, pipeline_*.log) are available. Checkpoint files themselves remain on the training server.

### Running Training & Evaluation Pipeline
```bash
# Main pipeline script - handles training and evaluation
bash scripts/run_pipeline.sh

# Key environment variables to set before running:
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac MPS
export PYTHONPATH=.

# Common training command:
python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
  --samples 87599 \
  --epochs 24 \
  --batch_size 64 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --dataset squad \
  --sequential_models \
  --warm_anchor_text "Answer: " \
  --first_token_ce_weight 0.5

# Common evaluation command:
python latentwire/eval.py \
  --ckpt runs/{RUN_NAME}/epoch{N} \
  --samples 200 \
  --max_new_tokens 12 \
  --dataset squad \
  --sequential_eval \
  --fresh_eval \
  --calibration embed_rms \
  --latent_anchor_mode text \
  --latent_anchor_text "Answer: " \
  --append_bos_after_prefix yes
```

### Testing & Validation
```bash
# Run sanity checks for tokenization alignment
python -c "from latentwire.train import _assert_t0_alignment; _assert_t0_alignment('meta-llama/Meta-Llama-3.1-8B-Instruct')"

# Quick smoke test
python latentwire/eval.py --ckpt {checkpoint} --samples 10 --debug
```

## High-Level Architecture

### Core System Design
LatentWire implements a **continuous interlingua** - a learned compressed representation that can condition multiple heterogeneous LLMs (Llama and Qwen) without retokenizing text. Key insights:

1. **Frozen LLMs**: The base models (Llama, Qwen) remain completely frozen. Only small adapters and an encoder are trained.

2. **Shared Latent Space**: A single encoder produces `Z ∈ R^{M × d_z}` soft tokens that both models consume via `inputs_embeds`.

3. **Model-Specific Adapters**: Small linear adapters map the shared latent into each model's embedding space while preserving statistical properties.

### Key Training Innovations

#### Phase A Improvements (from PLAN.md)
- **K-token teacher-forced CE** (`k_token_ce_from_prefix` in losses.py): Supervises first K tokens instead of just one
- **Prefix KD** (`kd_first_k_prefix_vs_text`): Distills text-prompted teacher distributions
- **Proper tokenization alignment**: Ensures exact t=0 alignment after anchor text
- **Per-example calibration**: Scales latents to match embedding RMS per example

#### Critical Bug Fixes (from LOG.md)
- **PAD token masking**: Fixed gradient contamination from left-padded tokens
- **BOS policy alignment**: Consistent BOS handling between train and eval
- **Anchor text consistency**: Same "Answer: " anchor used throughout

### Module Organization

```
latentwire/
├── train.py          # Main training loop with K-token objectives
├── eval.py           # Deterministic evaluation with multiple baselines
├── losses.py         # K-token CE and KD losses (critical for learning)
├── models.py         # Encoder, Adapter, LMWrapper classes
├── prefix_utils.py   # Calibration, BOS policy, anchor handling
├── data.py          # Dataset loading (SQuAD, HotpotQA)
├── metrics.py       # EM/F1 scoring, NLL computation
└── common.py        # Chat templates, text truncation utilities
```

### Key Configuration Parameters

Critical hyperparameters that significantly impact performance:

```python
# Latent dimensions
LATENT_LEN = 32       # M: number of soft tokens (compression vs capacity tradeoff)
D_Z = 256            # Latent dimension per token

# Training objectives
K = 4                # Number of tokens to supervise (k_token_ce)
FIRST_TOKEN_CE = 0.5 # Weight for first-token cross-entropy
KD_TAU = 1.0        # Temperature for knowledge distillation

# Calibration & anchoring
CALIBRATION = "embed_rms"      # Scale latents to match embedding statistics
WARM_ANCHOR_TEXT = "Answer: "  # Anchor text between prefix and answer
APPEND_BOS_AFTER_PREFIX = "yes" # BOS policy for first-token generation

# Decode hardening
FIRST_TOKEN_TOP_P = 0.95       # Nucleus sampling for first token
FIRST_TOKEN_TEMPERATURE = 0.7  # Temperature for first token
EOS_BAN_STEPS = 4              # Prevent early EOS tokens
```

### Evaluation Metrics & Baselines

The system evaluates against multiple baselines:
1. **Text baseline**: Full prompt via text (upper bound)
2. **Latent**: Compressed soft tokens
3. **Token-budget**: Text truncated to M tokens (fairness baseline)
4. **Joint rescoring**: Two-model ensemble picking best answer

Key metrics tracked:
- **Task quality**: EM/F1 scores
- **Conditioning**: NLL/token on gold answers
- **Efficiency**: Compression ratio, wire bytes, wall-clock time
- **First-token accuracy**: Critical for generation quality

### Common Pitfalls & Solutions

1. **PAD token contamination**: Always mask PAD tokens (-100) in labels and zero their attention
2. **BOS misalignment**: Ensure consistent BOS policy between train/eval
3. **Amplitude drift**: Use per-example calibration, not batch-level
4. **Tokenization mismatch**: Verify t=0 alignment with `_assert_t0_alignment()`
5. **Left padding issues**: Handle properly in teacher-forced sequences

### Compression & Wire Protocol

The system measures honest compression via:
- **Text bytes**: UTF-8 encoded prompt bytes
- **Latent bytes**: Quantized latent representation (fp16/int8/int6/int4)
- Group-wise quantization with scale overhead accounting
- Target: ≥4× compression while maintaining quality

### Current State & Next Steps

Based on recent experiments (8B_clean_answer_ftce run):
- Text baseline achieves F1 ~0.80-0.85
- Latent currently at F1 ~0.01-0.02, FirstTok@1 ~5-7%
- Recent fixes addressed PAD masking and BOS alignment
- Next focus: Improving first-token accuracy via K-token objectives

The codebase is actively implementing Phase A improvements from PLAN.md to achieve:
- FirstTok@1: 12-20% at M∈{32,48,64}
- F1: 0.10-0.20 with honest compression