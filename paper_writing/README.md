# Paper Writing Directory

**Working directory for 3-week paper development**
**Timeline**: Nov 8-28, 2024

---

## Contents

### Planning Documents
- **PLAN.md** - Overall 3-week timeline (experiments + writing)
- **README.md** - This file (quick reference)

### Scripts
- **cross_attention.py** - Main training script with BottleneckedGatedTranslator
- **run_ablations.sh** - Runs all ablation experiments (~9 hours on 4× H100)
- **run_phase2_swap.sh** - Bidirectional swap launcher (Llama source → Mistral target)
- **run_phase2_single_gpu_suite.sh** - Sequential single-GPU sweep with automatic preservation
- **benchmark_inference.py** - Standalone inference benchmarking utility (not actively used)
- **scripts/summarize_eval_jsonl.py** - Utility to report invalid vs valid answers in eval JSONLs

### Output (Generated During Experiments)
- **runs/** - All experiment outputs, logs, and checkpoints
  - `ablations_YYYYMMDD_HHMMSS/` - Timestamped ablation runs
    - `1a_stable_64tok/` - 64 tokens WITH stability fixes
    - `2a_stable_32tok/` - 32 tokens WITH stability fixes
    - `2b_stable_48tok/` - 48 tokens WITH stability fixes
    - `summary.log` - Consolidated results table
    - `analyze_ablations.py` - Generated analysis script

---

## Quick Start

### Step 1: Run Ablation Experiments (Week 1)

On HPC cluster with 4× H100 GPUs:

```bash
cd /path/to/LatentWire
git pull
rm -rf paper_writing/runs  # Clean previous runs
PYTHONPATH=. bash paper_writing/run_ablations.sh
```

**Expected runtime**: ~9 hours
**Output**: `paper_writing/runs/ablations_YYYYMMDD_HHMMSS/`

**GPU Scaling**: All launcher scripts auto-detect available GPUs (NUM_GPUS env var → `nvidia-smi` → PyTorch). Set `NUM_GPUS=N` before the command if you need to pin the world size (e.g., `NUM_GPUS=1 PYTHONPATH=. bash paper_writing/run_phase2_swap.sh`).

### Step 2: Analyze Results (Week 2)

```bash
cd paper_writing/runs/ablations_*/
python analyze_ablations.py
```

Generates:
- `ablation_results.json` - Raw data for all experiments
- Summary tables and statistics

For per-eval diagnostics on preserved runs:

```bash
python paper_writing/scripts/summarize_eval_jsonl.py paper_writing/preserved_data/phase1_full_20251116_201212/phase1_all_fix/
```


### Step 3: Generate Plots (Week 2)

```python
# TODO: Add plotting script for:
# - Training curves (accuracy over time)
# - Stability comparison (with vs without fixes)
# - Sequence length tradeoff (32/48/64 tokens)
# - Compression-quality curve
```

---

## Current Status (Nov 17, 2025)
- **Phase 1 all-fix baseline completed** (command: `PYTHONPATH=. bash paper_writing/run_ablations.sh`). Bridged accuracy: 0.680 peak → 0.645 final (source 0.540, target 0.770). Artifacts preserved under `paper_writing/preserved_data/phase1_full_20251116_201212/`.
- **Ablation B (KL only)** finalized via `bash paper_writing/run_ablation_B.sh`: 0.710 peak → 0.625 final. Demonstrates KL alone underperforms; logs in `paper_writing/preserved_data/ablB_20251116_234242/`.
- **Ablation C (KL + prompt alignment, no RoPE)** finalized via `bash paper_writing/run_ablation_C.sh`: plateau around 0.615 during training, recovered to 0.655 on the final checkpoint. Artifacts under `paper_writing/preserved_data/ablC_20251117_013909/`.
- **Next HPC run**: Phase 2 “bidirectional swap” to flip source and target models (Mistral ↔ Llama) while keeping the Phase 1 hyperparameters. Launch via the same torchrun invocation as Phase 1 but set `SOURCE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct` and `TARGET_MODEL=mistralai/Mistral-7B-Instruct-v0.3`. New logs should be timestamped `paper_writing/runs/phase2_swap_*` before being copied into `paper_writing/preserved_data/`.
- **Phase 2 swap attempt (Nov 18, soft+text)** preserved under `paper_writing/preserved_data/phase2_swap_20251118_192955/`: source-alone 0.765, target-alone 0.515, bridged 0.26 because prompt-teacher soft tokens were concatenated with the literal prompt. `run_phase2_swap.sh` now auto-selects `soft_only` whenever `dit_teacher=prompt` to prevent duplicated context.
- **Phase 2 swap attempt (Nov 18, soft-only)** preserved under `paper_writing/preserved_data/phase2_swap_20251118_213543/`: even with `soft_only`, prompt-teacher DiT collapsed (bridged ≤0.005) and emitted constant tokens, so future runs need stronger supervision or hybrid conditioning.

Consult `paper_writing/preserved_data/README.md` for detailed rationale, results, and publishability notes for each preserved run.

---

## Ablation Experiments

### Ablation 1: Stability Fixes (P0)
**Question**: Do InfoNCE + early stopping prevent collapse?

| Config | Description | Runtime | Status |
|--------|-------------|---------|--------|
| 1a_stable_64tok | WITH fixes | 3h | ✅ DONE (phase1_all_fix) |
| 1b_baseline_64tok | NO fixes | 0h | ✅ Reuse existing (81.5% → 36%) |

**Expected outcome**: 1a maintains >70% final vs 36% collapse in 1b

### Ablation 2: Sequence Length (P0)
**Question**: Compression vs quality tradeoff?

| Tokens | Compression | Runtime | Status |
|--------|-------------|---------|--------|
| 32 | 4.7× | 3h | ⏳ TODO |
| 48 | 3.1× | 3h | ⏳ TODO |
| 64 | 2.3× | 0h | ✅ Reuse 1a |

**Expected outcome**: Higher tokens = better quality but less compression

### Ablation 3: Inference Metrics (P0)
**Question**: Practical memory and latency savings?

**Baselines**:
- Source-alone (Mistral)
- Target-alone (Llama)
- Latent (our method)
- Token-budget (fair compression baseline)

**Metrics**: KV cache savings, latency, accuracy

**Expected outcome**: Demonstrate practical benefits with 2-5× compression

---

## Experiment Checklist

**Week 1 (Nov 8-14): Run Experiments**
- [ ] Run `run_ablations.sh` on HPC
- [ ] Monitor progress (check logs every few hours)
- [ ] Git pull results locally
- [ ] Verify all experiments completed successfully

**Week 2 (Nov 15-21): Analyze + Start Writing**
- [ ] Run `analyze_ablations.py` to extract results
- [ ] Create training curve plots
- [ ] Generate comparison tables
- [ ] Create LaTeX paper template
- [ ] Write Method section (can start early)
- [ ] Write Related Work section

**Week 3 (Nov 22-28): Complete Draft**
- [ ] Write Introduction
- [ ] Write Experiments section (fill in results)
- [ ] Write Analysis section
- [ ] Write Conclusion
- [ ] Create all figures and tables
- [ ] Multiple revision passes
- [ ] Final submission prep

---

## Key Results to Report

### Main Claim: Information Enrichment
- **Evidence**: 81.5% bridged > 73% text baseline on GSM8K
- **Source**: `successful_experiments/cross_model/85/train_high_capacity.log`

### Claim: Stability Fixes Work
- **Evidence**: Final accuracy >70% vs 36% collapse
- **Source**: Compare `1a_stable_64tok/` vs `1b_baseline_64tok`

### Claim: Compression-Quality Tradeoff
- **Evidence**: 32tok (4.7×, ~55%) → 48tok (3.1×, ~TBD%) → 64tok (2.3×, ~75%+)
- **Source**: `2a_stable_32tok/`, `2b_stable_48tok/`, `1a_stable_64tok/`

### Claim: Practical KV Cache Savings
- **Evidence**: 43-59 MB saved per inference with soft tokens
- **Source**: Inference metrics from Ablation 3

---

## Files Referenced

### From Main Codebase
- `successful_experiments/cross_model/85/` - Original 81.5% result
- `cross_model/experiments/cross_attention.py` - Training implementation

### Paper Assets (To Be Created)
- `paper.tex` - Main paper LaTeX
- `figures/` - All plots and diagrams
  - `training_curves.pdf` - Accuracy over time
  - `stability_comparison.pdf` - With vs without fixes
  - `compression_tradeoff.pdf` - Quality vs compression
- `tables/` - LaTeX table definitions
  - `main_results.tex`
  - `ablations.tex`
  - `compression.tex`

---

## Notes

### Architecture
- BottleneckedGatedTranslator with Flamingo-style gated cross-attention
- Orthogonal query initialization, RMSNorm, SwiGLU activation
- Gate parameters (sa_gate, cross_gate, ffn_gate) at 3× learning rate

### Training
- All experiments use `seed=1234` for reproducibility
- InfoNCE weight = 0.05 (starts after 50% warmup, computed in float32)
- Early stopping patience = 5 evaluations
- Fused AdamW optimizer for efficiency
- Pure greedy decoding (no repetition penalty)

### Evaluation
- 500 samples per evaluation (batched to avoid OOM)
- 8-shot Chain-of-Thought prompting with fixed seed=42 exemplars
- Stable tokenization spacing (clean_up_tokenization_spaces=False)

---

## Contact

Questions? Check:
1. **PLAN.md** - Overall timeline and detailed experiment specifications
2. **../CLAUDE.md** - Project-wide instructions
3. **../LOG.md** - Experiment history and findings
