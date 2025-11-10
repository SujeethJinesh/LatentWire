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
- **benchmark_inference.py** - Standalone inference benchmarking utility (not actively used)

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

### Step 2: Analyze Results (Week 2)

```bash
cd paper_writing/runs/ablations_*/
python analyze_ablations.py
```

Generates:
- `ablation_results.json` - Raw data for all experiments
- Summary tables and statistics

### Step 3: Generate Plots (Week 2)

```python
# TODO: Add plotting script for:
# - Training curves (accuracy over time)
# - Stability comparison (with vs without fixes)
# - Sequence length tradeoff (32/48/64 tokens)
# - Compression-quality curve
```

---

## Ablation Experiments

### Ablation 1: Stability Fixes (P0)
**Question**: Do InfoNCE + early stopping prevent collapse?

| Config | Description | Runtime | Status |
|--------|-------------|---------|--------|
| 1a_stable_64tok | WITH fixes | 3h | ⏳ TODO |
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
