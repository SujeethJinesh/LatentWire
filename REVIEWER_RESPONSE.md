# Reviewer Response Document

## Overview

This document outlines how we address each piece of reviewer feedback for the LatentWire paper submission.

---

## CRITICAL ISSUES (Already Fixed)

### 1. Parameter Count Inconsistency

**Reviewer Concern:** "Figure 1 claims 188K Bridge, ablation table shows 6.3M/16.8M (hidden row discrepancy)"

**Status:** FIXED (Commit 72a7904)

**Root Cause:** The paper claimed ~500K parameters but the actual PerceiverResampler implementation has **537M parameters**. This was a 1074x discrepancy.

**Resolution:**
- Updated all parameter claims from ~500K to 537M across 6 locations
- Clarified that classification experiments use full model dimension (d_z=4096, M=8)
- Added note distinguishing SQuAD ablation config (d_z=256, M=16) from classification config
- Updated efficiency comparison: now correctly states 13x reduction vs 7B fine-tuning (not 100x)
- Noted bridge is 3.6% of combined sender+receiver capacity (15B)

**Files Changed:** `paper.tex`

---

## ADDRESSED CONCERNS

### 2. ICAE/Prompt Compression Comparison

**Reviewer Concern:** Missing comparison to ICAE and 500xCompressor methods.

**Status:** ADDRESSED in Related Work

**Resolution:** Added clarification that ICAE/500xCompressor solve a fundamentally different problem:
- ICAE: Same-model compression (encoder = decoder LLM)
- LatentWire: Cross-model transfer (different encoder and decoder LLMs)

These are not directly comparable because ICAE cannot transfer information between heterogeneous models.

**Files Changed:** `paper.tex` (Related Work section)

### 3. Latency Sensitivity

**Reviewer Concern:** Latency measurements may be sensitive to hardware/conditions.

**Status:** ADDRESSED

**Resolution:** Added methodology section clarifying:
- All timings on H100 GPU with models pre-loaded in VRAM
- Each latency is mean of 3 runs over 200 samples
- Standard deviation <5% for Bridge, <10% for Text-Relay
- Speedup advantage is robust across all measured variance

**Files Changed:** `paper.tex` (after Table 3)

### 4. Baseline Fairness

**Reviewer Concern:** Need fairness baselines.

**Status:** ALREADY ADDRESSED

**Resolution:** Paper already includes:
- Token-budget baseline (text truncated to M tokens)
- Zero-shot baselines for both sender and receiver
- Prompt-tuning baseline (receiver-only, no sender)
- Text-relay baseline (for latency comparison)

---

## EXPERIMENTS TO RUN

### 5. Task Transfer Experiments

**Reviewer Concern:** "Task-specific bridges with no transfer" - bridges are trained per-task.

**Proposed Experiments:**
- Train bridge on SST-2 sentiment
- Evaluate zero-shot transfer on IMDB and Yelp Polarity (same sentiment task, different domains)
- Compare transfer accuracy vs training fresh bridges on target domains

**Expected Outcome:** Demonstrate that bridges generalize across domains within the same task type.

**Script:** `telepathy/run_enhanced_arxiv_suite.sh` (Phase 1)

**Datasets Added:** IMDB and Yelp Polarity configs in `run_unified_comparison.py`

### 6. Inverse Scaling Ablation

**Reviewer Concern:** "Inverse scaling under-theorized" - fewer tokens sometimes work better.

**Proposed Experiments:**
- Test M = [2, 4, 8, 16, 32] soft tokens
- Run on SST-2, AG News, and TREC
- Analyze: Does compression act as regularization?

**Expected Outcome:** Show that for simple tasks (binary classification), fewer tokens can match or exceed more tokens, supporting "compression as regularization" hypothesis.

**Script:** `telepathy/run_enhanced_arxiv_suite.sh` (Phase 2)

### 7. Multi-Seed Statistical Robustness

**Reviewer Concern:** Need statistical significance across multiple runs.

**Proposed Experiments:**
- Run all experiments with seeds [42, 123, 456]
- Report mean ± std for all metrics
- Compute significance tests between methods

**Script:** `telepathy/run_full_arxiv_suite.sh` (Phase 3)

---

## ACKNOWLEDGED LIMITATIONS

### 8. Single Model Pair

**Reviewer Concern:** Only tested Llama/Mistral pair.

**Response:** Acknowledged in Limitations section. We investigated Gemma but found 256K vocabulary mismatch makes comparison unfair. The Llama→Mistral pair demonstrates the core contribution (cross-family transfer with incompatible tokenizers). Additional model pairs (e.g., Phi-3, other Llama variants) would strengthen the paper but are not strictly necessary for the core claim.

**Action:** No code changes. Paper already acknowledges this limitation.

### 9. Reasoning Task Failure

**Reviewer Concern:** Bridge fails on reasoning tasks (GSM8K).

**Response:** This is a fundamental limitation, not a bug. Reasoning requires:
- Multi-step symbolic manipulation
- Exact numeric preservation
- Logical chain coherence

Soft tokens are inherently "blurry" and cannot preserve exact values needed for arithmetic. This is acknowledged in the paper and represents an important negative result.

**Action:** No code changes. Paper already discusses this limitation.

---

## EXPERIMENT EXECUTION PLAN

### Scripts Ready to Run

1. **Full ArXiv Suite** (recommended):
   ```bash
   cd /path/to/LatentWire
   git pull
   rm -rf runs
   PYTHONPATH=. bash telepathy/run_full_arxiv_suite.sh
   ```
   - Runs all baselines on SST-2, AG News, TREC
   - Multi-seed experiments (42, 123, 456)
   - Generates statistical summary
   - Estimated time: 3-4 hours on 4x H100

2. **Enhanced ArXiv Suite** (additional experiments):
   ```bash
   PYTHONPATH=. bash telepathy/run_enhanced_arxiv_suite.sh
   ```
   - Task transfer (SST-2 → IMDB/Yelp)
   - Inverse scaling ablation (M=2,4,8,16,32)
   - Estimated time: 5-7 hours on 4x H100

### Output Structure

```
runs/full_arxiv_suite_YYYYMMDD_HHMMSS/
├── phase1_baselines/           # Llama text baselines
├── phase2_unified/seed_42/     # Main experiments
├── phase3_multiseed/
│   ├── seed_123/
│   └── seed_456/
└── statistical_summary.json    # Aggregated mean ± std

runs/enhanced_arxiv_YYYYMMDD_HHMMSS/
├── phase1_transfer/            # SST-2 → IMDB/Yelp transfer
├── phase2_inverse_scaling/     # M ablation
│   ├── sst2_M2/, sst2_M4/, ... sst2_M32/
│   └── ...
└── phase3_multiseed/
```

---

## SUMMARY TABLE

| Concern | Status | Action Taken |
|---------|--------|--------------|
| Parameter count (CRITICAL) | FIXED | Updated paper: 537M not 500K |
| ICAE comparison | ADDRESSED | Clarified in Related Work |
| Latency sensitivity | ADDRESSED | Added methodology section |
| Baseline fairness | ALREADY DONE | Token-budget baseline exists |
| Task transfer | READY TO RUN | Scripts prepared |
| Inverse scaling | READY TO RUN | Scripts prepared |
| Multi-seed stats | READY TO RUN | Scripts prepared |
| Single model pair | ACKNOWLEDGED | In Limitations section |
| Reasoning failure | ACKNOWLEDGED | In Limitations section |

---

## Next Steps

1. Run `run_full_arxiv_suite.sh` on HPC (3-4 hours)
2. Run `run_enhanced_arxiv_suite.sh` on HPC (5-7 hours)
3. Pull results and update paper with:
   - Task transfer results table
   - Inverse scaling ablation figure
   - Multi-seed mean ± std in all tables
4. Update abstract/intro with any new key findings
