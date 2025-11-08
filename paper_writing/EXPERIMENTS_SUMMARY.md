# Complete Ablation Studies Summary

**Total Experiments**: 4 new runs + 1 reused baseline = 5 total configurations
**Total GPU Time**: ~12 hours on 4× H100
**Timeline**: Week 1 (Nov 8-14)

---

## Ablations Overview

### ✅ Already Have (Reuse from `successful_experiments/cross_model/85/`)

**1b_baseline_64tok** - No stability fixes
- **Config**: 64 tokens, depth=8, lr=1e-4, warmup=750
- **Stability**: NO InfoNCE, NO early stopping, NO repetition penalty
- **Dataset**: GSM8K
- **Result**: Peak 81.5% → Final 36.0% (45.5% degradation)
- **Purpose**: Baseline showing collapse without fixes
- **Status**: ✅ DONE (use existing log)

---

## New Experiments to Run

### ABLATION 1: Stability Fixes

#### 1a_stable_64tok ⏳ TODO
- **Config**: 64 tokens, depth=8, lr=1e-4, warmup=750
- **Stability**: InfoNCE (λ=0.05), early stopping (patience=5), repetition_penalty=1.1
- **Dataset**: GSM8K
- **Expected**: Maintain >70% final (vs 36% collapse)
- **Runtime**: ~3 hours
- **Purpose**: Prove stability fixes prevent collapse

**Comparison**: 1a vs 1b shows impact of stability fixes

---

### ABLATION 2: Sequence Length (Compression-Quality Tradeoff)

All configs WITH stability fixes (InfoNCE + early stopping + gen hygiene)

#### 2a_stable_32tok ⏳ TODO
- **Config**: 32 tokens, depth=4, bottleneck=768, heads=12, lr=1e-4, warmup=600
- **Compression**: ~4.7× (150 → 32 tokens)
- **KV Cache Saved**: 118 tokens × 0.5 MB = ~59 MB per request
- **Expected**: ~55% peak (based on old run), better stability
- **Runtime**: ~3 hours
- **Purpose**: High compression, moderate quality

#### 2b_stable_48tok ⏳ TODO
- **Config**: 48 tokens, depth=6, bottleneck=1024, heads=16, lr=1e-4, warmup=750
- **Compression**: ~3.1× (150 → 48 tokens)
- **KV Cache Saved**: 102 tokens × 0.5 MB = ~51 MB per request
- **Expected**: ~60-70% peak, improved over old 48-token runs
- **Runtime**: ~3 hours
- **Purpose**: Medium compression, good quality

#### 2c_stable_64tok (Same as 1a) ✅ REUSE
- **Config**: 64 tokens, depth=8, bottleneck=1024, heads=16, lr=1e-4, warmup=750
- **Compression**: ~2.3× (150 → 64 tokens)
- **KV Cache Saved**: 86 tokens × 0.5 MB = ~43 MB per request
- **Expected**: >75% final (with stability)
- **Runtime**: 0 hours (reuse 1a result)
- **Purpose**: Lower compression, best quality

**Comparison**: 2a vs 2b vs 2c shows compression-quality tradeoff

---

### ABLATION 3: Dataset Generalization

#### 3a_hotpotqa_64tok ⏳ TODO
- **Config**: 64 tokens, depth=8, bottleneck=1024, heads=16, lr=1e-4, warmup=750
- **Stability**: InfoNCE + early stopping + gen hygiene (same as 1a)
- **Dataset**: HotpotQA (multi-hop question answering)
- **Expected**: Beat text baseline on at least 1 checkpoint
- **Runtime**: ~3 hours
- **Purpose**: Show generalization beyond math reasoning

**Comparison**: 1a (GSM8K) vs 3a (HotpotQA) shows cross-dataset generalization

---

### ABLATION 4: Inference Metrics (KV Cache & Latency)

Post-hoc analysis, NO TRAINING REQUIRED

**Input**: Checkpoint from 1a_stable_64tok
**Analysis**: Benchmark all baselines on test set, measure practical benefits

#### Baselines (5 Total)
1. **Source-alone (Mistral)**: Question → Mistral → Answer (P0 - CRITICAL)
   - Purpose: Prove improvement isn't just from using Mistral
2. **Target-alone (Llama)**: Full prompt → Llama
   - Purpose: Standard single-model baseline
3. **Latent (Our method)**: Question → Mistral → Translator → Llama
   - Purpose: Cross-model translation via soft tokens
4. **Token-budget**: Truncated prompt (K tokens) → Llama
   - Purpose: Fair compression baseline
5. **Cascade**: Mistral text answer → Llama refinement (P2)
   - Purpose: Compare soft tokens vs discrete text transfer

**Per-Sample Metrics** (stored in JSONL):
- KV cache memory (MB) during generation
- End-to-end latency (seconds)
- Peak GPU memory (MB)
- Quality (EM/F1)
- Sequence lengths (input/output)

**Key Comparisons**:
- Does latent beat BOTH single models? (Information enrichment)
- Does latent beat cascade? (Soft tokens > text tokens)
- KV cache savings: target_alone vs latent
- Latency reduction for longer outputs

**Method**: `benchmark_inference.py` script
**Samples**: Full GSM8K test set (1,319 samples)
**Runtime**: 2-3 hours (5 baselines × ~1.3K samples)

**Purpose**: Determine paper narrative + practical benefits

---

## Summary Tables

### Execution Plan

| Experiment | Tokens | Dataset | Stability | Runtime | Status |
|------------|--------|---------|-----------|---------|--------|
| 1a_stable_64tok | 64 | GSM8K | YES | 3h | ⏳ TODO |
| 1b_baseline_64tok | 64 | GSM8K | NO | 0h | ✅ REUSE |
| 2a_stable_32tok | 32 | GSM8K | YES | 3h | ⏳ TODO |
| 2b_stable_48tok | 48 | GSM8K | YES | 3h | ⏳ TODO |
| 2c_stable_64tok | 64 | GSM8K | YES | 0h | ✅ REUSE 1a |
| 3a_hotpotqa_64tok | 64 | HotpotQA | YES | 3h | ⏳ TODO |
| 4_inference_metrics | - | - | - | 2-3h | ⏳ TODO |
| **TOTAL** | | | | **~14h** | |

### Paper Claims Matrix

| Claim | Evidence | Ablation | Status |
|-------|----------|----------|--------|
| **Cross-model fusion beats both** | Latent > source AND target | 4 (P0 - CRITICAL) | ⏳ TODO |
| Stability fixes work | >70% final vs 36% | 1a vs 1b | ⏳ TODO |
| Compression-quality tradeoff | 32→48→64 tokens | 2a, 2b, 2c | ⏳ TODO |
| Generalizes beyond math | HotpotQA beats baseline | 3a | ⏳ TODO |
| Soft tokens > text transfer | Latent > cascade | 4 (P2) | ⏳ TODO |
| KV cache savings | 43-59 MB saved | 4 | ⏳ TODO |

---

## Expected Results

Based on prior experiments and stability improvements:

| Config | Peak Acc | Final Acc | Degradation | Compression |
|--------|----------|-----------|-------------|-------------|
| 1a (64 tok, stable) | ~75-80% | ~70-75% | <10% | 2.3× |
| 1b (64 tok, unstable) | 81.5% | 36.0% | 45.5% | 2.3× |
| 2a (32 tok, stable) | ~55-60% | ~50-55% | <10% | 4.7× |
| 2b (48 tok, stable) | ~65-70% | ~60-65% | <10% | 3.1× |
| 3a (HotpotQA, 64 tok) | TBD | TBD | TBD | 2.3× |

**Key Hypothesis**: Stability fixes will:
1. ✅ Prevent catastrophic collapse (final >> 36%)
2. ✅ Maintain performance near peak (<10% degradation)
3. ⚠️ Possibly trade some peak performance for stability (peak ~75% vs 81.5%)

---

## Analysis Scripts

### During Training
- `run_ablations.sh` - Main experiment runner
- Logs saved to `runs/ablations_*/EXPERIMENT_NAME/train.log`
- Summary table in `runs/ablations_*/summary.log`

### Post-Training
- `analyze_ablations.py` - Extract metrics from logs (auto-generated)
- `analyze_compression.py` - Quantization analysis
- Custom plotting scripts (to be created in Week 2)

---

## Next Steps

1. **Review this plan** - Confirm scope is appropriate
2. **Test scripts locally** - Verify no syntax errors
3. **Run on HPC** - Execute `run_ablations.sh`
4. **Monitor progress** - Check logs every 2-3 hours
5. **Analyze results** - Week 2, after experiments complete
6. **Write paper** - Weeks 2-3

**Est. start date**: Nov 8, 2024
**Est. completion**: Nov 11-12, 2024 (assuming no failures)
**Buffer**: Nov 12-14 for re-runs if needed
