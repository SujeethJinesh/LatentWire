# Ablation Studies for Paper

**Goal**: Minimal set of experiments to support paper claims in 3 weeks
**Total GPU Time Budget**: ~20 hours on 4× H100

---

## Ablation 1: Stability Fixes (CRITICAL - P0)

**Research Question**: Do InfoNCE + early stopping + generation hygiene prevent collapse?

**Configurations**:
1. **Baseline (no fixes)**: Same as successful_experiments/cross_model/85/3_high_capacity
   - 64 tokens, depth=8, lr=1e-4, warmup=750
   - NO InfoNCE, NO early stopping, NO repetition penalty
   - Expected: Peak ~81.5%, collapse to ~36%

2. **With stability fixes**:
   - Same architecture (64 tokens, depth=8, lr=1e-4, warmup=750)
   - InfoNCE loss (λ=0.05, start after 50% warmup)
   - Early stopping (patience=5 on bridged accuracy)
   - Repetition penalty (1.1) + no_repeat_ngram_size=3
   - Expected: Maintain >70% final accuracy

**Metrics**:
- Peak bridged accuracy
- Final bridged accuracy
- Degradation (peak - final)
- Training curves (plot accuracy over time)

**Runtime**: 2 × 3 hours = 6 hours
**Dataset**: GSM8K (1,319 test samples)

---

## Ablation 2: Sequence Length (Architecture - P0)

**Research Question**: How does soft token count affect compression vs quality?

**Configurations** (all with stability fixes):
1. **32 tokens** (High compression)
   - Bottleneck=768, depth=4, heads=12, lr=1e-4, warmup=600
   - Compression: ~4.7× (150 → 32 tokens)
   - Expected: Moderate quality, best stability

2. **48 tokens** (Medium compression)
   - Bottleneck=1024, depth=6, heads=16, lr=1e-4, warmup=750
   - Compression: ~3.1× (150 → 48 tokens)
   - Expected: Better quality than 32, less stable

3. **64 tokens** (Lower compression)
   - Bottleneck=1024, depth=8, heads=16, lr=1e-4, warmup=750
   - Compression: ~2.3× (150 → 64 tokens)
   - Expected: Best quality, moderate stability

**Metrics**:
- Bridged accuracy vs soft token count
- Compression ratio vs quality tradeoff
- KV cache savings (tokens saved × 0.5 MB)
- Training stability (peak - final degradation)

**Runtime**: 3 × 3 hours = 9 hours
**Dataset**: GSM8K

---

## Ablation 3: Dataset Generalization (P1)

**Research Question**: Does method generalize beyond math reasoning?

**Configurations**:
1. **GSM8K** (Math reasoning) - Already trained
   - Use results from Ablation 1 & 2

2. **HotpotQA** (Multi-hop QA)
   - Same best config from Ablation 1 (64 tokens, stable)
   - Different reasoning type (knowledge retrieval + multi-hop)
   - Expected: Beat baseline on at least 1 checkpoint

**Metrics**:
- Bridged vs text baseline on HotpotQA
- Cross-dataset comparison (GSM8K vs HotpotQA)

**Runtime**: 1 × 3 hours = 3 hours
**Dataset**: HotpotQA (7,405 train, ~1,000 test)

---

## Ablation 4: Inference Metrics (KV Cache & Latency - P0)

**Research Question**: What are the practical memory and time savings?

**Goal**: Measure KV cache memory usage and end-to-end latency for longer outputs
- Why KV cache matters: Memory grows with sequence length during generation
- Longer outputs → more KV cache → more memory savings with compression

**Method**: Benchmark all baselines on evaluation samples
- Take trained checkpoint from Ablation 1 (64 tokens, stable)
- Evaluate on ALL test samples (full dataset)
- **Store per-sample raw data** for later analysis

**Baselines to Measure**:
1. **Source-alone (Mistral)**: Question → Mistral → Answer (P0 - CRITICAL)
   - Proves improvement isn't just from using Mistral
2. **Target-alone (Llama)**: Full prompt → Llama (standard baseline)
3. **Latent**: Question → Mistral → Translator (64 soft tokens) → Llama (our method)
4. **Token-budget**: Truncated prompt (M tokens) → Llama (fairness baseline)
5. **Cascade**: Mistral generates text → Llama refines (P2 - text vs soft token transfer)

**Per-Sample Metrics** (store raw data):
- KV cache memory usage (MB) during generation
- End-to-end latency (seconds) from prompt to completion
- Peak GPU memory (MB)
- Generated text quality (EM/F1)
- Input sequence length (tokens)
- Output sequence length (tokens)

**Aggregate Metrics**:
- Average KV cache: text vs latent
- KV cache reduction (MB and %)
- Average latency: text vs latent
- Latency reduction (seconds and %)
- Quality comparison (EM/F1)

**Output Format**:
- `per_sample_metrics.jsonl` - One JSON object per sample with all metrics
- `aggregate_metrics.json` - Summary statistics
- Allows flexible post-hoc analysis and plotting

**Runtime**: ~2-3 hours (depends on dataset size, 5 baselines per sample)
**Dataset**: GSM8K test set (1,319 samples) or subset

---

---

## Summary Table

| Ablation | Priority | GPU Hours | Configs | Purpose |
|----------|----------|-----------|---------|---------|
| 1. Stability Fixes | P0 | 3h | 1 new + 1 reuse | Prove fixes prevent collapse |
| 2. Sequence Length | P0 | 6h | 2 new + 1 reuse | Compression-quality tradeoff |
| 3. Dataset (HotpotQA) | P1 | 3h | 1 | Generalization beyond math |
| 4. Inference Metrics | P0 | 2h | Analysis | KV cache + latency measurement |
| **TOTAL TRAINING** | | **~12h** | **4 new runs** | |
| **TOTAL WITH ANALYSIS** | | **~14h** | | |

---

## What We Already Have (Reuse!)

From `successful_experiments/cross_model/85/`:
- ✅ Baseline without stability fixes (81.5% peak → 36% collapse)
- ✅ Sequence length: 32, 48, 64 tokens (without stability fixes)
- ✅ Architecture depth: 4, 6, 8 layers
- ✅ Learning rate: 5e-5, 1e-4, 2e-4

**Reuse Strategy**:
- Use existing results for "no stability fixes" baseline
- Focus new runs on "with stability fixes" configs
- **Saves ~6 hours** (don't need to re-run baseline)

---

## Revised GPU Time Budget

| Experiment | Hours | Notes |
|------------|-------|-------|
| Stability: With fixes (64 tokens) | 3h | New run |
| Stability: No fixes (64 tokens) | 0h | **Reuse 81.5% result** |
| Sequence: 32 tokens + stability | 3h | New run |
| Sequence: 48 tokens + stability | 3h | New run |
| Sequence: 64 tokens + stability | 0h | **Same as stability ablation** |
| Dataset: HotpotQA | 3h | New run |
| Inference metrics (KV cache + latency) | 2h | Per-sample measurement |
| **TOTAL REQUIRED** | **~14h** | Core experiments |

---

## Implementation Plan

### Script Structure

**Main sweep script**: `run_ablations.sh`
- Runs all P0 and P1 experiments sequentially
- Each config saves to `paper_writing/runs/ablation_name/`
- Logs captured with `tee` to timestamped files
- Summary script at end to compare all results

**Ablation configs**:
```bash
# Ablation 1: Stability
- 1a_stable_64tok (WITH fixes)
- 1b_baseline_64tok (NO fixes) → SKIP, use existing

# Ablation 2: Sequence Length (all with stability)
- 2a_stable_32tok
- 2b_stable_48tok
- 2c_stable_64tok → SAME as 1a_stable_64tok

# Ablation 3: Dataset
- 3a_hotpotqa_64tok (with stability)

# Ablation 4: Inference Metrics
- 4_inference_metrics.py (KV cache + latency benchmarking)
```

**Total unique runs needed**: 4 training runs (12h) + 1 inference analysis (2h) = 14 hours

---

## Success Metrics (Paper Claims)

### Claim 1: Cross-Model Information Enrichment
**Evidence**: Latent beats BOTH source-alone (Mistral) AND target-alone (Llama)
**Ablation**: Inference Metrics (P0 - will reveal which scenario we're in)
**Critical**: This determines if we have genuine cross-model fusion vs just knowledge transfer

### Claim 2: Stability Fixes Work
**Evidence**: Final accuracy >70% vs 36% collapse
**Ablation**: Stability (with vs without fixes)

### Claim 3: Compression-Quality Tradeoff
**Evidence**: 32 tok (4.7× comp, ~55% acc) → 64 tok (2.3× comp, ~81% acc)
**Ablation**: Sequence Length

### Claim 4: Generalizes Beyond Math
**Evidence**: Beats baseline on HotpotQA (different reasoning type)
**Ablation**: Dataset

### Claim 5: KV Cache Savings and Latency Reduction
**Evidence**: Reduced KV cache memory (43-59 MB saved) and faster inference for longer outputs
**Ablation**: Inference Metrics

---

## Next Steps

1. ✅ Copy scripts to paper_writing/
2. ⏳ Implement ablation sweep script
3. ⏳ Add quantization analysis script
4. ⏳ Create results aggregation script
5. ⏳ Run ablations (Week 1)
6. ⏳ Analyze results (Week 2)
7. ⏳ Write paper (Weeks 2-3)
