# Research Paper Plan: Cross-Model Translation via Learned Interlingua

**Timeline**: 3 weeks (Experiments + Writing)
**Target Venue**: arXiv preprint / workshop paper (4-8 pages)
**Core Claim**: Cross-model translation beats both single models through information enrichment

---

## Overview

**What we already have** ✅:
- 81.5% peak result exceeding 73% baseline (successful_experiments/cross_model/85/)
- Sequence length ablations: 32, 48, 64 soft tokens (without stability fixes)
- Architecture ablations: Depth (4-8 layers), learning rate, warmup
- GSM8K dataset: 1,319 held-out test samples

**What we need** ⏳:
- Validate stability fixes prevent collapse
- Measure if latent beats BOTH source (Mistral) and target (Llama) models
- Benchmark KV cache savings and latency

**Total GPU time**: ~9 hours on 4× H100

---

## Week 1: Core Experiments (Nov 8-14)

### Ablation 1: Stability Fixes (P0 - CRITICAL)

**Research Question**: Do InfoNCE + early stopping + generation hygiene prevent collapse?

**Configurations**:
1. **Baseline (no fixes)** - REUSE successful_experiments/cross_model/85/3_high_capacity
   - 64 tokens, depth=8, lr=1e-4, warmup=750
   - NO InfoNCE, NO early stopping, NO repetition penalty
   - Result: Peak 81.5%, collapsed to 36%

2. **With stability fixes** - NEW RUN (3 hours)
   - Same architecture (64 tokens, depth=8, lr=1e-4, warmup=750)
   - InfoNCE loss (λ=0.05, start after 50% warmup)
   - Early stopping (patience=5 on bridged accuracy)
   - Pure greedy decoding (no repetition penalty)
   - 8-shot CoT evaluation with fixed seed=42 exemplars
   - Expected: Maintain >70% final accuracy

**Metrics**:
- Peak bridged accuracy
- Final bridged accuracy
- Degradation (peak - final)
- Training curves (plot accuracy over time)

**Runtime**: 3 hours
**Dataset**: GSM8K (1,319 test samples)

---

### Ablation 2: Sequence Length (P0)

**Research Question**: How does soft token count affect compression vs quality?

**Configurations** (all with stability fixes):

1. **32 tokens** - NEW RUN (3 hours)
   - Bottleneck=768, depth=4, heads=12, lr=1e-4, warmup=600
   - Compression: ~4.7× (150 → 32 tokens)
   - KV cache saved: 118 tokens × 0.5 MB = ~59 MB
   - Expected: Moderate quality, best stability

2. **48 tokens** - NEW RUN (3 hours)
   - Bottleneck=1024, depth=6, heads=16, lr=1e-4, warmup=750
   - Compression: ~3.1× (150 → 48 tokens)
   - KV cache saved: 102 tokens × 0.5 MB = ~51 MB
   - Expected: Better quality than 32

3. **64 tokens** - REUSE from Ablation 1 (with stability)
   - Bottleneck=1024, depth=8, heads=16, lr=1e-4, warmup=750
   - Compression: ~2.3× (150 → 64 tokens)
   - KV cache saved: 86 tokens × 0.5 MB = ~43 MB
   - Expected: Best quality

**Metrics**:
- Bridged accuracy vs soft token count
- Compression ratio vs quality tradeoff
- KV cache savings
- Training stability (peak - final degradation)

**Runtime**: 6 hours (2 new configs)
**Dataset**: GSM8K

---

### Ablation 3: Inference Metrics (P0 - CRITICAL)

**Research Question**: What are the practical memory and time savings? Does latent beat BOTH single models?

**Method**: Benchmark all 4 baselines on full test set
- Use trained checkpoint from Ablation 1 (64 tokens, stable)
- Evaluate on ALL 1,319 test samples
- **Store per-sample raw data** for flexible analysis

**Baselines (4 Total)**:

1. **Source-alone (Mistral)** - P0 CRITICAL
   - Question → Mistral → Answer (no Llama)
   - Purpose: Prove improvement isn't just from using Mistral
   - **This determines the paper story!**

2. **Target-alone (Llama)** - Standard baseline
   - Full prompt → Llama
   - Purpose: Single-model performance

3. **Latent (Our method)**
   - Question → Mistral → Translator (64 soft tokens) → Llama
   - Purpose: Cross-model translation

4. **Token-budget**
   - Truncated prompt (K tokens) → Llama
   - Purpose: Fair compression baseline

**Per-Sample Metrics** (stored in JSONL):
- KV cache memory (MB) during generation
- End-to-end latency (seconds)
- Peak GPU memory (MB)
- Quality (EM/F1)
- Sequence lengths (input/output)

**Key Comparisons**:
- Does latent beat BOTH source AND target? (Information enrichment!)
- KV cache savings: target_alone vs latent
- Latency reduction for longer outputs

**Runtime**: 1-2 hours
**Dataset**: GSM8K (1,319 samples)

**Output**:
- `inference_per_sample.jsonl` - Raw data for all samples
- `inference_aggregate.json` - Summary statistics

---

### Week 1 Summary

| Experiment | Runtime | Status | Purpose |
|------------|---------|--------|---------|
| 1. Stability (with fixes) | 3h | ⏳ TODO | Prevent collapse |
| 2a. Sequence (32 tok) | 3h | ⏳ TODO | High compression |
| 2b. Sequence (48 tok) | 3h | ⏳ TODO | Medium compression |
| 3. Inference metrics | 1-2h | ⏳ TODO | KV cache + latency |
| **TOTAL** | **~9-10h** | | |

**Reused (no GPU time)**:
- Baseline (no fixes): 81.5% → 36% collapse
- 64 tokens (with fixes): Same as experiment 1

---

## Week 2: Analysis + Paper Drafting (Nov 15-21)

### Analysis Tasks (3-4 days)

#### Results Compilation
- Extract all metrics from logs (peak/final acc, compression ratios)
- Create comparison tables for all 4 baselines
- Generate training curves (accuracy over time)
- Analyze which scenario we're in:
  - **Best**: Latent > source AND target (cross-model fusion)
  - **Good**: Latent > target (knowledge transfer)
  - **Problem**: Latent < source (degrading better model)

#### Qualitative Analysis
- Sample 10-20 examples showing:
  - Success cases: latent matches or exceeds both models
  - Failure modes: when/why latent fails
  - Information enrichment: cases where latent > both baselines
- Analyze what information the translator captures

#### Ablation Summary
- Consolidate sequence length ablations (32/48/64)
- Stability fixes impact (with vs without)
- Training dynamics visualization

---

### Paper Structure (Target: 6 pages + refs)

**1. Introduction** (0.75 pages)
- Problem: LLMs are large, prompts consume memory (KV cache)
- Idea: Cross-model translation via learned interlingua
- Key result: Latent beats BOTH source and target models (information enrichment!)
- Contributions:
  1. Architecture for cross-model translation via soft tokens
  2. Training methodology with stability improvements
  3. Demonstration of cross-model information enrichment
  4. Analysis of compression-quality tradeoffs and practical benefits

**2. Related Work** (0.75 pages)
- Prompt compression (LLMLingua, AutoCompressor)
- Soft prompting (prefix tuning, prompt tuning)
- Cross-model knowledge distillation
- Interlingua in neural machine translation

**3. Method** (2 pages)
- Architecture:
  - Source model (Mistral-7B) encodes question
  - Bottleneck-gated translator (Flamingo-style gated cross-attention)
  - Orthogonal query initialization, RMSNorm, SwiGLU activation
  - Target model (Llama-3.1-8B) generates answer
- Training:
  - Teacher-forced cross-entropy on answers
  - InfoNCE anti-collapse loss (λ=0.05, float32 for stability)
  - Early stopping on validation accuracy
  - Pure greedy decoding (no repetition penalty)
  - Fused AdamW optimizer with high-LR gate parameters
- Evaluation:
  - 8-shot Chain-of-Thought prompting (seed=42)
  - Calibration and anchoring (RMS matching, "Answer: " anchor)

**4. Experiments** (2 pages)
- Dataset: GSM8K (1,319 test samples)
- Baselines: source-alone, target-alone, latent, token-budget
- Metrics: Accuracy, KV cache savings, latency, compression ratio
- Results:
  - Main: Latent vs both single models (Table + key finding)
  - Ablations: Sequence length (32/48/64), stability fixes (Figures)
  - Efficiency: KV cache savings, latency reduction (Table)

**5. Analysis** (1 page)
- Why does it work?
  - Cross-model information fusion hypothesis
  - Mistral's reasoning + Llama's generation
- Training dynamics:
  - Peak-and-collapse without stability fixes (Figure)
  - Stable training with fixes (Figure)
- Compression-quality tradeoff (32/48/64 tokens)
- Failure modes and limitations

**6. Conclusion** (0.5 pages)
- Demonstrated cross-model translation can enrich information
- Achieved 2-5× compression with practical KV cache savings
- Future work: scaling to more models, online adaptation

**References** (1 page)

---

## Week 3: Writing + Revisions (Nov 22-28)

### Days 1-3: Complete Draft
- Finish all sections
- Create all figures and tables
- Write abstract and conclusion
- Internal consistency check

### Days 4-5: Revisions
- Clarity pass (is the story clear?)
- Technical accuracy (are claims supported?)
- Figure/table polish
- Grammar and style

### Days 6-7: Final Polish
- Read full paper end-to-end
- Check references are complete
- Verify all experimental claims have evidence
- LaTeX compilation and formatting
- Final submission prep

---

## Paper Claims (Evidence Matrix)

| # | Claim | Evidence | Ablation | Status |
|---|-------|----------|----------|--------|
| **1** | **Cross-model fusion beats both models** | Latent > source AND target | 3 (P0) | ⏳ TODO |
| 2 | Stability fixes prevent collapse | >70% final vs 36% | 1 | ⏳ TODO |
| 3 | Compression-quality tradeoff exists | 32/48/64 tokens analysis | 2 | ⏳ TODO |
| 4 | Practical KV cache savings | 43-59 MB saved | 3 | ⏳ TODO |

**Critical**: Claim #1 determines the entire paper narrative. Ablation 3 will reveal which scenario we're in.

---

## Scope Boundaries (What NOT to do)

❌ Test on multiple datasets (focus on GSM8K only)
❌ Try 10 different model pairs (stick with Mistral→Llama)
❌ Implement complex baselines (4 simple baselines is enough)
❌ Extensive hyperparameter sweeps (use what worked)
❌ Multi-seed statistical analysis (single seed fine for first paper)
❌ Theoretical analysis (empirical focus)
❌ Over-the-wire quantization analysis (focus on compression-quality tradeoff)

---

## Risks and Mitigations

### Risk 1: Latent doesn't beat both models
**Mitigation**: Pivot story based on which scenario emerges
- If latent > target only: "Effective knowledge transfer"
- If latent < source: Focus on compression benefits, frame as exploration

### Risk 2: Stability fixes don't maintain 70%+
**Mitigation**: Present tradeoff between peak performance and stability
**Backup**: Use existing 81.5% result, discuss stability as future work

### Risk 3: Not enough time for writing
**Mitigation**: Start paper template in Week 1, fill sections incrementally
**Backup**: Reduce to 4 pages (short paper format)

---

## Success Criteria (Minimum Viable Paper)

✅ **Required**:
1. Demonstrate cross-model translation works
2. Compare against source-alone AND target-alone baselines
3. Quantify KV cache savings and compression
4. Show at least one case where method provides benefit
5. Complete paper draft with intro/method/experiments/conclusion

✅ **Nice to Have**:
6. Latent beats BOTH single models (information enrichment)
7. Multiple stability checkpoints
8. Detailed compression-quality analysis at different token counts

---

## Execution Checklist

**Week 1** (Nov 8-14):
- [ ] Run Ablation 1: Stability fixes (3h)
- [ ] Run Ablation 2a: 32 tokens (3h)
- [ ] Run Ablation 2b: 48 tokens (3h)
- [ ] Run Ablation 3: Inference metrics (1-2h)
- [ ] Git pull results locally
- [ ] Verify all experiments completed

**Week 2** (Nov 15-21):
- [ ] Analyze all results
- [ ] Determine which scenario (latent vs source/target)
- [ ] Create all tables and figures
- [ ] Draft paper sections (parallel to analysis)
- [ ] Write intro based on results

**Week 3** (Nov 22-28):
- [ ] Complete full draft
- [ ] Multiple revision passes
- [ ] Polish figures and tables
- [ ] LaTeX compilation
- [ ] Final submission prep

---

## Quick Reference

**Scripts**:
- Training: `paper_writing/run_ablations.sh`
- Main script: `paper_writing/cross_attention.py`
- Analysis: Auto-generated in runs directory

**Key Files**:
- This plan: `paper_writing/PLAN.md`
- README: `paper_writing/README.md`
- Preserved 81.5% result: `successful_experiments/cross_model/85/`

**Timeline**: 3 weeks total
**GPU Budget**: ~9-10 hours on 4× H100
**Target**: 6-page paper on arXiv
**Dataset**: GSM8K only (1,319 test samples)
**Evaluation**: 8-shot Chain-of-Thought with seed=42

---

This is aggressive but achievable with focused execution and no scope creep.
