# LatentWire Comprehensive Revision Plan

**Source**: Analysis of ALL_REVIEWS.md (50 reviewers: 10 Claude, 10 ChatGPT, 10 Kimi, 10 Gemini, 10 Grok)
**Current Verdict**: 5.5/10 - Weak Reject / Borderline at MLSys

---

## 🔔 PLAN REVIEW STATUS (Round 2)

| Reviewer | Verdict | Key Condition |
|----------|---------|---------------|
| Claude | ✅ APPROVED | Add decision point after linear probe |
| ChatGPT | ✅ APPROVED WITH CHANGES | Re-scope CRITICAL, add execution gates |
| Kimi | ✅ CONDITIONALLY APPROVED | Add fallback strategies, fix resource estimates |
| Gemini | ✅ APPROVED | Prioritize Generation task as "existential insurance" |
| Grok | ✅ APPROVED | Gate Major tasks behind Critical completion |

**Consensus**: Plan is approved. Key amendments incorporated below.

---

## ⚠️ EXECUTION GATES (Decision Points)

Execute experiments in order. Stop and reassess at each gate.

### Gate 1: After Statistical Rigor (Week 1)
- Run full test sets + 10 seeds on SST-2, AG News, TREC
- **Pass**: Headline accuracy preserved within ±3 points → Proceed
- **Fail**: Revise performance claims, debug training stability

### Gate 2: After Linear Probe Baseline (Week 1-2)
- Run logistic regression on Llama layer-16 hidden states
- **If linear probe < Bridge - 5%**: Proceed with current narrative
- **If linear probe ≈ Bridge**: PIVOT narrative to:
  1. Latency/efficiency advantages (constant-size protocol)
  2. Cross-model capability (probe is same-model only)
  3. Generative transfer potential (probe can't generate)

### Gate 3: After Generation Task (Week 2)
- Run Llama→Mistral summarization (XSUM or CNN/DailyMail)
- **Pass**: ROUGE ≥80% of text baseline → "Communication" claim validated
- **Fail**: Scope paper to "classification and label-relevant compression" only
- **CRITICAL**: If Generation fails AND Linear Probe works → Paper contribution is dead

### Gate 4: After Latency Measurement (Week 2-3)
- Measure wall-clock Bridge vs. optimized Text-Relay vs. Direct Mistral
- **If speedup ≥10×**: Keep systems framing
- **If speedup 5-10×**: Rewrite claims with exact measured numbers
- **If speedup <5×**: Pivot away from systems contribution

---

## 🛡️ FALLBACK STRATEGIES (Negative Result Plans)

### If Linear Probe Matches Bridge Accuracy
**Defense** (per ChatGPT): Linear probe is NOT equivalent because:
1. It's not a communication protocol (no portable message)
2. Doesn't produce fixed-size representation across varying contexts
3. Doesn't support same batching/serving story
4. Can't enable generative transfer

**Pivot**: Reframe as "Efficient Cross-Model Adapter" emphasizing:
- Latency gains (M tokens vs. full text)
- Bidirectional transfer capability
- Constant-size protocol for serving

### If Gist Tokens Work Cross-Model
- Emphasize Bridge's bidirectional capability (Gist is unidirectional)
- Highlight wider model family support
- Focus on 3B threshold discovery (novel regardless)

### If Multi-Turn Fails
- Scope paper to "single-turn cross-model transfer"
- Document information bottleneck as limitation
- Propose future work: context-aware latent accumulation

### If Qwen→Mistral Still Fails After Investigation
- **Drop "Interlingua" terminology completely** (per Gemini)
- Use "Learned Semantic Compression" or "Neural Bridge" instead
- Define explicit compatibility criteria (similar architecture, comparable size)
- Report compatibility matrix for tested pairs

---

## Executive Summary

### What Reviewers Agree Is GOOD

1. **3B Parameter Threshold** - Most robust finding, practically useful, well-supported empirically
2. **Training Stability Contributions** - First-token objective, curriculum learning, calibration are valuable
3. **Honest Failure Reporting** - Section on training challenges (3.4) is exemplary
4. **Bidirectional Transfer** - Impressive engineering achievement
5. **Problem Identification** - Prefill bottleneck analysis is accurate and important
6. **Comprehensive Ablations** - Table 16 is thorough

### What Reviewers Agree Is PROBLEMATIC

1. **27x Speedup Claim** - Compares to strawman baseline (text+summarization), not fair comparison
2. **200 Sample Evaluation** - Too small, +/-4-6% variance undermines all performance claims
3. **Only Classification Tasks** - Claims "communication" but only tests classification, no generation/reasoning
4. **Missing Key Baselines** - Gist Tokens, LLMLingua, linear probe, direct Mistral classification
5. **No Multi-Turn Demo** - Core motivation (long conversations) never validated experimentally
6. **Qwen->Mistral Fails** - Undermines "interlingua" universality claim
7. **Overstated Terminology** - "Telepathically," "interlingua," "first," "wire protocol" all overclaim

### Reviewer Score Distribution

| Source | Avg Score | Range |
|--------|-----------|-------|
| Claude | 5.3/10 | 4-6 |
| ChatGPT | 5.7/10 | 4-7 |
| Kimi | 5.6/10 | 3-8 |
| Gemini | 5.4/10 | 3-8 |
| Grok | 5.7/10 | 4-7 |
| **Overall** | **5.5/10** | **3-8** |

### Reviewer Type Concerns

| Type | Main Concerns |
|------|---------------|
| Systems (R1) | 27x speedup methodology, latency measurement, memory overhead |
| Soft Prompts (R2) | Gist/ICAE baselines, 3B threshold novelty |
| Multi-Agent (R3) | Agent benchmarks, multi-turn, task selection mismatch |
| Representation (R4) | CKA analysis, Platonic hypothesis, Qwen failure explanation |
| Skeptical (R5) | Statistical rigor, novelty, overclaims, small eval |
| Reproducibility (R6) | Seeds, sample size, 200 samples insufficient |
| NLP Apps (R7) | Benchmark quality, outdated tasks (SST-2 from 2011) |
| Compression (R8) | Pareto curves, ICAE, missing compression baselines |
| Writing (R9) | Terminology, structure, "telepathically," "interlingua" |
| Industry (R10) | Deployment cost, ROI, per-pair training impractical |

---

## CRITICAL Tasks (Must Fix for Acceptance)

**⚡ EXECUTION ORDER** (per reviewer consensus):
1. Statistical Rigor → 2. Linear Probe → 3. Latency → 4. Generation → 5. Code Release

---

### 1. Statistical Rigor & Sample Size (45+ reviewers) — FIRST

**Problem**: Current 200-sample evaluation with +/-4-6% variance undermines all performance claims.

**Tasks**:
- [ ] Run full test sets: SST-2 (872), AG News (7,600), TREC (500)
- [ ] Increase seeds from 3 to 10+
- [ ] Compute 95% confidence intervals
- [ ] Use paired bootstrap CI or McNemar test (per ChatGPT: more appropriate than t-tests)
- [ ] Report p-values for all baseline comparisons
- [ ] Apply Bonferroni correction for multiple comparisons (α/K for K≥30 comparisons)
- [ ] Update Table 13 with significance tests

**What to Report**:
- Full test set accuracies with mean +/- 95% CI over 10 seeds
- Statistical significance (p-values, corrected) for all method comparisons

---

### 2. Linear Probe Baseline (18+ reviewers) — SECOND (Kill Switch #1)

**Problem**: If simple logistic regression on Llama hidden states matches Bridge, core contribution collapses.

**Protocol**:
- [ ] Extract Llama layer-16 hidden states (4096-dim) for all datasets
- [ ] Train logistic regression classifiers (sklearn default)
- [ ] Evaluate with 10 seeds on SST-2, AG News, TREC
- [ ] Compare accuracy to Bridge

**Decision Point** (see Gate 2):
- If Bridge > Linear Probe + 5%: Proceed with "enabling cross-model transfer" narrative
- If Bridge ≈ Linear Probe: Pivot to efficiency/latency/generative narrative

**Defense if Linear Probe Works** (per ChatGPT):
- Probe is NOT a communication protocol
- Probe doesn't produce portable fixed-size message
- Probe can't enable generative transfer
- Probe doesn't support batching/serving benefits

---

### 3. Fair Baseline Comparisons (30+ reviewers) — THIRD

**Problem**: Text-Relay baseline uses summarization step; the critique is **baseline fairness**, not "missing measurements."

#### 3A. Direct Classification Baseline
- [ ] Implement Mistral direct classification on full text (no relay)
- [ ] Implement Mistral on truncated text (token-budget matched to M=8)
- [ ] Run both with 10 seeds on all tasks
- [ ] Compare latency and accuracy

#### 3B. Optimized Text-Relay Variants (per ChatGPT)
- [ ] Text-Relay with label-preserving compression prompt (not generic summarization)
- [ ] Text-Relay with strict summary token budget matched to M
- [ ] Document fair comparison with matched constraints

#### 3C. LLMLingua Comparison (compression baseline)
- [ ] Implement LLMLingua at 15-30x compression ratio
- [ ] Compare quality and speedup at matched compression

---

### 4. Measured End-to-End Latency (35+ reviewers) — FOURTH

**Problem**: 27x speedup is measured wall-clock, but baseline design (summarization) inflates the number. Need fair comparison.

**Tasks**:
- [ ] Add timing instrumentation to all pipeline stages
- [ ] Measure wall-clock latency breakdown: encoder, adapter, prefill, generation
- [ ] Report p50, p95, p99 latencies under concurrent requests
- [ ] Test batch sizes: 1, 4, 16, 64
- [ ] Measure throughput under concurrency
- [ ] Analyze GPU utilization and memory bandwidth
- [ ] Compare on same hardware: Text-Relay (optimized), Bridge, Direct Mistral
- [ ] Create latency breakdown charts
- [ ] Update speedup claims with proper caveats

---

### 5. Generation Task (35+ reviewers) — FIFTH (Kill Switch #2 / Existential)

**Problem**: All current tasks are classification. "Communication" claims require generation validation.

**Why Existential** (per Gemini): This is your insurance against the Linear Probe trap. A linear probe can classify, but it CANNOT transfer context for generation. If Bridge enables generative transfer, you have a strong contribution regardless of classification performance.

**Protocol**:
- [ ] Adapt Bridge architecture for generation (soft tokens → Mistral generates summary)
- [ ] Train Llama→Mistral bridge on XSUM (extreme compression) or CNN/DailyMail
- [ ] Minimum 10K training examples for stable training
- [ ] Evaluate ROUGE-1, ROUGE-2, ROUGE-L (report all three)
- [ ] Compare to text baseline (Mistral with full text)

**Success Criterion**: Bridge achieves ≥80% of text baseline ROUGE score

**Pivot Narrative if Successful**:
> "While simple probes suffice for classification, LatentWire is the only method that enables **generative transfer** (summarization, QA) without text decoding."

---

### 6. Code Release (30+ reviewers) — SIXTH (Parallelize)

**Non-negotiable for final acceptance at any venue.**

- [ ] Clean up training scripts
- [ ] Add evaluation scripts
- [ ] Document hyperparameters (all random seeds used)
- [ ] Provide exact reproduction command: `python train.py --config config.yaml`
- [ ] Prepare model checkpoints for release (or upload plan)
- [ ] Write exact prompts and decoding parameters
- [ ] Create README with reproduction instructions
- [ ] Add inference script: `python inference.py --bridge_ckpt [PATH] --text [PROMPT]`
- [ ] Add reproduction time estimates (e.g., "full SST-2 10-seed run: 24 GPU-hours on H100")
- [ ] Add reproducibility checklist

---

## MOVED TO MAJOR (Previously CRITICAL)

### Reasoning Benchmark — Now MAJOR (per ChatGPT: focus on ONE non-classification task)
- [ ] Train bridge on GSM8K (chain-of-thought transfer)
- [ ] Evaluate accuracy on test set
- [ ] Compare to LatentMAS results
- **Note**: Do this AFTER Generation task succeeds. One task is sufficient.

### Gist Tokens Reproduction — Now MAJOR (per ChatGPT: too expensive, failure-prone)
- [ ] Attempt Gist Tokens training on Llama
- [ ] Train linear projection to Mistral embedding space
- [ ] If reproduction fails, compare to reported numbers from paper
- **Note**: LLMLingua comparison (in 3C) is more practical

### Cross-Model QA — Now MAJOR
- [ ] Report SQuAD Llama→Mistral F1
- [ ] Add HotpotQA cross-model results

---

## MAJOR Tasks (Significantly Strengthens Paper)

### 7. Attention Analysis - Prove Soft Tokens Are Used (15+ reviewers)

**Problem**: Zero-prefix control shows coherent generation without prefix. Need proof soft tokens contribute.

**Tasks**:
- [ ] Generate attention heatmaps showing receiver heads attending to soft token positions
- [ ] Ablation: Mask soft tokens during generation, measure performance drop
- [ ] Counterfactual: Shuffle soft tokens, measure degradation
- [ ] Train linear probes on soft tokens to verify semantic content
- [ ] Visualize what soft tokens encode
- [ ] Add quantitative cluster quality metric (silhouette score or Davies-Bouldin) to Figure 2

---

### 7. Representation Analysis (20+ reviewers)

#### 7A. Cross-Model Similarity (CKA/SVCCA)
- [ ] Compute Centered Kernel Alignment between Llama layer-16 and Mistral embeddings
- [ ] Analyze Qwen->Mistral failure with CKA
- [ ] Test correlation between CKA and transfer success

#### 7B. Linear Probing of Soft Tokens
- [ ] Train linear classifiers on 8 soft tokens to predict semantic features
- [ ] Verify what information is preserved in compressed representation

#### 7C. Intrinsic Dimensionality Analysis
- [ ] Measure effective rank or participation ratio of soft token representations
- [ ] Determine if all dz=4096 dimensions are used or if it's low-rank

#### 7D. Nearest Neighbor Analysis
- [ ] For each soft token, find nearest text tokens in receiver embedding space
- [ ] Interpretability of what soft tokens represent

---

### 8. More Model Pairs & Compatibility Analysis (20+ reviewers)

#### 8A. Test Additional Model Families
- [ ] Train Llama->Gemma bridge
- [ ] Train Llama->Phi-3 bridge
- [ ] Test Qwen-72B->Mistral (isolate size vs. architecture)
- [ ] Test Mistral->Qwen (reverse direction)
- [ ] Document compatibility patterns

#### 8B. Analyze Qwen->Mistral Failure
- [ ] Use CKA to measure representation mismatch
- [ ] Test if larger Qwen helps
- [ ] Analyze vocabulary/embedding differences

---

### 9. Multi-Turn & Multi-Agent Experiments (25+ reviewers)

#### 9A. Multi-Turn Dialogue
- [ ] Implement 3-5 turn conversation framework
- [ ] Run experiments on MultiWOZ or debate dataset
- [ ] Measure cumulative latency and degradation
- [ ] Test if soft tokens can accumulate over turns

#### 9B. 3+ Agent System
- [ ] Test if shared interlingua works with 3 heterogeneous agents
- [ ] Llama, Mistral, and one other model

#### 9C. Agent Benchmark
- [ ] Evaluate on AgentBench, GAIA, or AutoGen-style task completion
- [ ] Multi-turn tool-use scenario
- [ ] Document agent applicability

---

### 10. Compression Baseline Comparisons (18+ reviewers)

#### 10A. LLMLingua Comparison
- [ ] Implement LLMLingua at 15-30x compression ratio
- [ ] Compare quality and speedup at similar compression ratios

#### 10B. ICAE and 500xCompressor
- [ ] Adapt ICAE for cross-model use with projection layer
- [ ] Compare 500xCompressor (if applicable)

#### 10C. Pareto Curve (Compression vs. Quality)
- [ ] Plot accuracy vs. compression ratio for M in {2, 4, 8, 16, 32, 64}
- [ ] Understand tradeoff

---

### 11. Ablation Studies with Multiple Seeds (15+ reviewers)

#### 11A. Re-run All Ablations
- [ ] Re-run Table 17 (M scaling) with 3+ seeds
- [ ] Re-run Table 19 (source layer) with 3+ seeds
- [ ] Re-run Table 20 (model pairs) with 3+ seeds
- [ ] Re-run Table 21 (training-free) with 3+ seeds

#### 11B. Factorial Analysis of SST-2 Fix
- [ ] Test each of 6 changes independently: diversity loss, M, LR, steps, layer, sampling
- [ ] Identify which changes actually matter
- [ ] Document interaction effects

#### 11C. Complete Source Layer Ablation
- [ ] Test every layer (0, 4, 8, 12, 16, 20, 24, 28, 31) with 3 seeds
- [ ] Comprehensive view beyond Table 19's 6 layers

#### 11D. PerceiverResampler Depth Ablation
- [ ] Test 1-layer, 2-layer, 4-layer PerceiverResampler

#### 11E. Adapter Architecture Ablation
- [ ] Test residual adapters, LoRA-style, bottleneck adapters vs. current full-rank
- [ ] 537M is large; smaller adapters might work

---

### 12. Scaling & Deployment Analysis (25+ reviewers)

#### 12A. 70B Model Validation
- [ ] Evaluate Llama-70B -> Mistral-7B and vice versa
- [ ] Validate scaling projections
- [ ] Measure latency scaling

#### 12B. Quantization Analysis
- [ ] Quantize bridge to int8
- [ ] Quantize bridge to int4
- [ ] Measure accuracy drop and memory savings

#### 12C. Memory Overhead Analysis
- [ ] Report VRAM overhead of 537M bridge
- [ ] Report KV-cache for M tokens
- [ ] Report peak memory during training

#### 12D. Distributed Systems Evaluation
- [ ] Measure latency when sender and receiver are on different GPUs/nodes
- [ ] Measure network transfer overhead
- [ ] Document deployment considerations

---

### 13. Cross-Task & Zero-Shot Generalization (15+ reviewers)

#### 13A. Train on One Task, Test on Another
- [ ] Train on AG News, test on TREC (both multi-class)
- [ ] Show if bridge learns task-agnostic representations

#### 13B. Universal Adapter Across Tasks
- [ ] Train single adapter on mixture of SST-2, AG News, TREC
- [ ] Test if same bridge works for multiple tasks

---

### 14. Error Analysis & Interpretability (12+ reviewers)

#### 14A. Qualitative Error Analysis
- [ ] Show concrete examples of success/failure cases with explanation
- [ ] Reveal failure modes

#### 14B. Confusion Matrices
- [ ] Show confusion matrices for AG News forward vs. reverse transfer
- [ ] Reveal which classes fail in asymmetric transfer

#### 14C. Soft Token Interpolation
- [ ] Linearly interpolate between two soft tokens
- [ ] Check if generation interpolates semantically

---

### 15. Training & Hyperparameter Analysis (10+ reviewers)

#### 15A. Training Data Scaling
- [ ] Test performance with 100, 1K, 10K, 32K training samples
- [ ] Cold-start analysis for new deployments

#### 15B. Hyperparameter Sensitivity
- [ ] Plot learning rate, batch size, M, dz sensitivity curves
- [ ] Not just optimal points, show robustness

#### 15C. Curriculum Learning Ablation
- [ ] Test without curriculum
- [ ] Test with different warm-up lengths
- [ ] Test with different phase triggers
- [ ] Provide algorithmic specification of curriculum schedule

---

## NICE TO HAVE Tasks (Strengthens Specific Claims)

### 16. Modern Benchmarks (12+ reviewers)
- [ ] Evaluate on MMLU (knowledge/reasoning)
- [ ] Evaluate on SuperGLUE (standardized NLU)
- [ ] Evaluate on Big-Bench (diverse capabilities)

### 17. Long-Context Evaluation (12+ reviewers)
- [ ] Test with 4K, 8K, 16K token inputs
- [ ] Evaluate compression benefits at longer contexts
- [ ] M scaling for longer contexts

### 18. Precision Task Evaluation - Negative Results (8+ reviewers)
- [ ] Test on arithmetic (GSM8K) - demonstrate failure
- [ ] Passkey retrieval
- [ ] Exact value extraction
- [ ] Document failure modes for precision tasks

### 19. Alternative Baselines (10+ reviewers)
- [ ] Prompt-tuning on Llama (sender) - current baseline is receiver only
- [ ] Joint sender+receiver prompt-tuning
- [ ] Relative representations (Moschella et al.) - training-free comparison

### 20. Calibration & Normalization Ablations (8+ reviewers)
- [ ] Compare RMS matching vs. mean-variance normalization vs. percentile matching vs. whitening
- [ ] Test pre-norm vs. post-norm in adapter

### 21. Diversity Loss Analysis (5+ reviewers)
- [ ] Plot token-to-token cosine similarity with/without diversity loss
- [ ] Test if diversity loss helps or hurts multi-class

### 22. Production & Deployment (15+ reviewers)
- [ ] Develop failure detection mechanism (confidence score or anomaly detection)
- [ ] Cost-benefit analysis (dollar cost for N model pairs x T tasks)
- [ ] Model versioning strategy (does bridge work after Llama-3.1 -> Llama-3.2?)

### 23. Security & Privacy (10+ reviewers)
- [ ] Test if malicious soft tokens can cause harmful outputs
- [ ] Test if bridge training leaks sensitive information

---

## WRITING & PRESENTATION Tasks

### Paper Restructuring (per Claude)
- [ ] Move SQuAD/HotpotQA results to appendix (classification is primary contribution)
- [ ] Restructure paper around cross-model classification as primary contribution
- [ ] Add "Key Differentiators" section before related work with architectural diagrams

### Terminology Changes (Must Do)

| Current Term | Problem | Fix | Reviewers |
|--------------|---------|-----|-----------|
| "Telepathically" | Informal, hyperbolic | **Remove entirely** | 15+ |
| "Interlingua" | Implies universality (Qwen fails) | **Drop completely** (per Gemini). Use "Learned Semantic Compression" or "Neural Bridge" | 20+ |
| "Wire protocol" | Implies standardization | "learned message format" or "fixed-size representation interface" | 12+ |
| "27x speedup" | Misleading baseline | "≤27× vs. summarization-based relay; [X]× vs. optimized relay" (fill X with measured) | 25+ |
| "First" (various) | Fragile claims | Remove or narrowly scope | 15+ |
| "3B threshold" | Seems like restatement | Reframe: "Cross-model transfer requires **3× larger models** than single-model prompt tuning (Lester et al. 2021)" | 10+ |
| "Universal/compatible" | Qwen fails | Narrow to "tested pairs" or add explicit compatibility criteria | 10+ |

### Key Differentiators to Emphasize (per Kimi)
- [ ] Add section comparing Bridge to alternatives:
  1. **Bidirectional transfer** (Gist is unidirectional)
  2. **Model-agnostic protocol** (same soft tokens for multiple receivers)
  3. **Constant-size communication** (independent of context length)
  4. **Training-free inference on new tasks** (plug and play)
- [ ] Side-by-side architectural diagram: Text-Relay vs. Linear-Probe vs. Bridge vs. Gist

### Content to Add

- [ ] SOTA context: "91.5% vs. 97% SOTA on SST-2" throughout
- [ ] System diagram: Visual comparison of Text-Relay vs. Bridge pipeline
- [ ] Algorithm pseudocode for training with curriculum schedule
- [ ] Notation table
- [ ] Reproducibility checklist

### Tables & Figures to Fix

- [ ] Consistent notation (dz vs. d_z)
- [ ] Error bars on all tables
- [ ] Figure 1 y-axis start at 0
- [ ] Add latency breakdown plots
- [ ] Add training curves

---

## Summary Statistics

### By Priority

| Priority | Categories | Affected Reviewers |
|----------|------------|-------------------|
| CRITICAL | 6 categories | 35-45+ each |
| MAJOR | 9 categories | 10-25 each |
| NICE TO HAVE | 8 categories | 5-15 each |

### Top 10 Most Requested (by reviewer count) — REVISED PRIORITY

1. **Full test sets + statistical rigor** (45+ reviewers) — CRITICAL #1
2. **Linear probe baseline** (18+ reviewers) — CRITICAL #2 (Kill Switch)
3. **End-to-end latency measurements** (35+ reviewers) — CRITICAL #4
4. **Generation task (summarization/QA)** (35+ reviewers) — CRITICAL #5 (Existential Insurance)
5. **Fair baseline (direct classification)** (30+ reviewers) — CRITICAL #3
6. **Code release** (30+ reviewers) — CRITICAL #6
7. **Multi-turn experiment** (25+ reviewers) — MAJOR (can cut if constrained)
8. **More model pairs** (20+ reviewers) — MAJOR (prove ONE other pair works)
9. **Qwen→Mistral investigation** (20+ reviewers) — MAJOR (fix or scope claims)
10. **Reasoning benchmark (GSM8K/MATH)** (25+ reviewers) — MAJOR (do AFTER generation)

---

## Venue Fit Assessment

### MLSys (Current Target)

**Pros**:
- Systems motivation (prefill bottleneck) is strong
- Throughput/batching analysis is relevant
- Constant-size protocol is architectural contribution

**Cons**:
- Limited systems evaluation (no multi-GPU, no serving, no production metrics)
- Core contribution is ML technique (soft prompting), not systems innovation
- Speedup claims need much stronger validation

**Required for MLSys acceptance**:
- End-to-end latency measurements
- Multi-GPU/distributed evaluation
- Memory bandwidth analysis
- Serving metrics (throughput under concurrency)
- Production cost analysis

### Alternative Venues

**ACL/EMNLP** (NLP Focus):
- Better fit if focus on cross-model transfer as NLP contribution
- Requires: Generation tasks, modern benchmarks, error analysis

**NeurIPS/ICLR** (ML Focus):
- Better fit if add representation analysis
- Requires: CKA/SVCCA, theoretical grounding, broader evaluation

---

## Consensus Issues (All 50 Reviewers Agree)

1. **3B threshold is solid** - Best contribution
2. **Training stability fixes are valuable** - Document honestly
3. **27x speedup claim is problematic** - Need fair baseline
4. **200 samples too small** - Need full test sets
5. **Only classification tasks** - Need generation/reasoning
6. **Gist Tokens missing** - Essential comparison
7. **Qwen->Mistral failure** - Limits "interlingua" claim
8. **No multi-turn demo** - Core motivation not validated

---

## Resource Requirements

### Compute (REVISED per Kimi - add 100% buffer)

| Experiment | Optimistic | Realistic (with debugging) |
|------------|------------|---------------------------|
| Full test set evaluation | 20h | 30h |
| 10 seeds × 3 tasks | 50h | 80h |
| Linear probe baseline | 10h | 15h |
| New baselines (LLMLingua, Text-Relay variants) | 30h | 50h |
| Generation task (XSUM) | 40h | 80h |
| Model pairs (2-3 additional) | 100h | 150h |
| Multi-seed ablations | 80h | 120h |
| Qwen→Mistral investigation | 30h | 50h |
| Buffer for failed runs | — | 150h |
| **Total** | ~360h | **~725h** |

**Recommended budget**: **800-1000 GPU-hours** (per Kimi)

### Minimum Viable Paper (If Resource-Constrained)

**Cannot cut** (jeopardizes acceptance):
- Statistical rigor (50h) — non-negotiable
- Linear probe baseline (15h) — existential threat
- Fair latency measurement (30h) — claim-critical
- Generation task (80h) — validates "communication"
- Code release (5h) — venue requirement

**Minimum viable total**: ~180 GPU-hours

**Can cut without losing core contribution**:
- ❌ 70B model scaling (saves 200-300h) — theoretical roofline analysis sufficient
- ❌ Multi-node distributed (saves 50h) — single-node multi-GPU is enough
- ❌ Full multi-agent benchmark suite (saves 80h) — one 3-turn dialogue demo sufficient
- ❌ Additional model pairs beyond 1-2 (saves 100h) — prove one other pair works
- ❌ Some ablations (adapter architecture, full layer sweep) — keep most impactful only

### Infrastructure
- H100 or A100 GPUs (4× recommended)
- Storage for checkpoints (~500GB)
- Experiment tracking (Weights & Biases recommended)

---

## Risk Mitigation

### If 70B Access Unavailable
- Focus on quantization and memory analysis instead
- Provide theoretical roofline analysis
- Acknowledge as limitation

### If Gist Tokens Reproduction Fails
- Document attempt honestly
- Compare to reported numbers from paper
- Focus on other baselines

### If Multi-Turn Results Are Weak
- Document failure honestly
- Analyze why (information loss, accumulation issues)
- Propose future work

### If Linear Probe Matches Bridge
- Major concern - need deeper analysis
- Focus on latency/efficiency advantages
- Emphasize cross-model capability (linear probe is same-model)

---

## Final Submission Checklist

### Experiments
- [ ] All CRITICAL experiments complete
- [ ] At least 70% of MAJOR experiments complete
- [ ] Results tables updated with new data
- [ ] All numbers verified and reproducible

### Writing
- [ ] Terminology fixed (no "telepathically")
- [ ] Speedup claims qualified
- [ ] SOTA context added
- [ ] System diagrams included
- [ ] Algorithm pseudocode added
- [ ] Notation table added
- [ ] References complete and formatted

### Code
- [ ] Training scripts cleaned
- [ ] Evaluation scripts tested
- [ ] README written
- [ ] Hyperparameters documented
- [ ] Checkpoints prepared (or upload plan)
- [ ] License added

### Submission
- [ ] Formatted for venue
- [ ] Abstract within limit
- [ ] Page count within limit
- [ ] Figures high-quality
- [ ] Supplementary materials ready
- [ ] Ethics statement (if required)
- [ ] Reproducibility checklist (if required)

---

## Venue Reassessment Strategy (per Grok)

**After experiments complete**, reassess venue based on results:

| If This Works Best | Target Venue | Narrative |
|--------------------|--------------|-----------|
| Latency/serving metrics strong | **MLSys** | "Constant-size protocol saves HBM bandwidth" |
| Representation analysis (CKA) strong | **ICLR/NeurIPS** | "Meaning transfers without words" |
| Generation task strong | **ACL/EMNLP** | "Cross-model generative transfer" |

---

**Bottom Line**: The 3B threshold finding is publication-worthy. All 5 AI reviewers approved the plan with amendments (now incorporated). Execute the gates in order—Linear Probe and Generation Task are the kill switches that determine paper viability.
