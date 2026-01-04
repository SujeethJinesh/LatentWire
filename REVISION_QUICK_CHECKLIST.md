# LatentWire Revision Quick Checklist

**Based on 50 reviewer feedback (Claude, ChatGPT, Kimi, Gemini, Grok)**

---

## MUST FIX (Critical for Acceptance)

### Statistical Rigor ⚠️
- [ ] Run full test sets (not 200 samples): SST-2 (872), AG News (7600), TREC (500)
- [ ] Increase from 3 seeds to 10+ seeds
- [ ] Report 95% confidence intervals
- [ ] Add paired t-tests with Bonferroni correction
- [ ] Report p-values for all comparisons

**Impact**: 45+ reviewers; foundational for all claims

---

### Fair Baselines ⚠️
- [ ] **Direct Mistral classification** on full text (no summarization)
- [ ] **Direct Mistral classification** on truncated text (token-budget matched to M=8)
- [ ] **Linear probe** baseline: Train classifier on Llama layer-16 hidden states
- [ ] **Gist Tokens + projection** baseline: Train Gist on Llama, project to Mistral

**Impact**: 30+ reviewers; speedup claim validity depends on this

---

### Measured Latency ⚠️
- [ ] End-to-end wall-clock time including encoder forward pass
- [ ] Breakdown: encoder time, adapter time, prefill time, generation time
- [ ] Report p50, p95, p99 latencies
- [ ] Throughput under concurrent requests (batch=1,4,16,64)
- [ ] Compare on same hardware: Text-Relay (optimized), Bridge, Direct Mistral
- [ ] GPU utilization and memory bandwidth analysis

**Impact**: 35+ reviewers; systems contribution validity

---

### Broader Tasks ⚠️
- [ ] **Generation task**: CNN/DM or XSUM with ROUGE scores for Llama→Mistral
- [ ] **Reasoning task**: GSM8K or MATH for Llama→Mistral
- [ ] **Cross-model QA**: Report SQuAD/HotpotQA F1 for Llama→Mistral (missing from Table 13)

**Impact**: 35+ reviewers; "communication" claims require generation validation

---

### Prove Soft Tokens Are Used ⚠️
- [ ] Attention heatmaps showing receiver attending to soft token positions
- [ ] Ablation: Mask soft tokens during generation, measure drop
- [ ] Counterfactual: Shuffle soft tokens, measure degradation

**Impact**: 15+ reviewers; validates method isn't learning dataset priors

---

## SHOULD FIX (Significantly Strengthens)

### Multi-Turn Experiments
- [ ] 3-5 turn dialogue experiment (e.g., MultiWOZ)
- [ ] Measure latency with cumulative soft tokens
- [ ] Test information degradation over rounds

**Impact**: 25+ reviewers; validates "constant overhead" claim

---

### Representation Analysis
- [ ] CKA similarity between Llama layer-16 and Mistral embedding space
- [ ] Linear probing: Train classifiers on 8 soft tokens
- [ ] Analyze Qwen→Mistral failure with CKA
- [ ] Intrinsic dimensionality of soft tokens (effective rank)

**Impact**: 20+ reviewers; explains mechanism

---

### More Model Pairs
- [ ] Test Llama→Gemma
- [ ] Test Llama→Phi-3
- [ ] Test Qwen-72B→Mistral (isolate size vs. architecture)
- [ ] Test Mistral→Qwen (reverse direction)

**Impact**: 20+ reviewers; "interlingua" generality

---

### Compression Baselines
- [ ] LLMLingua at same compression ratio (15-30×)
- [ ] ICAE with cross-model projection
- [ ] 500xCompressor comparison
- [ ] Pareto curve: accuracy vs. M ∈ {2,4,8,16,32,64}

**Impact**: 18+ reviewers; compression contribution

---

### Scaling & Deployment
- [ ] Evaluate on Llama-70B or Mistral-70B
- [ ] Quantize bridge to int8/int4, measure accuracy drop
- [ ] Report VRAM overhead (bridge + KV cache)
- [ ] Distributed setting: latency with network transfer

**Impact**: 25+ reviewers; production viability

---

### Ablations with Seeds
- [ ] Re-run Table 17 (M scaling) with 3+ seeds
- [ ] Re-run Table 19 (source layer) with 3+ seeds
- [ ] Factorial analysis of SST-2 fix (6 changes independently)
- [ ] Complete source layer sweep (every 4th layer, 3 seeds)
- [ ] PerceiverResampler depth ablation (1, 2, 4 layers)

**Impact**: 15+ reviewers; ablation validity

---

### Cross-Task Generalization
- [ ] Train on AG News, test on TREC (zero-shot transfer)
- [ ] Train universal adapter on mixture of tasks
- [ ] Test if same bridge works for multiple tasks

**Impact**: 15+ reviewers; practical deployment

---

## NICE TO HAVE (Strengthens Specific Claims)

### Modern Benchmarks
- [ ] MMLU (knowledge/reasoning)
- [ ] SuperGLUE (standardized NLU)
- [ ] Big-Bench (diverse capabilities)

---

### Agent Benchmarks
- [ ] AgentBench or GAIA evaluation
- [ ] 3+ agent system test
- [ ] Tool-use scenario

---

### Error Analysis
- [ ] Qualitative examples of success/failure
- [ ] Confusion matrices for AG News forward vs. reverse
- [ ] Soft token interpolation experiments

---

### Training Analysis
- [ ] Data scaling: 100, 1K, 10K, 32K samples
- [ ] Hyperparameter sensitivity curves
- [ ] Curriculum learning ablation
- [ ] Algorithmic specification of curriculum schedule

---

### Production Features
- [ ] Failure detection mechanism
- [ ] Cost-benefit analysis ($$ per model pair)
- [ ] Model versioning strategy
- [ ] Security analysis (adversarial soft tokens)

---

### Long Context
- [ ] Evaluate with 4K, 8K, 16K token inputs
- [ ] M scaling for longer contexts

---

### Precision Tasks (Document Failure)
- [ ] Test on arithmetic (demonstrate failure)
- [ ] Passkey retrieval
- [ ] Exact value extraction

---

### Alternative Baselines
- [ ] Prompt-tuning on Llama (sender)
- [ ] Joint sender+receiver prompt-tuning
- [ ] Relative representations (Moschella et al.)

---

### Calibration Methods
- [ ] RMS vs. mean-variance vs. percentile vs. whitening
- [ ] LayerNorm pre-norm vs. post-norm

---

## WRITING & PRESENTATION

### Terminology Changes
- [ ] Remove "telepathically" (15+ reviewers)
- [ ] Change "interlingua" → "learned semantic compressor" or "cross-model adapter"
- [ ] Qualify "27× speedup" → "vs. summarization-based text relay"
- [ ] Remove/soften "first" claims
- [ ] Contextualize "3B threshold" vs. Lester et al. 1B

---

### Add Content
- [ ] SOTA context: "91.5% vs. 97% SOTA on SST-2"
- [ ] System diagram (Text-Relay vs. Bridge)
- [ ] Algorithm pseudocode for training
- [ ] Notation table
- [ ] Reproducibility checklist

---

### Fix Tables/Figures
- [ ] Consistent notation (dz vs. d_z)
- [ ] Error bars on all tables
- [ ] Figure 1 y-axis start at 0
- [ ] Add latency plots
- [ ] Add training curves

---

## CODE RELEASE ⚠️

**CRITICAL: 30+ reviewers demand this**

- [ ] Training scripts
- [ ] Evaluation scripts
- [ ] Model checkpoints
- [ ] Hyperparameter configs
- [ ] Exact prompts and decoding parameters
- [ ] README with reproduction instructions

**This is non-negotiable for final acceptance.**

---

## PRIORITY ORDER

### Week 1-2: Critical Validity
1. Full test sets + 10 seeds + significance tests
2. Fair baselines (3 new baselines)
3. End-to-end latency measurements

### Week 3-4: Critical Tasks
4. Generation task (XSUM/CNN-DM)
5. Reasoning task (GSM8K)
6. SQuAD Llama→Mistral results
7. Attention analysis

### Week 5-6: Major Strengthening
8. Multi-turn experiment
9. Representation analysis (CKA, probing)
10. Quantization + memory analysis
11. 2-3 new model pairs

### Week 7-8: Comprehensive Ablations
12. LLMLingua + ICAE baselines
13. Multi-seed ablations
14. 70B validation
15. Pareto curves

### Week 9-10: Generalization
16. Agent benchmark
17. Cross-task generalization
18. Error analysis
19. Factorial SST-2 analysis

### Week 11-12: Polish
20. Modern benchmarks
21. Production analysis
22. Writing improvements
23. Code release preparation

---

## ESTIMATED TOTAL EFFORT

**10-14 weeks for comprehensive revision addressing all critical and major issues**

- Critical fixes: 4-6 weeks
- Major strengthening: 3-4 weeks
- Comprehensive additions: 2-3 weeks
- Publication polish: 1 week

---

## CONSENSUS ISSUES (All 50 Reviewers Agree)

1. ✅ **3B threshold is solid** - Best contribution
2. ✅ **Training stability fixes are valuable** - Document honestly
3. ❌ **27× speedup claim is problematic** - Need fair baseline
4. ❌ **200 samples too small** - Need full test sets
5. ❌ **Only classification tasks** - Need generation/reasoning
6. ❌ **Gist Tokens missing** - Essential comparison
7. ❌ **Qwen→Mistral failure** - Limits "interlingua" claim
8. ❌ **No multi-turn demo** - Core motivation not validated

---

## VENUE RECOMMENDATION

**Current consensus**: Weak Reject / Borderline at MLSys

**To achieve acceptance**:
1. Fix all CRITICAL issues (Weeks 1-4)
2. Address most MAJOR issues (Weeks 5-8)
3. Polish writing and release code

**Alternative venues if MLSys doesn't work**:
- ACL/EMNLP (if focus on cross-model NLP transfer)
- NeurIPS/ICLR (if add representation analysis)

---

## KEY NUMBERS TO REMEMBER

- **45+ reviewers**: Need full test sets + statistical rigor
- **35+ reviewers**: Need measured latency + generation tasks
- **30+ reviewers**: Need fair baselines + code release
- **25+ reviewers**: Need multi-turn + quantization
- **20+ reviewers**: Need more model pairs + scaling validation
- **15+ reviewers**: Need attention analysis + compression baselines
