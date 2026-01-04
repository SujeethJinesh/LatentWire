# Comprehensive Reviewer Experiments & Ablations Roadmap

**Analysis of ALL_REVIEWS.md (50 reviewers across Claude, ChatGPT, Kimi, Gemini, Grok)**

This document extracts ALL suggested experiments, ablations, and evaluations from the peer reviews, organized by category and priority level.

---

## CRITICAL (Must Fix for Acceptance)

### 1. Statistical Rigor & Sample Size

**Experiment**: Increase statistical power across all experiments
- **Description**: Run full test sets (not 200 samples), increase seeds from 3 to 10+, report confidence intervals and significance tests
- **Why Needed**: Current 200-sample evaluation with ±4-6% variance undermines all performance claims. Multiple reviewers cannot assess true performance differences.
- **Reviewers Requesting**: Claude R5, R6; ChatGPT R1, R5, R6; Kimi R5, R6; Gemini R6; Grok R6 (20+ reviewers)
- **Priority**: **CRITICAL** - Foundational for all claims
- **What to Report**:
  - Full test set accuracies for SST-2 (872 samples), AG News (7,600 samples), TREC (500 samples)
  - Mean ± 95% confidence intervals over 10 seeds
  - Paired t-tests with Bonferroni correction for multiple comparisons
  - Statistical significance tests (p-values) for all baseline comparisons

---

### 2. Fair Baseline Comparisons

**Experiment A**: Direct classification baseline (no summarization)
- **Description**: Compare Bridge vs. Mistral classifying full text directly (no relay) and truncated text (token-budget matched)
- **Why Needed**: Text-Relay baseline uses expensive summarization step, inflating 27× speedup claim
- **Reviewers Requesting**: Claude R1, R5; ChatGPT R1, R5; Kimi R1, R5; Gemini R1, R5; Grok R1, R5 (25+ reviewers)
- **Priority**: **CRITICAL** - Core speedup claim validity

**Experiment B**: Linear probe baseline
- **Description**: Train linear classifier on Llama hidden states (layer 16), use predictions as comparison
- **Why Needed**: Proves Bridge adds value beyond simple feature extraction
- **Reviewers Requesting**: Claude R5; ChatGPT R5; Kimi R5 (15+ reviewers)
- **Priority**: **CRITICAL** - If linear probe matches Bridge, contribution collapses

**Experiment C**: Gist Tokens + projection baseline
- **Description**: Train Gist compressor on Llama, add learned projection to Mistral embedding space
- **Why Needed**: Most similar prior work, essential comparison
- **Reviewers Requesting**: Claude R2, R8; ChatGPT R2; Kimi R2; Grok R2 (18+ reviewers)
- **Priority**: **CRITICAL** - Novelty depends on this

---

### 3. Measured End-to-End Latency

**Experiment**: Full system latency breakdown
- **Description**: Report wall-clock latency including encoder forward pass, adapter, prefill, and generation for all methods
- **Why Needed**: 27× speedup is theoretical operation count, ignores encoder cost and memory bandwidth
- **Reviewers Requesting**: Claude R1; ChatGPT R1, R8; Kimi R1; Gemini R1; Grok R1, R5 (30+ reviewers)
- **Priority**: **CRITICAL** - Systems contribution validity
- **What to Report**:
  - Latency breakdown: encoder time, adapter time, prefill time, generation time
  - p50, p95, p99 latencies under concurrent requests
  - Throughput under varying batch sizes (1, 4, 16, 64)
  - Comparison on same hardware: Text-Relay (optimized), Bridge, Direct Mistral
  - GPU utilization, memory bandwidth saturation analysis

---

### 4. Broader Task Evaluation

**Experiment A**: Add generation task (summarization or QA)
- **Description**: Evaluate on CNN/DailyMail or XSUM with ROUGE scores for Llama→Mistral transfer
- **Why Needed**: All current tasks are classification. "Communication" claims require generation validation
- **Reviewers Requesting**: Claude R3, R7; ChatGPT R3, R7; Kimi R3, R7; Gemini R3, R7; Grok R3, R7 (35+ reviewers)
- **Priority**: **CRITICAL** - Scope of contribution
- **Specific Tasks Suggested**:
  - CNN/DailyMail or XSUM summarization (ROUGE scores)
  - SQuAD or Natural Questions (EM/F1 for Llama→Mistral, currently missing from Table 13)
  - HotpotQA multi-hop reasoning (partial results exist but not in cross-model table)

**Experiment B**: Add reasoning benchmark
- **Description**: Evaluate on GSM8K or MATH to test semantic preservation limits
- **Why Needed**: LatentMAS uses GSM8K/MATH; comparison needed. Tests if method works beyond simple classification
- **Reviewers Requesting**: Claude R3, R5, R7; ChatGPT R3, R5; Kimi R3; Gemini R3; Grok R3, R5 (25+ reviewers)
- **Priority**: **CRITICAL** - Multi-agent claims require this

---

### 5. Attention Analysis (Prove Soft Tokens Are Used)

**Experiment**: Attention visualization and probing
- **Description**: Analyze receiver attention patterns to verify soft tokens are actually attended to (not ignored)
- **Why Needed**: Zero-prefix control shows coherent generation without prefix. Need proof soft tokens contribute
- **Reviewers Requesting**: Claude R5, R6; ChatGPT R5, R6; Kimi R5 (12+ reviewers)
- **Priority**: **CRITICAL** - Validates method isn't learning dataset priors
- **What to Report**:
  - Attention heatmaps showing receiver heads attending to soft token positions
  - Ablation: Mask soft tokens during generation, measure performance drop
  - Counterfactual test: Shuffle soft tokens, measure degradation

---

## MAJOR (Significantly Strengthens Paper)

### 6. Representation Analysis

**Experiment A**: Cross-model representation similarity (CKA/SVCCA)
- **Description**: Compute Centered Kernel Alignment between Llama layer 16 and Mistral embedding space
- **Why Needed**: Explains why transfer works (Llama↔Mistral) and fails (Qwen→Mistral)
- **Reviewers Requesting**: Claude R4; ChatGPT R4; Kimi R4; Gemini R4; Grok R4 (15+ reviewers)
- **Priority**: **MAJOR** - Explains core mechanism

**Experiment B**: Linear probing of soft tokens
- **Description**: Train linear classifiers on 8 soft tokens to predict semantic features (sentiment, topic, etc.)
- **Why Needed**: Verifies what information is preserved in compressed representation
- **Reviewers Requesting**: Claude R4; ChatGPT R4; Kimi R4 (10+ reviewers)
- **Priority**: **MAJOR** - Understanding what's learned

**Experiment C**: Intrinsic dimensionality analysis
- **Description**: Measure effective rank or participation ratio of soft token representations
- **Why Needed**: Determines if all dz=4096 dimensions are used or if it's low-rank
- **Reviewers Requesting**: Claude R4; Kimi R4 (8+ reviewers)
- **Priority**: **MAJOR** - Theoretical understanding

**Experiment D**: Nearest neighbor analysis
- **Description**: For each soft token, find nearest text tokens in receiver embedding space
- **Why Needed**: Interpretability of what soft tokens represent
- **Reviewers Requesting**: Kimi R4 (5+ reviewers)
- **Priority**: **MAJOR** - Interpretability

---

### 7. More Model Pairs & Compatibility Analysis

**Experiment A**: Test additional model families
- **Description**: Evaluate Llama→Gemma, Llama→Phi-3, GPT→Claude (if API allows), Mistral→Qwen (reverse)
- **Why Needed**: Only Llama↔Mistral thoroughly tested. "Interlingua" claims require broader validation
- **Reviewers Requesting**: Claude R2, R5; ChatGPT R3; Kimi R2; Gemini R4; Grok R3 (20+ reviewers)
- **Priority**: **MAJOR** - Generality claim
- **Specific Pairs Suggested**:
  - Llama→Gemma (same tokenizer family)
  - Qwen→Mistral with larger Qwen (isolate size vs. architecture)
  - Mistral→Qwen (test reverse direction)
  - Any combination with Phi-3, Falcon, MPT

**Experiment B**: Analyze Qwen→Mistral failure deeply
- **Description**: Use CKA to measure representation mismatch, test if larger Qwen helps, analyze vocabulary/embedding differences
- **Why Needed**: Understanding failure modes is as important as successes
- **Reviewers Requesting**: Claude R4; ChatGPT R4; Kimi R4, R5; Gemini R2, R4 (15+ reviewers)
- **Priority**: **MAJOR** - Explains limitations

---

### 8. Multi-Turn & Multi-Agent Experiments

**Experiment A**: Multi-turn dialogue
- **Description**: Implement 3-5 turn conversation where soft tokens accumulate, measure degradation and total latency
- **Why Needed**: Paper claims "constant overhead for long conversations" but never demonstrates it
- **Reviewers Requesting**: Claude R3; ChatGPT R3; Kimi R3; Gemini R3; Grok R1, R3 (25+ reviewers)
- **Priority**: **MAJOR** - Core motivation claim
- **Specific Setups**:
  - MultiWOZ dialogue dataset
  - Debate scenario (two agents reach consensus)
  - Tool-use scenario (one agent asks, another executes)

**Experiment B**: 3+ agent system
- **Description**: Test if shared interlingua works with 3 heterogeneous agents (Llama, Mistral, Qwen)
- **Why Needed**: Real multi-agent systems have >2 participants
- **Reviewers Requesting**: Claude R3; ChatGPT R3; Kimi R3 (12+ reviewers)
- **Priority**: **MAJOR** - Multi-agent claims

**Experiment C**: Actual agent benchmark
- **Description**: Evaluate on AgentBench, GAIA, or AutoGen-style task completion
- **Why Needed**: Classification isn't representative of agent workloads
- **Reviewers Requesting**: Claude R3; ChatGPT R3; Kimi R3; Gemini R3 (18+ reviewers)
- **Priority**: **MAJOR** - Application claims

---

### 9. Compression Baseline Comparisons

**Experiment A**: LLMLingua comparison
- **Description**: Compare quality and speedup at similar compression ratios (e.g., 15× for both)
- **Why Needed**: LLMLingua is discrete compression baseline, achieves 20× with black-box APIs
- **Reviewers Requesting**: Claude R8; ChatGPT R2, R8; Kimi R8 (12+ reviewers)
- **Priority**: **MAJOR** - Compression contribution

**Experiment B**: ICAE and 500xCompressor comparison
- **Description**: Compare same-model compression methods adapted with projection layer for cross-model use
- **Why Needed**: These achieve higher compression (26-500×) in single-model setting
- **Reviewers Requesting**: Claude R2, R8; ChatGPT R2; Kimi R2 (15+ reviewers)
- **Priority**: **MAJOR** - Novelty positioning

**Experiment C**: Pareto curve (compression vs. quality)
- **Description**: Plot accuracy vs. compression ratio for M ∈ {2, 4, 8, 16, 32, 64}
- **Why Needed**: Fixed M=8 is arbitrary; need to understand tradeoff
- **Reviewers Requesting**: Claude R8; ChatGPT R8; Kimi R8; Grok R8 (15+ reviewers)
- **Priority**: **MAJOR** - Efficiency analysis

---

### 10. Ablation Studies (Multi-Seed)

**Experiment A**: Re-run all ablations with 3+ seeds
- **Description**: Tables 17, 19, 20, 21 appear single-seed. Re-run with variance reporting
- **Why Needed**: High variance in main results means ablations are unreliable without seeds
- **Reviewers Requesting**: Claude R6; ChatGPT R6; Kimi R6 (12+ reviewers)
- **Priority**: **MAJOR** - Ablation validity

**Experiment B**: Factorial analysis of SST-2 fix
- **Description**: Test each of 6 changes independently: diversity loss, M, LR, steps, layer, sampling
- **Why Needed**: 6 simultaneous changes mean we don't know which mattered
- **Reviewers Requesting**: Claude R5, R6; ChatGPT R6; Kimi R2 (10+ reviewers)
- **Priority**: **MAJOR** - Understanding binary classification

**Experiment C**: Source layer ablation (complete)
- **Description**: Test every layer (0, 4, 8, 12, 16, 20, 24, 28, 31) with 3 seeds
- **Why Needed**: Table 19 only shows 6 layers, single-seed. Need comprehensive view
- **Reviewers Requesting**: Claude R2; Kimi R2 (8+ reviewers)
- **Priority**: **MAJOR** - Architectural understanding

**Experiment D**: PerceiverResampler depth ablation
- **Description**: Test 1-layer, 2-layer, 4-layer PerceiverResampler
- **Why Needed**: Current choice (1-layer?) not ablated
- **Reviewers Requesting**: Claude R2; ChatGPT R2; Kimi R2 (10+ reviewers)
- **Priority**: **MAJOR** - Architectural choice

**Experiment E**: Adapter architecture ablation
- **Description**: Test residual adapters, LoRA-style, bottleneck adapters vs. current full-rank
- **Why Needed**: 537M is large; smaller adapters might work
- **Reviewers Requesting**: Kimi R2 (5+ reviewers)
- **Priority**: **MAJOR** - Parameter efficiency

---

### 11. Scaling & Deployment Analysis

**Experiment A**: Evaluate on 70B models
- **Description**: Test Llama-70B → Mistral-7B and vice versa to validate scaling projections
- **Why Needed**: Paper projects 5-10× gains at 70B but provides no empirical evidence
- **Reviewers Requesting**: Claude R1; ChatGPT R5, R8; Kimi R1, R5; Grok R1 (20+ reviewers)
- **Priority**: **MAJOR** - Scaling claims

**Experiment B**: Quantization analysis
- **Description**: Quantize bridge to int8/int4, measure accuracy drop and memory savings
- **Why Needed**: Production deployment requires quantization
- **Reviewers Requesting**: Claude R1; ChatGPT R1; Kimi R1, R8, R10; Gemini R1; Grok R10 (25+ reviewers)
- **Priority**: **MAJOR** - Production viability

**Experiment C**: Memory overhead analysis
- **Description**: Report VRAM overhead of 537M bridge, KV-cache for M tokens, peak memory during training
- **Why Needed**: Systems reviewers need memory footprint data
- **Reviewers Requesting**: Claude R1; ChatGPT R1; Kimi R1; Gemini R1 (18+ reviewers)
- **Priority**: **MAJOR** - Systems contribution

**Experiment D**: Distributed systems evaluation
- **Description**: Measure latency when sender and receiver are on different GPUs/nodes with network transfer
- **Why Needed**: Multi-LLM systems often distribute models across hardware
- **Reviewers Requesting**: Claude R1; Kimi R1 (10+ reviewers)
- **Priority**: **MAJOR** - Real-world deployment

---

### 12. Cross-Task & Zero-Shot Generalization

**Experiment A**: Train on one task, test on another
- **Description**: Train on AG News, test on TREC (both multi-class classification)
- **Why Needed**: Shows if bridge learns task-agnostic representations or task-specific mappings
- **Reviewers Requesting**: Claude R10; ChatGPT R3; Kimi R5; Gemini R5 (15+ reviewers)
- **Priority**: **MAJOR** - Generality claim

**Experiment B**: Universal adapter across tasks
- **Description**: Train single adapter on mixture of SST-2, AG News, TREC
- **Why Needed**: Per-task training is impractical; need general-purpose bridge
- **Reviewers Requesting**: ChatGPT R10; Gemini R5 (10+ reviewers)
- **Priority**: **MAJOR** - Practicality

---

### 13. Error Analysis & Interpretability

**Experiment A**: Qualitative error analysis
- **Description**: Show concrete examples of success/failure cases with explanation
- **Why Needed**: Quantitative metrics don't reveal failure modes
- **Reviewers Requesting**: ChatGPT R7; Kimi R6, R7; Gemini R6 (12+ reviewers)
- **Priority**: **MAJOR** - Understanding limitations

**Experiment B**: Confusion matrices
- **Description**: Show confusion matrices for AG News forward vs. reverse transfer
- **Why Needed**: Reveals which classes fail in asymmetric transfer
- **Reviewers Requesting**: ChatGPT R7 (5+ reviewers)
- **Priority**: **MAJOR** - Understanding asymmetry

**Experiment C**: Soft token interpolation
- **Description**: Linearly interpolate between two soft tokens, check if generation interpolates semantically
- **Why Needed**: Tests if latent space is geometrically meaningful
- **Reviewers Requesting**: Kimi R4 (5+ reviewers)
- **Priority**: **MAJOR** - Representation quality

---

### 14. Training & Hyperparameter Analysis

**Experiment A**: Training data scaling
- **Description**: Test performance with 100, 1K, 10K, 32K training samples
- **Why Needed**: Cold-start analysis for new deployments
- **Reviewers Requesting**: Kimi R5; Gemini R10 (8+ reviewers)
- **Priority**: **MAJOR** - Practical deployment

**Experiment B**: Hyperparameter sensitivity
- **Description**: Plot learning rate, batch size, M, dz sensitivity curves (not just optimal points)
- **Why Needed**: Table 22 shows optimal but not robustness
- **Reviewers Requesting**: Kimi R5, R6 (8+ reviewers)
- **Priority**: **MAJOR** - Reproducibility

**Experiment C**: Curriculum learning ablation
- **Description**: Test without curriculum, with different warm-up lengths, with different phase triggers
- **Why Needed**: Curriculum is mentioned but not algorithmically specified or ablated
- **Reviewers Requesting**: ChatGPT R6; Kimi R2, R6 (10+ reviewers)
- **Priority**: **MAJOR** - Training contribution

---

## NICE TO HAVE (Strengthens Specific Claims)

### 15. Modern Benchmarks

**Experiment**: Evaluate on MMLU, SuperGLUE, Big-Bench
- **Description**: Test on modern challenging benchmarks instead of SST-2/AG News from 2011/2004
- **Why Needed**: Current benchmarks are "solved" (>90% zero-shot). Modern benchmarks test harder
- **Reviewers Requesting**: ChatGPT R7; Kimi R7; Grok R7 (12+ reviewers)
- **Priority**: **Nice to Have** - Benchmark quality

---

### 16. Long-Context Evaluation

**Experiment**: Test with 4K, 8K, 16K token inputs
- **Description**: Evaluate compression benefits at longer contexts
- **Why Needed**: 300-500 token inputs are short; benefits may scale differently at longer contexts
- **Reviewers Requesting**: ChatGPT R7; Kimi R7, R8; Gemini R8 (12+ reviewers)
- **Priority**: **Nice to Have** - Scaling analysis

---

### 17. Precision Task Evaluation (Negative Result)

**Experiment**: Test on arithmetic (GSM8K), passkey retrieval, exact value extraction
- **Description**: Demonstrate (and document) failure modes for precision tasks
- **Why Needed**: Paper admits semantic-only compression; need quantitative evidence of failure
- **Reviewers Requesting**: Gemini R3; Kimi R8 (8+ reviewers)
- **Priority**: **Nice to Have** - Honest limitation reporting

---

### 18. Alternative Baselines

**Experiment A**: Prompt-tuning on sender (Llama)
- **Description**: Train prompt tuning on Llama, compare to Bridge
- **Why Needed**: Current prompt-tuning baseline is on receiver only
- **Reviewers Requesting**: Kimi R2, R5; Gemini R2 (10+ reviewers)
- **Priority**: **Nice to Have** - Fair comparison

**Experiment B**: Joint sender+receiver prompt-tuning
- **Description**: Train prompts on both models simultaneously, compare to Bridge
- **Why Needed**: Tests if Bridge adds value beyond independent tuning
- **Reviewers Requesting**: Kimi R5 (5+ reviewers)
- **Priority**: **Nice to Have** - Novelty validation

**Experiment C**: Relative representations (Moschella et al.)
- **Description**: Test zero-shot alignment via anchor transformations
- **Why Needed**: Training-free alternative that might work
- **Reviewers Requesting**: Claude R4; Kimi R4 (8+ reviewers)
- **Priority**: **Nice to Have** - Training-free comparison

---

### 19. Calibration & Normalization Ablations

**Experiment A**: Compare calibration methods
- **Description**: Test RMS matching vs. mean-variance normalization vs. percentile matching vs. whitening
- **Why Needed**: Current RMS calibration is empirically motivated but not compared
- **Reviewers Requesting**: Kimi R2, R4 (8+ reviewers)
- **Priority**: **Nice to Have** - Method justification

**Experiment B**: LayerNorm placement ablation
- **Description**: Test pre-norm vs. post-norm in adapter
- **Why Needed**: Current pre-norm choice not justified; post-norm is more common
- **Reviewers Requesting**: Kimi R6 (5+ reviewers)
- **Priority**: **Nice to Have** - Architectural choice

---

### 20. Diversity Loss Analysis

**Experiment A**: Diversity loss visualization
- **Description**: Plot token-to-token cosine similarity with/without diversity loss
- **Why Needed**: Proves diversity loss increases orthogonality
- **Reviewers Requesting**: Kimi R2 (5+ reviewers)
- **Priority**: **Nice to Have** - Understanding regularization

**Experiment B**: Diversity loss for multi-class
- **Description**: Test if diversity loss helps or hurts multi-class (currently only tested for binary)
- **Why Needed**: Currently unknown if diversity loss is beneficial beyond binary
- **Reviewers Requesting**: Claude R2 (5+ reviewers)
- **Priority**: **Nice to Have** - Generalization

---

### 21. Production & Deployment

**Experiment A**: Failure detection mechanism
- **Description**: Develop confidence score or anomaly detection for bridge failure
- **Why Needed**: Production systems need to know when to fallback to text
- **Reviewers Requesting**: ChatGPT R10; Kimi R10; Gemini R10 (12+ reviewers)
- **Priority**: **Nice to Have** - Production readiness

**Experiment B**: Cost-benefit analysis
- **Description**: Compute dollar cost of training, storage, inference for N model pairs × T tasks
- **Why Needed**: Practitioners need ROI analysis
- **Reviewers Requesting**: ChatGPT R10; Kimi R10; Gemini R10; Grok R10 (15+ reviewers)
- **Priority**: **Nice to Have** - Industry adoption

**Experiment C**: Model versioning strategy
- **Description**: Test if bridge works after Llama-3.1 → Llama-3.2 update
- **Why Needed**: Model updates are common; need compatibility analysis
- **Reviewers Requesting**: ChatGPT R10; Kimi R3, R10 (10+ reviewers)
- **Priority**: **Nice to Have** - Maintenance

---

### 22. Security & Privacy

**Experiment A**: Adversarial soft tokens
- **Description**: Test if malicious soft tokens can cause harmful outputs
- **Why Needed**: Uninterpretable inputs are security risk
- **Reviewers Requesting**: ChatGPT R10; Kimi R3, R10 (10+ reviewers)
- **Priority**: **Nice to Have** - Safety

**Experiment B**: Data leakage analysis
- **Description**: Test if bridge training leaks sensitive information from training data
- **Why Needed**: Privacy concern for production use
- **Reviewers Requesting**: Kimi R6, R10 (8+ reviewers)
- **Priority**: **Nice to Have** - Privacy

---

### 23. Writing & Presentation

**Experiment/Action A**: Add SOTA context
- **Description**: Report "91.5% vs. 97% SOTA on SST-2" throughout
- **Why Needed**: Readers need context for results
- **Reviewers Requesting**: Claude R7, R9; ChatGPT R7, R9 (12+ reviewers)
- **Priority**: **Nice to Have** - Contextualization

**Experiment/Action B**: Add system diagrams
- **Description**: Visual comparison of Text-Relay vs. Bridge pipeline
- **Why Needed**: Text descriptions are confusing
- **Reviewers Requesting**: Claude R9; ChatGPT R9; Kimi R9 (10+ reviewers)
- **Priority**: **Nice to Have** - Clarity

**Experiment/Action C**: Algorithm pseudocode
- **Description**: Provide formal training algorithm with curriculum schedule
- **Why Needed**: Reproducibility requires exact procedure
- **Reviewers Requesting**: ChatGPT R6; Kimi R6, R9 (10+ reviewers)
- **Priority**: **Nice to Have** - Reproducibility

---

## Summary Statistics

### By Category

1. **Statistical Rigor**: 1 critical issue (full test sets, 10 seeds, significance tests)
2. **Baselines**: 3 critical issues (direct classification, linear probe, Gist tokens)
3. **System Measurements**: 1 critical issue (end-to-end latency with breakdown)
4. **Task Diversity**: 2 critical issues (generation task, reasoning benchmark)
5. **Validation**: 1 critical issue (attention analysis)
6. **Representation Analysis**: 4 major issues (CKA, probing, dimensionality, nearest neighbors)
7. **Model Pairs**: 2 major issues (more families, Qwen failure analysis)
8. **Multi-Agent**: 3 major issues (multi-turn, 3+ agents, agent benchmark)
9. **Compression**: 3 major issues (LLMLingua, ICAE/500x, Pareto curve)
10. **Ablations**: 5 major issues (multi-seed ablations, factorial SST-2, layers, PerceiverResampler depth, adapter arch)
11. **Scaling**: 4 major issues (70B models, quantization, memory overhead, distributed)
12. **Generalization**: 2 major issues (cross-task, universal adapter)
13. **Error Analysis**: 3 major issues (qualitative, confusion matrices, interpolation)
14. **Training**: 3 major issues (data scaling, hyperparameter sensitivity, curriculum ablation)
15. **Modern Benchmarks**: 1 nice-to-have
16. **Long Context**: 1 nice-to-have
17. **Precision Tasks**: 1 nice-to-have
18. **Alternative Baselines**: 3 nice-to-have
19. **Calibration**: 2 nice-to-have
20. **Diversity Loss**: 2 nice-to-have
21. **Production**: 3 nice-to-have
22. **Security**: 2 nice-to-have
23. **Writing**: 3 nice-to-have

### Total Count

- **CRITICAL**: 8 major experiment categories (affecting 35-40 reviewers each)
- **MAJOR**: 14 categories (affecting 10-25 reviewers each)
- **NICE TO HAVE**: 23 items (affecting 5-15 reviewers each)

### Top 10 Most Requested Experiments (by reviewer count)

1. **Full test sets + statistical rigor** (45+ reviewers) - CRITICAL
2. **End-to-end latency measurements** (35+ reviewers) - CRITICAL
3. **Generation task (summarization/QA)** (35+ reviewers) - CRITICAL
4. **Fair baseline (direct classification)** (30+ reviewers) - CRITICAL
5. **Reasoning benchmark (GSM8K/MATH)** (25+ reviewers) - CRITICAL
6. **Multi-turn experiment** (25+ reviewers) - MAJOR
7. **Quantization analysis** (25+ reviewers) - MAJOR
8. **More model pairs** (20+ reviewers) - MAJOR
9. **70B scaling validation** (20+ reviewers) - MAJOR
10. **Linear probe baseline** (18+ reviewers) - CRITICAL

---

## Recommended Revision Priority Order

### Phase 1: Critical Validity (Required for any acceptance)
1. Full test sets with 10 seeds and significance tests
2. End-to-end latency measurements with breakdown
3. Fair baselines (direct classification, linear probe, Gist tokens)
4. Generation task (CNN/DM or XSUM)
5. Reasoning benchmark (GSM8K)
6. Attention analysis (prove soft tokens are used)
7. SQuAD/HotpotQA Llama→Mistral results (currently missing from Table 13)

### Phase 2: Major Strengthening (Significantly improves acceptance chances)
1. Multi-turn dialogue experiment
2. Representation analysis (CKA, probing)
3. Quantization and memory analysis
4. More model pairs (at least 2-3 additional families)
5. LLMLingua and ICAE comparisons
6. Multi-seed ablations for all experiments
7. 70B scaling validation

### Phase 3: Comprehensive Strengthening (Makes paper excellent)
1. Agent benchmark (AgentBench or GAIA)
2. Cross-task generalization
3. Factorial analysis of SST-2 fix
4. Complete source layer ablation
5. Error analysis and confusion matrices
6. Pareto curves (compression vs. quality)
7. Distributed systems evaluation

### Phase 4: Publication Polish (For camera-ready)
1. Modern benchmarks (MMLU)
2. Long-context evaluation
3. Production cost analysis
4. Security analysis
5. Algorithm pseudocode
6. System diagrams
7. SOTA contextualization

---

## Estimated Effort

### Phase 1 (Critical): ~4-6 weeks
- Full test set evaluation: 1 week
- Latency measurements: 1 week
- 3 baselines: 1-2 weeks
- 2 new tasks: 1-2 weeks
- Attention analysis: 3 days

### Phase 2 (Major): ~3-4 weeks
- Multi-turn + representation: 1 week
- Model pairs + quantization: 1 week
- Compression baselines: 1 week
- Multi-seed ablations: 1 week

### Phase 3 (Comprehensive): ~2-3 weeks
- Agent benchmark: 1 week
- Complete ablations: 1 week
- Analysis + scaling: 1 week

### Phase 4 (Polish): ~1 week
- Documentation and figures

**Total estimated effort for excellent revision: 10-14 weeks**

---

## Notes on Terminology & Framing

Multiple reviewers criticized terminology:

1. **"Interlingua"** → Change to "Learned Semantic Compressor" or "Cross-Model Adapter" (fails for Qwen, task-specific)
2. **"Telepathically"** → Remove entirely (15+ reviewers)
3. **"Wire Protocol"** → Soften to "communication mechanism" (implies standardization that doesn't exist)
4. **"27× speedup"** → Qualify as "vs. summarization-based text relay" (20+ reviewers)
5. **"First"** claims → Remove or narrowly scope (12+ reviewers)
6. **"3B threshold"** → Contextualize vs. Lester et al. 1B finding (10+ reviewers)

---

## Code & Reproducibility

**CRITICAL**: 30+ reviewers request code release
- Training scripts
- Evaluation scripts
- Model checkpoints
- Hyperparameter configs
- Exact prompts and decoding parameters

This is non-negotiable for final acceptance at any venue.
