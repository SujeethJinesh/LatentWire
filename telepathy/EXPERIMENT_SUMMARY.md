# EXPERIMENT SUMMARY

**Project**: Latent Telepathy - Cross-Model Communication via Soft Tokens
**Models**: Llama 3.1 8B → Mistral 7B (primary focus)
**Date**: January 2025

---

## 1. EXPERIMENTS OVERVIEW

### Phase 1: Statistical Rigor & Core Evaluation
**Purpose**: Address reviewer concerns about statistical validity and test set completeness
**Experiments**:
- **Full test sets** on SST-2 (872 samples), AG News (7,600 samples), TREC (500 samples)
- **Multi-seed runs** (seeds: 42, 123, 456) for variance estimation
- **Bootstrap confidence intervals** (95% CI with BCa method, 10,000 resamples)
- **McNemar's test** for paired comparison significance
- **Cohen's d effect size** reporting

**Rationale**: Reviewers criticized small sample sizes (100-200) and lack of statistical testing. Full test sets with proper confidence intervals provide definitive evidence.

### Phase 2: Linear Probe Baseline
**Purpose**: Critical baseline comparing direct task-solving vs cross-model transfer
**Experiments**:
- **Sklearn LogisticRegression** on frozen Llama embeddings (layer 16, 24, 31)
- **Pooling strategies**: mean, last-token, first-token
- **Cross-validation**: 5-fold stratified with L2 regularization sweep
- **Direct comparison** with Bridge method on identical data splits

**Rationale**: Tests whether sender model already encodes task solution, establishing lower bound for cross-model transfer necessity.

### Phase 3: Fair Baseline Comparisons
**Purpose**: Comprehensive baseline evaluation under identical conditions
**Experiments**:
- **Text Relay**: Full text generation → parsing (upper bound)
- **Token Budget**: Text truncated to M tokens (compression fairness)
- **Random/Noise**: Random soft tokens (sanity check, expect 50% SST-2)
- **LLMLingua-2**: State-of-the-art prompt compression at M∈{32,64,128}
- **Zero-shot**: Direct prompting without examples

**Rationale**: Establishes performance bounds and validates Bridge gains aren't artifacts.

### Phase 4: Efficiency Measurements
**Purpose**: Quantify computational savings for real-world deployment
**Experiments**:
- **End-to-end latency**: Bridge vs text generation (ms/sample)
- **Throughput**: Samples/second at batch sizes 1, 4, 16, 32
- **Memory footprint**: Peak GPU memory usage
- **Compression metrics**: Bits/token with int8/int4 quantization

**Rationale**: Reviewers want concrete efficiency numbers beyond theoretical claims.

---

## 2. DATA COLLECTED

### Core Metrics
| Metric | Description | Collection Method |
|--------|-------------|-------------------|
| **Accuracy** | Classification correctness | Exact match on label |
| **F1 Score** | Precision-recall balance | Macro/micro averaging |
| **Latency** | ms per sample | torch.cuda.Event timing |
| **Throughput** | Samples/second | Batch timing with warmup |
| **Memory** | Peak GPU MB | torch.cuda.max_memory_allocated |

### Statistical Measures
| Measure | Implementation | Purpose |
|---------|----------------|---------|
| **95% CI** | Bootstrap BCa (10k samples) | Uncertainty quantification |
| **p-value** | McNemar's test | Significance testing |
| **Effect size** | Cohen's d (pooled) | Practical significance |
| **Cross-validation** | 5-fold stratified | Generalization estimate |

### Expected Outputs
- **JSON results**: `{dataset}/results_{seed}.json` with all metrics
- **Statistical summary**: `summary_table.csv` with CIs and p-values
- **Confusion matrices**: Per-dataset error analysis
- **Sample outputs**: 50 examples with predictions for qualitative analysis
- **Timing logs**: Detailed latency breakdowns

---

## 3. TECHNICAL DETAILS

### Models
**Primary Configuration** (Memory-constrained):
- **Sender**: Llama 3.1 8B Instruct (bfloat16)
- **Receiver**: Mistral 7B v0.3 Instruct (bfloat16)
- **Bridge**: PerceiverResampler + StatisticalNormalizer
- **Soft tokens**: K=8 (optimal from ablations)
- **Source layer**: 31 (near-final representations)

**Ablations** (Phase 6):
- Small: Llama 1B → Qwen 1.5B
- Medium: Llama 8B → Mistral 7B
- Large: Single-batch configuration

### Datasets

| Dataset | Task | Classes | Train | Test | Metric |
|---------|------|---------|-------|------|--------|
| **SST-2** | Sentiment | 2 (pos/neg) | 67,349 | 872 | Accuracy |
| **AG News** | Topic | 4 (world/sports/business/sci) | 120,000 | 7,600 | Accuracy |
| **TREC** | Question Type | 6 (ABBR/ENTY/DESC/HUM/LOC/NUM) | 5,452 | 500 | Accuracy |
| **XSUM** | Summarization | N/A | 204,045 | 11,334 | ROUGE-L |

### Baselines Compared

| Baseline | Description | Expected Performance |
|----------|-------------|---------------------|
| **Linear Probe** | LogisticRegression on Llama embeddings | 70-85% (task-dependent) |
| **Text Relay** | Full generation + parsing | 85-95% (upper bound) |
| **Token Budget** | Truncated text (M tokens) | 60-75% (degraded) |
| **LLMLingua-2** | SOTA compression | 70-80% at 4× compression |
| **Random** | Uniform guessing | 50%/25%/16.7% (2/4/6 classes) |
| **Noise** | Random soft tokens | ~Random |

---

## 4. EXPECTED RESULTS

### Primary Hypotheses
1. **Bridge > Linear Probe**: Cross-model transfer adds value beyond sender's encoding
2. **Bridge ≈ 0.8×Text Relay**: Reasonable quality/efficiency tradeoff
3. **Bridge > Token Budget**: Learned compression beats naive truncation
4. **Bridge competitive with LLMLingua**: Different approach, similar quality

### Performance Targets

| Dataset | Random | Linear Probe | Bridge (Target) | Text Relay |
|---------|--------|--------------|-----------------|------------|
| **SST-2** | 50% | 75-80% | **85-90%** | 92-95% |
| **AG News** | 25% | 65-70% | **70-75%** | 85-90% |
| **TREC** | 16.7% | 55-60% | **60-65%** | 75-80% |

### Statistical Significance
- **p < 0.05**: Bridge vs Linear Probe (McNemar's test)
- **Cohen's d > 0.5**: Medium effect size vs baselines
- **95% CI width < 5%**: Sufficient precision with 3 seeds

### Computational Efficiency
- **Latency reduction**: 40-50% vs text generation
- **Throughput gain**: 2-3× at batch=16
- **Memory stable**: <2× despite dual models
- **Compression**: 4-6× with int8 quantization

### Risk Factors
- **Posterior collapse**: Receiver ignoring soft tokens (Phase 15 fix via contrastive loss)
- **Distribution shift**: Test performance drop (mitigated by full test sets)
- **Memory constraints**: Batch size limited to 2-4 (accepted limitation)
- **XSUM length**: Generation quality on long outputs (backup: use shorter summaries)

---

**Document Status**: Comprehensive experiment plan for paper revision
**Last Updated**: January 2025
**Next Steps**: Execute Phase 1 → gate on results → continue if promising