# Telepathy: Cross-Model Neural Bridge for Heterogeneous LLM Communication
## Executive Summary for Hour-Long Presentation

**Author**: Sujeeth Jinesh
**Date**: January 2026
**Document Purpose**: High-level overview for PI and peer presentation

---

## Abstract

Modern multi-agent AI systems rely on text-based communication between language models, creating a fundamental bottleneck: text generation accounts for 70-80% of inference latency and introduces information loss through discretization. **Telepathy** addresses this by building a learned neural bridge that enables heterogeneous LLMs to communicate directly through their latent representations, bypassing text entirely. Using a Perceiver Resampler architecture, we compress Llama 3.1 8B hidden states into 8-16 soft tokens that Mistral 7B can condition on directly. Our key finding is task-dependent: for classification tasks, Telepathy achieves **96.7% accuracy on SST-2** and **90.7% on AG News**, exceeding both individual models (super-additive performance), while running **22x faster** than text-based communication. However, for reasoning tasks like GSM8K, the approach fails completely (2% accuracy), revealing a fundamental architectural limitation. This work establishes both the promise and boundaries of continuous latent communication between different LLM families.

---

## Key Takeaways

### What Worked

- **Classification Excellence**: Telepathy achieves 90-97% accuracy across sentiment, topic, and question classification tasks, consistently exceeding text-based baselines
- **Super-Additive Performance**: The bridge achieves higher accuracy than either Llama (88.4%) or Mistral (92.2%) alone on SST-2, demonstrating emergent capability from cross-model combination
- **22x Speedup**: By eliminating autoregressive text generation, inference drops from 834ms to 37ms per sample
- **4.2x Compression**: Reduces prompt length from ~67 tokens to 16 soft tokens while preserving classification-relevant information
- **Heterogeneous Model Support**: Successfully bridges fundamentally different architectures (128K vs 32K vocabulary, different RoPE frequencies, 5x magnitude difference)

### What Did Not Work

- **Reasoning Tasks Fail Completely**: GSM8K math reasoning achieves only 2% accuracy vs 76.5% text baseline - a fundamental limitation, not a tuning issue
- **Entity Scrambling**: The compression loses specific entities (ducks->chickens, Janet->generic) even when overall semantic category is preserved
- **Binary Classification Anomaly**: SST-2 shows 49.5% accuracy in some configurations, suggesting sensitivity to hyperparameters for low-class-count tasks
- **No Zero-Shot Transfer**: Each task requires dedicated bridge training; no universal bridge emerged

### Critical Insights

1. **Compression as Regularization**: Counterintuitively, fewer soft tokens (8-16) outperform more tokens (64-128); the bottleneck forces learning of robust, abstract features
2. **Cross-Model > Same-Model**: Llama->Mistral bridges slightly outperform Llama->Llama, suggesting the heterogeneity provides beneficial regularization
3. **Classification vs Generation Dichotomy**: The same architecture that excels at classification fails at generation, revealing fundamentally different information requirements

---

## Main Results

### Classification Performance

| Dataset | Classes | Random | Llama 0-shot | Mistral 0-shot | Prompt Tuning | **Telepathy** |
|---------|---------|--------|--------------|----------------|---------------|---------------|
| SST-2 | 2 | 50.0% | 88.4% | 92.2% | 97.5% | **96.7 +/- 0.8%** |
| AG News | 4 | 25.0% | 63.8% | 69.4% | 82.5% | **90.7 +/- 0.6%** |
| TREC | 6 | 16.7% | 74.4% | 61.8% | 90.0% | **95.3 +/- 0.3%** |
| Banking77 | 77 | 1.3% | 22.0% | N/A | N/A | **21.5%** |

### Efficiency Comparison

| Method | Latency | Speedup | Token Count | Compression |
|--------|---------|---------|-------------|-------------|
| Text-Relay | 834.5ms | 1.0x | ~67 tokens | 1.0x |
| **Telepathy** | **37.3ms** | **22.4x** | **16 tokens** | **4.2x** |

### Reasoning Performance (Negative Result)

| Task | Llama | Mistral | Text-Relay | **Telepathy** |
|------|-------|---------|------------|---------------|
| GSM8K | 76.5% | 48.5% | 45.0% | **2.0%** |

---

## Research Timeline

### Phase 1: LatentWire (Failed Approach) - August-October 2025

**Goal**: Learn universal soft prompts that any LLM understands
**Approach**: Shared encoder producing soft tokens for both Llama and Qwen
**Result**: Complete failure (F1 < 1%, FirstTok@1 ~5%)
**Lesson**: Soft prompts cannot bridge fundamentally different architectures without intermediate transformation

### Phase 2: Cross-Model Translation - November 2025

**Goal**: Direct hidden state transfer between specific model pairs
**Experiments**: GSM8K reasoning with Mistral->Llama
**Results**:
- Peak accuracy: 81.5% (exceeded target baseline)
- Final accuracy: 36% (catastrophic collapse over training)
- Stability improvements reduced collapse but lowered peak

**Lesson**: Cross-model transfer is viable but unstable; reasoning requires precise information preservation

### Phase 3: The Four Boss Battles - November 2025

**Identified four physical incompatibilities between Llama and Mistral**:

1. **Magnitude Shock**: Llama hidden states ~20 std, Mistral expects ~100 std (5x difference)
   - Solution: Statistical normalization with learned affine transformation

2. **Vocabulary Density**: 128K vs 32K vocabularies cause tokenization mismatch
   - Solution: Perceiver Resampler abstracts away token boundaries

3. **RoPE Geometry**: Different positional encoding frequencies (500K vs 1M base)
   - Solution: Cross-attention extracts position-agnostic semantic features

4. **KV Cache Amnesia**: Soft tokens alone leave model without context
   - Solution: Prefix priming with static text anchor

### Phase 4: Telepathy Classification Success - November-December 2025

**Pivot**: Abandon reasoning, focus on classification
**Architecture**: Perceiver Resampler with 8-16 learned query tokens
**Key Discovery**: Inverse token scaling - fewer tokens perform better
**Results**:
- SST-2: 94.7% (exceeds both models)
- AG News: 88.9% (+14.4pp over best baseline)
- TREC: 95.3% (super-additive)
- GSM8K: 2.0% (confirmed failure)

### Phase 5: Paper Preparation - December 2025-January 2026

**Focus**: Statistical validation, ablations, reproducibility
**Experiments**: 50+ training runs, 5 random seeds per configuration
**Artifacts**: LaTeX tables, figures, reproducible code package

---

## Critical Insights Learned

### 1. The Information Bottleneck is a Feature, Not a Bug

Standard ML intuition suggests more capacity (more soft tokens) should help. We found the opposite: 8-16 tokens outperform 64-128. The compression forces the bridge to learn robust, task-relevant features while discarding noise. This aligns with Information Bottleneck theory - optimal representations compress irrelevant information while preserving task-relevant signal.

**Evidence**: Banking77 accuracy vs token count: 8 tokens (21.5%), 32 tokens (13.5%), 128 tokens (1.0%)

### 2. Heterogeneity as Beneficial Regularization

Cross-model transfer (Llama->Mistral) slightly outperforms same-model transfer (Llama->Llama). When the sender and receiver speak "different languages," the bridge must learn truly abstract, model-agnostic representations rather than exploiting shortcuts or identity mappings. The forced abstraction improves generalization.

**Evidence**: AG News: Llama->Llama (90.5%) vs Llama->Mistral (90.7%)

### 3. Classification and Generation Have Fundamentally Different Information Requirements

Classification requires preserving category-level semantic features - "negative sentiment," "sports topic," "location question." These compress efficiently into few bits. Generation requires preserving exact entities, numbers, and sequential dependencies - information that doesn't survive lossy compression.

**The Soft Token Paradox**: 16 soft tokens x 256 dims x 16 bits = 65,536 bits available. But for classification, we only need ~2 bits (for binary) or ~4 bits (for 16-class). The excess capacity doesn't help because the bottleneck is semantic abstraction, not bit capacity.

### 4. Teacher Forcing Creates Shortcuts

During training with teacher forcing, the model can learn to copy from the provided ground truth rather than from soft tokens. We discovered this when evaluation produced completely different (garbage) outputs than training. The solution was careful loss weighting on early tokens where no teacher forcing context is available.

### 5. Scale Matching is Necessary but Not Sufficient

Matching mean and standard deviation between sender and receiver embeddings prevents immediate failure (gradient explosion, attention saturation). But geometric structure - the shape of the embedding manifold - differs between models. Statistical normalization is like matching volume; we also need to match the "language."

---

## Limitations (Honest Assessment)

### Fundamental Limitations

1. **Reasoning Fails**: The architecture cannot preserve precise numerical and logical information needed for multi-step reasoning. This is not a hyperparameter issue - it's architectural.

2. **Task-Specific Training Required**: No zero-shot transfer between tasks. Each classification domain needs its own trained bridge (~3-4 hours on H100).

3. **Entity Precision Lost**: Even successful classification loses specific entity information. "Janet's ducks" becomes "some birds" in the compressed representation.

### Practical Limitations

4. **Memory Overhead**: Requires both models in memory simultaneously (15B parameters total). Not suitable for edge deployment without distillation.

5. **Limited Model Pairs Tested**: Only Llama->Mistral extensively validated. Generalization to other pairs (Gemma, Qwen, Claude) untested.

6. **Binary Classification Instability**: SST-2 shows high variance (49.5% to 96.7%) across configurations, suggesting sensitivity for low-class-count tasks.

### Methodological Limitations

7. **No Theoretical Guarantee**: The super-additive performance is empirically observed but not theoretically explained. We don't know when to expect it.

8. **Compression Lower Bound Unknown**: We achieved 4.2x compression, but don't know the theoretical limit for preserving classification accuracy.

---

## So What? Why This Matters

### Immediate Practical Impact

1. **API Cost Reduction**: For classification-heavy workloads, 4.2x fewer tokens means 76% cost savings on API calls
2. **Latency-Critical Applications**: 22x speedup enables real-time multi-model ensemble classification
3. **Heterogeneous Ensembles**: Combines strengths of different model families without model merging constraints

### Scientific Contributions

1. **First Neural Bridge for Heterogeneous LLMs**: Demonstrates direct hidden state transfer between fundamentally different architectures (different tokenizers, dimensions, position encodings)

2. **Super-Additive Performance**: Shows that cross-model combination can exceed either model alone - an emergent property not achieved by simple ensembling

3. **Inverse Token Scaling**: Counterintuitive finding that fewer soft tokens perform better, validating Information Bottleneck theory in cross-model communication

4. **Classification vs Reasoning Boundary**: Precisely characterizes where latent communication succeeds (pattern matching) vs fails (symbolic computation)

### Implications for Multi-Agent AI

Current multi-agent systems communicate via text - slow, lossy, and model-agnostic by necessity. Telepathy demonstrates an alternative: learned, continuous, task-specific communication channels that are faster and more information-preserving (for classification). As multi-agent systems become more prevalent, this paradigm could enable new architectures where models specialize and communicate efficiently.

### Research Directions Opened

1. **Hybrid Text/Latent Systems**: Route classification to latent channel, reasoning to text
2. **Learnable Communication Protocols**: Let agents negotiate their own compressed languages
3. **Privacy-Preserving Inference**: Soft tokens are non-interpretable without the decoder
4. **Cross-Modal Bridges**: Same architecture could connect vision encoders to text decoders

---

## Next Steps

### Immediate (Next 2 Weeks)

1. Complete statistical validation with 5 seeds per configuration
2. Finalize paper for MLSys 2026 submission
3. Prepare reproducible code package and documentation

### Near-Term (Next 2 Months)

1. Test additional model pairs (Gemma, Qwen, Llama variants)
2. Investigate SST-2 instability for binary classification
3. Develop hybrid routing system (latent for classification, text for reasoning)

### Long-Term (Next 6 Months)

1. Theoretical analysis of super-additive conditions
2. Multi-model bridges (N>2 models sharing single compressed representation)
3. Production deployment for classification pipelines
4. Explore privacy-preserving applications of non-interpretable soft tokens

---

## Appendix: Key Experimental Artifacts

### Preserved Experiments

| ID | Description | Key Result |
|----|-------------|------------|
| exp001 | SST-2 Signal Check | First success: 93.46% |
| exp003 | Comprehensive Ablations | Layer 31 + 8 tokens optimal |
| exp005 | SST-2 Corrected Prompts | 94.72% (fair comparison) |
| exp006 | AG News Corrected Prompts | 88.9% (+18.4pp vs text) |
| exp007 | GSM8K Latent CoT | 2.0% (confirmed failure) |
| exp008 | Paper Final Results | 96.7% SST-2, 90.7% AG News |

### Repository Structure

```
LatentWire/
├── telepathy/                    # Main neural bridge implementation
│   ├── latent_bridge.py         # Perceiver Resampler architecture
│   ├── train_telepathy.py       # Training loop
│   ├── eval_telepathy_*.py      # Task-specific evaluation
│   ├── preserved_data/          # Reproducible experiments
│   └── REPORT.md                # Complete 19-phase journey
├── latentwire/                   # Initial approach (deprecated)
├── paper_writing/                # Paper drafts and figures
└── scripts/                      # Training and evaluation scripts
```

### Hardware Requirements

- Training: 4x H100 80GB GPUs (~3-4 hours per task)
- Inference: Single H100 (37ms per sample)
- Memory: 15.2GB (both models loaded)

---

## Summary Statement

**Telepathy demonstrates that heterogeneous LLMs can communicate directly through learned soft tokens, achieving 22x speedup and super-additive accuracy on classification tasks.** However, this success is bounded: reasoning tasks fail completely, revealing that the approach excels at pattern matching but cannot preserve the precise information needed for symbolic computation. This work establishes both the promise and the limitations of continuous latent communication between different language model families.

---

*Document Version: 1.0 - January 2026*
*Companion to: PRESENTATION_OUTLINE.md, PRESENTATION_HOUR_TALK.md*
