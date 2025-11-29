# Latent Telepathy: Technical Report

**Project**: Cross-Model Latent Communication
**Models**: Llama 3.1 8B → Mistral 0.3 7B
**Date**: November 28, 2025
**Status**: Phase 1 Complete, Phase 2 Ready

---

## 1. Abstract

Modern multi-agent AI systems communicate via text generation—a computationally expensive process that accounts for ~70-80% of inference latency. **Latent Telepathy** eliminates this bottleneck by capturing the internal hidden states ("thought vectors") of a Source Model and injecting them directly into a Target Model's embedding space.

This project built a neural bridge between Llama 3.1 8B and Mistral 0.3 7B. Phase 1 achieved mechanical success (stable training, 95% direction preservation) but 0% task accuracy due to **Posterior Collapse**—the target model learned to ignore the soft tokens. Phase 2 introduces contrastive learning to fix this.

---

## 2. The Problem: Tower of Babel

### Current Text-Based Communication

```
Agent A (Thinking) → Agent A (Speaking) → [TEXT] → Agent B (Reading) → Agent B (Understanding)
                     ↑                              ↑
                     SLOW                           REDUNDANT
                     (autoregressive)               (re-encoding)
```

### Proposed Latent Communication

```
Agent A (Thinking) → [LATENT VECTORS] → Agent B (Understanding)
                     ↑
                     FAST (single forward pass)
```

**Goal**: Reduce inter-agent latency by 40-50% by eliminating the text generation step.

---

## 3. Architecture: The Four Boss Battles

We identified four physical incompatibilities between Llama and Mistral that must be solved:

### Battle 1: Magnitude Shock

| Property | Llama | Mistral |
|----------|-------|---------|
| Hidden state range | ±20 | ±100 |
| Scale ratio | 1x | ~5x |

**Risk**: Direct injection causes gradient explosion or signal treated as noise.

**Solution**: `StatisticalNormalizer` - Affine transformation that whitens Llama's distribution and recolors to Mistral's statistics. Calibrated on 500 GSM8K samples.

### Battle 2: Vocabulary Density

| Property | Llama | Mistral |
|----------|-------|---------|
| Vocabulary size | 128,000 | 32,000 |
| Token density | High | Low |

**Risk**: Llama encodes "bioluminescence" as 1 token; Mistral expects 3.

**Solution**: `PerceiverResampler` - Cross-attention mechanism that compresses variable-length sequences (T tokens) into fixed-length latent packets (K=64 soft tokens).

### Battle 3: RoPE Geometry

| Property | Llama | Mistral |
|----------|-------|---------|
| RoPE base frequency | 500,000 | 1,000,000 |

**Risk**: Positional rotations are incompatible—"King" in Llama's coordinate system looks like gibberish to Mistral.

**Solution**: Implicit de-rotation via learned queries. The Perceiver extracts semantic meaning while discarding source-specific positional encoding.

### Battle 4: KV Cache Amnesia

**Risk**: Mistral needs context to generate. Injecting soft tokens without priming leaves the model "confused."

**Solution**: Prefix priming with static text: `"Analysis of received thought vector: "` fills the KV cache before soft token injection.

---

## 4. Implementation

### File Structure

```
telepathy/
├── __init__.py                 # Module exports
├── latent_bridge.py           # Core architecture
│   ├── StatisticalNormalizer  # Magnitude calibration
│   ├── PerceiverResampler     # Cross-attention compression
│   └── LatentBridge           # Combined module
├── phase1_calibration.py      # Statistics collection
├── train_telepathy.py         # Phase 1: LM + Reconstruction loss
├── train_telepathy_v2.py      # Phase 2: + Contrastive loss
└── eval_telepathy.py          # Held-out test evaluation

run_telepathy.sh               # Phase 1 execution
run_telepathy_eval.sh          # Evaluation script
run_telepathy_v2.sh            # Phase 2 execution (TODO)
```

### Training Pipeline

```
Phase 1: Calibration
├── Load Llama + Mistral
├── Run 500 samples through both
├── Compute mean/std of Llama Layer 20 hidden states
├── Compute mean/std of Mistral embedding layer
└── Save stats.pt

Phase 2: Training
├── Load frozen Llama + Mistral + Bridge
├── For each batch:
│   ├── Llama: question → hidden states (Layer 20)
│   ├── Bridge: hidden states → 64 soft tokens
│   ├── Mistral: [Primer] + [Soft Tokens] + [Answer] → LM loss
│   └── Compute reconstruction loss (cosine similarity)
└── Save checkpoint
```

---

## 5. Phase 1 Results

### Training Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Reconstruction Loss | 0.053 | 95% cosine similarity preserved |
| LM Loss | 0.77 | Perplexity ~2.16 |
| Training Steps | 3,000 | ~1.5 epochs on GSM8K |

### Calibration Statistics

| Statistic | Llama (Layer 20) | Mistral (Embeddings) |
|-----------|------------------|----------------------|
| Mean Norm | 13.6 | 0.029 |
| Scale Ratio | 1x | ~460x smaller |

The normalizer successfully bridged this ~460x scale difference.

### Evaluation Results (Held-Out Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 0% (0/20) |
| **Partial Success** | 75% (15/20) |
| **Total Failure** | 25% (5/20) |

### Failure Analysis

**Observed Behavior**: Mistral generated mathematically-structured responses that were completely unrelated to the input question.

```
Input (Llama sees):  "Janet sells 16 duck eggs for $2..."
Output (Mistral):    "First find the number of apples... 100 apples"
```

**Diagnosis**: **Posterior Collapse**

The target model learned to:
1. Recognize the primer text ("Analysis of received thought vector:")
2. Infer "I should generate a math problem"
3. **Ignore** the soft tokens entirely and hallucinate

This is a classic ML failure mode where the model finds an easier solution than the intended one.

---

## 6. Phase 2: The Fix

### Root Cause

The soft tokens for different inputs were too similar. Mistral couldn't distinguish between Question A and Question B, so it learned to ignore them.

### Solution: Contrastive Learning (InfoNCE)

```python
def contrastive_loss_fn(soft_tokens, temperature=0.07):
    # Mean pool: [B, K, D] → [B, D]
    features = F.normalize(soft_tokens.mean(dim=1), dim=1)

    # Similarity matrix [B, B]
    logits = torch.matmul(features, features.T) / temperature

    # Diagonal = positives, off-diagonal = negatives
    labels = torch.arange(B)

    return F.cross_entropy(logits, labels)
```

**Effect**: Forces the bridge to produce **unique** latent representations for each input. If Question A and Question B produce similar soft tokens, the loss increases.

### Phase 2 Changes

| Parameter | Phase 1 | Phase 2 | Rationale |
|-----------|---------|---------|-----------|
| Soft tokens | 64 | 128 | More capacity |
| Batch size | 4 | 8 | More negatives for contrastive |
| Learning rate | 1e-4 | 2e-4 | Faster adaptation |
| Contrastive weight | 0 | 0.5 | Force uniqueness |
| Training steps | 3,000 | 2,000 | Sufficient with stronger signal |

### Expected Outcomes

| Outcome | Indicator | Next Step |
|---------|-----------|-----------|
| Success | >10% accuracy | Paper-worthy result |
| Partial | >30% topic match | Increase capacity further |
| Collapse persists | Similar outputs | Increase contrastive weight to 1.0 |

---

## 7. How to Run

### Phase 1 (Complete)

```bash
# On HPC:
git pull && rm -rf runs && bash run_telepathy.sh
```

### Evaluate Phase 1

```bash
bash run_telepathy_eval.sh runs/telepathy_*/bridge_final.pt
```

### Phase 2 (Next)

```bash
# On HPC:
git pull && rm -rf runs && bash run_telepathy_v2.sh
```

---

## 8. Theoretical Implications

### If Successful

1. **Latency Reduction**: Eliminates autoregressive generation between agents
2. **Bandwidth Compression**: 64-128 float16 values vs. hundreds of text tokens
3. **Heterogeneous Agents**: Proves different model families can communicate directly

### Limitations

1. **Training Required**: Each model pair needs calibration and bridge training
2. **Task Specificity**: Bridge may not generalize across domains
3. **Frozen Models**: Requires access to internal hidden states

---

## 9. Related Work

- **Perceiver** (Jaegle et al., 2021): Cross-attention for variable-length inputs
- **Soft Prompts** (Lester et al., 2021): Learned continuous prompts
- **Model Stitching** (Lenc & Vedaldi, 2015): Layer-wise feature alignment
- **InfoNCE** (Oord et al., 2018): Contrastive representation learning

---

## 10. Appendix: Key Files

### latent_bridge.py

```python
class LatentBridge(nn.Module):
    def __init__(self, args, src_dim, tgt_dim):
        self.normalizer = StatisticalNormalizer(args.stats_path)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, args.soft_tokens)

    def forward(self, src_hidden, src_mask):
        normed = self.normalizer(src_hidden)      # Match distributions
        compressed = self.resampler(normed, mask)  # Compress to K tokens
        return compressed
```

### Loss Functions

```python
# Phase 1: Reconstruction
loss_recon = 1 - cosine_similarity(soft_tokens.mean(1), src_hidden.mean(1))

# Phase 2: + Contrastive
loss_contrastive = InfoNCE(soft_tokens, temperature=0.07)
total_loss = loss_lm + 0.5 * loss_contrastive
```

---

## 11. Changelog

| Date | Change |
|------|--------|
| 2025-11-28 | Initial implementation (Phase 1) |
| 2025-11-28 | Added evaluation script |
| 2025-11-28 | Phase 1 results: 0% accuracy, 75% partial |
| 2025-11-28 | Added Phase 2 with contrastive learning |
