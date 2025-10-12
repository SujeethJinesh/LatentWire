# Phase 1 Analysis: What Is The Adapter Actually Learning?

## TL;DR

The adapter is learning the **inverse of a linear PCA projection**. This is an easy problem, which explains why we get 87% cosine similarity with minimal training (~100 steps). The remaining 1150 steps only improve from 77% → 87%.

## The Fixed PCA Bottleneck

Phase 1 uses **fixed PCA compression**:

```
Original embeddings [batch, seq, 4096]
         ↓ (PCA projection, FIXED)
Compressed [batch, seq, 1024]
         ↓ (Adapter, LEARNED)
Reconstructed [batch, seq, 4096]
         ↓ (LLM generation)
Generated text
```

Key insight: **PCA is frozen**. It was fit once on 5k samples and never updated.

## What The Adapter Learns

The adapter learns two things:

### 1. Inverse PCA Projection

PCA does: `x_compressed = (x - mean) @ V`
Adapter learns: `x_reconstructed ≈ x_compressed @ V^T + mean`

Since PCA is **linear**, the inverse is also **linear**. The adapter (3-layer MLP) can easily approximate this.

### 2. Magnitude Correction

Llama embeddings have RMS ≈ 0.5 per token
Adapter output has RMS ≈ 64 per token (from LayerNorm)

We explicitly fix this with RMS scaling:
```python
reconstructed = reconstructed * (orig_rms / (recon_rms + 1e-8))
```

This is **not learned** - it's a fixed rescaling applied after the adapter.

## Learning Curve

From λ=0.001 diagnostics:

| Step | Cosine Similarity | What's Happening |
|------|------------------|-------------------|
| 10   | 0.40             | Learning basic linear map |
| 50   | 0.65             | Capturing main PCA directions |
| 100  | 0.77             | Good approximation found |
| 1250 | 0.87             | Refinement (10% improvement over 1150 steps) |

**Key observation**: The adapter learns 90% of what it needs in the first 100 steps. The remaining 1150 steps provide only marginal improvement (0.77 → 0.87).

## Why High Cosine Similarity ≠ Good Generation

Cosine similarity measures **direction** not **task-specific information**.

PCA preserves:
- ✅ Semantic content (facts, names, dates)
- ✅ Linguistic structure (grammar)
- ✅ Overall "meaning" of embeddings

PCA loses:
- ❌ Task framing ("this is QA, not text continuation")
- ❌ Format cues (stopping behavior)
- ❌ Fine-grained next-token predictions

This is why Phase 1a achieves:
- **87% cosine similarity** (excellent reconstruction)
- **24% F1** (poor task performance)

The embeddings are semantically close, but the model doesn't know to stop after the answer.

## Why Generation Objectives Fail

### Phase 1a: Pure Reconstruction (λ_gen = 0)

```
Loss = cosine_loss + 0.1 * mse_loss
     = 0.88 + 0.1 * 0.00014
     ≈ 0.88
```

Adapter optimizes purely for embedding similarity. Gets 87% cosine, 24% F1.

### Phase 1b: λ_gen = 0.001

```
Loss = recon_loss + 0.001 * (kce_loss + kd_loss)
     = 0.12 + 0.001 * (2.6 + 4.7)
     = 0.12 + 0.007
     = 0.127
```

Generation objectives contribute only 6% of total loss. Not enough signal to change behavior. Result: 87% cosine, 23% F1 (no improvement).

### Phase 1b: λ_gen = 0.005

```
Loss = 0.33 + 0.005 * (7.2 + 6.9)
     = 0.33 + 0.07
     = 0.40
```

Generation objectives now 17% of loss. Reconstruction degrades:
- Cosine drops from 87% → 67%
- F1 likely same or worse

### Phase 1b: λ_gen = 0.5 (Original Failure)

```
Loss = 0.90 + 0.5 * (5.1 + 4.6)
     = 0.90 + 4.85
     = 5.75
```

Generation objectives dominate (84% of loss). Result:
- Cosine collapses to 9.5%
- F1 drops to 0%
- Mode collapse: "the the the the"

## The Fundamental Problem

Generation objectives optimize in **token space** (discrete), while reconstruction optimizes in **embedding space** (continuous). These objectives **conflict** when using fixed compression:

1. **Reconstruction says**: "Make embeddings close to originals"
2. **K-token CE says**: "Predict specific next tokens correctly"
3. **Conflict**: Embeddings that are close (high cosine) don't necessarily lead to correct token predictions

With **learned encoders** (full LatentWire), both objectives can shape the latent space together. But with **fixed PCA**, the latent space can't adapt - only the decoder (adapter) can change. This creates conflicting gradients.

## Implications For Full LatentWire

The full system (latentwire/train.py) uses:
- k_ce_weight = 0.5 (default)
- kd_first_k_weight = 1.0 (default)

If these weights cause mode collapse with **fixed PCA**, they might also be too strong for **learned encoders** early in training.

### Recommendations:

1. **Start with weaker weights**: λ ≈ 0.01-0.05 instead of 0.5-1.0
2. **Use annealing**: Ramp from 0 → target over first few epochs
3. **Monitor cosine similarity**: If it drops below 60%, generation objectives too strong
4. **Watch for mode collapse early**: First 100 steps should show if it's happening

## Why Fast Sweep Is Sufficient

Since the adapter learns its basic behavior in ~100 steps:
- **1000 samples, 1 epoch** = ~125 steps
- **Enough to detect mode collapse**: Cosine will drop if λ too high
- **10× faster**: 2 min per λ instead of 16 min

We don't need full convergence to identify the threshold where generation objectives break reconstruction.

## Next Steps

1. **Run fast sweep**: Test λ ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
2. **Find threshold**: Identify max λ where cosine > 0.7 and F1 ≥ 0.20
3. **Use for full LatentWire**: Start with safe λ values from sweep
4. **Consider annealing**: If no single λ works, try gradual ramp-up
