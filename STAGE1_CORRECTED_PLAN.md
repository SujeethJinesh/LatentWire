# Stage 1 Corrected Implementation Plan

## Executive Summary

Current implementation has 5 critical issues preventing rigorous validation. This document provides a corrected two-phase approach for 100% scientific rigor.

## Critical Issues Found

### 1. **Train/Eval Mismatch** (BLOCKER)
**Current**: Training uses reconstructed_prompt + real_answer_embeddings (teacher forcing)
**Evaluation**: Uses only reconstructed_prompt (autoregressive generation)
**Impact**: Training optimizes wrong objective, results will be misleading

### 2. **Loss Magnitude Imbalance**
**Current**: `loss = 1.0 * recon_loss + 1.0 * ce_loss`
**Problem**: CE (~2-5) dominates MSE (~0.1-1.0) by 5-10×
**Impact**: Reconstruction loss essentially ignored

### 3. **Insufficient PCA Training Data**
**Current**: PCA fitted on 100 samples (~17k tokens)
**Applied to**: 10k samples (~2M+ tokens)
**Impact**: Principal components don't generalize, poor compression quality

### 4. **Quick Eval Metric Too Loose**
**Current**: Substring matching (false positives)
**Should be**: F1 score (matches full evaluation)

### 5. **Data Preparation** ✅
**Status**: CORRECT - verified load_squad_subset properly formats data

## Corrected Two-Phase Approach

### **Phase 1: Pure Reconstruction Baseline**

**Hypothesis**: Good reconstruction → Good generation

**Training**:
```python
# Only MSE reconstruction loss
recon_loss = MSE(reconstructed_embeddings, original_embeddings)
loss = recon_loss  # No CE loss
```

**Evaluation**:
```python
# Same as current: autoregressive generation
F1 score on generated answers
```

**Key Fixes**:
1. Remove CE loss entirely (no teacher forcing)
2. Fix PCA: fit on 5000 samples (50% of training data)
3. Fix quick eval: use F1 score instead of substring matching
4. Add reconstruction quality metrics to logs

**Success Criteria**:
- F1 ≥70%: Reconstruction alone sufficient → hypothesis validated!
- F1 50-70%: Reconstruction helps but needs generation training
- F1 <50%: Either compression too lossy OR architecture flawed

**What Phase 1 Tests**:
- Can PCA compress 8× without losing critical information?
- Can 19.9M parameter adapter reconstruct well enough for generation?
- Is reconstruction quality predictive of generation quality?

### **Phase 2: Add Generation-Aware Training** (Only if Phase 1 F1 <70%)

**Hypothesis**: Direct generation feedback improves adaptation

**Training**:
```python
# Reconstruction + prompt perplexity
recon_loss = MSE(reconstructed, original)

# Forward through model with ONLY reconstructed prompt
outputs = model(
    inputs_embeds=reconstructed,  # No answer embeddings!
    attention_mask=attention_mask,
    labels=input_ids  # Predict next tokens in prompt
)
prompt_ppl_loss = outputs.loss

# Balanced loss (after tuning α/β)
loss = α * recon_loss + β * prompt_ppl_loss
```

**Why This Works**:
- Matches evaluation (no teacher forcing)
- Optimizes next-token prediction quality
- Still computationally efficient (no full answer generation)

**Loss Weight Tuning**:
```python
# Normalize magnitudes
α = 1.0 / E[recon_loss]  # ~1.0 if MSE ≈ 1.0
β = 1.0 / E[prompt_ppl_loss]  # ~0.2 if CE ≈ 5.0
# This makes both terms contribute equally
```

## Implementation Changes Required

### File: `train_adapter_only.py`

**Changes for Phase 1**:

1. **Fix PCA fitting** (line 162-179):
```python
# Before: 100 samples
for item in tqdm(dataset[:100], desc="Collecting embeddings"):
    ...

# After: 5000 samples (50% of training data)
pca_samples = min(5000, len(dataset) // 2)
for item in tqdm(dataset[:pca_samples], desc="Collecting embeddings"):
    ...
```

2. **Remove CE loss** (line 258-288):
```python
# DELETE lines 258-288 (the teacher-forced CE loss)
# Keep only:
loss = recon_loss  # Pure reconstruction
```

3. **Fix quick eval** (line 374-375):
```python
# Before: Substring matching
if val_item['answer'].lower() in generated.lower():
    correct += 1

# After: F1 score
from latentwire.core_utils import batch_metrics
pred = generated.strip()
ref = val_item['answer']
_, f1 = batch_metrics([pred], [ref])
total_f1 += f1
```

4. **Add reconstruction metrics** (after line 256):
```python
# Log reconstruction quality
with torch.no_grad():
    # Cosine similarity
    cos_sim = F.cosine_similarity(
        reconstructed[attention_mask.bool()],
        orig_embeds[attention_mask.bool()],
        dim=-1
    ).mean()

    # Relative error
    rel_error = (reconstructed - orig_embeds).norm() / orig_embeds.norm()

    # Log these metrics
    log_diagnostics(diagnostic_log, step, epoch, {
        "recon_mse": recon_loss.item(),
        "recon_cosine_sim": cos_sim.item(),
        "recon_rel_error": rel_error.item()
    })
```

**Changes for Phase 2** (only if needed):

1. **Replace CE loss with prompt perplexity** (line 258-288):
```python
# NEW: Prompt-only perplexity loss
outputs = model(
    inputs_embeds=reconstructed,
    attention_mask=attention_mask,
    labels=input_ids  # Only prompt tokens, no answers
)
prompt_ppl_loss = outputs.loss

# Magnitude-balanced combination
alpha = 1.0 / (epoch_avg_recon_loss + 1e-8)  # Normalize by typical magnitude
beta = 1.0 / (epoch_avg_ppl_loss + 1e-8)
loss = alpha * recon_loss + beta * prompt_ppl_loss
```

### File: `scripts/run_stage1_h100.sh`

**Update documentation** (line 57-60):
```bash
echo "Expected Performance:"
echo "  - Phase 1 (reconstruction only): Test if good reconstruction → good generation"
echo "  - Target: ≥70% F1 validates hypothesis"
echo "  - If <70% F1: Move to Phase 2 (add generation training)"
```

## Testing Plan

### Unit Tests (Run Locally on MacBook):

1. **Test PCA compression/decompression**:
```python
def test_pca_reconstruction_quality():
    compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
    embeddings = torch.randn(1000, 128)
    compressor.fit(embeddings)

    # Test reconstruction error
    compressed = compressor.compress(embeddings[:100])
    # Should maintain reasonable quality
    assert compressed.shape == (100, 32)
```

2. **Test adapter forward pass**:
```python
def test_adapter_shape_preservation():
    adapter = Adapter(d_z=512, d_model=4096, ...)
    compressed = torch.randn(8, 20, 512)
    reconstructed = adapter(compressed)
    assert reconstructed.shape == (8, 20, 4096)
```

3. **Test data loading**:
```python
def test_squad_data_format():
    dataset = load_squad_subset("train", 10)
    assert len(dataset) == 10
    assert "source" in dataset[0]
    assert "answer" in dataset[0]
    assert "Context:" in dataset[0]["source"]
    assert "Question:" in dataset[0]["source"]
```

### Integration Test (Run on HPC):

```bash
# Quick smoke test with Phase 1
python train_adapter_only.py \
  --samples 1000 \
  --epochs 1 \
  --batch_size 32 \
  --compress_dim 512 \
  --recon_weight 1.0 \
  --ce_weight 0.0  # Pure reconstruction!
```

Expected: Should complete without errors, produce F1 score.

## Success Metrics

### Phase 1 Success:
- ✅ Training completes without errors
- ✅ Reconstruction loss decreases over time
- ✅ Cosine similarity >0.9 between reconstructed and original
- ✅ **F1 ≥70%**: Hypothesis validated!
- ✅ Logs show reconstruction metrics improving

### Phase 1 Partial Success (needs Phase 2):
- ✅ Training completes
- ✅ Reconstruction quality good (cos_sim >0.9, MSE <0.5)
- ⚠️ **F1 50-70%**: Need generation-aware training
- → Proceed to Phase 2

### Phase 1 Failure (investigate further):
- ❌ **F1 <50%** despite good reconstruction
- → Issue is either:
  1. PCA compression too lossy (try compress_dim=1024)
  2. Adapter architecture inadequate (try larger/different architecture)
  3. Training insufficient (try more epochs/samples)

## Timeline

1. **Implement Phase 1 fixes**: 1-2 hours
2. **Run unit tests locally**: 30 min
3. **Run integration test on HPC**: 2 hours (training time)
4. **Analyze results**: 1 hour
5. **If needed, implement Phase 2**: 1-2 hours
6. **Final validation run**: 4-6 hours (full 10k samples × 3 epochs)

**Total**: 1-2 days for complete rigorous validation

## Conclusion

The current implementation has fundamental correctness issues that would produce misleading results. The corrected two-phase approach:

1. **Tests the right hypothesis**: Good reconstruction → good generation
2. **Matches train/eval**: No teacher forcing mismatch
3. **Provides clear path forward**: Phase 2 only if Phase 1 insufficient
4. **Scientifically rigorous**: Tests one variable at a time

This is the CORRECT way to validate Stage 1.
