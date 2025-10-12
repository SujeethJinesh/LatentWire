# Stage 1 Phase 1 Training Diagnosis

## Current Status: COSINE FIX WORKED, BUT F1 STILL 0%

### Training Results (Latest Run)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|--------|---------|---------|---------|
| **Cosine Similarity** | 0.875 | 0.888 | 0.894 |
| **MSE Loss** | 0.978 | 0.962 | 0.962 |
| **F1 Score** | 0.0% | 0.0% | 0.0% |
| **EM Score** | 0.0% | 0.0% | 0.0% |

### ✅ What Worked
- **Loss weight fix successful**: Cosine similarity now **increases** during training (0.47 → 0.89)
- Previous run: Cosine **decreased** from 0.65 → 0.51 (MSE dominated)
- Current run: Cosine **increased** from 0.47 → 0.89 (cosine dominates)

### ❌ Critical Problem
**High cosine similarity (0.89) but zero generation quality (F1 = 0%)**

This indicates the embeddings are directionally aligned but semantically incorrect. The LLM is generating garbage text despite receiving "similar-looking" embeddings.

---

## Architecture Overview

```
INPUT TEXT (SQuAD Question+Context)
    ↓
[Tokenizer] → input_ids [batch, seq_len]
    ↓
[Frozen LLM Embeddings] → orig_embeds [batch, seq_len, 4096]
    ↓
[PCA Compressor] → compressed [batch, seq_len, 1024]  (4× compression)
    ↓
[Adapter Network] → reconstructed [batch, seq_len, 4096]
    ↓
[Frozen LLM Generation] → output_ids [batch, generated_len]
    ↓
[Tokenizer Decode] → generated_text
    ↓
[F1 Scoring] vs gold answer
```

## Possible Failure Points

### 1. **PCA Compression Loss** ⚠️ HIGH SUSPICION
**Problem**: PCA on token embeddings may destroy semantic structure

**Why this could fail**:
- **PCA assumes Gaussian structure**, but LLM embeddings are not Gaussian
- **Context-dependent semantics**: Each token's meaning depends on its context
- **Non-linear relationships**: PCA only captures linear correlations
- **Reported "100% variance"** is an artifact (only computing top-1024 of 4096 components)

**Evidence**:
- 4× compression (4096 → 1024) loses 75% of dimensions
- Even with 0.89 cosine, the reconstruction may be semantically off
- The "explained variance" metric from randomized SVD is misleading

**Test**: Try 2× compression (4096 → 2048) to see if less aggress

ive compression helps

### 2. **Adapter Architecture Mismatch** ⚠️ MEDIUM SUSPICION
**Problem**: Adapter may not have enough capacity or right structure

**Current adapter** (from latentwire/models.py):
```python
Adapter(
    d_z=1024,           # compressed dim
    d_model=4096,       # target dim
    latent_length=32,   # not used in reconstruction
    hidden_mult=4,      # hidden = 1024 × 4 = 4096
    dropout=0.1
)
```

**Potential issues**:
- Adapter may be learning shortcuts (e.g., outputting mean embedding)
- Linear projection may be insufficient for reconstructing complex semantic structure
- Dropout might be interfering with reconstruction

**Test**:
- Increase `hidden_mult` from 4 to 8 (more capacity)
- Remove dropout during reconstruction training
- Check if adapter outputs are degenerate (all similar)

### 3. **Evaluation Mismatch** ⚠️ LOW-MEDIUM SUSPICION
**Problem**: Model may need additional tokens/context after reconstruction

**Current evaluation**:
```python
outputs = model.generate(
    inputs_embeds=reconstructed,  # No BOS token!
    attention_mask=attention_mask,
    max_new_tokens=20,
    do_sample=False
)
```

**Potential issues**:
- **No BOS token**: LLM might expect a beginning-of-sequence token before generation
- **No prompt anchor**: Reconstructed embeddings go straight to generation without "Answer:" anchor
- **Attention mask mismatch**: We create a mask but LLM might expect different structure

**Test**:
- Add explicit BOS token before reconstruction
- Append "Answer:" tokens after reconstructed embeddings
- Try generation with original text prefix + reconstructed middle

### 4. **Embedding Space Shift** ⚠️ MEDIUM-HIGH SUSPICION
**Problem**: Reconstructed embeddings may be in the right direction but wrong magnitude/distribution

**Evidence**:
- MSE = 0.962 at end (quite high - embeddings have norm ~30-50)
- Relative error = 114 (reconstructed embeddings are very different in magnitude)
- Cosine = 0.89 only measures direction, not magnitude

**Why this matters**:
- LLMs are sensitive to embedding magnitudes
- LayerNorm in LLM helps, but may not fully correct large magnitude errors
- Embedding norms encode information (rare tokens often have higher norms)

**Test**:
- Normalize reconstructed embeddings to match original embedding RMS
- Check distribution of embedding norms (original vs reconstructed)
- Try per-token normalization instead of per-example

### 5. **Training Objective Mismatch** ⚠️ MEDIUM SUSPICION
**Problem**: Optimizing reconstruction != optimizing generation

**Current training**:
- Loss = `0.1 * MSE + Cosine`
- Optimizes: "Make embeddings point in same direction"
- **Does not optimize**: "Make LLM generate correct text"

**Why this could fail**:
- High cosine doesn't guarantee semantic preservation
- Compression may destroy critical tokens (e.g., entity names)
- Some dimensions may matter more than others for generation

**Test**: This is why Phase 2 exists (add generation-aware training)

---

## Recommended Next Steps (In Order)

### Step 1: Add Diagnostic Logging ⚡ ✅ **COMPLETED**
**Why**: We need to see what text is being generated

**UPDATE**: Logging has been added to `train_adapter_only_phase1.py`!

**`evaluate_full()` now logs:**
- First 10 generated examples with expected answers
- **Token-level reconstruction**: Shows what tokens the reconstructed embeddings map to
  - Original tokens (from input_ids)
  - Nearest tokens for reconstructed embeddings (via cosine similarity to vocab)
- Original vs reconstructed embedding norms
- Norm ratio with status indicator (TOO LOW / OK / TOO HIGH)
- Per-example cosine similarity

**`evaluate_quick()` now logs:**
- First example in each quick eval with token-level reconstruction
- Aggregate statistics: average norm ratio and cosine similarity

**Next run will show:**
1. What text the model is actually generating
2. If it's complete garbage or partially correct
3. **If reconstructed embeddings map to completely different tokens** (smoking gun!)
4. If magnitude mismatch is causing the issue (norm ratio)

### Step 2: Check Embedding Magnitude Distribution ⚡ **NOW LOGGED**
**Why**: Verify if magnitude mismatch is causing issues

**UPDATE**: Norm ratios are now logged in both evaluation functions!
- Quick eval: Shows aggregate avg norm ratio
- Full eval: Shows per-example norm ratio with status indicators

Next run will reveal if norm ratio is consistently too low/high.

### Step 3: Test Less Aggressive Compression (2×)
**Why**: 4× may be too much information loss

```bash
# In scripts/run_stage1_h100.sh
--compress_dim 2048  # was 1024 (4×, try 2× instead)
```

### Step 4: Test Magnitude Normalization
**Why**: Ensure reconstructed embeddings have correct scale

```python
# After adapter forward pass
reconstructed = adapter(compressed)

# Normalize to match original RMS
orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
recon_rms = reconstructed.pow(2).mean(dim=-1, keepdim=True).sqrt()
reconstructed = reconstructed * (orig_rms / (recon_rms + 1e-8))
```

### Step 5: If All Else Fails → Phase 2
**Why**: Pure reconstruction may be fundamentally insufficient

Move to generation-aware training:
- Add teacher forcing
- Add cross-entropy loss on generated tokens
- Add KL divergence between reconstructed and original logits

---

## Key Metrics to Track

### Reconstruction Quality
- ✅ **Cosine similarity**: 0.89 (good!)
- ❌ **MSE**: 0.962 (high - embeddings very different)
- ❌ **Relative error**: 114 (huge magnitude mismatch)
- ✅ **Embedding norm ratio**: NOW LOGGED (will show in next run)

### Generation Quality
- ❌ **F1 score**: 0.0% (broken)
- ❌ **EM score**: 0.0% (broken)
- ✅ **Generated text samples**: NOW LOGGED (first 10 examples per eval)
- ❓ **Token diversity**: Will check after next run

---

## Hypothesis Ranking (Most Likely → Least Likely)

1. **Embedding magnitude mismatch** (Relative error = 114!)
   - Fix: Normalize reconstructed embeddings to match original RMS

2. **PCA destroys semantic structure** (4× compression too aggressive)
   - Fix: Reduce to 2× compression (4096 → 2048)

3. **Training objective insufficient** (reconstruction ≠ generation)
   - Fix: Move to Phase 2 (generation-aware training)

4. **Adapter architecture inadequate** (can't learn proper reconstruction)
   - Fix: Increase hidden_mult, remove dropout, add residual connections

5. **Evaluation setup wrong** (missing BOS, attention mask issues)
   - Fix: Add BOS token, check attention mask alignment

---

## Status Summary

✅ **Logging Added** - Token-level diagnostic logging is now in place!

**What happens on next training run:**
1. Every 100 steps: Quick eval will show 1 example + aggregate norm/cosine stats + token mapping
2. End of each epoch: Full eval will show first 10 examples with detailed diagnostics
3. We'll see:
   - What text is actually being generated
   - If it's garbage, partially correct, or systematically wrong
   - **Token-level reconstruction**: What tokens the reconstructed embeddings map to
   - Norm ratios to diagnose magnitude mismatch
   - Per-example cosine similarity

**Key Diagnostic: Token-Level Reconstruction**

The reconstructed embeddings are mapped back to their nearest tokens in vocabulary space:
- `Original tokens:` → The actual input text tokens
- `Reconstructed →:` → What tokens the reconstructed embeddings are closest to

**What this reveals:**
- If tokens match perfectly → PCA+adapter preserving token-level semantics ✅
- If tokens are similar (synonyms) → Semantic drift but potentially recoverable
- If tokens are completely different → Compression destroying semantic information ❌

**Example scenarios:**
```
Original:     "Context: The Eiffel Tower is in Paris"
Reconstructed: "Context: The Eiffel Tower is in Paris"  ← GOOD: exact match

Original:     "Context: The Eiffel Tower is in Paris"
Reconstructed: "Context: The France structure is in Paris"  ← OK: semantic drift

Original:     "Context: The Eiffel Tower is in Paris"
Reconstructed: "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"  ← BAD: collapsed to common tokens
```

**Next steps after we see the logs:**
- If tokens match exactly but F1=0%: Check generation decode logic
- If tokens drift to synonyms: Try magnitude normalization or less compression
- If tokens collapse to common/garbage: PCA destroying semantics → Phase 2 or reduce compression
- If norm ratio is consistently <0.8 or >1.2: Try magnitude normalization (Step 4)
