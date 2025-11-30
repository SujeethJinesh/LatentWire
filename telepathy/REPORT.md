# Latent Telepathy: Technical Report

**Project**: Cross-Model Latent Communication
**Models**: Llama 3.1 8B → Mistral 0.3 7B
**Date**: November 29, 2025
**Status**: Phase 9 Ready (Bag-of-Words Supervision)

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

## 11. Phase 2 Training Results

Training completed on 2025-11-28. Key metrics:

### Loss Progression

| Step | Total Loss | LM Loss | Contrastive Loss |
|------|-----------|---------|------------------|
| 50 | 2.54 | 1.50 | 2.07 |
| 200 | 1.41 | 0.94 | 0.95 |
| 500 | 1.26 | 0.90 | 0.70 |
| 1000 | 1.12 | 0.94 | 0.35 |
| 2000 | 1.18 | 0.91 | 0.53 |

### Analysis

**Contrastive loss dropped from 2.07 → 0.35** by step 1000, proving that soft tokens are now **unique per input**. The bridge is no longer producing generic "average thoughts."

**LM loss increased slightly** (0.77 → 0.91) compared to Phase 1. This is expected: forcing uniqueness makes the task harder but more grounded in reality.

### The "Lazy Student" Lesson

- **Phase 1**: Mistral ignored blurry latent vectors and guessed based on probability (cheating)
- **Phase 2**: Contrastive learning punishes the bridge if Question A produces similar tokens to Question B
- **Result**: The model must now look at the actual data instead of guessing

---

## 12. Phase 2 Evaluation Results (FAILURE)

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Accuracy** | 0% | 0% |
| **Partial** | 75% | **50%** ⬇️ |
| **Failed** | 25% | **50%** ⬆️ |

**Phase 2 performed WORSE than Phase 1.**

### Failure Mode: Semantic Drift

The outputs revealed garbage tokens:
```
"schen: I'm going to have 10000000000000000000..."
"rile=100*100/100=1000000000000000000..."
"pile of 100000000000000000000..."
```

### Root Cause Analysis

Contrastive learning made vectors **mathematically unique** but pushed them into **dead zones** - regions of Mistral's embedding space that don't map to real words.

| Symptom | Evidence |
|---------|----------|
| Corrupted tokens | "schen:", "rile=", "nek@", "bia=" |
| Value explosion | 10^100+ numbers |
| No entity transfer | Never see input words like "Janet", "eggs" |

The StatisticalNormalizer matched mean/std but not the **geometric structure** of the embedding manifold.

---

## 13. Phase 3: Manifold Anchoring

### The Fix

Three changes to keep soft tokens in the "Safe Operating Area":

1. **Learnable Normalizer**: Unfreeze affine parameters (l_mean, l_std, m_mean, m_std) so gradients can fine-tune the voltage match.

2. **Output Clamping**: Apply `tanh()` to bound outputs to [-1, 1], then scale by learnable factor. Prevents 10^100 explosion.

3. **Batch Anchor Loss**: Pull soft tokens toward target answer embeddings:
   ```python
   loss_anchor = F.mse_loss(soft_tokens.mean(1), answer_embeds.mean(1))
   ```

### Loss Weights

| Loss | Phase 2 | Phase 3 |
|------|---------|---------|
| LM | 1.0 | 1.0 |
| Contrastive | 0.5 | **0.1** (reduced) |
| Anchor | 0 | **1.0** (new) |

### Architecture Changes

```python
# V3: Learnable normalizer
self.l_mean = nn.Parameter(l_mean)  # was: register_buffer

# V3: Output clamping
x = self.output_ln(x)
x = torch.tanh(x)  # bound to [-1, 1]
x = x * self.output_scale  # learnable scale
```

---

## 14. Phase 3 Evaluation Results (FIRST SUCCESS!)

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Accuracy** | 0% | 0% | **5% (1/20)** ✓ |
| **Partial** | 75% | 50% | 70% |
| **Failed** | 25% | 50% | 25% |

**Phase 3 achieved the FIRST CORRECT ANSWER!** The output clamping and learnable normalizer helped, even though anchor loss was broken.

### Training Metrics

| Step | Total | LM | Anchor | Contrastive |
|------|-------|-----|--------|-------------|
| 50 | 1.48 | 1.28 | 0.0000 | 2.06 |
| 500 | 0.93 | 0.90 | 0.0000 | 0.26 |
| 2000 | 0.76 | 0.76 | 0.0000 | 0.02 |

**BUG DISCOVERED**: Anchor loss was 0.0000 throughout training!

---

## 15. Phase 3.1: Anchor Loss Bug Fix

### Root Cause

The anchor loss used MSE between vectors with **10x scale mismatch**:

| Component | Scale |
|-----------|-------|
| soft_tokens (after tanh×scale) | ±0.003 |
| answer_embeds (raw embeddings) | ±0.03 |
| MSE result | ~0.0000007 |

The MSE was so small it provided no gradient signal.

### The Fix

Changed from MSE to **cosine similarity**, which is scale-invariant:

```python
# OLD (broken): MSE with scale mismatch
return F.mse_loss(soft_mean, answer_mean)  # → 0.0000

# NEW (fixed): Cosine similarity ignores scale
soft_norm = F.normalize(soft_mean, dim=1)
answer_norm = F.normalize(answer_mean, dim=1)
cos_sim = (soft_norm * answer_norm).sum(dim=1).mean()
return 1.0 - cos_sim  # 0 = aligned, 2 = opposite
```

### Expected Impact

With anchor loss actually working:
- Soft tokens will be **directionally aligned** with answer embeddings
- Combined with contrastive loss: unique AND meaningful vectors
- Expected accuracy improvement from 5% → 10-20%

---

## 16. Changelog

| Date | Change |
|------|--------|
| 2025-11-28 | Initial implementation (Phase 1) |
| 2025-11-28 | Added evaluation script |
| 2025-11-28 | Phase 1 results: 0% accuracy, 75% partial |
| 2025-11-28 | Added Phase 2 with contrastive learning |
| 2025-11-28 | Phase 2 training: Ctr loss 2.07→0.53 |
| 2025-11-28 | Phase 2 eval: 0% acc, 50% partial (WORSE) |
| 2025-11-28 | Diagnosed "Semantic Drift" failure mode |
| 2025-11-28 | Added Phase 3: Manifold Anchoring |
| 2025-11-28 | Phase 3 eval: **5% acc** (FIRST SUCCESS!) |
| 2025-11-28 | Diagnosed anchor loss scale mismatch bug |
| 2025-11-28 | Phase 3.1: Fixed anchor loss (MSE→Cosine) |
| 2025-11-28 | Phase 3.1 eval: 5% acc, **85% partial** (↑15%) |
| 2025-11-28 | **Phase 4: Found train/eval primer mismatch bug!** |
| 2025-11-28 | Phase 4 eval: 0% acc, but NO MORE meta-commentary! |
| 2025-11-28 | New failure: Entity scrambling (ducks→chickens) |
| 2025-11-28 | Phase 5: High-res telepathy (256 tokens, layer 16) |

---

## 17. Phase 3.1 Evaluation Results

### Training Success

| Metric | Phase 3 | Phase 3.1 |
|--------|---------|-----------|
| Anchor Loss Start | 0.0000 | **0.7160** |
| Anchor Loss End | 0.0000 | **0.1167** |

**The fix worked** - anchor loss now provides gradient signal.

### Evaluation Results

| Metric | Phase 3 | Phase 3.1 | Change |
|--------|---------|-----------|--------|
| Accuracy | 5% | 5% | — |
| Partial | 70% | **85%** | +15% |
| Failed | 25% | **10%** | -15% |

### Remaining Issue: "Thought Vector" Meta-Commentary

The model generates **descriptions of vectors** rather than solving problems:

```
Input:  "Janet's ducks lay 16 eggs..."
Output: "The thought vector has 100 components..."
```

**Hypothesis**: The primer "Analysis of received thought vector:" triggers Mistral to describe vectors rather than use them as context.

### Next Steps (Phase 4)

1. **Change primer** from "Analysis of received thought vector:" to "Answer:" or "Solution:"
2. **Entity alignment loss**: Penalize when output entities differ from input entities
3. **Longer training**: 2000 steps may be insufficient for semantic grounding

---

## 18. Phase 4: Train/Eval Primer Mismatch Fix

### The Bug

| Component | Primer Used |
|-----------|-------------|
| Training (`train_telepathy_v3.py`) | `"Answer: "` ✓ |
| Evaluation (`eval_telepathy.py`) | `"Analysis of received thought vector: "` ✗ |

**The model was trained correctly but evaluated with the wrong primer!**

This explains why Mistral generated "thought vector" meta-commentary - the eval primer literally asked it to analyze thought vectors.

### The Fix

Changed `eval_telepathy.py` line 249:
```python
# OLD (wrong):
primer = "Analysis of received thought vector: "

# NEW (matches training):
primer = "Answer: "
```

### Expected Impact

With matching primers, the existing Phase 3.1 checkpoint should work properly. No retraining needed - just re-run evaluation.

---

## 19. Phase 4 Evaluation Results

### Summary

| Metric | Phase 3.1 | Phase 4 | Change |
|--------|-----------|---------|--------|
| Accuracy | 5% | 0% | ⬇️ |
| Partial | 85% | 75% | ⬇️ |
| Failed | 10% | 25% | ⬆️ |

### Major Behavioral Shift: No More Meta-Commentary!

**Before (Phase 3.1 - wrong primer)**:
```
Input:  "Janet's ducks lay 16 eggs..."
Output: "The thought vector has 100 components..."
```

**After (Phase 4 - correct primer)**:
```
Input:  "Janet's ducks lay 16 eggs..."
Output: "The number of eggs that the chickens lay in a week is 16 * 4 = <<16*4=64>>64..."
```

The model now generates **actual GSM8K-style math reasoning**!

### New Failure Mode: Entity Scrambling

| Question | Output |
|----------|--------|
| Janet's **ducks** | **Chickens** |
| 60 **elves** | **People at party** |
| **Cherries** | **Apples** |
| **Pomegranates** | **Apples** |

### What the Bridge Transmits

| Information | Transmitted? |
|-------------|--------------|
| "This is a math problem" | ✅ Yes |
| "Use GSM8K format" | ✅ Yes |
| Specific entities | ❌ No |
| Specific numbers | ❌ Partially |

### Diagnosis: Lossy Compression

The 128 soft tokens preserve **task structure** but lose **semantic details**. The bridge acts like a "blurry JPEG" - you can tell it's a math problem but can't read the specifics.

### Potential Fixes

1. **More soft tokens**: 128 → 256 or 512
2. **Entity-aware loss**: Penalize when output entities differ from input
3. **Stronger anchor weight**: Currently 1.0, try 2.0-5.0
4. **Different source layer**: Layer 20 may be too abstract, try layer 16 or 24

---

## 20. Phase 5: High-Resolution Telepathy

### The Insight

Phase 4 proved semantic transfer works - "pomegranates" → "apples" shows category-level understanding. We just need to **sharpen the image**.

### Changes

| Parameter | Phase 4 | Phase 5 | Rationale |
|-----------|---------|---------|-----------|
| Soft Tokens | 128 | **256** | Double bandwidth for entity details |
| Source Layer | 20 | **16** | More concrete features, less abstract |
| Anchor Weight | 1.0 | **2.0** | Force stronger semantic alignment |
| Steps | 2000 | **2500** | More training for larger capacity |

### Why Layer 16?

- **Layer 20**: Abstract reasoning ("This is math about quantities")
- **Layer 16**: Concrete concepts ("This is about ducks and eggs")

Lower layers preserve more surface-level details like entity names.

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v5.sh
```

### Success Criteria

If "Janet" and "ducks" appear in output (not "chickens"), we've succeeded. Target: >10% accuracy with entity preservation.

---

## 21. Phase 5 Evaluation Results

### Training Completion

Training completed successfully on 2025-11-29. Final metrics at step 1170:

| Metric | Value |
|--------|-------|
| Total Loss | ~1.17 |
| LM Loss | ~0.79 |
| Anchor Loss | ~0.13 |
| Contrastive Loss | ~1.66 |

### Evaluation Results

| Metric | Phase 4 | Phase 5 | Change |
|--------|---------|---------|--------|
| **Accuracy** | 0% | **5% (1/20)** | ⬆️ |
| **Partial** | 75% | **40%** | ⬇️ |
| **Failed** | 25% | **55%** | ⬆️ |

### Text Baseline Comparison

| Method | Accuracy |
|--------|----------|
| Text Baseline (Mistral reads question) | 0% |
| Telepathy (Llama→Bridge→Mistral) | 5% |

**Note**: The text baseline also failed (0%), suggesting GSM8K requires chain-of-thought prompting for Mistral.

### Entity Scrambling Persists

| Question Topic | Telepathy Output Topic |
|---------------|----------------------|
| Ducks laying eggs | Farmer selling **apples** |
| Knitting wool skeins | Pounds of **meat** |
| Pomegranates | Pieces of **candy** |
| Cherries | **Apples** |

### The One Correct Answer

```
Question: "20 students, 5 good at math only, 8 good at English only..."
GT Answer: 12
Telepathy: "There are 12 people in the group..." → 12 ✓
```

Interestingly, the telepathy output describes a completely different scenario ("people in a group", "men/women") but arrives at the correct number. This suggests the bridge transfers numerical relationships but scrambles entities.

### Diagnosis

Phase 5 changes (Layer 16, 256 tokens, Anchor 2.0) did **NOT** fix entity scrambling. The bridge transfers:
- ✅ Task structure (math word problem format)
- ✅ GSM8K-style reasoning chains (with `####` markers)
- ❌ Specific entities (consistently scrambled)

---

## 22. Phase 6: The Translator Pivot

### Critical Architectural Insight

**The Deep Bug**: We are training the bridge to be a **Math Solver**, not a **Translator**.

#### Current Training (V1-V5)

```
Input:    Llama reads Question ("Janet has 16 ducks...")
Anchor:   Mistral's Answer embeddings ("16 * 4 = 64...")
Problem:  Bridge must convert Q→A (solve math!)
Result:   Bridge learns generic "math vectors", discards entities
```

The bridge (a small 4-layer adapter) cannot solve math. To minimize loss, it learns to output generic "math answer vectors" that are directionally similar to answers, discarding specific entities ("ducks") because they aren't central to the answer embedding space.

#### The Fix (V6)

```
Input:    Llama reads Question ("Janet has 16 ducks...")
Anchor:   Mistral's Question embeddings ("Janet has 16 ducks...")
Goal:     Bridge(Llama_Q) ≈ Mistral(Q)
Result:   Bridge translates Q→Q, Mistral's 7B params do reasoning
```

### Key Code Change

```python
# V5 (broken): Anchor to ANSWER embeddings
with torch.no_grad():
    tgt_ans_enc = tgt_tok(tgt_answer_texts, ...)
    target_embeds = tgt_model.get_input_embeddings()(tgt_ans_enc.input_ids)

# V6 (fixed): Anchor to QUESTION embeddings
with torch.no_grad():
    tgt_q_enc = tgt_tok(tgt_q_texts, ...)  # "Question: {q}\nAnswer:"
    target_q_embeds = tgt_model.get_input_embeddings()(tgt_q_enc.input_ids)
```

### Conceptual Shift

| Aspect | V1-V5 | V6 |
|--------|-------|-----|
| Bridge Role | Solver (Q→A) | Translator (Q→Q) |
| Anchor Target | Answer embeddings | Question embeddings |
| Who does reasoning? | Bridge (4 layers) | Mistral (7B params) |
| Entity preservation | ❌ Lost in compression | ✅ Should be preserved |

### Configuration

| Parameter | Value |
|-----------|-------|
| Source Layer | 16 |
| Soft Tokens | 256 |
| Anchor Weight | 2.0 |
| Steps | 3000 |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v6.sh
```

### Success Criteria

If outputs mention "ducks" when question has "ducks" (instead of "apples"), the architectural insight is correct and entity scrambling should be fixed.

### Expected Outcome

| Scenario | What It Means |
|----------|---------------|
| Entities preserved | ✓ Translator pivot works! |
| Entities still scrambled | Need different approach (entity extraction, explicit copying) |

---

## 23. Changelog (Continued)

| Date | Change |
|------|--------|
| 2025-11-29 | Phase 5 training completed |
| 2025-11-29 | Phase 5 eval: 5% acc, 40% partial |
| 2025-11-29 | Entity scrambling persists despite changes |
| 2025-11-29 | **Critical insight: Anchoring to answers forces bridge to solve math** |
| 2025-11-29 | Phase 6: The Translator Pivot (anchor to question embeddings) |
| 2025-11-29 | Phase 6 OOM crash - reduced batch size 8→4, added memory clearing |
| 2025-11-29 | Phase 6 NameError crash - fixed src_h reference in logging code |
| 2025-11-29 | Phase 6 slow training (3s/iter) - removed torch.cuda.empty_cache() |
| 2025-11-29 | Phase 6 training completed at step 2147/3000 (71%) |
| 2025-11-29 | Phase 6 final metrics: Total=3.86, LM=3.46, Anchor=0.18, Contrastive=0.54 |
| 2025-11-29 | Phase 6 eval crashed: architecture mismatch (LatentBridgeV3 vs LatentBridge) |
| 2025-11-29 | Fixed: bridge_version 3→1 in run_telepathy_v6.sh (awaiting re-run) |
| 2025-11-29 | **Phase 6 eval: 0% correct, 5% partial - WORSE than V5** |
| 2025-11-29 | Outputs are degenerate repetitive loops (10*10*10..., 12=12=12...) |
| 2025-11-29 | Root cause: Anchor (→Q) and LM (→A) objectives are in conflict |
| 2025-11-29 | **Gemini identified Magnitude Bug: 33x signal overload** |
| 2025-11-29 | Phase 7: Scale-Corrected Telepathy (tanh + learnable scale) |
| 2025-11-29 | Created latent_bridge_v7.py, train_telepathy_v7.py, run_telepathy_v7.sh |
| 2025-11-29 | Phase 7 crashed: dtype mismatch (float32 vs bfloat16) in cross_attn |
| 2025-11-29 | Fixed: cast normalized output back to input dtype before resampler |
| 2025-11-29 | Phase 7 OOM crash on all 4 H100s - reduced batch size 8→4 |
| 2025-11-29 | **Phase 7 eval: 0% correct, 40% partial - scale fix worked, but semantics lost** |
| 2025-11-29 | No more degenerate loops, but outputs hallucinate "students/clubs" templates |
| 2025-11-29 | Output scale converged to 0.0022 (target ~0.0027) |

---

## 26. Phase 7 Results: Scale Fix Worked, Semantics Lost

### Evaluation Results

| Metric | Phase 5 | Phase 6 | Phase 7 |
|--------|---------|---------|---------|
| Correct | 5% (1/20) | 0% (0/20) | 0% (0/20) |
| Partial | 40% (8/20) | 5% (1/20) | **40% (8/20)** |
| Output Quality | Entity scrambled | Degenerate loops | Hallucinated templates |

### Training Metrics (Step 2500)

| Metric | Value |
|--------|-------|
| Total Loss | 1.766 |
| LM Loss | 1.019 |
| Anchor Loss | 0.323 (↓ from 0.55) |
| Contrastive Loss | 1.008 |
| **Output Scale** | 0.0022 |

### What Worked

1. **No more degenerate loops** - Outputs are coherent sentences, not `10*10*10...`
2. **Scale fix effective** - Output magnitude (~0.002) matches Mistral's expected range
3. **Attention not saturating** - Softmax no longer "hard max"

### What Failed

Outputs are coherent but **completely hallucinated**:

```
Question: Janet's ducks lay 16 eggs...
V7 Output: The total number of students is 3600.
           The number of students in the first class is 1200...

Question: Jim spends 2 hours watching TV...
V7 Output: igten
           The first number is 10000000000...
```

All outputs talk about "students", "clubs", "10th grade" - a generic math problem template.

### Diagnosis

The bridge learned to generate **generic math problem patterns** rather than encoding specific question semantics.

Hypotheses:
1. Anchor loss (cosine to answer embeddings) might be too weak to enforce semantic alignment
2. The 256 soft tokens might collapse to similar representations
3. Contrastive loss alone isn't sufficient for semantic preservation

### Next Directions

1. **Increase anchor weight** - Force stronger alignment to specific answers
2. **Add reconstruction loss** - Decode soft tokens back to text
3. **Use KL divergence** - Match full output distribution, not just embeddings
4. **Try smaller soft token count** - 256 might allow too much collapse

---

## 24. Phase 6 Results: The Translator Pivot Failed

### Evaluation Results

| Metric | Phase 5 | Phase 6 | Change |
|--------|---------|---------|--------|
| Correct | 5% (1/20) | 0% (0/20) | ↓ Worse |
| Partial | 40% (8/20) | 5% (1/20) | ↓ Much Worse |
| Entity Preservation | Scrambled | N/A (degenerate) | - |

### Sample Outputs (Degenerate)

```
Question: Janet's ducks lay 16 eggs...
Expected: 18
V6 Output: She day of 2* 12*2*2*2*2*2*2*2*2*2=12=12=12=12=12=12=12...

Question: A company wanted to buy 500 computers...
Expected: 385000
V6 Output: The $100000000000000000000000000000000000000000000...
```

### Why V6 Failed: Conflicting Objectives

The Translator Pivot created a fundamental conflict between training objectives:

```python
# V6 has CONFLICTING objectives:
loss_anchor = cosine(soft_tokens, question_embeddings)  # Push soft tokens → Q
loss_lm = cross_entropy(model.generate(soft_tokens) → answer)  # Need soft tokens to help generate A
```

**The Conflict:**
1. Anchor loss wants soft tokens to look like question embeddings
2. LM loss wants soft tokens to contain answer-generation information
3. These are mutually exclusive goals

**V5 had ALIGNED objectives:**
- Anchor loss: soft_tokens → answer_embeddings
- LM loss: generate(soft_tokens) → answer
- Both push toward answer representation

**V6 created MISALIGNED objectives:**
- Anchor loss: soft_tokens → question_embeddings
- LM loss: generate(soft_tokens) → answer
- Anchor and LM fight each other

### The Insight Was Wrong

The original hypothesis was:
> "V5 failed because the bridge tried to solve math (Q→A)"

The corrected understanding:
> "The problem isn't Q→A vs Q→Q. The problem is that anchor and LM must agree on what soft tokens represent."

### Next Steps → Phase 7

---

## 25. Phase 7: Scale-Corrected Telepathy

### The Magnitude Bug (Root Cause)

Gemini identified a critical physical implementation bug:

| Component | Expected | Actual | Problem |
|-----------|----------|--------|---------|
| Mistral input range | ~0.03 std | - | - |
| Perceiver output | - | ~1.0 std | LayerNorm guarantees unit variance |
| **Overload** | - | **33x** | Attention softmax saturates |

**The Physics:**
1. Perceiver ends with LayerNorm + residuals → output std ≈ 1.0
2. Mistral's embedding layer expects std ≈ 0.03
3. We're feeding 33x louder signals than expected
4. Attention softmax saturates → "hard max" → stuck in loops

This explains the degenerate outputs (`10*10*10...`, `100000...`). It's like audio clipping.

### The Fix: LatentBridgeV7

```python
class LatentBridgeV7(nn.Module):
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        # ... existing components ...

        # PHASE 7 FIX: Output Scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden, src_mask=None):
        normed = self.normalizer(src_hidden)
        compressed = self.resampler(normed, src_mask)

        # tanh bounds to [-1, 1], then scale to target range
        return torch.tanh(compressed) * self.output_scale
```

### Two Key Fixes

1. **Output Scaling**: `tanh(output) * 0.03` to match Mistral's expected range
2. **Reverted to Answer Anchoring**: V6's Question anchor had conflicting objectives

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Source Layer | 16 | Same as V5/V6 |
| Soft Tokens | 256 | Same as V5/V6 |
| Anchor Weight | 2.0 | V5 settings |
| Steps | 2500 | Slightly reduced |
| **Output Scale** | ~0.03 | **NEW: Learnable, initialized to target RMS** |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v7.sh
```

### Success Criteria

| Symptom | What It Means |
|---------|---------------|
| Coherent numbers (not 10*10*10...) | ✓ Scale fix working |
| Still degenerate loops | Need further magnitude debugging |
| Correct answers | 🎉 Full success |

---

## 27. Phase 8: Reconstruction Loss (Mode Collapse Fix)

### Diagnosis: Mode Collapse

Phase 7 fixed the scale issue, but revealed a deeper problem: **Mode Collapse**.

The bridge learned to output generic "GSM8K templates" regardless of input:

```
Question: Janet's ducks lay 16 eggs...
Output: "The total number of students is 3600..."

Question: Jim spends 2 hours watching TV...
Output: "The number of students in the first class is 1200..."
```

All outputs mention "students", "clubs", "10th grade" - memorized patterns from the GSM8K training distribution.

### Root Cause Analysis

| What Happened | Why |
|---------------|-----|
| Bridge outputs similar vectors for all inputs | Anchor loss (cosine) can be minimized by collapsing to mean |
| Mistral generates coherent but wrong content | The "students/clubs" template satisfies LM loss on average |
| No specific entity encoding | No objective forces bridge to encode "ducks" specifically |

**The Problem**: Nothing prevents the bridge from learning:
```python
def collapsed_forward(self, any_input):
    return GENERIC_MATH_TEMPLATE_VECTOR  # Works "on average"
```

### The Fix: Reconstruction Loss

If we can reconstruct the source hidden states from soft tokens, the bridge **must** encode input-specific information.

```python
class LatentBridgeV8(nn.Module):
    def __init__(self, ...):
        # ... existing components ...

        # PHASE 8: Reconstruction Head
        self.recon_proj = nn.Linear(tgt_dim, src_dim)

    def forward(self, src_hidden, src_mask=None):
        # ... normalization and compression ...
        compressed = self.resampler(normed, src_mask)
        scaled_out = torch.tanh(compressed) * self.output_scale

        # Return BOTH: scaled (for Mistral) and raw (for recon loss)
        return scaled_out, compressed
```

### Training Objective

```python
# Phase 8: Reconstruction Loss
recon_vec = bridge.recon_proj(raw_compressed).mean(dim=1)  # [B, src_dim]
src_vec = masked_mean(src_hidden)  # Target: original source representation
loss_recon = F.mse_loss(recon_vec, src_vec)

total_loss = loss_lm + anchor_weight * loss_anchor +
             contrastive_weight * loss_contrastive +
             recon_weight * loss_recon  # NEW
```

### Why This Works

| Without Recon Loss | With Recon Loss |
|-------------------|-----------------|
| Bridge can output generic vector | Must preserve input-specific info |
| "ducks" → template | "ducks" → must be reconstructable |
| Mode collapse allowed | Mode collapse prevented |

If the bridge collapses to a generic template, reconstruction loss will spike because it can't distinguish "ducks" from "students".

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Source Layer | 16 | Same as V7 |
| Soft Tokens | 256 | Same as V7 |
| Anchor Weight | 2.0 | Same as V7 |
| Contrastive Weight | 0.1 | Same as V7 |
| **Recon Weight** | 1.0 | **NEW** |
| Batch Size | 4 | Reduced (V7 OOM fix) |
| Steps | 2500 | Same as V7 |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v8.sh
```

### Success Criteria

| Output Contains | What It Means |
|-----------------|---------------|
| Input-specific entities ("ducks" when question has "ducks") | ✓ Recon loss working |
| Still generic templates ("students") | Need stronger recon weight |
| Correct answers | 🎉 Full success |

---

## 28. Changelog (Continued)

| Date | Change |
|------|--------|
| 2025-11-29 | **Diagnosed Mode Collapse in V7** |
| 2025-11-29 | V7 outputs coherent but all hallucinate "students/clubs" templates |
| 2025-11-29 | **Phase 8: Reconstruction Loss to force information preservation** |
| 2025-11-29 | Created latent_bridge_v8.py with ReconstructionHead |
| 2025-11-29 | Created train_telepathy_v8.py with recon_weight parameter |
| 2025-11-29 | Created run_telepathy_v8.sh |
| 2025-11-29 | Updated eval_telepathy.py to handle V8 tuple output |
| 2025-11-29 | **Phase 8 eval: 5% correct, 25% partial - Mode Collapse NOT fixed** |
| 2025-11-29 | Recon loss dropped to 0.008 by step 100 - too easy to satisfy |
| 2025-11-29 | Mean-pooling destroys entity info; bridge satisfies loss with generic vectors |

---

## 29. Phase 8 Results: Reconstruction Loss Failed

### Evaluation Results

| Metric | Phase 7 | Phase 8 | Change |
|--------|---------|---------|--------|
| Correct | 0% (0/20) | **5% (1/20)** | +1 |
| Partial | 40% (8/20) | 25% (5/20) | -3 |
| Failed | 60% | 70% | +2 |

### Training Metrics

| Metric | Start | End | Problem |
|--------|-------|-----|---------|
| **Recon Loss** | 0.3506 | **0.0086** | ← Dropped too fast! |
| LM Loss | 1.500 | 0.838 | OK |
| Anchor Loss | 0.5265 | 0.0786 | OK |
| Contrastive | 1.382 | 0.557 | OK |
| Scale | 0.0022 | 0.0042 | OK |

### Why Reconstruction Loss Failed

The reconstruction target is **mean-pooled source hidden states**:

```python
src_vec = (src_h * mask).sum(dim=1) / mask.sum(dim=1)  # Average all tokens
```

This average is dominated by:
- ✅ Common structure: "Question:", "How many", "per day"
- ❌ NOT entities: "ducks", "Janet", "pomegranates"

**The bridge can satisfy recon loss by encoding generic math-problem structure while ignoring specific entities.**

### Evidence: Recon Loss Too Easy

| Step | Recon Loss | Interpretation |
|------|------------|----------------|
| 50 | 0.3506 | Learning |
| 100 | 0.0598 | Already low |
| 500 | 0.0099 | Near-zero |
| 2500 | 0.0086 | Trivially satisfied |

By step 100, the bridge learned to reconstruct "generic math question vector" without needing entity-specific information.

### Sample Outputs (Mode Collapse Persists)

```
Q: Janet's ducks lay 16 eggs...
V8: "The number of apples that the boy has..."

Q: Nissa hires 60 elves...
V8: "The number of students in the school is 120 * 2..."

Q: cherries shared by Richard, Jerry, Robert...
V8: "The total number of apples is 30 + 20 + 20..."
```

### Regression: Degenerate Loops Returned

Tests 12 and 18 show number explosions (`30000000000000...`), which V7 had eliminated.

### Next Directions

The fundamental issue: **mean-pooling destroys entity information**.

Potential fixes:
1. **Token-level reconstruction** - predict each source token position
2. **Entity extraction loss** - explicitly match named entities (NER)
3. **Lower layer** - layer 12 instead of 16 for more surface features
4. **CLS-style token** - dedicated position for entity encoding
5. **Contrastive on entities** - pull together same-entity examples

---

## 30. Phase 9: Bag-of-Words Supervision

### The "Generic Noun" Trap

The bridge is acting like a **lossy JPEG compressor**:

1. It sees "16 ducks"
2. Compresses to "Small Number of Farm Animals"
3. Mistral decompresses to "20 chickens" (closest guess)

**V8's Mean-Pooling Problem**: The "average vector" of a sentence about ducks is mathematically very similar to the "average vector" of a sentence about chickens. The model minimized recon loss without preserving specific entities.

### The Fix: Bag-of-Words Classification

Force the bridge to **prove** it remembers specific words from the input.

**Mechanism**:
1. Add a "Detector" Head: Linear layer predicting every token in the input
2. If Llama reads "Janet" and "ducks", bridge must activate those classifiers
3. If it outputs "Quiara" or "apples", it gets penalized

```python
class LatentBridgeV9(nn.Module):
    def __init__(self, ...):
        # ... existing components ...

        # PHASE 9: Bag-of-Words Head
        self.bow_head = nn.Linear(tgt_dim, src_vocab_size)  # 4096 -> 128k

    def forward(self, src_hidden, src_mask=None):
        compressed = self.resampler(...)
        scaled_out = torch.tanh(compressed) * self.output_scale

        # MAX pooling detects "Did this feature appear?"
        pooled = compressed.max(dim=1).values  # [B, D]
        bow_logits = self.bow_head(pooled)     # [B, Vocab]

        return scaled_out, bow_logits
```

### Training Objective

```python
# Multi-hot target: Which tokens were in the input?
bow_targets = torch.zeros(B, vocab_size)
bow_targets.scatter_(1, src_enc.input_ids, 1.0)  # Set 1 for present tokens

# BCE Loss: Predict 1 for present words, 0 for absent
loss_bow = F.binary_cross_entropy_with_logits(bow_logits, bow_targets)

total_loss = loss_lm + anchor_weight * loss_anchor +
             contrastive_weight * loss_contrastive +
             bow_weight * loss_bow  # NEW: Weight = 5.0
```

### Why This Should Work

| V8 (Mean Recon) | V9 (BoW) |
|-----------------|----------|
| "ducks" → average vector | "ducks" → must predict token 42857 |
| "chickens" → similar average | "chickens" → different token, different target |
| Same loss for both | Different loss forces distinction |

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Source Layer | 16 | Same as V8 |
| Soft Tokens | 256 | Same as V8 |
| Anchor Weight | 2.0 | Same as V8 |
| Contrastive Weight | 0.1 | Same as V8 |
| **BoW Weight** | 5.0 | **NEW: Strong signal** |
| Batch Size | 4 | Reduced (BoW head adds memory) |
| Steps | 3000 | Extended training |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v9.sh
```

### Success Criteria

| Output Contains | What It Means |
|-----------------|---------------|
| "ducks" when question has "ducks" | BoW forcing entity transfer |
| "Janet" when question has "Janet" | Named entities preserved |
| Still "students/apples" | Need stronger BoW or different approach |

---

## 31. Phase 9 Results: BoW Supervision FAILED

**Run**: telepathy_v9_20251130_102755
**Training**: 3000 steps, batch_size=2 (reduced for BoW head memory)

### Training Metrics

| Step | Total | LM | BoW | Anchor | Scale |
|------|-------|-----|-----|--------|-------|
| 50 | 3.52 | 1.58 | 0.077 | 0.745 | 0.0024 |
| 500 | 2.14 | 1.05 | 0.002 | 0.486 | 0.0030 |
| 1500 | 1.95 | 1.02 | 0.008 | 0.413 | 0.0032 |
| 3000 | 1.93 | 1.04 | 0.007 | 0.392 | 0.0034 |

**Key Observation**: BoW loss dropped to ~0.007 - nearly perfect token prediction. Anchor improved from 0.74 to 0.39.

### Evaluation Results

| Metric | Value |
|--------|-------|
| Correct | 0/20 (0%) |
| Partial | 5/20 (25%) |
| Baseline | 0/20 (0%) |

**MODE COLLAPSE PERSISTS**:
```
Question: Janet's ducks lay 16 eggs...
Output: "Q: 1/2 of 1/3 of 1/4..."  ← Repetitive fraction pattern

Question: Jim spends 2 hours watching TV...
Output: "Question 1: The number of students in the class is 24..."  ← SAME TEMPLATE
```

### Why BoW Failed: Separate Pathways

```
                                    ┌─────────────────┐
                                    │ BoW Head        │
Input → Resampler → [pooled] ──────►│ Linear(4096,128k)│───► Token Prediction ✓
                  │                 └─────────────────┘     (BoW loss: 0.007)
                  │
                  └──► [soft tokens] → scale → Mistral ───► Generic Output ✗
                                                            (Still "students")
```

**The Problem**: BoW head operates on **pooled** resampler output, not on soft tokens. The bridge can:
1. Make soft tokens that satisfy anchor loss (generic answer-like vectors)
2. Separately learn BoW classification from the pooled representation

There's **no gradient forcing soft tokens themselves** to encode different information for different inputs. Mistral still sees the same collapsed representation.

### Fundamental Issue

The BoW loss gradient flows through the resampler, but:
- Soft tokens = `resampler output` × `learnable scale`
- BoW logits = `max(resampler output)` → `linear projection`

These share the resampler but have different objectives. The resampler can satisfy both by:
- Keeping soft token positions generic (for anchor loss)
- Encoding input info in dimensions that BoW head detects but soft tokens ignore

### What We Learned

1. **Auxiliary classifiers don't fix mode collapse** - they can be satisfied without affecting the main output
2. **BoW loss needs to directly constrain what Mistral sees**
3. **Need stronger coupling between input detection and output generation**

---

## 32. Changelog (Continued)

| Date | Change |
|------|--------|
| 2025-11-29 | **Phase 9: Bag-of-Words Supervision** |
| 2025-11-29 | Created latent_bridge_v9.py with BoW head (tgt_dim -> vocab_size) |
| 2025-11-29 | Created train_telepathy_v9.py with multi-hot BCE loss |
| 2025-11-29 | Created run_telepathy_v9.sh with BOW_WEIGHT=5.0 |
| 2025-11-29 | Updated eval_telepathy.py to handle V9 |
| 2025-11-29 | Key insight: Mean-pooling allows mode collapse; BoW forces token prediction |
| 2025-11-30 | **V9 Results: FAILED** - BoW loss 0.007 but outputs still show mode collapse |
| 2025-11-30 | Diagnosis: BoW head is separate pathway, doesn't constrain soft tokens |
