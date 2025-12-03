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

---

## 33. Phase 10: Auto-Encoder Pivot (The Final Fix)

### Definitive Diagnosis: The Split Brain Problem

The bridge "cheated" in all previous phases:

1. **The Cheat**: Bridge packed entity data ("Janet", "ducks") into specific dimensions/frequencies that auxiliary heads (recon, BoW) could read easily
2. **The Failure**: It did NOT pack that data into the semantic shape Mistral's embedding space recognizes
3. **The Result**: `tanh` clamping and Mistral's pre-trained attention filtered out the "auxiliary-friendly" noise, leaving only generic "math" signal

**All previous approaches failed because they treated the bridge as a "black box"**:
- Phase 2: Contrastive constraint → Cheated by making different but equally generic representations
- Phases 3-7: Geometry fixes → Cheated by satisfying geometry without encoding semantics
- Phase 9: Side-channel BoW → Cheated by encoding in separate pathway

### The Fix: Stop Treating This as "Telepathy" - Treat it as "Neural Compression"

**The New Strategy**: Force Mistral to read the **Question** back to us.

| Before (V1-V9) | After (V10) |
|----------------|-------------|
| Input: [Soft Tokens] | Input: [Soft Tokens] |
| Target: [Answer] | Target: [Question] + [Answer] |

**Why This Works**: If we force Mistral to output "Janet has 16 ducks" based only on soft tokens, the bridge **MUST** encode "Janet", "16", and "ducks" in a format Mistral can decode. It cannot cheat by guessing "Students" because the loss directly penalizes wrong entities.

### Architecture

```
Llama reads: "Question: Janet has 16 ducks..."
                ↓
        [Source Hidden States]
                ↓
        ┌───────────────┐
        │    Bridge     │ (PerceiverResampler)
        └───────────────┘
                ↓
        [128 Soft Tokens]
                ↓
        ┌───────────────┐
        │    Mistral    │
        └───────────────┘
                ↓
        Must output: "Janet has 16 ducks...\nAnswer: 18"
                     ↑
        Loss penalizes if it says "students" instead of "ducks"
```

### Training Objective

```python
# V10: Target includes Question + Answer
tgt_texts = [f"{q}\nAnswer: {a}" for q, a in zip(questions, answers)]

# Soft tokens must encode enough info to regenerate question
combined_embeds = torch.cat([soft_tokens, tgt_embeds], dim=1)

# Labels: -100 on soft tokens, then target ids (question + answer)
labels = torch.cat([ignore_prefix, tgt_ids], dim=1)

# Single LM loss - no auxiliary losses needed!
loss = tgt_model(inputs_embeds=combined_embeds, labels=labels).loss
```

### Why This is the "Final Boss"

If Mistral can regenerate "Janet has 16 ducks" from the vectors, then the vectors **provably contain** that information in Mistral-readable format. Once that is proven, the math solving (which Mistral is already good at) will follow naturally.

This removes the "bridge is guessing" variable entirely.

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Source Layer | 16 | Same as previous |
| Soft Tokens | 128 | Compression challenge |
| Depth | 4 | Same architecture |
| Heads | 8 | Same architecture |
| Batch Size | 4 | Memory efficient |
| Steps | 3000 | Full training |
| Loss | LM only | No auxiliary losses needed |

### Success Criteria

| Entity Transfer Rate | Interpretation |
|---------------------|----------------|
| > 30% | Working! Auto-encoder approach successful |
| 10-30% | Partial success, may need more training |
| < 10% | Still failing, fundamental architecture issue |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v10.sh
```

---

## 34. Changelog (Continued)

| Date | Change |
|------|--------|
| 2025-11-30 | **Phase 10: Auto-Encoder Pivot** |
| 2025-11-30 | Created train_telepathy_v10.py - Target = Question + Answer |
| 2025-11-30 | Created eval_telepathy_v10.py - Entity transfer measurement |
| 2025-11-30 | Created run_telepathy_v10.sh |
| 2025-11-30 | Key insight: Auxiliary losses can be cheated; main pathway must be supervised |
| 2025-11-30 | V10 training completed: Loss 5.3→0.9, Scale 0.0027→0.0058 |
| 2025-11-30 | V10 eval: 0% entity transfer - ALL outputs are empty strings |
| 2025-11-30 | Diagnosed: Missing attention_mask and BOS token in eval generation |
| 2025-11-30 | Fixed eval_telepathy_v10.py with attention_mask and BOS embedding |

---

## 35. Phase 10 Results: Training Succeeded, Generation Failed

### Training Metrics

| Step | Loss | Output Scale |
|------|------|--------------|
| 50 | 1.42 | 0.0036 |
| 100 | 0.92 | 0.0035 |
| 500 | 0.87 | 0.0049 |
| 1000 | 0.93 | 0.0058 |
| 2000 | 0.92 | 0.0058 |
| 3000 | 0.92 | 0.0058 |

Loss converged well from 1.4 → 0.9. Target embedding RMS was measured at 0.0027, and output scale converged to 0.0058 (2x larger - reasonable).

### Evaluation Results

| Metric | Value |
|--------|-------|
| Numbers matched | 0/60 (0%) |
| Names matched | 0/67 (0%) |
| Nouns matched | 0/9 (0%) |
| **Overall Entity Transfer** | 0/136 (0%) |

**ALL OUTPUTS WERE EMPTY STRINGS.**

### Root Cause Analysis

The eval script had two critical bugs:

1. **No Attention Mask**: `tgt_model.generate(inputs_embeds=soft_tokens, ...)` was called without `attention_mask`. The model couldn't properly attend to soft tokens.

2. **No BOS Token**: During training, the sequence was `[soft_tokens] + [BOS] + [Question] + [Answer]`. But eval only provided `[soft_tokens]`, missing the BOS token that kicks off generation.

Warning in eval log:
```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token
```

### The Fix

Updated `eval_telepathy_v10.py` lines 122-142:

```python
# FIX: Add BOS token to kick off generation
bos_emb = tgt_model.get_input_embeddings()(
    torch.tensor([[tgt_tok.bos_token_id]], device=DEVICE)
).bfloat16()
inputs_embeds = torch.cat([soft_tokens, bos_emb], dim=1)

# FIX: Add attention mask - required for proper attention
attention_mask = torch.ones(
    1, inputs_embeds.shape[1], device=DEVICE, dtype=torch.long
)

# Generate from soft tokens + BOS
out_ids = tgt_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,
    pad_token_id=tgt_tok.eos_token_id
)
```

### Potential Remaining Issue

Even with the fix, there's a deeper concern about the training paradigm:

**During training**: The model sees `[soft_tokens] + [Q+A embeddings]` and predicts Q+A. With teacher forcing, it might learn to predict later tokens from the teacher-forced context rather than from soft tokens.

**Specifically**: To predict token Q[0] at position 129, the model sees soft_tokens (0-127) + BOS (128). This is correct - it must use soft tokens.

But to predict Q[1] at position 130, it sees soft_tokens + BOS + Q[0]. It can rely on the teacher-forced Q[0] more than soft tokens.

The loss of 0.9 might be achieved primarily by predicting tokens 130+ (which have rich context) while failing on tokens 128-129 (which rely on soft tokens).

### Next Steps

1. **Re-run eval** with the fixed script to see if BOS+attention_mask produces non-empty outputs
2. If outputs are still garbage/hallucinated, investigate if teacher forcing is allowing the model to ignore soft tokens
3. Consider adding **first-token loss weighting** to emphasize the critical initial tokens

### Execution

```bash
# Re-run eval only (no retraining needed)
git pull && bash run_telepathy_v10_eval_only.sh
```

---

## 36. Phase 10 Results (Fixed Eval): TEACHER FORCING SHORTCUT CONFIRMED

### Evaluation Results (With BOS + Attention Mask Fix)

| Metric | Value |
|--------|-------|
| Numbers matched | 25/60 (41.7%) |
| Names matched | 11/67 (16.4%) |
| Nouns matched | 0/9 (0.0%) |
| **Overall** | 36/136 (26.5%) |

**BUT WAIT** - these numbers are **FALSE POSITIVES**.

### The Smoking Gun: All Outputs Are IDENTICAL

Every single one of the 20 samples produced the **exact same output**:

```
Question 1:
1. Which of the following is not a characteristic of a simple interest?
A. The rate of change
B. The amount of money borrowed
C. The time period
D. The number of compounding periods

Answer: D. The number of compounding periods
```

The "entity matches" are coincidental - numbers like "10", "4" appear in this template and happen to match some question numbers.

### Diagnosis: Complete Mode Collapse via Teacher Forcing Shortcut

| Training Setup | Eval Setup |
|---------------|------------|
| Input: [Soft Tokens] + [Question Text] | Input: [Soft Tokens] + [BOS] |
| Model learns: "Copy from Question Text" | No Question Text to copy |
| Soft tokens = ignorable noise | Falls back to memorized template |

**The model learned to ignore soft tokens entirely.** When teacher-forced question embeddings are available, it copies from them. When they're not (eval), it outputs a generic memorized pattern.

### Soft Token Statistics

```
min: -0.0058, max: 0.0058, mean: 0.0001
```

The bridge is mechanically functional (bounded by tanh × scale), but the **semantic content is not being encoded** in a way the model uses.

### Why This Specific Template?

The "simple interest multiple choice" template is likely a high-frequency pattern from Mistral's pretraining. When the model has no useful context from soft tokens, it defaults to this as a "safe" output.

### Root Cause Analysis

The V10 training objective had a fatal flaw:

```python
# V10 Training: Teacher Forcing
combined_embeds = torch.cat([soft_tokens, tgt_embeds], dim=1)
# Position 129 predicts Q[0] by looking at:
#   - Positions 0-127: soft_tokens (noise)
#   - Position 128: BOS embedding
# But position 130 predicts Q[1] by looking at:
#   - Positions 0-128: soft_tokens + BOS
#   - Position 129: Q[0] embedding (TEACHER FORCED!)
```

The model minimized loss by:
1. Learning that position 128 always predicts something generic (BOS is constant)
2. Learning to copy from teacher-forced embeddings at positions 129+
3. **Never learning to extract information from soft tokens**

### The Fix for Phase 11

To prevent the Teacher Forcing Shortcut, we need to remove the "cheat sheet":

**Option A: Pure Generation Loss (Expensive)**
- Don't use teacher forcing at all
- Generate autoregressively during training
- Loss = compare generated tokens to target

**Option B: First-K Token Focus (Efficient)**
- Weight the loss heavily on the first K tokens after soft tokens
- These tokens have NO teacher-forced context to cheat from
- If the model can predict Q[0], Q[1], Q[2] correctly, it MUST be using soft tokens

**Option C: Masked Teacher Forcing**
- Randomly mask some of the teacher-forced embeddings during training
- Force model to sometimes rely on soft tokens even for later positions

### Conclusion

Phase 10 proved that:
1. ✅ The BOS/attention mask fix works (no more empty strings)
2. ✅ The bridge is mechanically functional (scale, normalization OK)
3. ❌ The training objective allows complete bypass of soft tokens
4. ❌ Teacher forcing creates a shortcut the model exploits

**Next Step: Phase 11 - First-K Token Loss Weighting**

---

## 37. Phase 11: Bottleneck Supervision

### The Strategy

The only moment the model truly relies on the bridge is when predicting the **very first word** of the question:

| Position | Context Available | Difficulty |
|----------|------------------|------------|
| 128 (Q[0]) | Soft tokens + BOS only | **HARD** (must use bridge) |
| 129 (Q[1]) | Soft tokens + BOS + Q[0] | Medium |
| 130 (Q[2]) | Soft tokens + BOS + Q[0:2] | Easy |
| 200+ | Full question context | Trivial (just copy) |

By weighting the first 10 tokens **100x**, we tell the optimizer: "I don't care if you get the rest wrong. If you miss 'Janet', you fail."

### Implementation

**Key change in `train_telepathy_v11.py`:**

```python
# Create weight mask
loss_weights = torch.ones_like(loss_per_token)

# Boost the first K text tokens (the "bottleneck zone")
bottleneck_start = K - 1  # First text token prediction
bottleneck_end = min(bottleneck_start + args.bottleneck_tokens, loss_weights.size(1))
loss_weights[:, bottleneck_start:bottleneck_end] = args.bottleneck_weight  # 100x

# Weighted loss
weighted_loss = (loss_per_token * loss_weights * valid_mask).sum()
weighted_loss = weighted_loss / (loss_weights * valid_mask).sum()
```

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Bottleneck Tokens | 10 | First 10 tokens after soft tokens |
| Bottleneck Weight | 100.0 | 100x more important than rest |
| Source Layer | 16 | Same as V10 |
| Soft Tokens | 128 | Same as V10 |
| Steps | 3000 | Same as V10 |

### Monitoring

Three loss values are tracked:
1. **Weighted Loss**: The actual training signal (emphasizes bottleneck)
2. **Unweighted Loss**: Standard LM loss (for comparison)
3. **Bottleneck Loss**: Loss on ONLY the first 10 tokens (CRITICAL metric)

If bottleneck loss decreases significantly, the bridge is learning to encode the first few words. If it stays high while unweighted loss drops, the model is still cheating on later tokens.

### Success Criteria

| Bottleneck Loss | Interpretation |
|-----------------|----------------|
| Decreases significantly | Bridge is encoding initial tokens |
| Stays high (>3.0) | Model struggling to read soft tokens |
| Low but outputs still identical | Need even higher weight or architecture change |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v11.sh
```

### Why This Should Work

The "cheat sheet" is the teacher-forced question embeddings. But for position 128 (predicting Q[0]), there IS no cheat sheet - only soft tokens and BOS. If we weight this position 100x, the optimizer MUST make the bridge encode "Janet" in a way Mistral can decode.

This is the "needle through the bottleneck" approach: squeeze all the information through the one position where cheating is impossible.

---

## 38. Phase 11 Results: COMPLETE FAILURE

### Training Metrics

| Step | Weighted Loss | Bottleneck Loss | Interpretation |
|------|---------------|-----------------|----------------|
| 50 | 9.95 | 10.05 | Near random |
| 100 | 8.49 | 8.52 | Some learning |
| 500 | 8.20 | 8.24 | **Plateaued** |
| 1500 | 8.22 | 8.25 | No improvement |
| 3000 | 8.22 | 8.25 | Stuck |

**For reference:**
- Random prediction: ln(32000) ≈ 10.4
- V11 final: 8.2 → perplexity ≈ 3600
- Good LM: perplexity ~10-50

The bottleneck loss barely improved from random. The bridge cannot encode the information.

### Evaluation Results

**All 20 outputs are IDENTICAL GARBAGE:**

```
Q
 /******/ Q
 /******/ Q
 /******/ Q Q
 /******/ Q
 /******/  /******/ Q
 /******/ Q

#
 ,� Question  Question
{
\ Question 1
\ Question <
...
```

### Comparison: V10 vs V11

| Metric | V10 | V11 |
|--------|-----|-----|
| Output quality | Coherent wrong English | **Garbage tokens** |
| Output pattern | "Simple interest" template | Code comment noise |
| Entity transfer | 0% (false positives) | 0% |
| Overall | Bad | **Much worse** |

**V11 is objectively worse than V10.** At least V10 produced grammatically correct English.

### Root Cause: Architectural Failure

The 100x bottleneck weighting exposed a fundamental truth: **the bridge architecture cannot encode semantic information**.

| Hypothesis | Evidence |
|------------|----------|
| Teacher forcing was the problem | ❌ Removing it made outputs worse |
| Bridge can encode entities | ❌ Bottleneck loss stuck at ~8.2 (near random) |
| Soft tokens are meaningful | ❌ Mistral outputs garbage from them |

**The problem is NOT the training objective. The problem is the information pathway.**

### Why V11 Produces Garbage

1. **V10**: Model could cheat by copying teacher-forced text → coherent output
2. **V11**: Model forced to rely on soft tokens → garbage (because soft tokens ARE garbage to Mistral)

The 100x weighting successfully forced the model to try using soft tokens. But soft tokens contain no usable information, so the model outputs random noise.

### The Deeper Issue

The bridge's Perceiver-Resampler architecture:
- Takes Llama hidden states (4096 dim)
- Compresses to 128 soft tokens (4096 dim each)
- Scales by ~0.007

**But Mistral's embedding space is not the same as Llama's.** Even with statistical normalization, the geometric structure is incompatible. The bridge is trying to:
1. Extract "Janet" from Llama's layer 16
2. Encode it in a format Mistral understands
3. With only learned linear projections

This is like trying to translate Chinese to French with only a volume knob. The normalization matches loudness, but not meaning.

### Implications

After 11 phases:
- ✅ Scale/magnitude issues solved
- ✅ Teacher forcing trap identified
- ❌ **Core problem unsolved: Llama and Mistral speak fundamentally different "languages"**

The statistical normalization (mean/std matching) is not sufficient for cross-model communication. We need either:
1. **Deeper alignment**: Learn a true translation, not just normalization
2. **Shared vocabulary**: Use a decoder that both models understand
3. **Reconstruction objective**: Force the bridge to decode back to Llama space first

### Next Steps

The experiment has hit a fundamental architectural limit. Options:

1. **Add a decoder**: Bridge must reconstruct Llama's hidden states (not Mistral's embeddings)
2. **Use text bottleneck**: Quantize to actual tokens, then embed in Mistral
3. **Joint fine-tuning**: Unfreeze Mistral's first layers to learn the bridge's "dialect"
4. **Different source layer**: Try layer 8 (more surface-level) instead of 16 (more abstract)

---

## 39. Phase 12: Diffusion Bridge (DiT + Rectified Flow)

**Date**: November 30, 2025
**Status**: Implemented, Ready to Run

### Scientific Pivot: From Regression to Generation

After 11 phases of systematic experimentation, we have accumulated sufficient evidence to diagnose the fundamental failure mode and propose a theoretically-grounded solution.

---

### The Regression Gap: Why V1-V11 Failed

**Theorem (Informal)**: *Any deterministic regression model trained with MSE loss on a one-to-many mapping will produce outputs that lie in the convex hull of valid targets, but not necessarily on the target manifold.*

#### Formal Problem Statement

Let:
- $\mathcal{M}_L$ = Llama's hidden state manifold
- $\mathcal{M}_M$ = Mistral's embedding manifold
- $f: \mathcal{M}_L \rightarrow \mathcal{M}_M$ = our bridge function
- $\mathcal{T}(x) = \{y \in \mathcal{M}_M : \text{decode}(y) = \text{decode}(x)\}$ = set of valid targets for input $x$

**The Core Issue**: For any input $x$, there exist multiple valid Mistral embeddings $y_1, y_2, ..., y_k \in \mathcal{T}(x)$ that all decode to semantically equivalent outputs. MSE training finds:

$$f^*(x) = \arg\min_f \mathbb{E}_{y \sim \mathcal{T}(x)} \|f(x) - y\|^2 = \mathbb{E}[y | x]$$

The optimal MSE solution is the **conditional expectation** - the centroid of valid targets.

#### Why the Centroid Fails

**Proposition**: *The centroid of points on a curved manifold generally lies off the manifold.*

Consider a simple example: points on a circle. The average of $(1,0)$ and $(-1,0)$ is $(0,0)$ - the center, not on the circle.

```
Mistral's embedding manifold (curved):

         y₁ ●
            \
             \  ← Manifold surface
              \
    Centroid → ✕ ← OFF the manifold (garbage region)
              /
             /
            /
         y₂ ●
```

**Evidence from our experiments:**

| Phase | Approach | Output | Diagnosis |
|-------|----------|--------|-----------|
| V1-V5 | Direct regression | Mode collapse | Centroid = single garbage point |
| V6-V8 | Reconstruction | Blurry entities | Averaging similar words |
| V9 | BoW supervision | Wrong content | Centroid in wrong semantic region |
| V10 | Auto-encoder | Template output | Centroid = most common pattern |
| V11 | Bottleneck weighting | Garbage tokens | Forced to use bad centroid |

**The Regression Gap** = the distance between the learned centroid and the nearest valid point on the manifold. Our experiments show this gap is catastrophic for cross-model transfer.

---

### Manifold Projection via Diffusion

**Key Insight**: Instead of predicting a point (regression), we learn to **project onto the manifold** (diffusion).

#### Theoretical Foundation

Diffusion models learn the **score function** $\nabla_x \log p(x)$, which points toward high-density regions of the data distribution. This is equivalent to learning to project arbitrary points onto the data manifold.

**Theorem (Score Matching)**: *A model trained to denoise Gaussian-corrupted data learns to approximate the score function, which defines a vector field pointing toward the data manifold.*

For our application:
1. **Training**: Learn $v_\theta(x_t, t, c)$ that predicts how to move from noise toward valid Mistral embeddings, conditioned on Llama hidden states $c$
2. **Inference**: Start from noise, follow the learned vector field, arrive ON the manifold

```
Regression (V1-V11):           Diffusion (V12):

   Llama → [Bridge] → ✕        Llama → [Bridge] → Noise
              ↓                           ↓
         OFF manifold                   Follow v(x,t)
              ↓                           ↓
           GARBAGE                    ● ON manifold
                                          ↓
                                       VALID OUTPUT
```

#### Why Diffusion Solves the Regression Gap

1. **No averaging**: Each sample follows a unique trajectory to a single point
2. **Manifold constraint**: The score function inherently pushes toward valid data
3. **Multimodality**: Can sample different valid outputs for the same input
4. **Proven theory**: Score matching + SDE integration is mathematically sound

#### Related Work Validating This Approach

| Paper | Year | Key Result |
|-------|------|------------|
| Diffusion-LM (Li et al.) | 2022 | Continuous diffusion for discrete text generation |
| PLAID (Gulrajani & Hashimoto) | 2023 | Latent diffusion for language |
| SED (Strudel et al.) | 2024 | Score entropy discrete diffusion |
| Flow Matching (Lipman et al.) | 2023 | Rectified Flow for efficient sampling |
| DiT (Peebles & Xie) | 2023 | Transformer architecture for diffusion |

---

### The Fix: DiT + Rectified Flow

We adopt **Rectified Flow** for its simplicity and efficiency:

**Rectified Flow** (Liu et al., 2022):
- Linear interpolation: $x_t = t \cdot x_1 + (1-t) \cdot x_0$ where $x_0 \sim \mathcal{N}(0,I)$, $x_1 \sim p_{data}$
- Constant velocity: $v = x_1 - x_0$
- Training: $\mathcal{L} = \mathbb{E}_{t,x_0,x_1} \|v_\theta(x_t, t, c) - v\|^2$
- Inference: ODE integration from $x_0$ to $x_1$

**Advantages over DDPM/Score-based**:
- Straight trajectories → fewer sampling steps
- Velocity is constant → easier to learn
- No variance schedule tuning

### Architecture: DiT + Rectified Flow

**DiT (Diffusion Transformer):**
- Self-attention + Cross-attention to source
- AdaLN (Adaptive Layer Norm) conditioning on timestep + source

**Rectified Flow:**
- Simplest diffusion training: linear interpolation
- x_t = t * target + (1-t) * noise
- Predict velocity: v = target - noise (constant)
- Loss: MSE(v_pred, v_true)

**Advantages over DDPM:**
- Single-step inference possible
- Straight trajectories = faster sampling
- Simpler math, easier to debug

### Implementation

**Files created:**
- `telepathy/latent_bridge_v12.py` - DiT architecture with Rectified Flow
- `telepathy/train_telepathy_v12.py` - Training loop with velocity prediction
- `telepathy/eval_telepathy_v12.py` - Eval with diffusion sampling
- `run_telepathy_v12.sh` - Execution script

**Key parameters:**
- DiT depth: 6 layers (vs 4 in Perceiver)
- Heads: 8
- Training steps: 5000
- Learning rate: 3e-4
- Diffusion steps (sampling): 10

### Training Objective

```python
# Rectified Flow training step:
t = torch.rand(B)  # Sample timestep
noise = torch.randn_like(target)
x_t = t * target + (1-t) * noise  # Linear interpolation
v_pred = model(x_t, t, source)  # Predict velocity
loss = MSE(v_pred, target - noise)  # Velocity loss
```

### Inference

```python
# Euler integration from noise to target:
x = torch.randn(B, L, D)  # Start from noise
for i in range(num_steps):
    t = i / num_steps
    v = model(x, t, source)
    x = x + v * (1/num_steps)  # Euler step
return x  # Generated soft tokens
```

### Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Output diversity | >50% unique | Not mode collapsed |
| Entity transfer | >30% | Information flowing |
| Output coherence | Readable | ON the manifold |

### Why This Should Work

1. **Diffusion outputs are ON the manifold** - by construction, the denoising process learns to push toward valid data points

2. **No averaging problem** - each sample generates a single coherent vector, not an average of possibilities

3. **Proven in language** - Diffusion-LM, PLAID, SED show continuous diffusion works for discrete language

4. **DiT is powerful** - State-of-the-art for image generation, should transfer to latent spaces

### Expected Challenges

1. **Training may be slow** - Diffusion typically needs more steps than regression
2. **Sampling quality** - May need >10 steps for good outputs
3. **Mode coverage** - May generate valid but different content than intended

### Ready to Run

```bash
bash run_telepathy_v12.sh
```

---

### Run 1: OOM Crash (2025-11-30)

**Result**: SIGKILL (exitcode -9) = Out of Memory

The first run crashed immediately after initializing the DiT bridge:
```
[LatentBridgeV12] DiT Diffusion Bridge initialized
  - num_latents: 128
  - depth: 6
  - heads: 8
  - src_dim: 4096 -> tgt_dim: 4096
Signal 9 (SIGKILL) received by PID 732643
```

**Root Cause**: batch_size=16 exceeded H100 80GB memory with:
- Llama 8B (~16GB)
- Mistral 7B (~14GB)
- DiT Bridge (6 layers x 4096 dim)
- Batch activations

**Fix**: Reduced batch_size from 16 to 4 (1 per GPU instead of 4 per GPU).

---

### Run 2: Still OOM (2025-11-30)

**Result**: SIGKILL (exitcode -9) = Still Out of Memory

Even with batch_size=4, the DiT bridge is too large:
```
[LatentBridgeV12] DiT Diffusion Bridge initialized
  - num_latents: 128
  - depth: 6
  - heads: 8
  - src_dim: 4096 -> tgt_dim: 4096
Signal 9 (SIGKILL) received by PID 759549
```

**Root Cause**: DiT with depth=6 is much larger than V11's Perceiver:
- 6 layers × (Self-Attention + Cross-Attention + FFN)
- Each attention layer: O(128² × 4096) activations per sample
- Cross-attention to full Llama sequence adds more

**Fix**: Reduced depth from 6 to 4 (matching V11's Perceiver depth).

---

### Run 3: DDP Attribute Error (2025-11-30)

**Result**: No more OOM! Training started but crashed with:
```
AttributeError: 'DistributedDataParallel' object has no attribute 'forward_loss'
```

The depth=4 fix worked - models loaded, training loop started, but crashed on first step.

**Root Cause**: When wrapped with DDP, custom methods like `forward_loss` must be accessed via `bridge.module.forward_loss()`, not `bridge.forward_loss()`.

**Fix**: Updated `train_step()` to handle DDP wrapper:
```python
bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
loss = bridge_module.forward_loss(src_h, tgt_embeds, src_mask)
```

---

### Run 4: Dtype Mismatch (2025-11-30)

**Result**: DDP fix worked! Training started but crashed with:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

Error at `latent_bridge_v12.py:199`:
```python
src_kv = self.src_proj(src_hidden.float()).to(x_t.dtype)
```

**Root Cause**: Called `.float()` on input but `src_proj` weights are bfloat16. Linear layer requires matching dtypes.

**Fix**: Convert input to match model weight dtype instead of forcing float32:
```python
src_kv = self.src_proj(src_hidden.to(self.src_proj.weight.dtype))
```

---

### Run 5: Another Dtype Mismatch (2025-11-30)

**Result**: Previous fix worked! But now crashing in TimestepEmbedding:
```
File "latent_bridge_v12.py", line 59, in forward
    return self.mlp(t_emb)
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**Root Cause**: `SinusoidalPositionEmbedding` uses `torch.arange` and `torch.exp` which produce float32. When model is bfloat16, the MLP weights are bfloat16 but input is float32.

**Fix**: Convert sinusoidal output to match MLP weight dtype:
```python
t_emb = t_emb.to(self.mlp[0].weight.dtype)
```

---

### Run 6: Dtype in src_pool (2025-11-30)

**Result**: Previous fix worked! But now crashing in src_pool:
```
File "latent_bridge_v12.py", line 212, in forward
    src_cond = self.src_pool(src_pooled)
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**Root Cause**: Mask operations use `.float()` which promotes tensors to float32, but `src_pool` weights are bfloat16.

**Fix**: Keep mask in same dtype as src_kv, and convert src_pooled before passing to src_pool:
```python
mask = src_mask.unsqueeze(-1).to(src_kv.dtype)  # Keep same dtype
src_cond = self.src_pool(src_pooled.to(self.src_pool[0].weight.dtype))
```

---

### Run 7: SUCCESS - Training Complete (2025-11-30)

**Result**: All dtype fixes worked! Training completed successfully.

**Training Log Summary:**

| Step | Flow Loss | LR |
|------|-----------|-----|
| 50 | 1.1336 | ~3e-4 |
| 500 | 0.95 | ~2.7e-4 |
| 1000 | 0.27 | ~2.1e-4 |
| 2500 | 0.20 | ~5e-5 |
| 5000 | 0.098 | ~1e-6 |

Flow loss decreased smoothly from 1.13 → 0.098. EMA checkpoints saved.

**Evaluation Results:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Answer Accuracy | 0/20 (0%) | No correct answers |
| Number Entity Transfer | 0/56 (0%) | No numbers preserved |
| Name Entity Transfer | 1/59 (1.7%) | Almost no names |
| **Overall Entity Transfer** | 1/115 (0.9%) | Near zero |
| **Output Diversity** | 20/20 (100%) | **NOT collapsed!** |

**Sample Outputs:**

```
Q: Janet's ducks lay 16 eggs...
Output: "ska of Question Question mark. Question of Question of..."

Q: Darrell and Allen's ages are in the ratio 7:11...
Output: "legends to the government. To be or not to be. To be. To be..."

Q: Lloyd has an egg farm. His chickens produce 252 eggs...
Output: "The best way to get there. ### you can. ### you can..."
```

**Soft Token Statistics:**
```
min: -4.8438, max: 4.8125, mean: -0.0034, std: 1.0612
```

---

### Phase 12 Analysis: Stability Achieved, Semantics Lost

**The Good News:**
1. ✅ Diffusion training is stable (Flow Loss 1.13 → 0.098)
2. ✅ No mode collapse (100% output diversity)
3. ✅ Soft token statistics are healthy (mean≈0, std≈1)
4. ✅ ODE solver produces varied outputs

**The Bad News:**
1. ❌ 0% entity transfer - Mistral doesn't "see" Janet, ducks, or numbers
2. ❌ Outputs are varied but semantically garbage
3. ❌ No question content flows through the bridge

### Root Cause: Two Critical Flaws

**Flaw 1: Global Pooling Bottleneck**

The source conditioning uses mean pooling:
```python
src_pooled = (src_kv * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
cond = t_emb + src_cond  # Single vector conditioning
```

All sequence information is destroyed. "Janet's 16 ducks" and "Bob's 160 chickens" likely average to nearly identical vectors. The DiT cannot "un-average" destroyed information.

**Flaw 2: Answer Embedding Supervision**

Training supervises on answer embeddings:
```python
tgt_texts = [f"{a}{tgt_tok.eos_token}" for a in answers]  # e.g., "18<eos>"
tgt_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
```

The bridge learns to generate vectors that *look like* "18" geometrically. But at inference, Mistral needs soft tokens that *point to* what answer to produce, not the raw answer embedding.

---

## 40. Phase 13: High-Fidelity Cross-Attention Diffusion

**Date**: November 30, 2025
**Status**: Ready to Implement

### The Two Fixes

**Fix 1: Remove Global Pooling**

| V12 (Broken) | V13 (Fixed) |
|--------------|-------------|
| `cond = t_emb + pooled_src` (1 vector) | Cross-attention to full sequence |
| Entity info destroyed | Entity info preserved |

The DiT will now have **full cross-attention** to the Llama sequence at every layer. Each position can attend to "Janet" (token 5) and "ducks" (token 12) individually.

**Fix 2: Question Reconstruction Target**

| V12 (Broken) | V13 (Fixed) |
|--------------|-------------|
| Target: Answer embeddings ("18") | Target: Question embeddings ("Janet...") |
| Bridge learns answer geometry | Bridge learns to translate Q→Q |
| Mistral gets "18-like" noise | Mistral sees question semantics |

If the bridge can reconstruct "Janet has 16 ducks" in Mistral's embedding space, Mistral will naturally solve it (because Mistral is good at math).

### Architecture Changes

```python
class DiTBlockV13(nn.Module):
    def __init__(self, dim, heads, cond_dim):
        # AdaLN for timestep only (not pooled source)
        self.adaLN_modulation = nn.Linear(dim, 6 * dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(dim, heads)

        # Cross-attention to FULL Llama sequence (NEW)
        self.cross_attn = nn.MultiheadAttention(dim, heads)

        # FFN
        self.mlp = nn.Sequential(...)
```

### Training Objective

```python
# V13: Target is QUESTION embeddings, not Answer
tgt_q_texts = [f"{q}" for q in questions]  # Just the question
tgt_enc = tgt_tok(tgt_q_texts, ...)
tgt_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)

# Rectified Flow loss as before
loss = bridge.forward_loss(src_h, src_mask, tgt_embeds)
```

### Why This Should Work

1. **No Bottleneck**: Cross-attention can read each token individually
2. **Correct Target**: Reconstructing questions is well-defined
3. **Proven Architecture**: This is exactly how Stable Diffusion conditions on CLIP
4. **Mistral Does Reasoning**: Once it "reads" the question, 7B params do the math

### Implementation Plan

| File | Changes |
|------|---------|
| `latent_bridge_v13.py` | DiT with cross-attention, no pooling |
| `train_telepathy_v13.py` | Question reconstruction target |
| `eval_telepathy_v13.py` | Pass src_mask to generate() |
| `run_telepathy_v13.sh` | Execution script |

### Success Criteria

| Entity Transfer Rate | Interpretation |
|---------------------|----------------|
| > 30% | **Success!** Cross-attention works |
| 10-30% | Partial, may need more capacity |
| < 10% | Still failing, need different approach |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v13.sh
```

---

## 41. Changelog (Continued)

| Date | Change |
|------|--------|
| 2025-11-30 | **Phase 12 Run 7: Training succeeded!** |
| 2025-11-30 | Flow Loss: 1.13 → 0.098, stable convergence |
| 2025-11-30 | Evaluation: 0% entity transfer, 100% diversity |
| 2025-11-30 | **Diagnosed: Global pooling destroys entities** |
| 2025-11-30 | **Diagnosed: Answer supervision is wrong target** |
| 2025-11-30 | Phase 13: Cross-attention + Question reconstruction |
| 2025-11-30 | **Phase 13 Results: Training failed to converge** |
| 2025-11-30 | Flow Loss plateaued at 1.58 (V12 reached 0.098) |
| 2025-11-30 | Entity transfer: 5.2% (slight improvement from 0.9%) |
| 2025-11-30 | Outputs still garbage - cross-attention not enough |

---

## 42. Phase 13 Results: Training Failed to Converge

### Training Comparison

| Metric | V12 | V13 |
|--------|-----|-----|
| Initial Loss | 1.13 | 2.80 |
| Final Loss | **0.098** | **1.58** |
| Convergence | Smooth to ~0.1 | Plateaued at ~1.6 |

**Critical Finding**: V13 training never converged. The loss got stuck at ~1.6 for thousands of steps.

### Evaluation Results

| Metric | V12 | V13 | Change |
|--------|-----|-----|--------|
| Answer Accuracy | 0% | 0% | — |
| Number Transfer | 0% | 7.1% | +7.1% |
| Name Transfer | 1.7% | 3.4% | +1.7% |
| **Overall Entity** | 0.9% | **5.2%** | **+4.3%** |
| Diversity | 100% | 100% | — |

### Sample Outputs (Still Garbage)

```
Q: Janet's ducks lay 16 eggs...
V13: "'s not here. ### What's not here. ### What's not here..."

Q: Two girls each got 1/6...
V13: "I I I I I I I I I I I I..."

Q: Lloyd has an egg farm...
V13: "1. The problem is that the answer to the question..."
```

### Why V13 Failed

**1. Training Never Converged**

The DiT with cross-attention couldn't learn the mapping:
- V12 (pooled conditioning): Loss → 0.098 ✓
- V13 (cross-attention): Loss → 1.58 ✗

Possible causes:
- Cross-attention adds complexity that requires more training
- Internal dim (512) too small for cross-attention patterns
- Learning rate may need adjustment for cross-attention

**2. Question Reconstruction May Be Wrong Target**

Even if training worked, the evaluation setup may be flawed:
- Training target: Mistral question embeddings
- Eval input: [Soft tokens] + "Answer: "
- Problem: Mistral sees soft tokens as prefix, not as "the question"

Mistral expects:
```
BOS + "Question: Janet has ducks..." + "Answer: "
```

We provide:
```
[128 soft tokens trying to encode question] + "Answer: "
```

Mistral doesn't understand soft tokens = question semantics.

**3. Cross-Attention Adds Training Difficulty**

V12's pooled conditioning is simpler:
- Single vector added to timestep
- Easy for DiT to learn

V13's cross-attention is complex:
- Every layer must learn to attend to relevant source tokens
- Requires more capacity and training

### Partial Success: Entity Transfer Improved

Despite training failure, entity transfer improved from 0.9% → 5.2%. This suggests:
- Cross-attention IS extracting some information
- But the extraction is very weak due to training plateau
- More training / higher capacity might help

### Next Directions

**Option A: Fix V13 Training**
- Increase internal_dim: 512 → 1024 or 2048
- Increase training steps: 5000 → 20000
- Lower learning rate for cross-attention layers
- Add gradient logging to diagnose attention patterns

**Option B: Change Target Again**
- Instead of question embeddings, use Mistral's hidden states
- Train: Bridge(Llama_Q) → Mistral_hidden(Q)
- This gives "how Mistral represents Q internally"

**Option C: Add LM Loss During Training**
- After generating soft tokens, run Mistral forward
- Add cross-entropy loss on generating the answer
- Forces functional correctness, not just geometric similarity

**Option D: Hybrid Approach** ← **CHOSEN FOR PHASE 14**
- Keep V12's training (answer embeddings, pooled conditioning)
- But add an auxiliary cross-attention path
- Use both pooled + cross-attention together

---

## 43. Phase 14: Hybrid Conditioning Diffusion (The Fix for V13's Collapse)

### The Problem: Conditioning Collapse

V13 failed because pure cross-attention was too weak a conditioning signal. The DiT defaulted to predicting "average velocity" → repetitive outputs like "I I I I..." or "What's not here...".

This is a well-known failure mode called **"Conditioning Collapse"** or "Posterior Collapse" in diffusion models.

### The Solution: Hybrid Conditioning

Phase 14 combines V12's global pooling (strong guide rail) with V13's cross-attention (entity details). This mirrors **Stable Diffusion XL**'s architecture:

| Component | V12 | V13 | V14 (Hybrid) |
|-----------|-----|-----|--------------|
| Global conditioning | ✓ Mean pooling | ✗ | ✓ Attention pooling |
| Cross-attention | ✗ | ✓ | ✓ |
| Internal dim | 512 | 512 | **1024** |
| Heads | 8 | 8 | **16** |
| Training steps | 5000 | 5000 | **10000** |

### Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                 Hybrid Conditioning (V14)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Llama Hidden States ─┬──► Attention Pooling ──► Global     │
│       [B, T, 4096]    │        (learnable query)     Gist   │
│                       │                               │     │
│                       │                               ↓     │
│                       │                    ┌─────────────┐  │
│                       │                    │ global_cond │  │
│                       │                    │  + t_emb    │  │
│                       │                    └─────┬───────┘  │
│                       │                          │          │
│                       └──► Cross-Attention ◄─────┘          │
│                              (each block)   AdaLN           │
│                                                             │
│  DiT Block Flow:                                            │
│    x ──► AdaLN(global) ──► Self-Attn ──► Cross-Attn ──► MLP │
│                                             ↑               │
│                                     Llama sequence          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Changes from V13

1. **Attention-based Global Pooling** (not mean pooling):
   ```python
   # Learnable query attends to source sequence
   self.cond_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
   self.cond_pool = nn.MultiheadAttention(dim, 8, batch_first=True)

   global_cond, _ = self.cond_pool(query, src_seq, src_seq)
   ```
   This extracts a strong "gist" vector that prevents collapse.

2. **AdaLN on Global Conditioning**:
   ```python
   # Global cond drives AdaLN gates (V12 style)
   combined_cond = global_cond + t_emb
   shift, scale, gate = self.adaLN_modulation(combined_cond)
   ```
   The DiT has a strong signal to follow, preventing drift to average.

3. **Cross-Attention for Details**:
   ```python
   # Cross-attention fetches specific entities
   cross_out, _ = self.cross_attn(query=x, key=src_seq, value=src_seq)
   ```
   After AdaLN sets the "topic", cross-attention fills in details.

4. **Increased Capacity**:
   - Internal dim: 1024 (was 512)
   - Heads: 16 (was 8)
   - Training steps: 10000 (was 5000)

### Why This Should Work

**V12 Problem**: Global pooling destroyed entity info
**V13 Problem**: Cross-attention alone was too weak → collapse

**V14 Solution**: Both together
1. Global pooling tells DiT "this is a math problem about animals"
2. Cross-attention lets DiT fetch "Janet", "ducks", "16"
3. AdaLN provides strong conditioning signal (prevents collapse)
4. More capacity handles the complexity

This is exactly how Stable Diffusion XL works:
- Pooled CLIP embeddings (global) + Full sequence (local)

### Implementation Files

| File | Purpose |
|------|---------|
| `latent_bridge_v14.py` | Hybrid DiT with attention pooling + cross-attention |
| `train_telepathy_v14.py` | Training with 10k steps, cosine LR |
| `eval_telepathy_v14.py` | Evaluation with entity tracking |
| `run_telepathy_v14.sh` | Execution script |

### Success Criteria

| Metric | Target | V13 Result | V12 Result |
|--------|--------|------------|------------|
| Flow Loss | **Converges** | 1.58 (failed) | 0.098 ✓ |
| Output Diversity | > 50% | 100% ✓ | 100% ✓ |
| Entity Transfer | **> 30%** | 5.2% | 0.9% |

### Expected Outcomes

| Scenario | Interpretation | Next Step |
|----------|---------------|-----------|
| Loss converges + Entity > 30% | **Success!** Hybrid fixed it | Add LM loss (Phase 15) |
| Loss converges + Entity < 30% | Hybrid helps but not enough | Increase capacity or change target |
| Loss fails to converge | Fundamental issue | Try completely different approach |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v14.sh
```

---

## 44. Changelog (Continued)

| Date | Change |
|------|--------|
| 2025-11-30 | **Phase 14: Hybrid Conditioning implemented** |
| 2025-11-30 | Architecture: Attention pooling + Cross-attention |
| 2025-11-30 | Increased capacity: 1024 dim, 16 heads, 10k steps |
| 2025-11-30 | Created latent_bridge_v14.py, train/eval scripts |
| 2025-11-30 | Ready for HPC execution |
| 2025-12-01 | **Phase 14 Results: COMPLETE FAILURE** |
| 2025-12-01 | Loss plateaued at 2.0 (worse than V13's 1.58) |
| 2025-12-01 | Entity transfer: 0.0% (regression from V13's 5.2%) |
| 2025-12-01 | Hybrid conditioning made things worse |
| 2025-12-01 | **Phase 15: VQ-Telepathy implemented** |
| 2025-12-01 | Architecture: Perceiver + Vector Quantization |
| 2025-12-01 | 4096 codebook entries, LM Loss + VQ Loss |
| 2025-12-01 | Back to ANSWER target (not question reconstruction) |

---

## 45. Phase 14 Results: COMPLETE FAILURE

### Execution

```
telepathy_v14_20251130_234649/
├── train.log           # Training output
├── eval_20251201_001443.log  # Evaluation output
└── eval_v14_results.json
```

### Training Analysis

| Metric | Target | V14 Result | V13 Result | V12 Result |
|--------|--------|------------|------------|------------|
| Flow Loss | Converges | **2.0 (PLATEAU)** | 1.58 | 0.098 ✓ |
| Output Diversity | > 50% | 100% ✓ | 100% ✓ | 100% ✓ |
| Entity Transfer | > 30% | **0.0%** | 5.2% | 0.9% |
| Answer Accuracy | > 0% | **0.0%** | 0.0% | 0.0% |

### Sample Output (Catastrophic)

```
Question: Janet's ducks lay 16 eggs per day...
Output: "Question is to be a certain is a certain is a certain is a
        or is await to to be await to be await to..."
```

### Diagnosis: Manifold Mismatch is Fatal

The progressive failure across phases reveals a fundamental problem:

| Phase | Approach | Loss | Entity | Problem |
|-------|----------|------|--------|---------|
| V7 | Regression | Low | Low | Blurry averages (semantic drift) |
| V12 | Diffusion Global | 0.098 | 0.9% | Lost entity details |
| V13 | Diffusion Cross-Attn | 1.58 | 5.2% | Failed to converge |
| V14 | Hybrid Conditioning | 2.0 | 0.0% | Made everything worse |

**Root Cause**: Continuous regression cannot bridge incompatible manifolds. When predicting Mistral embeddings from Llama states:
- Any error in continuous space leads to invalid vectors
- Averaging produces "blurry" concepts (Generic Bird instead of Ducks)
- Diffusion helps but cannot recover entity-specific information

---

## 46. Phase 15: VQ-Telepathy (Vector Quantized Bridge)

### The Insight: Discretization Prevents Manifold Mismatch

| Method | Manifold Mismatch | Entity Preservation |
|--------|-------------------|---------------------|
| Regression | Outputs off-manifold vectors | Averages to generic concepts |
| Diffusion | Converges to manifold but loses details | Cannot recover entities |
| **Vector Quantization** | **Forces valid vectors** | **Discrete codes = discrete entities** |

### Why VQ Solves This

1. **Prevents Blur**: Every output is a valid codebook entry
   - Cannot output "0.5 × Duck + 0.5 × Chicken"
   - Must commit to one code or another

2. **Prevents Drift**: Discrete codes map to specific concepts
   - "Ducks" → Code #42
   - "Chickens" → Code #99
   - Cannot average to "Code #70.5" (Generic Bird)

3. **Efficient**: 1-step inference (no diffusion iteration)

### Architecture

```
Llama Hidden States [B, T, 4096]
          ↓
  StatisticalNormalizer (Llama → Mistral distribution)
          ↓
  PerceiverResampler [B, 128, 4096]
          ↓
  VectorQuantizer (4096 codebook entries)
          ↓
    Quantized Soft Tokens [B, 128, 4096]
          ↓
  Output Scale (match Mistral RMS)
          ↓
    Mistral inputs_embeds
```

### Training

**Loss = LM Loss + λ × VQ Loss**

- **LM Loss**: Cross-entropy on answer generation (functional correctness)
- **VQ Loss**: Codebook loss + commitment loss (discrete bottleneck)
- **Target**: ANSWER (Mistral generates answer, not question)

The VQ loss has two components:
```python
# Codebook loss: pull codebook toward encoder outputs
q_latent_loss = MSE(quantized, encoder_output.detach())

# Commitment loss: pull encoder toward codebook
e_latent_loss = MSE(quantized.detach(), encoder_output)

vq_loss = q_latent_loss + 0.25 * e_latent_loss
```

Gradient flows via **Straight-Through Estimator**:
```python
quantized = encoder_output + (quantized - encoder_output).detach()
```

### Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| LM Loss | Decreasing | Bridge is learning |
| Perplexity | > 100 | Many codebook entries used |
| Output Diversity | > 50% | Not collapsed |
| Entity Transfer | **> 30%** | VQ preserved entities |

### If VQ Fails

If VQ-Telepathy also fails:
- The task (Llama → Mistral) may be fundamentally too hard
- Consider: Larger codebook, more training, hierarchical VQ
- Or: Task is impossible with current adapter size

### Files

| File | Purpose |
|------|---------|
| `latent_bridge_v15.py` | VQ-Telepathy bridge architecture |
| `train_telepathy_v15.py` | Training with LM + VQ loss |
| `eval_telepathy_v15.py` | Evaluation with entity tracking |
| `run_telepathy_v15.sh` | Execution script |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v15.sh
```

---

## 47. Phase 15 Implementation Issues & Fixes

### Issue 1: Tensor Stride Error
**Error:** `RuntimeError: view size is not compatible with input tensor's size and stride`

**Cause:** Perceiver attention outputs are non-contiguous tensors.

**Fix:** Added `.contiguous()` before `.view()` in VectorQuantizer:
```python
flat_input = inputs.contiguous().view(-1, self.embedding_dim)
```

### Issue 2: OOM (Out of Memory)
**Error:** `torch.OutOfMemoryError: CUDA out of memory`

**Cause:** DDP loads both Llama (16GB) and Mistral (14GB) on each GPU. With batch_size=8, activations exceed 80GB.

**Fix:** Reduced batch size + gradient accumulation:
- `batch_size`: 8 → 2
- `grad_accum`: 4 (effective batch = 2 × 4 = 8)

### Issue 3: VQ Loss Explosion → NaN
**Symptom:** VQ Loss exploded from 7 → 3548 → NaN in 48 steps.

**Cause:**
- Codebook initialized with tiny values (`±0.0002`)
- Perceiver outputs have much larger magnitude
- Huge MSE between them → loss explosion

**Fix:**
1. Added `LayerNorm` before VQ to normalize inputs
2. Changed codebook initialization to unit-normalized vectors:
```python
self.embedding.weight.data.normal_(0, 1)
self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)
```

### Issue 4: Codebook Collapse (Perplexity = 1)
**Symptom:** Only 1-2 codes used out of 4096. VQ Loss constant at 1.249.

**Cause:** L2 distance in high dimensions causes all inputs to map to same nearest code.

**Fix:** Changed from L2 distance to **cosine similarity**:
```python
# L2 normalize both for cosine similarity
flat_input_norm = F.normalize(flat_input, dim=1)
codebook_norm = F.normalize(codebook, dim=1)

# Use negative cosine similarity as distance
similarity = torch.matmul(flat_input_norm, codebook_norm.t())
distances = -similarity
```

**Expected Result:** Perplexity should increase to 100+ (many codes used).

### Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 2 | Reduced for OOM |
| grad_accum | 4 | Effective batch = 8 |
| soft_tokens | 128 | Latent sequence length |
| codebook_size | 4096 | VQ vocabulary |
| commitment_cost | 0.25 | VQ hyperparameter |
| lr | 2e-4 | With cosine + warmup |
| warmup_steps | 100 | LR warmup |
| steps | 3000 | Total training steps |

### Issue 5: Persistent Codebook Collapse Despite Cosine Similarity
**Symptom:** Perplexity collapsed from 12 → 1 within 10 steps, even after removing LayerNorm.

**Root Cause:** Encoder outputs not diverse enough. All inputs map to similar directions on unit sphere.

**Fix:** Added **entropy bonus** to VQ loss:
```python
entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
entropy_bonus = entropy / max_entropy  # Normalized [0, 1]
loss = loss - 0.1 * entropy_bonus  # Subtract to maximize entropy
```

**If entropy bonus fails:** Pivot to Finite Scalar Quantization (FSQ) - no codebook = no collapse.

### Training Verification Checklist

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| LM Loss | Decreasing over time | Stuck or increasing |
| VQ Loss | Varying, can be negative | Constant positive |
| Perplexity | 100+ (many codes) | 1-10 (collapse) |
| Outputs | Coherent text | Repetitive garbage |

---

## 48. Phase 15 Pivot: VQ → FSQ (Finite Scalar Quantization)

### Why VQ Failed

Despite multiple fixes, VQ consistently collapsed to perplexity=1:

| Attempt | Fix Applied | Result |
|---------|-------------|--------|
| 1 | L2 distance | Collapse to 1 code |
| 2 | Cosine similarity | Still collapsed |
| 3 | Removed LayerNorm | ppl started at 12, collapsed to 1 within 10 steps |
| 4 | Entropy bonus (0.1 weight) | Still collapsed by step 16 |

**Root Cause:** The entropy bonus (0.1) was too weak compared to LM loss (1.5-4.0). The LM loss gradient dominated, and all vectors collapsed to the same codebook entry.

**Additional Problem:** Periodic LM loss spikes (8-17) indicated catastrophic mismatch when the single used code didn't fit certain inputs.

### The FSQ Solution

Finite Scalar Quantization (Google Research, 2023) eliminates codebook collapse by removing the codebook entirely:

**Key Differences:**
| Aspect | VQ | FSQ |
|--------|-----|-----|
| Codebook | Learned embeddings | None |
| Collapse | Possible (and happened) | Impossible |
| Aux Loss | Commitment + codebook loss | None needed |
| Codes | 4096 explicit | 8^8 = 16M implicit |

**FSQ Architecture:**
```
4096-dim → Project(8) → Quantize each dim to 8 levels → Project(4096)
```

Each of the 8 dimensions is independently quantized to values in {-1, -0.71, -0.43, -0.14, 0.14, 0.43, 0.71, 1}. The effective codebook size is 8^8 = 16,777,216 codes.

### Implementation Details

**FSQ Class:**
```python
class FSQ(nn.Module):
    def __init__(self, levels=[8,8,8,8,8,8,8,8], input_dim=4096):
        self.proj_down = nn.Linear(input_dim, len(levels))  # 4096 → 8
        self.proj_up = nn.Linear(len(levels), input_dim)     # 8 → 4096
        
    def forward(self, x):
        z = self.proj_down(x)           # Project down
        z = torch.tanh(z)               # Bound to [-1, 1]
        z = round(z * half_levels) / half_levels  # Quantize (STE)
        return self.proj_up(z)          # Project back up
```

**Key Features:**
1. **No codebook = No collapse**
2. **No auxiliary loss** - Pure LM loss training
3. **Deterministic quantization** - Same input → same output
4. **16M effective codes** - Rich representation capacity

### Training Changes

| Parameter | VQ | FSQ |
|-----------|-----|-----|
| Auxiliary Loss | VQ loss (commitment + codebook) | 0 (none needed) |
| Diversity Metric | Perplexity (should be high) | Diversity ratio (should be high) |
| Expected Collapse | Possible | Impossible |

### Execution

```bash
git pull && rm -rf runs && bash run_telepathy_v15.sh
```

### Success Criteria

| Metric | Expected | If Failed |
|--------|----------|-----------|
| LM Loss | Decreasing | Check gradients |
| Diversity | 0.5-1.0 (high) | Already impossible to collapse |
| Outputs | Coherent answers | Check FSQ projections |

### Reference

"Finite Scalar Quantization: VQ-VAE Made Simple"
https://arxiv.org/abs/2309.15505

---

---

## 49. Phase 15 Issue: FSQ Also Collapsed (8-dim bottleneck too aggressive)

### Symptom

FSQ collapsed to diversity=0 by step 5, same as VQ:

| Step | LM Loss | Diversity | Observation |
|------|---------|-----------|-------------|
| 0-3 | 2.16-2.40 | 0.76-0.84 | Good start |
| 4 | 2.39 | 0.11 | Dropping |
| 5+ | 1.5-2.0 | **0.00** | **COLLAPSED** |

### Root Cause

The problem is NOT VQ vs FSQ. The problem is the **bottleneck compression ratio**.

8-dim FSQ:
```
Perceiver output: [128, 4096] → proj_down → [128, 8] → quantize → [128, 8] → proj_up → [128, 4096]
```

The 4096 → 8 projection (512× compression) throws away 99.8% of information. All 128 tokens project to nearly identical 8-dim vectors, which then quantize to the same code.

### Multi-Level Analysis

1. **Low-level (FSQ mechanics):** FSQ working correctly - quantizing inputs
2. **Medium-level (training dynamics):** LM loss decreasing, but diversity=0
3. **High-level (architecture):** Bottleneck too aggressive - no room for diversity

### Fix: Increase FSQ Dimensions

| Config | Dimensions | Levels | Effective Codes | Compression |
|--------|------------|--------|-----------------|-------------|
| Before | 8 | 8 | 16M | 512× |
| **After** | **32** | **5** | **5^32 ≈ 2.3×10²²** | **128×** |

32 dimensions preserves 4× more information through the bottleneck, giving diversity a better chance to survive quantization.

### Code Change

```python
# Before (collapsed)
fsq_levels = [8, 8, 8, 8, 8, 8, 8, 8]  # 8 dims

# After (fix)
fsq_levels = [5] * 32  # 32 dims, 5 levels each
```

### Expected Outcome

- Diversity should stay above 0.1-0.2 instead of collapsing to 0.00
- LM loss should still decrease
- If 32-dim still collapses, will need to abandon discrete bottleneck entirely

---

## 50. Phase 15 Issue: 32-dim FSQ Still Collapsed - Adding Diversity Loss

### Symptom

32-dim FSQ also collapsed, just slightly slower than 8-dim:

| Step | Diversity | Observation |
|------|-----------|-------------|
| 0-3 | 0.97-0.98 | Healthy |
| 4 | 0.94 | Starting to drop |
| 5 | 0.44 | Rapid decline |
| 6 | 0.28 | Accelerating |
| 7 | 0.01 | **Collapsed** |
| 8+ | 0.00 | Flatlined |

LM loss was decreasing (1.96 → 1.43 → 1.22 → 1.08) but this is meaningless - all 128 soft tokens collapsed to a single code.

### Pattern Recognition

| Approach | Result |
|----------|--------|
| VQ (4096 codes) | Collapsed to perplexity=1 |
| VQ + entropy bonus | Collapsed to perplexity=1 |
| FSQ 8-dim | Collapsed to div=0 by step 5 |
| FSQ 32-dim | Collapsed to div=0 by step 7 |

**All discrete bottleneck approaches collapse.** The common factor is not the implementation - it's that pure LM loss doesn't penalize collapse.

### Root Cause

The gradient signal from LM loss pushes all 128 latent tokens toward a single "safe" representation. The network learns that outputting 128 identical soft tokens minimizes answer prediction loss. Discreteness does not equal diversity.

### Fix: Add Diversity Loss

Added explicit penalty for low diversity:

```python
# Diversity loss: penalize low diversity to prevent collapse
# Log penalty: gentle at high diversity (0.9 → 0.1), harsh at low (0.01 → 4.6)
diversity_loss = -torch.log(diversity + 1e-8)
div_weight = 0.1

total_loss = lm_loss + div_weight * diversity_loss
```

| Diversity | Diversity Loss | Gradient Strength |
|-----------|----------------|-------------------|
| 0.90 | 0.11 | Gentle |
| 0.50 | 0.69 | Moderate |
| 0.10 | 2.30 | Strong |
| 0.01 | 4.61 | Very strong |

### Expected Outcome

- Diversity should stay HIGH (0.5-1.0) due to explicit penalty
- LM loss may be slightly higher (tradeoff with diversity)
- If this still collapses, will abandon discrete bottleneck entirely

---

## 51. Phase 15 Issue: Diversity Loss Had No Gradients - Fixed with Variance

### Symptom

Diversity loss was being computed and logged, but diversity still collapsed:
- Step 50: Div Loss=4.42, Diversity=0.12
- Step 100: Div Loss=5.39, Diversity=0.01
- Step 150: Div Loss=5.55, Diversity=0.00

The loss was increasing as expected, but the network was not responding.

### Root Cause

The diversity loss had **NO GRADIENT PATH** to the FSQ parameters:

```python
# BROKEN - no gradients!
diversity_loss = -torch.log(torch.tensor(diversity + 1e-8, device=device))
```

The `diversity` variable was a Python float computed from `torch.unique()` (non-differentiable). Creating a new tensor from it breaks the computation graph entirely.

### Fix: Differentiable Variance-Based Loss

Replaced non-differentiable diversity counting with variance of quantized values:

```python
# In FSQ.forward():
# z_quantized has gradients via straight-through estimator
z_variance = z_quantized.var(dim=[0, 1]).mean()  # DIFFERENTIABLE!

# In train_step():
diversity_loss = -torch.log(z_variance + 1e-8)  # Now has gradient path!
```

### Why This Works

1. `z_quantized` has gradients via the straight-through estimator (STE)
2. `var()` is a differentiable operation
3. When all tokens collapse to same code → variance = 0 → loss = infinity
4. Gradients now flow: loss → z_variance → z_quantized → proj_down weights

### Expected Outcome

- Z Variance should stay HIGH (> 0.1) due to gradient pressure
- Diversity should follow (high variance → diverse codes)
- LM loss may be slightly higher (tradeoff)

---

## 52. Phase 15 Pivot: Abandon Discrete Bottleneck → Continuous Soft Tokens

### The Evidence

After 6 failed attempts, discrete bottlenecks are proven incompatible with this task:

| # | Approach | Result |
|---|----------|--------|
| 1 | VQ (4096 codes) | Collapsed to perplexity=1 |
| 2 | VQ + entropy bonus | Collapsed to perplexity=1 |
| 3 | FSQ 8-dim | Collapsed to div=0 by step 5 |
| 4 | FSQ 32-dim | Collapsed to div=0 by step 7 |
| 5 | FSQ + diversity loss (non-diff) | Collapsed to div=0 |
| 6 | FSQ + variance loss (diff) | Collapsed to z_var=0 by step 100 |

### Root Cause Analysis

The discrete bottleneck creates **basins of attraction** that gradients cannot escape:

1. LM loss gradient pushes all 128 tokens toward a single "safe" representation
2. Discrete quantization rounds small perturbations back to the same bin
3. Even infinite gradient pressure (from variance loss) cannot overcome discrete rounding
4. Once collapsed, the network is stuck in a local minimum

### The Solution: Continuous Soft Tokens

Remove FSQ entirely. Use Perceiver resampler output directly:

```
Llama Hidden → Normalizer → Perceiver → Scale → Mistral
                              ↓
                    [128, 4096] continuous
                    (no quantization)
```

### Architecture Change

```python
# Before (FSQ - collapsed)
quantized, aux_loss, diversity, z_variance = self.fsq(compressed)
out = quantized * self.output_scale

# After (Continuous)
out = compressed * self.output_scale  # Direct from Perceiver
```

### Expected Outcome

- NO collapse possible (continuous values)
- LM loss should decrease steadily
- Quality depends on Perceiver's compression ability
- May see "blurry" outputs (original concern), but better than collapsed

### Potential Concern

The original motivation for discrete bottleneck was to prevent "blurry averages" from continuous regression. If continuous soft tokens produce blurry/generic outputs, we may need a different approach (e.g., contrastive learning, multiple hypotheses).

But first, let's establish if continuous works at all.

---

## 53. Phase 15 Fix: Training Instability → Add tanh Bounding

### Symptom

Continuous mode training showed catastrophic instability:

```
Step 0-90:  lm = 4.59 → 1.14 (healthy learning)
Step 91:    lm = 8.89  (8× SPIKE)
Step 92:    lm = 15.37 (continues spiking)
Step 100:   lm = 3.97, Z_var = 0.2038 (13× variance increase!)
Step 126:   lm = 8.17 → 12.58 (re-crash)
```

Key observation: Z Variance jumped 13× (0.0155 → 0.2038) = Perceiver outputs exploding.

### Root Cause

V15 continuous mode lacked the output bounding that V7 had:

```python
# V7 (stable): Bounded to [-scale, +scale]
return torch.tanh(compressed) * self.output_scale

# V15 continuous (unstable): UNBOUNDED
out = quantized * self.output_scale  # compressed can grow arbitrarily large!
```

Without tanh bounding, a feedback loop occurs:
1. Large gradients → Perceiver weights change → Larger outputs
2. Larger outputs → Mistral attention saturates → Worse loss
3. Worse loss → Larger gradients → Repeat

### Fix

Add tanh bounding to continuous mode output:

```python
# Before (unstable)
out = quantized * self.output_scale

# After (stable)
out = torch.tanh(quantized) * self.output_scale
```

This bounds output to `[-0.0027, +0.0027]` matching Mistral's expected embedding range.

### Expected Outcome

- NO more loss spikes (bounded outputs)
- Z Variance should stay stable (tanh prevents magnitude explosion)
- LM loss should decrease steadily throughout training
- May still see "blurry" outputs (continuous limitation), but training should be stable

### Alternative Options Considered

| Option | Description | Why Not Chosen |
|--------|-------------|----------------|
| Lower LR | 2e-4 → 5e-5 | Slower, may still be unstable |
| Stronger clip | grad_clip 0.1 | May slow learning |
| LayerNorm | Normalize outputs | Less proven than tanh |

tanh chosen because it's proven in V7 and simplest fix.

---

## 54. Phase 15 Fix: tanh Saturation → RMS Normalization

### Symptom

After adding tanh bounding (Section 53), training was stable but loss PLATEAUED:

| Step | LM Loss | Z Variance | Notes |
|------|---------|------------|-------|
| 50 | 1.375 | 0.017 | Healthy |
| 100 | 1.102 | 0.126 | Good |
| 200 | 1.078 | 19.1 | Variance exploding |
| 500 | 1.058 | 207.9 | Plateaued |
| 1400 | 1.054 | 208.9 | Stuck at ~1.05 |

Z Variance exploded 10,000× (0.017 → 208) while loss stayed flat.

### Root Cause: tanh Saturation

The Perceiver learned to "game" tanh:
1. Perceiver outputs grow unboundedly (variance → 200, std → 14)
2. tanh(14) ≈ 1.0 (saturated)
3. Output becomes essentially binary (±0.0027)
4. tanh gradient at saturation: sech²(14) ≈ 0 (no learning)
5. Loss plateaus because gradients vanish

**Analogy**: Like an audio amplifier driven into clipping - only outputs ±V_max.

### Why This Is New

Previous uses of tanh (V3, V7) worked because:
- Perceiver outputs were naturally in reasonable range
- No incentive to grow unboundedly

Current continuous mode:
- No FSQ to constrain values
- Perceiver discovered it can output ANY magnitude
- tanh acts as "free normalization" the network exploits

### Fix: RMS Normalization

Replace tanh with RMS-based normalization:

```python
# Before (saturates):
out = torch.tanh(quantized) * self.output_scale

# After (preserves structure):
rms = torch.sqrt((quantized ** 2).mean(dim=-1, keepdim=True) + 1e-8)
out = (quantized / rms) * self.output_scale
```

### Why RMS Works

1. **Self-correcting**: If variance grows, RMS grows, output stays bounded
2. **Preserves structure**: Unlike saturated tanh, relative differences maintained
3. **Gradients everywhere**: No saturation region with zero gradients
4. **Proven**: RMSNorm is used in Llama and other modern transformers

### Expected Outcome

- Z Variance may still grow, but output magnitude stays bounded
- LM Loss should continue decreasing (gradients flow)
- Output preserves semantic structure (not binary)

---

## Section 55: Periodic Evaluation During Training

**Date**: 2025-12-02
**Status**: Implemented

### The Problem

RMS normalization fix shows loss continuing to decrease (0.877 at step 650), but:
- Low loss ≠ good outputs
- Need early feedback on output quality
- Waiting until end of training wastes compute if outputs are garbage

### Solution: Quick Eval Every 500 Steps

Added periodic evaluation during training:

```python
# New arguments
--eval_every 500    # Run quick eval every N steps
--eval_samples 10   # Samples per eval

# In training loop
if (step + 1) % args.eval_every == 0:
    quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, device, args, step + 1)
```

### What Quick Eval Measures

1. **Accuracy**: Does predicted answer match ground truth?
2. **Entity Transfer**: Do numbers from question appear in output?
3. **Output Diversity**: Are outputs unique or collapsed?
4. **Visual Samples**: First 3 outputs printed for inspection

### Implementation Details

- Uses test split of GSM8K (not train)
- Sets bridge to eval mode, then back to train mode
- Only runs on rank 0 (no DDP sync needed)
- Generates with `do_sample=False` for deterministic comparison

### Also Fixed

Eval script (`eval_telepathy_v15.py`) was unpacking 3 return values but bridge returns 4:
```python
# Before (broken):
soft_tokens, vq_loss, perplexity = bridge(src_h, src_mask)

# After (fixed):
soft_tokens, aux_loss, diversity, z_variance = bridge(src_h, src_mask)
```

---

## Section 56: Mode Collapse and Batch Diversity Loss

**Date**: 2025-12-02
**Status**: Fix Implemented

### The Problem: Mode Collapse

Training with RMS normalization showed good loss (0.916 at step 500), but quick eval revealed catastrophic failure:

```
Q: Janet's ducks lay 16 eggs per day...
Output: #1. The number of students in the class is 100 * 1.5 = <<100*1.5=150>>150

Q: A new program had 60 downloads...
Output: #1. The number of students in the class is 100 * 1.5 = <<100*1.5=150>>150

Q: I have 10 liters of orange drink...
Output: #1. The number of students in the class is 100 * 1.5 = <<100*1.5=150>>150
```

**ALL 10 samples produced nearly identical output!**

Metrics:
- Accuracy: 0/10 (0.0%)
- Entity Transfer: 7/29 (24.1%) - coincidental
- Output Diversity: 1/10 (10.0%) - **COLLAPSED**

### Root Cause Analysis

RMS normalization preserves **direction** but removes **magnitude**:

```python
out = (x / ||x||_rms) * scale
```

The Perceiver discovered it could achieve low average loss by:
1. Outputting vectors that differ mainly in magnitude (not direction)
2. RMS normalization collapses them to the same point on the unit hypersphere
3. Result: single "average" representation that works reasonably well on average

**Physics Analogy**: Points at (1,1,1) and (10,10,10) both project to the same point on the unit sphere: (0.577, 0.577, 0.577).

**Information Theory**: The bridge found a shortcut - maximum entropy solution when differentiation isn't required by the loss.

### The Fix: Batch Diversity Loss

Added contrastive-style loss that penalizes high cosine similarity between different questions in the same batch:

```python
# Flatten soft tokens: [B, K, D] -> [B, K*D]
flat_tokens = soft_tokens.view(B, -1).float()
# Normalize for cosine similarity
flat_norm = F.normalize(flat_tokens, dim=1)
# Compute similarity matrix [B, B]
sim_matrix = torch.mm(flat_norm, flat_norm.t())
# Get off-diagonal similarities
mask = ~torch.eye(B, dtype=torch.bool, device=device)
off_diag_sim = sim_matrix[mask].mean()
# Diversity loss: penalize high similarity
batch_div_loss = off_diag_sim
```

Total loss becomes:
```python
total_loss = loss_lm + diversity_weight * batch_div_loss
```

Default `diversity_weight = 0.1`.

### Why This Should Work

1. **Direct signal**: Different inputs must produce different outputs
2. **Batch-level**: No need for large contrastive bank
3. **Cosine similarity**: Direction-sensitive, immune to magnitude normalization
4. **Gradient flow**: Smooth, differentiable loss

### Expected Outcome

- Batch similarity should decrease from ~1.0 to < 0.5
- Output diversity should increase from 10% to > 80%
- Entity transfer should improve as outputs become input-specific
- LM loss may increase slightly (tradeoff for diversity)

### Monitoring

New metrics in training logs:
- `Div Loss`: The diversity penalty (want this low after initial push)
- `Batch Sim`: Average cosine similarity between batch items (want < 0.5, collapse if ~1.0)

---
