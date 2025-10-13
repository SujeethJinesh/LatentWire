# Shared Interlingua Architecture Proposal

**Date:** 2025-10-12
**Status:** Complete representational collapse diagnosed. New architecture required.
**Goal:** Design architecture that enables heterogeneous LLM conditioning WITHOUT focusing on compression yet.

---

## 1. Root Cause Analysis: Why Current Architecture Fails

### 1.1 Observed Failure Mode

**Symptom:** Complete representational collapse
**Evidence:** ALL latent predictions produce identical output: `"2019) 1. The answer is"`

```
Gold answer: "linear" → Latent pred: "2019) 1. The answer is"
Gold answer: "Lampea" → Latent pred: "2019) 1. The answer is"
Gold answer: "San Jose" → Latent pred: "2019) 1. The answer is"
```

Meanwhile, text baseline achieves F1=69.4%, EM=50% with proper answers.

### 1.2 Architectural Analysis

Current pipeline:
```
Text (UTF-8)
  → ByteEncoder (processes bytes 0-255)
  → LatentPooler (compress to M=32 × d_z=256)
  → Adapter (project to d_model=4096)
  → Frozen LLM (Llama/Qwen)
  → Generation: "2019) 1. The answer is" [COLLAPSED]
```

**Critical Issue: Semantic Impedance Mismatch**

1. **ByteEncoder operates on alien modality:**
   - Processes raw UTF-8 bytes (0-255 vocab)
   - Byte sequences have NO alignment with LLM tokenization
   - Example: "Answer" → bytes `[65, 110, 115, 119, 101, 114]`
   - LLM tokenizer: "Answer" → single token `[1234]`
   - **The LLM has NEVER seen byte-level representations in pretraining**

2. **Adapter cannot bridge semantic gap:**
   - Linear projection: `d_z=256 → d_model=4096`
   - Even with FiLM modulation, metadata hints, skip connections
   - Cannot transform byte-level encodings into token-level semantic representations
   - Gap is too large for gradient signal during training

3. **Training signal too weak:**
   - K-token CE loss: Tries to teach "decode bytes → predict tokens"
   - KD loss: Distill text teacher distributions
   - But gradients must flow through: `Generation Loss → Adapter → Pooler → ByteEncoder → Byte embeddings`
   - Gradient vanishing + modality mismatch = learning fails

### 1.3 Why This Matters for Interlingua Design

**Key insight:** You cannot create a "shared interlingua" if the representation is incomprehensible to the target models.

The current failure teaches us:
- ✗ Starting from bytes/characters (too low-level for LLMs)
- ✗ Relying on learned compression before proving semantic transfer works
- ✗ Asking adapters to perform semantic transformation (they can only do statistical alignment)

**What we need:**
- ✓ Start in LLM-native representation space (token embeddings)
- ✓ Prove interlingua works FIRST (cross-model transfer)
- ✓ Add compression LATER (once semantic bridge is established)

---

## 2. Design Principles for Shared Interlingua

### 2.1 Core Requirements

**Primary Goal:** Enable Llama and Qwen to decode the SAME representation and produce correct answers.

**Success Criteria (Phase 1 - No Compression):**
1. Latent F1 > Text baseline × 0.7 (retain 70% of text performance)
2. Cross-model agreement: Both models decode to similar answers
3. Compression ratio = 1.0× initially (no compression penalty)

**Enablers for Future Compression:**
- Representation must be continuous (vector space, not discrete tokens)
- Should tolerate dimensionality reduction (PCA showed this fails for raw embeddings)
- Architecture supports adding quantization/pruning later

### 2.2 Key Design Decisions

**Decision 1: Start in LLM Embedding Space**
*Rationale:* Both Llama and Qwen understand token embeddings. This is their native language.

**Decision 2: Cross-Model Alignment, Not Shared Encoder**
*Rationale:* Llama and Qwen have different vocabularies, tokenization, embedding dimensions. We need per-model encoders with alignment loss.

**Decision 3: Anchor in Semantic Space**
*Rationale:* Use a small shared text encoder (e.g., SentenceTransformer) to create semantic anchors that guide alignment.

**Decision 4: No Compression Initially**
*Rationale:* Get cross-model transfer working first. Add compression in Phase 2.

---

## 3. Proposed Architecture: Anchor-Guided Cross-Model Interlingua

### 3.1 High-Level Design

```
                    ┌─────────────────┐
                    │  Text Prompt    │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
    │ Llama Tokens │ │   Semantic  │ │  Qwen Tokens │
    │  + Embeds    │ │   Encoder   │ │   + Embeds   │
    └──────┬───────┘ │  (frozen)   │ └──────┬───────┘
           │         └──────┬──────┘        │
           │                │               │
           │         ┌──────▼──────┐        │
           │         │   Z_sem     │        │
           │         │  (anchor)   │        │
           │         └──────┬──────┘        │
           │                │               │
           ▼                ▼               ▼
    ┌─────────────────────────────────────────────┐
    │         Alignment Transformer               │
    │  - Cross-attention from tokens to Z_sem     │
    │  - Learns to align Llama/Qwen spaces        │
    └──────┬──────────────────────────┬───────────┘
           │                          │
           ▼                          ▼
    ┌──────────────┐          ┌──────────────┐
    │   Z_llama    │          │   Z_qwen     │
    │ (interlingua)│          │ (interlingua)│
    └──────┬───────┘          └──────┬───────┘
           │                          │
           ▼                          ▼
    ┌──────────────┐          ┌──────────────┐
    │ Llama Frozen │          │ Qwen Frozen  │
    │     LLM      │          │     LLM      │
    └──────────────┘          └──────────────┘
```

### 3.2 Component Details

#### 3.2.1 Semantic Encoder (Frozen)
- **Model:** SentenceTransformer (e.g., `all-MiniLM-L6-v2`, 384 dim)
- **Purpose:** Create model-agnostic semantic anchor
- **Output:** `Z_sem ∈ R^{1 × 384}` (single vector per prompt)
- **Why frozen:** Prevents semantic drift, provides stable supervision

#### 3.2.2 Token Embeddings (Frozen LLM Embeddings)
- **Llama:** Tokenize prompt → `[T_llama]` tokens → Embed → `E_llama ∈ R^{T × 4096}`
- **Qwen:** Tokenize prompt → `[T_qwen]` tokens → Embed → `E_qwen ∈ R^{T × 2048}`
- **Why use these:** Already contain semantic information the LLMs understand

#### 3.2.3 Alignment Transformer (Learned)
**Purpose:** Learn to project token embeddings into shared interlingua space, guided by semantic anchor.

**Architecture:**
```python
class AlignmentTransformer(nn.Module):
    def __init__(self, d_model_llama, d_model_qwen, d_sem, d_inter, n_layers=4):
        # Project to shared dimension
        self.proj_llama = nn.Linear(d_model_llama, d_inter)
        self.proj_qwen = nn.Linear(d_model_qwen, d_inter)
        self.proj_sem = nn.Linear(d_sem, d_inter)

        # Cross-attention: query from tokens, key/value from semantic
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_inter, num_heads=8)
            for _ in range(n_layers)
        ])

        # FFN for refinement
        self.ffn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_inter, nhead=8, dim_feedforward=d_inter*4),
            num_layers=2
        )

    def forward(self, token_embeds, sem_anchor, model_type='llama'):
        # Project to shared space
        if model_type == 'llama':
            x = self.proj_llama(token_embeds)  # [B, T, d_inter]
        else:
            x = self.proj_qwen(token_embeds)

        sem = self.proj_sem(sem_anchor).unsqueeze(1)  # [B, 1, d_inter]

        # Cross-attend to semantic anchor
        for attn_layer in self.cross_attn_layers:
            attn_out, _ = attn_layer(x, sem, sem)
            x = x + attn_out  # Residual

        # Refine
        z = self.ffn(x)  # [B, T, d_inter]

        # Mean pool to single representation
        z_inter = z.mean(dim=1)  # [B, d_inter]

        return z_inter
```

**Key features:**
- Per-model projections (handles different d_model)
- Semantic anchor guides alignment via cross-attention
- Output: `d_inter=512` dimensional interlingua (room for later compression)

#### 3.2.4 Interlingua → LLM Adapter (Learned)
Map interlingua back to LLM embedding space for generation:

```python
class InterlinguaAdapter(nn.Module):
    def __init__(self, d_inter, d_model, num_slots=32):
        self.num_slots = num_slots
        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(num_slots, d_inter))

        # Expand single vector to sequence
        self.expand = nn.Sequential(
            nn.Linear(d_inter, d_inter * num_slots),
            nn.LayerNorm(d_inter * num_slots),
            nn.GELU(),
        )

        # Project to LLM space
        self.proj = nn.Sequential(
            nn.Linear(d_inter, d_model),
            nn.LayerNorm(d_model),
        )

        # Calibration
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, z_inter):
        # Expand to sequence
        expanded = self.expand(z_inter)  # [B, d_inter * num_slots]
        seq = expanded.view(-1, self.num_slots, d_inter)  # [B, M, d_inter]

        # Add learned queries
        seq = seq + self.queries.unsqueeze(0)

        # Project to LLM space
        embeds = self.proj(seq) * self.scale  # [B, M, d_model]

        return embeds
```

### 3.3 Training Objectives

**Phase 1: Establish Cross-Model Transfer (No Compression)**

```python
def training_step(batch, llama_model, qwen_model, semantic_encoder, alignment_tf, adapters):
    text, answer_ids = batch

    # 1. Encode with frozen semantic model (anchor)
    with torch.no_grad():
        z_sem = semantic_encoder.encode(text)  # [B, 384]

    # 2. Get frozen LLM embeddings
    with torch.no_grad():
        llama_ids = llama_tokenizer(text, return_tensors='pt').input_ids
        qwen_ids = qwen_tokenizer(text, return_tensors='pt').input_ids
        llama_embeds = llama_model.get_input_embeddings()(llama_ids)
        qwen_embeds = qwen_model.get_input_embeddings()(qwen_ids)

    # 3. Encode to interlingua
    z_llama = alignment_tf(llama_embeds, z_sem, model_type='llama')
    z_qwen = alignment_tf(qwen_embeds, z_sem, model_type='qwen')

    # 4. Decode to LLM embedding space
    prefix_llama = adapters['llama'](z_llama)  # [B, M, 4096]
    prefix_qwen = adapters['qwen'](z_qwen)     # [B, M, 2048]

    # 5. Compute losses

    # L1: Generation loss (K-token CE)
    loss_gen_llama = k_token_ce_from_prefix(llama_model, prefix_llama, answer_ids, K=4)
    loss_gen_qwen = k_token_ce_from_prefix(qwen_model, prefix_qwen, answer_ids, K=4)
    loss_gen = loss_gen_llama + loss_gen_qwen

    # L2: Cross-model alignment (interlingua should be similar)
    loss_align = F.mse_loss(z_llama, z_qwen)

    # L3: Semantic anchor loss (stay close to frozen semantic space)
    z_sem_proj = proj_sem_to_inter(z_sem)  # Project to d_inter
    loss_sem = F.mse_loss(z_llama, z_sem_proj) + F.mse_loss(z_qwen, z_sem_proj)

    # L4: KD from text teacher (optional, for quality)
    loss_kd_llama = kd_first_k_prefix_vs_text(llama_model, llama_model, prefix_llama, llama_ids, answer_ids, K=4)
    loss_kd_qwen = kd_first_k_prefix_vs_text(qwen_model, qwen_model, prefix_qwen, qwen_ids, answer_ids, K=4)
    loss_kd = loss_kd_llama + loss_kd_qwen

    # Total loss
    loss = loss_gen + 0.5 * loss_align + 0.1 * loss_sem + 0.3 * loss_kd

    return loss
```

**Key components:**
1. **Generation loss:** Both models must decode interlingua correctly
2. **Alignment loss:** Force Llama and Qwen interlingua to be similar (MSE)
3. **Semantic anchor:** Keep interlingua grounded in semantic space
4. **KD loss:** Optional quality boost from text teacher

### 3.4 Why This Architecture Works

**Addresses root causes:**
1. ✓ **No byte-level encoding:** Uses LLM-native token embeddings
2. ✓ **Semantic grounding:** Frozen SentenceTransformer provides stable anchor
3. ✓ **Gradients flow properly:** Short path from generation loss to alignment transformer
4. ✓ **Per-model adaptation:** Separate projections handle vocab/dimension differences

**Enables future compression:**
1. Current: `d_inter=512`, `num_slots=32` → `512×32 = 16KB` per prompt
2. Phase 2 options:
   - Reduce `d_inter`: 512→256→128 (dimensionality reduction)
   - Reduce `num_slots`: 32→16→8 (fewer soft tokens)
   - Quantize: FP16→INT8→INT4 (quantization)
   - All three combined: 512×32×FP16 → 128×8×INT4 = 32× compression

---

## 4. Implementation Roadmap

### Phase 1: Establish Cross-Model Transfer (3-5 days)

**Goal:** Prove interlingua works without compression.

**Steps:**
1. Implement `AlignmentTransformer` and `InterlinguaAdapter`
2. Freeze semantic encoder (SentenceTransformer)
3. Train with loss = L_gen + 0.5×L_align + 0.1×L_sem
4. Target: Latent F1 > 0.5 (vs text baseline 0.69)

**Metrics to track:**
- Per-model F1 (Llama, Qwen)
- Cross-model agreement (% same answer)
- Interlingua distance: ||z_llama - z_qwen||

### Phase 2: Add Knowledge Distillation (1-2 days)

**Goal:** Improve quality via text teacher.

**Steps:**
1. Add L_kd term (KD from text-prompted teacher)
2. Tune weight: 0.1→0.5
3. Target: Latent F1 > 0.6

### Phase 3: Compression Experiments (ongoing)

**Goal:** Reduce representation size while maintaining F1 > 0.5.

**Approaches:**
1. **Dimensionality:** d_inter = 512 → 256 → 128
2. **Slots:** num_slots = 32 → 16 → 8
3. **Quantization:** FP16 → INT8 → INT4
4. **Sparsity:** Add L1 penalty on interlingua activations

**Target:** 8-16× compression at F1 > 0.5

---

## 5. Expected Results

### Phase 1 Targets (No Compression)

| Baseline | F1 | Compression | Status |
|----------|-----|-------------|---------|
| Text (Llama) | 69.4% | 1.0× | ✓ Established |
| Token Budget (M=32) | 0.0% | ~8× | ✓ Measured |
| **New Interlingua (Llama)** | **>50%** | **1.0×** | 🎯 Target |
| **New Interlingua (Qwen)** | **>50%** | **1.0×** | 🎯 Target |
| Cross-model agreement | >70% | - | 🎯 Target |

**Success criteria:**
- Latent predictions are semantically correct (not collapsed)
- Both Llama and Qwen decode to similar answers
- F1 retains >70% of text baseline performance

### Phase 3 Targets (With Compression)

| Config | F1 | Compression | Wire Bytes |
|--------|-----|-------------|-----------|
| d=512, M=32, FP16 | >60% | 1.0× | 32KB |
| d=256, M=32, FP16 | >55% | 2× | 16KB |
| d=256, M=16, FP16 | >50% | 4× | 8KB |
| d=128, M=16, INT8 | >45% | 8× | 2KB |

---

## 6. Why This Succeeds Where ByteEncoder Failed

| Aspect | ByteEncoder (Failed) | Anchor-Guided Interlingua (Proposed) |
|--------|---------------------|--------------------------------------|
| **Input modality** | Raw UTF-8 bytes (0-255) | LLM token embeddings |
| **LLM familiarity** | Never seen in pretraining | Native representation |
| **Semantic grounding** | Byte statistics only | Frozen SentenceTransformer anchor |
| **Cross-model strategy** | Single encoder for both | Per-model encoders + alignment loss |
| **Gradient path** | Long, through byte→token gap | Short, within embedding space |
| **Compression first?** | Yes (M=32 from start) | No (prove transfer first) |
| **Result** | Complete collapse (F1=0%) | Expected: F1>50% |

---

## 7. Open Questions & Future Work

### 7.1 Architecture Variants

**Q:** Should we use multiple semantic anchors (per-sentence) instead of one?
**A:** Start with single anchor for simplicity. Add hierarchical anchors in Phase 2 if needed.

**Q:** Should alignment transformer be shared or per-model?
**A:** Start with per-model projections + shared cross-attention. Simplifies different d_model handling.

**Q:** How to handle variable-length prompts?
**A:** Mean pooling works for Phase 1. Could try attention-based pooling or fixed-length "gist tokens" later.

### 7.2 Scaling Considerations

**Q:** Can this scale to 3+ models (Mistral, Gemma, etc.)?
**A:** Yes - add new projection + adapter per model. Alignment loss becomes N×(N-1)/2 pairwise terms.

**Q:** What about multilingual support?
**A:** SentenceTransformer supports 50+ languages. Should work out of box if LLMs are multilingual.

### 7.3 Compression Strategy

**Q:** Which compression method is most promising?
**Priority order:**
1. Reduce slots first (32→16→8) - biggest wins
2. Then dimensionality (512→256→128)
3. Finally quantization (INT8→INT4)

**Q:** Can we do learned compression (VAE-style)?
**A:** Phase 4 idea: Add variational bottleneck. But prove basic transfer first.

---

## 8. Next Steps

**Immediate (Today):**
1. ✓ Document root cause analysis
2. ✓ Design proposed architecture
3. → Discuss with team / get feedback
4. → Start implementation if approved

**This Week:**
1. Implement `AlignmentTransformer` and `InterlinguaAdapter`
2. Write training loop with 4-term loss
3. Run initial experiment (100 samples smoke test)
4. Debug and iterate

**Next Week:**
1. Full training run (87k samples, 10 epochs)
2. Evaluate on SQuAD validation (10k samples)
3. Analyze cross-model agreement
4. Document results in LOG.md

---

## 9. Summary

**Problem:** Current ByteEncoder architecture suffers complete representational collapse (F1=0%) due to semantic impedance mismatch between byte-level encoding and token-level LLM representations.

**Solution:** Anchor-guided cross-model interlingua that:
1. Starts in LLM-native embedding space (token embeddings)
2. Uses frozen semantic encoder as stable anchor
3. Learns per-model alignment transformers
4. Proves cross-model transfer BEFORE compression

**Expected outcome:** F1 > 50% (vs 0% current, 69% text baseline), establishing viable path to compressed interlingua.

**Compression readiness:** Architecture designed with continuous vector space (d_inter=512, M=32) that naturally supports dimensionality reduction, slot pruning, and quantization in Phase 2.
