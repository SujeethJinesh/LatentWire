# Cross-Model Translation Log

**Experiment**: Cross-attention based interlingua for translating between heterogeneous LLMs (Mistral-7B → Llama-3.1-8B)
**Dataset**: GSM8K (math word problems)
**Goal**: Learn a translator that converts Mistral hidden states into soft tokens that condition Llama to generate correct answers

---

## Problem Statement

Training a bottlenecked gated cross-attention translator (BottleneckedGatedTranslator) that:
- Takes Mistral-7B hidden states as input
- Produces 48 soft tokens in Llama-3.1-8B embedding space
- Uses 6 layers of gated cross-attention with 1024 bottleneck dimension
- Trains only the translator (~143M params); both LLMs frozen

**Expected behavior**: Bridged accuracy should approach target-alone baseline (73%) as training progresses.

**Observed behavior**: Bridged accuracy peaks at 25.5% around step 750, then catastrophically collapses to 4% by step 2000.

---

## Initial Analysis (November 6, 2025)

### Performance Trajectory - The "Peak and Collapse" Pattern

```
Step  250: Bridged acc: 23.0%  (promising start)
Step  500: Bridged acc: 14.0%
Step  750: Bridged acc: 25.5%  (PEAK - best performance)
Step 1000: Bridged acc: 18.5%  (beginning decline)
Step 1250: Bridged acc:  9.0%  (rapid degradation)
Step 1500: Bridged acc:  2.5%  (catastrophic collapse)
Step 1750: Bridged acc:  6.5%
Step 2000: Bridged acc:  4.0%  (near-total failure)
```

Meanwhile, target-alone baseline remained stable at 73% throughout, confirming the frozen LLM is healthy.

### Degenerate Output Patterns Observed

After the peak (~750 steps), the translator produces pathological generations:

1. **Infinite zero loops**: `000000000000000000000000...` (hundreds of zeros)
2. **Special character repetition**: `################...` or `$ $ $ $ $ $ $...`
3. **Word repetition**: `bolts bolts bolts bolts bolts...`
4. **Empty generations**: No output at all, just whitespace
5. **Single-digit loops**: `2>>2>>2>>2>>2>>2>>2>>2...` or `3>>3>>3>>3>>3...`

### Claude's Initial Diagnosis

**Root Cause 1: Scale & Distribution Mismatch**
- The translator outputs soft tokens directly into the target LLM's embedding space
- If RMS scale or distribution drifts from the target's native embedding statistics, early decoding steps go low-entropy
- This causes "logit collapse" - the model snaps into repetitive token loops
- Classic failure mode when injecting learned prompts without normalization

**Root Cause 2: Gating Mechanism Issues**
- The `GatedCrossAttentionBlock` uses a single learnable gate: `self.cross_gate = nn.Parameter(torch.zeros(1))`
- Starting at zero means information flow is initially blocked
- Once the gate opens without proper constraints, soft tokens can dominate and destabilize early decoding
- Flamingo-style 0-init is correct, but needs schedule/regularization

**Root Cause 3: Objective Mismatch**
- Pure next-token cross-entropy allows the translator to find shortcuts
- The model can minimize loss by forcing the LLM into a narrow state-space region that scores well under teacher forcing
- But this region degenerates at inference time (the "000..." loops satisfy the training objective locally)
- Loss stabilizes at ~4.4-4.5 while generation quality collapses - the objective isn't capturing the failure

**Root Cause 4: Hygiene Issues**
- Decoder-only models (Llama, Mistral) require left padding for generation
- Without explicit `padding_side='left'` and `pad_token`, attention masks are incorrect
- This causes skewed attention distributions and instability

### ChatGPT's Analysis (Confirming + Literature Support)

ChatGPT's analysis confirmed all four root causes and provided literature references:

1. **Normalization stabilizes soft/prefix tuning** (ACL Anthology papers on prefix-tuning)
2. **BLIP-2 uses staged pretraining + strong regularization** to avoid connector collapse
3. **Flamingo 0-init tanh gating** is correct but needs LR multipliers to open gradually
4. **KL regularization** (like RLHF) prevents distribution drift into degenerate modes
5. **Left-padding is non-negotiable** for decoder-only generation (HuggingFace docs)

**Key Insight**: The 25.5% peak proves the translator *can* learn useful signals. The failure is **stability**, not capacity. Don't increase model size before fixing the interface.

---

## Solutions to Implement

### Tier 1: MUST IMPLEMENT (Critical for Stability)

#### 1.1 RMS Matching + LayerNorm on Translator Outputs

**Problem**: Soft tokens have different scale/distribution than target embeddings → logit collapse
**Solution**: Normalize and rescale soft tokens to match target embedding statistics

```python
# One-time initialization (after loading target model)
with torch.no_grad():
    tgt_embed = target_model.get_input_embeddings().weight  # (vocab_size, d_model)
    target_rms = tgt_embed.pow(2).mean(dim=1).sqrt().mean().item()

# Each forward pass (after translator produces soft tokens)
def rms(x, eps=1e-8):
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

soft_tokens = translator(src_hiddens)                         # (B, K, d_model)
soft_tokens = F.layer_norm(soft_tokens, (soft_tokens.size(-1),))  # normalize distribution
soft_tokens = soft_tokens / rms(soft_tokens) * target_rms     # match RMS scale
```

**Why this works**:
- LayerNorm centers and scales the distribution (prevents drift)
- RMS matching ensures magnitude is consistent with target's embeddings
- Prevents over-confident logits that lead to "0000..." loops
- Well-established in prefix/soft-prompt tuning literature

**Expected impact**: Eliminates degenerate repetitive patterns; allows stable training.

---

#### 1.2 Left Padding + Explicit Pad Token (Hygiene)

**Problem**: Decoder-only models generate autoregressively from left-to-right; right-padding breaks attention
**Solution**: Configure tokenizers correctly at initialization

```python
# At startup, before any forward passes
for tokenizer in [source_tokenizer, target_tokenizer]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # CRITICAL for decoder-only models
```

**Why this works**:
- Left padding ensures attention masks are correct during generation
- Prevents skewed attention distributions
- Standard practice for decoder-only LLMs (Llama, Mistral, GPT)

**Expected impact**: Correct attention computation; removes one source of instability.

---

### Tier 2: SHOULD IMPLEMENT (Prevents Collapse)

#### 2.1 KL Consistency Loss to Anchor Distributions

**Problem**: No constraint prevents translator from diverging into degenerate modes that minimize loss locally
**Solution**: Add KL divergence penalty to keep bridged distribution close to baseline

```python
# During training, for each batch:
with torch.no_grad():
    # Get baseline distribution (no bridge, just target LLM with text prompt)
    baseline_logits = target_model(input_ids=prompt_ids, attention_mask=prompt_mask).logits

# Get bridged distribution (with soft tokens)
bridged_logits = target_model(
    inputs_embeds=concat([soft_tokens, prompt_embeds]),
    attention_mask=concat([soft_mask, prompt_mask])
).logits

# KL penalty on first 10-20 decoding steps (where collapse happens)
kl_loss = F.kl_div(
    F.log_softmax(bridged_logits[:, :20], dim=-1),
    F.softmax(baseline_logits[:, :20], dim=-1),
    reduction="batchmean"
)

total_loss = nll_loss + 0.03 * kl_loss  # λ=0.03 (tune between 0.02-0.05)
```

**Why this works**:
- Explicitly prevents the bridged distribution from collapsing into "000..." or "####..." modes
- Baseline LLM is known to be healthy (73% accuracy) - use it as anchor
- Same principle as RLHF (KL to reference policy) and knowledge distillation
- Only penalize first ~20 tokens (early decoding) where collapse starts

**Expected impact**: Bridged accuracy should rise and **stay** instead of peaking and dropping.

---

#### 2.2 Gate Schedule (LR Multiplier) + Dropout

**Problem**: 0-init gates are too slow to open, or open too aggressively without constraints
**Solution**: Use higher LR for gate parameters + add dropout for regularization

```python
# When building optimizer:
gate_params = [p for n, p in translator.named_parameters() if "cross_gate" in n]
other_params = [p for n, p in translator.named_parameters() if "cross_gate" not in n]

optimizer = AdamW([
    {"params": other_params, "lr": base_lr, "weight_decay": 0.01},
    {"params": gate_params, "lr": 3 * base_lr, "weight_decay": 0.0},  # Gates open faster
], betas=(0.9, 0.98), eps=1e-8)

# In GatedCrossAttentionBlock:
self.dropout = nn.Dropout(0.1)

# Forward pass:
cross_attn_output = self.cross_attn(queries, src_seq, src_mask)
gated_output = torch.tanh(self.cross_gate) * self.dropout(cross_attn_output)
queries = queries + gated_output
```

**Why this works**:
- Gate LR multiplier (×3) allows gates to open gradually without cranking global LR
- Dropout (0.1) prevents over-reliance on soft tokens; target never *requires* the bridge
- Maintains Flamingo's 0-init stability while easing the gates open
- Dropout also regularizes the cross-attention layers themselves

**Expected impact**: Smoother training dynamics; less sensitivity to LR schedule.

---

### Tier 3: OPTIONAL (If Needed)

#### 3.1 LoRA Adapter on Target LLM Input Layer

**Problem**: Frozen target LLM might not adapt well to soft token inputs
**Solution**: Add lightweight LoRA (rank=4) to first attention block

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,                          # Low rank
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # Only first layer
    lora_dropout=0.1,
    layers_to_transform=[0],      # Only layer 0
)

target_model = get_peft_model(target_model, lora_config)
```

**Why this might help**:
- Allows frozen LLM to slightly adapt its input projection for soft tokens
- Very stable (widely used in adaptor-tuning / PEFT literature)
- Minimal parameters (~few hundred K)

**When to use**: Only if collapse persists after implementing Tier 1 + 2.

---

## Debugging Utilities to Add

### Analysis Function for Generation Diagnostics

This function helps identify *where* and *why* degeneration happens:

```python
def analyze_bridge_quality(soft_tokens, target_model, prompt_ids, prompt_mask, tokenizer):
    """
    Diagnose why bridged generations degenerate.

    Returns:
        - RMS statistics (soft tokens vs target embeddings)
        - First-token entropy (bridged vs baseline)
        - Top-5 token probabilities for both
        - Attention mass on soft tokens
    """
    with torch.no_grad():
        # 1. RMS scale comparison
        tgt_embed = target_model.get_input_embeddings().weight
        tgt_rms = tgt_embed.pow(2).mean(dim=1).sqrt()
        soft_rms = soft_tokens.pow(2).mean(dim=-1).sqrt()

        stats = {
            "soft_rms_mean": soft_rms.mean().item(),
            "soft_rms_max": soft_rms.max().item(),
            "tgt_embed_rms_mean": tgt_rms.mean().item(),
        }

        # 2. First-token distribution comparison
        baseline_out = target_model(input_ids=prompt_ids, attention_mask=prompt_mask)
        bridged_out = target_model(
            inputs_embeds=torch.cat([soft_tokens, target_model.get_input_embeddings()(prompt_ids)], dim=1),
            attention_mask=torch.cat([torch.ones(soft_tokens.size(0), soft_tokens.size(1), device=prompt_mask.device), prompt_mask], dim=1)
        )

        baseline_probs = baseline_out.logits[:, -1].softmax(-1)
        bridged_probs = bridged_out.logits[:, soft_tokens.size(1) + prompt_ids.size(1) - 1].softmax(-1)

        # Entropy (low entropy = over-confident = likely to loop)
        entropy = lambda p: (-p * p.clamp_min(1e-9).log()).sum(-1).mean().item()

        # Top-5 tokens
        def top5(p):
            vals, inds = p[0].topk(5)
            return [(tokenizer.decode([idx]), prob.item()) for idx, prob in zip(inds, vals)]

        analysis = {
            "entropy_baseline": entropy(baseline_probs),
            "entropy_bridged": entropy(bridged_probs),
            "top5_baseline": top5(baseline_probs),
            "top5_bridged": top5(bridged_probs),
        }

        return {**stats, **analysis}
```

**What to look for**:
- `soft_rms_mean >> tgt_embed_rms_mean`: Scaling problem (RMS matching not working)
- `entropy_bridged << entropy_baseline`: Logit collapse (add KL penalty)
- Top-1 token is a repeated digit/symbol: First-token pathology confirmed

---

## Implementation Plan

1. **Fix tokenizer hygiene** (5 lines, startup code)
2. **Add RMS matching + LayerNorm** (compute target_rms at init; apply normalization in forward)
3. **Implement KL consistency loss** (compute baseline logits; add KL term to loss)
4. **Add gate schedule + dropout** (separate param groups in optimizer; dropout in blocks)
5. **Add analysis utilities** (debugging function for eval)
6. **Test on conservative config** (same hyperparams, validate fixes work)
7. **Document results** (update this log with before/after metrics)

---

## Expected Outcomes After Fixes

### Before Fixes (Current State)
- Peak bridged accuracy: 25.5% @ step 750
- Final bridged accuracy: 4.0% @ step 2000
- Degenerate patterns: "000...", "####...", empty generations
- Loss stable but generation collapses

### After Fixes (Expected)
- Bridged accuracy should rise steadily (no collapse)
- Target: 30-40% bridged accuracy sustained through training
- No degenerate repetitive patterns
- Smoother loss curves; generation quality tracks loss

### Success Criteria
- ✅ No "000..." or "####..." loops in generation samples
- ✅ Bridged accuracy increases monotonically or plateaus (no collapse)
- ✅ `entropy_bridged` stays within 20% of `entropy_baseline`
- ✅ `soft_rms_mean` matches `tgt_embed_rms_mean` (within 10%)

---

## References & Literature

### Core Papers
- **BLIP-2** (Li et al., 2023): Bootstrapping language-image pre-training with frozen LLMs. Shows importance of connector normalization and staged training.
- **Flamingo** (Alayrac et al., 2022): Tanh-gated cross-attention with 0-init. Ablations confirm gating stabilizes training.
- **Prefix-Tuning** (Li & Liang, 2021): Reparameterization with MLP + normalization stabilizes soft prompt optimization.

### Key Techniques
- **RMS Normalization**: Standard in Llama, Mistral architectures. Critical for scale-invariant representations.
- **KL Regularization**: Used in RLHF (PPO), knowledge distillation, and constrained policy optimization to prevent distribution drift.
- **Left Padding**: HuggingFace documentation and community threads (Stack Overflow, GitHub issues) confirm this is non-negotiable for decoder-only models.

### Relevant to This Work
- Cross-modal bridging (vision → language) shares the same interface stability problems
- BLIP-2's Q-Former achieves robust bridging with ~32 queries (similar to our 48 soft tokens)
- The failure mode we observe (peak → collapse) is widely reported in soft prompt tuning without proper normalization

---

## Changelog

**2025-11-06**: Initial cross-attention sweep run
- Observed peak at 25.5% followed by catastrophic collapse to 4%
- Identified degenerate generation patterns

**2025-11-06**: Analysis and diagnosis
- Claude identified 4 root causes (scale mismatch, gating, objective mismatch, hygiene)
- ChatGPT confirmed with literature support
- Documented 3-tier solution (must/should/optional)

**2025-11-06**: Implementing fixes
- Adding RMS matching, LayerNorm, KL loss, gate schedule, dropout, hygiene fixes
- Next: Re-run conservative config and validate fixes
