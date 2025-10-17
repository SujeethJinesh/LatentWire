# Critical Training/Evaluation Issues in Sequence Compression Experiment

## Executive Summary

The sequence compression experiment (Experiment 1) has **fundamental training/evaluation mismatches** that likely explain the poor performance (2.88% F1 vs 4-6% token truncation baseline). These issues go beyond just "compression is hard" - the setup is not testing the hypothesis properly.

---

## Issue 1: Answer Length Mismatch ⚠️ CRITICAL

### The Problem

**Training**: Answers tokenized with `max_length=64` (line 272 of `train_sequence_compression.py`)
```python
answer_encoded = self.tokenizer(
    ex['answer'],
    truncation=True,
    max_length=64,  # Train on answers up to 64 tokens
)
```

**Evaluation**: Only generates `max_new_tokens=12` tokens (line 221)
```python
outputs = model.generate(
    inputs_embeds=compressed,
    max_new_tokens=12,  # Only generate 12 tokens!
)
```

### Impact

- Model trained to produce answers up to 64 tokens
- Evaluation truncates at 12 tokens
- Any answer >12 tokens is automatically wrong, lowering F1 score
- Need to check: What % of SQuAD answers exceed 12 tokens?

### Fix

Run diagnostic script to measure actual answer lengths, then set:
```python
max_new_tokens=32  # Should cover 95%+ of answers
```

---

## Issue 2: No Anchor Text ⚠️ CRITICAL

### The Problem

**Training input format** (line 363):
```python
inputs_embeds = torch.cat([compressed, answer_embeds], dim=1)
# Result: [comp_1, ..., comp_256, ans_embed_1, ans_embed_2, ...]
```

**Evaluation input format** (line 216-220):
```python
compressed = compressor(source_embeds, positions)
outputs = model.generate(inputs_embeds=compressed)
# Result: [comp_1, ..., comp_256] → generate
```

**No separator token!** The model receives compressed tokens and must immediately start generating, with no signal like:
- "Answer: " anchor text
- BOS token
- Any delimiter

### Comparison to LatentWire

LatentWire uses (from `CLAUDE.md`):
```bash
--warm_anchor_text "Answer: "
--append_bos_after_prefix yes
```

Input format: `[compressed_32] [BOS] "Answer: " [answer_tokens]`

This gives the model a clear signal: "context is over, now generate answer".

### Impact

- Model doesn't know when to transition from reading compressed context to generating
- No learned "start of answer" signal
- Compressed representations have no natural boundary marker
- Model must infer "now I should generate" from the compressed tokens alone

### Fix

Add anchor text between compression and answer:
```python
# Training
anchor_ids = tokenizer("Answer: ", add_special_tokens=False)['input_ids']
anchor_embeds = model.get_input_embeddings()(torch.tensor(anchor_ids))
inputs_embeds = torch.cat([compressed, anchor_embeds, answer_embeds], dim=1)

# Evaluation
inputs_embeds = torch.cat([compressed, anchor_embeds], dim=1)
outputs = model.generate(inputs_embeds=inputs_embeds, ...)
```

---

## Issue 3: Teacher Forcing Mismatch (Exposure Bias)

### The Problem

**Training** (lines 359-363):
```python
# Get answer embeddings (teacher-forced)
answer_embeds = model.get_input_embeddings()(answer_ids[:, :-1])

# Concatenate compressed prefix with answer embeddings
inputs_embeds = torch.cat([compressed, answer_embeds], dim=1)
```

At each training step, the model sees:
- Step 1: [compressed, **gold_ans_1_embed**] → predict ans_2
- Step 2: [compressed, **gold_ans_1_embed, gold_ans_2_embed**] → predict ans_3
- Step 3: [compressed, **gold_ans_1_embed, gold_ans_2_embed, gold_ans_3_embed**] → predict ans_4

**Evaluation** (lines 219-220):
```python
outputs = model.generate(inputs_embeds=compressed)
```

At each generation step, the model sees:
- Step 1: [compressed] → generate ans_1
- Step 2: [compressed, **pred_ans_1_embed**] → generate ans_2  (uses own prediction!)
- Step 3: [compressed, **pred_ans_1_embed, pred_ans_2_embed**] → generate ans_3

### Impact

- During training: model conditioned on perfect gold context
- During evaluation: model must use its own (imperfect) predictions
- Errors compound: wrong token at step 1 → wrong context at step 2 → worse token at step 3
- This is standard **exposure bias** but particularly severe here because:
  1. Compressed representations are novel (not natural text embeddings)
  2. No anchor text to help recover
  3. Model never practiced generating from compressed tokens alone

### Why This Matters More Here

In normal text generation:
- Training: [the, cat, sat] → predict "on"
- Eval: [the, cat, sat] → predict "on" (same context type)

In compressed generation:
- Training: [comp_256, gold_"the", gold_"cat"] → predict "sat"
- Eval: [comp_256, pred_"dog", pred_"ran"] → predict ??? (different context!)

The model has **never seen** a situation where it generates from compressed tokens without gold embeddings.

### Partial Fix

Use scheduled sampling during training:
```python
# With probability p, use model's own predictions instead of gold embeddings
if random.random() < scheduled_sampling_prob:
    with torch.no_grad():
        pred_ids = model.generate(...)
        answer_embeds = model.get_input_embeddings()(pred_ids)
else:
    answer_embeds = model.get_input_embeddings()(gold_answer_ids)
```

Increase `scheduled_sampling_prob` from 0 → 0.5 over training.

---

## Issue 4: Loss Objective Mismatch

### What is the loss against?

**Loss computation** (line 137):
```python
outputs = model(inputs_embeds=inputs_embeds, labels=labels)
loss = outputs.loss  # Cross-entropy on next-token prediction
```

**Labels setup** (lines 366-377):
```python
# Create labels: mask compressed prefix, predict answer
labels = torch.full(..., -100, ...)  # -100 = ignore
labels[:, target_len:] = answer_ids[:, 1:]  # Only supervise answer tokens

# Mask padding in labels
labels[:, target_len:][ans_mask_shifted == 0] = -100
```

### What this means

The loss is **ONLY** on answer token prediction:
- Compressed tokens: `labels = -100` (ignored)
- Answer tokens: `labels = [gold_token_2, gold_token_3, ...]` (supervised)

Loss = Cross-entropy on predicting `answer_tokens[i+1]` given `[compressed, answer_embeds[:i]]`

### What is NOT supervised

1. **Compression quality**: No signal for whether compressed representations preserve information
2. **Information preservation**: No penalty for losing critical context
3. **Query distinctiveness**: No signal preventing all compressed tokens from collapsing to same representation
4. **First token accuracy**: First generated token has same weight as 50th token

### Why This Fails

From the analysis (REPORT.md Hypothesis 6):
> "All M=32 queries converged to cosine similarity 0.999 - nearly identical"

The compressor learns:
1. "Produce SOME representation that minimizes CE loss when gold answer embeddings are available"
2. NOT: "Preserve input-specific information that enables generation without gold context"

The loss provides **no feedback** for:
- Maintaining distinct representations per input
- Preserving positional information needed for generation
- Preventing mode collapse to generic tokens

### Comparison to LatentWire

LatentWire uses multiple objectives (from `CLAUDE.md` and `losses.py`):

```python
# K-token teacher-forced CE
k_token_ce_from_prefix  # Supervise first K tokens, not all tokens

# Prefix KD
kd_first_k_prefix_vs_text  # Distill text-prompted teacher distributions

# First-token CE
first_token_ce_weight = 0.5  # Extra weight on first token accuracy
```

These provide:
- Explicit signal for first-token quality (critical for generation)
- Knowledge distillation from text baseline (compression target)
- Focus on early tokens (where errors compound)

---

## Issue 5: Are We Testing Things Properly?

### Summary of Testing Issues

❌ **Input format**: No anchor text between compression and answer
❌ **Generation length**: 12 tokens vs 64 token training
❌ **Training-eval gap**: Teacher forcing with gold embeddings vs autoregressive with predictions
❌ **Loss signal**: Only answer CE, no compression quality signal
❌ **First token**: Not prioritized, but critical for autoregressive generation

### Are we feeding inputs correctly?

**Training**: YES, inputs are fed correctly to the model
- Format: `[compressed_tokens] [answer_embeddings]`
- Model receives proper `inputs_embeds` tensor
- Labels correctly mask compressed prefix

**Evaluation**: PARTIALLY
- Compression works correctly
- Generation receives proper `inputs_embeds`
- BUT: Missing anchor text, truncated generation length

### Are we letting the LLM produce results as needed?

❌ **NO** - Evaluation is artificially constrained:
- Only 12 tokens when answers can be 64 tokens
- No anchor text to signal "start generating"
- Trained on 64-token answers, tested on 12-token generation

---

## Recommendations

### Tier 1: Critical Fixes (Required for Valid Experiment)

1. **Fix answer length**: Set `max_new_tokens=32` (run diagnostic script to confirm)
2. **Add anchor text**: Insert "Answer: " between compressed tokens and generation
3. **Add BOS token**: Prepend BOS after compressed sequence (if using instruct model)

### Tier 2: Architecture Improvements

4. **Anchor embeddings**: Train with anchor text consistently
5. **First-token loss**: Add extra weight/auxiliary loss on first generated token
6. **KL divergence**: Add KL loss comparing compressed→answer logits to text→answer logits

### Tier 3: Training Improvements

7. **Scheduled sampling**: Mix teacher forcing with autoregressive during training
8. **Compression regularization**: Add auxiliary loss preventing query collapse
9. **Multi-task**: Add auxiliary task predicting compression quality

---

## Diagnostic Next Steps

Run the diagnostic script to measure actual answer lengths:
```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
python scripts/diagnose_training_setup.py
```

This will show:
- Answer length distribution
- % of answers exceeding 12 tokens
- Sample examples
- Specific recommendations for `max_new_tokens`

Then implement Tier 1 fixes and rerun a short experiment (3 epochs, 1000 samples) to validate.

---

## Why This Explains the Poor Results

**Current results**: 2.88% F1 (worse than 4-6% token truncation)

**Explained by**:
1. **~50-80% of answers truncated** due to max_new_tokens=12 (need to confirm with diagnostic)
2. **No signal to start generating** (no anchor text)
3. **Never practiced autoregressive generation** from compressed tokens (teacher forcing gap)
4. **Loss doesn't supervise compression quality** (only answer CE with gold context)

These are not "compression is hard" problems - these are **experimental setup bugs** that prevent the model from learning and being evaluated properly.

Fix Tier 1 issues first, then reassess whether sequence compression itself is the problem.
