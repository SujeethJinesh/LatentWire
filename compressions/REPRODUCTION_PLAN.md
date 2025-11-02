# Reproduction Plan for Compression Experiments

## Current Status: NOT READY FOR REPRODUCTION ❌

Our current implementations are **inspired by** but **do not faithfully reproduce** the cited papers. This document outlines what's needed to make them reproducible.

## Critical Issues

### 1. Gist Tokens (Mu et al., NeurIPS 2023)

**Paper Requirements:**
- Attention mask modification during training (core innovation)
- Batch size = 1 (per GitHub repo)
- Instruction finetuning dataset (not SQuAD)
- Special rotary position embedding handling
- Meta-learning approach for zero-shot generalization

**Current Implementation:**
- ❌ No attention masking
- ❌ Batch size = 8
- ❌ Using SQuAD Q&A, not instruction tuning
- ❌ Standard position embeddings
- ❌ Supervised learning, not meta-learning

**Status:** Our "Gist" is just another cross-attention variant

**To Fix:**
1. Implement attention mask modification (key innovation)
2. Change to batch_size=1
3. Use instruction tuning dataset (e.g., FLAN)
4. Add gist position offset handling
5. Implement meta-learning training loop

**Estimated Effort:** 3-5 days

### 2. Missing Baselines

**Currently Have:**
- ✅ Uncompressed teacher model (upper bound)

**Missing (CRITICAL):**
- ❌ **Token-budget baseline:** Truncate to M tokens
  - Most important: proves compression beats naive truncation
  - Easy to implement
- ❌ **Prefix-only:** First M tokens
- ❌ **Suffix-only:** Last M tokens
- ❌ **Random selection:** Random M tokens
- ❌ **Extractive summarization:** Use BERT/Longformer
- ❌ **Abstractive summarization:** Use T5/BART

**To Fix:**
1. Add `TruncationBaseline` class (truncate to M tokens)
2. Add `PrefixBaseline` class
3. Add `SuffixBaseline` class
4. Add `RandomSelectionBaseline` class
5. Optionally: Add summarization baselines

**Estimated Effort:** 1 day

### 3. Cross-Attention (Perceiver)

**Paper Requirements (Jaegle et al., 2021):**
- Iterative refinement (6-8 passes)
- Learned position encodings
- Specific initialization scheme
- Tested on ImageNet, not NLP

**Current Implementation:**
- ❌ Single-pass only
- ❌ Standard PyTorch initialization
- ✅ Using cross-attention correctly

**Status:** Simplified version, not full Perceiver

**To Fix:**
1. Add iterative refinement loop
2. Implement learned position encodings
3. Match paper's initialization

**Estimated Effort:** 2 days

### 4. Convolutional Compressor

**Issue:** Not based on any specific paper
- WaveNet is for audio, not text compression
- No NLP compression paper uses this exact approach

**Status:** Custom method, no reproduction target

**Options:**
1. Find proper NLP compression paper to reproduce
2. Keep as "custom baseline" but don't claim reproduction
3. Remove entirely

### 5. Weighted Pooling

**Issue:** Completely custom, no paper

**Status:** Custom method, no reproduction target

**Options:**
1. Keep as simple baseline
2. Replace with real method from literature

## Required Baselines (Priority Order)

### Priority 1: MUST HAVE
1. **Token-Budget Baseline** (truncate to M tokens)
   - Proves compression > naive truncation
   - 1-2 hours to implement
2. **Prefix/Suffix Baselines**
   - Show position matters
   - 1 hour to implement

### Priority 2: SHOULD HAVE
3. **Random Selection Baseline**
   - Control for selection bias
   - 1 hour to implement
4. **Fixed Gist Implementation**
   - Follow paper exactly
   - 3-5 days to implement

### Priority 3: NICE TO HAVE
5. **Summarization Baselines**
   - BERT extractive
   - T5 abstractive
   - 2-3 days to implement

## Recommended Approach

### Option A: Quick Validation (1 week)
1. Add token-budget, prefix, suffix, random baselines (1 day)
2. Keep current architectures as "inspired by" (0 days)
3. Run experiments with honest baselines (1 day)
4. Update documentation to clarify limitations (1 hour)
5. Report results with caveats

**Pros:** Fast, honest about limitations
**Cons:** Can't claim to reproduce papers

### Option B: Faithful Reproduction (3-4 weeks)
1. Implement proper Gist with attention masking (1 week)
2. Implement proper Perceiver with iteration (3 days)
3. Find/implement proper conv baseline from literature (3 days)
4. Add all baselines (2 days)
5. Run comprehensive experiments (1 week)
6. Reproduce paper numbers

**Pros:** Reproducible, publishable
**Cons:** Time-intensive

### Option C: Hybrid (2 weeks)
1. Add all baselines (2 days)
2. Fix Gist implementation (1 week)
3. Keep other methods as "inspired by" (0 days)
4. Focus on comparing against baselines, not reproducing papers
5. Contribute Gist reproduction to community

**Pros:** Balanced, one solid reproduction
**Cons:** Still can't reproduce other papers

## Recommended Action

I recommend **Option A** (Quick Validation) for now:

1. **Immediately add baselines** to prove value
2. **Rename architectures** to be honest:
   - "Gist-inspired" instead of "Gist"
   - "Cross-attention" instead of "Perceiver"
3. **Update README** with limitations
4. **Run experiments** to see if any method beats baselines
5. **Then decide** if full reproduction is worth it

## Implementation Plan (Option A)

### Step 1: Add Baselines (4 hours)
```python
class TruncationBaseline:
    """Truncate to first M tokens."""
    def compress(self, input_ids, M):
        return input_ids[:, :M]

class SuffixBaseline:
    """Keep last M tokens."""
    def compress(self, input_ids, M):
        return input_ids[:, -M:]

class RandomBaseline:
    """Random M tokens."""
    def compress(self, input_ids, M):
        indices = torch.randperm(input_ids.size(1))[:M]
        return input_ids[:, indices.sort()[0]]
```

### Step 2: Update Eval (2 hours)
Add baseline comparison to evaluation:
```python
def evaluate_all_baselines(self):
    results = {
        'uncompressed': self.evaluate_baseline(),
        'truncation': self.evaluate_truncation(M=64),
        'suffix': self.evaluate_suffix(M=64),
        'random': self.evaluate_random(M=64),
        'learned': self.evaluate()  # Our methods
    }
    return results
```

### Step 3: Update Docs (1 hour)
Clarify in README:
- "Inspired by" not "reproduction of"
- List differences from papers
- Show baseline comparisons

### Step 4: Run Experiments (1 day)
Test if learned compression beats baselines

## Success Criteria

**Minimum (Option A):**
- [ ] Learned methods beat truncation baseline
- [ ] Results clearly show comparison to baselines
- [ ] Documentation is honest about limitations

**Full (Option B):**
- [ ] Reproduce Gist paper's 26× compression results
- [ ] Reproduce Perceiver results on appropriate task
- [ ] All baselines implemented
- [ ] Results match or exceed papers

## Next Steps

Please decide which option you prefer:
1. **Option A:** Quick validation with baselines (1 week)
2. **Option B:** Full faithful reproduction (3-4 weeks)
3. **Option C:** Hybrid approach (2 weeks)

I'll implement whichever you choose.
