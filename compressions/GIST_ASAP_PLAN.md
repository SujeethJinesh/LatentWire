# Gist Tokens ASAP Reproduction Plan

## What I've Done

1. **Cloned official Gist repo** to `compressions/gisting_reference/`
2. **Analyzed their implementation** - found the key components
3. **Extracted gist mask functions** - exact copy from their `src/data/gist.py`
4. **Created minimal reproduction** - `gist_minimal.py` with their masking logic

## Key Findings from Official Repo

### Critical Implementation Details

**1. Gist Masking (THE Innovation)**
- Located in `gisting_reference/src/data/gist.py` lines 69-118
- Creates a 4D attention mask `[batch, 1, seq_len, seq_len]`
- Behavior:
  - Tokens **before last gist**: attend to everything before last gist (but NOT after)
  - Tokens **after last gist**: attend ONLY to gist tokens
  - This forces compression into gist tokens!

Example with `G` = gist token:
```
      a b c G d
    a 1 1 1 1 0    <- 'a' can't see 'd' (after gist)
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0    <- gist sees everything
    d 0 0 0 1 1    <- 'd' can ONLY see gist 'G' (compression!)
```

**2. Model Modifications**
- They MODIFY transformers source: `gist_llama.py` (34KB file)
- Key changes:
  - Custom `GistLlamaRotaryEmbedding` with `gist_offset` support
  - Modified `GistLlamaAttention` to pass `gist_offset` to RoPE
  - Modified `GistLlamaModel.forward()` to accept `attention_mask_gist`
  - Combines gist mask with causal mask (lines 536-542)

**3. Training Setup**
- **Dataset**: Alpaca+ (instruction tuning, NOT Q&A)
- **Batch size**: 1 (REQUIRED - see line 116 comment "THIS WILL BREAK FOR BATCH SIZE > 1")
- **Padding**: Left padding for LLaMA
- **Token**: Add special `<GIST>` token to vocabulary
- **Hardware**: 4× A100 80GB with DeepSpeed stage 3
- **Transformers version**: Specific commit `fb366b9a` (newer versions break)
- **DeepSpeed version**: 0.8.3 (newer versions degrade performance)

**4. Gist Token Insertion**
- Format: `"Instruction: {text}\n<GIST> <GIST> ...\nOutput: {response}"`
- Initialize new `<GIST>` embedding to average of vocabulary (lines 197-200 of train.py)

## Two Paths Forward

### Option A: Quick Test (2-4 hours) ✅ READY NOW

**What's ready:**
- `compressions/gist_minimal.py` - runnable script with:
  - Exact gist mask functions from their repo
  - Alpaca dataset loading
  - `<GIST>` token insertion
  - Data collator with left padding
  - Training loop

**Limitations:**
- Doesn't integrate gist mask into model forward (requires modifying transformers)
- Gist tokens are in sequence but standard causal attention is used
- Won't reproduce paper results, but tests infrastructure

**Run it:**
```bash
cd compressions
python gist_minimal.py  # Tests with 100 samples, 1 epoch
```

**Expected outcome:**
- Verifies gist mask generation works
- Trains model with gist tokens in sequence
- Validates data pipeline
- **NOT a faithful reproduction** but proves infrastructure works

### Option B: Faithful Reproduction (1-2 days) 🎯 RECOMMENDED

**Steps:**

1. **Copy their `gist_llama.py`** (lines needed):
   - `GistLlamaRotaryEmbedding` class (lines 79-126)
   - `GistLlamaAttention` class (lines 128-260)
   - `GistLlamaDecoderLayer` class (lines 263-339)
   - `GistLlamaModel` class (lines 342-599)
   - `GistLlamaForCausalLM` class (lines 602-end)

2. **Adapt for Llama 3.1 8B**:
   - Update imports to work with current transformers
   - Change base class from `LlamaPreTrainedModel` to match Llama 3.1
   - Test position ID handling with Llama 3.1's RoPE

3. **Use their exact training config**:
   - Alpaca+ dataset
   - batch_size=1
   - DeepSpeed stage 3 (we have 4× H100)
   - Same hyperparameters from their configs

4. **Expected timeline**:
   - Day 1: Adapt gist_llama.py, test on small model
   - Day 2: Full training run (52K samples, 3 epochs)

**Expected outcome:**
- Should match or beat their LLaMA-7B results
- Up to 26× compression
- 40% FLOPs reduction
- ROUGE scores comparable to full prompt

## My Recommendation: Option B

**Why:**
- You said "I need reproduction ASAP"
- Option A tests infrastructure but won't give paper results
- Option B is only 1-2 days and gives real results
- We have BETTER hardware (4× H100 > 4× A100)
- Llama 3.1 8B ≈ their LLaMA-7B (similar arch)

**Concrete next steps:**

1. **Today (4 hours)**:
   - Copy `gisting_reference/src/gist_llama.py` → `compressions/gist_llama_31.py`
   - Update imports and class inheritance for current transformers
   - Test on small Llama model (debug config)

2. **Tomorrow (4 hours)**:
   - Integrate with Llama 3.1 8B
   - Test gist masking with forward pass
   - Verify position IDs work correctly

3. **Day 2 (8-12 hours)**:
   - Full training run on 4× H100
   - 52K Alpaca+ samples, 3 epochs
   - Evaluate with ROUGE metrics
   - Compare to paper results

## Files Created

1. `compressions/gist_minimal.py` - Quick test script (ready to run)
2. `compressions/gist_llama_wrapper.py` - Wrapper attempt (incomplete)
3. `compressions/gisting_reference/` - Official repo clone
4. `compressions/GIST_ASAP_PLAN.md` - This file

## Critical Success Factors

✅ **Get attention masking right** - Exact copy from their `gist.py`
✅ **Use batch_size=1** - Required for position IDs
✅ **Use Alpaca+ dataset** - Instruction tuning, not Q&A
✅ **Initialize gist token** - Average of vocabulary
⚠️ **Modify model forward()** - Requires adapting `gist_llama.py`
⚠️ **Handle RoPE position IDs** - Critical for Llama

## Questions for You

1. **Which option?**
   - A: Quick test (2-4 hours, not faithful)
   - B: Faithful reproduction (1-2 days, matches paper)

2. **If Option B, should I:**
   - Start adapting `gist_llama.py` now?
   - Test with small model first or go straight to Llama 3.1 8B?

3. **Hardware:**
   - Run on HPC cluster (4× H100)?
   - What's the job submission process?

Let me know and I'll proceed immediately!
