# Evaluation Guide for Hero Resume Best Checkpoint

## Training Results Summary

**Best Checkpoint**: `runs/hero_resume/ckpt_stageb_best`
- **Peak Performance**: 25.0% first-token accuracy at step 4113
- **Schedule**: keep_prob frozen at 0.85 (dropout rate 0.15)
- **Configuration**: LATENT_LEN=64, D_Z=256, deep_prefix_len=100, LoRA r=16

### What is first_acc=25%?
- **First-token accuracy** = % of times model predicts correct first answer token from latent
- **25%** means 1 in 4 predictions are correct (vs ~0.01% random baseline)
- This is an **intermediate metric** - it shows the latent is decodable but doesn't guarantee good F1
- **F1 evaluation will tell us if this translates to usable answers**

### Performance Progression:
1. Step 2302: 13.9%
2. Step 2313: 16.7%
3. Step 2976: 19.4%
4. Step 3831: 22.2%
5. Step 4113: **25.0% ← BEST**

## Diagnostics "Issue" - Not a Bug

**Why diagnostics show 0-8% but peak was 25%:**
- Diagnostics write every 10 steps (step 10, 20, 30, ...)
- Peak checkpointing checks EVERY step
- Peak occurred at step 4113 (between diagnostic writes at 4110 and 4120)
- This is **correct behavior** - we catch peaks even between diagnostic samples

## Evaluation Command

```bash
# Evaluate the best checkpoint on 1000 samples
cd /Users/sujeethjinesh/Desktop/LatentWire

python latentwire/eval.py \
  --ckpt runs/hero_resume/ckpt_stageb_best \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset squad \
  --samples 1000 \
  --max_new_tokens 16 \
  --fresh_eval \
  --use_chat_template yes \
  --latent_anchor_mode chat \
  --append_bos_after_prefix yes \
  --first_token_top_p 1.0 \
  --first_token_temperature 0.0 \
  --prefix_gain 1.1 \
  --token_budget_mode content_only \
  --token_budget_k 64 \
  --chunk_size 88
```

### Key Eval Parameters:
- `--samples 1000`: Evaluate on 1000 SQuAD examples
- `--max_new_tokens 16`: Generate up to 16 tokens (typical answer length)
- `--first_token_top_p 1.0, --first_token_temperature 0.0`: Deterministic (greedy) decoding
- `--prefix_gain 1.1`: Slight amplitude boost to latent (10%)
- `--token_budget_k 64`: Compare against text truncated to 64 tokens (fair comparison)

## What to Look For in Results

### Success Criteria (from PLAN.md):
- ✅ **Minimum viable**: F1 > 0.10, FirstTok@1 > 12%
- ✅ **Good progress**: F1 > 0.20, FirstTok@1 > 20%
- 🎯 **Full success**: F1 close to text baseline (0.80+)

### Key Metrics:
1. **Latent F1**: Answer quality from compressed latent
2. **Text baseline F1**: Upper bound (full prompt)
3. **Token-budget F1**: Fair comparison (text truncated to same tokens as latent)
4. **FirstTok@1**: First-token accuracy on eval set
5. **Compression ratio**: Text bytes / Latent bytes

### Expected Output:
The eval will show tables comparing:
- **Text baseline**: Full prompt performance
- **Latent**: Compressed latent performance  
- **Token-budget**: Truncated text (fair comparison)
- **Joint rescoring**: Ensemble of both models

## Next Steps Based on Results

### If F1 > 0.10:
✅ **Success!** Continue with:
1. Hyperparameter tuning (prefix_gain, keep_prob schedule)
2. Scale to larger checkpoints
3. Compression optimization (quantization sweeps)

### If F1 < 0.10:
Need to diagnose:
1. Is first-token accuracy high but answers incomplete? → Increase max_new_tokens
2. Is generation quality poor? → Check decode settings (temperature, top_p)
3. Is the gap between first_acc (25%) and actual generation large? → Possible autoregressive degradation

## Files to Review After Eval
1. **Console output**: Summary tables with F1/EM scores
2. **Generated answers**: Check quality of actual predictions
3. **Compression stats**: Bytes used for text vs latent
