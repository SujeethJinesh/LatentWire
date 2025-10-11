# Stage 1 Evaluation Fixes - Critical Bug Investigation

## Investigation Date: 2025-10-11

## Summary

Found and fixed critical evaluation bugs that caused F1 and EM scores to be 0.0%.

## Critical Findings

###  1. **Wrong Script Executed**
**Problem**: The HPC run used the OLD `train_adapter_only.py`, NOT the new `train_adapter_only_phase1.py`

**Evidence**:
- Log shows "ADAPTER-ONLY TRAINING" (should be "STAGE 1 PHASE 1: PURE RECONSTRUCTION")
- Has both `recon_loss` and `ce_loss` (Phase 1 should only have reconstruction loss)
- PCA fitted on 100 samples (should be 80k)
- Configuration includes `ce_weight=1.0` (Phase 1 doesn't have this parameter)

**Timeline**:
- User committed Phase 1 implementation: `6f7688b`
- User ran training BEFORE pulling Phase 1 changes
- User pushed logs: `6e0a722`
- Therefore logs are from OLD script, not new Phase 1 script

### 2. **Missing attention_mask in Generation (ROOT CAUSE OF 0.0% F1/EM)**
**Problem**: `model.generate()` calls did not pass `attention_mask` parameter

**Impact**:
```
Warning: The attention_mask is not set and cannot be inferred from input
because pad token is same as eos token. As a consequence, you may observe
unexpected behavior.
```

This caused the model to generate garbage/empty outputs, resulting in 0% overlap with reference answers.

**Locations**:
- `train_adapter_only.py` line 366 (quick eval)
- `train_adapter_only.py` line 469 (full eval)
- `train_adapter_only_phase1.py` line 440 (quick eval)
- `train_adapter_only_phase1.py` line 476 (full eval)

### 3. **Temperature/top_p Warnings**
**Problem**: Model's default generation config has `temperature=0.6` and `top_p=0.9`, but we set `do_sample=False`

**Warning**:
```
UserWarning: `do_sample` is set to `False`. However, `temperature` is set to
`0.6` -- this flag is only used in sample-based generation modes.
```

## Fixes Applied

### Fix 1: Add attention_mask to all generate() calls

**Before**:
```python
outputs = model.generate(
    inputs_embeds=adapted,
    max_new_tokens=20,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id
)
```

**After**:
```python
# Create attention mask (all 1s for embeddings we're passing)
attention_mask = torch.ones(
    adapted.shape[0], adapted.shape[1],
    dtype=torch.long, device=adapted.device
)

outputs = model.generate(
    inputs_embeds=adapted,
    attention_mask=attention_mask,  # CRITICAL FIX
    max_new_tokens=20,
    do_sample=False,
    temperature=None,  # Explicitly disable
    top_p=None,        # Explicitly disable
    pad_token_id=tokenizer.pad_token_id
)
```

### Fix 2: Explicitly set temperature=None and top_p=None
This prevents the warnings when `do_sample=False`.

## Files Modified

1. **train_adapter_only_phase1.py** (NEW Phase 1 script)
   - Line 440-454: Quick eval generation fixed
   - Line 485-499: Full eval generation fixed

2. **train_adapter_only.py** (OLD script)
   - Line 365-380: Quick eval generation fixed
   - Line 477-492: Full eval generation fixed

## Testing

All 23 tests pass (1 GPU-only test skipped):
```bash
pytest tests/test_stage1_phase1.py -v
# Result: 23 PASSED, 1 skipped in 10.34s
```

## Expected Results After Fix

With these fixes, the Phase 1 script should:
1. **No warnings** during generation
2. **Non-zero F1/EM scores** (actual performance depends on training quality)
3. **Proper evaluation** that matches training setup

## Next Steps

1. **Run Phase 1 script on HPC**: Use `bash scripts/run_stage1_h100.sh`
2. **Verify fixes work**: Check that F1/EM scores are non-zero
3. **Interpret results**:
   - F1 ≥70%: Hypothesis validated! (reconstruction → generation works)
   - F1 50-70%: Proceed to Phase 2 (add generation training)
   - F1 <50%: Investigate compression quality or architecture

## Technical Notes

### Why attention_mask is Critical

When using `inputs_embeds` (embedding inputs instead of token IDs):
- Model cannot infer attention mask from pad tokens
- Without mask, model doesn't know which positions are valid
- This causes undefined behavior in attention mechanisms
- Results in garbage outputs and 0% evaluation scores

### Why Phase 1 is Different from OLD Script

**OLD Script** (`train_adapter_only.py`):
- Training: Uses teacher forcing (real answer embeddings concatenated)
- Evaluation: Autoregressive generation
- **Mismatch**: Trains on one task, evaluates on another
- PCA: Fitted on 100 samples

**Phase 1 Script** (`train_adapter_only_phase1.py`):
- Training: Pure MSE reconstruction loss only
- Evaluation: Autoregressive generation
- **Match**: Tests hypothesis directly (reconstruction → generation)
- PCA: Fitted on full 80k training set

## Diagnostic Commands

Check if correct script is running:
```bash
head -50 runs/stage1_adapter_only/logs/training.log | grep "PHASE 1"
# Should see: "STAGE 1 PHASE 1: PURE RECONSTRUCTION"
```

Check for warnings:
```bash
grep -i "warning\|attention_mask" runs/stage1_adapter_only/logs/training.log
# Should see NO attention_mask warnings after fix
```

Check evaluation scores:
```bash
grep "F1:\|EM:" runs/stage1_adapter_only/logs/training.log
# Should see non-zero scores
```

## References

- Phase 1 commit: `6f7688b`
- Issue analysis: `STAGE1_ANALYSIS.md`
- Implementation plan: `STAGE1_CORRECTED_PLAN.md`
- Test suite: `tests/test_stage1_phase1.py`
