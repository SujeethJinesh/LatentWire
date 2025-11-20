# Claude's Analysis of Nov 19 Phase 2 Test Results

## Test Configuration (Commit 2a3c0a0)

The `run_phase2_single_gpu_suite.sh` script ran two configurations:

1. **prompt_softonly**:
   - `prompt_alignment_weight=1.0`
   - `dit_loss_weight=1.0`
   - `PROMPT_MODE=soft_only`

2. **answer_softplus**:
   - `prompt_alignment_weight=0.001`
   - `dit_loss_weight=0.1`
   - `PROMPT_MODE=soft_plus_text`

## Results Summary

| Configuration | Best Bridged | Source-Alone | Target-Alone | Gap to Target | Status |
|--------------|--------------|--------------|--------------|---------------|--------|
| prompt_softonly | **0.025** (2.5%) | 0.765 | 0.515 | **-49.0 pts** | ❌ Collapsed |
| answer_softplus | **0.360** (36.0%) | 0.765 | 0.515 | **-15.5 pts** | ⚠️ Below target |

### Detailed Breakdown (answer_softplus progression):
- Step 250: 24/200 correct (12%)
- Step 750: 60/200 correct (30%)
- Step 1000: 66/200 correct (33%)
- **Step 1250: 72/200 correct (36%)** ← BEST
- Step 1500: 66/200 correct (33%)
- Step 1750: 63/200 correct (31.5%)

## Critical Findings

### 1. ✅ Answer Supervision is Essential
- **prompt_softonly: 2.5%** vs **answer_softplus: 36%** = **14× improvement**
- Prompt-teacher DiT cannot learn meaningful cross-model translation
- **Conclusion**: Answer supervision is mandatory, not optional

### 2. ❌ Translator Degrades Target Performance
- Target-alone baseline: **51.5%**
- Best bridged result: **36.0%**
- **Gap: -15.5 percentage points**

This means soft tokens are **actively hurting** Mistral's native reasoning ability, not enhancing it.

### 3. ⚠️ Directional Asymmetry
- **Phase 1 (Mistral→Llama)**: 64.5% bridged vs 77.0% target = **-12.5 pts gap**
- **Phase 2 (Llama→Mistral)**: 36.0% bridged vs 51.5% target = **-15.5 pts gap**

Phase 2 has a **24% larger gap**, suggesting Llama→Mistral is fundamentally harder.

### 4. 📈 Modest Improvement from Tuning
- Nov 18 (original weights): 26% bridged
- Nov 19 (tuned weights): 36% bridged
- **Improvement: +10 pts** but still not viable

## Assessment of Codex's Proposed Fixes (NEXT_STEPS.md)

### ❌ DISAGREE: Prioritizing Soft-Only Fixes (Section 2.3)

**Codex proposes:**
- Curriculum learning (soft_plus_text → soft_only fade)
- Contrastive prompt loss
- Probe/auxiliary losses for soft-only

**Claude's objection:**
Even the **answer-supervised soft+text** configuration underperforms the target baseline by 15.5 pts. Investing resources in soft-only improvements is premature when soft+text doesn't work yet.

**Recommendation**: DEFER all soft-only work until soft+text achieves ≥ target baseline.

---

### ⚠️ PARTIALLY AGREE: Tokenizer/RoPE Alignment (Section 2.4)

**Codex proposes:**
> "design an explicit loss that forces the DiT projection to mimic Llama's positional/vocabulary geometry (e.g., align rotary phases or KL-match logits on a shared sub-vocab)"

**Claude agrees:**
- Real architectural mismatch exists (32K vocab vs 128K, different RoPE geometries)
- This addresses a genuine technical gap

**Claude's concern:**
- Won't fix the core issue: soft tokens **override** literal prompt semantics
- Expected gain: ~5-10 pts at best (won't close 15.5 pt gap alone)

**Recommendation**: Implement, but don't expect it to solve the degradation problem by itself.

---

### ✅ STRONGLY RECOMMEND: Hybrid Conditioning FIRST (Section 2.2)

**Codex proposes (but lower priority):**
> "keep literal prompts in place and inject DiT outputs via adapters or residual addition"

**Claude's argument for prioritization:**
1. **Fastest diagnostic test** (~2 GPU hours)
2. **Directly addresses root cause**: Tests if soft tokens override literal prompts
3. **Unblocks decision**: Determines if Phase 2 is viable at all

**Expected outcomes:**
- **If hybrid works** (bridged ≥ target): Issue is "soft token override" → proceed with tokenizer alignment
- **If hybrid fails** (bridged < target): Fundamental architectural mismatch → abandon Phase 2

**Recommendation**: Run THIS FIRST before any other Phase 2 work.

---

## Claude's Recommended Action Plan

### Priority 1: Hybrid Conditioning Test (2 GPU hours)
- Keep literal prompts fully intact
- Inject DiT outputs via learned adapters or residual addition
- **Decision criterion**: If bridged ≥ 51.5% (target baseline), continue Phase 2; otherwise abandon

### Priority 2 (if hybrid works): Tokenizer/RoPE Alignment
- Implement vocabulary projection layer
- Add RoPE phase alignment loss
- **Expected gain**: +5-10 pts

### Priority 3 (if hybrid + alignment work): Compression Tuning
- Optimize soft token count
- Fine-tune loss weights
- **Expected gain**: +2-5 pts

### DEFERRED: All Soft-Only Work
- Only pursue after soft+text reaches ≥target baseline
- Likely requires fundamental architectural changes

---

## Should Phase 2 Continue?

**Arguments FOR continuing:**
- 36% is better than 26% (progress being made)
- Tokenizer alignment might add 5-10 pts
- Hybrid conditioning might unlock target-level performance

**Arguments AGAINST continuing:**
- Phase 1 (Mistral→Llama) has better results (64.5% vs 36%)
- Phase 1 gap is smaller (-12.5 pts vs -15.5 pts)
- Limited GPU resources better spent on Phase 1 improvements
- Phase 2 translator actively degrades target performance

**Claude's recommendation:**
1. Run hybrid conditioning test FIRST (2 GPU hours)
2. If bridged < target → **abandon Phase 2**, refocus on Phase 1
3. If bridged ≥ target → continue with tokenizer alignment

**Do NOT spend 4×H100 hours on Phase 2 without hybrid test results.**

---

## Comparison with Phase 1 Results

| Direction | Source→Target | Best Bridged | Target-Alone | Gap | Viable? |
|-----------|--------------|--------------|--------------|-----|---------|
| Phase 1 | Mistral→Llama | 64.5% | 77.0% | -12.5 pts | ⚠️ Maybe |
| Phase 2 | Llama→Mistral | 36.0% | 51.5% | -15.5 pts | ❌ No |

Phase 1 is **24% closer** to target baseline than Phase 2.

**Strategic recommendation**: Prioritize Phase 1 over Phase 2 for GPU allocation.

---

## Action Items for User/Codex

1. **DECIDE**: Run hybrid conditioning test on Phase 2? (2 GPU hours)
2. **IF NO**: Update NEXT_STEPS.md to deprioritize/remove Phase 2 soft-only fixes
3. **IF YES**: Update NEXT_STEPS.md to reorder: hybrid (first) → tokenizer (second) → defer soft-only
4. **UPDATE docs**: Add Nov 19 results to LOG.md, EXPERIMENTS_SUMMARY.md with analysis
