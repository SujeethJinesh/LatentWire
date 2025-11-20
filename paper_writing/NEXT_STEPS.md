# Next Steps — Finalized Execution Plan

## 0. Prep (Today)
- Sync to latest `main`, archive prior run logs, and reserve a 4× H100 slot.
- Double-check module stack (`gcc/13.1.0`, `conda/24.3.0-0`, `stockcuda/12.6.2`, `cudnn/cuda12/9.3.0.75`) before launching any job.
- Run a quick single-GPU smoke check (20 steps, `per_device_batch=1`, `--dit_steps_train=1`, decode loss off) to ensure the translator compiles/logs correctly before consuming H100 hours.
- For overnight single-GPU sweeps, use `bash paper_writing/run_phase2_single_gpu_suite.sh` (auto-runs prompt soft-only + answer soft-plus-text variants, copies each run into `paper_writing/preserved_data/`).

## 1. Phase 1 Run — All Fixes at Once (ETA: 2 hrs GPU)
Implement in `paper_writing/cross_attention.py`:
1. **KL Slice Alignment** — compare answer tokens to answer tokens by re-tokenizing baselines with answer text and shifting the bridged logits past `soft_tokens + prompt_length`.
2. **Prompt Alignment Weight** — reduce coefficient from `0.05` to `0.001`.
3. **Decode-Aware Supervision** — leave the code path available but keep `decode_loss_weight=0` in the main Phase 1 job (recent runs OOM around step 400). Re-enable in Phase 1.5 ablations once the baseline finishes end-to-end.
4. **RoPE/Tokenizer Projection** — insert a learnable projection layer right after we capture Mistral hidden states to map the 32 768-token/roto-phase space into Llama’s 128 256-token geometry before diffusion.

After code changes:
- Run `git pull`, stage edits, and launch `PYTHONPATH=. bash paper_writing/run_ablations.sh` but **only** the Phase 1 config (1b_dit_4step_64tok_soft_plus_text) with the updated flags (`PROMPT_MODES=("soft_plus_text")`, `PER_DEVICE_BATCH=2`, `eval_batch=36`). Log run ID and GPU node.
- Success criterion: bridged accuracy ≥ 70 %.

## 1.5 Conditional Ablations (only if Phase 1 < 70 %)
- **Status:** Completed. Ablation B (KL only) final bridged 0.625; ablation C (KL + prompt alignment drop) final 0.655. Both runs kept decode loss disabled to avoid OOM. Artifacts preserved under `paper_writing/preserved_data/ablB_20251116_234242/` and `paper_writing/preserved_data/ablC_20251117_013909/` for reproducibility.
- **If rerun required:** use `paper_writing/run_ablation_B.sh` or `_C.sh` directly (same configs), keeping decode loss off. Only rerun if new code changes warrant revalidation.

## 2. Phase 2 Experiments (post-success)
1. **Bidirectional Swap:** run Llama 3.1 as source and Mistral as target using the stabilized translator; reuse Phase 1 hyperparameters. _Status:_ first attempt (Nov 18) with `dit_teacher=prompt` + `soft_plus_text` collapsed to 0.26 bridged accuracy because the soft tokens duplicated the literal question. Scripts now auto-switch to `soft_only` in this mode—relaunch after pulling to validate whether the translated prompt alone can lift Mistral past 0.515.
2. **Hybrid Conditioning Baseline:** keep literal prompts in place and inject DiT outputs via adapters or residual addition, verifying whether textual anchoring alone closes the remaining gap.
3. **Soft-only alignment fixes:**
   - Add a curriculum mode that starts training with `soft_plus_text` and linearly fades to `soft_only` after 1k steps so the DiT learns content-bearing soft tokens before we remove the literal prompt.
   - Introduce a contrastive prompt loss (InfoNCE between soft-token pools and target prompt embeddings) controlled by `--prompt_contrast_weight` so constant outputs are penalized.
   - Implement a lightweight probe/auxiliary loss that predicts the source prompt embedding (or category) from the soft tokens; this ties soft-only outputs back to the question even when no text is shown at inference.
   - Strengthen the format loss coupling when `prompt_alignment_weight` is large (e.g., scale the 0.1 coefficient accordingly) to keep invalid outputs suppressed.
4. **Tokenizer/RoPE alignment:** design an explicit loss that forces the DiT projection to mimic Llama’s positional/vocabulary geometry (e.g., align rotary phases or KL-match logits on a shared sub-vocab) so cross-model vocab differences don’t manifest as degenerate soft tokens.

## 3. Phase 3 (only if GPU queue opens up)
- Compression sweep (64/128/256/512 soft tokens) using the stabilized codepath.
- Attention-pooling diagnostics and DiT + cross-attention hybrids.
- Broader evals (HotpotQA, MATH) and larger targets (Llama 70B) once GSM8K is solid.

## 4. Reporting
- After every run, update `paper_writing/CODEX_REPORT.md`, `LOG.md`, and `EXPERIMENTS_SUMMARY.md` with metrics plus JSONL paths.
- If conditional ablations fire, add a short appendix summarizing the comparative gains so Claude’s publication concerns are satisfied.

---

## CLAUDE'S ALTERNATIVE PLAN (Based on Nov 19 Results)

**Context:** Nov 19 test showed answer_softplus=36% vs target=51.5% → **translator degrades performance by 15.5 pts**

### Critical Issue

| Direction | Best Bridged | Target | Gap | Status |
|-----------|--------------|--------|-----|--------|
| Phase 1 (Mistral→Llama) | 64.5% | 77.0% | -12.5 pts | ⚠️ Viable |
| Phase 2 (Llama→Mistral) | 36.0% | 51.5% | **-15.5 pts** | ❌ Degrading |

**Phase 2 translator actively hurts target performance.** Need diagnostic before investing 4×H100.

---

### RECOMMENDED: Validation-First Approach

#### Step 1: Hybrid Conditioning Test (2 GPU hrs) — DO THIS FIRST

**Goal:** Diagnose if "soft token override" causes degradation

**Implementation:**
- Keep literal prompts intact
- Inject DiT via learned adapters (not prepending to embeddings)
- Test if preserving text fixes degradation

**Decision Criterion:**
- ✅ bridged ≥ 51.5% → Continue Phase 2 (Step 2)
- ❌ bridged < 51.5% → ABANDON Phase 2 (Option B)

---

#### Step 2: IF Hybrid Works — Add Alignment (2 GPU hrs)
- Tokenizer/RoPE projection (32K→128K)
- Expected: +5-10 pts
- **Decision:** bridged ≥ 60% → Step 3, else → Option B

#### Step 3: IF Alignment Works — Full Training (4×H100)
- Target: bridged ≥ 60%

---

### OPTION B: Refocus on Phase 1 (If Phase 2 Fails)

Phase 1 baseline is **24% better** than Phase 2.

#### B1. Phase 1 Improvements (4×H100, 2 hrs)
- KL slice alignment
- Reduce prompt_alignment_weight → 0.001
- RoPE/tokenizer projection
- **Target:** ≥75% (within 2 pts of target)

#### B2. Compression Sweep (4×H100, ~6 hrs)
- Test 32/48/64/128 tokens
- Find optimal compression/accuracy tradeoff

#### B3. Publication
- Phase 1 ≥75% accuracy
- Compression analysis
- Ablations (done)

---

### Claude's Assessment of Codex's Plan (Section 2)

❌ **DISAGREE: Soft-only fixes (2.3) premature**
- Soft+text already underperforms by 15.5 pts
- Don't invest in soft-only when soft+text doesn't work
- **Defer until soft+text ≥ target**

⚠️ **PARTIAL: Tokenizer alignment (2.4) won't fix alone**
- Addresses real issue but won't close 15.5 pt gap
- Expected: +5-10 pts
- **Do AFTER hybrid test**

✅ **AGREE: Hybrid conditioning (2.2) CRITICAL**
- 2 hrs to diagnose root cause
- **DO THIS FIRST**

---

### Decision Flow

```
Hybrid Test (2h) → bridged≥51.5%? 
                      ├─YES→ Alignment(2h) → bridged≥60%?
                      │                        ├─YES→ Full training
                      │                        └─NO → Phase 1
                      └─NO → Phase 1 (safer investment)
```

---

### ACTION REQUIRED

**Choose:**
- [ ] Run hybrid test tomorrow (2 GPU hrs, data-driven)
- [ ] Skip Phase 2, do Phase 1 fixes (safer)

**Claude's vote:** Hybrid test first (2 hrs decides everything)

