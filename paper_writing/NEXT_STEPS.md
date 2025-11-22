# Next Steps — Finalized Execution Plan

## 0. Prep (Today)
- Sync to latest `main`, archive prior run logs, and reserve a 4× H100 slot.
- Double-check module stack (`gcc/13.1.0`, `conda/24.3.0-0`, `stockcuda/12.6.2`, `cudnn/cuda12/9.3.0.75`) before launching any job.
- Run a quick single-GPU smoke check (20 steps, `per_device_batch=1`, `--dit_steps_train=1`, decode loss off) to ensure the translator compiles/logs correctly before consuming H100 hours.
- For overnight single-GPU sweeps, use `bash paper_writing/run_phase2_single_gpu_suite.sh` (auto-runs prompt soft-only + answer soft-plus-text variants, copies each run into `paper_writing/preserved_data/`).

## Latest Result (Nov 20, 2025 — Phase 2 hybrid adapter)
- Config: `soft_injection=adapter`, prompt teacher, `soft_tokens=64`, `token_alignment_weight=0.1`, `adapter_scale=1.0`, eval prompt mode `soft_plus_text`.
- Outcome: Bridged 45.5–46.0% (plateaued through step 1000, early stop), target-alone 54.0%, source-alone 77.0%; invalids ~25% (48–50/200, step0 70/200). Run: `paper_writing/preserved_data/phase2_hybrid_adapter_phase2_swap_20251120_224319/`.
- Interpretation: Hybrid adapters reliably improve ~+8 pts over prompt-aligned but still **below target by 8.0–8.5 pts**. Decision criterion (≥51.5%) still not met → stay pivoted to Phase 1.

## 1. Phase 1 Status and Next Push
- **Latest:** 128-token run peaked 75.0%, final 73.5% (invalid ≤6%), gap to Llama target 3–4 pts. 96-token + light decode peaked 74.5%, final 73.0% (invalid ~4%), early-stopped at step 1000.
- **Goal:** Close the last 2–3 pts to ≥75–77%, maintain low invalid rate.
- **Next runs (overnight-ready, 4× H100):**
  1) **128-token short rerun (must complete):** Re-run the 1500-step schedule (no early stop) and ensure full evals; previous attempt truncated at ~step 340 with only step-250 metrics logged.
  2) **160-token headroom (optional):** `soft_tokens=160`, same stable weights, to test if a small capacity bump clears the remaining gap without destabilizing. Abort if invalids rise >10%.
- **If these miss ≥75%:** try smaller LR (8e-5) with 128 tokens and plateau decay at step 1000.

## 2. Phase 2 Status (still paused)
- Hybrid adapter reruns plateau at 45–46% (−8.5 pts); still below the 54% target.
- **Default:** continue pause until Phase 1 is ≥75%.
- **Optional single follow-up (only if idle GPUs):** answer-teacher + adapter_scale=0.3, token_alignment_weight=0.2; 2 hrs max; drop Phase 2 if still < target.
- **Deferrals:** soft-only curriculum, prompt contrastive, tokenizer/RoPE alignment remain on hold until a text-anchored setup beats target.

## 3. Phase 3 (only if GPU queue opens up)
- Compression sweep (64/128/256/512 soft tokens) using the stabilized codepath.
- Attention-pooling diagnostics and DiT + cross-attention hybrids.
- Broader evals (HotpotQA, MATH) and larger targets (Llama 70B) once GSM8K is solid.

## 4. Reporting
- After every run, update `paper_writing/CODEX_REPORT.md`, `LOG.md`, and `EXPERIMENTS_SUMMARY.md` with metrics plus JSONL paths.
- If conditional ablations fire, add a short appendix summarizing the comparative gains so Claude’s publication concerns are satisfied.

---

## CLAUDE'S ALTERNATIVE PLAN (Based on Nov 19 Results)

**Context (updated Nov 20):** Hybrid adapter test hit 45.5% vs target=54.0% (−8.5 pts). Translator still degrades target performance, though less severely than the Nov 19 runs.

### Critical Issue

| Direction | Best Bridged | Target | Gap | Status |
|-----------|--------------|--------|-----|--------|
| Phase 1 (Mistral→Llama) | 64.5% | 77.0% | -12.5 pts | ⚠️ Viable |
| Phase 2 (Llama→Mistral) | 45.5% | 54.0% | **-8.5 pts** | ⚠️ Still degrading (hybrid) |

**Phase 2 translator actively hurts target performance.** Need diagnostic before investing 4×H100.

---

### RECOMMENDED: Pivot to Phase 1

- Hybrid diagnostic missed the ≥51.5% bar; per decision tree, shift compute to Phase 1 until ≥75% bridged is achieved.
- Retain Phase 2 only for a single low-risk follow-up (answer-teacher + small adapter_scale) once Phase 1 jobs are underway; abort Phase 2 if it still underperforms target.

### Claude's Assessment of Codex's Plan (Section 2)

❌ **DISAGREE: Soft-only fixes remain premature**
- Soft+text/adapter still under target; pause soft-only curriculum until a text-anchored setup beats target.

⚠️ **PARTIAL: Tokenizer alignment might add 5–10 pts but won't erase the gap alone**
- Only consider after Phase 1 milestone or if the optional Phase 2 follow-up crosses the target baseline.

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
