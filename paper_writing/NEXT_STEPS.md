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

## 1. Phase 1 Push (pivot path)
- **Goals:** Close the remaining 5–6 pt gap to target (77%) and reach ≥75% bridged.
- **Next runs (4× H100, ~2 hrs each):**
  1) **128-token DiT**: `soft_tokens=128`, prompt weight 0.001, dit_loss_weight 0.1, token_alignment_weight 0, eval_every=250. Expect +2–3 pts vs 96tok without destabilizing.
  2) **96-token refine with light decode loss**: reuse breakthrough config, add `--decode_loss_weight 0.02 --decode_interval 100 --decode_samples 2` to see if answer-formatting improves final accuracy; abort if memory >70 GB.
- **If both fail to reach ≥75%:** revert to 64-token stable config and adjust LR schedule (plateau LR decay after step 1000) as a low-cost follow-up.

## 2. Phase 2 Status (paused unless high-ROI change)
- Hybrid adapter diagnostic (rerun) again missed the ≥51.5% bar (45–46%); Phase 2 still degrades target performance.
- **Default:** pause Phase 2 until Phase 1 hits ≥75%.
- **Optional single follow-up (only if spare cycles):** answer-teacher + adapter injection with a gentler residual (`adapter_scale=0.3`, `token_alignment_weight=0.2`) to test whether answer supervision + smaller override helps fidelity. One shot, 2 hrs max; abort Phase 2 entirely if this also underperforms target.
- **Deferrals:** soft-only curriculum, prompt contrastive, and tokenizer/RoPE alignment remain deprioritized until a text-anchored setup beats target.

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
