# Preserved Runs (Paper-Writing)

Every directory below mirrors the exact artifact layout that came off the HPC cluster (`train.log`, evaluation JSONL dumps, checkpoints). Use this README as the high-level manifest for why we kept each run, how close it is to paper-ready quality, and which script/command produced it.

---

## `phase1_full_20251116_201212` — Phase 1 Baseline (All Fixes)
- **Why we ran it**: First end-to-end verification that the DiT bridge with KL + prompt alignment + RoPE projection can finish 2k steps without decode-aware OOMs. This is the candidate “main result” for the paper.
- **HPC command**: `PYTHONPATH=. bash paper_writing/run_ablations.sh` (only the `phase1_all_fix` block enabled). Hardware: 4× H100 80 GB, `per_device_batch=2`, `eval_every=250`.
- **Key metrics** (from `phase1_all_fix/train.log`):
  - Peak bridged accuracy 0.680 at step 1500.
  - Final bridged accuracy 0.645 (source-alone 0.540, target-alone 0.770). Degradation: 0.035.
  - Stability achieved: no post-peak collapse, InfoNCE and early stopping behave as expected.
- **Publishability**: Serves as the control row in the main table (latent beats source but is still ~12 pts below target). Needs a follow-up experiment that either closes the 0.125 gap or proves the directionality hypothesis before we can claim information enrichment.
- **Artifacts to cite**:
  - Training log: `phase1_all_fix/train.log`
  - Eval dumps: `phase1_all_fix/eval_samples_step_{250,...,final}.jsonl`
  - Summary entry: `summary.log`

---

## `ablB_20251116_234242` — KL-Only Stack
- **Why we ran it**: Quantify how much of Phase 1’s success comes purely from KL re-targeting versus the newer prompt/RoPE alignment tricks. Acts as an internal control when reviewers ask for component ablations.
- **HPC command**: `PYTHONPATH=. bash paper_writing/run_ablation_B.sh` (same hardware + batch settings as Phase 1, but prompt/RoPE alignment disabled and decode loss kept at 0).
- **Key metrics**:
  - Peak bridged accuracy 0.710 at step 1750.
  - Final bridged accuracy 0.625 (degradation 0.085), source 0.540, target 0.770.
  - Observed recovery after step 1250 but lingering instability once eval reaches 2k steps.
- **Publishability**: Not paper-worthy on its own—it underperforms Phase 1 by ~2 pts and still lags target by ~15 pts—but the comparison proves that KL alone cannot deliver the stability we need. Keep this directory to justify why prompt/RoPE work mattered.
- **Artifacts**:
  - `ablB_kl_only/train.log`
  - `ablB_kl_only/eval_samples_step_*.jsonl`
  - `summary.log` with per-run metadata.

---

## `ablC_20251117_013909` — KL + Prompt Alignment (No RoPE)
- **Why we ran it**: Determine whether prompt-alignment, not RoPE projection, is carrying the stability gains. This answers the reviewer-facing question “do I really need the RoPE matching layer?”
- **HPC command**: `PYTHONPATH=. bash paper_writing/run_ablation_C.sh` (prompt alignment on, RoPE projection off, decode loss 0). Same compute footprint as above.
- **Key metrics**:
  - Bridged accuracy hovered 0.605–0.615 throughout training and recovered to 0.655 on the final checkpoint.
  - Source 0.540, target 0.770. Degradation <0.01 from mid-training, indicating high stability even without RoPE.
  - Confirms that prompt alignment is the dominant contributor; RoPE projection adds ~2 pts at best.
- **Publishability**: Close to the Phase 1 row and therefore useful for the ablation table, but still below target. We will reference it in the paper to argue that prompt anchoring is the minimal set of fixes.
- **Artifacts**:
  - `ablC_kl_prompt/train.log`
  - `ablC_kl_prompt/eval_samples_step_{0,...,1500,final}.jsonl`
  - `summary.log`

---

## `phase2_swap_20251118_192955` — Bidirectional Swap (Llama ➞ Mistral, Prompt Teacher)
- **Why we ran it**: Test the “strong-to-weak guidance” hypothesis by swapping the bridge direction (Llama source at 76.5% → Mistral target at 51.5%) while supervising the DiT on prompt embeddings instead of answers.
- **Command**: `PYTHONPATH=. bash paper_writing/run_phase2_swap.sh` (default settings at the time forced `eval_prompt_mode=soft_plus_text` even under `dit_teacher=prompt`, later identified as a bug).
- **Key metrics** (see `phase2_swap_all_fix/train.log`):
  - Source-alone accuracy: 0.765 (Llama 3.1 Instruct)
  - Target-alone accuracy: 0.515 (Mistral Instruct)
  - Bridged accuracy: peaked at 0.290 (steps 750 & 1500), final 0.260
  - Invalid generations: 100% at step 0, ~30–40% after 250+ steps (see JSONL counts)
- **What went wrong**:
  - The prompt-supervised DiT reproduced the question text, but we also kept the literal question in `soft_plus_text` mode. Mistral therefore saw `[translated question || original question]`, treating the soft tokens as noisy duplicates and regressing below its solo baseline.
  - All `bridged_full` outputs look like re-encoded questions rather than helpful latent prompts; no answer supervision was present in this configuration.
  - Directory name initially included quotes because `RUN_ID` was quoted; fixed for future runs but this snapshot already exists.
- **Publishability**: Not directly (bridged < target), but preserved as evidence that prompt-supervised DiT requires `soft_only` decoding to avoid destructive interference. We will cite it when motivating the script fix that auto-selects `soft_only` whenever `dit_teacher=prompt`.
- **Artifacts**:
  - `phase2_swap_all_fix/train.log`
  - `phase2_swap_all_fix/eval_samples_step_{0,250,500,750,1000,1250,1500,final}.jsonl`
  - `summary.log`

---

## `phase2_swap_20251118_213543` — Prompt Teacher + Soft-Only (Single H100)
- **Why we ran it**: Repeat the bidirectional swap with the corrected `soft_only` evaluation so the prompt-supervised DiT is the *only* context Mistral sees. This run executed on a single H100 (DDP world size 1) to validate the overnight workflow.
- **Command**: `NUM_GPUS=1 PYTHONPATH=. DIT_TEACHER=prompt PROMPT_MODE=soft_only bash paper_writing/run_phase2_swap.sh`.
- **Key metrics** (`phase2_swap_all_fix/train.log`):
  - Source-alone 0.765 (unchanged), target-alone 0.515.
  - Bridged accuracy never exceeded 0.005 (step 500) and early-stopped immediately afterward.
  - `eval_samples_step_0.jsonl`: 100% `[invalid]` answers; by step 500 only 15/200 samples emitted any digits and most outputs repeated `#### 1000`.
- **Root cause**: With soft-only prompting, the prompt-supervised DiT failed to produce useful latent prompts—most embeddings collapsed to a constant value, so Llama never answered the GSM8K question. We need either answer supervision or a more informative conditioning path before reattempting.
- **Artifacts**:
  - `phase2_swap_all_fix/train.log`
  - `phase2_swap_all_fix/eval_samples_step_{0,250,500}.jsonl`
  - `summary.log`

---

**How to use these entries**:
1. Link the relevant directory whenever citing numbers in `PLAN.md`, `EXPERIMENTS_SUMMARY.md`, or the paper draft.
2. When planning new HPC jobs, note why each preserved run failed to beat the target model so we can justify the next configuration (e.g., bidirectional swap, hybrid conditioning).
3. If any future run surpasses target accuracy, create a new subsection here with the same level of detail so reviewers have a lineage of publishable metrics.
