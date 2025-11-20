# Phase 2 Status Report — Nov 19, 2025

This report summarizes every experiment completed so far, highlights the overnight Phase 2 suite, and captures the concrete next steps (aligned with `paper_writing/NEXT_STEPS.md`).

## 1. Executive Summary
- **Phase 1 + Ablations (Mistral→Llama)** remain stable: bridged accuracy 0.64–0.68 consistently beats the source model but still trails Llama by ~0.12.
- **Phase 2 (Llama→Mistral)** now has two preserved variants:
  - Answer-teacher + `soft_plus_text` reached **0.36** bridged accuracy (step 1250) and held **0.315** at step 1750 before the job timed out. Invalid outputs dropped to ~64/200; 63 answers matched the gold label in the latest eval JSONL.
  - Prompt-teacher + `soft_only` still collapses despite 1.0 prompt-alignment and DiT weights; bridged accuracy stays ≤0.025 with near-constant outputs (`#### 1`, `#### 10`).
- **Blocking issue**: the translator struggles to align Llama embeddings to Mistral without literal prompt context. We now have high-priority action items (curriculum, contrastive prompt loss, tokenizer/RoPE alignment) to address this.

## 2. Experiment Table

| Run ID | Config | Bridged Acc (peak → final) | Source / Target Acc | Sample Output (bridged) | Notes |
| --- | --- | --- | --- | --- | --- |
| `phase1_full_20251116_201212` | Phase 1 all fixes (KL + prompt alignment + RoPE, decode off) | 0.680 → 0.645 | 0.540 / 0.770 | “Janet eats 3 eggs… she makes $18. #### 18” | Stable baseline; artifacts in `paper_writing/preserved_data/phase1_full_20251116_201212/`. |
| `ablB_20251116_234242` | KL only (prompt/RoPE disabled) | 0.710 → 0.625 | 0.540 / 0.770 | Same GSM8K reasoning as Phase 1 but slightly less stable | Shows KL alone nearly recovers baseline; downstream collapse slightly worse (`…/ablB_20251116_234242/`). |
| `ablC_20251117_013909` | KL + prompt alignment (no RoPE) | plateau ≈0.615 → 0.655 | 0.540 / 0.770 | “Janet’s ducks lay 16 eggs… #### 18” | Prompt alignment is dominant; RoPE adds marginal gains (`…/ablC_20251117_013909/`). |
| `phase2_swap_20251118_192955` | Llama→Mistral, prompt teacher, `soft_plus_text` | 0.290 → 0.260 | 0.765 / 0.515 | Bridged outputs duplicate the 8-shot prompt | Duplicated literal question; led to destructive interference (`paper_writing/preserved_data/phase2_swap_20251118_192955/`). |
| `prompt_softonly_phase2_swap_20251119_001243` | Llama→Mistral, prompt teacher, `soft_only`, `prompt_alignment_weight=1.0`, `dit_loss_weight=1.0` | 0.025 → 0.025 | 0.765 / 0.515 | Dominant outputs “#### 1”, “#### 10” | Translator emits constant vectors despite stronger losses (`paper_writing/preserved_data/prompt_softonly_phase2_swap_20251119_001243/`). |
| `answer_softplus_phase2_swap_20251119_020705` | Llama→Mistral, answer teacher, `soft_plus_text`, standard weights | **0.360** (step 1250) → 0.315 (step 1750) | 0.765 / 0.515 | “Q: Janet’s ducks… #### 18” with ~70/200 correct | Clear progress vs older swap; job timed out before final eval; rerun with longer wall clock needed (`paper_writing/preserved_data/answer_softplus_phase2_swap_20251119_020705/`). |

### Sample Outputs
- **Phase 1 baseline** (from `phase1_all_fix/eval_samples_step_final.jsonl`):
  ```
  Q: Janet’s ducks lay 16 eggs per day…
  A: Janet eats 3 eggs for breakfast… She sells 9 eggs for $2 each, so she makes $18.
  #### 18
  ```
- **Phase 2 answer run** (step 1250 JSONL):
  ```
  Q: Josh decides to try flipping a house…
  A: Josh spent $80,000… Profit is $70,000.
  #### 70000
  ```
- **Phase 2 prompt soft-only run** (final JSONL):
  ```
  #### 10
  
  ###### Explanation:
  ```
  (same answer regardless of question, demonstrating collapse)

## 3. Analysis Highlights
- **Phase 2 answer run**: Invalid outputs dropped from 78 to ~64 per eval as training progressed. Gold matches peaked at 72/200 (step 1250). Loss curve shows steady decline around 1.3 and no divergence. Because the job timed out at step 2000, rerun with `--time >= 06:00:00` (or resume from checkpoint) is recommended to capture the final evaluation.
- **Phase 2 prompt soft-only run**: Invalid rate reached 0 %, but the model learned to output constant numerals. This indicates the prompt-alignment and DiT loss alone cannot anchor soft tokens to the input question; the bridge needs either textual scaffolding or a more informative alignment objective.

## 4. Action Items (mirror `NEXT_STEPS.md`)
1. **Re-run answer supervised Phase 2** with sufficient wall clock to complete the final evaluation (`answer_softplus_phase2_swap`); confirm whether bridged accuracy can hold ≥0.33 at convergence.
2. **Soft-only alignment upgrades** (new tasks):
   - Curriculum: start each run with `soft_plus_text` and fade to `soft_only` after 1 k steps.
   - Contrastive prompt loss (`--prompt_contrast_weight`) to punish constant outputs.
   - Auxiliary probe that predicts the source prompt embedding or category from the soft tokens.
   - Scale the format penalty with `prompt_alignment_weight` to keep invalid outputs suppressed.
3. **Tokenizer/RoPE alignment loss**: design a KL/phase-alignment objective so the DiT projection explicitly matches Llama’s positional/vocabulary geometry and is less sensitive to embedding mismatches.

## 5. References
- All preserved runs live under `paper_writing/preserved_data/` with full logs and JSONL dumps.
- JSONL diagnostics can be reproduced with `python paper_writing/scripts/summarize_eval_jsonl.py <path>`.
- `paper_writing/LOG.md` and `EXPERIMENTS_SUMMARY.md` contain line-by-line history if deeper code references are needed.
