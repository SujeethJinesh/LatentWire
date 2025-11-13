# Codex Report – 2025-11-13

## Overview
All eight experiments from `run_ablations.sh` completed under `paper_writing/runs/ablations_20251112_234456/`. Training now uses the same 8-shot GSM8K prompt as evaluation, so baseline (target-alone) accuracy stabilized at 1.000 for every eval. However, every bridged configuration ultimately collapsed to 0.0 accuracy. Only `1c_dit_attn_64tok` briefly reached 21.5% at step 750 before dropping to zero. Logs and JSONL outputs show bridged generations devolving into the memorized “museum” reasoning template instead of the question at hand, often missing the required `####` marker.

## Experiment Summaries
| Experiment | Key Variant | Peak Bridged | Final Bridged | Notes |
|------------|-------------|--------------|---------------|-------|
|1a_dit_2step_64tok|DiT steps_train=2|0.5%|0.0%|Bridge outputs remain template text from step 0 onward.|
|1b_dit_4step_64tok|DiT steps_train=4|0.5%|0.0%|Extra diffusion steps do not change behavior; bridged JSONL identical to 1a.|
|1c_dit_attn_64tok|DiT pool=attn|21.5%|0.0%|First run to break symmetry: bridged answers briefly matched gold (e.g., `bridged_extracted=2` for can-recycling prompt). Accuracy collapses after step 750 and never recovers.|
|1d_dit_cfg_64tok|DiT CFG=1.5|0.0%|0.0%|CFG adds noise and accelerates collapse.|
|1e_dit_prompt_teacher_64tok|Teacher=prompt, flow warmup=500|0.5%|0.0%|Prompt-teacher does not help; outputs revert to template.|
|2a_stable_64tok|Cross bridge + stability fixes|0.0%|0.0%|Despite RMS/InfoNCE/early-stop, bridged accuracy never rises.|
|3a_stable_32tok|32 token bridge|0.0%|0.0%|Higher compression fails immediately.|
|3b_stable_48tok|48 token bridge|0.0%|0.0%|Slightly better than 32, but still no correct answers.|
|2b_baseline_64tok|Legacy high-capacity cross run|81.5%|36.0%|Logged separately (successful_experiments); shows target behavior we’re trying to regain.|

## Detailed Findings
1. **Prompts now match, but translator fails to encode them.** Training and eval both wrap GSM8K in the 8-shot prefix. JSONL records confirm baseline answers solve the questions, yet bridged answers rarely reference the prompt. The soft tokens appear stuck representing a memorized reasoning chain (“museum exhibit” text). This indicates the DiT bridge is not learning from the long prompt despite prompt parity.
2. **Loss behavior mirrors failure:** In `1c_dit_attn_64tok`, loss steadily drops to ~1.1, but once bridged accuracy spikes at step 750 the loss also dips briefly. After that, loss and accuracy diverge: the LM loss keeps falling but bridged accuracy hits zero, showing the translator optimizes teacher-forced tokens but not the free generation objective.
3. **No numerical instability / OOM now:** With `per_device_batch=4` and eval batch=36, none of these runs reported CUDA OOM. GPU memory stayed under ~70 GB according to the logs.
4. **Bridged outputs still lack `####`:** Even when reasoning is correct (early steps of 1c), the `bridged_extracted` field is often `[invalid]` because the chain stops before the `####` marker. When it is present, the numeric answer is wrong 4/5 times, suggesting the translator never incorporates the final “please end with ####” instruction.
5. **Template collapse persists across variants:** Diffusion steps, attention pooling, CFG, prompt teacher, fewer tokens—all runs converge to the same `target_full` and `bridged_full` strings after ~500 steps. The repeated museum text in eval logs proves we’re stuck in a local minimum where the translator feeds Llama a fixed soft prefix regardless of question.

## Root Cause Hypothesis
- **Insufficient supervision on free-form generation:** Training optimizes NLL on ground-truth tokens with teacher forcing. During eval we force Llama to read only soft tokens (no text starter except BOS). There is no explicit loss tying the translator’s outputs to the actual prompt semantics once generation starts. Hence it memorizes a high-probability reasoning template to minimize LM loss, and the InfoNCE loss (using prompt embeddings) is too weak to prevent collapse.
- **DiT bridge never sees textual instruction tokens:** We entirely replace the full prompt with soft tokens during eval. If those tokens don’t perfectly encode “Answer the following question… end with ####”, Llama has no textual guidance and drifts into whatever reasoning it learned earliest. That explains the consistent template text.

## Recommendations
1. **Provide textual anchors during generation:** Instead of BOS-only targets in eval (`build_batch_inputs`), prepend a short hard prompt (“Solve the following GSM8K problem… end with ####”). Let soft tokens encode the few-shot examples + question, while the textual tail reminds Llama of formatting. This should dramatically increase chances of producing `####` and reduce template drift.
2. **Strengthen semantic supervision:** Add a contrastive loss between translator outputs and the actual text prompt (e.g., compare Llama hidden states for the real prompt vs. soft prompt). Alternatively, mix teacher-forced decoding with soft tokens + textual prompt so gradients directly penalize wrong answers.
3. **Monitor bridged accuracy during training:** Write the per-eval metrics to TensorBoard/CSV so we can detect collapse earlier and inspect when 1c achieved 21.5%. Use that checkpoint to analyze which questions were solved and why collapse followed.
4. **Validate translator outputs:** Add a debugging mode that decodes Llama with both text prompt and soft tokens simultaneously to visualize how much information the soft tokens encode. If they’re identical regardless of question (as current JSON suggests), we need additional conditioning (e.g., cross-attention residual pathways).

## Next Steps
- Run `paper_writing/scripts/validate_run.sh` on a single GPU to confirm the new early-stop logic and prompt parity code behave as expected before the next 4-GPU sweep.
- Implement the textual anchor + strengthened contrastive supervision, then rerun a single configuration (e.g., 1c) to see if bridged accuracy stays above zero.
- Once stable, rerun the full ablation list to obtain meaningful numbers for the paper.

## Response to CLAUDE_REPORT.md
Claude correctly flagged a severe evaluation bug: we log and score gold answers by running `extract_final_answer()` over the concatenated **prompt + solution**, so the first `####` token in the few-shot prefix (“#### 12”) becomes the “gold” answer for every sample. Because the accuracy loop compares predictions against that mis-extracted value, the reported `Target-alone acc: 1.000` is meaningless. After reviewing the JSONL files, I confirmed every `gold_extracted` entry is 12 regardless of the question, so I fully agree that the existing results cannot be trusted until this bug is fixed.

Where I differ slightly is on the root cause of the low bridged accuracy: now that training also uses the 8-shot prompt, the remaining collapse looks less like a prompt mismatch and more like missing textual anchors/weak supervision (the bridged generations still emit the museum template). That nuance aside, our action items now align:

1. **Fix gold extraction** so logging and metrics call `extract_final_answer()` on `sample.tgt_answer` only; keep the prompt separate for display.
2. **Re-run the promising configuration (1c)** once the evaluator is repaired, saving checkpoints around step 750 to understand why accuracy spikes to 21.5% and then collapses.
3. **Apply the training improvements from the previous section** (textual anchors, stronger contrastive loss, monitor bridged accuracy) so that, after the evaluator is trustworthy, we can meaningfully tackle the collapse problem.

Until the evaluation bug is fixed, neither Claude’s report nor this one can make quantitative claims. The good news is that both analyses now converge on the same next steps: repair gold extraction, rerun the key experiments, then iterate on the translator architecture with accurate metrics in hand.
