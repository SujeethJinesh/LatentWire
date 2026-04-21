# Next Code Ablation After Span-ALM Negative

Scope: the smallest next implementable ablation that is materially different from local KL / interaction stacking, after the span-ALM teacher failed to rescue the controlled slice.

## Decision

The next ablation should be the existing **readout-distilled dynalign** path:

- `bridge_ridge_qk_readout_adapter`

If we want a more descriptive label for the paper narrative, this is the same lane as an `attnrefine`-style dynalign ablation, but the important part is that the code already exists and the supervision target is different from local KL or pairwise interaction loss.

Why this is the right smallest step:

- it reuses the current dynalign / span alignment harness;
- it changes the teacher target from token-mass / interaction stacking to **prompt-local readout geometry**;
- it already has parser, calibration, and translator wiring;
- it is cheap enough to run as a controlled GSM replay before any tokenizer surgery or new bridge family.

## Ranked next 5 attempts

| Rank | Method attempt | Compose with current span-ALM branch? | Cheap? | New code? | Why it is next |
|---|---|---|---|---|---|
| 1 | **Readout-distilled dynalign** (`bridge_ridge_qk_readout_adapter`) | Yes. It sits on the same `spanalign` / `ctxalign` / `dynalign` plumbing and only changes the supervision target. | **Yes** | **No** | Span-ALM already says token likelihood is too brittle; readout supervision is the smallest materially different teacher that is already implemented. |
| 2 | **Aligned-span preference distillation** (CTPD-style) | Yes. Reuses the same aligned-span machinery but changes the supervision target. | **Medium** | **Yes** | If readout supervision is not enough, the next step is a stronger span-level teacher that is still tokenizer-agnostic. |
| 3 | **Contextual dynamical mapping + soft alignment** | Yes. Best as a routing / correspondence layer feeding the teacher above. | **Medium** | **Yes** | The failure mode is now partly the correspondence itself, so the mapping must move with context. |
| 4 | **Attention-aware token initialization** | Partially. Useful once span supervision is working, as a target-side repair. | **Medium** | **Yes** | This is the smallest tokenizer-side repair before full vocab surgery. |
| 5 | **TokAlign / token alignment** | Yes, but only as a later compatibility layer. | **Low** | **Yes** | It is the heaviest tokenizer bridge and should stay after the span/readout stack. |

## Exact hook points

### `latent_bridge/calibrate.py`

Use the existing branch wiring that already handles:

- `bridge_ridge_qk_readout_adapter`
- `bridge_ridge_qk_attnkl_adapter`
- `bridge_ridge_qk_cab_adapter`
- `bridge_ridge_qk_emkd_adapter`
- `bridge_ridge_qk_dynalign_ctxonly_module_replace`
- `bridge_ridge_qk_dynalign_module_replace`
- `bridge_ridge_qk_dynalign_dwakd_module_replace`
- `bridge_ridge_qk_dynalign_likelihood_module_replace`
- `bridge_ridge_qk_dynalign_spanalm_module_replace`
- `bridge_ridge_qk_dynalign_dwainteract_module_replace`
- `bridge_ridge_qk_dynalign_interact_module_replace`

The relevant existing calibration hooks are already there:

- `collect_dynamic_prompt_position_mixtures(...)`
- `collect_aligned_prediction_teacher(...)`
- `collect_alignment_confidence_weights(...)`
- `collect_prediction_confidence_weights(...)`
- `collect_aligned_query_features(...)`
- `translator.set_bridge_sample_prompt_ids(...)`
- `translator.set_bridge_sample_query_features(...)`
- `translator.set_bridge_prediction_teacher(...)`
- `translator.set_bridge_sample_weights(...)`
- `translator.set_bridge_runtime_template_bank(...)`

For the readout branch, the important calibration side is that it already builds `readout_partner` tensors and passes aligned query features into the translator.

### `latent_bridge/translator.py`

The smallest code path is the existing query-residual fitter:

- `_fit_bridge_query_residual_adapter(...)`

It already supports the relevant knobs:

- `attention_kl_weight`
- `interaction_distill_weight`
- `readout_distill_weight`
- `prediction_distill_weight`
- `readout_partner`
- `readout_partner_kind`

That means the ablation is not a new bridge family. It is a different teacher objective on top of the same residual fitter.

### `latent_bridge/evaluate.py`

No new scoring math is required.

The only expected changes are:

- surface the new branch label cleanly in the report;
- keep the same exact-match / bytes / latency reporting;
- compare against the same held-out slice and the same nulls.

### `latent_bridge/README.md`

Update the branch list only if we add a new alias such as `bridge_ridge_qk_dynalign_attnrefine_module_replace`.
If not, the existing `bridge_ridge_qk_readout_adapter` entry is sufficient.

## Required tests

### Must-have

- `tests/test_calibrate_and_ablation.py`
  - parse-args smoke test for `bridge_ridge_qk_readout_adapter`
  - if an alias is added, a parse-args smoke test for the alias too

- `tests/test_translator_core.py`
  - `test_fit_from_pairs_bridge_ridge_qk_readout_adapter_populates_k_and_v_residuals`
  - add a shape/null-control assertion that the readout partner is required and that K/V residuals still populate when readout weight is on

- held-out GSM replay
  - run the readout branch on the same held-out Qwen pair
  - compare against `bridge_ridge_qk_dynalign_spanalm_module_replace`
  - compare against `bridge_ridge_qk_dynalign_ctxonly_module_replace`

### Nice-to-have

- seed-repeat / bootstrap check on the held-out split
- paired null row with `bridge_ridge_qk_attnkl_adapter`
- bytes / latency in the same table as accuracy

## Why this is not just another saturated teacher

Span-ALM already failed as a direct likelihood-mass teacher in the current readout, and local KL / interaction stacking has already been explored in the dynalign ladder.

This ablation is different because it moves the teacher from:

- token likelihood
- or prompt-local interaction similarity

to:

- **prompt-local readout geometry over aligned spans**

That is the smallest teacher-side change that is already implemented, still composes with the existing bridge, and is materially different from the saturated local KL / interaction stack.

## Practical run order

1. current baseline: `bridge_ridge_qk_dynalign_spanalm_module_replace`
2. next ablation: `bridge_ridge_qk_readout_adapter`
3. null control: `bridge_ridge_qk_dynalign_ctxonly_module_replace`
4. secondary contrast: `bridge_ridge_qk_attnkl_adapter`

## Bottom line

The next implementable ablation should be the existing **readout-distilled** path, not another KL or interaction stack. It is the smallest teacher change that is already wired through the calibration / translator code and it gives us a clean way to test whether prompt-local readout geometry helps where token-level likelihood does not.
