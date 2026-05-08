# SSM Shape-Conditioned Codec Phase 0 Preregistration

**Frozen on**: 2026-05-08
**Frozen by**: swarm pivot after `KILL_SSML_PHASE0_STATE_STABLE`
**Pivot source**: `experimental/ssm_lifecycle/diagnostic.md`
**Pivot depth**: 1 from SSM-State Lifecycle

## Status note

This is a fresh preregistration in a new branch directory. It does not modify
or relax `experimental/ssm_lifecycle/phase0/preregister_ssml_phase0.md`.

The killed SSM-State Lifecycle gate showed statistically detectable state
distribution shifts but no preregistered 2x magnitude drift. This pivot tests a
different positive-method hypothesis: whether shape-conditioned state
codebooks, rather than age-based magnitude scaling, reduce reconstruction error
for SSM state compression on held-out traces.

## Hypothesis

For Granite-4.0-H-Tiny SSM states, the shape of the state distribution differs
enough between early and late decode positions that age-specific 4-bit
quantile codebooks reconstruct held-out SSM state tensors better than a single
age-agnostic 4-bit quantile codebook.

## Model

- `ibm-granite/granite-4.0-h-tiny`

## Promptset

- Source: AIME-2025
- File: `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Held-out indices: 12-23 only
- Calibration prompts: indices 12-17
- Test prompts: indices 18-23
- The killed SSM-State Lifecycle run used indices 0-11. This pivot must not
  inspect indices 12-23 before the runner executes.

## Procedure

1. Capture real `past_key_values.ssm_states` from Granite-4.0-H-Tiny for all
   36 Mamba layers at decode positions `{100, 10000}` on held-out AIME-2025
   indices 12-23.
2. For each Mamba layer, fit 16-value quantile codebooks on calibration
   prompts:
   - **Age-agnostic baseline**: one pooled codebook fitted from positions 100
     and 10000 together.
   - **Shape-conditioned method**: separate codebooks for position 100 and
     position 10000.
3. Evaluate both codecs only on test prompts 18-23. For every
   `(prompt, layer, position)` test tensor, quantize each state element to the
   nearest codebook value and compute normalized MSE:
   `mean((state - reconstructed)^2) / (mean(state^2) + 1e-12)`.
4. Compute relative NMSE reduction:
   `(baseline_nmse - method_nmse) / baseline_nmse`.
5. Aggregate by test prompt, then bootstrap over the six held-out test prompts
   with seed `20260508` and `n=1000`.

## Decision rule

### PASS (decision string: `PASS_SSC_PHASE0_SHAPE_CODEC_GAIN`)

All of the following must hold:

1. Mean relative NMSE reduction across test prompts is at least `0.10`.
2. Bootstrap 95% CI lower bound is greater than `0.05`.
3. The method is not worse than baseline on either evaluated position when
   averaged over layers and test prompts.
4. Artifact checker exits 0.

### KILL (decision string: `KILL_SSC_PHASE0_NO_CODEC_GAIN`)

Any of the following triggers a kill:

1. Mean relative NMSE reduction is below `0.10`.
2. Bootstrap 95% CI lower bound is not greater than `0.05`.
3. The method is worse than baseline on either evaluated position.
4. Packet cannot be made artifact-complete.

## Forbidden actions

- Using AIME indices 0-11 for this pivot's decision.
- Tuning thresholds after observing indices 12-23.
- Changing calibration/test split after observing indices 12-23.
- Replacing 4-bit quantile codebooks with a different codec after observing
  indices 12-23.
- Treating a PASS as a decode-quality claim. This Phase 0 gate is an offline
  state reconstruction gate only.

## On pass

Author a Phase 1 preregistration before running any decode-quality experiment.
Phase 1 must test whether the shape-conditioned codec preserves greedy decode
quality and transfers to at least one larger or cross-family hybrid.

## On kill

Write `experimental/KILLED_ssm_shape_codec_no_gain/README.md` with artifact
SHAs, decision metrics, and the reason classification. Do not write a paper for
the killed pivot.
