# SSQ-LR Preregistered Mac Gates

- date: 2026-05-06
- status: preregistered before measurement
- branch: recurrent SSM state quantization for hybrid reasoners

## S1: State Distribution Heterogeneity

Run a hybrid model on reasoning traces and dump recurrent SSM state at early,
middle, and late positions.

First admissible target: `ibm-granite/granite-4.0-h-tiny` with the revision
recorded in `experimental/shared/results/hybrid_model_eligibility_20260506/`.
If that target is not loadable on the current host, record the blocker and do
not substitute a dense model. The first packet must use at least 12 fixed
reasoning prompts, seed list `[1, 2, 3]` where stochastic decoding is used,
context buckets `{prefill_end, 2k_or_end, 8k_or_end, final_minus_128}`, BF16
reference state, and the shared `architecture_map_hash`.

Pass if state magnitude varies at least 2x across reasoning length with a
cluster bootstrap 95% lower bound above 1.25, or if early/late state
distributions differ by a two-sample test at Holm-corrected `p < 0.01`.
At least 25% of SSM layers, or at least three SSM layers when the model has few
SSM layers, must satisfy the criterion.

## S2: Quantization Sensitivity

Simulate INT8, FP8-style, and MXFP4-style quantization on SSM state only,
leaving weights and activations unchanged.

Pass if at least one recipe keeps quality within 1% absolute accuracy or a
matched continuation-NLL drift of at most 0.01 nats/token relative to the BF16
state baseline while reducing state storage by at least 4x. Required controls:
BF16 no-op state dump/replay with rel-L2 drift `<=1e-5`, INT8 state-only,
FP8-style state-only, MXFP4-style state-only, random same-L2 state noise, scale
shuffle, and byte accounting including scale/metadata overhead.

## S3: Cross-Model Robustness

Repeat the frozen S2 recipe on at least two additional hybrid or Mamba-family
models without retuning.

Pass if the recipe remains within 2% absolute accuracy or the preregistered NLL
tolerance on at least two of three models. The report must include paired
bootstrap intervals per model and an explicit no-retuning statement.

## Kill Rule

Kill if S1 finds no state heterogeneity, if all sub-FP16 recipes fail S2, or if
S3 requires per-model retuning to preserve quality.
