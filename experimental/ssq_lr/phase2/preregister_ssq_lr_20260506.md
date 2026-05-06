# SSQ-LR Preregistered Mac Gates

- date: 2026-05-06
- status: preregistered before measurement
- branch: recurrent SSM state quantization for hybrid reasoners

## S1: State Distribution Heterogeneity

Run a hybrid model on reasoning traces and dump recurrent SSM state at early,
middle, and late positions.

Pass if state magnitude varies at least 2x across reasoning length or if
early/late state distributions differ by a preregistered two-sample test at
`p < 0.01`.

## S2: Quantization Sensitivity

Simulate INT8, FP8-style, and MXFP4-style quantization on SSM state only,
leaving weights and activations unchanged.

Pass if at least one recipe keeps quality within 1% absolute accuracy or a
matched NLL tolerance of the BF16 state baseline while reducing state storage by
at least 4x.

## S3: Cross-Model Robustness

Repeat the frozen S2 recipe on at least two additional hybrid or Mamba-family
models without retuning.

Pass if the recipe remains within 2% absolute accuracy or the preregistered NLL
tolerance on at least two of three models.

## Kill Rule

Kill if S1 finds no state heterogeneity, if all sub-FP16 recipes fail S2, or if
S3 requires per-model retuning to preserve quality.
