# HellaSwag Trained-SAE Scout and Validation[0:6144] Stress

## Status

The trained sparse-autoencoder common-basis branch is weakened for this cycle,
while the dense bagged hidden-innovation branch is strengthened.

Current paper story: the `2B` raw / `5B` framed source-private dense
hidden-innovation packet is still the only robust HellaSwag positive method.
PCA, anchor-relative bases, QJL/sign sketches, simple sparse dictionaries, and
now a bounded trained SAE scout do not yet support a strong common-basis claim.

## Why These Gates Were Run

The previous sparse residual dictionary scout ruled out simple k-means-like
feature words. This turn tested the next stronger common-language hypothesis:
train a sparse autoencoder on train-only source residuals and use its sparse
activations inside the same source-private packet selector.

In lay terms: instead of hand-making a small dictionary of hidden-state feature
words, we trained a little compressor to invent those feature words. It still
did not preserve the useful hint. Then we checked whether the original dense
hint keeps working on the next untouched chunk of HellaSwag.

## Trained-SAE Scout Result

Artifact:
`results/source_private_hellaswag_trained_sae_dictionary_scout_20260501_qwen05_validation4096_5120/hellaswag_sparse_residual_dictionary_scout.json`

Decision slice: HellaSwag validation rows `4096:5120`.

Best variant: `sae64_saedecision_relu_top8_dw0p2_l10p001`.

- accuracy: `0.499023`
- best label-copy: `0.500000`
- delta vs label-copy: `-0.000977`
- CI95 low vs label-copy: `-0.015625`
- score-only bagged: `0.497070`
- delta vs score-only: `+0.001953`
- dense hidden-innovation reference: `0.503125`
- delta vs dense reference: `-0.004102`
- scout pass: `false`

The reconstruction-only SAE and label-permuted decision SAE also failed. This
does not prove SAE/crosscoder-style alignment is impossible, but it rules out a
cheap one-layer train-only SAE as the missing common basis.

## Validation[5120:6144] Dense Stress

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation5120_6144/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=true`.

- selected accuracy: `0.555664`
- best label-copy: `0.512695`
- delta vs label-copy: `+0.042969`
- CI95 low vs label-copy: `+0.019019`
- score-only bagged: `0.501953`
- delta vs score-only: `+0.053711`
- zero-hidden delta: `+0.053711`
- wrong-example hidden: `0.478516`
- candidate-roll hidden: `0.454102`
- jackknife: `3/3`
- packet: `2B` raw / `5B` framed

This is the sixth contiguous HellaSwag validation slice to pass.

## Validation[0:6144] Aggregate

Artifact:
`results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_6144/hellaswag_hidden_innovation_multi_slice_stress.json`

Result: `pass_gate=true`.

- slices passing: `6/6`
- total eval rows: `6144`
- weighted selected accuracy: `0.511882`
- weighted best label-copy: `0.470052`
- weighted score-only: `0.464030`
- min delta vs best label-copy: `+0.034180`
- min CI95 low vs best label-copy: `+0.011719`
- min delta vs score-only: `+0.041016`
- min delta vs zero-hidden: `+0.041016`
- corrupted hidden controls below label-copy: `true`
- source-private packet: `true`

## Interpretation

Promoted:

1. Dense bagged hidden innovation is now stronger: it clears validation
   `0:6144`, not only `0:5120`.
2. The HellaSwag story is less likely to be prefix-slice luck.

Weakened:

1. Cheap trained SAE compression is not enough.
2. The paper still cannot claim a solved cross-model common language.

Still blocked:

1. Remaining HellaSwag validation rows `6144:10042`.
2. One strict cross-family falsification pair.
3. Native NVIDIA/vLLM/SGLang systems rows against target-only, text, C2C,
   KVComm, QJL, and TurboQuant-style state-transfer baselines.

## Next Gate

The next highest-value gate is another dense heldout validation slice,
`6144:7168`, using the same frozen train samples, split seeds, packet bytes, and
controls. If the remaining slices keep passing, HellaSwag becomes a much more
comfortable ICLR headline row. If dense validation fails, stop widening and use
the failure map to analyze where label-copy or score-only catches up.
