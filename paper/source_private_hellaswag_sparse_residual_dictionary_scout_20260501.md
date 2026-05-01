# HellaSwag Sparse Residual Dictionary Scout

## Status

The sparse residual dictionary branch is demoted. Both the unsupervised
candidate-residual dictionary and the train-label contrastive dictionary fail on
the same HellaSwag decision slice where `pca256` passed.

## Why This Gate Was Run

Dense hidden innovation passes the five-slice HellaSwag gate, while anchor
coordinates, QJL/JL sign sketches, and PCA residual bases do not yet support a
stable common-basis claim. This gate tested the next cheapest common-language
hypothesis: learn a small train-only sparse dictionary of residual directions
and use sparse atom activations inside the sender-side packet selector.

In lay terms: instead of sending a model's whole private hunch, we asked whether
the hunch can be summarized as a few learned "feature words." The answer here is
no for simple clustered dictionaries.

## Unsupervised Dictionary Result

Artifact:
`results/source_private_hellaswag_sparse_residual_dictionary_scout_20260501_qwen05_validation4096_5120/hellaswag_sparse_residual_dictionary_scout.json`

Decision slice: HellaSwag validation rows `4096:5120`.

Best variant: `dict128_cand_signed_top4`.

- accuracy: `0.497070`
- best label-copy: `0.500000`
- delta vs label-copy: `-0.002930`
- CI95 low vs label-copy: `-0.016138`
- score-only bagged: `0.497070`
- delta vs score-only: `+0.000000`
- dense hidden-innovation reference: `0.503125`
- delta vs dense reference: `-0.006055`
- scout pass: `false`

This rules out simple unsupervised sparse clustering of candidate residuals as a
useful replacement for the dense hidden selector.

## Contrastive Dictionary Result

Artifact:
`results/source_private_hellaswag_sparse_residual_dictionary_contrastive_scout_20260501_qwen05_validation4096_5120/hellaswag_sparse_residual_dictionary_scout.json`

Best variant: `dict64_gold_signed_top4`.

- accuracy: `0.498047`
- best label-copy: `0.500000`
- delta vs label-copy: `-0.001953`
- CI95 low vs label-copy: `-0.015161`
- score-only bagged: `0.497070`
- delta vs score-only: `+0.000977`
- dense hidden-innovation reference: `0.503125`
- delta vs dense reference: `-0.005078`
- scout pass: `false`

The supervised atom sources were:

- `gold_residual`
- `gold_minus_top_wrong`
- `gold_minus_wrong_mean`

Even these contrastive train-label atom dictionaries do not recover the PCA
single-slice lift, so another k-means / top-k sparse-code tweak is unlikely to
be a high-value branch.

## Systems Trace

Both scouts preserve the source-private packet contract:

- packet: `2B` raw / `5B` framed
- source text exposed: `false`
- source KV exposed: `false`
- raw hidden vector transmitted: `false`
- raw score vector transmitted: `false`
- raw sparse code transmitted: `false`
- dictionary atoms transmitted at runtime: `false`
- dictionary public/preloaded: `true`
- native kernel status: `mac_python_trace_only`

Selected-variant dictionary residency:

- unsupervised best: `3.9375` MiB across the nine bagged selector components
- contrastive best: `1.96875` MiB across the nine bagged selector components

This is a systems accounting row, not a native speed claim.

## Interpretation

Ruled out:

1. Simple unsupervised residual clustering is not enough.
2. Simple train-label contrastive atom dictionaries are not enough.
3. Sparse top-k atom features do not explain the dense hidden-innovation lift on
   the strongest single slice.

Still alive:

1. Dense hidden innovation remains the only fully promoted positive HellaSwag
   method.
2. PCA remains useful as a diagnostic that learned coordinates can matter, but
   not as a stable method.
3. A real SAE/crosscoder objective remains alive only if it jointly optimizes
   reconstruction, sparsity, source-private decision lift, and atom knockout.

## Next Gate

The next common-basis branch should not be another static dictionary tweak. It
should be a trained sparse/crosscoder objective with:

- reconstruction loss on train-only residuals
- decision loss against the source-private candidate selector
- label-permutation control
- atom-ID shuffle and atom-value shuffle controls
- top-atom knockout showing causal dependence
- dictionary public/preloaded systems accounting

If that is too costly on Mac, the paper should temporarily focus on the dense
hidden-innovation method plus the negative common-basis ladder for COLM, while
leaving SAE/crosscoder as the ICLR extension.
