# C4 Drift-Aware Pairing Gate

## Decision

Status: `NEEDS_ACTIVATION_CACHE`.

The cached block-output activation trajectories are enough to define and
compare deterministic pairing templates, but they are not enough to score the
exact ParoQuant rotation surface. ParoQuant rotates the last dimension of each
eligible weight tensor. The available caches are transformer block-output
channel magnitudes. They can show that drift-aware pairings would be materially
different from the current deterministic ParoQuant edge schedule, but not that
they improve quantization error.

## Candidate Pairings

All candidates are deterministic within each rotation group.

1. `paroquant_current`: current code path from
   `run_om_paroquant_baseline.build_pair_indices`, pairing channel IDs in the
   first half of each group with reversed IDs in the second half, with a shift
   by rotation index.
2. `static_activation_high_low`: sort channels by early calibration magnitude
   and pair high with low within each group.
3. `drift_complementary`: sort channels by relative late-minus-early drift and
   pair positive-drift channels with negative-drift channels.
4. `quantile_high_low`: sort channels by calibration quantile rank and pair
   high-quantile with low-quantile channels.
5. `anti_correlated_temporal_proxy`: use the cached per-position trajectory to
   estimate a temporal contrast and pair opposite contrast channels. This is a
   proxy for anti-correlation because full pairwise correlations are too large
   for the compact artifact.

## What The Cached Data Says

Using the first four layers of the cached block-output trajectories, every
candidate differs materially from the current ParoQuant edge schedule. Mean
edge-overlap with current ParoQuant was roughly:

| Model | static high-low | drift-complementary | quantile high-low | anti-correlated proxy |
|---|---:|---:|---:|---:|
| Granite | 0.0085 | 0.0071 | 0.0087 | 0.0074 |
| Nemotron | 0.0080 | 0.0078 | 0.0084 | 0.0061 |
| DeepSeek | 0.0120 | 0.0078 | 0.0120 | 0.0042 |
| Falcon-H1 | 0.0103 | 0.0088 | 0.0103 | 0.0059 |

This passes only the "materially different from baseline" screen. It does not
yet prove useful range reduction or recovery.

## Promotion Gate

Promote to GPU smoke only after collecting a rotation-surface activation cache:

- per eligible linear/expert weight tensor;
- tensor input activations before the weight multiply;
- early and late decode windows;
- enough channels to form full group-size pairings;
- current ParoQuant config included exactly as baseline.

Run the smoke only if the exact-surface candidate predicts at least one of:

- group range reduction greater than 5% relative to current ParoQuant;
- lower post-rotation CVaR range on known ParoQuant tail traces;
- materially different edge schedule with lower drift-coupled group variance.

## Caveat

The current block-output edge-overlap numbers should not be reported as a
method result. They are a queueing diagnostic.
