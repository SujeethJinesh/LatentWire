# M26 Stable-Core Size Check

Generated: 2026-05-18 UTC

## Purpose

M26 stable-core protection is only worth preregistering if the stable core
`C` is neither too small to cover meaningful outlier mass nor so large that it
collapses into ordinary static protection. The authorized suitability band is:

- Too small: `|C| < 0.05%` of channels.
- Sweet spot: roughly `0.2%` to `0.8%` of channels.
- Too large: `|C| > 2%` of channels.

`C` is defined per model/layer as the intersection of prompt-averaged top-1%
channel sets across all measured decode positions.

## Method

For each source packet, I averaged channel magnitudes over prompts at each
decode position, selected the top 1% channels per layer/position, intersected
those sets across all available positions, then summed stable-core channels
over layers. This is the same prompt-averaged readout used in
`experimental/outlier_migrate/phase9/post_m18_analysis/always_protected_channels.md`,
with the additional conversion from "fraction of the top-1% set" to "percent
of all channels".

## Results

| Model | Source packet | Positions | Layers | Total channels | `|C|` channels | `|C|` percent of channels | Suitability |
|---|---|---:|---:|---:|---:|---:|---|
| Granite-4.0-H-Small | `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z` | 6 | 40 | 163840 | 847 | 0.516967773438% | suitable |
| Nemotron-3-Nano | `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z` | 6 | 52 | 139776 | 866 | 0.619562728938% | suitable |
| DeepSeek-R1-Distill-Qwen-1.5B | `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z` | 6 | 28 | 43008 | 228 | 0.530133928571% | suitable |
| Falcon-H1 | `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z` | 6 | 36 | 36864 | 226 | 0.613064236111% | suitable |

Layer-level ranges:

| Model | Mean core count per layer | Median core count per layer | Min `|C|` percent | Max `|C|` percent | Zero-core layers |
|---|---:|---:|---:|---:|---:|
| Granite-4.0-H-Small | 21.175000000000 | 21.000000000000 | 0.439453125000% | 0.659179687500% | 0 |
| Nemotron-3-Nano | 16.653846153846 | 16.500000000000 | 0.520833333333% | 0.818452380952% | 0 |
| DeepSeek-R1-Distill-Qwen-1.5B | 8.142857142857 | 8.000000000000 | 0.390625000000% | 0.781250000000% | 0 |
| Falcon-H1 | 6.277777777778 | 6.000000000000 | 0.488281250000% | 0.781250000000% | 0 |

## Decision

M26 is suitable to preregister and run on Granite-4.0-H-Small. The stable core
is in the intended range on all four measured models, and Granite-Small's
`|C| = 0.516967773438%` is large enough to test whether protecting stable
channels helps without degenerating into top-1% or top-2% static protection.

## Caveats

- This is a prompt-averaged activation readout, not a trace-bootstrap gate.
- The core is computed from existing calibration packets; it should not be
  interpreted as a causal proof that stable channels carry the recoverable
  perplexity gap.
- M26 must still include a random matched-size control and static-1% baseline,
  because a stable core of this size could still fail to cover the channels
  responsible for quality loss.
