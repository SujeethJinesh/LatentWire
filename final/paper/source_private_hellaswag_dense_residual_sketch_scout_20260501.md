# HellaSwag Dense Residual Sketch Scout

## Status

Dense residual sign sketches are demoted for this cycle. They do not preserve
enough of the dense hidden-innovation signal on the strongest available
HellaSwag scout slice.

## Why This Gate Was Run

Anchor-relative and local anchor variants failed to recover the dense
hidden-innovation result. This scout tested a different common-basis hypothesis:
maybe a public random projection or QJL-style sign sketch of the source hidden
residual can preserve the sender's private decision evidence.

In lay terms: instead of translating hidden vectors through a learned map, we
measured them with the same random ruler on both sides and asked whether those
measurements still point to the right answer.

## Result

Artifact:
`results/source_private_hellaswag_dense_residual_sketch_scout_20260501_qwen05_validation4096_5120/hellaswag_dense_residual_sketch_scout.json`

Decision slice: HellaSwag validation rows `4096:5120`.

| Variant | Accuracy | Delta vs label-copy | CI95 low vs label-copy | Delta vs score-only | Scout pass |
| --- | ---: | ---: | ---: | ---: | --- |
| qjl_sign64 | 0.497070 | -0.002930 | -0.016138 | 0.000000 | false |
| qjl_sign128 | 0.500977 | +0.000977 | -0.013672 | +0.003906 | false |
| qjl_norm_sign128 | 0.501953 | +0.001953 | -0.011719 | +0.004883 | false |
| jl_value64 | 0.499023 | -0.000977 | -0.014648 | +0.001953 | false |
| jl_norm_value64 | 0.500000 | 0.000000 | -0.013672 | +0.002930 | false |

Best label-copy baseline: `0.500000`.
Score-only bagged control: `0.497070`.

## Interpretation

Simple random projections and sign sketches collapse to score-only or near
score-only on this slice. Wrong-projection and random-sign controls also
collapse, which is good for the control design, but the matched sketch does not
carry enough useful signal.

This weakens the hypothesis that the current dense hidden-innovation signal can
be recovered by a simple data-oblivious sketch. The next branch should use a
learned residual basis or sparse/crosscoder dictionary.

## Contribution Boundary

Do not claim QJL/JL/sign sketching as a new method. QJL and TurboQuant already
cover low-bit randomized vector-state compression. The only possible LatentWire
claim would be a source-private packet-selection interface using a sketch as a
local sender front end; this scout does not support that claim yet.
