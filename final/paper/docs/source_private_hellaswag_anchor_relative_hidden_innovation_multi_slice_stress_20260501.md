# HellaSwag Anchor-Relative Hidden-Innovation Multi-Slice Stress

- artifact: `results/source_private_hellaswag_anchor_relative_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_5120/`
- gate: `source_private_hellaswag_anchor_relative_hidden_innovation_multi_slice_stress`
- status: negative common-basis diagnostic, not a promoted method

## Result

The train-only anchor-relative version keeps the same source-private packet
contract as the dense HellaSwag hidden-innovation gate: `2B` raw / `5B` framed,
with no source text, source KV, raw hidden vectors, or raw scores transmitted.

Across the five frozen 1024-row HellaSwag validation slices:

- strict slices passed: `0/5`
- total rows: `5120`
- selected accuracy: `0.469531`
- best label-copy accuracy: `0.461523`
- score-only bagged control: `0.456445`
- weighted lift over best label-copy: `0.008008`
- weighted lift over score-only: `0.013086`
- dense hidden-innovation selected accuracy: `0.503125`
- anchor-relative gap versus dense hidden innovation: `-0.033594`
- min per-slice lift over best label-copy: `0.004883`
- min CI95 low versus best label-copy: `-0.014648`

The anchor-destroying controls and corrupted-hidden controls stay below
label-copy, so the diagnostic is not showing obvious leakage. The problem is
that the common-basis bottleneck removes too much of the dense hidden signal.

## Interpretation

This weakens the paper's common-basis story. The dense `2B` hidden-innovation
packet remains the strongest HellaSwag branch, but we should not claim that the
current anchor-relative coordinate system is a robust shared latent language.

For ICLR, this is still useful: it tells reviewers we tested the most obvious
"you just fit raw Qwen coordinates" objection and recorded the failure instead
of overclaiming. The next common-basis branch should be a bounded follow-up:
top-k/RBF anchor similarities, spectral anchor graph coefficients, or a learned
sparse crosscoder-style basis, all on the same five-slice decision surface.
