# ARC Soft-Prefix Residual Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, public-basis benchmark
  gates, and systems byte/exposure accounting are defensible.
- Exact gap: a positive source-necessary target-loss soft-prefix/query
  connector is still missing.

## Gate

Updated script:
`scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`

New feature modes:

- `hashed_selected_residual`
- `hf_selected_hidden_residual`

The residual mode computes selected-choice feature minus the row mean over all
candidate choices, then L2-normalizes it. This mirrors the row-centered
candidate residuals used by the packet gates, but feeds the resulting feature
into a target-loss soft-prefix connector.

Artifact:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_qwen_hidden_residual_n8_cpu_label_choice/`

## Result

Qwen-source-hidden-residual ARC n8 CPU smoke, `label_and_choice` continuation:

- pass gate: `False`
- fit/eval rows: `4 / 4`
- matched soft-prefix accuracy: `0.250`
- target-only accuracy: `0.250`
- slots-only/static prefix accuracy: `0.500`
- zero-source accuracy: `0.000`
- shuffled-source accuracy: `0.250`
- same-norm noise accuracy: `0.250`
- label-shuffled accuracy: `0.250`
- same-byte visible text accuracy: `0.250`
- source-label-copy audit upper bound: `0.750`
- matched mean margin: `-0.345896`
- best pass-control margin: `-0.195899`
- matched minus best-control margin: `-0.149997`
- runtime: about `245.7s`
- peak RSS: about `7.1 GiB`

Compared with absolute selected hidden features, residualization improved
matched accuracy from `0/4` to `1/4` and improved the margin deficit from about
`-0.751` to `-0.150`, but it still lost to the slots-only/static prefix.

## Decision

Promote row-centered residual source features as better than absolute selected
hidden features, but do not widen this exact selected-vector connector yet. It
does not pass source necessity, and CPU runtime is too high for meaningful
seed repeats on the Mac.

The next exact method branch should be tokenwise/query pooling:

- learned 16-64 query tokens over all source candidate token states;
- explicit absolute-vs-residual-vs-score-only ablation;
- static prefix and shuffled/zero/noise controls;
- n64 or larger validation slice with three seeds;
- NVIDIA if available.

## Lay Explanation

The previous run gave the target model a clue about the source model's chosen
answer in isolation. This run instead gave it the difference between the chosen
answer and the other answers. That helped a little, but not enough: a generic
learned prefix still did better, so the connector is not yet proving real
source-to-target communication.
