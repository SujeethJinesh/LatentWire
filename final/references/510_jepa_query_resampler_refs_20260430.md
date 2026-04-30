# JEPA Query Resampler Receiver References

- date: `2026-04-30`
- purpose: primary-source grounding for the `jepa_query_resampler` receiver
  smoke and the decision not to promote the first formulation.

## BLIP-2 / Q-Former

- source: https://arxiv.org/abs/2301.12597
- blocker helped: the semantic-anchor receiver is too hand-designed; the paper
  needs a learned connector precedent.
- mechanism idea: use a small query bottleneck between a frozen/source-side
  representation and a target-side model interface.
- next experiment change: implement candidate-conditioned queries over decoded
  packet atom keys/values rather than another global bilinear map.
- role: architecture inspiration / paper framing.

## Flamingo / Perceiver Resampler

- source: https://arxiv.org/abs/2204.14198
- blocker helped: cross-model communication needs a fixed-size interface from
  variable source state to a target consumer.
- mechanism idea: resample source packet atoms into a small learned latent set
  and measure accuracy against byte/query-count budgets.
- next experiment change: report query count and hidden dimension as the
  receiver rate/capacity axis.
- role: architecture inspiration / systems framing.

## Perceiver IO

- source: https://arxiv.org/abs/2107.14795
- blocker helped: supports the idea that cross-attention queries can map
  structured inputs to structured outputs without token-level alignment.
- mechanism idea: candidate features generate queries over source packet atom
  keys/values; the attended contexts score candidate compatibility.
- next experiment change: add candidate-conditioned attention and query
  entropy/effective-rank diagnostics.
- role: architecture inspiration / ablation design.

## I-JEPA

- source: https://arxiv.org/abs/2301.08243
- blocker helped: motivates predicting useful target-side latent state rather
  than reconstructing private source text.
- mechanism idea: train the receiver to score target candidate latents from a
  source-private packet under destructive negatives.
- next experiment change: keep a latent-prediction interpretation, but require
  the result to beat target and controls before calling it communication.
- role: objective framing.

## VICReg / Anti-Collapse Diagnostics

- source: https://openreview.net/forum?id=xm6YD62D1Ub
- blocker helped: query bottlenecks can look safe only because they collapse to
  target-only behavior.
- mechanism idea: record query entropy, effective rank, and context variance.
- next experiment change: reject any query-resampler win with collapsed query
  rank or zero context variance.
- role: diagnostic / anti-collapse support.

## Bottom Line

The sources justify trying a query-resampler connector, but the first local
smoke is negative: it gives asymmetric signal and clean query-rank telemetry,
but it does not clear bidirectional held-out controls or oracle/headroom
requirements. This weakens the current random-feature JEPA-Q formulation; a
future attempt needs trained query/key/value factors or a stronger feature
source, not more threshold tuning.
