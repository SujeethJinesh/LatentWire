# SVAMP32 Next Method: Conditional Residual Query Codec

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The live paper story is still blocked at the SVAMP32 exact-ID
gate: target self-repair reaches `14/32`, while the current matched source
rows fail to add source-necessary C2C-only recoveries.

Estimated distance to ICLR readiness: medium-high. The project still needs one
same-pair positive method that clears the clean residual gate, seed stability,
and one strict cross-family falsification pair before benchmark widening.

## Current Story

The current story is not yet "latent transfer works." It is:

- frozen SVAMP32 target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- best current source-conditioned matched rows: below target_self_repair and
  at `0/6` or `1/6` clean residual recovery

The useful clue is that C2C exposes ten target-missed teacher wins, of which
six remain clean residual IDs after target-self-repair and source-control
filtering. A publishable method must recover at least `2/6` of those clean IDs
without losing the target-self-repair floor.

## Saturated Or Negative

- scalar runtime gate retuning around query innovation
- clean-ID scalar over-weighting
- low-weight zero/shuffle contrastive source-control training
- simple one-gate, value-bank, and query-bank residual routing

These branches either fail to preserve target self-repair or fail to create
clean source-necessary residual recoveries.

## Best Concrete Idea

Build a target-self-repair-preserving conditional residual query codec.

Mechanism:

1. Run the target self-repair path as the protected base.
2. Add a tiny learned query bottleneck that cross-attends to matched source
   K/V traces and emits only a residual innovation sidecar, not a full
   replacement cache.
3. Condition the query codec on target-side self-repair state, so the source
   message is a Wyner-Ziv-style conditional delta: transmit what source adds
   given what target already knows.
4. Apply AWQ-style selective protection inside the sidecar: keep the top
   sensitivity query/channel atoms high precision and quantize/drop the rest.
5. Gate the sidecar with a verifier-style no-harm constraint: default to exact
   target_self_repair unless the sidecar predicts one of the clean C2C-only
   residual cases.

This is a hybrid of:

- multimodal Q-Former / Perceiver bottlenecks: learned queries bridge frozen
  backbones instead of assuming a shared latent basis
- quantization/selective precision: protect only the salient residual atoms
  rather than transporting a dense source cache
- decoder-side-information coding: encode source innovation conditional on
  target self-repair, not source state in isolation
- verifier-gated repair: make the source path opt-in, not always-on

Primary references already tracked in the repo:

- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
- Flamingo gated cross-attention / Perceiver-style resampler:
  https://arxiv.org/abs/2204.14198
- AWQ salient-channel protection: https://arxiv.org/abs/2306.00978
- Wyner-Ziv decoder-side-information coding:
  https://doi.org/10.1109/TIT.1976.1055508

## Why It Matters

This maps directly to the current blocker. The issue is not generic SVAMP
accuracy; target_self_repair already solves most of the reachable target-side
cases. The missing object is a small source-specific residual that activates on
the six clean C2C-only IDs while staying silent elsewhere.

The codec framing also explains why recent branches failed:

- scalar gates changed how much source signal entered, but not what signal was
  encoded
- clean-ID over-weighting made the same interface louder, not more conditional
- naive source-control contrast suppressed residual energy instead of learning
  a useful conditional code
- route-bank variants added capacity before solving no-harm preservation

## Minimal Experiment

Run the cheapest SVAMP32 exact-ID screen only; do not widen benchmarks.

1. Freeze source and target.
2. Use the existing SVAMP32 exact-ID artifacts and clean target set:
   `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`.
3. Train only a small residual query codec:
   - `8` or `16` learned queries
   - one cross-attention block over source top-layer K/V plus target
     self-repair summary
   - rank-`16` residual output into the existing target injection point
   - top-`k` protected query/channel atoms selected by target-loss sensitivity
4. Loss:
   - preserve target_self_repair logits on all target-self-repair-correct IDs
   - distill C2C teacher deltas only on target_self_repair-missed C2C-only IDs
   - penalize nonzero residual norm on zero-source and shuffled-source controls
   - add a margin only on clean residual IDs, not on every example
5. Decode matched exact IDs with a tiny gate sweep: no-sidecar, low, medium.
6. Analyze with `scripts/analyze_svamp32_paper_gate.py` in promotion mode.

Promotion criterion for this screen:

- preserve at least `14/32` target_self_repair correct
- recover at least `2/6` clean residual IDs
- at most `1` target-correct loss
- clean residual wins not retained by zero-source or shuffled-source controls

## Likely Failure Mode

The six clean residual IDs may be too few for the query codec to learn a stable
conditional code rather than memorizing ID-specific corrections. If matched
accuracy rises but zero/shuffled controls retain the same clean IDs, the branch
is not source communication; it is target-cache repair or leakage and should be
killed. If the sidecar causes more than one target_self_repair loss, the method
violates the no-harm requirement and should be redesigned as a stricter
oracle-gated/offline selector before more connector capacity is added.

## Decision

Promote this as the next single method branch. It is the highest expected-value
move because it attacks the exact `>=2/6` clean residual blocker while
explicitly preserving the target_self_repair floor. Do not spend another cycle
on scalar objective tuning or broad benchmark widening until this branch either
clears or fails the SVAMP32 paper gate.
