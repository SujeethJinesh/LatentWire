# Candidate-Local Cross-Family Gate References

- date: `2026-04-30`
- purpose: primary-source memo for the same-family vs cross-family separation
  artifact attached to the live candidate-local residual receiver.

## Claim Boundary

The cross-family gate supports a narrow statement:

> On the current n512 candidate-family packet surface, the live
> candidate-local residual receiver passes same-family and bidirectional
> cross-family rows under strict destructive controls, while RR anchor
> coordinates are clean but asymmetric.

It does not claim first latent communication, full dense latent transfer, or a
systems win over C2C/KVComm/TurboQuant. Those are separate comparisons.
Two simple RR repair probes are also measured now and do not change the
cross-family conclusion.

## Closest Prior Art

- Relative Representations are the closest anchor-coordinate latent
  communication prior. They show that similarities to anchors can create
  invariance to latent isometries and rescalings, enabling latent-space
  communication and model stitching. The current LatentWire RR row is therefore
  a serious baseline, not a strawman.
  Source: https://arxiv.org/abs/2209.15430
- LSTIRP / inverse relative projection turns relative spaces into direct
  latent-space translation. The current packet row is only a same-slice
  LSTIRP-lite diagnostic, not a full dense reproduction.
  Source: https://arxiv.org/abs/2406.15057
- Product-of-invariances and relative/geodesic representation papers reinforce
  the broader point that common bases can be constructed by quotienting away
  nuisance transforms. LatentWire should not claim anchor or invariance
  coordinates are new; the claim is the candidate-local side-information chart
  plus controls.
  Sources: https://arxiv.org/abs/2310.01211 and
  https://openreview.net/forum?id=4JnZvkVssS
- Latent Functional Maps and model stitching are nearby representation reuse
  methods. They motivate why reviewers will ask for dense latent-transfer and
  stitching comparisons.
  Sources:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/79be41d858841037987964e3f5caf76d-Abstract-Conference.html
  and https://openreview.net/forum?id=ak06J5jNR4
- C2C projects and fuses source KV cache into target KV cache, making it the
  closest direct semantic communication systems competitor. It is not
  privacy-equivalent to an 8B source-private packet because source KV is
  exposed.
  Source: https://arxiv.org/abs/2510.03215
- KVComm shares selected KV pairs across models and reports communication with
  a subset of layers' KV pairs. This is source-KV-visible and should be
  compared as a native systems baseline, not as the same boundary condition.
  Source: https://arxiv.org/abs/2510.03346
- TurboQuant is a strong recent quantization reference for vector/KV byte
  floors and inner-product preservation. It is a systems caveat until measured
  through a native KV sharing or serving path.
  Source: https://arxiv.org/abs/2504.19874

## Result-Guided Implication

Measured n512 separation:

- live candidate-local residual cross-family rows: `6/6` pass;
- live candidate-local residual same-family rows: `3/3` pass;
- live cross-family matched accuracy minimum: `0.500`;
- live cross-family best-control maximum: `0.260`;
- RR anchor-coordinate cross-family rows: `3/6` pass;
- RR core-to-holdout rows: `3/3` pass;
- RR holdout-to-core rows: `0/3` pass;
- RR same-family rows: `3/3` pass;
- RR innovation residual rows: `3/9` pass, with all gains confined to
  core-to-holdout and control leakage elsewhere;
- ranked RR innovation residual rows: `0/9` pass, with holdout-to-core below
  target.

Safe claim:

> The live row is not same-family-only on this packet surface; RR remains the
> strongest clean mathematical competitor but is direction-asymmetric. Simple
> RR innovation repairs did not make it bidirectional.

Unsafe claim:

> This does not prove transfer across unrelated real LLM families, defeat full
> dense RR/LSTIRP, or beat source-KV-exposing systems such as C2C/KVComm.
