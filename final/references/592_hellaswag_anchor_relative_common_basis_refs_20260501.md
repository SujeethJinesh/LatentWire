# HellaSwag Anchor-Relative Common-Basis References

## Result Boundary

The May 1 anchor-relative HellaSwag gate tests whether the dense
hidden-innovation packet survives a train-only common-basis bottleneck. It
does not promote a new method: across validation rows 0:5120, the
anchor-relative packet is `0.4695` accurate versus `0.4615` best label-copy and
`0.4564` score-only, but every 1024-row slice misses the strict `+0.02` margin
and label-copy CI rule. The dense hidden-innovation packet remains the live
positive HellaSwag branch at `0.5031`.

## Primary Related Work Boundary

- Relative representations show that anchor-relative coordinates can align
  independently trained latent spaces. This means our anchor-coordinate idea is
  not itself novel; the paper should claim only the stricter fixed-byte,
  source-private communication gate and its destructive controls.
  Source: https://arxiv.org/abs/2209.15430
- SAE universality and shared sparse-basis work motivate common language
  hypotheses for model internals, but do not establish our per-example
  `2B`/`5B` task packet. They are motivation and future-method baselines, not
  proof of novelty for anchor similarities.
  Sources: https://arxiv.org/abs/2410.06981, https://arxiv.org/abs/2502.03714
- Sparse crosscoders learn shared feature dictionaries across models/layers.
  They are the closest interpretability analogue for a future learned common
  basis. Our current result is weaker: train anchors preserve only a small
  HellaSwag lift and should be framed as a failed robustness gate.
  Source: https://transformer-circuits.pub/2024/crosscoders/index.html
- Prefix tuning and prompt tuning condition frozen models with learned soft
  prompt/prefix parameters. LatentWire differs because it sends a per-example
  fixed-byte candidate packet and does not expose source text, source KV, raw
  hidden vectors, or raw scores at the communication boundary.
  Sources: https://arxiv.org/abs/2101.00190, https://arxiv.org/abs/2104.08691
- C2C and KVComm/KVCOMM communicate or fuse source KV/cache state. Our
  boundary is much smaller and more private, but the anchor-relative failure
  means we cannot yet claim a general shared latent language on HellaSwag.
  Sources: https://arxiv.org/abs/2510.03215, https://arxiv.org/abs/2510.03346,
  https://arxiv.org/abs/2510.12872
- QJL and TurboQuant motivate hardware-friendly sketches and low-bit state
  compression. They are systems comparators and possible implementation
  inspirations; the current packet is not a KV/vector quantizer.
  Sources: https://arxiv.org/abs/2406.03482, https://arxiv.org/abs/2504.19874

## Reviewer-Safe Claim

Use this result to avoid overclaiming. The anchor-relative front end is
hardware-legible because it uses a static anchor bank and dot products before
emitting the same `2B` raw / `5B` framed packet, but it currently fails the
strict accuracy gate. The next shared-coordinate branch should test top-k/RBF
anchor features, spectral anchor graph coefficients, or a learned sparse
crosscoder-style basis against the same five-slice HellaSwag surface.
