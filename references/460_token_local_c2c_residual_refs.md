# Token-Local C2C Residual References

- date: `2026-04-26`
- problem: C2C exposes Qwen2.5-Math -> Qwen3 SVAMP32 headroom, but source-only
  sidecars, source-hidden summaries, and token-local C2C residual query probes
  recover `0/6` clean C2C-only IDs.

## Sources

1. Zandieh, Daliri, Hadian, and Mirrokni. "TurboQuant: Online Vector
   Quantization with Near-optimal Distortion Rate." arXiv:2504.19874.
   https://arxiv.org/abs/2504.19874

   - Helps with: compression/transport design when the method has a real signal
     but needs a systems contribution.
   - Mechanism: random rotations plus scalar quantizers, then a 1-bit QJL
     residual correction to reduce inner-product bias.
   - Experiment impact: do not use it to rescue the failed C2C readout; use it
     later only if a source-derived vector signal clears controls and needs a
     byte/latency tradeoff.
   - Role: inspiration and systems ablation, not a fair baseline yet.

2. Google Research. "TurboQuant: Redefining AI efficiency with extreme
   compression." 2026.
   https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

   - Helps with: framing why compact vector communication needs both quality
     and systems evidence.
   - Mechanism: PolarQuant-style geometry simplification plus QJL sign-bit
     residual correction for KV/search bottlenecks.
   - Experiment impact: supports a future compress-after-signal gate, not more
     C2C trace probing.
   - Role: systems motivation.

3. Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
   Image Encoders and Large Language Models." ICML 2023 / OpenReview.
   https://openreview.net/pdf?id=KU9UojoX7U

   - Helps with: query bottleneck design between frozen modules.
   - Mechanism: learned query tokens extract task-relevant features from one
     frozen module before a projection into another frozen model.
   - Experiment impact: already tested as query-bottleneck residue probes; the
     token-local failure says the current supervision/features are wrong, not
     that all connector bottlenecks are impossible.
   - Role: inspiration and architectural precedent.

## Decision Update

The next branch should not be another C2C trace readout. The literature points
to two bounded uses only:

- if a deployable source-side signal appears, apply TurboQuant/QJL-style
  residual coding as a byte/latency ablation;
- if a connector branch is revived, change the supervision/source surface
  rather than only increasing query capacity.

