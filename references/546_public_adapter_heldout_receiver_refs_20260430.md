# Public Adapter Held-Out Receiver References

- date: `2026-04-30`
- purpose: primary-source memo for the public semantic-anchor teacher adapter
  and the next receiver-conditioned residual/codebook branch.

## Novelty Boundary

The new public adapter is a receiver-side calibration layer: it trains frozen
text features to predict public semantic-anchor coordinates from calibration
surfaces, then decodes source-private packets against held-out candidate text.
This is not a claim that anchors, adapters, query bottlenecks, or residual
codebooks are new by themselves.

Closest prior art:

- Relative Representations formalize anchor-relative common coordinates for
  latent-space communication and model stitching. Our safe claim must therefore
  be about source-private byte packets plus destructive controls, not the
  existence of anchors or relative coordinates.
  Source: https://arxiv.org/abs/2209.15430 and
  https://openreview.net/forum?id=SrC-nwieGJ
- C2C directly projects and fuses source KV cache into target KV cache for
  inter-LLM communication, with accuracy and latency claims. It is a direct
  competitor for broad "latent communication" framing.
  Source: https://arxiv.org/abs/2510.03215
- KVCOMM-style cache communication/reuse work is a systems competitor for
  multi-agent LLM serving. Our systems claim must stay on source-private bytes,
  no source text/KV exposure, and receiver-kernel accounting until GPU serving
  telemetry exists.
  Source: https://arxiv.org/abs/2510.12872
- QINCo learns implicit residual codebooks that depend on previous residual
  state. This motivates, but does not subsume, receiver-conditioned residual
  packets where the decoder codebook is conditioned on public candidate geometry.
  Source: https://arxiv.org/abs/2401.14732
- Perceiver IO, Flamingo, and BLIP-2/Q-Former establish learned query/adaptor
  bottlenecks between frozen or heterogeneous systems. We should use these as
  architectural precedent, not claim the adapter form itself as novel.
  Sources: https://arxiv.org/abs/2107.14795,
  https://arxiv.org/abs/2204.14198, and https://arxiv.org/abs/2301.12597
- TurboQuant and QJL are strong quantization/rotation baselines for compressed
  KV/vector state. Our current public adapter is not a KV compression method.
  Sources: https://arxiv.org/abs/2504.19874 and
  https://arxiv.org/abs/2406.03482

## Result-Guided Implication

The public adapter can produce large held-out matched-packet lifts, but the
permuted-teacher negative control also passes some individual rows. That means
the next technical contribution should not be "a learned public adapter" alone.
The stronger branch is a receiver-conditioned residual/codebook method with:

1. candidate-local normalization so shuffled source packets collapse;
2. deranged/permuted teacher negative controls in the main pass rule;
3. warm-path systems accounting that separates offline calibration from packet
   decoding;
4. no source text or source KV exposure.
