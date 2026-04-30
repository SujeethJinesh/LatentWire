# Candidate-Local Residual Receiver References

- date: `2026-04-30`
- purpose: primary-source memo for the candidate-local residual receiver that
  turns the public semantic-anchor adapter into a positive held-out packet gate.

## Novelty Boundary

The new claim is not that anchors, residual codebooks, or quantization are new.
The claim is narrower: a source-private byte packet is decoded only after the
receiver centers candidate public features into a local coordinate chart, and
the method is evaluated with shuffled, private-random, atom-deranged, and
permuted-teacher controls.

Closest prior art and safe boundaries:

- Relative Representations propose anchor-relative coordinates for latent-space
  communication and model stitching. LatentWire should cite this as the closest
  "common basis" precedent and claim novelty only for the source-private packet
  protocol plus candidate-local destructive controls.
  Sources: https://arxiv.org/abs/2209.15430 and
  https://openreview.net/forum?id=SrC-nwieGJ
- C2C directly communicates through projected/fused KV caches and is the direct
  competitor for broad "LLMs communicate beyond text" claims. LatentWire should
  not claim general superiority to C2C without matched hardware and benchmark
  runs.
  Source: https://openreview.net/forum?id=LeatkxrBCi
- KVComm/KVCOMM/Q-KVComm own cache sharing, reuse, and compression for
  multi-agent LLM systems. LatentWire's safe systems claim is the far-left-rate,
  no-source-text, no-source-KV point with byte/exposure accounting.
  Sources: https://openreview.net/forum?id=F7rUng23nw,
  https://arxiv.org/abs/2510.12872, and https://arxiv.org/abs/2512.17914
- QINCo learns residual codebooks conditioned on prior quantization state. This
  motivates receiver-conditioned residual decoding, but does not subsume the
  candidate-local side-information setting or the source-private controls.
  Sources: https://arxiv.org/abs/2401.14732 and
  https://openreview.net/forum?id=NBAc36V00H
- Perceiver IO, Flamingo, and BLIP-2/Q-Former establish query bottlenecks and
  frozen-system connectors. Use them as architectural precedent for local query
  maps, not as identical methods.
  Sources: https://arxiv.org/abs/2107.14795,
  https://arxiv.org/abs/2204.14198, and https://arxiv.org/abs/2301.12597
- TurboQuant and QJL give strong quantization/rotation systems baselines for
  vector and KV-cache compression. LatentWire currently does not optimize KV
  cache memory and should cite these only for compression/rotation baselines and
  future systems comparisons.
  Sources: https://arxiv.org/abs/2504.19874 and
  https://arxiv.org/abs/2406.03482
- Slepian-Wolf and Wyner-Ziv remain the right theory analogy: the packet is not a
  universal latent vector; it is a small side-information-coded syndrome whose
  meaning depends on public receiver state.
  Sources:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  and https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf

## Result-Guided Implication

Candidate-local residual normalization fixes the main public-adapter failure:
the receiver subtracts candidate-pool common signal before scoring packet atoms.
On the held-out synonym benchmark, this collapses the in-run permuted-teacher
receiver and private-random atom packets on the passing rows.

Safe paper claim:

> Candidate sets can serve as a local public basis for decoding source-private
> byte packets. In our held-out synonym gate, candidate-local residual scoring
> gives a learned public receiver bidirectional cross-family transfer while
> shuffled/private-random/permuted controls fall back to target-only.

Unsafe paper claim:

> This is not universal cross-model latent transfer, not a new residual
> quantizer, and not a replacement for C2C/KV-cache communication. The result is
> currently a Mac-scale positive method that still needs matched competitor
> baselines and larger seed repeats.
