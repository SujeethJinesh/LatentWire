# References: CommonsenseQA Score-Packet and Fusion Probe

Date: 2026-05-01

## Primary Benchmarks

- CommonsenseQA: https://arxiv.org/abs/1811.00937
  - Relevance: this is the non-science commonsense QA validation probe used in
    the score-packet and fusion artifacts.
  - Boundary: the current result is diagnostic only because same-byte text
    saturates the source top-label signal.

- HellaSwag: https://arxiv.org/abs/1905.07830
  - Relevance: recommended next non-science gate because it is a four-choice
    commonsense completion benchmark with longer candidate endings.
  - Boundary: use it as candidate-selection generality, not as open-ended QA.

- OpenBookQA: https://arxiv.org/abs/1809.02789
  - Relevance: current promoted second benchmark; useful contrast because the
    3B packet beats same-byte text on validation/test seed stability.

## Closest Communication and Systems Baselines

- Cache-to-Cache (C2C): https://openreview.net/forum?id=LeatkxrBCi
  - Relevance: direct LLM-to-LLM communication through projected/fused KV cache.
  - Boundary: C2C exposes dense cache semantics and uses a learned projector;
    LatentWire's current result is a few-byte source-private task packet.

- KVCOMM: https://arxiv.org/abs/2510.12872
  - Relevance: cross-context KV-cache communication for efficient multi-agent
    prefilling.
  - Boundary: KVCOMM reuses/aligns KV cache state across contexts; the current
    probe sends task-level score evidence and does not expose source KV.

- QJL: https://arxiv.org/abs/2406.03482
  - Relevance: Johnson-Lindenstrauss plus sign-bit quantization motivates a
    future distribution-QJL packet ISA.
  - Boundary: QJL compresses KV cache vectors; our score-distribution packet is
    not a KV-cache quantizer.

- TurboQuant: https://arxiv.org/abs/2504.19874
  - Relevance: systems motivation for online data-oblivious vector
    quantization and unbiased inner-product estimation.
  - Boundary: TurboQuant is a vector/KV quantization method, not a
    source-private model-to-model reasoning packet.

## Novelty Note

The safe novelty claim, if a future score/distribution packet passes, is:

> A fixed-byte source-private packet can communicate candidate-aligned source
> distribution evidence that improves receiver decisions beyond top-label text,
> while using only public candidate side information.

The current CommonsenseQA artifacts do not yet support that claim. They support
the narrower diagnostic claim that top-label copying is a real threat model and
that source/receiver top-2 oracle headroom remains available for a stronger
selector.
