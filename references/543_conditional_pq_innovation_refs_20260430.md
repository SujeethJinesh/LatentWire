# Conditional PQ Innovation References

Date: `2026-04-30`

Purpose: primary-source novelty and comparison memo for the source-private
conditional PQ innovation gate.

## Core Frame

- Wyner and Ziv, "The rate-distortion function for source coding with side
  information at the decoder," IEEE Transactions on Information Theory, 1976.
  URL: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder

  Relevance: the target has public side information, so the source should send a
  conditional innovation rather than a full reconstruction of source state.

- Slepian and Wolf, "Noiseless coding of correlated information sources," IEEE
  Transactions on Information Theory, 1973.
  URL: https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources

  Relevance: motivates communication under correlated source/decoder state, but
  LatentWire is lossy/task-directed rather than pure lossless compression.

- Tishby, Pereira, and Bialek, "The Information Bottleneck Method," 1999.
  URL: https://arxiv.org/abs/physics/0004057

  Relevance: supports rate-limited task information as the right objective, not
  source-state reconstruction.

## Product Quantization And Rotated Residual Coding

- Jégou, Douze, and Schmid, "Product Quantization for Nearest Neighbor Search,"
  IEEE TPAMI, 2011.
  URL: https://ieeexplore.ieee.org/document/5432202

  Relevance: product-codebook packets and public-distance decoding are standard
  compressed nearest-neighbor machinery; LatentWire's novelty must be the
  source-private conditional packet protocol and controls.

- Ge et al., "Optimized Product Quantization," CVPR 2013.
  URL: https://openaccess.thecvf.com/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf

  Relevance: supports the basis/rotation concern that plain PQ can fail when
  the coordinate partition is not aligned with task geometry.

- Qian et al., "QJL: 1-Bit Quantized Johnson-Lindenstrauss Transform for KV
  Cache Quantization with Zero Overhead," 2024.
  URL: https://arxiv.org/abs/2406.03482

  Relevance: strong quantized-sketch/KV-cache baseline; do not claim our packet
  is a better general KV compressor.

- "TurboQuant: Efficient and Accurate LLM Weight Quantization with
  Turbo-Transformed Low-Rank Compensation," 2025.
  URL: https://arxiv.org/abs/2504.19874

  Relevance: recent rotation plus residual quantization comparator; useful for
  systems/quantization framing, not a direct source-private communication
  method.

## Shared Bases And Connectors

- Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023.
  URL: https://openreview.net/forum?id=SrC-nwieGJ

  Relevance: anchor-relative coordinates are the nearest prior for
  basis-invariant communication. Our gate differs by sending a byte-capped
  conditional source-private packet and requiring destructive controls.

- Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms," NAACL 2022.
  URL: https://aclanthology.org/2022.naacl-main.319/

  Relevance: useful analogy for common bases; not a latent communication method.

- Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs
  & Outputs," 2021.
  URL: https://arxiv.org/abs/2107.14795

  Relevance: learned-query connector baseline class for future receiver work.

- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models," 2023.
  URL: https://arxiv.org/abs/2301.12597

  Relevance: Q-Former is a proven connector pattern, but a raw Q-Former-style
  bridge is not unique to LatentWire.

- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning,"
  2022.
  URL: https://arxiv.org/abs/2204.14198

  Relevance: Perceiver-resampler style connectors are competitor/inspiration,
  not novelty by themselves.

## Closest Latent/KV Communication Threats

- Fu et al., "Cache-to-Cache: Fast Latent Communication for Model Collaboration,"
  2025.
  URL: https://arxiv.org/abs/2510.03215

  Relevance: closest broad cross-model cache communication competitor. LatentWire
  should not claim first model-to-model latent communication; our claim is
  source-private, far-left byte-scale packets with explicit side information.

- KVComm, OpenReview.
  URL: https://openreview.net/forum?id=F7rUng23nw

  Relevance: KV-cache communication/compression competitor. Our packet interface
  is not a KV-cache sharing method.

- KVCOMM, "Communication-Efficient Collaborative Inference via KV Cache Reuse,"
  2025.
  URL: https://arxiv.org/abs/2510.12872

  Relevance: direct systems comparator for collaborative inference; LatentWire
  must compare on access model, bytes, private-state exposure, and task scope.

## Novelty Boundary

Promote:

- Conditional source-private packet: `source(matched) - source(answer-masked)`.
- Target/public side-information decoding: candidate innovation relative to
  target prior.
- Disjoint-ID success with deranged public-basis and opaque-slot collapse.
- 2-byte n500 rows with reduced payload uniqueness and reused-payload success.

Do not promote:

- Product quantization itself.
- Rotation/Hadamard transforms themselves.
- Q-Former/Perceiver connector novelty.
- Broad cross-family latent transfer.
- Production GPU/vLLM throughput.

## Comparison Sentence For Paper

"Unlike C2C/KV cache sharing methods, LatentWire transmits a source-private,
rate-capped conditional innovation packet and decodes it against public
target-side candidate side information; unlike generic PQ/QJL/TurboQuant, the
objective is task-directed source-private communication, not reconstruction or
cache compression."
