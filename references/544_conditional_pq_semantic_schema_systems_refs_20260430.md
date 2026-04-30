# Conditional PQ Semantic/Schema/Systems References

Date: `2026-04-30`

Purpose: primary-source memo for the conditional PQ semantic/no-diagnostic
stress, cross-family basis/schema grid, and packet-ISA waterfall update.

## Side-Information And Task-Oriented Coding

- Wyner and Ziv, "The rate-distortion function for source coding with side
  information at the decoder," IEEE Transactions on Information Theory, 1976.
  URL: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder

  Relevance: the conditional innovation packet uses target-side public
  candidate state as decoder side information.

- Slepian and Wolf, "Noiseless coding of correlated information sources," IEEE
  Transactions on Information Theory, 1973.
  URL: https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources

  Relevance: motivates communicating only what is not already predictable from
  correlated decoder state, although this work is task-directed and lossy.

- Enttsel and Corlay, "Model-Aware Rate-Distortion Limits for Task-Oriented
  Source Coding," 2026.
  URL: https://arxiv.org/abs/2602.12866

  Relevance: supports evaluating rate, downstream task accuracy, and deployed
  receiver behavior together rather than optimizing source reconstruction.

- Zeng et al., "V2X-DSC: Multi-Agent Collaborative Perception with Distributed
  Source Coding Guided Communication," 2026.
  URL: https://arxiv.org/abs/2602.00687

  Relevance: independently motivates conditional coding in multi-agent
  perception: receivers need complementary innovation beyond local context, not
  redundant dense feature transmission.

## Quantization And Residual Codebooks

- Huijben et al., "Residual Quantization with Implicit Neural Codebooks,"
  ICML 2024.
  URL: https://arxiv.org/abs/2401.14732

  Relevance: QINCo suggests a next branch where residual codebooks are
  conditioned on previous reconstruction or target-public state. It does not
  subsume the current result, which is source-private conditional communication
  with destructive controls rather than vector-search compression.

- Jégou, Douze, and Schmid, "Product Quantization for Nearest Neighbor Search,"
  IEEE TPAMI, 2011.
  URL: https://ieeexplore.ieee.org/document/5432202

  Relevance: product-codebook packetization is inherited machinery; novelty is
  the conditional source-private packet protocol and controls.

- Ge et al., "Optimized Product Quantization," CVPR 2013.
  URL: https://openaccess.thecvf.com/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf

  Relevance: supports the basis/rotation concern behind the protected-Hadamard
  and anchor-relative packet variants.

- Qian et al., "QJL: 1-Bit Quantized Johnson-Lindenstrauss Transform for KV
  Cache Quantization with Zero Overhead," 2024.
  URL: https://arxiv.org/abs/2406.03482

  Relevance: strong KV quantization comparator and byte-floor row for the
  systems table; not a source-private task packet method.

- "TurboQuant: Efficient and Accurate LLM Weight Quantization with
  Turbo-Transformed Low-Rank Compensation," 2025.
  URL: https://arxiv.org/abs/2504.19874

  Relevance: rotation plus residual quantization comparator for the systems and
  compression discussion.

## Common Bases, Connectors, And Competitors

- Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023.
  URL: https://openreview.net/forum?id=SrC-nwieGJ

  Relevance: nearest prior for common-basis/anchor-relative communication; the
  new grid shows anchor-relative and other static bases do not rescue
  bidirectional held-out-family transfer in our current task.

- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models," 2023.
  URL: https://arxiv.org/abs/2301.12597

  Relevance: Q-Former-style receiver-conditioned slots are a plausible next
  connector branch, but not unique by themselves.

- Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs
  & Outputs," 2021.
  URL: https://arxiv.org/abs/2107.14795

  Relevance: supports learned-query bottleneck connectors as future work.

- Fu et al., "Cache-to-Cache: Fast Latent Communication for Model
  Collaboration," 2025.
  URL: https://arxiv.org/abs/2510.03215

  Relevance: closest broad latent/KV communication competitor. LatentWire must
  frame its novelty as source-private, byte-scale conditional packets rather
  than generic latent collaboration.

- KVCOMM, "Communication-Efficient Collaborative Inference via KV Cache Reuse,"
  2025.
  URL: https://arxiv.org/abs/2510.12872

  Relevance: systems comparator for KV reuse; unlike the conditional packet
  ISA, it exposes or reuses cache state.

## Claim Boundary

Promote:

- Semantic/no-diagnostic same-family n500 success: the packet does not require
  direct public diagnostic handles.
- Cross-family basis/schema grid failure: existing static bases do not solve
  unseen-family communication.
- Packet ISA waterfall: 2-byte and 4-byte conditional packets can be stated as
  5-byte and 7-byte boundary records with Mac-local cache/DMA byte accounting.

Do not promote:

- Broad unseen-family latent transfer.
- Product quantization, QINCo, or Hadamard transforms as standalone novelty.
- Measured GPU serving, HBM, PCIe/NVLink, or energy results before NVIDIA runs.

Next method implication: pursue public-conditioned residual codebooks or
ontology-calibrated receiver-conditioned slots, because the static-basis grid
does not rescue cross-family transfer despite high target-innovation oracle
headroom.
