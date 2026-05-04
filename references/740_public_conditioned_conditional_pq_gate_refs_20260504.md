# Public-Conditioned Conditional PQ Gate References

Date: `2026-05-04`

Purpose: reference and novelty boundary memo for the public-conditioned
conditional-PQ held-out-family gate. The gate keeps the source-private
conditional packet interface fixed but normalizes packet and candidate
innovations by each row's public candidate geometry before PQ decoding.

## Conditional Codebook Prior Art

- Huijben et al., "Residual Quantization with Implicit Neural Codebooks,"
  ICML 2024.
  URL: https://openreview.net/forum?id=NBAc36V00H

  Use: QINCo is prior art for conditioning residual codebooks on previous
  reconstruction. Boundary: LatentWire cannot claim conditional residual
  codebooks as new; the possible contribution is source-private packet
  communication with target-public side information and destructive controls.

- Huijben et al., "QINCo2: Improved Implicit Neural Codebooks for Vector
  Compression," ICLR 2025.
  URL: https://openreview.net/forum?id=2zMHHZ569S

  Use: recent evidence that implicit/conditional codebooks can improve vector
  compression. Boundary: this is vector compression, not LLM-to-LLM
  source-causal evidence transfer.

- Jégou, Douze, and Schmid, "Product Quantization for Nearest Neighbor Search,"
  IEEE TPAMI, 2011.
  URL: https://inria.hal.science/inria-00514462v2

  Use: product-codebook machinery. Boundary: PQ itself is not novel.

- Ge et al., "Optimized Product Quantization," CVPR 2013.
  URL: https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html

  Use: coordinate conditioning/rotation is a known way to improve PQ geometry.
  Boundary: LatentWire needs communication evidence, not only better vector
  compression.

## Side-Information Coding

- Wyner and Ziv, "The rate-distortion function for source coding with side
  information at the decoder," IEEE Transactions on Information Theory, 1976.
  URL: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder

  Use: decoder side information is the right theoretical frame for using target
  public candidate geometry.

- Slepian and Wolf, "Noiseless coding of correlated information sources," IEEE
  Transactions on Information Theory, 1973.
  URL: https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources

  Use: correlated source/target evidence motivates sending only residual
  information. Boundary: the LatentWire result is lossy and task-directed.

## Dense / KV Competitor Boundary

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," 2025/2026.
  URL: https://arxiv.org/abs/2510.03215

  Use: strongest dense cross-model cache fusion baseline. LatentWire should
  compare byte movement and private-state exposure, not claim first latent
  model communication.

- "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing,"
  2025/2026.
  URL: https://arxiv.org/abs/2510.03346

  Use: selective KV sharing competitor. Boundary: sends selected KV state, not
  tiny auditable packets.

- "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
  Multi-agent Systems," 2025.
  URL: https://arxiv.org/abs/2510.12872

  Use: training-free KV reuse competitor. Boundary: optimizes cache reuse for
  shared context, not source-private conditional evidence transfer.

- KIVI, "A Tuning-Free Asymmetric 2bit Quantization for KV Cache," 2024.
  URL: https://arxiv.org/abs/2402.02750

  Use: low-bit KV byte floor for systems comparison.

## Gate Outcome And Novelty Boundary

The public-zscore gate is a useful negative:

- public-only and answer-masked controls stayed at target-only, so the public
  conditioner alone did not become the answer path;
- label-shuffled, permuted-code, and random same-byte controls remained too
  strong, so row-public normalization did not produce clean source-causal
  cross-family transfer;
- target innovation oracle stayed high, so the blocker is learnable packet
  causality, not absence of target-side headroom.

Next implication: simple row-public normalization is not enough. The next
conditional-PQ branch needs a stronger public-conditioned residual codebook or
a learned receiver with corruption-to-no-op training, and it must preserve the
same destructive controls before any ICLR claim.
