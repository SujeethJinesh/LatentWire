# Conditional PQ Corruption-Noop Receiver References

Date: `2026-05-04`

Purpose: reference and novelty-boundary memo for the conditional-PQ
corruption-to-noop receiver gate.

## Receiver Objective

- Vincent et al., "Stacked Denoising Autoencoders," JMLR 2010.
  URL: https://jmlr.csail.mit.edu/papers/v11/vincent10a.html

  Use: corruption training precedent. Boundary: LatentWire is not reconstructing
  corrupted inputs; it trains destructive packet families to decode to no-op.

- Hendrycks et al., "Deep Anomaly Detection with Outlier Exposure," ICLR 2019.
  URL: https://arxiv.org/abs/1812.04606

  Use: explicit bad-packet negatives are analogous to outlier exposure.
  Boundary: the receiver output is a task decision, not an OOD score.

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks,"
  NeurIPS 2017.
  URL: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

  Use: risk/coverage framing for "apply packet only when useful." Boundary:
  this gate uses no-op training, not a final calibrated selective gate.

- Mozannar and Sontag, "Consistent Estimators for Learning to Defer to an
  Expert," ICML 2020.
  URL: https://proceedings.mlr.press/v119/mozannar20b.html

  Use: defer/no-op receiver motivation. Boundary: LatentWire's source packet is
  not an external expert prediction exposed as text or full logits.

## Quantized Packet And Side-Information Boundary

- Jégou, Douze, and Schmid, "Product Quantization for Nearest Neighbor Search,"
  IEEE TPAMI 2011.
  URL: https://inria.hal.science/inria-00514462v2

  Use: product-codebook machinery. Boundary: PQ itself is not novel.

- Ge et al., "Optimized Product Quantization," CVPR 2013.
  URL: https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html

  Use: rotation/conditioning precedent for product quantization geometry.

- Wyner and Ziv, "The rate-distortion function for source coding with side
  information at the decoder," IEEE Transactions on Information Theory, 1976.
  URL: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder

  Use: decoder-side public candidate geometry is a side-information channel.

- Huijben et al., "QINCo: Residual Quantization with Implicit Neural
  Codebooks," ICML 2024.
  URL: https://openreview.net/forum?id=NBAc36V00H

  Use: conditional/implicit codebook prior art. Boundary: LatentWire's novelty
  must be source-private task communication and strict destructive controls, not
  conditional quantization alone.

- Huijben et al., "QINCo2: Improved Implicit Neural Codebooks for Vector
  Compression," ICLR 2025.
  URL: https://openreview.net/forum?id=2zMHHZ569S

  Use: stronger conditional codebook baseline to consider if we revisit this
  branch.

## Dense And KV Competitor Boundary

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," ICLR 2026.
  URL: https://openreview.net/forum?id=LeatkxrBCi

  Use: dense KV-cache fusion baseline. Boundary: C2C projects and fuses source
  KV into target KV; LatentWire should claim a lower-rate, source-private,
  auditable packet interface if evidence supports it, not first latent
  communication.

- Shi et al., "KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing," ICLR 2026.
  URL: https://arxiv.org/abs/2510.03346

  Use: selective KV sharing competitor. Boundary: still communicates selected
  KV pairs rather than byte-scale source-private packets.

- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," arXiv 2025.
  URL: https://arxiv.org/abs/2504.19874

  Use: low-bit vector/KV compression and systems byte-floor comparator.
  Boundary: TurboQuant compresses KV/vector data; it is not a source-private
  task packet protocol.

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,"
  ICML 2024.
  URL: https://arxiv.org/abs/2402.02750

  Use: low-bit KV cache floor for systems comparisons.

## Gate Consequence

The corruption-to-noop receiver does not solve the held-out-family
conditional-PQ blocker. Strong no-op weights collapse all packets to target
prior. Weak weights restore matched source lift but also restore random,
permuted, and label-shuffled packet controls. This weakens simple receiver
corruption training and motivates moving to a richer source-code packet or
implicit public-conditioned codebook rather than widening this exact branch.
