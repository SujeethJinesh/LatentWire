# Conditional PQ ICLR / COLM_v2 Status References

Date: `2026-05-04`

Purpose: paper-facing reference refresh for the current LatentWire_v2 live
branch decision. The conclusion is that conditional PQ innovation is the
strongest scoped positive method, while ICLR remains blocked by cross-family or
broader-benchmark evidence.

## Live Method Boundary

- Wyner and Ziv, "The rate-distortion function for source coding with side
  information at the decoder," IEEE Transactions on Information Theory, 1976.
  URL: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder

  Use: conditional PQ is best framed as task-directed side-information coding:
  the target has public candidate state and the source sends only a compact
  innovation.

- Slepian and Wolf, "Noiseless coding of correlated information sources," IEEE
  Transactions on Information Theory, 1973.
  URL: https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources

  Use: motivates communication when sender and receiver have correlated but not
  identical evidence. Boundary: LatentWire is lossy and task-directed, not
  lossless distributed compression.

- Jégou, Douze, and Schmid, "Product Quantization for Nearest Neighbor Search,"
  IEEE TPAMI, 2011.
  URL: https://ieeexplore.ieee.org/document/5432202

  Use: PQ is prior machinery. Do not claim PQ itself as novelty; claim the
  source-private conditional protocol and destructive evaluation.

- Ge et al., "Optimized Product Quantization," CVPR 2013.
  URL: https://openaccess.thecvf.com/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf

  Use: supports the basis/rotation sensitivity behind protected and
  anchor-relative packet variants.

## Competitor Boundary

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," OpenReview / ICLR 2026.
  URL: https://openreview.net/forum?id=LeatkxrBCi

  Use: closest high-bandwidth latent/KV communication baseline. LatentWire must
  differentiate on source-private byte-scale packets, exposure, auditability,
  and utility per byte, not on being first to use latent model-to-model
  communication.

- "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing."
  URL: https://arxiv.org/abs/2510.03346

  Use: selective KV-sharing competitor. Boundary: it shares selected KV pairs,
  whereas LatentWire sends tiny task packets and does not expose source KV.

- "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
  Multi-agent Systems."
  URL: https://arxiv.org/abs/2510.12872

  Use: training-free cross-context KV reuse competitor. Boundary: it optimizes
  prefill/cache reuse for overlapping contexts, not source-private
  side-information packet causality.

- Qian et al., "QJL: 1-Bit Quantized Johnson-Lindenstrauss Transform for KV
  Cache Quantization with Zero Overhead," 2024.
  URL: https://arxiv.org/abs/2406.03482

  Use: low-bit KV quantization byte floor. Boundary: QJL compresses dense KV
  vectors; it does not provide an auditable source-private task packet.

- "TurboQuant: Efficient and Accurate LLM Weight Quantization with
  Turbo-Transformed Low-Rank Compensation," 2025.
  URL: https://arxiv.org/abs/2504.19874

  Use: strong recent quantization/systems threat. Boundary: discuss as
  compression baseline and do not claim GPU/HBM wins without native measurement.

## Selective / Defer Boundary

- Mozannar and Sontag, "Consistent Estimators for Learning to Defer to an
  Expert," ICML 2020.
  URL: https://arxiv.org/abs/2006.01862

  Use: deferral and expert selection are prior art. If LatentWire adds a gate,
  novelty requires source-private packet causality beyond ordinary deferral.

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks,"
  NeurIPS 2017.
  URL: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

  Use: risk/coverage and reject-option reporting. Boundary: selection alone is
  not model-to-model communication.

- Angelopoulos et al., "Conformal Risk Control," 2022.
  URL: https://arxiv.org/abs/2208.02814

  Use: calibrated packet firing or abstention thresholds. Boundary: conformal
  gating can certify when to act, but cannot prove the packet carries source
  evidence by itself.

- Pradhan and Ramchandran, "Distributed Source Coding Using Syndromes
  (DISCUS): Design and Construction," IEEE Transactions on Information Theory,
  2003.
  URL: https://doi.org/10.1109/TIT.2002.808103

  Use: practical syndrome/coset coding with decoder side information. Boundary:
  LatentWire's remaining novelty is task-directed LLM packet transfer under
  strict destructive controls.

## Current Decision

Promote for COLM_v2:

- conditional source-private innovation packet;
- 2-byte and 4-byte packet rows with Mac-local packet-ISA accounting;
- semantic/no-diagnostic same-family success;
- explicit cross-family failure boundary.

Do not promote for ICLR yet:

- broad unseen-family latent communication;
- sparse atom/resonance transfer, after recent controls saturated it;
- native GPU, HBM, energy, PCIe, or NVLink claims.

Next gate: public-conditioned residual/codebook decoding or a broader
less-synthetic benchmark, with source-index/rank/score, same-byte text,
wrong-row, candidate-roll, code-permutation, random-byte, and target-derived
controls preserved.
