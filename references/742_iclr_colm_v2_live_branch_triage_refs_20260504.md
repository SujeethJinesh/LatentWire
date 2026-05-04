# ICLR / COLM_v2 Live Branch Triage References

Date: `2026-05-04`

Purpose: reference and novelty-boundary memo for the live branch triage after
the conditional-PQ and HellaSwag receiver failures.

## Dense Communication Competitors

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," ICLR 2026.
  URL: https://arxiv.org/abs/2510.03215
  Code: https://github.com/thu-nics/C2C

  Use: primary dense KV-cache fusion baseline. C2C projects and fuses source
  model KV cache into the target cache. LatentWire should not claim to beat C2C
  on raw accuracy without a direct run; the meaningful boundary is lower-rate,
  source-private, auditable packet transfer with utility-per-byte reporting.

- Shi et al., "KVComm: Enabling Efficient LLM Communication Through Selective
  KV Sharing," ICLR 2026.
  URL: https://openreview.net/pdf/fb0426d7900d0d9d2c2252532541eb292b23765a.pdf

  Use: selective KV-sharing competitor. Boundary: KVComm reduces dense cache
  transfer by selecting KV pairs/layers, but it still sends KV state rather
  than a byte-scale source-private task packet.

## Quantization And Systems Byte Floors

- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," arXiv 2025.
  URL: https://arxiv.org/abs/2504.19874

  Use: low-bit vector/KV quantization comparator and motivation for
  rate-distortion accounting. Boundary: TurboQuant compresses vectors/KV
  caches; it is not a source-private model-to-model task protocol.

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,"
  ICML 2024.
  URL: https://arxiv.org/abs/2402.02750

  Use: low-bit KV-cache compression byte floor and hardware-friendly systems
  comparator. Boundary: KIVI reduces KV memory movement; LatentWire's current
  byte accounting is packet-size-only unless native serving measurements are
  run.

## Conditional Codebook Direction

- Jégou, Douze, and Schmid, "Product Quantization for Nearest Neighbor Search,"
  IEEE TPAMI 2011.
  URL: https://inria.hal.science/inria-00514462v2

  Use: product-codebook machinery. Boundary: PQ itself is prior work.

- Ge et al., "Optimized Product Quantization," CVPR 2013.
  URL: https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html

  Use: rotation/conditioning precedent for product quantization geometry.

- Huijben et al., "QINCo: Residual Quantization with Implicit Neural
  Codebooks," ICML 2024.
  URL: https://openreview.net/forum?id=NBAc36V00H

  Use: stronger public-conditioned residual codebook inspiration for the next
  conditional-PQ branch.

- Huijben et al., "QINCo2: Improved Implicit Neural Codebooks for Vector
  Compression," ICLR 2025.
  URL: https://openreview.net/forum?id=2zMHHZ569S

  Use: follow-up implicit codebook baseline to consider if the next branch
  becomes a learned public-conditioned codebook.

## Triage Consequence

The current evidence supports a narrow COLM_v2 claim but not an ICLR full-paper
claim. The stale recommendation to continue HellaSwag protected top-2/rival
switchers is superseded: those runs already fail or no-op under receiver
calibration, harm control, denoising syndrome decoding, and sparse/common-basis
top-2 atom controls. The next ICLR gate should either implement a richer
public-conditioned residual/codebook receiver for conditional PQ or shift to
target-self-resonance with a held-out encoder preflight.
