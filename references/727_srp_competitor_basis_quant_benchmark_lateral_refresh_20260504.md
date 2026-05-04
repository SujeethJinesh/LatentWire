# Reference Memo 727: SRP Competitor, Basis, Quantization, Benchmark, And Lateral Refresh

Date: 2026-05-04

## Current Paper Status

Paper readiness: not ICLR-ready. COLM_v1 is frozen and COLM_v2/ICLR are now
centered on Sparse Resonance Packets, but no strict positive packet gate has
survived controls.

Current story: SRP should be framed as a low-rate, source-private,
interpretable communication protocol, not as generic latent communication,
SAE discovery, or KV-cache compression.

Exact blocker: a strict packet must beat target-only, source-index/rank/score,
same-byte text, target-derived packets, wrong-row/candidate-roll/bit-header
destructive controls, and source-family substitution with paired positive
uncertainty.

## Competitor And Systems Baselines

- C2C / Cache-to-Cache, OpenReview:
  https://openreview.net/pdf?id=LeatkxrBCi
  Use: closest dense KV/cache fusion baseline.

- C2C implementation:
  https://github.com/TencentARC/Cache-to-Cache
  Use: runnable reference for future dense-baseline comparison.

- CacheGen:
  https://arxiv.org/abs/2310.07240
  Use: cache compression/streaming precedent for network and serving bytes.

- PagedAttention / vLLM:
  https://arxiv.org/abs/2309.06180
  Use: serving-memory baseline; do not claim native serving wins without this
  style of comparison.

- FlexGen:
  https://arxiv.org/abs/2303.06865
  Use: offload/memory movement baseline for distinguishing interface bytes from
  end-to-end system traffic.

- LatentMAS:
  https://arxiv.org/abs/2507.11273
  Use: adjacent latent communication threat; SRP must emphasize source-private
  sparse packets and destructive controls.

- Activation steering:
  https://arxiv.org/abs/2308.10248
  Use: boundary against viewing SRP as ordinary activation intervention.

## Sparse Bases And Feature Alignment

- Toy Models of Superposition:
  https://www.transformer-circuits.pub/2022/toy_model/
  Use: sparse feature/superposition intuition.

- Towards Monosemanticity:
  https://transformer-circuits.pub/2023/monosemantic-features/
  Use: SAE feature dictionaries for language models.

- Sparse Autoencoders Find Highly Interpretable Features:
  https://openreview.net/forum?id=F76bwRSLeK
  Use: ICLR SAE precedent.

- Gemma Scope:
  https://arxiv.org/abs/2408.05147
  Use: open SAE infrastructure precedent.

- Transcoders Find Interpretable LLM Feature Circuits:
  https://arxiv.org/abs/2406.11944
  Use: behavior/circuit-oriented sparse atoms, stronger than reconstruction
  atoms for SRP.

- Sparse Crosscoders:
  https://transformer-circuits.pub/2024/crosscoders/
  Use: shared/private feature dictionaries across models/layers.

- Crosscoder sparsity artifact warning:
  https://arxiv.org/abs/2504.02922
  Use: reviewer caveat; do not claim atom canonicity without causal controls.

- Sparse Autoencoders Do Not Find Canonical Units of Analysis:
  https://openreview.net/forum?id=9ca9eHNrdH
  Use: direct warning against a universal feature alphabet claim.

- Relative Representations Enable Zero-shot Latent Space Communication:
  https://openreview.net/forum?id=SrC-nwieGJ
  Use: common-coordinate/relative latent communication precedent.

- CKA representation similarity:
  https://arxiv.org/abs/1905.00414
  Use: representation-alignment baseline and analysis tool.

## Quantization And Hardware Byte Floors

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  Use: low-bit vector/KV-style quantization threat to SRP byte claims.

- Google Research TurboQuant blog:
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  Use: official high-level TurboQuant systems framing.

- KIVI:
  https://arxiv.org/abs/2402.02750
  Use: 2-bit KV-cache quantization floor.

- KVQuant:
  https://arxiv.org/abs/2401.18079
  Use: long-context KV quantization baseline.

- No Token Left Behind:
  https://arxiv.org/abs/2402.18096
  Use: mixed-precision KV quantization baseline.

- LeanKV:
  https://arxiv.org/abs/2412.03131
  Use: unified KV quantization and sparsity.

- Scissorhands:
  https://arxiv.org/abs/2305.17118
  Use: KV eviction/compression baseline.

- StreamingLLM:
  https://arxiv.org/abs/2309.17453
  Use: attention sink / streaming cache behavior.

- SnapKV:
  https://arxiv.org/abs/2404.14469
  Use: prompt-compression KV baseline.

- H2O:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf
  Use: heavy-hitter KV eviction baseline.

- QJL:
  https://arxiv.org/abs/2406.03482
  Use: sketch/sign-transform threat to utility-per-byte framing.

Reviewer-safe formula:

```text
KV_fp16_bytes = B * T * L * 2 * H_kv * d_head * 2
KV_q_bytes = B * T * L * H_kv * d_head * (b_K + b_V) / 8 + metadata
SRP_packet_bytes = payload_bits / 8 + framing + cacheline/DMA accounting
```

Claims requiring measurement: native speedup, HBM savings, kernel efficiency,
and sparse-packet throughput.

## Benchmark And Reviewer-Risk Sources

- ARC:
  https://arxiv.org/abs/1803.05457

- HellaSwag:
  https://aclanthology.org/P19-1472/

- LLMs are not robust MCQ selectors:
  https://openreview.net/pdf?id=shr9PXz7T0

- Option-order sensitivity:
  https://arxiv.org/abs/2308.11483

- MCQ limitations:
  https://arxiv.org/abs/2401.07955

- Paired statistical tests:
  https://doi.org/10.1162/089976698300017197

- Bootstrap significance in NLP:
  https://people.csail.mit.edu/people/koehn/publications/bootstrap2004.pdf

- Calibration:
  https://proceedings.mlr.press/v70/guo17a.html

- Selective classification:
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

- Contamination detection:
  https://arxiv.org/abs/2310.16789

- LiveBench:
  https://arxiv.org/abs/2406.19314

Controls to preserve: option-text permutation, answer-position balance,
source-index/rank/score, source-score quantization, target-derived packet,
wrong-row packet, candidate-roll/derangement, bit/header shuffles, parity
flip, same-byte text, Qwen/source-family substitution, paired bootstrap,
helps/harms, and risk-coverage curves for fired rows.

## Lateral Methods

- Diffusion Transformers:
  https://arxiv.org/abs/2212.09748

- Consistency Models:
  https://arxiv.org/abs/2303.01469

- Mixture-of-Depths:
  https://arxiv.org/abs/2404.02258

- Flamingo frozen-model connectors:
  https://arxiv.org/abs/2204.14198

- BLIP-2 / Q-Former:
  https://arxiv.org/abs/2301.12597

- Neural entropic Gromov-Wasserstein alignment:
  https://arxiv.org/abs/2312.07397

- Information bottleneck:
  https://www.princeton.edu/~wbialek/our_papers/tishby%2Bal_99.pdf

- Deep VIB:
  https://arxiv.org/abs/1612.00410

Highest expected-value method idea: an event-triggered innovation/defer gate
over target uncertainty, source reliability, source-target disagreement, and
score-shape features. The confidence/ECOC packet can remain the 2-4 byte
transport, but the receiver must learn helpability instead of relying on
hand-binned source confidence.

## Novelty Boundary

Not novel: SAEs, crosscoders, transcoders, activation steering, C2C dense KV
sharing, KV compression, or MCQ source-score relays.

Potentially novel: a fixed-byte, source-private sparse packet interface whose
receiver passes destructive controls, reports utility per byte, exposes no
source text/KV, and beats source-choice and target-cache baselines with paired
positive uncertainty.
