# Residual/Syndrome Packet Reference Refresh

Date: 2026-05-04

This memo records the targeted literature and competitor scan for the
Residual/Syndrome Sparse Resonance Packet gate. It is not a full related-work
section; it is a decision memo for what the current ARC gate can and cannot
claim.

## Distributed Source Coding And Syndrome Framing

- Slepian and Wolf (1973), "Noiseless coding of correlated information
  sources." DOI: `10.1109/TIT.1973.1055037`.
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
- Wyner and Ziv (1976), "The rate-distortion function for source coding with
  side information at the decoder." DOI: `10.1109/TIT.1976.1055508`.
  https://cir.nii.ac.jp/crid/1360564063947537280
- Pradhan and Ramchandran (2003), "Distributed source coding using syndromes
  (DISCUS): design and construction." DOI: `10.1109/TIT.2002.808103`.
  https://www.kiphub.com/paper/61e505d01d7bbffcc2f622d0
- Liveris, Xiong, and Georghiades, "The equivalence between Slepian-Wolf coding
  and channel coding under density evolution." IBM research page:
  https://research.ibm.com/publications/the-equivalence-between-slepian-wolf-coding-and-channel-coding-under-density-evolution
- Whang et al. (2021), "Neural Distributed Source Coding."
  https://arxiv.org/abs/2106.02797
- Ozyilkan et al. (2021), "Neural Distributed Image Compression using Common
  Information." https://arxiv.org/abs/2106.11723
- Mital et al. (2023), "Neural Distributed Compressor Discovers Binning."
  https://arxiv.org/abs/2310.16961

Use in paper: these sources motivate decoder-side information and syndrome
packets. They do not make the SRP idea novel by themselves. The novel claim
must be the cross-model communication contract, source-private exposure
accounting, and strict destructive-control evidence.

## Error-Correcting Candidate Codes

- Dietterich and Bakiri (1995), "Solving Multiclass Learning Problems via
  Error-Correcting Output Codes." https://arxiv.org/abs/cs/9501101
- Guruswami and Sudan (1999), "Improved Decoding of Reed-Solomon and
  Algebraic-Geometry Codes." IEEE ITS page:
  https://www.itsoc.org/publications/papers/improved-decoding-of-reed-solomon-and-algebraic-geometry-codes
- Bennatan et al. (2018), "Deep Learning for Decoding of Linear Codes - A
  Syndrome-Based Approach." https://arxiv.org/abs/1802.04741

Use in paper: ECOC/list/syndrome decoding are prior-art tools. For LatentWire,
they are packet mechanics, not the headline contribution.

## C2C/KV Communication And Systems Baselines

- C2C / Cache-to-Cache, arXiv `2510.03215` and OpenReview:
  https://arxiv.org/abs/2510.03215
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm / selective KV communication, OpenReview:
  https://openreview.net/forum?id=F7rUng23nw
- KVCOMM / cross-context KV reuse, arXiv `2510.12872`:
  https://arxiv.org/abs/2510.12872
- Q-KVComm / compressed KV transmission, arXiv `2512.17914`:
  https://arxiv.org/abs/2512.17914
- CacheGen, "Fast Context Loading for Language Model Applications," arXiv:
  https://arxiv.org/abs/2310.07240
- vLLM/PagedAttention, arXiv:
  https://arxiv.org/abs/2309.06180
- DistServe, arXiv:
  https://arxiv.org/abs/2401.09670

Use in paper: C2C/KV systems are not defeated unless run natively. SRP can
compare as a different access-model and byte-rate boundary: source-private
fixed packet versus source-KV-visible cache transfer.

## KV Quantization Byte Floors

- KIVI, 2-bit asymmetric KV quantization:
  https://arxiv.org/abs/2402.02750
- KVQuant, sub-4-bit KV cache quantization:
  https://arxiv.org/abs/2401.18079
- PolarQuant, polar-transform KV quantization:
  https://arxiv.org/abs/2502.02617
- TurboQuant, online vector/KV quantization with QJL residual:
  https://arxiv.org/abs/2504.19874
  https://openreview.net/forum?id=tO3ASKZlok
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

Use in paper: these define aggressive KV byte floors and hardware-friendly
quantization baselines. SRP should report payload bytes, framed bytes, cache
line/DMA accounting, and utility per byte, while avoiding unmeasured GPU
throughput claims.

## SAE / Crosscoder / Common-Basis Boundary

- Bricken et al. / ICLR 2024 sparse autoencoder line:
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/1fa1ab11f4bd5f94b2ec20e794dbfa3b-Abstract-Conference.html
- Anthropic Crosscoders:
  https://www.anthropic.com/research/crosscoder-model-diffing
- Crosscoder sparsity artifact warning, arXiv:
  https://arxiv.org/abs/2504.02922
- Transcoders for sparse circuit replacement:
  https://arxiv.org/abs/2406.11944
- Relative/common-coordinate communication prior, ICLR 2023 OpenReview:
  https://openreview.net/forum?id=SrC-nwieGJ

Use in paper: do not claim sparse atoms, cross-model dictionary discovery, or a
canonical feature alphabet as novel. SRP may use these tools later, but the
novelty must stay on fixed-rate source-private communication with causal
packet controls.

## Benchmark And Reviewer Controls

- MCQ option-order sensitivity: Zheng et al. (ICLR 2024/arXiv).
  https://arxiv.org/abs/2309.03882
- MCQ option permutation sensitivity: Pezeshkpour and Hruschka (NAACL Findings
  2024/arXiv). https://arxiv.org/abs/2308.11483
- Benchmark leakage cards: Xu et al.
  https://arxiv.org/abs/2404.18824
- Data/order contamination tests: Oren et al.
  https://arxiv.org/abs/2310.17623
- Paired significance for NLP: Dror et al. (ACL 2018).
  https://aclanthology.org/P18-1128/
- Bootstrap significance: Koehn (EMNLP 2004).
  https://aclanthology.org/W04-3250/

Use in paper: any positive SRP row needs paired deltas, help/harm counts,
label/order controls, wrong-row controls, target-derived controls, and
same-byte/source-score baselines before it can be widened.

## Claim Boundary

Reviewer-safe wording:

> Unlike C2C/KV communication, SRP does not expose source KV caches or raw
> hidden states. Unlike SAE/crosscoder/transcoder work, SRP does not claim a
> universal sparse feature alphabet. It asks whether a fixed-byte residual or
> syndrome packet, decoded with target-side information, carries source-needed
> evidence beyond source-choice codes, source-score compression, same-byte
> visible text, target-cache effects, and destructive packet controls.

Current ARC residual/syndrome result: negative. Pairwise score-syndrome packets
should not be the headline method unless a later receiver/message shape clears
the strict gate.
