# COLM_v2 Review Packet Baseline Matrix References

Date: 2026-05-04.
Scope: reviewer-facing primary-source boundaries for the LatentWire COLM_v2
baseline matrix. This memo supports the generated review packet in
`results/latentwire_colm_v2_review_packet_20260504/`.

## Dense KV / Cache Communication

- Cache-to-Cache (C2C): `https://openreview.net/forum?id=LeatkxrBCi`
  - Boundary: high-bandwidth source-KV projection/fusion baseline. LatentWire
    should only claim a different low-rate source-private packet point unless
    direct native C2C runs are added.
- Latent Space Communication via K-V Cache Alignment:
  `https://arxiv.org/abs/2601.06123`
  - Boundary: direct dense K/V latent-communication competitor. LatentWire is
    not a broad cache-alignment method.
- DroidSpeak: `https://arxiv.org/abs/2411.02820`
  - Boundary: compatible cache reuse across distributed LLM nodes, not
    source-private packet transfer.
- KVCOMM: `https://arxiv.org/abs/2510.12872`
  - Boundary: cache communication/reuse for multi-agent inference; keep
    separate from byte-scale source-private packets.
- RelayCaching: `https://arxiv.org/abs/2603.13289`
  - Boundary: collaborative decoding cache reuse, not opaque task-level packet
    transfer.
- CacheGen: `https://arxiv.org/abs/2310.07240`
  - Boundary: cache streaming/reuse system; native serving comparisons remain
    future work.

## Activation / Latent Communication

- Communicating Activations Between Language Model Agents:
  `https://arxiv.org/abs/2501.14082`
  - Boundary: direct activation exchange pressures generic latent-communication
    novelty. LatentWire should claim source-private byte-accounted packets and
    destructive controls instead.
- CIPHER / Let Models Speak Ciphers:
  `https://arxiv.org/abs/2310.06272`
  - Boundary: embedding-level inter-model messages for debate. LatentWire is
    narrower and must not claim embedding communication is new.
- Direct Semantic Communication via Vector Translation:
  `https://arxiv.org/abs/2511.03945`
  - Boundary: translated dense semantic vectors, not strict low-byte
    source-private packet controls.
- InterLat: `https://arxiv.org/abs/2511.09149`
  - Boundary: intermediate latent transfer; compare as a dense/latent transfer
    competitor if making broad ICLR claims.

## Prompt / Query Bottlenecks

- Prefix-Tuning: `https://aclanthology.org/2021.acl-long.353/`
  - Boundary: target adaptation by continuous prefixes, not row-specific
    source-to-target communication.
- Gist Tokens:
  `https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html`
  - Boundary: context compression within a model; same-byte visible text is the
    local LatentWire control.
- LLMLingua: `https://arxiv.org/abs/2310.05736`
- LongLLMLingua: `https://arxiv.org/abs/2310.06839`
  - Boundary: visible prompt compression baselines, not source-private latent
    packets.
- BLIP-2 / Q-Former: `https://arxiv.org/abs/2301.12597`
- Flamingo: `https://arxiv.org/abs/2204.14198`
- Perceiver IO: `https://arxiv.org/abs/2107.14795`
  - Boundary: learned query/latent bottlenecks are architectural prior art.

## Sparse / Common-Basis Methods

- Sparse Crosscoders: `https://transformer-circuits.pub/2024/crosscoders/`
  - Boundary: shared/private feature discovery across models. LatentWire must
    show packetized atoms cause downstream utility under destructive controls.
- Universal Sparse Autoencoders: `https://arxiv.org/abs/2502.03714`
  - Boundary: common sparse coordinates alone are not a communication result.
- Transcoders: `https://arxiv.org/abs/2406.11944`
  - Boundary: behavior-feature decomposition is a plausible future packet
    representation, not current COLM_v2 evidence.
- SAEBench: `https://proceedings.mlr.press/v267/karvonen25a.html`
  - Boundary: SAE evaluation quality check if Sparse Resonance Packets move
    from PCA/SVD to learned feature bases.

## Quantization, KV Compression, and Serving Systems

- TurboQuant: `https://arxiv.org/abs/2504.19874`
- KVQuant: `https://arxiv.org/abs/2401.18079`
- KIVI: `https://arxiv.org/abs/2402.02750`
  - Boundary: dense same-model KV/vector compression. These are byte-floor and
    native-systems comparators, not source-private packet methods.
- H2O: `https://arxiv.org/abs/2306.14048`
- SnapKV: `https://arxiv.org/abs/2404.14469`
- Quest: `https://arxiv.org/abs/2406.10774`
- KVzip: `https://arxiv.org/abs/2505.23416`
  - Boundary: long-context KV eviction/compression methods; include for
    systems reviewers but do not treat as direct packet-transfer baselines.
- vLLM / PagedAttention: `https://dl.acm.org/doi/10.1145/3600006.3613165`
- SGLang / RadixAttention:
  `https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html`
  - Boundary: required native serving substrates for any future TTFT, TPOT,
    goodput, HBM, latency, or throughput claim.

## Local Destructive Controls

- Source-index/source-label, source-rank/source-score, same-byte visible text,
  wrong-row, source-row shuffle, same-source-choice wrong-row, target-derived,
  zero-source, candidate roll/derangement, atom shuffle, and coefficient
  corruption are internal LatentWire controls.
- Reviewer-risk memo: `paper/source_private_strict_benchmark_reviewer_risk_design_20260504.md`
- MCQ/order-sensitivity references are tracked in
  `references/723_strict_source_private_benchmark_reviewer_risk_refs_20260504.md`.
