# Reference Memo 719: Sparse Resonance Packet Competitors and Systems Boundary

Date: 2026-05-04

## Local Claim Boundary

Sparse Resonance Packets should be positioned against C2C and KV-cache
communication as a low-rate, source-private, interpretable packet interface.
The current implementation does not yet beat strict controls, so the claim is
a method direction plus a falsification gate, not a positive ICLR result.

## Dense Cache Communication Competitors

- Fu et al., 2026, "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models."
  Source: https://openreview.net/forum?id=LeatkxrBCi
  Boundary: strongest direct competitor. C2C projects and fuses source KV-cache
  state into the receiver with learned layer gating and reports accuracy and
  latency gains. LatentWire must not claim novelty for "latent communication"
  alone.

- C2C official code.
  Source: https://github.com/thu-nics/C2C
  Boundary: documents the Rosetta wrapper, per-layer projector/fuser training,
  and supported source/receiver pairs. It is the right implementation baseline
  for dense KV fusion when NVIDIA hardware is available.

- CacheBlend.
  Source: https://arxiv.org/abs/2405.16444
  Boundary: KV-cache reuse/fusion for RAG-like settings. It is a serving
  competitor, not source-private model-to-model communication.

- vLLM / PagedAttention.
  Source: https://arxiv.org/abs/2309.06180
  Boundary: foundational KV serving substrate. Use it as a native systems
  baseline when measuring real TTFT/TPOT, not as a Mac-local claim.

- SGLang / RadixAttention.
  Source: https://arxiv.org/abs/2312.07104
  Boundary: second native serving substrate for structured generation and KV
  reuse. Include when reporting production-style serving measurements.

## Quantized / Sparse KV Baselines

- TurboQuant, "Online Vector Quantization with Near-optimal Distortion Rate."
  Source: https://arxiv.org/abs/2504.19874
  Boundary: strong low-bit vector/KV comparator. Use as a compression baseline
  and byte-floor competitor; do not claim speedups without reproducing them.

- Google Research TurboQuant blog.
  Source:
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  Boundary: useful official systems framing, but blog performance claims should
  not substitute for local measured throughput.

- KIVI.
  Source: https://arxiv.org/abs/2402.02750
  Boundary: tuning-free asymmetric 2-bit KV cache quantization. This sets an
  aggressive byte floor for dense KV baselines.

- KVQuant.
  Source: https://arxiv.org/abs/2401.18079
  Boundary: sub-4-bit KV cache quantization with outlier handling and
  non-uniform datatypes.

- QJL.
  Source: https://arxiv.org/abs/2406.03482
  Boundary: 1-bit Johnson-Lindenstrauss-style sign sketch for KV quantization.
  Relevant as a low-bit continuous-state comparator.

## Sparse / Common-Basis Representation Sources

- Universal Sparse Autoencoders.
  Source: https://arxiv.org/abs/2502.03714
  Boundary: motivates a shared concept space, but evaluates alignment and
  reconstruction rather than fixed-byte downstream communication.

- Relative Representations.
  Source: https://openreview.net/forum?id=SrC-nwieGJ
  Boundary: strong common-coordinate baseline and novelty risk. Use
  anchor-relative coordinates as an initializer/control.

- SVCCA.
  Source:
  https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability
  Boundary: useful for diagnosing shared subspaces; not a causal packet method.

- Prefix-Tuning.
  Source: https://arxiv.org/abs/2101.00190
  Boundary: target-native soft prefixes are not novel by themselves. LatentWire
  must show source-conditioned, low-rate, destructive-control-surviving packet
  transfer.

## Utility-Per-Byte Reporting

Report paired utility per communicated byte:

```text
UPB = mean_i(correct_method_i - correct_target_i) / framed_packet_bytes
```

Use paired bootstraps over item IDs. Keep separate columns for:

- communicated payload bytes;
- framed packet bytes;
- modeled dense/quantized KV byte floor;
- source text exposed;
- source KV exposed;
- native throughput measured: true/false.

For dense KV communication, the byte floor is:

```text
B_dense_KV = 2 * L * H_kv * T_source * d_head * bytes_per_value
```

For quantized KV:

```text
B_quant_KV = 2 * L * H_kv * T_source * d_head * bits_per_value / 8 + metadata
```

For Sparse Resonance Packets:

```text
B_packet = candidates * top_k * (ceil(log2(atom_count)) + coefficient_bits) / 8
```

The current ARC n8 sparse PCA packet used rank 4, top-2, 3-bit coefficients:
`1.25` bytes/candidate, approximately `5.0` bytes/row at a 4-candidate packet
pool. It failed strict controls, so this is a systems boundary measurement,
not a positive method result.
