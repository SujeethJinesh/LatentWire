# HellaSwag Non-Qwen Receiver-Family Multi-Slice References

Date: 2026-05-03

## Purpose

This memo records the literature and novelty boundary after the second
non-Qwen HellaSwag receiver-family slice. The new evidence strengthens
fixed-byte packet utility across TinyLlama -> Phi-3, but it does not close the
learned receiver-fusion gate because the selected receiver still trails
packet-only.

## Current Evidence Boundary

- Artifact:
  `results/source_private_hellaswag_nonqwen_receiver_family_multislice_summary_20260503_validation1024_2048/`
- Range: HellaSwag validation `1024:2048`.
- Packet contract: `2B` raw / `5B` framed, source-private.
- Weighted Phi-3 target-only accuracy: `0.263021`.
- Weighted TinyLlama packet-only accuracy: `0.506510`.
- Weighted receiver accuracy: `0.477865`.
- Weighted target-or-packet oracle accuracy: `0.619792`.
- Decision: packet utility transfers to a non-Qwen receiver family on `2/2`
  adjacent slices, but receiver fusion fails on `0/2` slices.

## Closest Communication Priors

- C2C / Cache-to-Cache:
  https://openreview.net/forum?id=LeatkxrBCi
  - C2C uses a learned projection and fusion path for source KV-cache state.
    This is the closest direct model-to-model communication competitor.
    LatentWire's current distinction is the fixed-byte source-private packet
    rather than source KV/cache transfer.
- KVComm:
  https://openreview.net/forum?id=F7rUng23nw
  - Selective KV sharing is a cache/state-transfer comparator. It raises the
    systems bar for quality and latency, but its communication object scales
    with source KV state rather than a constant `5B` record.
- Communicating Activations Between Language Model Agents:
  https://arxiv.org/abs/2501.14082
  - Direct activation exchange is close to latent communication, but it is not
    the same privacy/rate contract as a fixed discrete packet.
- CIPHER:
  https://openreview.net/forum?id=Yf7PaRar7T
  - Embedding-level debate/communication is a related latent-channel prior.
    LatentWire must avoid claiming general non-text communication novelty.

## Common-Basis And Receiver-Fusion Priors

- Relative Representations:
  https://arxiv.org/abs/2209.15430
  - Anchor-relative coordinates are a direct prior for common-basis alignment.
    Future LatentWire common-basis packets must claim downstream packet utility
    and destructive controls, not the anchor idea itself.
- SAE feature-space universality:
  https://arxiv.org/abs/2410.06981
  - SAE feature universality supports sparse/common-feature packet hypotheses
    but is also novelty pressure. Feature-id shuffle and shared/private feature
    controls are required.
- Prefix-Tuning:
  https://arxiv.org/abs/2101.00190
  - Continuous prefixes are a prior for target-conditioning with virtual
    tokens. LatentWire soft-token variants require zero-source/static-prefix
    and same-byte visible-token controls.
- Gist Tokens:
  https://arxiv.org/abs/2304.08467
  - Prompt compression with cached gist tokens is a strong prompt-side
    compression baseline. It differs from source-conditioned private packets,
    but must remain in related work and controls.

## Systems And Quantization Priors

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  and
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  - TurboQuant uses random rotations, scalar quantization, and QJL residual
    correction for vector/KV quantization. It is a byte-floor and native
    systems comparator, not the same as transmitting a task-evidence packet.
- QJL:
  https://arxiv.org/abs/2406.03482
  - QJL-style sign sketches motivate compact residual controls and byte floors.
- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
- SGLang / RadixAttention:
  https://arxiv.org/abs/2312.07104
  - These define native serving metrics for eventual NVIDIA runs. Mac-local
    evidence can report packet bytes and serialization/accounting, not
    production TTFT/ITL/throughput wins.

## Method Decision

The branch is alive because the oracle headroom is large: the two-slice
target-or-packet oracle is `0.113281` above packet-only. The current ridge
receiver is weakened because it cannot exploit this headroom and trails
packet-only.

The next highest-value method should be a query-conditioned sparse/common-basis
innovation packet or receiver. Required controls:

- source-label, source-rank/index, and quantized source-score controls;
- zero-source, row-shuffle, candidate-roll, and candidate-score-roll controls;
- target-derived packet and same-byte visible text controls;
- feature-id shuffle and magnitude shuffle controls for sparse/common-basis
  packets;
- packet-only as the main receiver-fusion baseline.
