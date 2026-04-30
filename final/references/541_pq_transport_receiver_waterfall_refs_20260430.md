# PQ Transport + Receiver Waterfall References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://doi.org/10.1109/TPAMI.2010.57
  - Role: canonical product-codebook compression primitive.
  - Use in paper: PQ is the adjacent codec primitive; novelty is the
    source-private residual-packet protocol, destructive controls, and joined
    packet transport plus receiver accounting.

- Optimized Product Quantization:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Role: rotation-based PQ baseline.
  - Use in paper: supports OPQ-Procrustes and utility-OPQ rows, but OPQ itself
    is not claimed as novel.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Role: recent vector-quantization and rate-distortion comparator.
  - Use in paper: motivates stronger quantization baselines and residual
    coding variants, while preserving the distinction between codec quality and
    source-private model-to-model communication.

- QJL:
  https://arxiv.org/abs/2406.03482
  - Role: 1-bit JL/KV sketching baseline.
  - Use in paper: byte-floor comparator for source-KV exposure and future
    packet-sketch inspiration.

- KIVI:
  https://arxiv.org/abs/2402.02750
  - Role: asymmetric 2-bit KV-cache quantization.
  - Use in paper: KV compression baseline; it still communicates source KV
    state, unlike the PQ packet.

- KVQuant:
  https://arxiv.org/abs/2401.18079
  - Role: sub-4-bit KV-cache quantization for long-context inference.
  - Use in paper: KV byte-floor and exposure comparator.

- Cache-to-Cache:
  https://arxiv.org/abs/2510.03215
  https://openreview.net/forum?id=LeatkxrBCi
  - Role: closest broad cache-level inter-model communication competitor.
  - Use in paper: C2C projects and fuses source KV-cache state; LatentWire's
    claim must stay scoped to tiny source-private packets unless we implement a
    comparable cache baseline.

- KVComm:
  https://arxiv.org/abs/2510.03346
  https://openreview.net/forum?id=F7rUng23nw
  - Role: selective KV-pair communication competitor.
  - Use in paper: relevant systems baseline; it sends selected source KV pairs
    instead of a few-byte source-private decision packet.

- KVCOMM:
  https://arxiv.org/abs/2510.12872
  - Role: online cross-context KV-cache communication.
  - Use in paper: related multi-agent cache communication; compare on
    state-exposure and byte boundary, not native benchmark wins.

- Q-KVComm:
  https://arxiv.org/abs/2512.17914
  - Role: adaptive KV-cache compression for multi-agent communication.
  - Use in paper: reinforces that reviewers will expect KV-compression
    baselines and exposure accounting.

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
  - Role: serving memory-management baseline.
  - Use in paper: future integration target and metric vocabulary for
    production serving claims.

- DistServe:
  https://arxiv.org/abs/2401.09670
  - Role: TTFT/TPOT/goodput and prefill/decode disaggregation framing.
  - Use in paper: systems metrics to report once NVIDIA serving experiments are
    available.

- FlashAttention:
  https://arxiv.org/abs/2205.14135
  - Role: IO-aware attention systems precedent.
  - Use in paper: supports reporting actual memory movement and boundary
    traffic rather than only algorithmic byte counts.

- CacheGen:
  https://arxiv.org/abs/2310.07240
  - Role: KV-cache compression/streaming for serving.
  - Use in paper: related cache-transfer systems baseline and non-claim
    boundary for the Mac packet-ring result.

- Infinite-LLM:
  https://arxiv.org/abs/2401.02669
  - Role: distributed KV-cache serving for long context.
  - Use in paper: future systems comparator for cache movement and serving
    architecture.

- LMCache:
  https://lmcache.ai/tech_report.pdf
  - Role: KV-cache layer for production inference systems.
  - Use in paper: future production-serving comparator and non-claim boundary.

- Wyner-Ziv coding with side information:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - Role: theoretical side-information framing.
  - Use in paper: target public candidate state is decoder side information;
    LatentWire does not claim a new source-coding theorem.

- Distributed indirect source coding with decoder side information:
  https://arxiv.org/abs/2405.13483
  - Role: recent side-information source-coding reference.
  - Use in paper: mathematical analogy for source-private packets with decoder
    side information.

## Novelty Boundary

Novel:

- The artifact joins measured packet-ring transport and exact target-side PQ
  receiver timing for the same 7-byte source-private packet record.
- The waterfall separates source-private packet traffic from private-text
  relay and source-KV byte floors.
- The gate ties systems accounting to the same destructive-control method
  branch that produced the n500 PQ residual-code lift.

Not novel:

- PQ, OPQ, Hadamard/rotation preprocessing, table lookup, and vector
  quantization.
- KV-cache quantization, KV-cache sharing, or production serving optimization.
- Cache-to-cache semantic communication through source KV/cache state.

## Claims Enabled

- A 7-byte source-private PQ residual packet can be transported and decoded
  exactly on the Mac-local gate, with packet transport p95 below `1 us/request`
  and receiver batch/resident p95 below `0.25 ms/request`.
- Query-aware text uses `2x` record bytes and exposes private text; KV
  byte-floor rows expose source KV and are hundreds of times slower in this
  packet-ring copy microbench.
- The result supports a boundary-traffic plus receiver-kernel systems
  contribution for source-private packets.

## Non-Claims

- No NVIDIA/vLLM TTFT, TPOT, goodput, HBM, PCIe, or NVLink result.
- No claim that LatentWire beats C2C, KVComm, KVCOMM, or Q-KVComm on their
  native cache-sharing tasks.
- No protocol-free cross-model latent reasoning.
- No new quantization primitive.
