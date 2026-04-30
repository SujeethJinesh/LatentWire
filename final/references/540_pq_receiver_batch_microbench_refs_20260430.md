# PQ Receiver Batch Microbench References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://doi.org/10.1109/TPAMI.2010.57
  - Role: canonical product-codebook primitive.
  - Use in paper: PQ is the adjacent codec primitive; novelty comes from the
    source-private residual-packet evaluation and destructive controls.

- Optimized Product Quantization:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Role: rotation-based PQ baseline.
  - Use in paper: supports the OPQ-Procrustes and utility-OPQ geometry rows.

- QuIP#:
  https://arxiv.org/abs/2402.04396
  - Role: incoherence/Hadamard preprocessing precedent for low-bit inference.
  - Use in paper: motivates structured rotations as a systems-plausible
    geometry mitigation, not as a new privacy mechanism.

- QuaRot:
  https://openreview.net/forum?id=dfqsW38v1X
  - Role: rotation-based low-bit transformer inference.
  - Use in paper: supports the hardware-friendly structured-rotation framing.

- SpinQuant:
  https://openreview.net/forum?id=ogO6DGE6FZ
  - Role: learned rotations for low-bit LLM quantization.
  - Use in paper: related rotation/quantization precedent; LatentWire does not
    claim learned rotation novelty.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Role: recent quantization/rate-distortion baseline.
  - Use in paper: positions PQ packets against quantization systems work while
    preserving the distinction between codec quality and source-private
    communication.

- QJL:
  https://arxiv.org/abs/2406.03482
  - Role: 1-bit Johnson-Lindenstrauss/KV sketching baseline.
  - Use in paper: related compact sketching and KV byte-floor comparator.

- KIVI:
  https://arxiv.org/abs/2402.02750
  - Role: asymmetric 2-bit KV-cache quantization.
  - Use in paper: KV compression/exposure comparator, not a source-private
    packet baseline.

- KVQuant:
  https://arxiv.org/abs/2401.18079
  - Role: sub-4-bit KV-cache quantization.
  - Use in paper: same KV byte-floor and source-KV exposure boundary.

- Cache-to-Cache:
  https://arxiv.org/abs/2510.03215
  - Role: closest broad cross-model cache communication competitor.
  - Use in paper: reviewers will compare any broad latent-communication claim
    to C2C; LatentWire must stay scoped to tiny source-private packets unless
    it measures comparable cache/serving rows.

- KVComm:
  https://arxiv.org/abs/2510.03346
  - Role: selected KV-pair sharing for inter-LLM communication.
  - Use in paper: relevant systems competitor, but it communicates source KV
    state rather than a few-byte packet.

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
  - Role: serving and KV-memory baseline.
  - Use in paper: metric vocabulary and future serving integration boundary,
    not a result claimed by the Mac microbench.

- DistServe:
  https://arxiv.org/abs/2401.09670
  - Role: prefill/decode disaggregation and SLO framing.
  - Use in paper: future TTFT/TPOT/goodput telemetry framing.

- FlashAttention:
  https://arxiv.org/abs/2205.14135
  - Role: IO-aware attention systems precedent.
  - Use in paper: supports the need to report memory movement and kernel
    boundaries rather than only algorithmic FLOPs.

## Novelty Boundary

Novel:

- The batch microbench evaluates a source-private PQ residual packet as a
  decision-causal communication object with exact destructive-control context.
- The receiver timing is tied to the actual public candidate side-information
  table used by the packet decoder.
- The artifact reports exact prediction parity, ID hashes, packet-record
  amortization, and resident/public-table cost separation across canonical,
  OPQ, and protected-Hadamard PQ variants.

Not novel:

- PQ, OPQ, Hadamard rotations, table lookup, and vector quantization.
- KV/cache compression and cache-level model communication.
- Serving-stack optimization.

## Claims Enabled

- Geometry-mitigated PQ packets can be decoded by a small resident
  target-side table lookup with zero prediction mismatches across batch sizes.
- The Mac-local receiver microkernel is sub-0.25 ms/request on the n500 gate,
  with 7-byte packet-record traffic amortized to 7 bytes/request at batch 256.
- The result supports a boundary-traffic and receiver-kernel systems claim.

## Non-Claims

- No GPU/vLLM TTFT, TPOT, goodput, HBM, PCIe, or NVLink result.
- No protocol-free latent reasoning.
- No claim that Hadamard rotations are cryptographic protection.
- No claim that LatentWire beats C2C/KVComm on their native high-rate cache
  communication objective.
