# PQ Systems Comparison References

Date: 2026-04-30

## Primary Sources And Role

- Product Quantization for Nearest Neighbor Search:
  https://doi.org/10.1109/TPAMI.2010.57
  - Role: base product-codebook comparator.
  - Use in paper: canonical PQ is an adjacent codec baseline; the contribution
    is the source-private decoder-side-information evaluation and control
    suite.

- Optimized Product Quantization:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Role: rotation-based PQ baseline.
  - Use in paper: utility-OPQ is the public-mean-sensitive geometry mitigation.

- QuIP#:
  https://arxiv.org/abs/2402.04396
  - Role: Hadamard incoherence and lattice/codebook quantization precedent.
  - Use in paper: motivates structured rotations for protected Hadamard PQ.

- QuaRot:
  https://openreview.net/forum?id=dfqsW38v1X
  - Role: rotation-based low-bit inference systems precedent.
  - Use in paper: supports sign/permutation/Hadamard transforms as a
    hardware-plausible alternative to dense learned rotations.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Role: recent online vector quantization and QJL-residual baseline.
  - Use in paper: positions PQ/Hadamard packets against modern
    rate-distortion systems work rather than claiming PQ itself is new.

- QJL:
  https://arxiv.org/abs/2406.03482
  - Role: 1-bit Johnson-Lindenstrauss / KV quantization baseline.
  - Use in paper: motivates the KV byte-floor and residual-sketch comparator.

- KIVI:
  https://arxiv.org/abs/2402.02750
  - Role: 2-bit asymmetric KV-cache quantization baseline.
  - Use in paper: KV rows are accounting contrasts that expose source KV state,
    not source-private packet baselines.

- KVQuant:
  https://arxiv.org/abs/2401.18079
  - Role: sub-4-bit KV-cache quantization baseline.
  - Use in paper: same KV byte-floor caveat as KIVI.

- Cache-to-Cache:
  https://arxiv.org/abs/2510.03215
  - Role: direct semantic communication via projected/fused source and target
    KV caches.
  - Use in paper: strongest relevant cross-model communication competitor, but
    it communicates source KV/cache state rather than a tiny private packet.

- KVComm:
  https://arxiv.org/abs/2510.03346
  - Role: selective KV-pair sharing for inter-LLM communication.
  - Use in paper: reviewer-relevant KV-sharing comparator; not privacy-equivalent
    because source KV state crosses the boundary.

- Q-KVComm:
  https://arxiv.org/abs/2512.17914
  - Role: adaptive KV-cache compression for multi-agent communication.
  - Use in paper: related systems direction for KV transfer, distinct from
    source-private residual packets.

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
  - Role: serving and KV-memory manager baseline.
  - Use in paper: compare as serving infrastructure only, not as a communication
    method.

- vLLM disaggregated prefill documentation:
  https://docs.vllm.ai/en/latest/features/disagg_prefill.html
  - Role: current production-style prefill/decode split and KV-transfer
    framing.
  - Use in paper: motivates future TTFT/TPOT/goodput telemetry and the
    non-claim boundary for Mac-local results.

- DistServe:
  https://arxiv.org/abs/2401.09670
  - Role: prefill/decode disaggregation and TTFT/TPOT/goodput SLO framing.
  - Use in paper: systems metric vocabulary for future NVIDIA/server runs.

## Table Claims Enabled

- Report bytes crossing the source-target boundary, packet record bytes, cache
  line/DMA floor where available, receiver compute, source text exposure, source
  KV exposure, cached decode, and control pass/fail.
- Mark C2C/KVComm/Q-KVComm/vLLM/DistServe as reference or serving/KV rows, not
  source-private method wins.
- Keep query-aware text and full-log relay as necessary text baselines because
  text can reach oracle when allowed enough private disclosure.

## Non-Claims

- Hadamard rotations are not a privacy mechanism without a formal keyed threat
  model.
- KV/cache compression baselines are not byte-equivalent unless all selected
  layers, positions, K/V tensors, dtype, metadata, and transfer padding are
  counted.
- vLLM and DistServe are serving baselines, not cross-model communication
  methods.
- The PQ systems table does not provide production GPU TTFT, TPOT, goodput, or
  HBM bandwidth evidence.
