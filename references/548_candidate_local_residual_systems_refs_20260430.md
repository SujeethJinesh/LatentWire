# Candidate-Local Residual Systems References

- date: `2026-04-30`
- purpose: primary-source memo for the systems waterfall attached to the live
  candidate-local residual receiver.

## Systems Boundary

The systems claim is a boundary-traffic and receiver-cache claim, not a native
GPU serving claim. The live method sends source-private byte packets, keeps
public MiniLM/candidate features at the receiver, and decodes against cached
candidate-local residual features.

Closest systems comparisons and caveats:

- C2C communicates directly through projected/fused KV caches. This is the
  closest broad "direct semantic communication" competitor, but it moves source
  KV/cache state rather than a source-private byte packet.
  Source: https://openreview.net/forum?id=LeatkxrBCi
- KVComm/KVCOMM-style systems reuse or share KV caches for multi-agent
  inference. They are the right high-rate cache-sharing baselines for future
  NVIDIA/vLLM runs, not baselines defeated by the current Mac artifact.
  Sources: https://openreview.net/forum?id=F7rUng23nw and
  https://arxiv.org/abs/2510.12872
- PagedAttention/vLLM is the relevant production serving reference for KV-cache
  memory management, batching, and metrics. Future ICLR systems rows should
  report TTFT, TPOT/inter-token latency, E2E latency, throughput/goodput, and
  queue/scheduler counters using vLLM-style metrics.
  Sources: https://arxiv.org/abs/2309.06180 and
  https://docs.vllm.ai/en/stable/design/metrics.html
- DistServe motivates goodput/SLO reporting and separating prefill from decode
  in serving claims. Our Mac artifact does not yet make this production claim.
  Source: https://arxiv.org/abs/2401.09670
- FlashAttention motivates IO-aware memory traffic accounting. The current
  artifact therefore reports raw payload bytes, record bytes, 64B cache-line
  bytes, 128B DMA bytes, batched bytes/request, and resident decode read bytes.
  Source: https://arxiv.org/abs/2205.14135
- TurboQuant and QJL/TurboQuant-style KV quantization are compression baselines,
  especially once source/target KV-cache baselines run on NVIDIA hardware.
  Sources: https://arxiv.org/abs/2504.19874 and
  https://arxiv.org/abs/2406.03482

## Result-Guided Implication

The Mac systems waterfall for the live candidate-local residual receiver should
be framed as:

> LatentWire occupies a far-left-rate, no-source-text/no-source-KV point. On the
> n512 held-out gate, the live 8B packet is an 11B record, amortizes to 11B of
> 64B-line traffic per request at batch 64, and a resident sparse decode over
> cached public candidate residuals reproduces the Python decoder exactly in the
> representative seed59 n512 microbench.

Safe non-claims:

- no HBM, PCIe, NVLink, TPOT, goodput, or energy result yet;
- no claim of beating C2C/KVComm/KV cache quantization on their native axes;
- no claim that cold MiniLM feature build is free. It is receiver-local public
  state and must be reported separately from resident decode.
