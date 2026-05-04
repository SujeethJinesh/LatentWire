# Source-Private Systems, Hardware, And Quantization Review

Date: 2026-05-04

Purpose: targeted systems/hardware/quantization review for the source-private
latent communication branch after the Mac packet-ring artifact passed as
accounting evidence and the native NVIDIA/vLLM/SGLang rows remained pending.

## Current Gate Read

- Paper readiness: not ICLR-ready. The systems story is honest but incomplete:
  Mac packet transport supports byte/exposure accounting; native serving,
  HBM, throughput, and energy are still blockers.
- Current story: LatentWire sends tiny source-private task packets decoded
  against public target-side side information. This is a different point from
  C2C/KVComm/DroidSpeak, which move or reuse cache-like internal state.
- Highest-priority gap: clear a positive method gate on larger frozen slices,
  seed repeats, and strict cross-family falsification; then attach native
  serving measurements instead of claiming systems wins from Mac proxies.

## What We Can Honestly Show On Mac Now

The current Mac-local systems win is a measured packed-record transport and
traffic-accounting row, not a serving acceleration row.

Canonical local artifact:

- `paper/source_private_mac_packet_ring_transport_post_receiver_fail_20260504.md`
- packet profile: 1B payload / 4B framed record
- batch64 packet p50: 0.642475 ns/request
- batch64 packet p95: 0.646144 ns/request
- batch64 line/DMA bytes: 4.0/4.0 B/request
- PQ packet profile: 4B payload / 7B framed record
- PQ batch64 p95: 0.680982 ns/request
- local ledger still marks native readiness false

Safe claim: LatentWire has hardware-readable source-private byte accounting and
a stable Mac packed-record microbenchmark showing that the current packet is
cheap to pack/copy/verify in a local synthetic transport loop.

Do not claim: GPU throughput, HBM reduction, TTFT/TPOT/goodput improvement,
energy savings, vLLM/SGLang integration, or wins over native cache-sharing or
KV-quantization methods.

## Native NVIDIA/vLLM/SGLang Runs Needed Later

Minimum native run set:

1. vLLM online serving, OpenAI-compatible endpoint:
   - rows: target-only, same-byte text, structured text, LatentWire packet
     receiver, source-shuffled/zero/target-derived packet controls;
   - metrics: successful requests, TTFT, TPOT/ITL, e2e latency, output token
     throughput, request throughput, queue time, prefill time, KV cache usage,
     prefix-cache queries/hits;
   - sweeps: concurrency 1/4/16/64, fixed prompt/output lengths, and at least
     one long-context/shared-prefix workload.
2. SGLang online serving with RadixAttention/prefix-cache enabled:
   - same rows and prompt IDs as vLLM;
   - record SGLang `cache_hit_rate`, token usage, TTFT, TPOT, e2e latency, and
     request replay artifacts.
3. Native GPU counter run:
   - Nsight Systems/Compute or equivalent for HBM read/write bytes, kernel
     time, GPU utilization, PCIe/NVLink transfer bytes when multi-process or
     multi-GPU communication is used;
   - `nvidia-smi`/NVML power samples for energy/request if stable.
4. Competitor/native comparator rows:
   - C2C cache-fusion where code/model pair is runnable;
   - KVComm/KVCOMM/DroidSpeak-style cache sharing or reuse where assumptions
     match;
   - KIVI/KVQuant/QJL/TurboQuant or best available KV quantization kernels as
     same-quality KV-cache byte-floor comparators.
5. Matched-quality protocol:
   - same frozen prompts, same sampling/decode settings, same model snapshot
     IDs, same answer parser, paired uncertainty, and source-private controls;
   - strict same-family and strict cross-family separation.

## Defensible Mac-Local Next Experiment

Run a rate-and-traffic ledger extension, not another speed claim:

- Extend `build_source_private_mac_packet_ring_transport_microbench.py` with a
  predeclared packet format sweep:
  - 1B/4B candidate packet
  - 2B/5B legacy packet
  - 4B/7B PQ packet
  - 8B/12B source-score packet
  - 12B/16B train-donor packet
- Add modeled KV-cache byte floors for the actual target/source model shapes:
  - fp16 KV
  - KIVI-like 2-bit KV
  - KVQuant-like 3-bit KV
  - QJL/TurboQuant-like 3 to 3.5-bit KV
- Output p50/p95 local transport time, coefficient of variation, cache-line
  rounded bytes, DMA-burst rounded bytes, and ratio to full hidden-log and
  quantized-KV floors.
- Pair the systems table with the current target-self-resonance soft-prefix
  capacity rows, but label that as capacity/headroom, not source-private
  transfer.

This is defensible because it only claims rate/traffic accounting and local
packet handling. It also creates the exact table native vLLM/SGLang runs must
later validate or falsify.

## Primary Sources And What They Establish

1. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   https://arxiv.org/abs/2504.19874
   - Establishes the strongest current rate-distortion pressure on any claim
     that communicating vectors/cache state is expensive by default. The
     abstract reports quality neutrality at 3.5 bits/channel and marginal
     degradation at 2.5 bits/channel for KV cache quantization.
   - Use in LatentWire: byte-floor comparator and quantization inspiration, not
     evidence that LatentWire beats TurboQuant.

2. ICLR 2026 TurboQuant poster.
   https://iclr.cc/virtual/2026/poster/10006985
   - Establishes venue status and the authors' framed contribution: random
     rotation, scalar quantizers, and 1-bit QJL residual correction for MSE and
     inner-product distortion.
   - Use in LatentWire: cite as current systems/quantization frontier.

3. Google Research TurboQuant blog.
   https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
   - Establishes the authors' official applied framing: KV bottlenecks, zero
     overhead, QJL residual bias correction, and long-context benchmark claims.
   - Use cautiously: blog numbers need independent reproduction before being
     used as a hard comparator.

4. Revisiting RaBitQ and TurboQuant.
   https://arxiv.org/abs/2604.19528
   - Establishes a live reproducibility dispute around TurboQuant runtime,
     recall, and KV-cache quantization results.
   - Use in LatentWire: avoid leaning on unreproduced TurboQuant speed claims;
     require our own native runs.

5. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead. https://arxiv.org/abs/2406.03482
   - Establishes the sign-bit JL sketch plus asymmetric inner-product estimator
     and reports more than 5x KV-cache memory reduction at 3 bits without
     accuracy loss in their evaluated setting.
   - Use in LatentWire: lower-bound-style byte-floor comparator for vector/KV
     communication.

6. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   https://arxiv.org/abs/2402.02750
   - Establishes asymmetric KV quantization: keys per-channel, values
     per-token, with a hardware-friendly 2-bit cache path and 2.6x lower peak
     memory including weights in their report.
   - Use in LatentWire: practical native KV compression baseline.

7. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization. https://arxiv.org/abs/2401.18079
   - Establishes sub-4-bit KV-cache quantization techniques: per-channel keys,
     pre-RoPE key quantization, non-uniform datatypes, and dense/sparse
     outlier handling; reports 3-bit quantization with less than 0.1 perplexity
     degradation and very long-context serving claims.
   - Use in LatentWire: stronger byte-floor than naive fp16 KV accounting.

8. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models. https://arxiv.org/abs/2510.03215
   - Establishes the closest broad competitor: source KV projection/fusion into
     the target cache with learnable gating; reports accuracy gains over
     individual/text communication and latency speedup.
   - Boundary: not source-private. It communicates cache semantics, so it must
     be separated from LatentWire's fixed-byte private task packet.

9. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   https://arxiv.org/abs/2510.03346
   - Establishes selective KV-pair sharing as an ICLR 2026 communication
     competitor, with layer-wise selection and as few as 30 percent of layers'
     KV pairs transmitted in their report.
   - Boundary: moves KV pairs; not equivalent to a source-private packet.

10. KVCOMM: Online Cross-context KV-cache Communication for Efficient
    LLM-based Multi-agent Systems. https://arxiv.org/abs/2510.12872
    - Establishes training-free online cache reuse across agent contexts by
      aligning offset variance with an anchor pool; reports over 70 percent
      reuse without quality degradation.
    - Boundary: cache reuse/serving systems competitor, not private latent task
      evidence.

11. DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM
    Serving. https://arxiv.org/abs/2411.02820
    - Establishes same-family or compatible-model cross-LLM communication by
      reusing embeddings/KV cache to reduce prefill latency.
    - Boundary: useful native serving comparator, but its compatibility
      assumptions differ from cross-family source-private packets.

12. vLLM/PagedAttention.
    https://arxiv.org/abs/2309.06180
    - Establishes the serving substrate: paged KV cache management, near-zero
      KV memory waste, cache sharing, and 2-4x throughput over prior serving
      systems in their evaluation.
    - Use in LatentWire: native target for packet endpoint benchmarking.

13. vLLM metrics documentation.
    https://docs.vllm.ai/en/v0.15.0/design/metrics/
    - Establishes available Prometheus metrics: KV cache usage, prefix-cache
      queries/hits, prompt/generation tokens, TTFT, inter-token latency, e2e
      latency, queue time, and prefill time.
    - Use in LatentWire: required native reporting fields.

14. vLLM benchmark CLI.
    https://docs.vllm.ai/en/v0.20.1/benchmarking/cli/
    - Establishes the supported online serving benchmark path and output fields:
      request throughput, token throughput, TTFT, TPOT, and ITL.
    - Use in LatentWire: runbook foundation for native benchmark scripts.

15. SGLang/RadixAttention.
    https://arxiv.org/abs/2312.07104
    - Establishes RadixAttention for KV-cache reuse and compressed FSMs for
      structured output; reports up to 6.4x throughput in evaluated programs.
    - Use in LatentWire: second native serving target, especially for shared
      prefixes and structured/choice decoding.

16. SGLang production metrics documentation.
    https://docs.sglang.io/docs/references/production_metrics
    - Establishes Prometheus metrics including token usage, cache hit rate,
      TTFT, e2e latency, and time per output token.
    - Use in LatentWire: required SGLang reporting fields.

17. NVIDIA GenAI-Perf documentation.
    https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2500/user-guide/docs/perf_analyzer/genai-perf/README.html
    - Establishes an NVIDIA-supported benchmarking path for LLM throughput and
      latency, including output token throughput, TTFT, ITL, and request
      throughput.
    - Use in LatentWire: independent client-side benchmark tool for native
      endpoint runs.

18. EdgeBERT.
    https://arxiv.org/abs/2011.14203
    - Establishes Thierry Tambe-style algorithm-hardware co-design for
      latency-aware energy optimization on NLP inference.
    - Use in LatentWire: systems bar is co-designed algorithm plus hardware
      counters, not post-hoc byte arithmetic alone.

19. 12nm 18.1 TFLOPs/W sparse transformer processor.
    https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf
    - Establishes the hardware contribution style: entropy-based early exit,
      mixed-precision predication, and fine-grained power management measured
      on silicon.
    - Use in LatentWire: a serious hardware claim needs per-query latency,
      energy/power, precision/dataflow, and memory-hierarchy measurements.

20. BlockDialect.
    https://arxiv.org/abs/2501.01144
    - Establishes mixed-format quantization with block-wise format selection
      and low-precision integer-compatible formats for energy-efficient LLM
      inference.
    - Use in LatentWire: packet/latent formats should be presented as a
      formatbook/rate-distortion co-design if we make hardware claims.

21. SemanticDialect.
    https://arxiv.org/abs/2603.02883
    - Establishes a current Tambe-lab direction: semantic-aware mixed-format
      quantization with low online selection cost.
    - Use in LatentWire: source-private packet formats should be tied to
      semantic importance and cheap online selection if we want a hardware
      contribution rather than only an ML protocol.

## Hardware Implications

For a Thierry Tambe-style systems contribution, the eventual story must be
co-designed across algorithm, serving runtime, and hardware counters:

- Algorithm: source-private packet rate, packet entropy, selective precision,
  and source-destroying controls.
- Runtime: vLLM/SGLang integration, prefix/cache behavior, request scheduling,
  and structured/choice decoding overhead.
- Hardware: cache-line/DMA traffic, HBM bytes, kernel time, power/energy per
  request, and low-precision format compatibility.

Mac evidence can support the first accounting layer. Native GPU and serving
evidence are required for the second and third layers.

## Claims To Avoid

- Avoid "native speedup", "HBM reduction", "energy efficiency", or "GPU
  throughput" until NVIDIA rows exist.
- Avoid "beats TurboQuant/QJL/KIVI/KVQuant" unless those baselines are run or
  modeled only as clearly labeled byte floors.
- Avoid "beats C2C/KVComm/DroidSpeak" because those are cache-sharing
  competitors with different privacy and compatibility assumptions.
- Avoid "source-private latent communication in general" while strict
  cross-family gates remain negative.
- Avoid "Mac packet-ring time implies endpoint speed"; it only proves packed
  record handling under a synthetic local loop.
- Avoid using TurboQuant blog/H100 claims as ground truth during an active
  reproducibility dispute; cite them as author claims and require local native
  reproduction.

## Decision

Promote the Mac packet-ring result only as a systems-boundary and accounting
artifact. The next Mac-local systems experiment should widen the rate/traffic
ledger and quantized-KV byte floors. The next paper-critical method branch
remains target self-resonance/query-resampler capacity, but source-private
claims are blocked until the encoder generalizes and beats destructive controls
on larger slices and a strict cross-family pair.
