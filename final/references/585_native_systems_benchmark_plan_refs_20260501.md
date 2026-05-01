# Native Systems Benchmark Plan References

Web check: 2026-05-01. Scope: primary sources for the native vLLM/SGLang
systems benchmark plan, GPU metric schema, cache/KV communication baselines,
and the safe non-claim boundary.

## Local Result

Artifact:
`results/source_private_native_systems_benchmark_plan_20260501/`

- pass gate: `True`
- native systems complete: `False`
- required baselines: `11`
- required metrics: `44`
- headline benchmarks: `ARC-Challenge`, `OpenBookQA`
- diagnostic benchmark: `HellaSwag`
- serving substrates: `vLLM`, `SGLang`

## Serving Metrics And Substrates

- Kwon et al., Efficient Memory Management for Large Language Model Serving
  with PagedAttention, SOSP 2023. vLLM/PagedAttention is the main native
  serving substrate; the paper reports KV-cache memory-management and
  throughput gains, so our native table must measure TTFT, TPOT, goodput,
  peak GPU memory, and cache traffic rather than relying on byte accounting.
  https://arxiv.org/abs/2309.06180
- Zheng et al., SGLang: Efficient Execution of Structured Language Model
  Programs, NeurIPS 2024. SGLang/RadixAttention is the second serving
  substrate because it optimizes KV reuse and structured decoding, giving a
  scheduler/runtime sensitivity check for LatentWire packets.
  https://arxiv.org/abs/2312.07104
- MLCommons, MLPerf Inference 5.1 Small LLM task note, 2025. This official
  MLPerf note separates prompt-phase TTFT and generation-phase TPOT for LLM
  inference, matching the latency metrics required by the runbook.
  https://mlcommons.org/2025/09/small-llm-inference-5-1/

## Cache Communication And Quantized Source-State Baselines

- Fu et al., Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models, ICLR 2026. C2C is the closest direct semantic cache
  communication baseline and claims accuracy and latency gains versus text,
  but it exposes source KV state and must be compared natively.
  https://arxiv.org/abs/2510.03215
- Shi et al., KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing, ICLR 2026. KVComm selectively shares KV pairs/layers, so it is a
  required source-state baseline with explicit source-KV exposure flags.
  https://arxiv.org/abs/2510.03346
- Ye et al., KVCOMM: Online Cross-context KV-cache Communication for
  Efficient LLM-based Multi-agent Systems, 2025. KVCOMM targets online
  cross-context KV reuse; fair comparison needs native prefill/decode timing,
  memory, and source-state exposure accounting. https://arxiv.org/abs/2510.12872
- Zandieh et al., QJL: 1-Bit Quantized JL Transform for KV Cache
  Quantization with Zero Overhead, 2024. QJL motivates low-bit sign-sketch
  source-state rows and HBM/cache-memory metrics, but it is not a
  source-private fixed-packet method. https://arxiv.org/abs/2406.03482
- Zandieh et al., TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate, 2025. TurboQuant motivates low-bit online vector/KV
  baselines and residual quantization ideas; our claim boundary requires
  native quality and GPU traffic measurements before comparison.
  https://arxiv.org/abs/2504.19874

## Safe Boundary

The benchmark plan strengthens the systems contribution by making the future
GPU table exact: every row must report benchmark/split/model-pair identity,
implementation and commit hash, GPU/driver/CUDA metadata, serving engine,
load shape, quality, paired uncertainty, TTFT, TPOT, inter-token latency,
goodput, token throughput, peak GPU memory, HBM read/write, PCIe-or-NVLink
bytes, payload/framed/source-state bytes, source-exposure flags, token
lengths, and wall time. It does not close the native systems blocker. The
paper may say the native systems gate is specified and ready to run; it may
not claim throughput, latency, memory, or C2C/KVComm/QJL/TurboQuant
superiority until those rows are measured.
