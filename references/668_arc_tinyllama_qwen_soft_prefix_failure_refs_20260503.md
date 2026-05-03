# ARC TinyLlama-to-Qwen Soft-Prefix Failure References

Date: 2026-05-03

## Purpose

This memo records the literature and systems boundary after the negative ARC
TinyLlama-to-Qwen n64 soft-prefix/query repair gate. The result weakens
answer-CE soft-prefix repair as the missing ICLR-positive method and sharpens
how LatentWire must be framed if a later connector succeeds.

## Not Novel By Itself

- Prefix-tuning:
  https://arxiv.org/abs/2101.00190
  - Continuous learned prefix vectors for a frozen LM are established prior
    art. LatentWire must not claim novelty from soft tokens alone.
- Prompt tuning:
  https://arxiv.org/abs/2104.08691
  - Learned prompt embeddings are also established.
- Input-dependent soft prompting:
  https://arxiv.org/abs/2506.05629
  and
  https://arxiv.org/abs/2405.18203
  - Dynamic prompt generation increases overlap risk. The needed distinction
    is source-private cross-model communication under a fixed-byte boundary.

## Compression And Query Bottleneck Boundary

- Gist tokens:
  https://arxiv.org/abs/2304.08467
- AutoCompressors:
  https://arxiv.org/abs/2305.14788
- Perceiver IO:
  https://arxiv.org/abs/2107.14795
- Flamingo Perceiver Resampler:
  https://arxiv.org/abs/2204.14198
- BLIP-2 / Q-Former:
  https://arxiv.org/abs/2301.12597

These works make learned latent query bottlenecks and soft-token compression
well-established. The defensible novelty is not the connector architecture;
it is a per-example source-conditioned fixed-byte packet that survives
source-destroying and target-cache controls.

## Cross-Model Communication Boundary

- C2C / Cache-to-Cache:
  https://arxiv.org/abs/2510.03215
- KVComm:
  https://arxiv.org/abs/2510.03346
- KVCOMM:
  https://arxiv.org/abs/2510.12872
- Communicating activations between language-model agents:
  https://arxiv.org/abs/2501.14082
- InterLat:
  https://arxiv.org/abs/2511.09149

These are the closest problem-family baselines. They transmit dense
activations or KV state, while LatentWire aims for extreme-rate packet
communication with explicit byte and privacy accounting.

## Common-Basis And Resonance Boundary

- Relative representations:
  https://arxiv.org/abs/2209.15430
- SVCCA:
  https://arxiv.org/abs/1706.05806
- CKA:
  https://arxiv.org/abs/1905.00414
- SAE feature universality:
  https://arxiv.org/abs/2410.06981
- Universal SAEs:
  https://arxiv.org/abs/2502.03714
- Crosscoders:
  https://www.anthropic.com/research/crosscoder-model-diffing

These works support the common-basis/resonance framing, but they do not by
themselves provide a downstream fixed-byte source-private communication
protocol. They are better treated as diagnostics or regularizers for the next
branch.

## Activation Steering Boundary

- Activation addition:
  https://arxiv.org/abs/2308.10248
- Representation Engineering:
  https://arxiv.org/abs/2310.01405
- Contrastive Activation Addition:
  https://arxiv.org/abs/2312.06681
- Inference-Time Intervention:
  https://arxiv.org/abs/2306.03341

If LatentWire uses a static vector, reviewers can reasonably classify it as
activation steering. The communication claim requires per-example,
source-conditioned packets and wrong-source controls.

## Systems Boundary

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
- SGLang:
  https://arxiv.org/abs/2312.07104
  and
  https://docs.sglang.io/docs/developer_guide/bench_serving
- LMCache:
  https://arxiv.org/abs/2510.09665
  and
  https://docs.lmcache.ai/developer_guide/architecture.html
- CacheGen:
  https://arxiv.org/abs/2310.07240
- SnapKV:
  https://arxiv.org/abs/2404.14469
- Quest:
  https://arxiv.org/abs/2406.10774
- H2O:
  https://arxiv.org/abs/2306.14048
- TurboQuant:
  https://arxiv.org/abs/2504.19874
  and
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- QJL:
  https://arxiv.org/abs/2406.03482

The systems claim is currently defensible only as a byte/privacy/interface
story. Native NVIDIA goodput, HBM traffic, and serving superiority require
vLLM/SGLang/LMCache/KV-compression rows with TTFT, TPOT, throughput, peak
memory, HBM read/write, interconnect bytes, and cache residency.

Relevant measurement references:

- NVIDIA GenAI-Perf metrics:
  https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2510/user-guide/docs/perf_analyzer/genai-perf/README.html
- Nsight Compute profiling guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Nsight Systems user guide:
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- PyTorch MPS notes:
  https://docs.pytorch.org/docs/2.9/notes/mps.html
- Hugging Face Apple Silicon caveats:
  https://huggingface.co/docs/transformers/perf_train_special

## Decision Boundary

The ARC TinyLlama-to-Qwen n64 soft-prefix gate fails:

- matched soft-prefix: `0.218750`;
- target-only: `0.406250`;
- packet-only source index: `0.468750`;
- Qwen-substituted packet: `0.437500`;
- same-byte visible text: `0.500000`;
- pass gate: `False`.

Rule out shallow answer-CE soft-prefix/query repair on the current Mac-local
ARC surface. If a future branch succeeds, the novelty claim should be:

1. per-example source-conditioned fixed-byte packets;
2. source-private communication across a model boundary;
3. destructive controls showing the receiver uses the source packet rather
   than target cache, static prefix, visible text, or source-label leakage.
