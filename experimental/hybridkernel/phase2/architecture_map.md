# HybridKernel Phase 2 Architecture Map

This Mac-local gate estimates whether layer-type boundary movement is large enough to justify GPU profiling later.
It is an activation-stream upper-bound calculation, not an end-to-end latency measurement.

| Config | Layers | Attn | SSM/linear | Boundaries | Boundary bytes/token | Stream fraction | 60% recovered | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ibm-granite-4.0-h-small.config.json | 40 | 4 | 36 | 8 | 131072 | 20.0% | 12.0% | pass_theoretical_gate |
| ibm-granite-4.0-h-tiny.config.json | 40 | 4 | 36 | 8 | 49152 | 20.0% | 12.0% | pass_theoretical_gate |
| qwen3-next-80b-a3b-instruct.config.json | 48 | 12 | 36 | 23 | 188416 | 47.9% | 28.7% | pass_theoretical_gate |

## Decision

Granite 4.0 H Tiny/Small and Qwen3-Next clear the >=3% theoretical activation-stream gate under this upper-bound model.
This keeps HybridKernel alive only for native NVIDIA/vLLM profiling; the source/control audits and integration map are complete enough for Mac-local work.
This map does not prove an end-to-end GPU speedup, and the next gate is to determine whether the apparent boundary cost survives actual vLLM/vendor implementation details in server-side Nsight traces.
