# HybridKernel Control Feasibility Matrix

- date: 2026-05-06
- purpose: pre-NVIDIA reviewer-risk hardening
- status: Mac-local planning artifact; not benchmark evidence

## Native Profiling Controls

| Control | Why it matters | Mac status | GPU status needed | Use in paper |
|---|---|---|---|---|
| Granite 4.0 H Tiny | Closest public Granite Mamba2/Transformer target with local config and 8 boundaries. | Config fetched; architecture map complete. | vLLM native run with server-side Nsight. | Primary if profiling works. |
| Granite 4.0 H Small | Same architecture family, larger hidden size. | Config fetched; architecture map complete. | vLLM native run if memory allows. | Scale check, not required for first gate. |
| Qwen3-Next-80B-A3B | Many inferred hybrid boundaries and lower pre-GPU threshold. | Config fetched; boundary pattern inferred from config. | Native runtime availability and memory feasibility unknown. | Secondary because boundary type differs from Granite Mamba2. |
| Pure or mostly Transformer control | Separates hybrid-boundary effects from ordinary inter-layer handoff. | Not runnable as timing control on Mac. | Matched vLLM profile with same request shape. | Required before speed claim. |
| Mostly SSM/Mamba control | Separates attention/SSM boundary overhead from SSM internals. | Not runnable as timing control on Mac. | Matched vLLM profile if model/runtime available. | Required if boundary overhead is claimed. |
| Warmup/graph-capture control | Prevents mistaking startup or CUDA graph capture for boundary overhead. | Runbook covers repeated fixed requests only. | At least three steady-state metric rows after warmup. | Required by artifact checker and profiler parser. |
| Client-only profile rejection | Avoids profiling HTTP request overhead instead of kernels. | Artifact checker rejects client-only scope. | Server-side Nsight Systems and Compute artifacts. | Required admissibility check. |

## Decision

No Mac-side control can replace native timing. The paper should state that the
control plan is ready, but the controls remain placeholders until a GPU packet
contains server-side traces and repeated metric rows.
