# HybridKernel Phase 2 Runtime Boundary Audit

Status: **ALIVE BUT WEAKENED; pivot to profiler-driven boundary overhead.**

This gate asks whether the activation-stream upper bound from
`architecture_map.md` still looks like a real systems opportunity after checking
current runtime implementation details.

## Source Audit

| Source | Relevant implementation detail | Effect on HybridKernel |
|---|---|---|
| vLLM hybrid SSM disaggregated serving blog, 2026-04-21 | vLLM describes Hybrid Memory Allocator shared tensors, dual descriptor views, DS conv-state layout, 3-descriptor conv transfer, no staging buffers, and no post-transfer reshuffling for hybrid SSM/FA disaggregated serving. | Strongly weakens any claim that hybrid state transfer/layout is an open gap. The remaining possible wedge is narrower: per-layer compute-boundary fusion, not state movement between P/D workers. |
| vLLM `ssm_conv_transfer_utils` docs | The source docs explicitly decompose Mamba2 conv state into contiguous x/B/C subregions under DS layout for NIXL descriptor registration. | Confirms the highest-risk layout-transfer path is already engineered. HybridKernel must not claim to save those bytes. |
| vLLM `mamba_mixer2` docs | Mamba2 has its own backend and CUDA path; the docs describe input-dependent Delta/B/C and Mamba2 backend plumbing. | Confirms optimized Mamba internals. I still did not find evidence of a fused attention-output-to-next-SSM compute kernel, but absence from docs is not enough for a systems claim. |
| Prior local architecture map | Granite 4.0 H Tiny/Small and Qwen3-Next clear the theoretical activation-stream gate, with 12.0% to 28.7% recovered activation-stream upper bounds under the optimistic 60% recovery assumption. | Keeps the project alive as a profiling target, but those bytes are mostly ordinary inter-layer hidden-state traffic. They are not automatically removable boundary overhead. |

## Decision

The architecture-map gate overestimated actionability because it counted
boundary-crossing hidden-state bytes. In real inference, most of that stream is
the normal residual/hidden-state handoff between layers, not an extra format
conversion or avoidable transfer.

The branch remains alive only if native profiling later finds one of these:

1. a distinct attention-to-SSM layout conversion kernel;
2. a launch/materialization gap at the attention/SSM boundary;
3. redundant residual writes/reads that a fused boundary operator can remove;
4. measurable scheduler/cache locality loss around hybrid layer boundaries.

## Current Gate Outcome

| Criterion | Outcome |
|---|---|
| Existing hybrid state-transfer/layout implementation found? | yes, in vLLM HMA/NIXL hybrid SSM path |
| Existing attention-to-SSM layer-boundary compute fusion found? | no evidence in docs/quick audit |
| Does activation-byte upper bound alone justify implementation? | no |
| Should this move to Mac-only implementation? | no |
| Should this move to native GPU profiling later? | yes, only as a profiler-driven falsification gate |

## Next Gate

Do not build more HybridKernel code on Mac. The next real gate is a native GPU
profiling runbook that measures whether hybrid layer boundaries create separate
conversion/materialization/launch overhead in vLLM or vendor runtimes.

Minimum native evidence required before implementation:

- per-layer kernel timeline around attention/SSM boundaries;
- memory-read/write counters or Nsight Compute evidence for extra materialized
  hidden-state traffic;
- matched pure-SSM, pure-attention, and hybrid-layer segments if available;
- an ablation showing the boundary overhead is at least 3% end-to-end or a
  larger localized percentage with a credible route to end-to-end gain.

Until then, HybridKernel is a **future systems spinout**, not COLM_v3 evidence.
