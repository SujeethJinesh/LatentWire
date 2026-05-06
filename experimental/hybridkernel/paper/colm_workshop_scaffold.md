# HybridKernel: Profiler-Driven Boundary Fusion for Hybrid Attention/SSM Serving

- venue target: COLM-style workshop shell, not a submission draft
- status: **weakly alive**
- date: 2026-05-06
- evidence level: Mac-local config audit, runtime source audit, CPU reference
  scaffold, Triton interpreter correctness, and pre-GPU threshold model

## Abstract

Hybrid language models that interleave attention and state-space layers are an
attractive target for efficient serving, but current serving stacks already
optimize much of the hybrid state-management path. HybridKernel investigates a
narrow remaining question: whether attention/SSM layer transitions create
measurable compute-boundary overhead that could be reduced by a fused boundary
operator. Current evidence does not support a performance claim. A Mac-local
architecture map shows that boundary-crossing activation streams are large
enough to inspect, while a runtime audit weakens the opportunity because vLLM
already provides sophisticated hybrid state layout and transfer mechanisms. The
next contribution gate is therefore a native NVIDIA/vLLM profiling protocol
that can either expose distinct boundary conversion, materialization, launch, or
locality overhead, or kill the branch before unnecessary kernel work.

## 1. Introduction

Hybrid attention/SSM models such as Granite 4.0 H, Nemotron-H, Bamba, and
Qwen3-Next motivate systems work beyond pure Transformer serving. The appealing
hypothesis is simple: when an attention layer hands off to an SSM layer, or vice
versa, the runtime may materialize, reshape, schedule, or reload hidden-state
data in a way that a boundary-aware kernel could avoid.

The current evidence forces a narrower framing. The Phase 1 source audit found
no direct evidence of an existing fused attention-to-SSM layer-boundary compute
kernel, but the Phase 2 runtime audit found that vLLM already handles important
hybrid SSM serving mechanics through HMA shared tensors, dual descriptor views,
DS conv layout, and no-buffer/no-reshuffle transfer paths. This means
HybridKernel cannot claim a broad hybrid-serving gap. It can only proceed if
native profiling reveals an avoidable per-layer compute-boundary cost.

The paper story, if the next gate succeeds, is:

> Hybrid serving systems optimize state layout and transfer, but still expose a
> measurable compute-boundary overhead at attention/SSM transitions; a small
> fused boundary operator removes that overhead while preserving model
> semantics.

The paper story, if the next gate fails, is:

> The apparent activation-byte opportunity is mostly ordinary hidden-state
> handoff rather than avoidable boundary overhead, so this branch should not be
> submitted as a positive method.

## 2. Current Method Hypothesis

HybridKernel would fuse only the narrow computation adjacent to an
attention/SSM layer transition. Candidate effects include:

- avoiding a standalone layout conversion or materialization kernel;
- reducing redundant residual stream reads/writes around the transition;
- collapsing launch gaps between adjacent layer-type kernels;
- improving cache locality for the hidden-state handoff.

The method is not a replacement for vLLM HMA/NIXL transfer machinery, Mamba
internals, FlashAttention, or SSM scan kernels. It must compose with those
systems and demonstrate a residual boundary-local gain after they are enabled.

## 3. Measurement Plan

The next gate is defined in
`experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`.

Primary target:

- Granite 4.0 H Tiny or Small under vLLM on a local NVIDIA Linux host.

Required traces:

- Nsight Systems timeline with CUDA, NVTX, OS runtime, fork tracing, and CUDA
  graph node tracing;
- Nsight Compute counter pass on only suspicious boundary kernels;
- repeated fixed-request runs with paired uncertainty;
- same-family controls where available;
- explicit separation between source communication and target-cache/runtime
  cache effects.

Promotion threshold:

- at least 3% estimated end-to-end gain, or a larger localized boundary cost
  with a concrete path to 3% end-to-end;
- repeated-run stability;
- one strict same-family control;
- one cross-family falsification pair before widening benchmark claims.

## 4. Current Evidence

| Artifact | Result | Interpretation |
|---|---|---|
| `phase1/lit_review.md` | No fused attention-to-SSM compute-boundary kernel found in the quick audit; vLLM hybrid SSM work is highly relevant. | Proceed only with a narrow compute-boundary claim. |
| `phase2/architecture_map.md` | Granite 4.0 H Tiny/Small show 8 boundaries and 20.0% boundary stream fraction; Qwen3-Next shows 23 inferred boundaries and 47.9%. | Activation streams are large enough to inspect, but this is not latency evidence. |
| `phase2/runtime_boundary_audit.md` | vLLM already handles important hybrid state-transfer and layout paths. | Broad hybrid-layout novelty is weakened; profiler evidence is mandatory. |
| `phase2/pre_gpu_threshold_model.md` | Granite needs about 25.0% of boundary traffic to be genuinely avoidable at 60% recovery to clear a 3% proxy gain; Qwen3-Next needs about 10.4%. | Mac-only implementation is not justified. |
| `phase3/reference/boundary.py` and tests | CPU boundary blend scaffold exists. | Useful for semantics if profiling promotes implementation, but not evidence of speed. |
| `phase4/kernel/boundary_triton.py` and tests | Triton interpreter tests pass under the repo-local `triton-cpu` source install. | Kernel logic only; no GPU or Mac performance claim. |
| `phase0/local_preflight.json` and `.md` | PyTorch 2.6.0 imports with MPS available; CUDA is unavailable; `triton==3.7.0+git270e696d` is importable from source; package-index checks for `triton`, `triton-cpu`, and `triton-nightly` still find no compatible wheel. | Local Phase 4 correctness is unblocked, but native performance evidence is still absent. |
| `phase2/profiler_driver.py` | Fixed-request OpenAI-compatible driver dry-runs locally; runbook now profiles the vLLM server and drives it from a second local terminal. | Avoids client-only Nsight traces. |
| `phase2/check_profiler_run_artifacts.py` | Future native run directories are checked for metadata, server-side Nsight Systems and Compute scope, Nsight artifacts, logs, readout rows, distinct repeated metric rows, and matching profiler-analysis outputs. | GPU evidence must be artifact-complete, server-side, independently repeated, and analytically fresh before the draft cites it. |
| `phase2/tests/fixtures/synthetic_profiler_run_packet/` | Synthetic packet fixture exercises the checker without GPU access. | Documents packet shape only; it is not profiler evidence. |
| `phase2/native_run_packet_checklist.md` | Single checklist for the packet a GPU operator must return. | Local Mac readiness is saturated; wait for native data. |

## 5. Limitations

- No NVIDIA/vLLM profiling has been run yet.
- No Mac result can support a GPU performance claim.
- Triton package-index wheels are unavailable from the current Mac ARM64
  venv/index, but a source-built `triton-cpu` install now runs the interpreter
  tests locally.
- The architecture map counts boundary-crossing hidden-state bytes, many of
  which are ordinary inter-layer traffic rather than avoidable overhead.
- vLLM already implements sophisticated hybrid state layout and disaggregated
  transfer paths, so the novelty window is narrow.
- Qwen3-Next is less directly matched to the Granite Mamba2 boundary-fusion
  hypothesis because its boundary type differs.
- The branch is not yet tied to a benchmark-backed positive method for the main
  latent-transfer paper.

## 6. Next-Gate Checklist

- [ ] Run the NVIDIA/vLLM profiler runbook locally on a GPU host.
- [ ] Save immutable environment metadata, exact vLLM version, and command
  lines.
- [ ] Save `metadata/profile_scope.json` showing that Nsight captured the
  vLLM server or a single-process vLLM benchmark for both Nsight Systems and
  Nsight Compute, not only the HTTP client.
- [ ] Annotate attention/SSM boundaries against the Nsight Systems timeline.
- [ ] Run Nsight Compute only on suspicious boundary kernels and matched
  same-type controls.
- [ ] Compute paired uncertainty over repeated fixed-request runs.
- [ ] Run `phase2/check_profiler_run_artifacts.py --run-dir "$HWK_RUN"` and
  save `artifact_check.json`; the checker must confirm
  `profiler_analysis_gate.json`/`.md` were generated from the same metric rows.
- [ ] Return the full packet described in
  `phase2/native_run_packet_checklist.md`.
- [ ] Decide promote, pause, or kill using the 3% end-to-end threshold.
- [ ] If promoted, implement only the smallest fused boundary operator needed
  to test the observed overhead.
- [ ] If killed, record that activation-byte upper bounds did not translate to
  avoidable runtime overhead.

## 7. Draft Claim Guardrails

Allowed now:

- "HybridKernel is a weakly alive profiler-driven systems branch."
- "Mac-local evidence motivates an NVIDIA/vLLM profiling gate."
- "No GPU performance claim has been established."

Not allowed now:

- "HybridKernel improves vLLM throughput."
- "HybridKernel reduces GPU memory traffic."
- "HybridKernel is COLM-ready evidence."
- "The boundary bytes in the architecture map are avoidable."
