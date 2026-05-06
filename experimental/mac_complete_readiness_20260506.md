# Experimental Projects Mac-Complete Readiness Audit

- date: 2026-05-06
- environment: `./venv_arm64`
- Triton status: `triton-cpu` source install available; interpreter gates pass
- scope: HybridKernel, SinkAware, ThoughtFlow-FP8 side projects

## Summary

The three experimental projects are now as far as the current Mac hardware can
take them without changing the scientific question.

| Project | Readiness | What is complete locally | What still requires different evidence |
|---|---|---|---|
| HybridKernel | Mac-complete handoff | Source/runtime audit, threshold model, vLLM fixed-request driver, profiler packet generator/checker, toy Triton interpreter correctness, COLM-style draft | Native NVIDIA/vLLM profiler packet with server-side Nsight Systems and Nsight Compute data |
| SinkAware | Mac-complete pre-GPU candidate | Exact branch killed, approximate rank-2 branch stress-tested on GPT2/OPT controls, downstream patch controls, 48-trace rank frontier, native packet validator, Triton interpreter correctness, COLM-style draft | Native GPU timing/memory traffic and preservation of downstream-control behavior |
| ThoughtFlow-FP8 | Mac-complete diagnostic note | Sparse-cache falsification ladder, `rdu_topk` demotion on alternate/independent surfaces, `psi_topk` and `vwac_topk` fresh-surface failures, current decision manifest, int8 Triton interpreter primitive, COLM-style draft | A new preregistered utility signal on a fresh/larger frozen sparse-cache surface |

## Stop Conditions

HybridKernel should not receive more Mac kernels or benchmark scripts until a
native profiler packet shows separable boundary overhead. The local primitive
is intentionally a plumbing check, not the proposed systems kernel.

SinkAware should not receive more Mac quality sweeps on the current branch
unless a reviewer asks for a specific missing local control. The current Mac
surface already covers sink counts 2/4, lengths 64/96, GPT2/OPT families,
split seeds, downstream patch controls, a rank frontier, and a native packet
validator. The next evidence must be native timing or memory traffic.

ThoughtFlow-FP8 should not receive more current-branch tuning, GPU work, or
FP8 claims. Reopening requires a new preregistered utility family before any
fresh measurement; the consumed `rdu_topk`, `psi_topk`, and `vwac_topk` branches
should not be retuned.

## Reviewer Pack Links

- HybridKernel: `experimental/hybridkernel/paper/reviewer_pack.md`
- SinkAware: `experimental/sinkaware/paper/reviewer_pack.md`
- ThoughtFlow-FP8: `experimental/thoughtflow_fp8/paper/reviewer_pack.md`

## Native Handoff Link

- `experimental/native_gpu_handoff_20260506.md`

## PDF Links

- `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf`
- `experimental/sinkaware/paper/sinkaware_colm2026.pdf`
- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`

## Validation Command

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/tests \
  experimental/hybridkernel/phase0/tests \
  experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests \
  experimental/hybridkernel/phase4/tests \
  experimental/sinkaware/phase2/tests \
  experimental/sinkaware/phase3/tests \
  experimental/sinkaware/phase4/tests \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

Expected current result after the latest handoff hardening: 167 owned Mac tests
pass, with only the opt-in non-interpreter Triton CPU-backend gate skipped on
this Mac.
