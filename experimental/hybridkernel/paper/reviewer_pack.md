# HybridKernel Reviewer Pack

- status: internal pre-GPU handoff artifact, not a systems result
- current decision: weakly alive only if native profiling finds separable
  attention/SSM boundary overhead

## Paper Link

- Draft PDF: `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf`
- Draft TeX: `experimental/hybridkernel/paper/hybridkernel_colm2026.tex`

## Current Claim

HybridKernel does not claim a GPU speedup. It provides a pre-registered native
profiling packet for testing whether hybrid attention/SSM transitions have
avoidable boundary-local overhead after current vLLM hybrid serving machinery is
enabled.

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| architecture map | Granite has 8 boundaries and 20.0% boundary stream fraction; Qwen3-Next has 23 inferred boundaries and 47.9% | worth profiling, not speed evidence |
| runtime/source audit | vLLM hybrid SSM support already handles important layout and transfer paths | broad novelty weakened |
| threshold model | Granite needs about 25% avoidable boundary traffic at 60% recovery to clear a 3% proxy gain | Mac kernels not justified |
| fixed-request driver | local dry-run plus optional `/start_profile`/`/stop_profile` bracketing | reduces client-only or startup-trace risk |
| parser/checker | metric rows require dtype, graph state, batch shape, control segment, three distinct same-config repeats, matching analysis outputs, and separate Nsight server/client replay logs | stale, mixed-config, warmup-only, incomplete-log, and placeholder evidence rejected |
| Triton interpreter | toy boundary primitive matches CPU reference under `TRITON_INTERPRET=1` | indexing/kernel-plumbing only |

## Reviewer Risks

- No NVIDIA/vLLM profile exists yet.
- The architecture-byte opportunity may be ordinary inter-layer traffic, not
  avoidable overhead.
- vLLM already narrows the novelty window for hybrid SSM layout/transfer.
- The toy Triton primitive is not the proposed production kernel.
- The draft should be read as a measurement protocol, not as a method result.

## Next Exact Gate

Run the user-operated NVIDIA/vLLM packet from
`experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md`, then pass:

1. `experimental/hybridkernel/phase2/check_profiler_run_artifacts.py`
2. `experimental/hybridkernel/phase2/analyze_profiler_metrics.py`

Promote only if at least three same-model/same-config native rows clear the 3%
recoverable-gain gate. Kill or shelve if repeated native summaries show less
than 1% recoverable gain.
