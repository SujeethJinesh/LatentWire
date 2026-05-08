# Decode Microkernel Phase 2 Infra Diagnostic

Current paper readiness: not camera-ready. Phase 1 remains replay evidence only.

Current story: Decode Microkernel Consolidation is still a positive-method
pivot, but it has not yet entered a real serving loop. No paper-level serving
speedup claim is admissible.

Exact blocking gap: the repo has no supported vLLM decode micro-operation
replacement hook or serving-side scheduler integration that implements the
fixed Phase 1 DMC schedule in inference.

## Gate Result

- Packet: `experimental/decode_microkernel/phase2/results/dmc_phase2_20260508T0045Z`
- Runner decision hint: `FAIL_INFRA_DMC_PHASE2`
- Checker decision: `FAIL_INFRA_DMC_PHASE2`
- Checker reasons:
  - method does not declare a real serving integration
  - method does not declare real serving rows
  - primary: row count 0 below required 12
  - same_family: row count 0 below required 12
  - cross_family: row count 0 below required 12

This is an infrastructure failure, not a scientific kill. There are no serving
rows, so `KILL_DMC_PHASE2_NO_SERVING_GAIN` is not supported.

## Hard Infra Checklist

- Exact error / failure mode inspected: current runner records no real serving
  integration and zero rows.
- Web/upstream check: vLLM V1 offline `LLM.generate()` per-request metrics are
  known to return `None`; the upstream RFC for restoring offline per-request
  metrics was closed as not planned:
  `https://github.com/vllm-project/vllm/issues/26298`.
- vLLM serving benchmark path checked: vLLM's online serving benchmark supports
  TTFT, TPOT, ITL, and E2EL metrics, but that measures an existing serving
  backend rather than implementing DMC:
  `https://docs.vllm.ai/en/v0.10.2/api/vllm/benchmarks/serve.html`.
- Alternative configs tested in `.debug/`: eager and CUDA graph modes on
  Granite Tiny, using in-process V1 engine and lower-level step outputs.
- Fresh subagent review completed: independent checker run confirmed
  `FAIL_INFRA_DMC_PHASE2` with fixed Phase 1 hashes and prompt manifest intact.

## Observability Diagnostic

Non-gate diagnostic artifacts are summarized in:

`experimental/decode_microkernel/phase2/results/dmc_phase2_20260508T0045Z/observability_diagnostic.json`

The diagnostic shows:

- Granite Tiny loads and generates on this GPU.
- In-process vLLM lower-level stepping exposes generated token IDs and
  per-step wall times.
- Nsight Systems captures an NVTX decode range and launch activity.
- CUDA graph mode emits `cudaGraphLaunch` activity.
- The DMC-disabled harness is token-identical to baseline and therefore is not
  a DMC implementation.

## Positive-Method Implication

The next positive-method path is not to reinterpret this as a negative paper.
The next valid move is a fresh implementation branch that adds a real DMC
serving integration before any Phase 2 row is collected. Plausible directions:

1. A vLLM model-local custom-op path inside Granite/Nemotron hybrid decode
   layers that replaces the fixed Phase 1 schedule with a packed Triton/CUDA
   path.
2. A serving-side scheduler pass that batches the fixed repeated micro-op
   schedule into fewer launches inside decode.
3. A persistent-kernel implementation for the fixed DMC schedule, with token
   equality and Nsight launch audits as the first gate.

Any of these requires implementation before the Phase 2 checker can return
PASS or KILL. Existing CUDA graph toggles alone are not scientifically a DMC
method.
