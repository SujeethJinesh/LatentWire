# HybridKernel Reviewer Pack

- status: pre-GPU handoff artifact, not a systems result
- current decision: weakly alive only if native profiling finds separable
  attention/SSM boundary overhead

## Paper Link

- Draft PDF: `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf`
- Draft TeX: `experimental/hybridkernel/paper/hybridkernel_colm2026.tex`

## Current Claim

HybridKernel does not claim a GPU speedup. It provides a pre-registered native
profiling packet for testing whether hybrid attention/SSM transitions have
avoidable boundary-local overhead after current vLLM hybrid serving machinery is
enabled. A kernel contribution exists only if native traces reveal boundary
overhead not already handled by vLLM's hybrid SSM machinery.

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| architecture map | Granite has 8 boundaries and 20.0% boundary stream fraction; Qwen3-Next has 23 inferred boundaries and 47.9% | worth profiling, not speed evidence |
| runtime/source audit | vLLM hybrid SSM support already handles important layout and transfer paths | broad novelty weakened |
| threshold model | Granite needs about 25% avoidable boundary traffic at 60% recovery to clear a 3% proxy gain | Mac kernels not justified |
| fixed-request driver | local dry-run plus optional `/start_profile`/`/stop_profile` bracketing | reduces client-only or startup-trace risk |
| parser/checker | metric rows require dtype, graph state, batch shape, prompt/decode/request-token shape, control segment, boundary direction, model/control-family role, explicit same-family `control_window_ids`, explicit NCU launch-selection provenance, reduction command, recoverable-fraction basis, SHA-256 artifact hashes, structured `environment.json`, model snapshot manifests with matching hashes, three distinct same-config repeats, primary-repeat bootstrap CI low above zero, matching analysis outputs, separate Nsight server logs, and non-dry-run client replay JSON with all request statuses `ok`; the GPU gate command now runs the checker with `--require-full-matrix` so primary-only packets fail promotion | stale, mixed-config, wrong-control, missing-control, warmup-only, dry-run, failed-request, shape-mismatched, incomplete-log, hash-mismatched, weak-CI, and placeholder evidence rejected |
| Triton interpreter | toy boundary primitive matches CPU reference under `TRITON_INTERPRET=1` | indexing/kernel-plumbing only |

The real native row schema also requires row role, control family, boundary
direction, kernel names, boundary indices, stable same-family non-boundary
`control_window_ids`, reduction time window, NCU launch selection
(`kernel_regex`, launch skip/count, source Nsight Systems artifact, matching
source time window, and derivation notes), recoverable fraction basis, reduction
command, reduction notes, and relative in-packet Nsight artifact paths with
matching SHA-256 digests. Nsight Compute artifacts
are optional only for an explicit `no_boundary_signal_kill` packet whose rows
are a clean below-1% kill, whose readout/reduction notes record no suspicious
boundary-local Nsight Systems signal, and whose rows use
`ncu_artifact: "not_run_no_boundary_signal"`.
The runbook now also requires a reduction input manifest tying each reduced row
to exact Nsight exports, windows, commands, and script hashes; analyst-selected
timeline windows without that manifest are review risks, not camera-ready
evidence.
Structured environment metadata and model snapshot manifests are now part of
the packet contract: `metadata/environment.json` records parseable GPU, profiler,
Python, and package versions, while each model provenance row points to a
snapshot manifest whose SHA-256 must match the file copied into the packet.
The GPU packet handoff also includes
`phase2/reduction_worksheet_template.tsv` and
`phase2/cross_family_control_replacement_template.json`. The first makes manual
timeline reduction auditable; the second is the only admissible way to replace
Qwen3-Next with a smaller cross-family hybrid before profiling. Any
cross-family replacement chosen after inspecting profiler output is audit-only.

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

Promote only if at least three distinct primary native repeats all clear the 3%
recoverable-gain gate, the primary-repeat bootstrap CI low end is above zero,
and three same-shape same-family plus three same-shape cross-family controls
stay below that gate. This would promote only the boundary-overhead
investigation and prototype-kernel plan, not a paper-level throughput or vLLM
speedup claim. If Qwen3-Next or another preregistered feasible matrix control
is missing, the packet is audit-only. Kill or shelve if repeated native
summaries show less than 1% recoverable gain.
