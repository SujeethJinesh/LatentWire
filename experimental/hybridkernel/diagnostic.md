# HybridKernel Kill Diagnostic

- Date: 2026-05-07
- Packet: `phase2/results/hybridkernel_profiler_gate_20260507T212428Z`
- Paper readiness: not ICLR-ready. Estimated distance is one successful fresh positive-method branch plus benchmark-quality repeats; the original HybridKernel boundary-fusion branch is now 0% as a paper unless a new preregistered hypothesis passes.
- Current story: vLLM/native serving does not expose a separable attention/SSM layer-boundary conversion or materialization cost on the tested full matrix. The observed traces are dominated by ordinary GEMV, MoE alignment/fused-MoE, direct-copy, and selective-scan style kernels rather than a boundary-local operator.
- Blocking gap: no measurable, preregistered positive mechanism remains. The current packet gives no speedup claim and no kernel target with recoverable headroom.

## Inputs Read

- `swarm/goal.md`, "Persistence and pivot behavior": kills require diagnosis, not paper drafting; pivots need fresh preregistration and cannot loosen thresholds, cherry-pick, or swap models/datasets post hoc.
- `experimental/hybridkernel/paper/reviewer_pack.md`: HybridKernel was alive only if native profiling found separable attention/SSM boundary overhead.
- `experimental/hybridkernel/progress.md`: Mac/source/kernel scaffolding was saturated before the GPU gate.
- Profiler packet readout/checker:
  - `artifact_check.json`: `PASS`, 9 rows, full matrix, `packet_mode: no_boundary_signal_kill`, `metrics_status: KILL or shelve: native profiler summaries show less than 1% recoverable gain.`
  - `readout.md`: no distinct boundary conversion/materialization kernel, no boundary-local launch gap, no layer-boundary NVTX range, no NCU target selected.
  - `profiler_analysis_gate.md`: all primary, same-family, and cross-family rows have 0.00% mean gain upper bound and primary 95% CI `[0.00%, 0.00%]`.

## Proximate Failure Type

**(a) Hypothesis wrong.**

The killed hypothesis was that hybrid attention/SSM transitions in the native vLLM serving path would reveal a separable boundary-local conversion/materialization or launch/locality cost large enough to justify a fused boundary kernel. The full matrix did not show that signal:

- 3 Granite primary rows, 3 Granite same-family controls, and 3 Nemotron cross-family replacement controls all reduce to `no_boundary_signal_kill`.
- Recoverable-gain upper bound is exactly `0.000000` for every row.
- Nsight Systems found ordinary serving kernels but no isolatable boundary-local kernel or NVTX boundary window.
- Nsight Compute was correctly skipped because there was no preregistered kernel/window to profile.

This is not an infrastructure issue: the artifact checker passes and native Nsight artifacts are present. It is not preregistration ambiguity: the runbook and checker explicitly allow a clean `no_boundary_signal_kill` packet. It is not best classified as setup-insufficient for the original claim, because a production-relevant server-side trace with same-family and cross-family controls was the intended decision surface.

## Fair Retest Assessment

No fair retest of the original hypothesis is recommended. Re-running the same gate with different seeds, thresholds, post-hoc model substitutions, or selected layers would violate the pivot rules.

A related retest would only be fair as a fresh hypothesis if it changes the observable mechanism before seeing new rows, for example "source-instrumented per-layer NVTX reveals a hidden boundary-local cost that server-level traces cannot isolate." That would need a new preregistration, new pass/fail thresholds, and source-level instrumentation criteria. It should not be treated as a rescue run for HybridKernel; it would be an instrumentation-first measurement branch, and its COLM value is low unless it predicts a concrete optimization.

## Alternative Positive-Method Hypotheses

| Hypothesis | Trace hint | Plausibility | Likely paper claim if it passes | COLM competitiveness | Fresh preregistration? |
|---|---|---:|---|---|---|
| Decode microkernel consolidation for hybrid/MoE serving | Granite rows are dominated by repeated BF16 GEMV and fused-MoE/MoE-align kernels; Nemotron rows are dominated by GEMV plus small selective-scan updates. The missing boundary signal suggests the opportunity is not at layer boundaries but in decode-time microkernel launch density and small-batch GEMV shape handling. | Medium | A scheduler/kernel packing method reduces decode latency for hybrid/MoE LLM serving without relying on boundary-local conversion overhead. | Medium if it beats vLLM baselines across frozen same-family and cross-family slices with paired latency CIs; weak if it is only a local kernel trick. | Yes. New gate must predefine kernel classes, batch/request shapes, latency metric, controls, and minimum speedup. |
| Selective-scan state update specialization | Nemotron replacement traces show `_selective_scan_update_kernel`, but it is not boundary-local and is small relative to GEMV. A method targeting scan-state update locality or batching could be a more honest hybrid-specific systems branch than attention/SSM boundary fusion. | Low-medium | A state-update specialization improves hybrid SSM decode throughput while preserving logits/quality. | Medium-low: competitive only if gains survive against native vLLM/FlashInfer-style serving and are not model-specific. | Yes. New prereg should require isolated scan-update headroom before implementation and strict quality equivalence. |
| Source-instrumented layer-lifecycle profiler | The packet lacked layer-boundary NVTX ranges, but that absence killed the kernel hypothesis rather than proving all lifecycle costs are absent. A positive method could add reusable lifecycle instrumentation plus an optimization only if it exposes stable hidden costs across models. | Low | A profiler-guided lifecycle method finds and removes hidden state movement or synchronization costs in hybrid LLM serving. | Low unless the first instrumented run finds a stable, optimizable anomaly; high risk of becoming a measurement-only result. | Yes, but only as a measurement-gated branch with a stop rule before any kernel work. |

## Decision

- Ruled out: original HybridKernel attention/SSM boundary-fusion branch under the current preregistered profiler gate.
- Saturated: Mac-only kernel scaffolding, source audit, packet/checker mechanics, and the current boundary-overhead decision surface.
- Still alive: only fresh-preregistered systems hypotheses driven by observed non-boundary trace structure.
- Highest-priority next branch, if continuing from this diagnostic: decode microkernel consolidation, because it is closest to the dominant observed cost and has the clearest path to a positive systems method. Do not run it under the killed HybridKernel preregistration.
