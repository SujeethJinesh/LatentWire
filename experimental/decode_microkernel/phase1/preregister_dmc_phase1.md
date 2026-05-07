# Decode Microkernel Consolidation Phase 1 Preregistration

- Frozen: 2026-05-07
- Branch: `decode_microkernel`
- Gate: `dmc_phase1`
- Gated by: `decode_microkernel_phase0` PASS

## Paper Readiness

Current paper readiness is not ICLR-ready. Phase 0 showed a trace-level
consolidation opportunity, not a speedup. Phase 1 is the first implementation
gate for a positive method.

## Fresh Positive-Method Hypothesis

Hybrid/MoE decode serving spends measurable time launching many repeated
microkernels from a small number of kernel families. A bounded Decode
Microkernel Consolidation prototype can reduce trace-derived decode replay
latency by consolidating repeated launch-heavy micro-operations while preserving
numerical equivalence.

This is not a rescue of the killed HybridKernel boundary-fusion hypothesis.
No attention/SSM boundary-local claim is allowed. The method target is
decode-time microkernel launch density and repeated micro-operation packing.

## Fixed Inputs

Phase 1 must use the Phase 0 PASS packet:

`experimental/decode_microkernel/phase0/results/decode_microkernel_phase0_20260507T233130Z`

The runner must also use the sanitized HybridKernel profiler packet only for
trace-derived operation-family weights and artifact provenance:

`experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z`

No paper speedup claim may cite Phase 1 unless this checker passes.

## Method Surface

The prototype must implement a trace-derived consolidation method for repeated
decode micro-operations. Acceptable bounded implementations include:

- a Triton or CUDA fused/packed replay kernel for repeated elementwise,
  reduction, gating, or routing-style micro-operations selected before timing;
- a PyTorch/Triton scheduler that batches repeated small operations into fewer
  launches using a fixed operation schedule derived from Phase 0;
- a CUDA graph or persistent-kernel replay that reduces launch count for the
  frozen micro-operation schedule.

Forbidden:

- changing Phase 0 thresholds after seeing Phase 1 numbers;
- selecting only favorable rows after timing;
- claiming boundary-fusion speedup;
- benchmarking only CPU or only Python overhead;
- replacing the fixed Phase 0 packet with a different trace source.

## Required Rows

The result packet must include at least 9 paired rows:

- 3 `primary_hybrid` trace-derived schedules;
- 3 `same_family_control` trace-derived schedules;
- 3 `cross_family_falsification` trace-derived schedules.

Each row must run baseline and consolidated replay on the same GPU, with:

- at least 100 warmup iterations not counted;
- at least 1000 measured iterations;
- fixed random seeds recorded;
- identical input tensors between baseline and consolidated replay;
- maximum absolute output error and relative error recorded;
- launch count or kernel-call count recorded for both paths;
- timing from CUDA events or an equivalent GPU-side timer, not wall-clock Python
  alone.

## Metrics

For each paired row:

- `baseline_ms_median`
- `consolidated_ms_median`
- `latency_reduction_fraction =
  (baseline_ms_median - consolidated_ms_median) / baseline_ms_median`
- `launch_reduction_fraction =
  (baseline_launch_count - consolidated_launch_count) / baseline_launch_count`
- `max_abs_error`
- `max_rel_error`

Aggregate by role with a bootstrap 95% CI over rows for latency reduction.

## PASS Decision

The checker returns `PASS_DMC_PHASE1_CONSOLIDATED_REPLAY` and exits 0 only if
all conditions hold:

1. Phase 0 source packet is exactly the fixed PASS packet above and its checker
   decision is PASS.
2. At least 3 valid rows per role are present.
3. Numerical equivalence:
   - every row has `max_abs_error <= 1e-2`;
   - every row has `max_rel_error <= 1e-2`.
4. Launch reduction:
   - every row has `launch_reduction_fraction >= 0.25`.
5. Latency reduction:
   - primary median latency reduction is at least 0.08;
   - same-family median latency reduction is at least 0.08;
   - cross-family median latency reduction is at least 0.05;
   - bootstrap 95% CI lower bound is greater than 0 for each role.

## KILL Decision

The checker returns `KILL_DMC_PHASE1_NO_REPLAY_GAIN` and exits nonzero if the
packet is complete but any numerical-equivalence, launch-reduction, or
latency-reduction threshold fails.

## INFRA Decision

The checker returns `FAIL_INFRA_DMC_PHASE1` and exits nonzero if artifacts are
missing, inputs are not the fixed Phase 0 PASS packet, timing is not GPU-side,
rows are fewer than required, or the checker cannot mechanically reproduce the
decision.

## On PASS

Promote to a Phase 2 serving integration gate. A paper draft is still not
camera-ready until the method improves real vLLM/serving latency on frozen
prompt slices with paired uncertainty.

## On KILL

Write a KILL manifest and diagnostic. Do not draft a negative-result paper.
