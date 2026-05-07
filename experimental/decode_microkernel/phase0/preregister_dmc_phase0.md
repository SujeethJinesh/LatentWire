# Decode Microkernel Consolidation Phase 0 Preregistration

- Frozen: 2026-05-07
- Branch: `decode_microkernel`
- Gate: `dmc_phase0`
- Status: frozen before any Decode Microkernel Consolidation Phase 0 result rows are reduced.

## Paper Readiness

Current paper readiness is not ICLR-ready. The killed HybridKernel boundary-fusion branch produced a useful diagnostic packet but no positive method. Estimated distance to ICLR readiness is one successful fresh positive-method branch, followed by larger frozen slices, seed repeats, paired latency uncertainty, oracle/headroom diagnostics, and strict same-family versus cross-family separation.

## Current Story

The killed HybridKernel profiler gate did not find a separable attention/SSM boundary conversion or materialization cost. Its diagnostic instead showed that decode traces are dominated by repeated small decode kernels: BF16 GEMV, MoE alignment/fused-MoE, selective-scan update, and nearby elementwise/reduction kernels. The fresh positive-method story is therefore not boundary fusion. It is Decode Microkernel Consolidation: before authoring any production fused kernel, test whether native hybrid/MoE serving has enough decode-time kernel launch density and concentrated repeated kernel families to justify a consolidation method.

## Blocking Gap

The blocking gap is whether there is a measurable consolidation opportunity at all. Phase 0 must answer that using fixed profiler artifacts and client logs only. It must not relax or reinterpret the killed HybridKernel preregistration, must not run new GPU inference, and must not claim a speedup.

## Fresh Pivot Statement

This is a fresh pivot from the killed HybridKernel diagnostic dated 2026-05-07. The killed HybridKernel hypothesis remains killed: no boundary-local operator is promoted, no boundary threshold is changed, and no paper result is claimed from the killed packet. This Phase 0 gate uses the prior sanitized profiler artifacts only as an auditable measurement surface for a different positive-method candidate.

## Positive-Method Hypothesis

If hybrid/MoE decode serving is dominated by dense launches from a small set of repeated decode microkernel families, then a later Decode Microkernel Consolidation method may reduce decode latency by batching, packing, scheduling, or fusing those microkernels. Phase 0 passes only if the fixed traces show enough launch density and top-kernel concentration to justify a Phase 1 implementation gate.

## Fixed Inputs

The only allowed input packet is:

`experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z`

The runner must use:

- `artifact_check.json`
- `profiler_metrics.json`
- `metadata/environment.json`
- `metadata/reduction_input_manifest.json`
- `nsys/*.sanitized.sqlite`
- `logs/client_*.log`

The runner must not launch vLLM, Nsight, CUDA kernels, remote commands, or GPU inference.

## Metrics

For each admissible row, compute from the sanitized Nsight Systems `KERNEL_SUMMARY` table and matching client log:

- `decode_tokens_total`: sum of successful requested completion tokens from the client log.
- `total_kernel_launches`: sum of `KERNEL_SUMMARY.launches`.
- `total_kernel_time_ms`: sum of `KERNEL_SUMMARY.total_ns / 1e6`.
- `launches_per_decode_token`: `total_kernel_launches / decode_tokens_total`.
- `top3_time_fraction`: fraction of kernel time in the three largest kernel names by time.
- `top5_launch_fraction`: fraction of launches in the five largest kernel names by launch count.
- `candidate_launch_fraction`: launch fraction in predefined candidate classes.
- `candidate_time_fraction`: time fraction in predefined candidate classes.
- `candidate_classes_present`: candidate classes with nonzero launches.

Candidate classes are frozen as:

- `gemv`: kernel name contains `gemv`.
- `moe`: kernel name contains `moe` or `expert`.
- `selective_scan`: kernel name contains `selective_scan` or `DeviceScan`.

## PASS Decision

The checker must return `PASS` and exit 0 only if all conditions hold:

1. Input admissibility:
   - upstream `artifact_check.json.status` is `PASS`;
   - upstream `profiler_metrics.json.packet_mode` is `no_boundary_signal_kill`;
   - at least 8 of 9 source rows are admitted;
   - admitted coverage is at least 3 `primary_hybrid`, 2 `same_family_control`, and 3 `cross_family_falsification` rows;
   - each admitted row has a nonempty `KERNEL_SUMMARY`, valid client log, positive decode token count, and matching sanitized SQLite SHA-256.
2. Launch density:
   - median `launches_per_decode_token` is at least 500 for `primary_hybrid`;
   - median `launches_per_decode_token` is at least 500 for `same_family_control`;
   - median `launches_per_decode_token` is at least 300 for `cross_family_falsification`;
   - every admitted row has `launches_per_decode_token` at least 250.
3. Concentration:
   - median `candidate_time_fraction` is at least 0.65 for every row role;
   - median `candidate_launch_fraction` is at least 0.20 for every row role;
   - median `top3_time_fraction` is at least 0.60 for `primary_hybrid`;
   - median `top3_time_fraction` is at least 0.60 for `same_family_control`;
   - median `top3_time_fraction` is at least 0.85 for `cross_family_falsification`.
4. Class support:
   - admitted `primary_hybrid` rows include both `gemv` and `moe`;
   - admitted `same_family_control` rows include both `gemv` and `moe`;
   - admitted `cross_family_falsification` rows include both `gemv` and `selective_scan`.

A PASS promotes only a Phase 1 implementation gate. It is not a paper result and cannot be cited as a speedup.

## KILL Decision

The checker must return `KILL` and exit nonzero if input admissibility passes but any density, concentration, or class-support condition fails. KILL means the decode microkernel consolidation candidate is not justified from this fixed diagnostic surface.

## INFRA Decision

The checker must return `INFRA` and exit nonzero if the packet cannot be mechanically audited: missing fixed inputs, bad JSON, unreadable SQLite, hash mismatch, too few admissible rows, invalid client logs, or threshold metadata mismatch.

## Required Result Packet

The runner must write `experimental/decode_microkernel/phase0/results/<run_id>/` with:

- `environment.json`
- `input_artifact_manifest.json`
- `metrics.json`
- `command_metadata.json`
- `logs/stdout.log`
- `logs/stderr.log`

The checker must print a JSON object containing at least `decision`, `run_dir`, and `reasons`.
