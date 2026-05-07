# HybridKernel Phase 2 Profiler Gate Preregistration

**Frozen on**: 2026-05-07
**Frozen by**: human author (codifies pre-existing decision rule from runbook)
**Frozen at git SHA**: <SET BY /goal AT started_at_sha>

## Status note

This preregistration codifies the pre-existing HybridKernel decision rule
documented in:
  - experimental/hybridkernel/phase2/nvidia_vllm_profiler_runbook.md
  - experimental/HANDOFF.md ("HybridKernel Decision Rule" section)

The criteria below were established before any GPU profiler data was
collected. This file freezes them into the swarm contract. If the runbook
contains stricter or more specific criteria than what is written here, the
runbook is authoritative and this file MUST be amended (with audit trail)
before the gate executes. /goal must not interpret a runbook-prereg
discrepancy as license to relax criteria.

## Hypothesis

Hybrid Mamba-Transformer attention/SSM boundaries create a separable
conversion / materialization / launch / locality overhead that admits a
boundary-fusion systems contribution.

## Models

- **Primary**: ibm-granite/granite-4.0-h-tiny
- **Same-family control**: ibm-granite/granite-4.0-h-small
- **Cross-family falsification control**: Qwen/Qwen3-Next-80B-A3B-Instruct
  (or whatever model is committed to
  experimental/hybridkernel/phase2/cross_family_control_replacement_template.json
  BEFORE profiler execution; post-hoc substitution is forbidden and triggers
  KILL_HYBRIDKERNEL_BELOW_SHELF.)

## Required artifacts (admissible packet)

The result packet must contain, all under the run_dir:

- metadata/environment.txt and metadata/environment.json
- metadata/profile_scope.json
- metadata/model_provenance.json
- copied metadata/native_control_matrix.json
- metadata/reduction_input_manifest.json
- filled reduction worksheet or equivalent cited source file
- row-specific client replay logs
- server-side Nsight Systems logs and artifacts
- server-side Nsight Compute logs and artifacts (unless explicit
  no-boundary-signal kill mode is used per runbook)
- profiler_metrics.json
- profiler_analysis_gate.json
- profiler_analysis_gate.md
- artifact_check.json

## Required rows

profiler_metrics.json must contain at least 9 valid reduced rows:
- 3 primary HybridKernel rows
- 3 same-shape same-family controls
- 3 same-shape cross-family falsification rows

Rows must have distinct run IDs, distinct artifacts, distinct time windows,
and SHA-256 hashes tying back to Nsight artifacts.

## Decision rule

### PROMOTE (decision string: PROMOTE_HYBRIDKERNEL_PROFILER_GATE)

ALL of the following must hold:
1. Repeated primary recoverable gain ≥ 3% (preregistered shelf).
2. Bootstrap / interval readout: 95% CI lower bound > 0.
3. Same-family controls: each row stays below 3% recoverable gain.
4. Cross-family controls: each row stays below 3% recoverable gain.
5. check_profiler_run_artifacts.py --require-full-matrix exits 0.

### KILL (decision string: KILL_HYBRIDKERNEL_BELOW_SHELF)

ANY of the following triggers a kill:
1. Repeated native summaries show < 1% recoverable gain.
2. Same-family or cross-family controls reproduce the same signal at or
   above the 3% gate.
3. Packet cannot be made artifact-complete.
4. Cross-family substitution is post-hoc rather than committed pre-profile
   to the replacement template.
5. Any preregistration drift detected by audit subagent.

### AMBIGUOUS (decision string: KILL_HYBRIDKERNEL_BELOW_SHELF)

If primary gain is 1-3% (between kill floor and promotion ceiling), the
result is treated as kill. There is no ambiguous-promote path. The shelf
exists precisely to prevent ambiguous promotions.

## Forbidden actions

- Adding GPU numbers to any paper before this gate's
  artifact_check.json shows pass.
- Substituting the cross-family control after profiler runs without
  pre-committed template.
- Re-running primary rows after observing same-family or cross-family
  results (selection bias).
- Modifying this preregistration after started_at_sha.

## On promote

If gate promotes, queue.yml's on_pass appends:
  - hybridkernel_kernel_prototype

The Triton kernel prototype must match the profiler-predicted gain within
±20%. That is a separate downstream gate, not part of this preregistration.

## On kill

Write experimental/KILLED_hybridkernel_below_shelf/README.md with the full
kill manifest including:
- artifact SHAs of every row
- decision metric values
- reason classification (below-floor, controls-reproduce, packet-incomplete,
  substitution-violation, prereg-drift)
- date of kill

The HybridKernel paper draft is preserved (not deleted) but marked as
killed in HANDOFF.md.
