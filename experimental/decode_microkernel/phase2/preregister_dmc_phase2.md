# Decode Microkernel Consolidation Phase 2 Preregistration

- Frozen: 2026-05-08
- Branch: `decode_microkernel`
- Gate: `dmc_phase2`
- Gated by: `decode_microkernel_phase1` PASS

## Paper Readiness

Current paper readiness is not camera-ready. Phase 1 showed a positive
trace-derived replay result, not an end-to-end serving result. Phase 2 is the
first gate allowed to support a paper-level serving acceleration claim.

## Fresh Positive-Method Hypothesis

Decode Microkernel Consolidation can be integrated into a real decode-serving
loop so that repeated launch-heavy decode micro-operations selected from the
fixed Phase 1 packet reduce end-to-end serving decode latency without changing
model outputs.

This remains separate from the killed HybridKernel boundary-fusion hypothesis.
No attention/SSM boundary-local speedup claim is allowed.

## Fixed Inputs

Phase 2 must use the Phase 1 PASS packet:

`experimental/decode_microkernel/phase1/results/dmc_phase1_20260508T000525Z`

The runner may use the fixed HybridKernel trace packet only as provenance for
trace-derived schedule selection:

`experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z`

The runner must not replace these inputs or select only favorable Phase 1 rows.

## Models

- Primary: `ibm-granite/granite-4.0-h-tiny`
- Same-family: `ibm-granite/granite-4.0-h-small`
- Cross-family: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`

If a model cannot be loaded for infrastructure reasons, the gate returns
`FAIL_INFRA_DMC_PHASE2`. Substitution requires a fresh preregistration before
any serving rows are inspected.

## Promptset

- Source: AIME-2025
- Count: 12 prompts for the primary model and 12 prompts for each control
- Selection: deterministic prompts indexed 0-11 from the canonical AIME-2025
  source already used by the swarm
- Prompt SHA-256: computed by the runner and recorded before inference

## Method Surface

The runner must implement a serving integration of the Phase 1 packed replay
idea. Acceptable implementations include:

- a vLLM-compatible decode replay hook that replaces a fixed schedule of
  repeated gating/routing/state-update micro-operations with a packed Triton or
  CUDA path;
- a serving-side scheduler that batches the same fixed micro-operation schedule
  into fewer GPU launches inside the decode path;
- a CUDA graph or persistent-kernel integration that reduces per-token launch
  count for the fixed DMC schedule while preserving generated tokens.

The baseline and DMC paths must run the same model, same prompt, same sampling
parameters, same seed, and same maximum decode length on the same GPU.

Forbidden:

- CPU-only or replay-only timing;
- using Phase 1 replay latency as a serving-speedup proxy;
- changing prompt slices, thresholds, seeds, or model set after observing
  serving results;
- dropping prompts post-hoc;
- claiming attention/SSM boundary fusion;
- substituting models without a fresh preregistration.

## Required Rows

The result packet must contain at least 36 paired serving rows:

- 12 primary rows;
- 12 same-family rows;
- 12 cross-family rows.

Each row must record:

- baseline and DMC output text;
- exact match of generated token IDs between baseline and DMC, or a documented
  deterministic-token mismatch that triggers kill;
- baseline and DMC end-to-end decode latency;
- baseline and DMC tokens/sec;
- p50, p95, and p99 per-token latency;
- baseline and DMC GPU launch count from Nsight Systems or an equivalent
  per-row GPU-launch audit;
- CUDA graph state, dtype, batch shape, decode length, seed, and sampling
  parameters;
- stdout/stderr logs and exact command line;
- environment snapshot, model provenance, prompt SHA, and artifact hashes.

## Metrics

For each row:

- `decode_latency_reduction_fraction =
  (baseline_decode_ms - dmc_decode_ms) / baseline_decode_ms`
- `tokens_per_second_gain_fraction =
  (dmc_tokens_per_second - baseline_tokens_per_second) /
  baseline_tokens_per_second`
- `launch_reduction_fraction =
  (baseline_launch_count - dmc_launch_count) / baseline_launch_count`
- `generated_token_match`

Aggregate by role with bootstrap 95% CI over prompts.

## PASS Decision

The checker returns `PASS_DMC_PHASE2_SERVING_GAIN` and exits 0 only if all
conditions hold:

1. Phase 1 source packet is exactly the fixed PASS packet above and its checker
   decision is `PASS_DMC_PHASE1_CONSOLIDATED_REPLAY`.
2. At least 12 valid paired serving rows per role are present.
3. Every row has `generated_token_match == true`.
4. Every row has `launch_reduction_fraction >= 0.10`.
5. Median decode latency reduction:
   - primary: at least 0.05;
   - same-family: at least 0.05;
   - cross-family: at least 0.03.
6. Bootstrap 95% CI lower bound for decode latency reduction is greater than
   0 for every role.
7. Median tokens/sec gain is positive for every role.
8. The checker can mechanically validate all required artifacts.

## KILL Decision

The checker returns `KILL_DMC_PHASE2_NO_SERVING_GAIN` and exits nonzero if the
packet is complete but any generated-token, launch-reduction, latency, or
tokens/sec threshold fails.

## INFRA Decision

The checker returns `FAIL_INFRA_DMC_PHASE2` and exits nonzero if artifacts are
missing, fixed inputs are replaced, model loading fails, prompt hashes are not
recorded, launch audits are absent, fewer rows than required exist, or the
checker cannot mechanically reproduce the decision.

## On PASS

Promote to paper development for Decode Microkernel Consolidation. The paper
must report Phase 1 as replay evidence and Phase 2 as serving evidence, with
the preregistered thresholds printed beside measured values.

## On KILL

Write a KILL manifest and diagnostic. Do not draft a negative-result paper.
