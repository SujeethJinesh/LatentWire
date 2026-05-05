# Experimental Project Workshop Review

- date: 2026-05-05
- scope: HybridKernel, SinkAware, ThoughtFlow-FP8 side-paper readiness

## Ranking

| Rank | Project | Workshop acceptance likelihood | Distance to completion | Current decision |
|---:|---|---|---|---|
| 1 | SinkAware | low-medium | medium | all-rank2 approximate branch weakly alive; simple validation head selection weakened |
| 2 | ThoughtFlow-FP8 | low-medium | medium-long | mixed after hidden/KV telemetry and CPU sparse-cache quality probe |
| 3 | HybridKernel | low-medium | long without GPU | weakly alive as a server-side profiler-gate systems paper |

## SinkAware

Strongest current claim: exact static sink reuse is invalid, but approximate
per-head low-rank fixed-sink logit prediction is measurable and bounded on
Mac-local distilgpt2 traces.

New update:

- Added the Phase 3 approximate-attention CPU reference.
- Tests verify exact sink logits reproduce exact attention and approximate
  sink logits affect only the sink side of the score vector.
- Added the Phase 4 Triton-interpreter scaffold for the same approximate
  operator. It is ready to run once `triton` is installed in the repo-local
  venv; current local tests skip because Triton is unavailable.
- Added per-head paired evidence: aggregate rank-2 improves output rel-L2
  `0.1408` vs position-only `0.1700`, but only `20/72` layer-heads improve,
  so the branch is weakly alive rather than cleanly positive.
- Added the head-selective validation/test gate. It selected 19/72 rank-2
  heads on validation but failed held-out with output rel-L2 `0.2035`, worse
  than position-only `0.1724` and all-rank2 `0.1419`.

Needed for completion:

1. Run split/seed repeats for the all-rank2 branch or design a better
   stability mechanism; simple validation head selection is weakened.
2. Run Triton interpreter correctness against the Phase 3 reference after
   installing `triton` in `venv_arm64`.
3. Native GPU comparison of exact attention, exact decomposition, rank-2, rank-4,
   and rank-8 approximate paths.
4. At least one larger/newer model family beyond distilgpt2.
5. Quality drift measured at output, token logprob, and short continuation
   levels.

## ThoughtFlow-FP8

Strongest current claim: phase-aware retention by itself is weak, but a
saliency+recent successor nearly matches the strongest retained-prefix proxy at
the same keep budget, and a small train/held-out sweep reaches a tie-range
result.

New update:

- Added `thoughtflow_saliency_recent`.
- Retained-context NLL improves from ThoughtFlow-recent 3.562 to 3.434.
- Fixed strict budget matching for ThoughtFlow-family policies.
- Added `phase2/policy_sweep.py`: a 12-trace train / 12-trace held-out sweep.
- Held-out best ThoughtFlow-family policy reaches NLL 3.480 versus R-KV-like
  3.482, margin +0.001. This is a tie, not a robust win.
- Added real hidden/KV saliency telemetry with distilgpt2 attention, final
  hidden norm, key norm, value norm, and combined KV norm retention. ThoughtFlow
  improves phase recall over value-norm top-k by `+0.508` with positive CI, but
  math-state recall improves only `+0.073` with CI crossing zero. The branch is
  still mixed, not revived.
- Added CPU sparse-cache quality: process full prefix, prune `past_key_values`,
  and score continuation from the sparse cache. Best ThoughtFlow-family policy
  NLL is `3.432` versus R-KV-like `3.435`, paired delta `-0.003` with 95% CI
  `[-0.037,+0.034]`. This is stronger than retained-text proxy evidence but
  still a tie-range result.

Needed for completion:

1. A new successor policy with a utility signal closer to future loss than
   marker/phase protection alone.
2. More held-out traces and at least one newer model.
3. Matched-budget continuation NLL win over R-KV-like, ThinKV-like, and
   LongFlow-like before GPU work.

## HybridKernel

Strongest current claim: the hypothesis is now measurable and falsifiable, but
not yet positive.

New update:

- Added a profiler-summary parser and promote/kill rule.
- Extended the NVIDIA/vLLM profiler runbook with exact parser input fields and
  the command for generating the paper-facing gate.
- Added native profiler artifact verification so GPU evidence must include
  metadata, server-side profile scope, logs, Nsight artifacts, readout rows, and
  at least 3 valid repeated metric rows before paper-facing claims. Client-only
  profiles are now rejected.
- Current output is pending native profiler data, so no speed claim is allowed.

Needed for completion:

1. Native vLLM/Nsight run on Granite 4.0 H or a close hybrid SSM target.
2. Three or more repeated runs clearing the 3% recoverable-gain upper-bound gate.
3. Matched non-boundary control to rule out warmup, graph capture, batching, and
   unrelated kernel gaps.
4. Prototype only if the profiler gate clears.

## Current Recommendation

Push SinkAware only if all-rank2 survives repeatability or a better stability
mechanism. Keep ThoughtFlow alive only for a new loss-oriented policy, since
the actual CPU sparse-cache gate tied R-KV-like. Do not spend more Mac
implementation time on HybridKernel beyond profiler artifact semantics until
native server-side profiler traces exist.
