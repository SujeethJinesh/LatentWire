# Experimental Project Workshop Review

- date: 2026-05-05
- scope: HybridKernel, SinkAware, ThoughtFlow-FP8 side-paper readiness

## Ranking

| Rank | Project | Workshop acceptance likelihood | Distance to completion | Current decision |
|---:|---|---|---|---|
| 1 | SinkAware | medium | medium | alive as approximate low-rank fixed-sink prediction |
| 2 | ThoughtFlow-FP8 | low-medium | medium-long | mixed after saliency+recent policy nearly ties R-KV-like proxy |
| 3 | HybridKernel | low-medium | long without GPU | weakly alive as a profiler-gate systems paper |

## SinkAware

Strongest current claim: exact static sink reuse is invalid, but approximate
per-head low-rank fixed-sink logit prediction is measurable and bounded on
Mac-local distilgpt2 traces.

New update:

- Added the Phase 3 approximate-attention CPU reference.
- Tests verify exact sink logits reproduce exact attention and approximate
  sink logits affect only the sink side of the score vector.

Needed for completion:

1. Triton interpreter implementation matching the Phase 3 reference.
2. Native GPU comparison of exact attention, exact decomposition, rank-2, rank-4,
   and rank-8 approximate paths.
3. At least one larger/newer model family beyond distilgpt2.
4. Quality drift measured at output, token logprob, and short continuation
   levels.

## ThoughtFlow-FP8

Strongest current claim: phase-aware retention by itself is weak, but a
saliency+recent successor nearly matches the strongest retained-prefix proxy at
the same keep budget.

New update:

- Added `thoughtflow_saliency_recent`.
- Retained-context NLL improves from ThoughtFlow-recent 3.562 to 3.434.
- R-KV-like remains slightly better at 3.419, so no positive method claim.

Needed for completion:

1. Held-out policy sweep over recent reserve, phase bonus, math-state bonus, and
   anchor budget.
2. Separate train/validation trace split to avoid overfitting the 24 traces.
3. A real hidden/KV saliency signal, not just text-token heuristics.
4. Matched-budget continuation NLL win over R-KV-like, ThinKV-like, and
   LongFlow-like before GPU work.

## HybridKernel

Strongest current claim: the hypothesis is now measurable and falsifiable, but
not yet positive.

New update:

- Added a profiler-summary parser and promote/kill rule.
- Current output is pending native profiler data, so no speed claim is allowed.

Needed for completion:

1. Native vLLM/Nsight run on Granite 4.0 H or a close hybrid SSM target.
2. Three or more repeated runs clearing the 3% recoverable-gain upper-bound gate.
3. Matched non-boundary control to rule out warmup, graph capture, batching, and
   unrelated kernel gaps.
4. Prototype only if the profiler gate clears.

## Current Recommendation

Push SinkAware first. Keep ThoughtFlow alive for one more Mac policy-search
gate because the new NLL gap is small. Do not spend more Mac implementation time
on HybridKernel until native profiler traces exist.
