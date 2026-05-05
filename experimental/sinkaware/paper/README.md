# SinkAware COLM-Style Workshop Shell

This directory is scoped to the SinkAware side branch. It is a workshop-paper
shell, not a claim that the method is ready for a main-conference submission.

## Current Position

SinkAware is currently **not ICLR-ready**. The exact static prior is killed:
fixed early-token K/V contributions cannot be reused exactly without preserving
query-dependent sink logits. The remaining live idea is narrower:
an approximate per-head low-rank sink-logit prior that may reduce fixed-sink
score work while preserving attention quality.

Estimated distance to ICLR readiness: high. The branch still needs a GPU
implementation gate, larger frozen slices, seed repeats, strict same-family vs
cross-family separation, and competitor baselines before it can support a
positive-method paper.

## Approximate Low-Rank Sink Prior

The live method hypothesis is:

1. Keep non-sink attention scores exact.
2. Replace only fixed early-token sink logits with a cheap per-head predictor.
3. Use low-rank query features plus lightweight position information.
4. Accept the method only if softmax/output drift stays small and GPU cost is
   lower than exact sink-token QK.

This is not learned denominator-only sink support, and it should not be framed
as a generic sink-aware attention kernel. Existing systems already cover learned
sink terms in the denominator.

## Exact Static Prior Killed

The exact static-prior branch is ruled out by the Phase 2 decomposition gate.
Sink logits depend on the current query, so a fixed precomputed sink output term
cannot replace `QK_sink` exactly under standard softmax attention.

The killed claim:

> Fixed BOS/sink K/V tokens can be skipped exactly by adding a static
> precomputed output contribution.

The surviving claim is only approximate and must be evaluated as an error/cost
tradeoff.

## Current Evidence

- Source audit: no immediate fixed-token precomputed-output kill in audited
  kernels, but learned sink support in FlashInfer, FlashMLA, and GPT-OSS creates
  high novelty risk for broad claims.
- Synthetic probe: static prediction fails, while low-rank query features
  recover favorable low-rank/clustered synthetic sink logits.
- Real distilgpt2 QK-logit probe: hidden+position features predict mean sink
  logits better than position-only structure.
- Cost model: rank-2 is the only currently plausible low-rank setting below
  exact four-sink QK multiply-add cost.
- Softmax/output probe: rank-2 improves held-out distilgpt2 output relative-L2
  over position-only (`0.134` versus `0.173`) while keeping non-sink scores
  exact. This promotes the branch only to a GPU gate; it is not end-to-end
  quality evidence.

## Limitations

- distilgpt2 is a small same-family diagnostic, not a benchmark result.
- Saved text traces are convenience traces, not a frozen benchmark slice.
- Current evidence is CPU/Mac-local and does not measure kernel latency.
- The approximation uses exact non-sink scores, so it isolates sink-logit drift
  but does not prove end-to-end serving speedups.
- No cross-family falsification pair has passed.
- No paired uncertainty, seed repeats, or long-context competitor comparisons
  are available yet.

## Next GPU Gate

The Mac-local softmax/output probe shows bounded rank-2 drift on distilgpt2
traces, so the next gate should be a small GPU implementation or fused-kernel
prototype that measures:

1. exact attention baseline,
2. exact fixed-sink decomposition that still computes `QK_sink`,
3. rank-2 approximate sink-logit path,
4. output drift against exact attention,
5. latency/throughput under matched sequence lengths and sink-token counts.

Promotion criterion: rank-2 must preserve attention output quality while
showing a real latency or memory-layout advantage over exact sink QK. If it
does not, SinkAware should remain a negative systems note rather than a
positive-method branch.
