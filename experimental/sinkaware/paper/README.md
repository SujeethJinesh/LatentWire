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
implementation gate, broader frozen benchmark slices, strict same-family vs
cross-family controls, and competitor baselines before it can support a
positive-method paper. The latest per-head readout weakens the claim: aggregate
rank-2 output drift improves over position-only and survives randomized
token-split repeats, a small length/sink sweep, trace-level frozen splits, and
a repeated OPT-family length diagnostic. The larger 48-trace downstream control
repeat now also favors rank-2 over position-only at lengths 64 and 96, but this
is still Mac-local and not benchmark evidence.

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
  over position-only (`0.141` versus `0.170`) while keeping non-sink scores
  exact. The new layer-head paired readout is weaker (`+0.0297 +/- 0.0378`
  output rel-L2 improvement; 20/72 head wins), so this promotes the branch only
  to a correctness/repeatability gate; it is not end-to-end quality evidence.
- Head-selective gate: a simple validation rule selected 19/72 rank-2 heads but
  failed held-out (`0.204` output rel-L2 versus `0.172` for position-only and
  `0.142` for all-rank2), so this rescue is ruled out for now.
- Split/seed stability gate: all-head rank-2 beat position-only on three
  randomized token-level splits (`+0.0368 +/- 0.0006` output rel-L2
  improvement), but the layer-head output win rate stayed low
  (`0.282 +/- 0.024`).
- Length/sink sweep: all-head rank-2 stayed positive for
  `max_length={64,96}` and `sink_tokens={2,4}` across three split seeds per
  configuration (`+0.0366 +/- 0.0024` output rel-L2 improvement; minimum config
  `+0.0342`), but layer-head output win rate remained low (`0.286 +/- 0.010`).
- Trace-level frozen split repeat: all-head rank-2 stayed positive on three
  whole-trace held-out splits over 48 traces (`+0.0379 +/- 0.0014` output
  rel-L2 improvement; minimum split `+0.0367`), but layer-head output win rate
  remained low (`0.278 +/- 0.016`).
- Held-out/model-family repeat: separately fit per-model rank-2 predictors
  stayed positive on 48 traces and split seeds `0,1,2`; distilgpt2 improved by
  `+0.0306 +/- 0.0023` output rel-L2 and facebook/opt-125m improved by
  `+0.0788 +/- 0.0069`. This is not cross-model predictor transfer.
- Cross-family length stability: the same 48-trace GPT2/OPT-family gate stayed
  positive at lengths 64 and 96; aggregate model/length output rel-L2
  improvement was `+0.0535 +/- 0.0262`, with minimum row `+0.0301` and
  layer-head output win rate `0.982 +/- 0.008`. This is still attention-output
  drift evidence only.
- Downstream quality/control smoke: on distilgpt2 and facebook/opt-125m, 24
  traces, and split seeds `0,1,2`, exact replacement is a no-op and rank-2 is
  closer than position-only in causal-LM loss drift and KL. Aggregate absolute
  loss-delta improvement is `+0.0809 +/- 0.0815`; minimum model improvement is
  `+0.0393`. This is a small Mac-local control diagnostic, not benchmark or
  speed evidence.
- Downstream length/sink sweep: lengths `64/96` and sink counts `2/4` all stay
  positive with minimum model loss improvement at least `+0.0272`.
- Larger downstream repeats: 48 traces, sink counts `2/4`, lengths `64/96`,
  and split seeds `0,1,2` stay positive on both model rows. Exact replacement
  remains a no-op; rank-2 beats position-only by loss drift and KL. Minimum
  model loss improvement is `+0.0263`, and top-1 disagreement remains
  non-negligible, so this is still a quality-control diagnostic.
- Downstream rank frontier: on 48 traces at length `96` and sink count `4`,
  ranks `1/2/4/8` monotonically reduce downstream drift, but rank4/rank8 lose
  the multiply-add wedge against exact four-sink QK. Rank2 remains the live
  compromise.

## Limitations

- distilgpt2 is a small same-family diagnostic, not a benchmark result.
- Saved text traces are convenience traces, not a frozen benchmark slice.
- Current evidence is CPU/Mac-local and does not measure kernel latency.
- The approximation uses exact non-sink scores, so it isolates sink-logit drift
  but does not prove end-to-end serving speedups.
- Per-head gains are concentrated; a reviewer can reasonably ask whether a
  rank-2 path should be stabilized or killed for unstable heads.
- Cross-family length stability has passed only as a Mac-local, separately fit
  per-model diagnostic.
- No long-context competitor comparison or native GPU timing is available yet.

## Next GPU Gate

The Mac-local softmax/output probe now shows bounded aggregate rank-2 drift
under randomized token splits, a small length/sink sweep, trace-level frozen
splits, GPT2/OPT-family length stability, Triton interpreter correctness, and
48-trace downstream quality/control repeats. The next gate is now a native GPU
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
