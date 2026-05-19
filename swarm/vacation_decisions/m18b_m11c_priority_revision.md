# Vacation Decision: M18b and M11c Priority Revision

Date: 2026-05-19 UTC

## Situation

A re-read of the Phase 9 method record surfaced three underweighted findings:

1. M18 activation-plus-K coupling had the strongest signal-versus-random margin
   among the failed methods. Its proposed activation+K variant had median
   recovery around `-0.34`, while the random coupled control was around
   `-4.66`, a margin of roughly `4.3`.
2. M11b produced the first mechanical positive decision at top-5 budget, with
   top5 median recovery `0.4492840911245966`, while top1 and top10 were weaker.
   This non-monotonic pattern suggests a budget sweet spot rather than a simple
   monotone "more protection is always better" story.
3. Phase 4 and later reused packets show substantial trace-level
   heterogeneity, with many traces having no measurable BF16-vs-static gap.

## Decision

The next new method candidates after the currently queued gates are:

1. **M18b cross-tensor coupling at top-5 budget.** This combines M18's
   strongest signal source with M11b's empirically useful top-5 budget and
   EMA smoothing.
2. **M11c budget resolution sweep.** This tests whether M11b's top-5 result is
   a real budget peak by measuring top3 and top7 around the existing top1,
   top5, and top10 points.

M33 is demoted. It remains a conditional combination method only if M18b and
M11c both kill and GPU budget remains.

## Updated Queue After KL

After KL accumulation lands, the current order is:

1. ParoQuant Granite-Small baseline.
2. M11b Nemotron-3-Nano replication.
3. M18b cross-tensor top-5, if the preceding gates do not force a different
   paper path.
4. M11c budget resolution sweep.
5. Conditional M33 only if M18b and M11c kill.
6. Paper integration, committee review, synthesis.

If budget becomes tight, M18b stays above M11c because it tests the strongest
failed signal source. M11c is an explanatory curve experiment and can be
deferred more safely.

## Explicit Non-Authorizations

The following remain unauthorized for this sprint:

- M16 dwell-time filtering.
- M17 eviction cooldown.
- M11-slow with very small alpha.
- M27 layer-stratified protection.
- M34 vulnerability stratification.
- M35 scale-axis methods.
- M36 per-channel dynamics classification.

## Paper Findings To Preserve

The paper should integrate these findings regardless of later method outcomes:

- Phase 4 no-gap rate shows trace-level quantization vulnerability
  heterogeneity.
- M2 and M10 random controls show that discontinuous switching can be actively
  harmful.
- M10's scale-bin behavior suggests scale-axis interventions may be more
  tolerant of randomization than channel-axis interventions.
- Top-1 budget methods appear bounded well below full BF16-gap recovery,
  while M11b suggests budget is a binding constraint.
- M18's cross-tensor signal was strong relative to random despite failing at
  top-1 budget.
- Per-component dissection showed only marginal attention-versus-SSM drift
  differences.
- The stable core accounts for roughly half of the per-layer top-1% structure,
  motivating but not validating stable-core protection.

## What Would Invalidate This Decision

If ParoQuant strongly recovers the BF16 gap, reviewer-facing baseline work may
preempt new-method exploration. If M11b Nemotron replication kills cleanly,
M18b remains informative but the positive-method story weakens substantially.
If KL accumulation finds a strong compound-error dynamic, a time-axis method
such as M31 may become more urgent than M11c.
