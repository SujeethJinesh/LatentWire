# ThoughtFlow-FP8 COLM Workshop Shell

Status date: 2026-05-05

## Current Policy Status

**Mixed/weakened.** The current anchor/phase-protected retention family is not
ready to support a positive-method claim. It preserved synthetic phase markers,
matched the local LongFlow-like importance proxy, and beat the strongest real
hidden/KV saliency proxy on phase recall. However, its math-state margin over
that real-saliency proxy is uncertain, and the retained-context NLL evidence is
only a held-out tie against R-KV-like rather than a robust win.

This shell is a scoped workshop-paper scaffold, not a submission draft. The
current evidence is useful for deciding the next method gate, but it is not yet
strong enough for an ICLR/COLM positive-method paper.

## Candidate Story If Revived

Most deployers need training-free KV compression for already released reasoning
models, not only new architectures trained around compressed attention. A
revived ThoughtFlow-FP8 story would be:

> Explicit anchor/fair-span/phase-transition retention plus FP8 byte budgeting
> can reduce KV footprint for existing reasoning models while avoiding the
> delayed-utility and span-bias failures that pure current-query or redundancy
> selectors miss.

This story only survives if the policy beats strong matched-budget baselines on
quality or perplexity, not just on protected-token recall.

## Current Evidence

| Artifact | Evidence | Decision |
|---|---|---|
| `phase2/phase_eviction_analysis.md` | Synthetic traces: ThoughtFlow preserves phase and anchor labels at matched keep rate. | Alive only as synthetic policy evidence. |
| `phase2/real_trace_retention_sweep.md` | Saved real generated traces: ThoughtFlow never beats the strongest proxy across keep fractions 0.10-0.35. | Weakened. |
| `phase2/hidden_saliency_retention_probe.md` | Distilgpt2 attention, final-hidden, key, value, and KV-norm telemetry: ThoughtFlow beats `value_norm_topk` on phase recall by +0.508 paired mean, but math-state CI crosses zero and LongFlow-like ties it. | Mixed; diagnostic, not a revival. |
| `phase2/perplexity_impact_proxy.md` | Distilgpt2 retained-context NLL: ThoughtFlow-saliency-recent beats old ThoughtFlow, LongFlow-like, and ThinKV-like, but loses to R-KV-like. | Weakened but more diagnostic. |
| `phase2/policy_sweep.md` | Train-selected ThoughtFlow-family policy ties R-KV-like on 12 held-out traces, 3.480 vs 3.482 NLL. | Mixed tie-range result. |

The most recent proxy scored 24 saved traces at 0.20 retained-prefix budget:

| Policy | NLL | Delta NLL vs full | Mean PPL |
|---|---:|---:|---:|
| Full context reference | 2.101 | 0.000 | 9.7 |
| R-KV-like | 3.419 | 1.319 | 38.7 |
| ThinKV-like | 3.583 | 1.482 | 44.6 |
| LongFlow-like | 3.961 | 1.861 | 74.2 |
| ThoughtFlow | 3.961 | 1.861 | 74.2 |
| ThoughtFlow-recent | 3.562 | 1.461 | 44.2 |

Interpretation: a simple recent-token reserve repairs part of the failure mode,
which means pure phase-marker protection was too far from the continuation
objective. The branch is still not revived because the R-KV-like retained-prefix
proxy remains stronger at the same keep rate.

The real hidden/KV telemetry sharpens the diagnosis but does not change the
decision:

| Probe row | Keep rate | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|---:|
| ThoughtFlow | 0.207 | 1.000 | 1.000 | 0.918 |
| LongFlow-like | 0.207 | 1.000 | 1.000 | 0.918 |
| ThinKV-like | 0.207 | 1.000 | 0.948 | 0.848 |
| best real hidden/KV proxy: value-norm top-k | 0.207 | 0.000 | 0.492 | 0.845 |
| train-selected ThoughtFlow sweep policy | 0.207 | 1.000 | 0.430 | 0.956 |

Paired against `value_norm_topk`, the best ThoughtFlow-family row has phase
recall delta +0.508 with 95% CI [+0.436, +0.579], but math-state recall delta
+0.073 with 95% CI [-0.078, +0.223]. This rules out using phase-recall telemetry
alone as a reviewer-facing positive claim.

## What Would Revive The Branch

The branch should be revived only if a bounded next policy clears at least one
of these gates:

1. **Quality/perplexity gate:** ThoughtFlow or a successor policy beats
   LongFlow-like, R-KV-like, and ThinKV-like matched-budget proxies by at least
   0.03 mean continuation NLL on saved traces, with paired per-trace deltas.
2. **Hidden/KV saliency gate:** The policy beats the strongest hidden/KV
   retention proxy on phase/control and math-state recall, not merely pure
   attention-received saliency.
3. **Bias-control gate:** The policy shows lower eviction bias on anchors,
   problem spans, phase transitions, and recurring math state at the same byte
   budget without becoming a sink+recent policy in disguise.

The highest-value method branch is no longer raw phase-marker protection. A
revival attempt should combine anchor/fair-span protection with a utility signal
closer to future continuation loss, recurrence, or hidden-state contribution.

Saturated: synthetic marker-retention and text-prefix-only policy tuning.
Still alive: cache-dropping quality validation for a train-fixed successor
policy, but only as a falsification gate. Promoted: real KV/hidden telemetry as
diagnostics to explain failure modes and eviction bias.

## Limitations

- The retained-context NLL proxy is not sparse-KV decoding; it rebuilds a shorter
  text prefix and scores the same continuation.
- Distilgpt2 is a small model and not a reasoning model; this is a Mac-local
  falsification proxy, not a benchmark result.
- Current saved traces are small and not seed-repeated.
- There is no real FP8 numerical drift measurement in this gate.
- There is no latency, throughput, or serving-system result.
- Current baselines are local proxies, not faithful implementations of LongFlow,
  R-KV/R-KVHash, ThinKV, PM-KVQ, or RaaS.

## Next GPU/KV Gate

Do not move to a broad GPU benchmark yet. The next exact gate is:

1. Implement real cache-dropping or sparse-KV validation on one small
   same-family pair and one strict cross-family falsification pair.
2. Score paired continuation NLL or task accuracy under full KV, matched
   sink+recent, LongFlow-like, R-KV-like, ThinKV-like, and ThoughtFlow-successor
   policies.
3. Report paired uncertainty, per-span keep telemetry, recurrence misses, and
   FP8 round-trip error separately.
4. Promote only if the successor policy beats the strongest proxy on quality at
   matched bytes while preserving anchor/fair-span telemetry.

Until that gate clears, the current policy should remain a negative/mixed
experiment rather than a paper claim.
