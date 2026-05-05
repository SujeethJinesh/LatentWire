# ThoughtFlow-FP8 COLM Workshop Shell

Status date: 2026-05-05

## Current Policy Status

**Mixed/weakened.** The current anchor/phase-protected retention family is not
ready to support a positive-method claim. It preserved synthetic phase markers
and matched the strongest hidden-saliency proxy on marker recall. A bounded
successor, `thoughtflow_recent`, improved the retained-context NLL proxy by
combining phase/anchor protection with a small recent-token reserve, but it
still loses to the R-KV-like retained-prefix proxy.

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
| `phase2/hidden_saliency_retention_probe.md` | Distilgpt2 attention-received saliency: ThoughtFlow beats pure saliency on phase recall but ties LongFlow-like recall. | Mixed. |
| `phase2/perplexity_impact_proxy.md` | Distilgpt2 retained-context NLL: ThoughtFlow-recent beats old ThoughtFlow, LongFlow-like, and ThinKV-like, but loses to R-KV-like. | Weakened but more diagnostic. |

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

1. Implement a real sparse-KV or cache-dropping validation on one small
   same-family pair and one strict cross-family pair.
2. Score paired continuation NLL or task accuracy under full KV, matched
   sink+recent, LongFlow-like, R-KV-like, ThinKV-like, and ThoughtFlow-successor
   policies.
3. Report paired uncertainty, per-span keep telemetry, recurrence misses, and
   FP8 round-trip error separately.
4. Promote only if the successor policy beats the strongest proxy on quality at
   matched bytes while preserving anchor/fair-span telemetry.

Until that gate clears, the current policy should remain a negative/mixed
experiment rather than a paper claim.
