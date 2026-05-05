# LongFlow Failure Hypothesis

Date: 2026-05-05

## Status

Concrete enough to proceed to a focused Phase 1/2 pivot, but not enough to
proceed to kernel implementation. The hypothesis is based on the LongFlow paper,
official OpenReview reviews, and the Pitfalls paper. It is not based on a local
LongFlow reproduction.

## What Sank LongFlow

The failure was not a single bug. It was a reviewer-visible mismatch between a
strong systems story and weaker quality/theory/evaluation evidence.

### Failure 1: Not Pareto-Dominant

Reviewers explicitly viewed LongFlow as a throughput/memory trade-off with worse
accuracy than R-KV. They asked for Pareto sweeps showing whether LongFlow can
simultaneously improve throughput, memory, and accuracy, rather than only being
faster at lower quality.

ThoughtFlow implication: every result must be plotted as a memory/throughput/
accuracy frontier against LongFlow, R-KV/R-KVHash, ThinKV, and full KV.

### Failure 2: No Real Production End-to-End Speedup

Reviewers asked for real inference acceleration in vLLM or SGLang. Kernel-level
speedups alone were not enough.

ThoughtFlow implication: Phase 5+ must include an end-to-end serving path or must
avoid claiming a systems win. Mac-only Phase 0/1 cannot produce this artifact.

### Failure 3: Fragile Importance Approximation

LongFlow simplifies the global objective of preserving all future generations
into preserving the immediate next-step attention output. It approximates future
queries using the current query. Reviewers considered this under-justified and
asked for broader statistical validation and ablations.

ThoughtFlow implication: phase-aware eviction should be justified by measured
reasoning-token lifecycle/recurrence, not by a single-step query proxy.

### Failure 4: Numerical Stability and Outlier Channels

One review specifically flagged outlier key channels and the omission of running
maximum qk scores in the FlashAttention-style kernel as possible causes of
attention precision degradation.

ThoughtFlow implication: FP8 cannot be added casually. The design needs a
numerical audit: qk max/range telemetry, outlier-channel preservation, FP8
round-trip error, and downstream quality under anchor/fair spans.

### Failure 5: Eviction Bias Against Low-Current-Attention Critical Spans

Pitfalls shows that KV eviction can disproportionately remove some instruction
spans, causing instruction ignoring and prompt leakage. LongFlow's current-query
importance is especially suspect for spans that are globally critical but not
currently attended: system instructions, problem statements, phase transitions,
and early reasoning lemmas.

ThoughtFlow implication: this is the strongest project wedge. Anchor protection
should be formalized as "fair-span and phase-transition retention" and audited
with keep-rate telemetry by span type.

## Specific Hypothesis To Test

LongFlow loses quality because current-query single-step importance plus
aggressive fused eviction under-protects tokens whose utility is delayed,
recurring, or instruction-like. This creates:

- lower quality than R-KV on reasoning tasks;
- hidden eviction bias against anchors/problem spans/phase transitions;
- numerical sensitivity when approximation and compression are combined.

ThoughtFlow-FP8 should only proceed if Phase 2 shows that an anchor/fair-span/
phase-transition policy protects tokens that LongFlow-like eviction would drop,
without merely becoming full-KV in disguise.

## Quick Test Plan

Mac-only possible before GPU:

1. Use cached or synthetic reasoning traces with token spans:
   system/problem/prefill, plan, derivation, verification, final.
2. Simulate LongFlow-like current-attention or recency importance if true
   attention logs are unavailable.
3. Compare keep-rate by span type under:
   uniform, sink+recent, LongFlow-like, RaaS-like prefill protection,
   anchor+phase protection.
4. Flag a proceed condition only if anchor+phase changes protected-token sets in
   a way that maps to Pitfalls/LongFlow review risks.

## Kill Trigger

Kill the original implementation plan if Phase 2 cannot show a concrete token
class that ThoughtFlow protects and LongFlow/ThinKV/R-KV-style baselines do not.
Without that, the project is an incremental crowded kernel variant.
