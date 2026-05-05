# SinkAware Workshop Paper Outline

## Title

SinkAware Attention as an Approximate Low-Rank Fixed-Sink Prior

## Abstract Placeholder

We study whether fixed early-token attention sinks can be exploited as a
systems prior without changing attention outputs materially. An exact static
prior is impossible under standard softmax attention because fixed sink-token
scores remain query-dependent. We therefore test a narrower approximate method:
per-head low-rank query predictors for fixed sink-token logits, with exact
non-sink scores preserved. Current evidence is diagnostic only and does not yet
support a benchmark-backed positive claim.

## Core Claim Boundary

Claim:

Per-head low-rank sink-logit prediction may approximate fixed early-token
attention contributions cheaply enough to justify a GPU kernel gate.

Non-claims:

- Exact static reuse of fixed sink K/V.
- A generic new attention-sink kernel class.
- End-to-end model quality preservation.
- Cross-family latent transfer.
- GPU speedup.

## Sections

1. Motivation: attention sinks and fixed early-token work.
2. Related systems: learned sink denominators, streaming sinks, sparse attention.
3. Exact decomposition and counterexample.
4. Approximate low-rank method.
5. Mac-local probes:
   - synthetic predictability,
   - real QK sink-logit predictability,
   - cost model,
   - per-head softmax/output error.
6. Limitations and threats to validity.
7. GPU gate and benchmark plan.

## Evidence Table Placeholder

| Gate | Status | Interpretation |
| --- | --- | --- |
| Source audit | Mixed | Broad novelty risky; fixed-token precompute not directly killed. |
| Exact static prior | Killed | Query-dependent sink logits block exact reuse. |
| Synthetic low-rank prior | Alive only as approximation | Useful for branch selection, not paper evidence. |
| Real QK-logit prediction | Alive | Hidden+position beats position-only on distilgpt2 traces. |
| Cost model | Alive only at rank-2 | Higher ranks are too expensive under current estimate. |
| Softmax/output error | Alive for GPU gate | Rank-2 reduces held-out output drift versus position-only on distilgpt2 traces; not an end-to-end quality result. |

## Reviewer-Risk Notes

- If the paper sounds like learned attention sinks, it is already occupied by
  existing systems.
- If the method only works on distilgpt2 traces, it is not a general method.
- If rank-2 output drift is large, the branch should be killed before GPU work.
- If rank-2 is accurate but slower than exact sink QK, the method is not useful.
