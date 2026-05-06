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
   - per-head softmax/output error on 48 distilgpt2 traces,
   - layer-head paired uncertainty against position-only,
   - all-rank2 split/seed repeat,
   - bounded sequence-length/sink-token sweep,
   - trace-level frozen split repeat,
   - small held-out/cross-family falsification smoke.
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
| Softmax/output error | Weakly alive | On 48 distilgpt2 traces, rank-2 reduces aggregate held-out output rel-L2 drift to 0.141 versus 0.170 for position-only; not an end-to-end quality result. |
| Layer-head paired error | Fragile | Rank-2 output rel-L2 improvement is +0.0297 +/- 0.0378 over position-only across 72 layer-head cells, with 20/72 head wins. |
| Validation head selection | Weakened | Selecting 19/72 heads on validation gives held-out output rel-L2 0.204, worse than position-only 0.172 and all-rank2 0.142. |
| Split/seed all-rank2 repeat | Alive but weak | Across 3 randomized token splits, all-rank2 beats position-only by +0.0368 +/- 0.0006 output rel-L2, but layer-head win rate is only 0.282 +/- 0.024. |
| Length/sink all-rank2 sweep | Alive but bounded | Across max lengths 64/96, sink tokens 2/4, and 3 seeds per config, all configs remain positive; mean improvement is +0.0366 +/- 0.0024, min config +0.0342. |
| Trace-level frozen split repeat | Alive but bounded | Across 3 whole-trace held-out splits on 48 traces, all splits remain positive; mean improvement is +0.0379 +/- 0.0014, min split +0.0367, but head win rate is only 0.278 +/- 0.016. |
| Held-out/cross-family smoke | Alive but bounded | On a smallest Mac-feasible 12-trace, one-seed gate, per-model rank-2 predictors beat position-only on distilgpt2 by +0.0329 output rel-L2 and on facebook/opt-125m by +0.0709. This is not cross-model predictor transfer and not promotion evidence. |
| Triton interpreter readiness | Blocked locally | `TRITON_INTERPRET=1` readiness reports `triton` is not importable in `./venv_arm64`; no interpreter correctness pass yet. |

## Reviewer-Risk Notes

- If the paper sounds like learned attention sinks, it is already occupied by
  existing systems.
- If the method only works on distilgpt2 traces, it is not a general method.
- If rank-2 output drift is large, the branch should be killed before GPU work.
- If rank-2 only wins a small subset of heads, add a better stability gate or
  kill any per-head robustness claim before claiming a general method.
- If the OPT-family smoke does not survive larger trace slices and split seeds,
  keep the method Mac-local and do not claim cross-family robustness.
- If rank-2 is accurate but slower than exact sink QK, the method is not useful.
