# SinkAware Phase 2 Decomposition Decision

## Result

Status: **KILL as exact static sink-prior kernel; pivot only if approximate sink
priors become the goal.**

The exact fixed sink-token contribution cannot be reused without recomputing
query-dependent `QK_sink`. Fixed sink keys and values are constant, but the
softmax numerator and denominator use `exp(q @ k_sink)` for each query. A
query-independent sink logit prior therefore fails exact attention except in
degenerate cases where every query has the same dot product with the sink keys.

## Local Test

`phase2/tests/test_sink_static_prior_counterexample.py` constructs two queries
with the same fixed sink key/value and shows that a static sink-logit prior
deviates from exact attention by more than 1.0 in scalar output.

## Decision Boundary

What remains possible:

- a small fused path that computes `QK_sink` separately and cheaply;
- an approximate or learned sink prior with bounded quality loss;
- a low-rank sink prior.

What is ruled out:

- exact reuse of fixed sink-token contribution while skipping per-query
  `QK_sink` computation.
