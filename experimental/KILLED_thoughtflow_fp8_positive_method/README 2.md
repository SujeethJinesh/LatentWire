# KILLED: ThoughtFlow-FP8 Positive Method

## What Was Tried

ThoughtFlow-FP8 tested anchor/recent/phase/math sparse-cache retention and
three preregistered successor signals: `rdu_topk`, `psi_topk`, and `vwac_topk`.

## Why It Died

The original policy family failed strict sparse-cache gates. `rdu_topk` cleared
one frozen gate but failed independent reproduction; `psi_topk` and
`vwac_topk` failed fresh-surface checks. No current branch supports a positive
training-free sparse-KV method claim.

## Salvage Value

The branch is still valuable as a falsification methodology paper: it documents
pre-registered signals, fresh-surface failures, oracle/headroom diagnostics,
and failure regimes for reasoning-trace sparse-cache retention.
