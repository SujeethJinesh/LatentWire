# KILLED: SinkAware Exact Static Prior

## What Was Tried

The original SinkAware branch tested whether fixed sink-token attention logits
could be reused as a static prior and avoid recomputing `QK_sink`.

## Why It Died

The exact branch is mathematically invalid: sink logits remain query-dependent.
A counterexample in the SinkAware gate shows exact fixed-sink reuse cannot skip
query-dependent `QK_sink` without changing attention outputs.

## Salvage Value

The work produced reusable sink-mass probes, fixed-sink decomposition tests,
and the approximate rank-2 predictor machinery. Those are now background for
SinkKV and attention-behavior diagnostics, not a systems speedup claim.
