# KILLED: SinkAware Systems Framing

## What Was Tried

The approximate SinkAware branch tested whether rank-2 sink-logit prediction
could reduce attention work while preserving output quality.

## Why It Died

The approximation improves quality metrics versus position-only, but the
systems wedge is too narrow. A few sink tokens are a tiny fraction of long
context QK work, so even a good approximation cannot plausibly create a
reviewer-clean wall-clock speedup at long context.

## Salvage Value

The rank-2 probes remain useful as evidence that sink logits have a
query-dependent component. That insight motivates SinkKV: protect sink K/V
precision instead of approximating away sink logits.
