# 2026-05-28T02:05Z M-FISH feasibility decision

## Finding

M-FISH requires per-channel diagonal Fisher scores at the same activation
surfaces used by the long-decode W4A16 endpoint scorer. Existing M11b and M-PRED
packets cache activation magnitudes only. They do not preserve autograd graphs or
loss gradients.

A faithful implementation would need a new gradient-capture runner that either:

1. retains a full 10K-token decode computation graph while collecting gradients
   at every protected layer, or
2. reconstructs the 10K-token context layer-by-layer under teacher forcing and
   accumulates gradients per channel.

Both are outside the existing endpoint scoring path and exceed the queued
"M-FISH backward pass infrastructure >2h" override threshold.

## Decision

Defer M-FISH as `DEFERRED_INFRA_MFISH_BACKWARD_GRAPH` rather than substitute a
proxy from cached activation magnitudes. A proxy would not test the Fisher method
that was proposed.

## Consequence

M-PRED is killed on Granite, DeepSeek, and Falcon. M-FISH is not killed; it is an
infrastructure-deferred branch requiring human authorization for a new gradient
runner. Because Tier 1.5 methods were gated on both M-PRED and M-FISH killing,
Tier 1.5 is not automatically triggered by the current evidence.

## Next safe work

Continue non-GPU integration work: update the experiment ledger, paper framing,
and final queue state to reflect M-PRED uniform KILL and M-FISH deferral.
