# 2026-05-26 Stage-1 GPU Roadmap Cap Pause

## Context

The human authorized a new Stage-1 GPU roadmap with hard cap
`gpu_hours_used <= 280` and priority order E1 through E5. The same
authorization requires pausing if cumulative GPU use approaches 280 before
Stage 1 completes.

## Ledger Check

`swarm/state.json` currently records:

- `gpu_hours_used = 278.643`
- hard cap from the new authorization: `280`
- remaining headroom: approximately `1.36` GPU hours

E1 is estimated at about 8 GPU hours. Starting E1 would violate the hard cap.

## Decision

Do not start any Stage-1 GPU experiment under the current cap. Continue only
non-GPU work that supports the roadmap:

- add the required scoop-hedge citations and wording to the paper;
- record a source memo in `references/`;
- prepare preregistration material if useful, but do not launch GPU work.

## Human Input Needed

To run Stage 1, the human needs to raise the hard cap or explicitly choose a
smaller no-GPU-only scope. The minimum useful cap increase for E1 alone is
roughly 8 GPU hours plus margin; completing all Stage-1 experiments would
require substantially more.
