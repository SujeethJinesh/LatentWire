# Source-Private ARC-Challenge Seed-Stability Gate, 2026-05-01

## Status

- code: `scripts/build_source_private_arc_challenge_seed_stability.py`
- validation artifact:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_validation/`
- test artifact:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_test/`
- test: `tests/test_build_source_private_arc_challenge_seed_stability.py`
- references: `references/569_arc_challenge_seed_stability_refs_20260501.md`

## Method

This gate reuses the answer-key-forbidden source-choice cache from the ARC
fixed-packet Qwen2.5-0.5B source run. It then recomputes only the fixed `12B`
packet projection and random-control seeds for `47/53/59/61/67`.

The point is to separate "the ARC row worked because seed 47 was lucky" from
"the compact packet interface is stable enough to be paper evidence."

## Results

| Split | Seeds | Pass | Matched mean/min/max | Target | Same-byte text | Min lift vs target | Min CI95 low |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 5 | 5/5 | 0.384 / 0.381 / 0.385 | 0.244 | 0.348 | 0.137 | 0.065 |
| test | 5 | 5/5 | 0.344 / 0.341 / 0.346 | 0.265 | 0.311 | 0.076 | 0.038 |

On test, every seed beats target-only, the best destructive control, and the
same-byte structured-text comparator. Candidate derangement remains below
target-only plus the predeclared tolerance.

## Interpretation

This closes the random-projection luck objection for the public ARC row. It
does not close the remaining source-model caveat: the source scorer is still a
local Qwen choice log-likelihood bridge rather than a native cross-model latent
endpoint.

## Next Gate

The next ICLR gate should be either:

- a second public benchmark using the same source-cache and fixed-packet
  protocol, or
- a true source/target endpoint variant, followed by native NVIDIA/vLLM
  systems rows.
