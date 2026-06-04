# Morning Brief

Row-safe Stage-1 screen completed on parseable cached families. It consumed `183653` dev/gate rows/traces and `0` confirm rows. Coverage is in `dashboard/cache_parse_coverage.md`; raw evidence is in `results/stage1/`.

Status counts: `{'AMBIGUOUS': 100, 'CPU_SCREENED': 57, 'KILLED': 12, 'PARKED_NEEDS_GPU': 2}`.

The held-out answer is unfavorable for final claims: many caches are prior test/validation/full-eval artifacts or lack explicit confirm naming, so Mac screens can kill or rank branches, but final confirmation needs quarantined row-specific confirm handling or fresh data.

## Overnight V3 Patch Status

- C_A1 is `NEXT_GPU_BACKFILL` with `promotion_allowed=false`; the GPU packet must run paired ParoQuant-vs-tight-clip native rows on Granite, DeepSeek, and Falcon.
- L_SCORECOMP, L_B1, and L_Q1 are terminal killed score-packet branches and must not be queued.
- L_A2 is `INCONCLUSIVE_UNDERPOWERED_NEEDS_VERIFIER_CACHE`; the next cache must use at least 300 generated-solution prompts, include `verifier_score`, and screen only if the correct-candidate subset is large enough.
- `queues/gpu_foreground.yaml` remains empty.
