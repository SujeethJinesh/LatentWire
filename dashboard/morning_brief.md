# Morning Brief

Row-safe Stage-1 screen completed on parseable cached families. It consumed `183653` dev/gate rows/traces and `0` confirm rows. Coverage is in `dashboard/cache_parse_coverage.md`; raw evidence is in `results/stage1/`.

Status counts: `{'AMBIGUOUS': 100, 'CPU_SCREENED': 57, 'KILLED': 12, 'PARKED_NEEDS_GPU': 2}`.

The held-out answer is unfavorable for final claims: many caches are prior test/validation/full-eval artifacts or lack explicit confirm naming, so Mac screens can kill or rank branches, but final confirmation needs quarantined row-specific confirm handling or fresh data.
