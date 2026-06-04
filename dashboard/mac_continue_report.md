# Mac Continue Report

- terminal headline: `BOUNDED_NEGATIVE`
- LatentWire status: `MAC_DONE`
- mac_continue drained: `true`
- confirm rows scored: `0`
- fresh-data I_beyond bits: `2.231310`
- WZ gate delta vs best baseline: `{'n': 23, 'delta': -0.043478260869565216, 'ci95_low': -0.30434782608695654, 'ci95_high': 0.2608695652173913}`
- WZ delta_beyond_score: `{'n': 23, 'delta': 0.0, 'ci95_low': -0.2608695652173913, 'ci95_high': 0.21739130434782608}`
- WZ delta_beyond_label: `{'n': 23, 'delta': 0.043478260869565216, 'ci95_low': -0.13043478260869565, 'ci95_high': 0.2608695652173913}`
- L-B1 AURC delta vs source-index+confidence: `-0.222541`
- L-B1 paired delta vs source-index: `{'n': 23, 'delta': -0.08695652173913043, 'ci95_low': -0.30434782608695654, 'ci95_high': 0.13043478260869565}`
- L-B1 damage reduction: `2` / `2`
- L-B1 repair rate: `0.000000`
- L-A2 rerank slice: `{'status': 'MAC_DONE', 'rows': 3, 'generated': 3, 'parked': False, 'path': 'results/mac_continue/fresh_mmlu_pro/rerank_generation_rows.jsonl'}`

Channel-Set remains GPU-backfill only from prior triage: C_A1 has limited offline headroom, C_F is mixed/control-contaminated, CE13 has no parseable cache, C_D1 is killed, and C_A2 is tests-only.
