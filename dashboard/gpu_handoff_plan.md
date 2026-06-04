# GPU Handoff Plan

No foreground GPU job is authorized from this triage.

- PROVISIONAL_PROMOTE_TO_GPU rows: `0`
- Foreground confirmation queue: empty
- Channel-Set state: C_A1 has limited positive offline gate medians, C_F is mixed and control-contaminated, C_D1 is negative, CE13 has no parseable cache, and C_A2 is tests-only.
- LatentWire state: keep on Mac. WZ/source-copy cached family is killed on this surface; L-B1 needs fresh risk/confidence rows before it is a live lead.

## Allowed GPU Backfill Only

Backfill may prepare Channel-Set parity/cache evidence, not confirmation claims:
- ParoQuant parity replay for the exact C_A1 cached gate packets.
- OSC/DecDEC cache shard only to replace the current underpowered negative row.
- Hazard/static-control shard for C_F only if paired denominator and random-control collapse are defined first.

## Hard Stop

Do not run foreground GPU confirmation until a future dashboard contains at least one `PROVISIONAL_PROMOTE_TO_GPU` row or a Channel-Set row with positive offline gate headroom, matched-budget baselines, and control collapse.
