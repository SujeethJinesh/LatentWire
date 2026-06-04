# GPU Handoff Plan

No foreground GPU job is authorized from this Mac continuation.

- PROVISIONAL_PROMOTE_TO_GPU rows: `0`
- Foreground confirmation queue: empty unless a future row is explicitly promoted.
- LatentWire remains Mac-complete for this pass.

## Allowed Backfill Only

- C_A1 ParoQuant parity: replay exact cached gate packets with native W4A16 forwards; est 2-4 GPU hours; no promotion allowed.
- C_F hazard controls: backfill paired static/random controls before any native forward confirmation; est 2-6 GPU hours; no promotion allowed.
- C_D1 OSC/DecDEC: only materialize a preregistered replacement shard for the current underpowered negative row; est 2-4 GPU hours; no promotion allowed.
- CE13 warmup policy: no parseable cache exists; spend 0 GPU hours now and only consider 2-4 GPU hours after a tiny dev/gate cache exists.

## Hard Stop

Do not run foreground GPU confirmation until a future dashboard contains at least one `PROVISIONAL_PROMOTE_TO_GPU` row or a Channel-Set row with positive offline gate headroom, matched-budget baselines, and control collapse.
