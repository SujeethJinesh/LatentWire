# GPU Handoff Plan

No foreground GPU job is authorized from the powered LatentWire oracle ladder.

- PROVISIONAL_PROMOTE_TO_GPU rows: `0`
- Foreground confirmation queue: empty unless a future row is explicitly promoted.
- LatentWire remains Mac-complete for this pass; the score-packet path is `not-deployable` on powered dev/gate screening.

## Allowed Backfill Only

- L-A2 generated-solution rerank cache: materialize at least 300 dev/gate prompts x16 candidates with verifier/source/target score surfaces; require `verifier_score` and at least about 80 prompts with one correct candidate before screening; est 0-2 GPU hours; no promotion allowed.
- C_A1 ParoQuant parity and tail/CVaR grid: replay exact cached gate packets with paired native ParoQuant baseline versus tight-clip C_A1 rows on Granite, DeepSeek, and Falcon; DeepSeek/Falcon denominator-only rows are invalid sentinels; est 2-4 GPU hours; no promotion allowed.
- C_D1 OSC/DecDEC: only materialize a preregistered replacement shard for the current underpowered negative row; est 2-4 GPU hours; no promotion allowed.
- CE13 warmup policy: no parseable cache exists; spend 0 GPU hours now and only consider 2-4 GPU hours after a tiny dev/gate cache exists.

## Hard Stop

Do not run foreground GPU confirmation until a future dashboard contains at least one `PROVISIONAL_PROMOTE_TO_GPU` row or a Channel-Set row with positive offline gate headroom, matched-budget baselines, and control collapse.
