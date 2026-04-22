# Same-Pair Expansion Blocker Gates Refs (2026-04-22)

Purpose: freeze widening order and stop us from mixing incomparable benchmark
 blocks before the same-pair method is genuinely alive.

## Strongest Sources

1. GSM8K
   Link: https://arxiv.org/abs/2110.14168
2. C2C
   Link: https://arxiv.org/abs/2510.03215
3. KVComm
   Link: https://arxiv.org/abs/2510.03346
4. Latent Space Communication via K-V Cache Alignment
   Link: https://arxiv.org/abs/2601.06123
5. RULER
   Link: https://arxiv.org/abs/2404.06654
6. SCBench
   Link: https://arxiv.org/abs/2412.10319
7. LongBench v2
   Link: https://arxiv.org/abs/2412.15204

## Block Order

1. frozen same-pair GSM8K32
2. `RULER-32`
3. one matched cross-family pair
4. `SCBench`
5. `LongBench v2`

## Promotion Gates

- beat `target_alone` and keep coverage
- either beat the current live same-pair row or clearly win on bytes/latency at matched accuracy
- only then widen

## Current Read

- The main risk now is not lack of ideas; it is widening tables before the
  same-pair lane actually survives.
