# Benchmark Expansion Order Refs (2026-04-21)

Purpose: keep the benchmark story medium-split and interpretable while we move
past the current frozen GSM8K32 same-pair smoke.

## Strongest Sources

1. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: live same-pair comparator for latent / KV communication.

2. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: strongest selective-KV communication baseline with a budget.

3. LatentMAS
   Link: https://arxiv.org/abs/2511.20639
   Why it matters: latent multi-agent communication reference; keep appendix-only
   until heterogeneity is matched.

4. ReasonBridge
   Link: https://arxiv.org/abs/2506.22865
   Why it matters: strongest reasoning-transfer comparator, but should stay in
   a separate table block from direct communication.

5. RULER
   Link: https://arxiv.org/abs/2404.06654
   Why it matters: smallest clean long-context contract expansion after the
   current GSM8K32 smoke.

6. SCBench
   Link: https://arxiv.org/abs/2412.10319
   Why it matters: strongest KV-centric benchmark if we need an
   efficiency-aware block.

7. LongBench v2
   Link: https://arxiv.org/abs/2412.15204
   Why it matters: strongest realistic long-context follow-up once the next
   synthetic contract is stable.

8. MMMU-Pro
   Link: https://arxiv.org/abs/2409.02813
   Why it matters: strongest multimodal row anchor once the text-only story is
   stable enough to widen.

## Recommended Order

1. Frozen `RULER` 32-example same-pair contract
2. `SCBench`
3. `LongBench v2`
4. multimodal rows like `MMMU-Pro`

## Table Blocks

1. same-pair / same-family communication
2. cross-family text communication
3. reasoning transfer
4. multimodal communication

## Current Read

- Do not widen into a mixed leaderboard.
- One new block should only be added after the current block has at least one
  stable, fair row.
