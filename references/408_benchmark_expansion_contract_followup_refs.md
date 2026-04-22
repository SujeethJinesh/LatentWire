# Benchmark Expansion Contract Follow-Up Refs (2026-04-21)

Purpose: freeze the next fair benchmark order after the GSM8K32 same-pair smoke
so we keep tables interpretable and medium-split.

## Strongest Sources

1. RULER
   Link: https://arxiv.org/abs/2404.06654
   Why it matters: cleanest synthetic long-context control suite for retrieval,
   aggregation, and multi-hop tracing.

2. SCBench
   Link: https://arxiv.org/abs/2412.10319
   Why it matters: best KV-lifecycle benchmark for a communication paper.

3. LongBench v2
   Link: https://arxiv.org/abs/2412.15204
   Why it matters: strongest realistic long-context reasoning follow-up after
   the synthetic gate.

4. InfinityBench
   Link: https://arxiv.org/abs/2402.13718
   Why it matters: stress-tests extreme context length instead of only moderate
   long-context regimes.

5. LongGenBench
   Link: https://arxiv.org/abs/2410.04199
   Why it matters: catches long-form generation failures that retrieval-only
   metrics miss.

6. ActionReasoningBench
   Link: https://arxiv.org/abs/2406.04046
   Why it matters: strongest structured state/action reasoning split for
   interpretable failure decomposition.

7. LongBench Pro
   Link: https://arxiv.org/abs/2601.02872
   Why it matters: recent realism-focused long-context successor if we need a
   2026-aligned follow-up block.

## Recommended Order

1. `RULER`
2. `SCBench`
3. `LongBench v2`
4. `InfinityBench`
5. `LongGenBench`
6. `ActionReasoningBench`

## Table Blocks

1. same-pair communication
2. synthetic long-context control
3. KV lifecycle / shared-context reuse
4. realistic long-context reasoning
5. generation-specific long-context rows
6. structured state/action reasoning
7. multimodal rows kept separate

## Exact Failure Checks

- retrieval hit rate
- trace consistency
- aggregation correctness
- numeric extraction coverage
- per-length-bucket accuracy
- generation constraint satisfaction
- state tracking vs action-executability breakdown

## Current Read

- Do not widen into a mixed leaderboard.
- The next block should only be added after a method survives the frozen
  GSM8K32 smoke with better-than-target accuracy and clean extraction coverage.
