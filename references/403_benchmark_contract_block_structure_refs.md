# Benchmark Contract Block Structure Refs (2026-04-21)

Purpose: freeze the current benchmark-table structure so same-pair, cross-family,
reasoning-transfer, and multimodal communication rows do not get mixed into one
unfair macro table.

## Strongest Sources

1. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: clearest direct comparator for same-pair latent / KV
   communication.

2. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: strongest selective KV-sharing comparator with an explicit
   communication budget.

3. LatentMAS
   Link: https://arxiv.org/abs/2511.20639
   Why it matters: strongest training-free latent multi-agent communication
   reference, but should stay appendix-only until heterogeneity is matched.

4. ReasonBridge
   Link: https://arxiv.org/abs/2506.22865
   Why it matters: strongest reasoning-transfer comparator and best anchor for
   a separate transfer block rather than a communication block.

5. RULER
   Link: https://arxiv.org/abs/2404.06654
   Why it matters: clean minimal long-context contract expansion after the
   current GSM8K32 smoke.

6. LongBench v2
   Link: https://arxiv.org/abs/2412.15204
   Why it matters: strongest realistic long-context follow-up once the minimal
   contract is stable.

7. SCBench
   Link: https://arxiv.org/abs/2412.10319
   Why it matters: strongest KV-cache-centric benchmark for efficiency-aware
   rows.

8. MMMU-Pro
   Link: https://arxiv.org/abs/2409.02813
   Why it matters: best multimodal row anchor once the text-only story is
   stable enough to widen.

## Recommended Table Blocks

1. Same-pair / same-family communication
   Rows: `target-alone`, `text-to-text`, `C2C`, `KVComm`, `ours`

2. Cross-family text communication
   Rows split by model pair, not averaged into same-pair rows.

3. Reasoning transfer
   Keep adapter / distillation / guidance methods separate from direct latent
   communication.

4. Multimodal communication
   Separate table block; never mixed with text-only contracts.

## Minimum Next Benchmark Expansion

1. Frozen `RULER` 32-example contract on the same Qwen2.5 -> Qwen3 pair
2. If that survives, widen to `LongBench v2`
3. Only then consider multimodal rows such as `MMMU-Pro`

## Current Read

- The final paper table must be medium-split and budgeted.
- Same-pair, cross-family, and multimodal rows should remain separate until we
  have at least one contract win in each block.
