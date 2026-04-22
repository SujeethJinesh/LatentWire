# Same-Pair Block Contract Refs (2026-04-22)

Purpose: freeze the main-table structure so we do not widen into a mixed
leaderboard before a new same-pair row survives GSM8K32.

## Tightened Contract

- Main Table A: same-pair direct communication only
- Main Table B: long-context controls only
- Appendix: multi-agent workflow, compression-only, and non-direct-communication
  papers
- Hard widening gate: beat `target_alone` on frozen GSM8K32 with numeric
  extraction coverage `>= 31/32`, then replay matched peers on the same IDs

## Exact Same-Pair Block Order

1. `target_alone`
2. `text_to_text`
3. `target_self_repair` if present
4. `C2C`
   Link: https://arxiv.org/abs/2510.03215
5. `KVComm`
   Link: https://arxiv.org/abs/2510.03346
6. `Latent Space Communication via K-V Cache Alignment`
   Link: https://arxiv.org/abs/2601.06123
7. `Ours`

## Long-Context Block Order

1. `RULER-32`
   Link: https://arxiv.org/abs/2404.06654
2. `SCBench`
   Link: https://arxiv.org/abs/2412.10319
3. `LongBench v2`
   Link: https://arxiv.org/abs/2412.15204
4. `LongBench Pro` only after core stability
   Link: https://arxiv.org/abs/2601.02872

## Appendix-Only for Now

- `LatentMAS`
  Link: https://arxiv.org/abs/2511.20639
- `Q-KVComm`
  Link: https://arxiv.org/abs/2512.17914
- `Communication to Completion`
  Link: https://arxiv.org/abs/2510.19995

## Current Read

- Do not widen beyond GSM8K32 until a new same-pair row beats `target_alone`
  cleanly and survives a matched peer replay.
