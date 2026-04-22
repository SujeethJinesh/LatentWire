# Benchmark Block Structure Follow-Up Refs (2026-04-21)

Purpose: freeze the next benchmark-expansion order so later paper tables stay
interpretable and do not mix incompatible communication settings.

## Strongest Sources

1. GSM8K
   Link: https://arxiv.org/abs/2110.14168
   Why it matters: keep `GSM8K32` as the frozen same-pair smoke gate.

2. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: current external same-pair bar on the frozen slice.

3. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: matched direct-communication baseline for long-context and
   cache-style transport.

4. RULER
   Link: https://arxiv.org/abs/2404.06654
   Why it matters: next exact long-context follow-up after the same-pair smoke.

5. SCBench
   Link: https://arxiv.org/abs/2412.10319
   Why it matters: structured long-context benchmark after `RULER`.

6. LongBench v2
   Link: https://arxiv.org/abs/2412.15204
   Why it matters: broader long-context block after `SCBench`.

7. MMMU-Pro
   Link: https://arxiv.org/abs/2409.02813
   Why it matters: multimodal belongs in a separate block and should come last.

## Exact Block Order

1. same-pair / same-family communication
2. cross-family text communication
3. long-context
4. multimodal

## Exact Next Expansion Order

1. keep `GSM8K32` frozen
2. rerun the live same-pair lane on `RULER-32`
3. add one matched cross-family text pair
4. expand to `SCBench`
5. then `LongBench v2`
6. keep multimodal last and separate

## Fixed Control Ladder

- target-alone
- text relay
- target self-repair
- C2C
- KVComm
- method under test

## Current Read

- Do not collapse same-pair, cross-family, long-context, and multimodal rows
  into one mixed table.
- Keep `ReasonBridge` and `LatentMAS` in a separate appendix block unless they
  are ported into a directly comparable contract.
