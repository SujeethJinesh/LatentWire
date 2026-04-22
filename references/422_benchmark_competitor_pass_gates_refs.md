# Benchmark Competitor Pass-Gates Refs (2026-04-22)

Purpose: freeze the widening order and competitor split so later tables stay
 interpretable.

## Strongest Sources

1. GSM8K
   Link: https://arxiv.org/abs/2110.14168

2. C2C
   Link: https://arxiv.org/abs/2510.03215

3. KVComm
   Link: https://arxiv.org/abs/2510.03346

4. RULER
   Link: https://arxiv.org/abs/2404.06654

5. SCBench
   Link: https://arxiv.org/abs/2412.10319

6. LongBench v2
   Link: https://arxiv.org/abs/2412.15204

7. LongBench Pro
   Link: https://arxiv.org/abs/2601.02872

8. LatentMAS
   Link: https://arxiv.org/abs/2511.20639

## Exact Block Structure

1. same-pair / same-family communication
2. cross-family text communication
3. long-context
4. multimodal
5. optional systems appendix

## Pass Gates

1. beat target on frozen `GSM8K32` with clean coverage
2. then `RULER-32`
3. then `SCBench`
4. then `LongBench v2`
5. only later `LongBench Pro` and multimodal

## Current Read

- Keep `C2C` and `KVComm` as the main external direct-communication baselines.
- Keep `LatentMAS` in appendix unless it is ported into the exact same
  contract.
