# Train-Donor Anti-Shuffle References, 2026-05-01

## Role

This memo supports the train-only anti-shuffle sender and its systems/novelty
boundary.

## Primary Sources

- Slepian-Wolf distributed source coding:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
- Wyner-Ziv coding with decoder side information:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- DISCUS syndrome coding:
  https://www.researchgate.net/publication/2352091_Distributed_Source_Coding_Using_Syndromes_DISCUS_Design_and_Construction
- Error-correcting output codes:
  https://arxiv.org/abs/cs/9501101
- Relative Representations:
  https://openreview.net/forum?id=SrC-nwieGJ
- QJL:
  https://arxiv.org/abs/2406.03482
- TurboQuant:
  https://arxiv.org/abs/2504.19874
- C2C:
  https://arxiv.org/abs/2510.03215
- KVComm:
  https://arxiv.org/abs/2510.03346
- KVCOMM:
  https://arxiv.org/abs/2510.12872
- KIVI:
  https://arxiv.org/abs/2402.02750
- KVQuant:
  https://arxiv.org/abs/2401.18079
- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
- ARC-Challenge:
  https://arxiv.org/abs/1803.05457

## Boundary

The train-donor anti-shuffle sender is not a new source-coding theorem or a
new quantizer. Its defensible novelty is the controlled LLM protocol:

- train-only sender donors instead of eval-donor contrast;
- byte-scale source-private packets rather than source text or source KV/cache;
- candidate-side-information receiver;
- destructive controls for shuffled, random, permuted, and text-matched source
  shortcuts;
- Mac-local byte/transport accounting with explicit GPU non-claims.

Do not claim native superiority over C2C, KVComm, KVCOMM, TurboQuant, QJL,
KIVI, or KVQuant until those systems are rerun in their native serving access
model.
