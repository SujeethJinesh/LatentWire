# Anti-Shuffle Innovation Sender References, 2026-05-01

## Role

This memo supports the anti-shuffle innovation sender gate and its novelty
boundary.

## Primary Sources

- Slepian-Wolf distributed source coding:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
- Wyner-Ziv coding with decoder side information:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- DISCUS syndrome coding:
  https://www.researchgate.net/publication/2352091_Distributed_Source_Coding_Using_Syndromes_DISCUS_Design_and_Construction
- ECOC output codes:
  https://arxiv.org/abs/cs/9501101
- QJL:
  https://arxiv.org/abs/2406.03482
- TurboQuant:
  https://arxiv.org/abs/2504.19874
- Relative Representations:
  https://openreview.net/forum?id=SrC-nwieGJ
- C2C:
  https://arxiv.org/abs/2510.03215
- KVComm:
  https://arxiv.org/abs/2510.03346
- KVCOMM:
  https://arxiv.org/abs/2510.12872
- ARC-Challenge:
  https://arxiv.org/abs/1803.05457

## Boundary

Anti-shuffle innovation should be framed as a source-private LLM packet
selection rule under destructive controls, not as a new source-coding theorem
or a new vector quantizer. It is most defensible as a task-directed
decoder-side-information protocol: the receiver has public candidate side
information, and the sender emits only a tiny packet meant to change the
receiver's decision while controls show the packet is source-specific.
