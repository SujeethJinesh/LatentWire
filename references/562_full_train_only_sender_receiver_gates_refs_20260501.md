# Full Train-Only Sender+Receiver Gates References, 2026-05-01

## Role

This memo supports the full train-only sender+receiver gate and the next
anti-shuffle innovation sender branch.

## Primary Sources To Cite

- Slepian-Wolf distributed source coding:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
- Wyner-Ziv lossy coding with decoder side information:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
- QJL-style randomized residual sketches:
  https://arxiv.org/abs/2406.03482
- TurboQuant residual/rotation quantization:
  https://arxiv.org/abs/2504.19874
- Relative Representations:
  https://openreview.net/forum?id=SrC-nwieGJ
- C2C cache-to-cache communication:
  https://arxiv.org/abs/2510.03215
- KVComm selective KV sharing:
  https://arxiv.org/abs/2510.03346
- ARC-Challenge benchmark:
  https://arxiv.org/abs/1803.05457

## Local Novelty Boundary

The failed branches here are not claims of novelty. They are controls that
separate three ideas:

- common-basis projection is insufficient by itself;
- answer-vs-decoy innovation helps but does not fully block shuffled-source;
- the next branch must optimize packet atoms against matched-vs-shuffled
  receiver gain directly.

Safe framing: the contribution is not generic KV compression or generic model
stitching. The emerging method claim is source-private, byte-scale packet
communication with decoder-side candidate information and explicit destructive
source controls.
