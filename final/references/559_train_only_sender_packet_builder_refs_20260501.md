# Train-Only Sender Packet Builder References

Date: 2026-05-01

## Local Result

The train-only sender source-prioritized packet builder passes `9/9` n512
seed-repeat rows using a 12B packet:

```text
packet_vector = source_to_candidate_ridge_train_only(source_atoms) + 0.75 * source_atoms
```

The sender builder uses `train_only` calibration. The receiver dictionary uses
`all_public_eval_disjoint`, so the result removes public calibration from
sender construction but not from receiver-basis construction.

## Primary Sources And Novelty Boundaries

- C2C and KV communication papers occupy broad latent/KV model communication.
  LatentWire should not claim first latent LLM communication. The distinction is
  source-private byte-scale packets rather than projected/fused source KV.
  - https://openreview.net/forum?id=LeatkxrBCi
  - https://arxiv.org/abs/2510.03215
  - https://arxiv.org/abs/2510.03346
  - https://arxiv.org/abs/2510.12872

- DroidSpeak and other cache-sharing methods are relevant same-base systems
  baselines. They reuse/share internal KV state; LatentWire avoids source KV
  exposure.
  - https://arxiv.org/abs/2411.02820

- Vector-translation and representation-stitching work weaken any novelty claim
  based only on a ridge/linear map. The contribution should be the controlled
  source-private packet protocol, not linear transport by itself.
  - https://arxiv.org/abs/2511.03945
  - https://arxiv.org/abs/2506.06609

- Soft/prefix prompt methods are mandatory target-only and candidate-only
  controls for “tiny learned vector” communication.
  - https://aclanthology.org/2021.emnlp-main.243/

- ECOC and syndrome-style side-information coding motivate candidate-side
  decoding, but are not identical: LatentWire packets are per-instance
  source-derived evidence, not fixed class codewords.
  - https://arxiv.org/abs/cs/9501101
  - https://www.jmlr.org/papers/v1/allwein00a.html
  - https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf

- Quantization and transform-coding papers motivate the source-prioritized
  residual and future common-basis repair, but they compress internal tensors or
  KV caches, not source-private cross-model semantic evidence packets.
  - https://arxiv.org/abs/2504.19874
  - https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  - https://arxiv.org/abs/2406.03482
  - https://arxiv.org/abs/2402.04396

## Required Reviewer Ablations

- full train-only receiver+sender calibration;
- target-only learned packet at matched bytes;
- candidate-only ECOC/Hadamard/random code packets;
- source packet without candidate side information;
- option-order/candidate-label permutation;
- text answer/rationale/MoA relays with token counts;
- native vLLM TTFT/TPOT/goodput systems rows.
