# Reference Memo 711: Qwen-To-Phi Quantized Score Packet Gate

Date: 2026-05-04

## Local Result Boundary

The Qwen-to-Phi equal-byte quantized source-score packet gate failed on
HellaSwag validation `1024:2048`. The best row was a rotated uniform z-score
packet at `2B` raw / `5B` framed, reaching `0.468750` accuracy versus fixed
hybrid `0.467448`, but paired CI95 low was negative (`-0.003906`) and the
second validation slice regressed. Raw source-score logit fusion was harmful
(`0.391927`), while the source top1/top2 oracle stayed high (`0.675781`).

Consequence: score-level source evidence has headroom only under an oracle
switch. Shallow score packets, top-pair buckets, and quantized score-vector
transport should be treated as saturated on this Qwen-to-Phi surface unless a
new source-specific causal feature is introduced.

## Closest Prior Work And Boundary

- Knowledge distillation / dark knowledge: Hinton, Vinyals, and Dean show that
  teacher score distributions can carry useful training signal beyond hard
  labels. Our gate differs because it is inference-time, fixed-byte, and
  source-private, and it intentionally tests whether a tiny score packet is
  enough without training Phi on raw teacher scores.
  Source: https://arxiv.org/abs/1503.02531

- Decoding-time logit composition: DExperts, contrastive decoding, and
  proxy-tuning combine model predictions or prediction differences at decoding
  time. The failed raw-score and quantized-score rows show that LatentWire
  cannot claim novelty from generic score fusion; any positive method must beat
  this family under equal-byte controls.
  Sources: https://arxiv.org/abs/2105.03023,
  https://arxiv.org/abs/2210.15097,
  https://arxiv.org/abs/2401.08565

- Selective prediction / defer systems: selective classifiers and
  learning-to-defer methods overlap with the idea of switching only when a
  calibrated receiver expects benefit. LatentWire should report fixed-coverage
  paired accuracy, not abstention-only gains.
  Sources: https://arxiv.org/abs/1705.08500,
  https://arxiv.org/abs/1901.09192

- Cache-to-Cache communication: C2C directly projects and fuses source KV-cache
  state into target caches and reports accuracy/latency gains over text
  communication. It is the closest cross-model systems competitor, but it
  exposes dense KV state rather than a fixed-byte source-private packet.
  Source: https://arxiv.org/abs/2510.03215

- QJL and TurboQuant: these motivate the random-rotation/scalar-quantization
  comparator. QJL uses a JL transform plus sign-bit quantization for KV-cache
  compression, and TurboQuant uses random rotations plus scalar quantizers to
  approach vector quantization distortion limits. Our rotated score packet is
  only a four-coordinate MCQ score-vector comparator; it is not a KV-cache
  compression method or a systems replacement for those baselines.
  Sources: https://arxiv.org/abs/2406.03482,
  https://arxiv.org/abs/2504.19874

- KV-cache compression systems: KIVI and KVQuant define hard systems baselines
  for compressing dense KV state and making long-context inference practical.
  LatentWire's Mac packet-ring evidence is much narrower: fixed-byte packet
  movement, not native GPU serving, HBM reduction, or KV-cache acceleration.
  Sources: https://arxiv.org/abs/2402.02750,
  https://arxiv.org/abs/2401.18079

## Consequence For The Next Gate

Do not continue iterating on shallow score transport unless a new result
explains why a source top1/top2 answer should override Phi on specific rows.
The highest-value next branch is a target-native resonance receiver:

- use target full-text activations/logits as the reference state;
- train compact source-derived soft slots to match answer-relevant late-layer
  or logit subspaces rather than full hidden state;
- include RMS/on-manifold regularization;
- audit wrong-row, candidate-roll, target-derived, zero-source, and same-byte
  random controls;
- report fixed hybrid, candidate-only, raw/quantized score packets, and C2C/KV
  exposure boundaries as mandatory baselines.
