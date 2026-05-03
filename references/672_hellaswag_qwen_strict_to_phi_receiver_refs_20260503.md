# HellaSwag Qwen Strict Packet To Phi Receiver References

Date: 2026-05-03

## Purpose

This memo records the novelty boundary after testing the current strongest
Qwen HellaSwag strict hidden-innovation packet against cached Phi-3 receiver
scores. The result shows useful packet transfer but fails receiver fusion, so
it motivates a common score-simplex receiver instead of another generic
ridge/confidence gate.

## Evidence Boundary

- Artifact:
  `results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_multislice_20260503_validation1024_2048/`
- Weighted Phi target-only accuracy: `0.263021`.
- Weighted Qwen strict packet-only accuracy: `0.455729`.
- Weighted candidate ridge receiver accuracy: `0.401042`.
- Weighted target-or-packet oracle accuracy: `0.593750`.
- Decision: source packet signal transfers, but receiver improvement fails on
  `0/2` adjacent slices.

## Primary Related Work

- Relative Representations:
  https://arxiv.org/abs/2209.15430
  - Provides an anchor-relative common-coordinate prior. It motivates the next
    common-basis branch but also prevents us from claiming anchor coordinates
    as novel by themselves.
- Sparse Autoencoders Reveal Universal Feature Spaces Across LLMs:
  https://arxiv.org/abs/2410.06981
  - Motivates sparse shared features across models. LatentWire must use sparse
    features as packet coordinates with destructive controls, not as a standalone
    novelty claim.
- Sparse Crosscoders:
  https://transformer-circuits.pub/2024/crosscoders/index.html
  - Shared/private feature decomposition is relevant to future hidden-basis
    packets, but current Phi artifacts only cache scores, so the immediate
    Mac-local gate should use the public candidate score simplex.
- Prefix-Tuning:
  https://arxiv.org/abs/2101.00190
  and Gist Tokens:
  https://arxiv.org/abs/2304.08467
  - Required baselines for any soft-token or compressed prompt variant.
- C2C / Cache-to-Cache:
  https://openreview.net/forum?id=LeatkxrBCi
  - Closest direct model-to-model communication competitor. It transfers and
    fuses source KV state, unlike the fixed-byte source-private packet here.
- KVComm:
  https://openreview.net/forum?id=F7rUng23nw
  - Selective KV sharing is a cache/state-transfer systems comparator.
- QJL:
  https://arxiv.org/abs/2406.03482
  and TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Byte-floor and vector/KV quantization comparators. They inspire sign-sketch
    controls but do not evaluate source-private task-evidence packets.
- Consistency Models:
  https://arxiv.org/abs/2303.01469
  - Inspires one-step repair/accept heads. Treat as inspiration unless a
    LatentWire receiver is actually trained and controlled.

## Next-Method Implication

The immediate common-basis path should use the agreed multiple-choice candidate
basis rather than hidden states:

- row-center Tiny/Qwen and Phi score vectors;
- project the four candidate scores into an orthonormal contrast basis;
- optionally fit train-prefix SVD/CCA on source/target score contrasts;
- train a receiver that can override packet-only only when the target score
  simplex indicates packet harm;
- include basis-permutation, sign-flip, row-shuffle, candidate-roll,
  target-derived packet, source-rank/index, source-score, and same-byte controls.

Promote only if it beats packet-only with positive paired CI on both adjacent
slices and does not survive source-destroying controls.
