# HellaSwag Non-Qwen Score-Simplex Receiver References

Date: 2026-05-03

## Purpose

This memo records the literature boundary after the TinyLlama-to-Phi
score-simplex Fourier/SVD receiver gate. The result is negative for receiver
fusion, but it sharpens the paper story: LatentWire currently has strong
fixed-byte packet evidence, while common-basis receiver learning remains the
ICLR blocker.

## Evidence Boundary

- Artifact:
  `results/source_private_hellaswag_nonqwen_score_simplex_receiver_gate_20260503_validation1024_2048/`
- Weighted Phi target-only accuracy: `0.263021`.
- Weighted TinyLlama packet-only accuracy: `0.506510`.
- Weighted score-simplex receiver accuracy: `0.442708`.
- Weighted target-or-packet oracle accuracy: `0.619792`.
- Decision: score-simplex Fourier/SVD fusion is weakened because receiver
  improvement holds on `0/2` adjacent slices.

## Primary Related Work

- Relative Representations:
  https://arxiv.org/abs/2209.15430
  - Closest common-coordinate precedent. LatentWire must distinguish itself by
    fixed-byte source-private packets, destructive controls, and task utility,
    not merely by using a shared coordinate system.
- Product of Invariances / Bricks to Bridges:
  https://openreview.net/forum?id=vngVydDWft
  - Extends relative representations with invariance products. This is a
    natural comparison for any future anchor-relative score packet.
- Relative Semantic Channel Equalization:
  https://arxiv.org/abs/2411.19719
  - Uses anchor-relative spaces for heterogeneous semantic communication. This
    directly pressures any "common language" claim.
- Model Stitching:
  https://arxiv.org/abs/2106.07682
  - A trainable frozen-network compatibility layer is a representation
    comparison baseline, but it does not enforce the LatentWire fixed-byte
    source-private packet contract.
- SVCCA:
  https://papers.neurips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability
  - SVD/CCA alignment is old representation-analysis machinery; our novelty
    cannot be "we used SVD."
- Prefix-Tuning:
  https://arxiv.org/abs/2101.00190
  and Gist Tokens:
  https://arxiv.org/abs/2304.08467
  - Required baselines for any learned continuous-token or prompt-compression
    variant.
- Sparse Crosscoders:
  https://transformer-circuits.pub/2024/crosscoders/index.html
  and SAE feature universality:
  https://arxiv.org/abs/2410.06981
  - Motivate sparse shared/private feature packets, but are not yet available
    in the cached Phi score-only gate.
- C2C / Cache-to-Cache:
  https://openreview.net/forum?id=LeatkxrBCi
  and KVComm:
  https://openreview.net/forum?id=F7rUng23nw
  - Closest direct model-to-model communication systems baselines. LatentWire's
    current differentiator is byte/privacy/exposure, not raw cache-transfer
    capability.
- QJL:
  https://arxiv.org/abs/2406.03482
  and TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Required byte-floor comparisons for any systems claim; they quantize vectors
    or KV-like state rather than sending task-evidence packets.

## Reviewer Implication

Reviewers will collapse score-simplex Fourier/SVD into relative
representations, SVCCA, or model stitching unless we show a strict fixed-byte
packet that beats packet-only competitors and fails under source-destroying
controls. The current result does not clear that bar.

## Next Branch

The next Mac-local branch should compare:

- packet-only;
- anchor-relative score packet at the same byte budget;
- packet-preserving target acceptor;
- source-destroying controls: row shuffle, candidate roll, basis permutation,
  basis sign flip, target-derived packet, and same-byte random packet.

Promote only if the method beats packet-only with positive paired CI on both
adjacent non-Qwen slices.
