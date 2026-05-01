# ARC-Challenge Anchor-Control References, 2026-05-01

## Local Finding

Artifacts:

- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_validation/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_test/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_validation/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_test/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_validation/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_test/`

The controls reuse the answer-key-forbidden Qwen source-choice cache and vary
only the public anchor coordinate chart. Anchor-ID and anchor-value mismatch
controls fail `0/5` validation seeds and `0/5` test seeds, collapsing to
target-level accuracy. Shared random anchors pass `5/5` validation seeds and
`5/5` test seeds.

The result supports a common-basis packet claim: source and receiver need an
agreed coordinate chart. It does not support a semantic train-anchor superiority
claim because random shared anchors are sufficient on ARC.

## Primary Sources

- ARC / AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
- Relative Representations. https://arxiv.org/abs/2209.15430
- Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- KVComm selective KV sharing. https://arxiv.org/abs/2510.03346
- QJL. https://arxiv.org/abs/2406.03482
- TurboQuant. https://arxiv.org/abs/2504.19874
- Diffusion Transformers. https://arxiv.org/abs/2212.09748
- LaDiR latent diffusion reasoner. https://arxiv.org/abs/2510.04573

## Safe Claim Boundary

Safe: fixed-byte source-private packets require a shared public coordinate chart
on ARC; mismatching coordinate identities or anchor values destroys transfer.

Safe: random public anchors work about as well as train anchors on ARC, so the
method is best framed as common-basis communication.

Unsafe: semantic train-anchor superiority, novel relative representations, or
new JL/QJL theory. Those are prior ideas; LatentWire's distinct contribution is
the source-private fixed-byte packet protocol plus destructive controls and
systems accounting.
