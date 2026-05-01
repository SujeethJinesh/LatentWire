# ARC-Challenge Anchor-Relative Basis References, 2026-05-01

## Local Finding

Artifacts:

- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_validation/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_test/`

The anchor-relative ARC endpoint reuses the answer-key-forbidden Qwen source
choice cache from the public hashed-basis run, but changes the receiver-side
packet basis. Each question/candidate pair is encoded by its similarities to a
deterministic public set of train-split question/candidate anchors, then
centered, normalized, projected, and packed into the same fixed `12B` sparse
packet.

Validation passes `5/5` projection seeds. Matched mean/min/max is
`0.386/0.381/0.388`, target is `0.244`, same-byte structured text is `0.348`,
minimum lift over same-byte text is `0.033`, and minimum CI95 lower bound
versus target is `0.065`.

Test passes `5/5` projection seeds. Matched mean/min/max is
`0.344/0.344/0.345`, target is `0.265`, same-byte structured text is `0.311`,
minimum lift over same-byte text is `0.033`, and minimum CI95 lower bound
versus target is `0.039`.

## Primary Sources

- ARC / AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
- Relative Representations. https://arxiv.org/abs/2209.15430
- Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- KVComm selective KV sharing. https://arxiv.org/abs/2510.03346
- QJL. https://arxiv.org/abs/2406.03482
- TurboQuant. https://arxiv.org/abs/2504.19874
- Diffusion Transformers. https://arxiv.org/abs/2212.09748
- LaDiR latent diffusion reasoner. https://arxiv.org/abs/2510.04573

## Claim Boundary

Safe: public anchor-relative coordinates can replace direct hashed coordinates
while preserving the ARC fixed-byte packet pass under seed repeats and
destructive controls.

Also safe: this is robustness evidence for the common-basis story, not an
anchor-relative superiority claim. The direct hashed shared-basis endpoint is
at least as strong on the current rows.

Unsafe: claiming that anchor-relative coordinates alone solve raw hidden-state
alignment. This result is a public coordinate-chart endpoint with a Qwen
source-choice cache, not a learned hidden-state communication endpoint.

## Control Follow-Up

Anchor-identity shuffle, anchor-value shuffle, and random-anchor controls are
now recorded in `references/573_arc_challenge_anchor_controls_refs_20260501.md`.
The next literature check should focus on whether a learned anchor set,
Procrustes/Gromov-Wasserstein regularization, or diffusion-style denoising
receiver can turn this public coordinate chart into a true hidden-state endpoint
without losing the source-private controls.
