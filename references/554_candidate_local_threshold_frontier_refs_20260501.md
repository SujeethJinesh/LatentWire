# Candidate-Local Threshold Frontier References, 2026-05-01

## Purpose

This memo supports the candidate-local threshold frontier diagnostic. The local
artifact replays stored per-candidate scores across receiver thresholds and
checks whether matched source-private packets retain a clean control-separated
operating band.

Local artifact:
`results/source_private_candidate_local_threshold_frontier_20260501/`

Code:
`scripts/build_source_private_candidate_local_threshold_frontier.py`

## Primary Sources and Boundaries

- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  arXiv:2510.03215. https://arxiv.org/abs/2510.03215
  - Boundary: C2C transfers/fuses source KV cache. The threshold frontier tests
    a no-source-text/no-source-KV packet receiver with destructive controls.
- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
  arXiv:2510.03346. https://arxiv.org/abs/2510.03346
  - Boundary: selective KV sharing is a high-rate internal-state communication
    baseline, not a strict source-private packet diagnostic.
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  arXiv:2504.19874. https://arxiv.org/abs/2504.19874
  - Boundary: random rotations, scalar quantization, and QJL-style residual
    correction are prior work for vector/KV compression. LatentWire should not
    claim novelty for those mechanisms.
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
  - Boundary: public random projection plus sign-bit sketches are prior work.
    The local contribution is the source-private control frontier, not the
    sketching primitive.
- Relative representations enable zero-shot latent space communication.
  arXiv:2209.15430. https://arxiv.org/abs/2209.15430
  - Boundary: RR is the strongest clean mathematical competitor in the current
    artifact, but the replay shows it has no all-row clean threshold because
    holdout-to-core collapses.
- Diffusion Transformers. arXiv:2212.09748.
  https://arxiv.org/abs/2212.09748
  - Boundary: transformer denoising in latent spaces motivates iterative
    refinement as a future branch, but it is not source-private model
    communication.
- Consistency Models. arXiv:2303.01469.
  https://arxiv.org/abs/2303.01469
  - Boundary: one/few-step refinement motivates a future candidate-logit
    refiner, but the threshold frontier itself is a deterministic score replay.
- REPresentation Alignment for Generation. arXiv:2410.06940.
  https://arxiv.org/abs/2410.06940
  - Boundary: alignment to pretrained representations supports the public-basis
    analogy, but it is generation-side representation regularization rather
    than cross-model source-private transfer.
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders
  and Large Language Models. arXiv:2301.12597.
  https://arxiv.org/abs/2301.12597
  - Boundary: Q-Former-style bottleneck connectors are useful inspiration for a
    learned receiver, but they do not provide the LatentWire source-destroying
    control protocol.

## Local Finding

The live candidate-local residual receiver has a clean threshold band:
`0.45-0.48`. At `0.48`, all `9/9` n512 replay rows are clean; minimum matched
accuracy is `0.500`, maximum best destructive control is `0.260`, and minimum
matched-control gap is `0.240`.

RR anchor-coordinate dot product has no all-row clean threshold. Public
random-rotation sign sketch has no all-row clean threshold. This supports a
safe claim that the live residual receiver is not explained by generic
public-basis transport, random projection, or threshold luck.

## Next Gate

Use this diagnostic in the top-level ICLR evidence bundle, then implement a
true candidate-conditioned residual code with control regularization if the
next turn pursues a method contribution.
