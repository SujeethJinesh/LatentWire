# Candidate-Local Margin Atlas References, 2026-05-01

## Purpose

This memo supports the candidate-local margin atlas. The local artifact uses
stored per-candidate scores to separate real packet confidence from target-only
fallbacks, destructive controls, and common-basis leakage.

Local artifact:
`results/source_private_candidate_local_margin_atlas_20260501/`

Code:
`scripts/build_source_private_candidate_local_margin_atlas.py`

## Primary Sources and Boundaries

- Predicting the Generalization Gap in Deep Networks with Margin
  Distributions. ICLR 2019. https://openreview.net/forum?id=HJlQfnCqKX
  - Boundary: margin distributions are prior art for diagnosing learned
    decision surfaces. LatentWire uses margins as a receiver diagnostic, not as
    a new generalization bound.
- On Calibration of Modern Neural Networks. ICML 2017.
  https://proceedings.mlr.press/v70/guo17a.html
  - Boundary: calibration explains why score confidence must be audited. The
    margin atlas is not claiming calibrated probabilities; it checks paired
    margin separation against destructive controls.
- Selective Classification for Deep Neural Networks. NeurIPS 2017.
  https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
  - Boundary: abstention/risk-coverage methods motivate thresholded decisions.
    The local contribution is source-private packet acceptance under controls.
- Revisiting Model Stitching to Compare Neural Representations. NeurIPS 2021.
  https://openreview.net/forum?id=ak06J5jNR4
  - Boundary: stitching/common-basis transfer is a representation-comparison
    baseline. The local Procrustes row is explicitly falsified by a
    permuted-teacher destructive control.
- Latent Communication in Artificial Neural Networks.
  https://arxiv.org/abs/2406.11014
  - Boundary: latent communication is close motivation, but the local method is
    a source-private candidate-local packet with byte and control accounting.
- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  arXiv:2510.03215. https://arxiv.org/abs/2510.03215
  - Boundary: C2C projects/fuses source KV cache. The margin atlas keeps the
    source text and source KV outside the transmitted object.
- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
  OpenReview. https://openreview.net/forum?id=F7rUng23nw
  - Boundary: KV sharing is a systems competitor with internal-state exposure.
    It is not the same threat model as an 8-byte source-private packet.
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  arXiv:2504.19874. https://arxiv.org/abs/2504.19874
  - Boundary: TurboQuant and related QJL residual correction are strong
    compression baselines. LatentWire should cite them for systems byte floors
    and avoid claiming novelty for vector quantization primitives.
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
  - Boundary: one-bit random sketches are prior art. The local random-rotation
    sign row is weakened because it does not find a clean control-separated
    operating point.
- Vector Quantised-Variational AutoEncoder. NeurIPS 2017.
  https://papers.nips.cc/paper/7210-neural-discrete-representation-learning
  - Boundary: discrete latent codebooks motivate future learned residual-code
    packets. The current atlas is a diagnostic, not a learned VQ interface.

## Local Finding

The live candidate-local residual packet has positive gold-candidate margins
on `0.750` of matched rows, compared with `0.375` for its best strict
destructive control. The live matched p50 margin is `0.243`, and oracle
candidate atoms reach `0.875` positive-margin rate.

Procrustes common-basis transfer has matched/control positive-margin rates of
`0.750` / `0.750`. That is useful because it shows a common basis can move the
score surface, but it fails the source-private destructive-control test.

## Next Gate

Train or select a candidate-conditioned residual code against this margin
surface: increase matched margins, preserve the `0.48` clean threshold band,
and keep zero-source, shuffled-source, random-same-byte, atom-derangement,
private-random-source, and permuted-teacher controls near target.
