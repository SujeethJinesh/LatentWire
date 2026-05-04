# Anchor-Relative Feature-Mode Grid References

Date: 2026-05-04

This memo records the literature boundary for the failed HellaSwag
anchor-relative top-k/RBF common-basis grid.

## Closest Prior Work

- Relative Representations use similarities to anchors as a shared coordinate
  system for latent communication and stitching:
  https://openreview.net/forum?id=SrC-nwieGJ
- LSTIRP extends the relative-representation idea through inverse relative
  projection:
  https://arxiv.org/abs/2406.15057
- SVCCA and CKA are representation-comparison/alignment diagnostics, not
  communication evidence by themselves:
  https://arxiv.org/abs/1706.05806
  https://arxiv.org/abs/1905.00414
- ICAE, Gist Tokens, and Prefix-Tuning cover compact soft/context interfaces
  for frozen or lightly adapted LMs:
  https://openreview.net/forum?id=uREj4ZuGJE
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html
  https://aclanthology.org/2021.acl-long.353/
- Sparse crosscoders and sparse autoencoders motivate shared/interpretable
  feature bases, but they do not by themselves establish low-rate
  source-private communication:
  https://transformer-circuits.pub/2024/crosscoders/index.html
  https://arxiv.org/abs/2309.08600
- C2C and KVComm are mandatory direct-communication competitors because they
  move projected/fused or selective KV-cache state between LMs:
  https://openreview.net/forum?id=LeatkxrBCi
  https://openreview.net/forum?id=F7rUng23nw
- QJL, KIVI, and TurboQuant define systems-side compressed-vector/KV-cache
  baselines and byte-floor comparators:
  https://arxiv.org/abs/2406.03482
  https://arxiv.org/abs/2402.02750
  https://openreview.net/forum?id=tO3ASKZlok

## Boundary

Anchor-relative coordinates are prior art. The failed top-k/RBF grid should be
used as a negative ablation showing that simply sharpening the anchor chart does
not preserve the useful HellaSwag hidden-innovation signal.

The safe novelty claim remains narrower:

> A fixed-byte, source-private task packet must improve a frozen receiver under
> paired held-out benchmarks and destructive source controls, while exposing far
> fewer bytes than KV/cache/vector-state baselines.

## Next Branch

The next common-basis-adjacent branch should be conditional coding rather than
more anchor retuning:

- Wyner-Ziv style decoder-side information:
  https://doi.org/10.1109/TIT.1976.1055508
- Slepian-Wolf source coding motivation:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
- diffusion/iterative-denoising inspiration for bounded receiver updates:
  https://arxiv.org/abs/2402.07754
  https://arxiv.org/abs/2503.09573

The experiment should transmit tiny source syndromes over candidate-pair
innovations or sparse residual atoms and test whether one to three target-side
denoising steps beat target-only and source-destroyed controls.
