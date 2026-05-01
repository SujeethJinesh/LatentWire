# References: HellaSwag Anchor-Variant Scout

## Purpose

This memo records the primary-source boundary for the failed HellaSwag
anchor-variant scout. The result should be used as a negative diagnostic, not as
a claim that common-basis latent communication is impossible.

## Directly Relevant Prior Work

- Relative Representations show how model activations can be expressed through
  relations to anchors, motivating the anchor-relative branch.
  Source: https://arxiv.org/abs/2209.15430

- Sparse crosscoders learn shared sparse features across models or layers,
  making them a stronger next branch than hand-designed anchor similarities.
  Source: https://transformer-circuits.pub/2024/crosscoders/index.html

- Sparse autoencoder universality and universal SAE work motivate learned
  shared dictionaries, but they do not by themselves provide a fixed-byte
  source-private packet protocol.
  Sources: https://arxiv.org/abs/2410.06981 and https://arxiv.org/abs/2502.03714

- QJL motivates randomized sign sketches as a systems-friendly low-bit basis
  for vector information, but our scout only used QJL-style signs over anchor
  similarities. The next branch should test dense residual sign sketches.
  Source: https://arxiv.org/abs/2406.03482

- TurboQuant is a useful systems comparator for low-bit vector/cache
  quantization, but it compresses vector state rather than selecting an
  extreme-byte source-private decision packet.
  Source: https://arxiv.org/abs/2504.19874

- Prefix tuning and prompt tuning are not the same mechanism: they optimize
  target-side continuous or prompt-prefix conditioning, while this project is
  evaluating a sender-private fixed-byte packet selected using source evidence.
  Sources: https://arxiv.org/abs/2101.00190 and https://arxiv.org/abs/2104.08691

## Uniqueness Boundary

Do not claim that anchor coordinates, RBF features, graph spectral bases, JL/QJL
sketches, SAEs, or crosscoders are new. The publishable claim must be narrower:
a source model uses private internal evidence to select a tiny packet that helps
a receiver under strict label-copy, score-only, and corrupted-source controls.

The anchor-variant scout does not yet support that common-basis claim. It shows
that simple hand-designed anchor charts are insufficient on the strongest
currently available HellaSwag decision surface.

## Experimental Boundary

This scout compared variants on one frozen heldout slice, so it is a branch
selection diagnostic. Even if a variant had passed, it would still need a
predeclared all-five-slice gate before entering the paper as evidence.

Current outcome:

- Best variant: `cosine_full`.
- Best accuracy: `0.512695`.
- Best label-copy accuracy: `0.500000`.
- Delta vs label-copy: `+0.012695`.
- Paired CI95 low vs label-copy: `-0.002930`.
- Scout pass: false.

## Next Literature-Backed Branch

Test dense residual sign sketches inspired by QJL before moving to learned
sparse/crosscoder dictionaries. This separates two hypotheses:

1. Hand-designed anchor charts are too lossy.
2. The dense residual signal cannot be compressed into a systems-friendly common
   basis at all.
