# References: HellaSwag Sparse Residual Dictionary Scout

## Purpose

This memo records the prior-work boundary for the sparse residual dictionary
scout. The branch fails, but it is still useful because it separates simple
dictionary clustering from a true SAE/crosscoder method.

## Primary Sources

- Sparse autoencoders are a standard way to decompose language-model activations
  into sparse features. The HellaSwag dictionary scout is not an SAE because it
  has no learned encoder/decoder reconstruction objective.
  Source: https://arxiv.org/abs/2309.08600

- Sparse Crosscoders learn shared sparse features across layers/models. This is
  the closest methodological inspiration for a future LatentWire common-basis
  branch, but our current scout only uses clustered residual atoms.
  Source: https://transformer-circuits.pub/2024/crosscoders/index.html

- Universal Sparse Autoencoders motivate cross-model concept alignment through a
  shared overcomplete sparse space. LatentWire should not claim this until a
  source-private packet survives the frozen gates.
  Source: https://arxiv.org/abs/2502.03714

- Relative Representations motivate anchor/common-coordinate latent interfaces.
  The negative anchor and sparse-dictionary results show that a shared
  coordinate system alone is not sufficient here.
  Source: https://arxiv.org/abs/2209.15430

- Prefix tuning and prompt tuning learn continuous prompt vectors. LatentWire's
  packet is not a soft prompt because no continuous prompt tokens are inserted
  into the receiver; the packet is a per-example fixed-byte selector record.
  Sources: https://arxiv.org/abs/2101.00190 and https://arxiv.org/abs/2104.08691

- C2C and KVComm remain the closest latent/cache communication competitors. They
  communicate or fuse KV/cache state, while this scout transmits no source KV,
  raw hidden vector, raw sparse code, or raw score vector.
  Sources: https://arxiv.org/abs/2510.03215 and https://arxiv.org/abs/2510.03346

- QJL and TurboQuant already cover sign/JL and vector-quantization ideas for KV
  or vector state compression. The sparse dictionary scout is not a quantization
  contribution.
  Sources: https://arxiv.org/abs/2406.03482 and https://arxiv.org/abs/2504.19874

## Experimental Outcome

Unsupervised candidate-residual dictionary:

- best variant: `dict128_cand_signed_top4`
- accuracy: `0.497070`
- best label-copy: `0.500000`
- delta vs label-copy: `-0.002930`
- score-only: `0.497070`
- scout pass: `false`

Contrastive train-label dictionary:

- best variant: `dict64_gold_signed_top4`
- accuracy: `0.498047`
- best label-copy: `0.500000`
- delta vs label-copy: `-0.001953`
- score-only: `0.497070`
- scout pass: `false`

## Non-Claim Boundary

Do not claim novelty for sparse dictionaries, SAEs, crosscoders, prefix/prompt
tuning alternatives, or vector quantization. The safe statement is:

> Simple train-only sparse residual dictionaries do not recover the dense
> hidden-innovation lift on HellaSwag; a real common-basis claim requires a
> trained SAE/crosscoder objective with destructive packet controls.

## Reviewer Controls To Add If Revived

- label-permutation dictionary fit
- atom-ID shuffle
- atom-value/sign shuffle
- top-atom knockout
- same-byte label-code control
- same-byte visible text / prompt-surrogate control
