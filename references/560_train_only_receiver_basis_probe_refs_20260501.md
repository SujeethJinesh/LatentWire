# Train-Only Receiver-Basis Probe References

This memo supports `paper/source_private_train_only_receiver_basis_probe_20260501.md`.

## Closest Latent-Communication Baselines

- Cache-to-Cache (C2C) directly projects and fuses source-model KV cache into a
  target model, making it the closest high-rate internal-state communication
  baseline for this project: https://arxiv.org/abs/2510.03215 and
  https://openreview.net/forum?id=LeatkxrBCi.
- LatentMAS moves multi-agent collaboration into continuous latent working
  memory; the distinction here is rate-capped source-private packets and strict
  source-destroying controls rather than shared hidden-state memory:
  https://arxiv.org/abs/2511.20639.
- Interlat transmits last hidden states between agents; the distinction here is
  a low-byte packet decoded against receiver candidate side information rather
  than dense hidden-state transfer: https://arxiv.org/abs/2511.09149.

## Coding And Receiver-Side-Information Framing

- Slepian-Wolf coding motivates the broad idea that an encoder can send only
  information missing from decoder side information:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources.
- Error-correcting output codes motivate candidate-codeword and syndrome-style
  decoding controls: https://arxiv.org/abs/cs/9501101.

## Common-Basis And Quantized-Projection Inspiration

- Relative Representations motivate comparing models through coordinates
  defined by shared anchors rather than raw latent axes:
  https://openreview.net/forum?id=SrC-nwieGJ.
- QJL motivates randomized low-bit projection/sketching as a systems-conscious
  latent-vector compression control: https://arxiv.org/abs/2406.03482.
- TurboQuant motivates treating KV/cache compression as a systems comparator,
  not as a direct duplicate of source-private packet communication:
  https://arxiv.org/abs/2504.19874.

## Local Outcome

The train-only receiver-basis probes did not clear the strict gate:

- semantic-anchor receiver: useful source signal, but permuted-teacher control
  rises above the clean-control band
- train-only sender plus semantic-anchor receiver: same-family improvement, but
  no cross-family improvement over the base source packet
- candidate-local payload-innovation centering: same-family improvement, but
  cross-family collapse

This weakens simple common-basis and payload-centering variants. The next
receiver-side method needs a control-blocking mechanism before n512 widening.
