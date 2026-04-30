# Masked Consistency Label-Blind Stress References

- date: `2026-04-30`
- purpose: primary-source grounding for label-blind, anti-shortcut, and
  side-information controls in the learned masked-consistency receiver stress.

## Annotation Artifacts in NLI

- source: https://aclanthology.org/N18-2017/
- blocker helped: high receiver accuracy can come from public metadata or
  candidate artifacts rather than private source evidence.
- mechanism idea: include hypothesis-only analogues: candidate-only,
  packet-free, opaque-slot, and metadata-only controls.
- next experiment change: opaque slot-remap collapse is mandatory; a future
  public-only learned receiver ablation should be added for semantic rows.
- role: ablation / control framing.

## HellaSwag / Adversarial Filtering

- source: https://aclanthology.org/P19-1472/
- blocker helped: easy distractors and simple negative controls can make
  shortcut-based methods look valid.
- mechanism idea: adversarially harden candidate views and remap/derange
  public candidate tables.
- next experiment change: keep remapped opaque slots and future hard negative
  candidate maps before promoting a broad claim.
- role: benchmark design / ablation inspiration.

## Wyner-Ziv Source Coding With Decoder Side Information

- source: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
- blocker helped: explains why tiny packets can be useful without carrying the
  entire private source trace.
- mechanism idea: decoder-side public candidate semantics act as side
  information, while source bytes carry conditional residual evidence.
- next experiment change: report candidate-view dependence explicitly instead
  of calling the result protocol-free latent transfer.
- role: theory support / framing.

## Distributed Indirect Source Coding With Decoder Side Information

- source: https://arxiv.org/abs/2405.13483
- blocker helped: the source observes private evidence about a task variable,
  not necessarily the answer text itself.
- mechanism idea: optimize task distortion under source observation plus
  decoder-only side information.
- next experiment change: keep downstream candidate accuracy and destructive
  controls as the primary distortion, not packet reconstruction.
- role: theory support / experiment framing.

## Denoising Autoencoders

- source: https://icml.cc/Conferences/2008/papers/592.pdf
- blocker helped: a receiver can memorize clean packet-code associations unless
  trained to handle corrupted inputs.
- mechanism idea: train clean and corrupted views to recover the same target
  state.
- next experiment change: retain masked packet views and add no-corruption
  learned-decoder ablations when widening.
- role: method / ablation.

## Consistency Models

- source: https://proceedings.mlr.press/v202/song23a.html
- blocker helped: motivates one-step recovery from noisy packet observations.
- mechanism idea: learn a direct map from corrupted/noisy states to a common
  clean endpoint.
- next experiment change: keep one-step receiver as the headline row and use
  iterative refinement only as a future ablation.
- role: method inspiration / framing.

## BLIP-2 / Q-Former

- source: https://arxiv.org/abs/2301.12597
- blocker helped: reviewers may ask for a learned bridge between frozen source
  and frozen target states.
- mechanism idea: small query bottlenecks can extract task-relevant evidence
  from a frozen source representation before feeding a frozen receiver.
- next experiment change: if byte receiver saturates, revisit a small
  Q-Former-style source-private query bottleneck as a distinct method branch.
- role: inspiration / alternate branch.

## Bottom Line

The label-blind stress should be framed as an anti-shortcut control. The
positive claim is strongest when full/semantic public side information passes,
opaque remapped slots collapse, and train/eval IDs are disjoint. The result
supports source-private communication with decoder side information, not
protocol-free latent transfer.
