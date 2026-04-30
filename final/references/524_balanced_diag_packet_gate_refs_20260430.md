# Balanced Diagnostic Packet Gate References

- date: `2026-04-30`
- purpose: grounding for the balanced public-only ablation and the cleaner
  source-private diagnostic-packet contribution.

## Annotation Artifacts in NLI

- source: https://aclanthology.org/N18-2017/
- blocker helped: public candidate wording can solve the task without source
  evidence.
- mechanism idea: train a public-only classifier and treat success as evidence
  of annotation/artifact leakage.
- next experiment change: public-only diagnostic receiver is now a required
  ablation for balanced candidate tables.
- role: ablation / shortcut-control framing.

## HellaSwag / Adversarial Filtering

- source: https://aclanthology.org/P19-1472/
- blocker helped: easy public distractors can make a benchmark look solved by a
  communication method.
- mechanism idea: harden distractors so a shortcut model cannot rely on
  superficial artifacts.
- next experiment change: replace obvious `X*` distractor diagnostics with
  plausible diagnostic-code decoys.
- role: benchmark design inspiration.

## Wyner-Ziv Source Coding With Decoder Side Information

- source: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
- blocker helped: tiny packets need a formal explanation.
- mechanism idea: the receiver has public candidate side information; the
  source sends only the private residual needed to resolve it.
- next experiment change: report direct diagnostic packets as side-information
  coding, not full latent transfer.
- role: theory support.

## DISCUS / Distributed Source Coding Using Syndromes

- source: https://doi.org/10.1109/TIT.2002.808103
- blocker helped: compact source-private identifiers need a coding-theory
  analogue.
- mechanism idea: send a syndrome-like index that becomes useful only with
  correlated decoder side information.
- next experiment change: frame diagnostic packets as exact low-rate syndrome
  keys over a public candidate table.
- role: theory support / framing.

## Consistency Models

- source: https://proceedings.mlr.press/v202/song23a.html
- blocker helped: learned receiver branch should eventually recover the direct
  diagnostic behavior under corruption.
- mechanism idea: learn one-step maps from corrupted/noisy states to a clean
  endpoint.
- next experiment change: keep the failed balanced learned-syndrome probe as a
  target for a stronger learned denoising receiver.
- role: method inspiration / future branch.

## TurboQuant

- source: https://arxiv.org/abs/2504.19874
- blocker helped: byte-scale packet claims need comparison against strong
  quantization systems work.
- mechanism idea: vector state can be rotated/quantized/residually corrected,
  but task-level source causality still needs destructive controls.
- next experiment change: treat quantization as a codec baseline/future bridge,
  not as evidence that public-only semantics are controlled.
- role: systems/compression baseline.

## C2C / Cache-to-Cache

- source: https://arxiv.org/abs/2510.03215
- blocker helped: distinguishes our low-rate private packet lane from
  high-rate KV/cache communication.
- mechanism idea: C2C transfers projected source KV/cache state into the target
  model; LatentWire sends a far-left-rate task packet.
- next experiment change: compare assumptions and boundary traffic, not claim
  C2C is beaten without native cache runs.
- role: closest adjacent baseline / claim boundary.

## Bottom Line

The balanced diagnostic gate should be framed as a rigorous source-causality
test. It removes obvious public artifacts and shows that a tiny private packet
adds information a public-only receiver cannot recover.
