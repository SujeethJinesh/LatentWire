# Conditional Debiasing And Side-Information References

Date: `2026-04-26`

## Why This Memo Exists

The Perceiver answer-teacher branch failed because target-only, slots-only,
zero-source, or shuffled-source controls explain every positive clean margin.
This memo records primary-source motivations for treating the blocker as
conditional coding plus nuisance debiasing, and for pivoting toward source-only
sidecars or matched-vs-control objectives.

## Primary Sources

- [The Conditional Entropy Bottleneck](https://arxiv.org/abs/2002.05379)
  Problem: target-side state can already explain part of the answer.
  Mechanism: compress only the input information necessary beyond a condition.
  Experiment impact: train/evaluate source messages as target-conditioned
  innovations, not source reconstructions. Role: objective theory.
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
  Problem: small slices make connector memorization and over-wide messages too
  easy. Mechanism: stochastic rate-limited predictive bottlenecks. Experiment
  impact: future learned sidecars should sweep rate/noise and report bytes
  against accuracy. Role: architecture regularizer.
- [Unsupervised Domain Adaptation by Backpropagation](https://proceedings.mlr.press/v37/ganin15.html)
  Problem: the communicated representation should preserve task signal while
  discarding nuisance/control identity. Mechanism: gradient reversal against a
  nuisance classifier. Experiment impact: target-only, slot, and shuffle
  recoverability can be made adversarial losses. Role: objective inspiration.
- [On Adversarial Removal of Hypothesis-only Bias in NLI](https://aclanthology.org/S19-1028/)
  Problem: target-prior leakage is analogous to hypothesis-only shortcuts.
  Mechanism: adversarially remove single-side bias while preserving task
  accuracy. Experiment impact: treat target-only/slots-only as first-class
  shortcut baselines and penalize gains that survive them. Role: debiasing
  template.
- [Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/)
  Problem: hidden shortcut signals can dominate benchmark improvements.
  Mechanism: expose artifacts with hypothesis-only baselines. Experiment
  impact: negative controls must appear in the main table, not just appendix
  ablations. Role: evaluation precedent.
- [Noiseless Coding of Correlated Information Sources](https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf)
  Problem: target and source caches are correlated, so the sender should not
  transmit what the decoder already has. Mechanism: distributed source coding.
  Experiment impact: source-only sidecars should transmit syndrome-like
  information that decoder-side state cannot infer alone. Role: theory support.
- [The Rate-Distortion Function for Source Coding with Side Information at the Decoder](https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf)
  Problem: LatentWire needs lossy, task-relevant communication with decoder
  side information. Mechanism: Wyner-Ziv conditional rate-distortion.
  Experiment impact: report rate/bytes curves conditioned on target-side
  candidate pools. Role: theory support.
- [Neural Distributed Image Compression using Common Information](https://arxiv.org/abs/2106.11723)
  Problem: practical neural codecs can use side information available only at
  the decoder. Mechanism: learned compression with common/correlated
  information. Experiment impact: architecture precedent for source-only code
  fused with target-side state only at decode time. Role: adjacent inspiration.

## Practical Read

The next LatentWire branch should avoid forming the transmitted signal from
target-only memory or learned slots. A source-only sidecar/router is the
cleanest falsification: if it recovers clean IDs while zero-source,
shuffled-source, label-shuffle, target-only, and slots-only controls do not,
then the improvement is closer to conditional communication than target-prior
repair.
