# Learned Syndrome Probe References

Date: `2026-04-24`

## Why This Memo Exists

The pooled source-hidden syndrome probe failed on SVAMP32. The next bounded
test was whether a small learned query bottleneck over source token states
could recover the compact C2C-derived residue syndrome while preserving the
same target-candidate controls.

## Primary Sources

- [Cache-to-Cache Communication](https://arxiv.org/abs/2510.03215)
  Direct cross-model cache communication baseline and source of the current
  oracle residue labels.
- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
  Frozen-backbone learned-query bottleneck precedent for extracting compact
  information from one model/interface into another.
- [Flamingo](https://arxiv.org/abs/2204.14198)
  Perceiver-style resampling and gated cross-attention precedent for
  low-interference source conditioning.
- [Set Transformer](https://arxiv.org/abs/1810.00825)
  Attention pooling over variable-size sets; useful for treating source token
  traces as a set/sequence without committing to final-token pooling.
- [Double/Debiased Machine Learning](https://arxiv.org/abs/1608.00060)
  Cross-fitting motivation: out-of-fold predictions reduce small-slice
  overfit risk when using learned nuisance predictors.
- [Self-Refine](https://arxiv.org/abs/2303.17651)
  Candidate-generation plus evaluation/refinement framing for a later verifier
  gate if a source-derived candidate signal exists.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
  Search/evaluation separation precedent for verifier-gated repair surfaces,
  but not a substitute for source-necessary communication evidence.

## Practical Read

The learned query bottleneck is only worth promoting if matched source-token
predictions recover the oracle-clean residual IDs and all source-destroying
controls recover none. If matched remains below target-only or clean recovery
stays at `0/6`, the next live branch should shift toward C2C-residual
distillation or a different source signal rather than adding verifier layers.
