# HellaSwag Terminal-Tail Stability References

Date: 2026-05-04

## Purpose

This memo records the reference boundary for the 2026-05-04 no-method-change
terminal-tail stability rerun.

## Benchmark

- Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?",
  ACL 2019 / arXiv 1905.07830.
  - arXiv: https://arxiv.org/abs/1905.07830
  - ACL Anthology PDF: https://aclanthology.org/P19-1472.pdf

HellaSwag is the correct hard diagnostic benchmark here because short natural
language answer choices make same-byte text and label-copy controls meaningful
threats. The terminal-tail rerun should therefore remain strict: positive
aggregate accuracy is insufficient if any predeclared jackknife subbag fails
paired uncertainty against the strongest label/source-index/score-only
controls.

## Uncertainty And Controls

- The local runner uses paired bootstrap confidence intervals over identical
  examples for method-control deltas. Bootstrap CIs are a standard way to
  estimate uncertainty by resampling observed data; the local result uses this
  only as an empirical robustness check, not as a formal theorem.
  - BBC-CV discussion of bootstrap CIs for model validation:
    https://link.springer.com/article/10.1007/s10994-018-5714-4

## Novelty Boundary

The terminal-tail rerun does not create a new method. It strengthens the
reviewer-facing evidence boundary:

- safe claim: HellaSwag validation `0:9216` is a strong source-private
  fixed-byte result, and the terminal tail has positive aggregate lift;
- unsafe claim: full HellaSwag validation is solved under the strict
  predeclared gate;
- next method requirement: recover the remaining HellaSwag headroom with a new
  mechanism, not a looser confidence rule.

## Related Systems / Communication Context

The full paper still needs to distinguish fixed-byte source-private packets
from source-state communication and KV-cache compression:

- C2C, direct source/target KV-cache fusion:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm, selective KV sharing:
  https://openreview.net/forum?id=F7rUng23nw
- QJL, 1-bit quantized JL KV-cache compression:
  https://arxiv.org/abs/2406.03482
- KIVI, 2-bit asymmetric KV-cache quantization:
  https://arxiv.org/abs/2402.02750
- TurboQuant, online vector/KV quantization:
  https://openreview.net/forum?id=tO3ASKZlok
