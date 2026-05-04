# Source-Codebook Candidate Repair References

Date: 2026-05-04

This memo records the literature and novelty boundary after the failed
source-codebook candidate repair gates.

## Closest Non-Novelty Risks

- Prefix-Tuning and Prompt Tuning show that learned continuous task prompts can
  steer frozen LMs:
  https://arxiv.org/abs/2101.00190
  https://arxiv.org/abs/2104.08691
- Gist tokens, AutoCompressors, and ICAE compress context into model-side
  soft tokens or memories:
  https://arxiv.org/abs/2304.08467
  https://arxiv.org/abs/2305.14788
  https://arxiv.org/abs/2307.06945
- DExperts, contrastive decoding, and Proxy-Tuning are important logit-fusion
  / output-distribution baselines. A candidate repair head that simply adds
  source logits to target logits would not be novel:
  https://arxiv.org/abs/2105.03023
  https://arxiv.org/abs/2210.15097
  https://arxiv.org/abs/2401.08565

## Communication And Systems Boundary

- C2C communicates through learned source-to-target KV-cache projection and
  fusion:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm and Q-KVComm communicate or compress KV-cache state:
  https://arxiv.org/abs/2510.12872
  https://arxiv.org/abs/2512.17914
- KIVI, KVQuant, QJL, and TurboQuant are low-bit KV/vector compression
  baselines. They make "quantizing vectors" non-novel by itself:
  https://arxiv.org/abs/2402.02750
  https://arxiv.org/abs/2401.18079
  https://arxiv.org/abs/2406.03482
  https://openreview.net/forum?id=tO3ASKZlok

## Interpretability And Shared-Basis Motivation

- SVCCA, relative representations, sparse autoencoders, and sparse crosscoders
  motivate common-basis and feature-level diagnostics, but do not by
  themselves provide a candidate-repair communication protocol:
  https://arxiv.org/abs/1706.05806
  https://arxiv.org/abs/2209.15430
  https://arxiv.org/abs/2309.08600
  https://arxiv.org/abs/2603.05805

## Boundary After This Gate

Naive source-codebook candidate repair is now weakened. The stronger novelty
claim remains possible only if the learned receiver:

```text
matched source code > fixed source-index/hybrid packet
matched source code > source top-1 shortcut
matched source code > source-row shuffle / candidate roll / target-derived code
```

with paired uncertainty and across larger frozen slices.

The current failures show that source top-1/top-2 contains recoverable
headroom, but simple score-space buckets do not extract it. A stronger method
must avoid degenerating into logit fusion or source-choice copying.
