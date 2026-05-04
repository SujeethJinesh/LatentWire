# 2026-05-04 Receiver-Calibrated Top1/Top2 Ambiguity-Code References

## Purpose

This memo records the novelty boundary for the failed receiver-calibrated
top1/top2 ambiguity-code gate. The result should be framed as a negative
ablation of shallow sparse/common-basis packet communication, not as evidence
for universal latent alignment or SAE-based cross-model communication.

## Primary Boundary Sources

- HellaSwag benchmark: `https://arxiv.org/abs/1905.07830`.
- Prefix-Tuning: `https://aclanthology.org/2021.acl-long.353/`.
- Prompt Tuning: `https://aclanthology.org/2021.emnlp-main.243/`.
- Gist tokens: `https://arxiv.org/abs/2304.08467`.
- AutoCompressors: `https://arxiv.org/abs/2305.14788`.
- ICAE / In-context Autoencoder: `https://arxiv.org/abs/2307.06945`.
- Relative Representations: `https://openreview.net/forum?id=SrC-nwieGJ`.
- Sparse Autoencoders for language-model features:
  `https://arxiv.org/abs/2309.08600`.
- Universal Sparse Autoencoders: `https://arxiv.org/abs/2502.03714`.
- Sparse crosscoders/model diffing:
  `https://transformer-circuits.pub/2024/crosscoders/index.html`.
- Selective Classification: `https://arxiv.org/abs/1705.08500`.
- Learning to Defer: `https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer`.
- Universal Logit Distillation: `https://arxiv.org/abs/2402.12030`.
- Proxy-tuning: `https://arxiv.org/abs/2401.08565`.
- C2C cache-to-cache communication: `https://arxiv.org/abs/2510.03215`.
- KVComm: `https://openreview.net/forum?id=F7rUng23nw`.
- KVCOMM: `https://arxiv.org/abs/2510.12872`.
- QJL: `https://arxiv.org/abs/2406.03482`.
- TurboQuant: `https://arxiv.org/abs/2504.19874`.
- KIVI: `https://arxiv.org/abs/2402.02750`.

## Novelty Boundary

The ambiguity-code packet is distinct from prompt/prefix/gist/context
compression only if it is per-example source-model evidence rather than a
persistent or text-derived soft prompt. It is distinct from relative
representations, SAEs, and crosscoders only if sparse/common-basis atoms become
a causal communication alphabet under source-destroying controls. It is
distinct from logit fusion, source-rank communication, and selective prediction
only if the sparse atom improves beyond top1/top2-only, source-score, rank-only,
target-derived, and label-copy controls.

This run does not meet that bar.

## Resulting Claim

Safe claim:

> A receiver-calibrated 1B top1/top2 plus sparse-atom packet fails to improve
> over packet-only on HellaSwag validation `1024:2048`, despite a large source
> top1/top2 oracle.

Unsafe claim:

> Sparse common-basis atoms form a working cross-model latent language.

The latter is not supported by the artifact.
