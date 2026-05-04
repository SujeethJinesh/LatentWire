# HellaSwag Qwen-To-Phi Protected Rival References

Date: 2026-05-04

## Purpose

This memo supports the failed protected top-rival packet gate. The safe claim
is narrow: sending the source model's protected hybrid plus top rival exposes a
large candidate-ID oracle, but a small train/select receiver cannot yet harvest
that headroom.

## Primary Sources And Boundaries

| Area | Sources | Boundary For This Gate |
|---|---|---|
| HellaSwag benchmark | Zellers et al., 2019, https://arxiv.org/abs/1905.07830 | HellaSwag is the current MCQ decision surface; do not generalize to open-ended reasoning without additional benchmarks. |
| Communication / shared code | Shannon, 1948, https://www.mpi.nl/publications/item2383162/mathematical-theory-communication | The candidate option IDs are the shared alphabet/basis. This is a framing, not a new information-theory theorem. |
| Decoder side information | Slepian-Wolf, https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources; Wyner-Ziv, https://cir.nii.ac.jp/crid/1360564063947537280 | Phi scores are decoder side information. LatentWire should be described as an empirical source-private packet protocol, not formal optimal coding. |
| Unequal error protection | Borade et al., 2009, https://arxiv.org/abs/0803.2570 | Supports the systems intuition that high-value fields deserve protection; our packet does not introduce UEP theory. |
| Knowledge distillation | Hinton et al., 2015, https://arxiv.org/abs/1503.02531; RankDistil, Reddi et al., 2021, https://proceedings.mlr.press/v130/reddi21a.html | Top-rank/dark-knowledge transfer is prior art. LatentWire differs only in the strict inference-time, byte-scale, source-private packet setting. |
| Private teacher outputs | Papernot et al., 2017, https://arxiv.org/abs/1610.05755; Shao et al., 2024, https://www.nature.com/articles/s41467-023-44383-9 | Private prediction sharing and selective distillation are prior art; the novelty claim cannot be "private teacher labels." |
| Selective prediction / calibration | Geifman and El-Yaniv, 2017, https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html; Guo et al., 2017, https://arxiv.org/abs/1706.04599 | We are not proposing a new calibration or reject-option method. These are baselines for switch/coverage analysis. |
| Top-k and pairwise reranking | Lapin et al., 2015, https://papers.nips.cc/paper_files/paper/2015/hash/0336dcbab05b9d5ad24f4333c7658a0e-Abstract.html; PairReranker, Jiang et al., 2022, https://arxiv.org/abs/2212.10555 | Top-k/reranking is crowded prior art. The protected-rival packet is unique only as a source-private cross-model communication primitive. |
| Prefix/gist tokens | Prefix tuning, https://aclanthology.org/2021.acl-long.353/; Gist tokens, https://arxiv.org/abs/2304.08467 | LatentWire packets are not learned soft prompts or human-readable prompt compression. |
| Cross-model latent/KV communication | C2C, https://openreview.net/forum?id=LeatkxrBCi and https://arxiv.org/abs/2510.03215; KVComm, https://openreview.net/forum?id=F7rUng23nw and https://arxiv.org/abs/2510.03346 | These are close competitors for hidden/KV communication. This gate is an extreme-rate discrete packet, not KV transfer. |
| Quantized systems baselines | QJL, https://arxiv.org/abs/2406.03482; KIVI, https://arxiv.org/abs/2402.02750; KVQuant, https://arxiv.org/abs/2401.18079; TurboQuant, https://arxiv.org/abs/2504.19874; TurboQuant OpenReview, https://openreview.net/forum?id=tO3ASKZlok | These motivate protected/mixed-precision packet fields but remain KV/vector compression baselines. Do not claim native systems wins without GPU rows. |
| Serving systems | FlashAttention, https://arxiv.org/abs/2205.14135; vLLM/PagedAttention, https://arxiv.org/abs/2309.06180; SGLang, https://arxiv.org/abs/2312.07104 | Future systems comparisons must use native serving rows. Mac-local rows support byte/exposure accounting only. |

## Claim After This Gate

Safe:

- A `2B` raw / `5B` framed protected-rival packet was tested under strict
  source-private controls and did not beat fixed hybrid.
- The candidate-ID shared basis has large oracle headroom:
  hybrid-rival oracle `0.678385` versus fixed hybrid `0.467448`.
- The failure localizes the bottleneck to receiver-side pair choice rather
  than merely missing rival transmission.

Unsafe:

- Claiming positive cross-model latent reasoning.
- Claiming a new top-k/reranking/calibration method.
- Claiming novelty over knowledge distillation or private teacher-output
  sharing.
- Claiming systems superiority over C2C, KVComm, QJL, KIVI, KVQuant,
  TurboQuant, vLLM, SGLang, or FlashAttention without native measurements.

## Next Literature Need

Before another Qwen-to-Phi receiver attempt, review official-train calibration,
learning-to-rank with calibrated pairwise utilities, and conformal/selective
risk guarantees for paired decision rules. The method need is not another
hand-coded top-rival field; it is a stable larger-data utility model or a new
receiver interface.
