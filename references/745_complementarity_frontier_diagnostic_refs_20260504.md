# Complementarity-Frontier Diagnostic References

Date: `2026-05-04`

Purpose: reference and novelty-boundary memo for the HellaSwag
complementarity-frontier diagnostic. The diagnostic asks whether a low-rate,
source-private packet can decide when source evidence should override a target
or fixed-hybrid baseline, without collapsing into model routing, selective
deferral, source-rank shortcuts, or dense KV transfer.

## Selective Prediction And Deferral Risk

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks,"
  arXiv 2017.
  URL: https://arxiv.org/abs/1705.08500

  Use: risk/coverage framing for selective packet application. Novelty risk:
  if LatentWire only rejects or abstains based on target confidence, it is
  selective prediction rather than source-private communication.

- Geifman and El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated
  Reject Option," ICML 2019.
  URL: https://proceedings.mlr.press/v97/geifman19a.html

  Use: learned reject/select heads and risk-coverage curves. Novelty risk:
  packet gates must be compared against target-derived confidence gates and
  source-free selective predictors.

- Madras et al., "Predict Responsibly: Improving Fairness and Accuracy by
  Learning to Defer," NeurIPS 2018.
  URL: https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer

  Use: classifier-plus-deferral setup for combining systems with different
  strengths. Novelty risk: Qwen-to-Phi packet switching can be interpreted as
  deferring to the source unless the target output changes because of
  source-causal packet evidence under destructive controls.

- Mozannar and Sontag, "Consistent Estimators for Learning to Defer to an
  Expert," ICML 2020.
  URL: https://proceedings.mlr.press/v119/mozannar20b.html

  Use: consistent surrogate for learning classifier/rejector systems. Novelty
  risk: source-index, source-rank, and source-score baselines are mandatory if
  the method behaves like defer-to-source.

- Hemmer et al., "Learning to Defer with Limited Expert Predictions," AAAI
  2023.
  URL: https://arxiv.org/abs/2304.07306

  Use: low-label deferral training where expert predictions are scarce. Novelty
  risk: a small train/select split can overfit deferral boundaries; paired
  held-out uncertainty and destructive controls are necessary.

## Model Routing Competitors

- Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing,"
  ICLR 2024.
  URL: https://proceedings.iclr.cc/paper_files/paper/2024/hash/b47d93c99fa22ac0b377578af0a1f63a-Abstract-Conference.html

  Use: quality-aware routing between model sizes. Novelty risk: LatentWire must
  show more than "choose the model likely to be correct"; it needs packet
  utility under fixed receiver identity and source-private controls.

- Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," ICLR
  2025.
  URL: https://arxiv.org/abs/2406.18665

  Use: routing LLM requests by preference/cost trade-offs. Novelty risk:
  complementarity-frontier plots should include router/source-choice baselines
  and report bytes/privacy exposure.

## Decoder Side Information And Packet Framing

- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder," IEEE Transactions on Information Theory 1976.
  URL: https://doi.org/10.1109/TIT.1976.1055508

  Use: theoretical analogy for source packets that encode residual information
  relative to receiver-side state. Boundary: LatentWire does not claim a
  theorem; this motivates utility-per-byte and decoder-side-information
  accounting.

- Yilmaz et al., "Distributed Deep Joint Source-Channel Coding with
  Decoder-Only Side Information," ICMLCN 2024.
  URL: https://arxiv.org/abs/2310.04311

  Use: neural decoder-side-information design precedent. Boundary: this is
  image/channel coding, not LLM source-private communication; the analogy is
  receiver-side state integration.

## Dense Communication Competitors

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," ICLR 2026.
  URL: https://arxiv.org/abs/2510.03215
  Code: https://github.com/thu-nics/C2C

  Use: closest dense high-bandwidth positive baseline. Boundary: C2C projects
  and fuses source KV cache; LatentWire should compete on source privacy,
  auditability, packet bytes, and destructive controls, not raw accuracy until
  we run C2C directly.

- Shi et al., "KVComm: Enabling Efficient LLM Communication Through Selective
  KV Sharing," arXiv 2025.
  URL: https://arxiv.org/abs/2510.03346

  Use: selective KV-sharing communication baseline. Boundary: KVComm reduces
  cache transfer but still transmits KV state rather than byte-scale auditable
  source-private task packets.

## Current Decision Boundary

The complementarity-frontier diagnostic shows a real source/target opportunity:
source top1/top2 could repair many fixed-hybrid errors. But the selected
frontier made zero held-out overrides, so the existing source top1/top2 plus
margin/entropy packet fields do not expose a stable decision surface.

A richer cached-policy packet follow-up also fails: the multi-signal packet
selector reaches `0.455729` versus the fixed hybrid's `0.467448`, with CI95 low
`-0.023470`, despite the fixed-or-source top1/top2 oracle staying at
`0.694010`. This weakens cached Qwen policy-prediction packets as a repair
frontier; the issue is not only that the first selector was too conservative.

Next gate:

- Do not train another HellaSwag selector on the same packet fields.
- Do not continue cached hidden/score/vote policy-prediction packets on this
  slice without a qualitatively new source-causal feature.
- Add a genuinely new source-causal feature, or move to a benchmark where
  complementarity is separable from rank/score shortcuts.
- Main-table comparators must include target-only, fixed hybrid, source-index,
  source-rank, source-score, same-byte, target-derived, wrong-row, and
  candidate-roll controls.
