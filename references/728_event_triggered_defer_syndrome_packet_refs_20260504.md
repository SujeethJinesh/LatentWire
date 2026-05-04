# Reference Memo 728: Event-Triggered Defer And Residual-Syndrome Packet Gates

Date: 2026-05-04

## Current Paper Status

Paper readiness: not ICLR-ready. COLM_v1 is frozen, COLM_v2/ICLR are centered
on Sparse Resonance Packets, and the newest learned defer gate still failed to
beat source-choice and same-byte controls.

Current story: the defensible SRP novelty is a source-private, rate-capped
packet interface with a receiver that can suppress harmful source information.

Exact blocker: candidate-identity packets, even with learned defer, do not
produce positive paired lift over target-only and shortcut controls.

## Selective Prediction, Calibration, And Learning To Defer

- Geifman and El-Yaniv, 2017, "Selective Classification for Deep Neural
  Networks."
  Source: https://arxiv.org/abs/1705.08500
  Use: risk-coverage framing for packet firing.

- SelectiveNet, Geifman and El-Yaniv, 2019.
  Source: https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf
  Use: coverage and selective-risk definitions.

- Guo et al., 2017, "On Calibration of Modern Neural Networks."
  Source: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf
  Use: ECE/MCE, reliability diagrams, NLL/Brier, temperature scaling.

- Romano, Sesia, and Candes, 2020, "Classification with Valid and Adaptive
  Coverage."
  Source: https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
  Use: target conformal candidate sets as decoder side information.

- Angelopoulos et al., 2024, "Conformal Risk Control."
  Source: https://arxiv.org/abs/2208.02814
  Use: finite-sample risk control for packet-firing thresholds.

- Learn then Test.
  Source: https://arxiv.org/abs/2110.01052
  Use: calibrating predictive algorithms to satisfy explicit risk constraints.

- Madras, Pitassi, and Zemel, 2018, "Predict Responsibly: Improving Fairness
  and Accuracy by Learning to Defer."
  Source: https://papers.neurips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer.pdf
  Use: deferral-rate and system-accuracy framing.

- Learning-to-defer with limited expert predictions.
  Source: https://arxiv.org/abs/2304.07306
  Use: deferral when expert labels are sparse/costly.

## Event-Triggered Control And Communication

- Tabuada, 2007, "Event-Triggered Real-Time Scheduling of Stabilizing Control
  Tasks."
  Source: https://dblp.org/rec/journals/tac/Tabuada07.html
  Use: act/communicate only when state warrants intervention.

- Heemels, Johansson, and Tabuada, 2012, "An Introduction to Event-Triggered
  and Self-Triggered Control."
  Source: https://doi.org/10.1109/CDC.2012.6425820
  Use: trigger-rate and event-conditioned action framing.

- Event-triggered MMSE state estimation with confidence levels.
  Source: https://arxiv.org/abs/2403.15289
  Use: confidence-triggered communication-rate analogy.

- Adaptive Computation Time.
  Source: https://arxiv.org/abs/1603.08983
  Use: allocate action/compute conditionally by difficulty.

## Side-Information And Residual/Syndrome Coding

- Wyner and Ziv, 1976.
  Source: https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  Use: decoder-side-information rate-distortion framing.

- Slepian and Wolf, 1973.
  Source: https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
  Use: compress correlated source evidence without sending the full source
  state.

- DISCUS / distributed source coding using syndromes.
  Source: https://doi.org/10.1109/TIT.2002.808103
  Use: practical syndrome/coset code analogy.

- LDPC coset codes for Slepian-Wolf.
  Source: https://arxiv.org/abs/cs/0607021
  Use: practical code construction for residual parity packets.

- ECOC.
  Source: https://arxiv.org/abs/cs/9501101
  Use: candidate codewords and destructive bit controls.

## Benchmarks And Reviewer-Risk Controls

- ARC:
  https://arxiv.org/abs/1803.05457

- OpenBookQA:
  https://arxiv.org/abs/1809.02789

- HellaSwag:
  https://arxiv.org/abs/1905.07830

- Option-ID / MCQ selector bias:
  https://arxiv.org/abs/2309.03882

- Option-order sensitivity:
  https://arxiv.org/abs/2308.11483

- Choices-only artifact risk:
  https://arxiv.org/abs/2402.12483

- Benchmark leakage:
  https://arxiv.org/abs/2404.18824

- Paired significance in NLP:
  https://aclanthology.org/P18-1128/

Required controls for the next gate: source-index, source-rank, source-score,
source-score quantization, same-byte visible text, zero-source, wrong-row,
label/candidate roll, target-derived syndrome, source-family substitution,
paired bootstrap, helps/harms, risk-coverage, and option-order/label audits
before paper-facing claims.

## Competitor Byte And Exposure Baselines

- C2C:
  https://openreview.net/forum?id=LeatkxrBCi

- KVComm:
  https://openreview.net/forum?id=F7rUng23nw

- KIVI:
  https://proceedings.mlr.press/v235/liu24bz.html

- KVQuant:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html

- TurboQuant:
  https://arxiv.org/abs/2504.19874

- CacheGen:
  https://arxiv.org/abs/2310.07240

Use the shared byte formula:

```text
KV_bytes = tokens * layer_fraction * L * 2 * H_kv * d_head * bits_per_element / 8
```

and clearly distinguish source-private task packets from source-KV-exposing
communication.

## Novelty Boundary

Occupied: sparse bases, SAEs, crosscoders, transcoders, LLM routing,
learning-to-defer, C2C/KV sharing, and candidate source-choice packets.

Still alive: residual/syndrome packets that send only source evidence not
already recoverable from target logits, decoded with target side information,
under strict source-private destructive controls and utility-per-byte reporting.
