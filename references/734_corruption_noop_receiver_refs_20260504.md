# Corruption-To-No-Op Receiver Reference Refresh

Date: 2026-05-04

## Why This Branch Exists

The behavior-atom packet decoder showed a useful but non-causal lift: matched
packets could help Qwen3 on the strict ARC n16 disagreement slice, but
same-source-choice wrong-row packets, candidate roll, top-atom knockout, and
Qwen-substitution controls remained too competitive. This branch moves the
corruption logic into the residual decoder objective: matched packets should
decode to useful residuals, while zero, wrong-row, target-derived, shuffled,
knocked-out, and candidate-rolled packets should decode to no-op.

## Denoising And Corruption Objectives

- [Stacked Denoising Autoencoders](https://jmlr.csail.mit.edu/papers/v11/vincent10a.html)
  motivate learning useful representations from corrupted inputs. LatentWire's
  receiver uses the opposite target for destructive packets: corruptions should
  collapse to no-op, not reconstruct the matched residual.
- [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) and
  [R-Drop](https://arxiv.org/abs/2106.14448) motivate prediction consistency
  under perturbations. For Sparse Resonance Packets, stability is only desired
  for mild coefficient noise around a matched packet; semantic corruptions and
  row/candidate misalignment should not be invariant.
- [Outlier Exposure](https://arxiv.org/abs/1812.04606) and
  [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759)
  are useful analogies for explicit bad-packet negatives. The SRP receiver
  should learn that destructive packet families are outside the accepted packet
  distribution.

## Selective And Defer-To-Source Framing

- [On Optimum Recognition Error and Reject
  Tradeoff](https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff)
  is the classical reject-option setup.
- [Selective Classification for Deep Neural
  Networks](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks)
  and [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)
  establish risk/coverage reporting requirements. LatentWire must report
  full-slice accuracy, accepted coverage, helps, harms, and paired uncertainty.
- [Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html)
  and [Learning to Defer to Multiple Experts](https://arxiv.org/abs/2210.16955)
  motivate a receiver that applies an external signal only when predicted
  source gain exceeds harm risk.

## Sparse Atom Basis And Novelty Boundary

- [Towards Monosemanticity](https://arxiv.org/abs/2309.08600),
  [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410),
  [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/),
  [Dedicated Feature Crosscoders](https://arxiv.org/abs/2602.11729),
  [Delta-Crosscoder](https://arxiv.org/abs/2603.04426), and
  [Transcoders](https://arxiv.org/abs/2406.11944) make common sparse feature
  bases a crowded area. LatentWire should claim novelty only for source-private
  low-rate packet transfer, receiver no-op controls, and utility-per-byte
  systems evaluation, not for sparse bases by themselves.

## Direct Systems Competitors

- [C2C](https://openreview.net/forum?id=LeatkxrBCi),
  [KVComm](https://openreview.net/forum?id=F7rUng23nw), and
  [Communicating Activations Between Language Models](https://arxiv.org/abs/2501.14082)
  are dense or high-bandwidth activation/cache communication baselines. SRP must
  be framed as lower-rate, more private, and more auditable rather than as a
  drop-in accuracy replacement.
- [TurboQuant](https://arxiv.org/abs/2504.19874),
  [KVQuant](https://arxiv.org/abs/2401.18079),
  [QJL](https://arxiv.org/abs/2406.03482), and
  [KIVI](https://arxiv.org/abs/2402.02750) set strong low-bit KV compression
  baselines. We should compare bytes and memory movement, but avoid GPU
  throughput or energy claims until measured natively.

## Current Diagnostic Consequence

Weighted corruption-no-op receiver training is alive but not sufficient. A
balanced `0.1` corruption weight restores matched lift on ARC n16, but
candidate-roll and top-atom-knockout controls remain competitive, and no-op
residual norms for several destructive packets are still close to matched
norms. That points to a candidate-alignment and atom-causality failure, not a
simple receiver-threshold problem.
