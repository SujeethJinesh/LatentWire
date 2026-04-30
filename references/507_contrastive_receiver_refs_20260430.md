# Contrastive Receiver References

- date: `2026-04-30`
- purpose: primary-source grounding for the source-control contrastive receiver
  smoke and the next query-resampler / information-bottleneck receiver gate.

## Contrastive Predictive Coding / InfoNCE

- source: https://arxiv.org/abs/1807.03748
- blocker helped: the frozen receiver needs an output-aware objective, not only
  post-hoc candidate feature alignment.
- mechanism idea: learn a score that distinguishes the correct future/paired
  representation from negatives; in this repo that becomes
  `score(candidate, source_packet)` with shuffled-source negatives.
- next experiment change: keep source-destroying negatives in the training
  objective, not only in evaluation.
- role: method inspiration / objective support.

## CLIP-Style Paired Contrastive Learning

- source: https://arxiv.org/abs/2103.00020
- blocker helped: motivates paired scoring between heterogeneous views while
  keeping each encoder frozen or lightly adapted.
- mechanism idea: treat source-private packets and target-visible candidates as
  paired views and train a compact receiver to rank the matched candidate over
  distractors.
- next experiment change: replace scalar ridge scoring with a low-rank or
  query-resampler contrastive receiver over frozen features.
- role: method inspiration / baseline framing.

## I-JEPA / Representation Prediction

- source: https://arxiv.org/abs/2301.08243
- blocker helped: the current semantic-anchor receiver is too symbolic; the
  next receiver should predict target-useful candidate state under a bottleneck
  rather than reconstruct source text.
- mechanism idea: train a predictor from visible context plus packet into a
  target representation, with collapse-preventing negatives or masking.
- next experiment change: test a query-resampler / representation-prediction
  receiver with a rate-distortion sweep.
- role: inspiration / paper framing.

## BLIP-2 Q-Former

- source: https://arxiv.org/abs/2301.12597
- blocker helped: cross-model communication needs a learned connector rather
  than a brittle global latent alignment.
- mechanism idea: use a small set of learned query vectors to extract the
  source-private information needed by the target interface.
- next experiment change: the next receiver should use a tiny query bottleneck
  or low-rank bilinear factorization with explicit byte/rate accounting.
- role: architecture inspiration.

## Flamingo / Perceiver Resampler

- source: https://arxiv.org/abs/2204.14198
- blocker helped: gives a scalable precedent for compressing one model's
  high-volume state into a fixed number of learned latent tokens before another
  model consumes it.
- mechanism idea: rate-cap source communication by the number and precision of
  learned resampler latents, then plot accuracy against bytes.
- next experiment change: use fixed query count as the next rate-distortion
  axis after the current packet-byte axis.
- role: architecture inspiration / systems framing.

## QJL and TurboQuant

- source: https://arxiv.org/abs/2406.03482
- source: https://arxiv.org/abs/2504.19874
- blocker helped: reviewers will ask whether tiny packets are just a weak
  version of modern quantized state transport.
- mechanism idea: compare against random projections, sign/low-bit packets,
  residual coding, and explicit byte-rate frontiers.
- next experiment change: keep reporting same-byte random/sign controls and
  byte floors for KV/cache-style transfer.
- role: baseline / systems comparison.

## Bottom Line

The literature supports the direction but not a claim yet: contrastive scoring
and query bottlenecks are plausible receiver mechanisms, but the BGE smoke shows
that source-destroying negatives must be part of training before frozen-feature
receivers can be promoted.
