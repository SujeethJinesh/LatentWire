# Binary Receiver Deranged-Control and Next-Branch References

- date: `2026-04-30`
- purpose: references for the cross-family binary-verifier deranged-table gate,
  systems/quantization comparison boundary, and the next less-protocol-shaped
  receiver branches.

## TurboQuant

- source: https://arxiv.org/abs/2504.19874
- blocker helped: prevents overclaiming that tiny packets beat modern KV/state
  compression.
- mechanism idea: rotation plus quantization and residual correction compress
  vector/KV state when cache exposure is allowed.
- next experiment change: keep TurboQuant-style bit floors in the systems
  table; compare access assumptions and source-state exposure, not only bytes.
- role: systems/compression baseline.

## QJL

- source: https://arxiv.org/abs/2406.03482
- blocker helped: one-bit random-projection sketches are a strong mathematical
  comparator for compact latent/residual transport.
- mechanism idea: Johnson-Lindenstrauss projection plus sign quantization with
  inner-product recovery.
- next experiment change: any latent/residual packet must beat QJL/sign-sketch
  controls under the same source-destroying gates.
- role: mathematical baseline / ablation.

## KIVI

- source: https://arxiv.org/abs/2402.02750
- blocker helped: naive fp16 KV byte floors are weak.
- mechanism idea: asymmetric 2-bit KV-cache quantization.
- next experiment change: use KIVI-style compressed KV floors for same-model or
  source-KV-exposed settings.
- role: systems baseline.

## KVQuant

- source: https://arxiv.org/abs/2401.18079
- blocker helped: stronger KV compression baselines use outlier-aware and
  pre-RoPE quantization, not uniform quantization.
- mechanism idea: per-channel key quantization plus dense-sparse outlier
  handling.
- next experiment change: future cache/latent comparisons should include
  outlier-aware KV byte floors.
- role: systems/compression baseline.

## FlashAttention

- source: https://arxiv.org/abs/2205.14135
- blocker helped: raw source-packet bytes are not the same as realized hardware
  IO.
- mechanism idea: IO-aware algorithms separate arithmetic from memory traffic.
- next experiment change: continue reporting raw bytes, 64B cache-line bytes,
  128B DMA bytes, and batch-packed packet bytes.
- role: systems framing / overclaim guard.

## vLLM / PagedAttention

- source: https://arxiv.org/abs/2309.06180
- blocker helped: mature serving stacks define the relevant KV/cache baseline.
- mechanism idea: paged KV blocks, sharing, and scheduler-level batching.
- next experiment change: future NVIDIA run should evaluate packet/text/KV
  rows under a serving scheduler rather than a one-off script.
- role: serving baseline.

## DistServe

- source: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- blocker helped: reviewers need TTFT, TPOT, SLO attainment, and goodput, not
  only p50 local latency.
- mechanism idea: disaggregate prefill/decode and report serving SLO metrics.
- next experiment change: native server validation must report TTFT, TPOT,
  p95, goodput, and HBM/KV counters.
- role: systems metric standard.

## GainSight

- source: https://arxiv.org/abs/2504.14866
- blocker helped: hardware-facing story needs data lifetime/locality, not a
  toy byte count.
- mechanism idea: profile fine-grained traffic through memory hierarchy and
  data lifetime.
- next experiment change: add a packet trace card: object lifetime, reuse,
  exposure, line/burst traffic, and whether source text/KV crosses boundary.
- role: hardware-facing systems framing.

## Diffusion Transformers

- source: https://arxiv.org/abs/2212.09748
- blocker helped: current binary receiver is too hand-shaped.
- mechanism idea: process latent tokens with a small transformer and denoising
  objective.
- next experiment change: if pursued, use a tiny candidate-token denoiser over
  packet atoms, not a large image DiT.
- role: architecture inspiration.

## Consistency Models

- source: https://proceedings.mlr.press/v202/song23a.html
- blocker helped: need one-step learned decoding from corrupted/partial packet
  views.
- mechanism idea: train corrupted views to map to the same endpoint; train
  destructive views to preserve target prior.
- next experiment change: run n500 masked-consistency with public-only
  separation, then try a posterior-consistency receiver if it survives.
- role: objective design / next method.

## Flow Matching

- source: https://arxiv.org/abs/2210.02747
- blocker helped: static selectors do not model the path from target prior to
  source-conditioned posterior.
- mechanism idea: learn a vector field from prior logits/noisy candidate state
  to source-informed posterior.
- next experiment change: a tiny 4-candidate posterior-flow receiver is
  Mac-feasible as an ablation against consistency loss.
- role: objective variant.

## I-JEPA

- source: https://arxiv.org/abs/2301.08243
- blocker helped: reconstructing private logs/text is the wrong objective for
  source-private communication.
- mechanism idea: predict useful latent representations from context, not raw
  observations.
- next experiment change: train receivers to predict decision-relevant
  candidate state with collapse/rank diagnostics.
- role: framing and anti-reconstruction objective.

## V-JEPA

- source: https://arxiv.org/abs/2404.08471
- blocker helped: latent prediction should support downstream action/selection
  rather than surface reconstruction.
- mechanism idea: masked latent prediction with target representations.
- next experiment change: any revived JEPA receiver should include
  source-control negatives and effective-rank telemetry.
- role: inspiration / objective design.

## Flamingo / Perceiver Resampler

- source: https://arxiv.org/abs/2204.14198
- blocker helped: variable source states need a fixed-size target interface.
- mechanism idea: resample source features into a small set of learned latent
  queries.
- next experiment change: query count and hidden dimension can become explicit
  rate axes for less protocol-shaped receivers.
- role: connector architecture.

## BLIP-2 / Q-Former

- source: https://arxiv.org/abs/2301.12597
- blocker helped: need a learned bridge between a frozen producer and frozen
  consumer.
- mechanism idea: learned query tokens extract task-relevant information from a
  frozen encoder.
- next experiment change: only revive Q-Former-like receivers with
  source-destroying negatives and public-only ablations.
- role: learned connector baseline.

## Distributed Indirect Source Coding With Decoder Side Information

- source: https://arxiv.org/abs/2405.13483
- blocker helped: formalizes recovering a task variable from source messages
  and decoder side information.
- mechanism idea: optimize task distortion, not exact source reconstruction.
- next experiment change: report LatentWire as rate-distortion over candidate
  recovery with side information and destructive controls.
- role: theory / claim boundary.

## Bottom Line

The deranged-table binary receiver gate should be framed as source-private
side-information communication. It is useful because the target follows the
source packet only when the public side-information table is correct, and
fails when that table is deranged. The next method contribution must move
beyond exact handle equality: either n500 masked-consistency with public-only
separation, or a tiny posterior-consistency/flow receiver with destructive
views mapped to target prior.
