# Target Behavior-Transcoder Probe Reference Refresh

Date: 2026-05-04

## Why This Branch Exists

The source-only and paired BatchTopK atom scouts fit the tiny train slice but
did not produce held-out source-private packet utility. This diagnostic asks a
more basic precursor question: before trying to make TinyLlama transmit sparse
atoms, can Qwen target-native sparse atoms causally improve Qwen's own ARC
candidate margins under destructive atom and candidate controls?

This is a target-hidden oracle feasibility probe. It is not a source-private
communication result and should not be presented as one.

## Primary Sources And Novelty Boundary

- [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944)
  motivates replacing raw activations with sparse predictive components that
  can expose feature-level circuits. LatentWire needs source-private packet
  transfer and strict utility-per-byte controls to be distinct.
- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
  uses cross-layer transcoders inside replacement models for attribution graphs.
  This is strong prior art for target-native sparse computational bases, not for
  low-rate cross-model packet communication.
- [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) shows that sparse
  features can be causally ablated or edited for model behaviors. Our target
  probe borrows that causal feature-testing spirit, but the paper claim must
  eventually be communication, not just interpretability.
- [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)
  and [Steering Language Model Refusal with Sparse Autoencoders](https://arxiv.org/abs/2411.11296)
  are activation/feature steering baselines. If LatentWire only adds target-side
  steering, novelty is weak.
- [Causal Language Control in Multilingual Transformers via Sparse Feature Steering](https://arxiv.org/abs/2507.13410)
  is another direct warning that sparse features can control behavior without
  any source model. LatentWire must show source-specific information transfer.
- [Gemma Scope](https://arxiv.org/abs/2408.05147) provides large open SAE
  resources and reinforces that sparse target features are becoming standard
  tooling. Our differentiator should be the packet protocol, destructive
  controls, and systems accounting.
- [TurboQuant](https://arxiv.org/abs/2504.19874) is relevant for the systems
  boundary: dense KV-cache compression can reduce bytes aggressively, so our
  byte/latency claims must compare against quantized dense-cache baselines where
  possible rather than full-precision KV only.

## Probe Outcome

Implemented `scripts/build_arc_challenge_target_behavior_transcoder_probe.py`.
The script fits Qwen target-native BatchTopK behavior atoms from target hidden
public innovations, quantizes sparse top-k atom packets, decodes them through a
target-conditioned residual receiver, and evaluates matched packets against
target-only, target-derived, zero, row-shuffle, atom-shuffle,
coefficient-shuffle, top-atom-knockout, candidate-roll, and
candidate-derangement controls.

Two `.debug/` n8 scouts were run:

| Variant | Packet bytes/row | Matched | Target | Best control | Helps | Harms | Decision |
|---|---:|---:|---:|---|---:|---:|---|
| rank16 top2 q4 | 8 | 0.2500 | 0.3750 | coefficient shuffle, 0.5000 | 0 | 1 | fail |
| rank32 top4 q4 | 18 | 0.2500 | 0.3750 | target-only, 0.3750 | 0 | 1 | fail |

The learned receiver overfits train behavior: the selected gate reaches
`1.0000` train accuracy and fires on every held-out row, but matched packets do
not improve held-out target-only accuracy. A fixed-weight diagnostic also picks
weight `0.0` as best for matched packets in both settings, while a corrupted
coefficient-shuffle packet can outperform matched on the tiny slice. This means
the current target-native atom/residual decoder has no held-out causal lift.

## Consequence

This weakens the target-native sparse behavior-transcoder branch as the next
positive method. It does not prove sparse packets are impossible, but it blocks
the simple story "first validate target atoms, then make source transmit them."

The next promoted branch should change the decision surface rather than scale
this diagnostic unchanged. Highest-value options are:

1. move from answer-margin residual steering to a lower-variance calibration or
   verifier/defer target where sparse packets only decide when to trust the
   source;
2. use a receiver objective with held-out no-harm calibration before selecting
   residual weights;
3. shift back to source-private side information that predicts target error or
   uncertainty, because target-side atom steering alone is not showing lift on
   the Mac-local ARC slice.

