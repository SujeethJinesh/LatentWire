# Learned Synonym Dictionary Packet References

- date: `2026-04-29`
- blocker: the hand sparse packet passed native ontology but failed synonym
  stress; the paper needs a learned/calibrated dictionary that is stronger than
  literal phrase rules while staying honest about leakage and calibration.
- role: source memo for
  `paper/source_private_learned_synonym_dictionary_packet_gate_20260429.md`.

## Primary Sources And Experiment Impact

1. **Towards Monosemanticity**
   - Source: https://transformer-circuits.pub/2023/monosemantic-features/
   - Blocker helped: motivates sparse feature dictionaries as interpretable
     interfaces rather than opaque dense vectors.
   - Mechanism/design idea: sparse features should be logged, knocked out, and
     evaluated causally rather than only by reconstruction.
   - Experiment change: report top-feature knockout and random-feature
     knockout for learned dictionary packets.
   - Role: interpretability support.

2. **Sparse Crosscoders**
   - Source: https://transformer-circuits.pub/2024/crosscoders/
   - Blocker helped: motivates shared/private feature splits across
     representation streams.
   - Mechanism/design idea: a receiver can consume a shared dictionary feature
     interface rather than raw text or full cache state.
   - Experiment change: keep the method framed as crosscoder-inspired until a
     neural activation crosscoder is actually trained.
   - Role: method inspiration and caveat.

3. **SAEBench**
   - Source: https://arxiv.org/abs/2503.09532
   - Blocker helped: reconstruction metrics are not enough for sparse-feature
     claims.
   - Mechanism/design idea: use downstream behavior, stability, and causal
     tests.
   - Experiment change: promote only rows with task lift, source controls,
     paired uncertainty, and feature knockout.
   - Role: evaluation support.

4. **I-JEPA**
   - Source: https://arxiv.org/abs/2301.08243
   - Blocker helped: synonym robustness should predict stable latent/candidate
     structure rather than reconstruct surface words.
   - Mechanism/design idea: representation prediction under alternative views.
   - Experiment change: next gate should split synonym clusters and optimize
     invariant candidate features, not just calibrate the current synonym map.
   - Role: method inspiration.

5. **SimCSE**
   - Source: https://aclanthology.org/2021.emnlp-main.552/
   - Blocker helped: learned dictionaries need paraphrase invariance metrics.
   - Mechanism/design idea: contrastive agreement across semantically
     equivalent sentence views.
   - Experiment change: future runs should log feature Jaccard/top-k overlap
     across native and synonym-stressed candidate descriptions.
   - Role: ablation inspiration.

6. **Sentence-BERT**
   - Source: https://arxiv.org/abs/1908.10084
   - Blocker helped: provides a standard paraphrase/semantic embedding baseline
     for target candidate text.
   - Mechanism/design idea: compare learned dictionary calibration against
     sentence-embedding nearest-neighbor candidate matching.
   - Experiment change: future Mac-local comparator can use cached sentence
     embeddings if dependencies are available.
   - Role: baseline inspiration.

7. **End-to-End Optimized Image Compression**
   - Source: https://arxiv.org/abs/1611.01704
   - Blocker helped: packet methods should be evaluated as rate-distortion
     systems, not single operating points.
   - Mechanism/design idea: bitrate penalty plus downstream distortion.
   - Experiment change: keep reporting 4-byte and 8-byte rows separately; do
     not promote 8-byte rows when controls collide.
   - Role: systems/theory framing.

8. **Wyner-Ziv Source Coding**
   - Source: https://doi.org/10.1109/TIT.1976.1055508
   - Blocker helped: the receiver has target-side candidate side information,
     so the packet should encode only private residual evidence.
   - Mechanism/design idea: communicate conditional innovation rather than full
     source trace.
   - Experiment change: source-destroying controls and matched-byte text remain
     mandatory for every learned packet.
   - Role: theory framing.

## Decision

The learned synonym-dictionary gate can be claimed as a calibrated
agreed-protocol contribution. It should not yet be claimed as a discovered
activation-level crosscoder or general semantic latent transfer. The next gate
must hold out synonym clusters or train from model/candidate embeddings without
the exact frozen synonym mapping used in evaluation.
