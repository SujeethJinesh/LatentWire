# Anchor-Relative Sparse Packet Gate References

- date: `2026-04-29`
- blocker: scalar Wyner-Ziv and canonical RASP fail bidirectional cross-family;
  reviewers asked for deeper, more mathematical model-to-model communication
  mechanisms.

## Sources And Experiment Implications

1. **Relative Representations**
   - source: https://arxiv.org/abs/2209.15430
   - blocker helped: absolute latent coordinates are not comparable across
     models or families.
   - mechanism/design idea: express examples by relationships to anchors rather
     than raw coordinates.
   - experiment change: AR-SIP sends sparse candidate-anchor atoms, and includes
     anchor-id permutation as a destroy-the-signal control.
   - role: method inspiration and diagnostic support.

2. **Sparse Crosscoders**
   - source: https://transformer-circuits.pub/2024/crosscoders/
   - blocker helped: need interpretable shared/model-specific latent atoms.
   - mechanism/design idea: transmit top-k sparse source-private atoms rather
     than a dense hidden state.
   - experiment change: AR-SIP packet format sends top sparse anchor ids plus
     quantized weights; future learned version should add atom-permutation
     controls.
   - role: interpretability inspiration.

3. **I-JEPA**
   - source: https://arxiv.org/abs/2301.08243
   - blocker helped: avoid reconstructing source text or copying answers.
   - mechanism/design idea: predict missing target-side state from private
     source evidence.
   - experiment change: AR-SIP is treated as an innovation packet and tested
     with answer-masked source controls.
   - role: objective/framing inspiration.

4. **Diffusion Transformers**
   - source: https://arxiv.org/abs/2212.09748
   - blocker helped: latent patch/state transport can be more useful than raw
     token relay.
   - mechanism/design idea: a packet can be viewed as a small denoising
     innovation over target candidate state.
   - experiment change: only used as design inspiration; the current gate is a
     sparse deterministic smoke, not a diffusion model.
   - role: inspiration.

5. **Representation Autoencoders for Diffusion Transformers**
   - source: https://arxiv.org/abs/2510.11690
   - blocker helped: modern high-dimensional representation latents require a
     learned decoder to be useful.
   - mechanism/design idea: if static sparse packets fail, a learned receiver or
     query bottleneck is the more literature-backed next step.
   - experiment change: AR-SIP failure points toward learned query bottlenecks,
     not more static coordinate tuning.
   - role: future-method framing.

6. **Model Stitching**
   - source: https://arxiv.org/abs/2106.07682
   - blocker helped: frozen networks may require a small learned bridge to
     expose functional representation similarity.
   - mechanism/design idea: static packet decoding may be underpowered without
     a target-side stitch.
   - experiment change: the next cross-family branch should include a small
     receiver/stitch if endpoint systems work is not prioritized.
   - role: theory support and limitation.

7. **BLIP-2 Q-Former / Flamingo**
   - sources: https://arxiv.org/abs/2301.12597 and
     https://arxiv.org/abs/2204.14198
   - blocker helped: strong frozen-model connectors generally use learned query
     bottlenecks.
   - mechanism/design idea: preserve the target while injecting source evidence
     through a tiny gated query interface.
   - experiment change: if AR-SIP remains failed, implement query-bottleneck
     receiver rather than another hand-designed sparse code.
   - role: architecture inspiration.

## Gate Outcome

The static AR-SIP smoke fails bidirectionally. Holdout-to-core passes at some
budgets, but core-to-holdout does not beat target-only and anchor controls can
outperform matched packets. This prunes the shallow claim that relative sparse
coordinates alone solve cross-family communication. It does not prune learned
anchor-relative receivers or Q-Former/Perceiver-style query bottlenecks.
