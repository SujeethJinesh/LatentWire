# Query-Bottleneck Residue Predictor References

- date: `2026-04-26`
- status: local memo
- problem: pooled and all-layer source hidden summaries do not predict C2C
  residue sidecars on the strict SVAMP32 source-control surface
- next experiment impact: move from pooled ridge readouts to cross-fitted
  learned query bottlenecks or token/layer-level C2C-residual distillation

## Sources

1. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
   Encoders and Large Language Models
   - link: https://arxiv.org/abs/2301.12597
   - helps with: pooled source summaries may discard task-relevant residue
     signal.
   - mechanism idea: use a small learned Q-Former with fixed query tokens
     cross-attending over source token/layer states, trained to emit residue
     posteriors or target-consumable sidecar tokens.
   - changes next experiment: yes; replace pooled ridge features with a
     cross-fitted learned query bottleneck under identical controls.
   - category: inspiration / baseline

2. Flamingo: a Visual Language Model for Few-Shot Learning
   - link: https://arxiv.org/abs/2204.14198
   - helps with: variable-length source hidden traces need compression without
     flattening token structure away.
   - mechanism idea: Perceiver Resampler-style fixed latent slots over source
     cache tokens, followed by gated sidecar consumption by the target.
   - changes next experiment: yes; sweep residue-query slots such as `8`,
     `16`, and `32`, with same-byte controls.
   - category: inspiration / ablation

3. Perceiver IO: A General Architecture for Structured Inputs and Outputs
   - link: https://arxiv.org/abs/2107.14795
   - helps with: source-derived residue prediction may need output-specific
     querying rather than a single source embedding.
   - mechanism idea: encode source states into latent bottleneck slots and
     decode with output queries for residue moduli, candidate IDs, or target
     error coordinates.
   - changes next experiment: yes; make the sidecar head output-query
     conditioned.
   - category: baseline / architecture

4. Relative Representations Enable Zero-Shot Latent Space Communication
   - link: https://arxiv.org/abs/2209.15430
   - helps with: absolute hidden coordinates may be unstable across
     models/layers even when relational signal exists.
   - mechanism idea: represent each source state by similarities to fixed
     anchors, then feed anchor-relative features into the query bottleneck.
   - changes next experiment: yes, as an ablation; promote only if it beats
     raw source features under controls.
   - category: ablation / theory

5. Deep Variational Information Bottleneck
   - link: https://arxiv.org/abs/1612.00410
   - helps with: current probes lack an explicit rate/predictiveness tradeoff
     and may learn target priors or noise.
   - mechanism idea: train a stochastic source sidecar `z` to preserve
     residue/candidate contrast while penalizing capacity.
   - changes next experiment: yes; add a rate-distortion curve over slots,
     bytes, and KL/noise strength.
   - category: theory / ablation

6. Cross-Tokenizer Distillation via Approximate Likelihood Matching
   - link: https://arxiv.org/abs/2503.20083
   - helps with: cross-family residue prediction can be confounded by
     incompatible token boundaries and likelihood spaces.
   - mechanism idea: align teacher/student predictions through approximate
     likelihood matching over spans rather than shared token IDs.
   - changes next experiment: not first on same-Qwen SVAMP32; use for
     cross-family falsification after a same-family bottleneck clears.
   - category: baseline / cross-family control

## Next Gate

The first summary-token query-bottleneck smoke gate failed on SVAMP32 with
matched `9/32`, target-self `2/3`, and clean source-necessary `0/6`. Do not
scale that exact variant. The remaining literature-backed gate is a full
token/layer or C2C-residual version:

- input: source token/layer states, optionally anchor-relative features
- output: residue factor queries or candidate-contrast queries
- controls: matched, zero-source, shuffled-source, label-shuffle, target-only,
  and slots-only
- promotion rule: matched `>=14/32`, target-self `3/3`, clean
  source-necessary `>=2/6`, numeric coverage `>=31/32`, exact ordered ID
  parity, and clean control union `0/6`
