# JEPA Anti-Collapse References

Date: `2026-04-27`

## Blocker Helped

The current blocker is not generic representation quality. It is specific:
source-to-target side information must avoid collapse into target-cache priors,
fixed slots, source final-answer leakage, shuffled-source controls, or harmful
target perturbations. JEPA-style predictive latent objectives are useful only if
they become source-destroying controls and anti-collapse telemetry in the
LatentWire harness.

## Sources And Implications

### I-JEPA

- Source: Assran et al., "Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture"
- URL: https://arxiv.org/abs/2301.08243
- Role: inspiration, ablation, paper framing
- Mechanism: predict latent target-block representations from an informative
  context block instead of reconstructing pixels.
- Experiment change: train a source-to-target predictor on answer-masked source
  views and frozen target latent/KV targets; require matched source to beat
  shuffled and zero-source predictors on answer-likelihood, not just latent MSE.

### V-JEPA

- Source: Bardes et al., "Revisiting Feature Prediction for Learning Visual
  Representations from Video"
- URL: https://arxiv.org/abs/2404.08471
- Role: inspiration, ablation, systems framing
- Mechanism: feature prediction alone can learn useful video representations
  without pixel reconstruction, negatives, text, or pretrained image encoders.
- Experiment change: use hard target-state masking in the receiver so a bridge
  must fill missing target-side KV/hidden slots from source state; compare
  matched source against shuffled-source and target-only slot controls.

### LeJEPA / SIGReg

- Source: Balestriero and LeCun, "LeJEPA: Provable and Scalable
  Self-Supervised Learning Without the Heuristics"
- URL: https://arxiv.org/abs/2511.08544
- Role: anti-collapse regularizer, diagnostics
- Mechanism: pair JEPA prediction with sketched isotropic Gaussian
  regularization, targeting complete and dimensional collapse without relying
  on teacher-student heuristics.
- Experiment change: add sideinfo effective-rank, per-dimension variance,
  covariance off-diagonal, and sketched-isotropy telemetry to source sidecars;
  test a SIGReg-style penalty only after answer-unexplained target-pool headroom
  exists.

### LLM-JEPA

- Source: Huang, LeCun, and Balestriero, "LLM-JEPA: Large Language Models Meet
  Joint Embedding Predictive Architectures"
- URLs: https://openreview.net/forum?id=qp4vlNfoB8 and
  https://arxiv.org/abs/2509.14252
- Role: language-domain inspiration, ablation
- Mechanism: formulate JEPA-style latent prediction for LLMs rather than pure
  input-space reconstruction.
- Experiment change: fit normalized/cosine source-to-target hidden/KV predictors
  and compare against current ridge/L2 bridges at matched bytes; report collapse
  telemetry and source-destroying controls.

### VL-JEPA

- Source: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
- URLs: https://openreview.net/forum?id=tjimrqc2BU and
  https://arxiv.org/abs/2512.10942
- Role: output-interface inspiration
- Mechanism: predict continuous target-text embeddings instead of
  autoregressively generating tokens at every step.
- Experiment change: add an answer-embedding/verifier diagnostic before full
  decoding: source sideinfo should move the target latent toward the correct
  answer embedding more than shuffled-source or target-only sideinfo.

### M3-JEPA

- Source: "M3-JEPA: Multimodal Alignment via Multi-gate MoE based on the
  Joint-Embedding Predictive Architecture"
- URL: https://openreview.net/forum?id=tYwKQMMjJA
- Role: multimodal-collapse analogy
- Mechanism: JEPA-style multimodal alignment is framed around reducing modality
  collapse and aligning heterogeneous embeddings.
- Experiment change: use multi-gate source-fault routing only as a control-aware
  receiver: gates must route matched answer-masked source differently from
  zero/shuffled/slots-only source, and preserve target-correct IDs.

### VICReg And Barlow Twins

- Sources: VICReg https://arxiv.org/abs/2105.04906 and Barlow Twins
  https://proceedings.mlr.press/v139/zbontar21a
- Role: anti-collapse regularizer, diagnostics
- Mechanisms: VICReg explicitly penalizes low per-dimension variance and high
  covariance; Barlow Twins reduces redundancy through cross-correlation
  matching.
- Experiment change: for dual answer-masked source views, report sideinfo
  variance floor, effective rank, covariance off-diagonal mass, Barlow diagonal
  and off-diagonal terms, and matched-vs-control answer-likelihood margin.

## Next Experiment Consequence

Do not train another connector merely to improve latent MSE. The next connector
branch should wait until a discovery surface has `answer_unexplained_clean_in_pool
> 0`. Once that exists, train a rate-capped source-memory connector with:

- stop-grad or frozen target latent targets
- answer-masked dual source views
- matched-source margin over zero/shuffled/target-only/slots-only controls
- target-preservation loss on target-correct IDs
- VICReg/Barlow/SIGReg collapse telemetry
- explicit byte/rate reporting

Until then, JEPA is a harness/design constraint, not a headline method.
