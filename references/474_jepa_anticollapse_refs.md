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

## 2026-04-27 Committee Addendum

The follow-up JEPA committee readout keeps the same decision boundary after the
Math-7B selected-disagreement scout:

1. Blocker helped: representation collapse is secondary to answer leakage. JEPA
   mechanisms help only if they make the side information source-destroyable and
   answer-masked.
2. Mechanism/design idea: use dual answer-masked source views, predict frozen
   target latent/KV summaries, and train only the source innovation over the
   target baseline rather than reconstructing the entire target state.
3. Next experiment change: no full connector training on the current selected
   surface; matched source still needs a non-leaky target in the candidate pool.
4. Role: inspiration, ablation, anti-collapse diagnostics, and paper framing.

Cheap collapse telemetry to attach to any future JEPA-style sidecar:

- finite score coverage
- per-dimension variance min/mean and variance floor pass rate
- effective rank
- covariance off-diagonal mean absolute value
- route/label entropy and zero-margin rate
- matched-vs-answer-only cosine
- matched-vs-control paired margin distribution

## 2026-04-27 Clean6 Gate Addendum

The SVAMP32 clean6 target-only sampling gate changes the next experiment but not
the promotion status:

1. Blocker helped: target/no-source generation now has strict-small reachable
   headroom (`2/6` clean IDs), but numeric source sidecars select no clean
   correct candidates.
2. Mechanism/design idea: use JEPA as an answer-masked process/latent ranking
   objective over the reachable candidate pool, not as another decoded-number
   sidecar.
3. Next experiment change: restrict the next smoke to the two reachable clean
   IDs, predict frozen target/candidate latent or KV summaries from dual
   answer-masked source views, and require matched source to beat answer-only,
   zero-source, shuffled-source, target-only, slots-only, and random same-byte
   controls.
4. Role: inspiration, anti-collapse diagnostics, ablation design, and paper
   framing. Still not method evidence.

## 2026-04-27 Full32 Reachability Addendum

The SVAMP32 full32 target-only sampling gate keeps JEPA/LeJEPA/V-JEPA as design
guidance, not positive evidence:

1. Blocker helped: broader target/no-source sampling reaches `14/32` raw sample
   oracle and `18/32` merged target-side oracle, but it does not expand C2C-clean
   residual reachability beyond the same two clean IDs from the clean6 gate.
2. Mechanism/design idea: a JEPA-style connector should predict source
   innovation over a frozen target-prior candidate pool, not reconstruct all
   target states or rank target-prior samples by latent similarity alone.
3. Next experiment change: use the full32 no-source pool as the target-prior
   baseline. Train or evaluate only a bounded source-conditioned generator or
   frozen-latent/rate-capped connector with matched-source margins over
   zero-source, shuffled-source, target-only/slots-only, and random same-byte
   controls.
4. Role: inspiration, anti-collapse diagnostics, ablation design, and paper
   framing. The full32 target-only gate is not communication evidence.

## 2026-04-27 Source Sampling Reachability Addendum

The SVAMP32 source-sampling smoke changes the next JEPA-style connector target
but still does not create a method claim:

1. Blocker helped: source sampling reaches two C2C-clean residual IDs beyond the
   target/no-source full32 pool (`6e9745b37ab6fc45`,
   `de1bf4d142544e5b`), giving a sharper surface for a matched-source selector
   or connector.
2. Mechanism/design idea: train a Query-JEPA-style source innovation encoder
   over the source-sampled candidate surface, not a full target-state
   reconstructor. Predict only the residual signal that separates matched
   answer-masked source from zero, shuffled, target-only/slots-only,
   answer-only, answer-masked, and random same-byte controls.
3. Next experiment change: restrict the next smoke to the two new source-only
   C2C-clean residual IDs and require matched-only recovery before any broader
   connector training.
4. Role: inspiration, anti-collapse diagnostics, ablation design, and paper
   framing. The source-sampling gate is discovery evidence, not communication
   evidence.
