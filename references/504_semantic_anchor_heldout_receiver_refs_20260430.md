# Semantic Anchor Held-Out Receiver References

- date: `2026-04-30`
- blocker: the learned synonym dictionary passed calibrated synonym stress but
  failed the held-out paraphrase split, so the paper needed a receiver that
  generalizes public candidate meaning without exact transformed-surface
  calibration overlap.
- role: primary-source memo for the semantic-anchor, target-preserving
  held-out receiver gate.

## Primary Sources And Experiment Impact

1. **Relative Representations Enable Zero-Shot Latent Space Communication**
   - Source: https://openreview.net/forum?id=SrC-nwieGJ
   - Blocker helped: raw surface/coordinate features did not transfer to held-
     out paraphrases.
   - Mechanism/design idea: represent candidate meanings through stable public
     anchors rather than absolute feature coordinates.
   - Experiment change: add `semantic_anchor` receiver features and require
     exact transformed held-out surface overlap to remain zero.
   - Role: method inspiration and paper framing.

2. **Sentence-BERT**
   - Source: https://arxiv.org/abs/1908.10084
   - Blocker helped: target-side candidate surfaces need comparable semantic
     representations rather than token-exact matching.
   - Mechanism/design idea: use semantically comparable candidate embeddings /
     anchors as the receiver side information.
   - Experiment change: the current Mac-local implementation uses deterministic
     semantic-anchor features; a future stronger baseline can replace them with
     SBERT-style frozen sentence embeddings.
   - Role: inspiration and future baseline.

3. **SimCSE**
   - Source: https://aclanthology.org/2021.emnlp-main.552/
   - Blocker helped: the receiver should be robust to paraphrases while
     separating unrelated candidates.
   - Mechanism/design idea: contrast paraphrase positives against in-batch
     negatives when moving beyond deterministic anchors.
   - Experiment change: next learned receiver should train paraphrase-positive /
     candidate-negative objectives on public candidate text only.
   - Role: method inspiration and ablation design.

4. **Sparse Crosscoders for Cross-Layer Features and Model Diffing**
   - Source: https://transformer-circuits.pub/2024/crosscoders/index.html
   - Blocker helped: reviewers may view hand anchors as too protocol-shaped.
   - Mechanism/design idea: shared sparse dictionaries can expose interpretable
     features across representations and support causal feature knockout.
   - Experiment change: semantic anchors are a deterministic stand-in; the
     larger next contribution should test a learned sparse crosscoder/anchor
     dictionary with the same source-destroying controls.
   - Role: inspiration and interpretability support.

5. **I-JEPA**
   - Source: https://arxiv.org/abs/2301.08243
   - Blocker helped: held-out transfer should predict latent semantic targets,
     not reconstruct text.
   - Mechanism/design idea: learn receiver-side invariant predicates/anchors
     from public context with masked/corrupted positives.
   - Experiment change: future branch should train a JEPA-style receiver to
     predict candidate anchor relations from source-private traces.
   - Role: method inspiration.

6. **BLIP-2 / Q-Former**
   - Source: https://arxiv.org/abs/2301.12597
   - Blocker helped: the paper needs a path from the current public-anchor
     receiver toward frozen-model connector architectures.
   - Mechanism/design idea: small query bottlenecks can bridge two frozen
     systems while preserving the receiver.
   - Experiment change: next non-hand-coded receiver should be a zero-init or
     target-preserving query bottleneck over source features.
   - Role: method inspiration.

7. **Perceiver IO**
   - Source: https://arxiv.org/abs/2107.14795
   - Blocker helped: source state may be long, multi-layer, or tool-trace-like.
   - Mechanism/design idea: fixed latent queries can compress arbitrary inputs
     into a small task-shaped interface.
   - Experiment change: use Perceiver-style resampling if source activations
     replace diagnostic traces.
   - Role: method inspiration.

8. **QJL and TurboQuant**
   - Sources: https://arxiv.org/abs/2406.03482 and
     https://arxiv.org/abs/2504.19874
   - Blocker helped: reviewers will ask whether sign sketches or low-bit
     vector quantization explain the systems win.
   - Mechanism/design idea: compression baselines matter after source-derived
     information is proven.
   - Experiment change: keep QJL/TurboQuant as systems/compression baselines;
     do not prioritize them over the held-out receiver gate until the method
     clears controls.
   - Role: systems baseline and caveat.

## Decision

The held-out receiver should be framed as a **semantic-anchor, target-preserving
side-information decoder**: the source still sends only a rate-capped private
packet, while the target uses public candidate anchors to decode it. This is
not yet an activation-level latent bridge, but it directly addresses the
previous held-out paraphrase failure and creates a stronger method contribution
than the earlier calibrated dictionary alone.
