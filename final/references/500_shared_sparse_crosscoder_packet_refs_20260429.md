# Shared Sparse Crosscoder Packet References

This memo supports the shared sparse crosscoder packet gate in
`paper/source_private_shared_sparse_crosscoder_packet_gate_20260429.md`.

## Sources

1. **Sparse Crosscoders / Model Diffing**
   - Source: https://transformer-circuits.pub/2024/crosscoders/index.html
   - Blocker helped: reviewers need to know why shared/private sparse atoms are
     a plausible interface instead of arbitrary labels.
   - Mechanism/design idea: learn or define sparse features that separate
     shared structure from model/source-specific structure.
   - Experiment change: use shared atom IDs plus magnitudes and report private
     atom knockout rather than only reconstruction.
   - Role: theory support and framing.

2. **Cross-Architecture Model Diffing with Crosscoders**
   - Source: https://arxiv.org/abs/2602.11729
   - Blocker helped: cross-family transfer is the current learned-method
     failure mode.
   - Mechanism/design idea: dedicated feature crosscoders can isolate shared
     and architecture-specific features.
   - Experiment change: require bidirectional core -> holdout and holdout ->
     core rows, not a one-way positive.
   - Role: baseline/framing for future learned crosscoder extension.

3. **Universal Sparse Autoencoders**
   - Source: https://arxiv.org/abs/2502.03714
   - Blocker helped: reviewers may ask whether one sparse code can work across
     models/families.
   - Mechanism/design idea: a shared sparse concept space can reconstruct
     activations across multiple models.
   - Experiment change: future learned variant should compare against a shared
     dictionary reconstruction baseline; current gate uses task delta and atom
     knockout instead of reconstruction-only claims.
   - Role: baseline and motivation.

4. **SPARC**
   - Source: https://arxiv.org/abs/2507.06265
   - Blocker helped: atom-ID alignment can be arbitrary if streams do not share
     dimensions.
   - Mechanism/design idea: encourage multiple streams to activate the same
     latent dimensions.
   - Experiment change: motivates fixed shared atom IDs and atom-ID derangement
     as a destructive control.
   - Role: inspiration and ablation support.

5. **SAEBench**
   - Source: https://arxiv.org/abs/2503.09532
   - Blocker helped: sparse-autoencoder proxy metrics can fail to predict
     useful downstream behavior.
   - Mechanism/design idea: evaluate utility and interpretability with task
     probes rather than reconstruction alone.
   - Experiment change: require task accuracy, source-destroying controls, and
     causal atom knockout.
   - Role: evaluation support.

6. **Sparse Autoencoders Can Interpret Randomly Initialized Transformers**
   - Source: https://arxiv.org/abs/2501.17727
   - Blocker helped: sparse atoms may look interpretable even when not
     causally meaningful.
   - Mechanism/design idea: interpretability claims need random and causal
     controls.
   - Experiment change: include random same-byte atom packets, atom-ID
     derangement, and top-atom knockout.
   - Role: threat model.

7. **Relative Representations**
   - Source: https://arxiv.org/abs/2209.15430
   - Blocker helped: latent spaces have gauge/permutation instability.
   - Mechanism/design idea: compare representations through anchor-relative
     coordinates.
   - Experiment change: keep anchor-relative/static sparse packets as a
     baseline, but promote shared sparse atoms only if cross-family controls
     pass.
   - Role: theory and ablation.

8. **Cache-to-Cache**
   - Source: https://arxiv.org/abs/2510.03215
   - Blocker helped: C2C is a stronger dense cross-model communication
     baseline.
   - Mechanism/design idea: transfer projected source KV-cache into a target
     cache with learned gating.
   - Experiment change: the sparse packet should be framed as an extreme-rate,
     interpretable source-private alternative, not raw accuracy superiority.
   - Role: baseline.

9. **QJL**
   - Source: https://arxiv.org/abs/2406.03482
   - Blocker helped: tiny binary packets need systems/quantization grounding.
   - Mechanism/design idea: 1-bit quantized Johnson-Lindenstrauss sketches
     preserve inner products with very low overhead.
   - Experiment change: report exact packet bytes and compare against binary
     sign-sketch packet baselines where relevant.
   - Role: systems inspiration.

10. **Wyner-Ziv Source Coding**
    - Source: https://ieeexplore.ieee.org/document/1055508/
    - Blocker helped: theoretical framing for source-private messages decoded
      with target-side side information.
    - Mechanism/design idea: encode a residual/syndrome rather than a full
      source state when the decoder has side information.
    - Experiment change: treat target candidate atoms as decoder side
      information and source atoms as the private residual.
    - Role: theory support.
