# Consistency Posterior Packet References

- date: `2026-04-29`
- purpose: primary-source memo for the consistency-distilled canonical posterior
  packet branch and the next systems-rate frontier gate

## Sources And Implications

1. **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture**
   - URL: https://arxiv.org/abs/2301.08243
   - Blocker helped: need a method that predicts abstract target-side state
     rather than reconstructing private source text.
   - Mechanism: predict latent target representations from context in embedding
     space.
   - Experiment impact: motivated `consistent_posterior_packet`, where the
     source predicts a smoothed candidate posterior centroid.
   - Role: inspiration/theory support.

2. **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations
   from Video**
   - URL: https://arxiv.org/abs/2404.08471
   - Blocker helped: avoid source-text reconstruction and negative-example
     dependence.
   - Mechanism: feature prediction over masked/perturbed views.
   - Experiment impact: supports source-feature dropout and candidate-negative
     dropout in the posterior-packet ablation.
   - Role: inspiration/ablation support.

3. **Consistency Models**
   - URL: https://proceedings.mlr.press/v202/song23a.html
   - Blocker helped: one-shot/few-step target decoding from a compact packet.
   - Mechanism: map corrupted/noisy states directly to clean endpoints.
   - Experiment impact: supports training a packet that decodes to a stable
     canonical posterior under perturbations.
   - Role: inspiration/theory support.

4. **Latent Consistency Models**
   - URL: https://arxiv.org/abs/2310.04378
   - Blocker helped: distilling a stronger teacher into a compact latent-state
     interface.
   - Mechanism: latent-space consistency distillation.
   - Experiment impact: suggests a future teacher-student packet with a
     full-source posterior teacher; not implemented in the current NumPy gate.
   - Role: inspiration/future ablation.

5. **Trajectory Consistency Distillation**
   - URL: https://arxiv.org/abs/2402.19159
   - Blocker helped: endpoint-only consistency can collapse or miss trajectory
     information.
   - Mechanism: distill whole denoising/solution trajectories.
   - Experiment impact: suggests a future ablation with posterior checkpoints
     instead of one centroid target.
   - Role: inspiration/future ablation.

6. **Relative Representations Enable Zero-Shot Latent Space Communication**
   - URL: https://arxiv.org/abs/2209.15430
   - Blocker helped: latent gauge mismatch across models/representations.
   - Mechanism: represent examples by similarities to anchors.
   - Experiment impact: canonical RASP and posterior packets should remain
     candidate/anchor-relative rather than raw hidden-state-relative.
   - Role: theory/framing.

7. **QJL: Quantized Johnson-Lindenstrauss Representations**
   - URL: https://arxiv.org/abs/2406.03482
   - Blocker helped: reviewers need a principled low-bit sketch baseline.
   - Mechanism: random projection plus sign/quantized sketches for inner-product
     preservation.
   - Experiment impact: keep raw sign sketch and QJL/TurboQuant-style rows as
     matched-byte comparator baselines.
   - Role: baseline/theory support.

8. **TurboQuant**
   - URL: https://arxiv.org/abs/2504.19874
   - Blocker helped: packet claims need modern quantization baselines, not weak
     strawmen.
   - Mechanism: random rotations plus scalar quantization and QJL-style residual
     correction.
   - Experiment impact: existing QJL/TurboQuant comparator remains necessary;
     future systems table should report encode/decode cost and calibration.
   - Role: baseline.

9. **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization**
   - URL: https://arxiv.org/abs/2401.18079
   - Blocker helped: systems reviewers will compare packet communication to KV
     compression/transport.
   - Mechanism: per-channel key quantization, pre-RoPE keys, non-uniform
     datatypes, outlier handling.
   - Experiment impact: relevant baseline for any future cache-transfer branch;
     not required for the current CPU packet gate.
   - Role: baseline/framing.

10. **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**
    - URL: https://arxiv.org/abs/2402.02750
    - Blocker helped: stronger asymmetric K/V compression competitor.
    - Mechanism: keys and values use different quantization axes.
    - Experiment impact: cite as a cache-compression baseline; compare if the
      paper makes cache-transfer claims.
    - Role: baseline/framing.

11. **Communicating Activations Between Language Model Agents**
    - URL: https://arxiv.org/abs/2501.14082
    - Blocker helped: closest latent/activation communication competitor class.
    - Mechanism: fuse activations from one model into another during generation.
    - Experiment impact: reinforces that LatentWire should claim compact
      source-private packets, not generic activation communication.
    - Role: baseline/framing.

12. **Cache-to-Cache**
    - URL: https://arxiv.org/abs/2510.03215
    - Blocker helped: closest high-dimensional KV communication competitor.
    - Mechanism: project/fuse source KV-cache into target KV-cache with gating.
    - Experiment impact: future competitor row; current packet claim should
      emphasize much smaller rate and strict source-private controls.
    - Role: baseline/framing.

## Resulting Method Decision

The consistency-posterior packet is distinct from KV/cache/activation transfer
because it sends a tiny task-instance posterior packet decoded with public
candidate side information. The implemented NumPy version did not clear
bidirectional cross-family controls, so it is an ablation rather than a new
claim. The next highest-value gate is a systems-rate frontier with TTFT/latency,
bytes, token counts, encode/decode cost, and structured text/compression
baselines.
