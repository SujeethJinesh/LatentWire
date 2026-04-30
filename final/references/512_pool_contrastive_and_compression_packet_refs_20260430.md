# Pool-Contrastive Receiver And Compression-Packet References

- date: `2026-04-30`
- purpose: primary-source framing for the whole-candidate-pool receiver
  falsification and the next compression-native packet branches.

## Learned Receiver / Latent Refinement

1. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture", arXiv `2301.08243`,
   https://arxiv.org/abs/2301.08243.
   - Blocker: non-hand-coded receiver needs latent prediction motivation.
   - Mechanism: predict target embeddings in representation space rather than
     reconstructing raw inputs.
   - Experiment implication: keep JEPA/Q-Former-style receiver as a learned
     connector baseline, but current feature path is pruned after whole-pool
     failure.
   - Role: inspiration/framing.

2. Li et al., "BLIP-2", arXiv `2301.12597`,
   https://arxiv.org/abs/2301.12597.
   - Blocker: query bottleneck receiver family needs a serious architectural
     precedent.
   - Mechanism: Q-Former query bottleneck extracts useful source-side features
     for a frozen language model.
   - Experiment implication: next connector should use stronger source/target
     hidden-state features, not more random or semantic-anchor features.
   - Role: architecture inspiration.

3. Peebles and Xie, "Scalable Diffusion Models with Transformers", arXiv
   `2212.09748`, https://arxiv.org/abs/2212.09748.
   - Blocker: current one-shot receivers may be too brittle.
   - Mechanism: transformer denoising over latent tokens with conditioning.
   - Experiment implication: possible future denoising packet receiver, but it
     should be tested only after stronger features exist.
   - Role: inspiration.

4. Song et al., "Consistency Models", arXiv `2303.01469`,
   https://arxiv.org/abs/2303.01469.
   - Blocker: source-destroying controls must stay stable under packet
     corruption.
   - Mechanism: consistent predictions across noise levels.
   - Experiment implication: future receiver losses can require matched packet
     improvement while corrupted packets preserve target-only outputs.
   - Role: ablation/theory support.

## Compression-Native Packet Branches

5. QJL, "1-Bit Quantized JL Transform for KV Cache Quantization", arXiv
   `2406.03482`, https://arxiv.org/abs/2406.03482.
   - Blocker: systems reviewers expect mathematical compression baselines.
   - Mechanism: JL-style sketching and low-bit quantization.
   - Experiment implication: implement a `packet_rotation_sign` source vector
     packet and compare against deterministic atom packets and text relays.
   - Role: baseline inspiration, not direct same-budget competitor.

6. "TurboQuant", arXiv `2504.19874`, https://arxiv.org/abs/2504.19874.
   - Blocker: quantization-aware transport can make the systems contribution
     deeper.
   - Mechanism: mixed-precision / protected-channel quantization for KV-style
     states.
   - Experiment implication: implement mixed-precision packet fields only if
     moving to continuous residual/source vectors.
   - Role: inspiration and future baseline.

7. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI
   2011, https://dblp.org/rec/journals/pami/JegouDS11.
   - Blocker: hand-coded packet IDs look too symbolic.
   - Mechanism: represent vectors by indices into sub-codebooks.
   - Experiment implication: a `packet_pq_codebook` variant could make bytes
     code learned source residuals rather than named atoms.
   - Role: baseline/inspiration.

8. KIVI, "A Tuning-Free Asymmetric 2bit Quantization for KV Cache", arXiv
   `2402.02750`, https://arxiv.org/abs/2402.02750.
   - Blocker: fair systems comparison against KV compression.
   - Mechanism: asymmetric 2-bit KV quantization.
   - Experiment implication: keep as a high-rate KV byte-floor row, not a
     same-budget source-private packet baseline.
   - Role: systems baseline/caveat.

## Gate Implication

The whole-pool receiver negative result means the current learned connector
failure is not just an independent-row training artifact. The next strongest
technical contribution is likely compression-native packets over learned or
continuous source vectors, because those can produce a genuinely new method
family and make TurboQuant/QJL/PQ comparisons fairer.
