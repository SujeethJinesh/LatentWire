# ICLR Strengthening And Anchor-Relative References

- date: `2026-04-29`
- blocker: the paper now has a strong source-private packet story, but ICLR
  reviewers can still reject broad novelty, systems, and cross-family claims.

## Related-Work Positioning

1. **C2C: Cache-to-Cache Communication**
   - source: https://arxiv.org/abs/2510.03215
   - blocker helped: closest broad non-text communication framing.
   - mechanism/design idea: C2C-style cache transfer is a high-rate baseline;
     LatentWire should claim a lower-rate source-private regime, not first
     latent communication.
   - next experiment: keep C2C/KV rows as right-frontier comparators when
     endpoint or GPU resources are available.
   - role: baseline and novelty threat.

2. **KVComm / KVCOMM cache communication**
   - sources: https://arxiv.org/abs/2510.03346 and
     https://arxiv.org/abs/2510.12872
   - blocker helped: reviewers may view packet transfer as another cache
     sharing method.
   - mechanism/design idea: distinguish source-private candidate evidence from
     shared-context cache reuse; report payload bytes and TTFT separately.
   - next experiment: endpoint frontier with packet, full-log relay, and
     cache-style right-frontier byte accounting.
   - role: systems baseline.

3. **DroidSpeak**
   - source: https://arxiv.org/abs/2411.02820
   - blocker helped: same-architecture cross-LLM KV reuse already exists.
   - mechanism/design idea: do not claim general same-architecture cache reuse;
     emphasize interpretable source-private packets and source-destroying
     controls.
   - next experiment: no immediate Mac-local gate; cite as competitor framing.
   - role: related work and baseline pressure.

4. **Communicating Activations Between Language Model Agents**
   - source: https://arxiv.org/abs/2501.14082
   - blocker helped: activation-level agent communication is a direct novelty
     threat.
   - mechanism/design idea: activation channels need source-shuffle and random
     activation controls; LatentWire's contrast is extreme-rate interpretable
     packets.
   - next experiment: keep activation/KV methods out of the headline unless a
     fair local activation baseline is added.
   - role: related work and baseline.

5. **CIPHER: Let Models Speak Ciphers**
   - source: https://arxiv.org/abs/2310.06272
   - blocker helped: non-natural-language agent messages are known.
   - mechanism/design idea: compare against learned/embedding messages with
     answer-only and answer-masked controls.
   - next experiment: use as framing for why source-private residual evidence is
     stricter than free-form cipher communication.
   - role: related work.

6. **LLMLingua / LongLLMLingua**
   - sources: https://aclanthology.org/2023.emnlp-main.825/ and
     https://arxiv.org/abs/2310.06839
   - blocker helped: compressed text is the strongest text-relay objection.
   - mechanism/design idea: matched-byte and query-aware compressed text must be
     in the rate frontier.
   - next experiment: endpoint TTFT/E2E frontier should include query-aware
     compressed text and full-log relay.
   - role: baseline.

## Mechanism Inspirations

7. **Relative Representations**
   - source: https://openreview.net/forum?id=SrC-nwieGJ
   - blocker helped: absolute scalar coordinates fail bidirectional
     cross-family transfer.
   - mechanism/design idea: express source evidence by similarities to public
     anchors/candidates, then send sparse relative innovations.
   - next experiment: anchor-relative sparse innovation packet smoke.
   - role: method inspiration and diagnostic.

8. **Sparse Crosscoders**
   - source: https://transformer-circuits.pub/2024/crosscoders/
   - blocker helped: need separate shared versus model-specific features.
   - mechanism/design idea: learn sparse atoms over relative source/target
     feature differences and transmit only top-k source-private atoms.
   - next experiment: top-k atom packet with atom permutation and shuffled-source
     controls.
   - role: method inspiration and interpretability support.

9. **BLIP-2 Q-Former / Flamingo Perceiver Resampler**
   - sources: https://arxiv.org/abs/2301.12597 and
     https://arxiv.org/abs/2204.14198
   - blocker helped: a target-preserving receiver should use a small query
     bottleneck rather than overwrite the target.
   - mechanism/design idea: frozen target plus gated query bottleneck for
     source-private evidence, with zero-init/no-source controls.
   - next experiment: only after packet receiver scale; current Mac-local gate
     should stay CPU-light.
   - role: future architecture inspiration.

10. **I-JEPA / V-JEPA / LeJEPA**
    - sources: https://arxiv.org/abs/2301.08243,
      https://arxiv.org/abs/2404.08471, and https://arxiv.org/abs/2406.15137
    - blocker helped: avoid collapse and answer-copying in source-private
      latent objectives.
    - mechanism/design idea: predict target-side candidate/posterior states
      from masked source evidence while tracking variance/effective rank and
      answer-masked controls.
    - next experiment: use as loss design only if a trainable query-bottleneck
      branch is opened.
    - role: theory/method inspiration.

11. **Diffusion Transformers / representation-autoencoder DiT variants**
    - sources: https://arxiv.org/abs/2212.09748 and
      https://arxiv.org/abs/2510.11690
    - blocker helped: high-dimensional latent patch transport suggests a
      denoising/score-matching view of packet decoding.
    - mechanism/design idea: treat the packet as a sparse denoising innovation
      over target candidate states, not as a literal answer code.
    - next experiment: bounded inspiration for anchor-relative sparse packet;
      do not add a diffusion model unless the sparse smoke passes.
    - role: inspiration.

12. **TurboQuant / QJL / KIVI**
    - sources: https://arxiv.org/abs/2504.19874,
      https://arxiv.org/abs/2406.03482, and https://arxiv.org/abs/2402.02750
    - blocker helped: systems reviewers expect modern quantization and KV
      compression comparators.
    - mechanism/design idea: random rotations, JL sketches, and asymmetric
      quantization motivate protected-head/sign-tail packet ablations.
   - next experiment: endpoint frontier, not another protected-residual tweak,
     because the current protected residual gate is a near-miss rather than a
     promoted method.
   - role: systems baseline and ablation.

## Sprint Decision

The safest ICLR framing is not broad latent communication. It is
source-private, extreme-rate evidence handoff with decoder side information,
strict source-destroying controls, frozen receiver evidence, and an honest
systems frontier.

The next new-method branch should be **anchor-relative sparse innovation
packets**. It directly targets the observed failure mode: scalar WZ and
canonical RASP are useful in same-family/remap settings but fail bidirectional
cross-family. The first smoke should be CPU-only, use the existing
core-to-holdout and holdout-to-core surfaces, and require both directions to
beat target-only while anchor/atom permutation controls collapse.
