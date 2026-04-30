# Systems Comparison And Next-Method References

- date: `2026-04-29`
- blocker: after the learned synonym-dictionary seed repeat and held-out
  paraphrase failure, the paper needs a cleaner systems comparison table and a
  higher-upside held-out receiver branch.
- role: source memo for the systems comparison table and next-method planning.

## Primary Sources And Experiment Impact

1. **Cache-to-Cache: Direct Semantic Communication Between LLMs**
   - Source: https://arxiv.org/abs/2510.03215
   - Blocker helped: closest direct latent/cache communication competitor.
   - Mechanism/design idea: compare against cache/vector fusion as a dense
     cross-model baseline, while emphasizing our extreme-rate, auditable packet
     interface.
   - Experiment change: systems table must mark C2C/KV rows as different
     access assumptions unless implemented on the same source-private task.
   - Role: baseline and reviewer-risk framing.

2. **Direct Semantic Communication Between LLMs via Vector Translation**
   - Source: https://arxiv.org/abs/2511.03945
   - Blocker helped: direct vector-space translation threatens broad novelty
     claims.
   - Mechanism/design idea: avoid claiming first latent communication; claim a
     public, discrete, source-private evidence interface instead.
   - Experiment change: keep vector-translation/C2C as dense-access baselines,
     not as packet-byte controls unless adapted to the task.
   - Role: baseline and claim boundary.

3. **Latent Collaboration in Multi-Agent Systems / LatentMAS**
   - Source: https://arxiv.org/abs/2511.20639
   - Blocker helped: multi-agent latent memory already covers "agents
     communicate without text" language.
   - Mechanism/design idea: distinguish shared latent workspace from
     source-private, rate-capped evidence packets.
   - Experiment change: paper should compare assumptions and avoid "first
     latent multi-agent communication" language.
   - Role: related work and reviewer-risk framing.

4. **LLMLingua**
   - Source: https://aclanthology.org/2023.emnlp-main.825/
   - Blocker helped: compressed text is a strong systems baseline.
   - Mechanism/design idea: compare packet accuracy against matched-byte and
     higher-byte structured/text relays.
   - Experiment change: systems table reports same-byte text and query-aware /
     structured text rate rows separately.
   - Role: baseline.

5. **Gist Tokens**
   - Source: https://arxiv.org/abs/2304.08467
   - Blocker helped: learned soft-token prompt compression overlaps with
     compact communication.
   - Mechanism/design idea: report whether the interface is model-internal
     soft context or externally auditable packet bytes.
   - Experiment change: keep future soft-token bridge as a separate branch from
     current public packet protocol.
   - Role: baseline and inspiration.

6. **In-context Autoencoder**
   - Source: https://arxiv.org/abs/2307.06945
   - Blocker helped: memory-slot context compression is adjacent to our
     rate-limited receiver side information.
   - Mechanism/design idea: compare rate curves rather than single operating
     points.
   - Experiment change: table includes same-byte controls, higher-rate
     structured text, and full hidden-log rows.
   - Role: baseline and systems framing.

7. **xRAG**
   - Source: https://arxiv.org/abs/2405.13792
   - Blocker helped: one-token or dense-feature evidence compression is a
     strong RAG-style neighbor.
   - Mechanism/design idea: source-selected evidence can be represented as
     compact features, but source privacy and controls must be explicit.
   - Experiment change: future wider benchmark should include RAG/document
     evidence tasks only after current packet controls hold.
   - Role: baseline and inspiration.

8. **TurboQuant**
   - Source: https://arxiv.org/abs/2504.19874
   - Blocker helped: reviewers will ask whether low-bit quantization makes the
     systems contribution trivial.
   - Mechanism/design idea: use TurboQuant as byte-floor / vector-compression
     context, not as direct proof against source-private task communication.
   - Experiment change: systems comparison marks TurboQuant/KV numbers as
     accounting-only unless a real kernel is run later.
   - Role: systems baseline and caveat.

9. **QJL**
   - Source: https://arxiv.org/abs/2406.03482
   - Blocker helped: 1-bit sign sketches are a natural compact vector
     baseline.
   - Mechanism/design idea: compare source-feature packets against scalar and
     QJL-style residual sketches when adapted to the same task.
   - Experiment change: systems table includes the local QJL-style residual
     source-code comparator and its control-clean status.
   - Role: baseline and systems inspiration.

10. **KIVI**
    - Source: https://arxiv.org/abs/2402.02750
    - Blocker helped: same-model KV cache quantization is a major systems
      neighbor.
    - Mechanism/design idea: report KV byte floors separately from packet
      payload bytes.
    - Experiment change: current table does not claim to beat KIVI; it reports
      estimated KV payload byte ratios only.
    - Role: systems caveat and baseline.

11. **Product Quantization**
    - Source: https://doi.org/10.1109/TPAMI.2010.57
    - Blocker helped: codebook amortization and vector distortion are relevant
      to compact packet baselines.
    - Mechanism/design idea: future packet baselines should include codebook
      bytes or state an amortization horizon.
    - Experiment change: next systems gate can add PQ distortion / recall on
      cached source/candidate features.
    - Role: baseline and ablation design.

12. **Relative Representations**
    - Source: https://openreview.net/forum?id=SrC-nwieGJ
    - Blocker helped: held-out paraphrase failure shows raw surface features do
      not generalize.
    - Mechanism/design idea: represent source/candidate evidence through
      anchor-relative similarities, reducing coordinate/surface dependence.
    - Experiment change: next receiver branch should test anchor-relative
      sparse innovation packets on the held-out family-B split.
    - Role: next-method inspiration.

13. **BLIP-2 / Q-Former**
    - Source: https://arxiv.org/abs/2301.12597
    - Blocker helped: need a stronger connector than hashed ridge word/char
      features.
    - Mechanism/design idea: learned query bottlenecks can bridge mismatched
      encoders and decoders.
    - Experiment change: one-month high-upside branch is a tiny query bottleneck
      over source-private traces with strict same-byte controls.
    - Role: method inspiration.

14. **Flamingo**
    - Source: https://arxiv.org/abs/2204.14198
    - Blocker helped: supports the resampler interface as a scalable connector
      pattern.
    - Mechanism/design idea: Perceiver-style query resampling as a rate-capped
      source interface.
    - Experiment change: if anchor-relative packets fail, test a small
      Perceiver/Q-Former packet builder over cached features.
    - Role: method inspiration.

15. **I-JEPA / V-JEPA**
    - Sources: https://arxiv.org/abs/2301.08243 and
      https://arxiv.org/abs/2404.08471
    - Blocker helped: held-out synonym transfer likely needs representation
      prediction rather than surface reconstruction.
    - Mechanism/design idea: train receiver-side features to predict invariant
      candidate predicates under masked/corrupted evidence.
    - Experiment change: possible denoising/JEPA receiver branch only after the
      anchor-relative receiver has been tested.
    - Role: method inspiration.

16. **Diffusion Transformers / DiT**
    - Source: https://arxiv.org/abs/2212.09748
    - Blocker helped: iterative denoising suggests a refinement-style receiver,
      but it is higher implementation risk.
    - Mechanism/design idea: treat source packet decoding as low-step latent
      denoising with target-side candidate context.
    - Experiment change: add only if simple anchor-relative or query-bottleneck
      receiver shows held-out signal but remains noisy.
    - Role: inspiration, not immediate baseline.

## Decision

The paper's safest systems claim is **source-private far-left-rate task
communication**, not superiority over KV compression kernels. The next method
branch should be an anchor-relative sparse innovation receiver on the exact
held-out family-B split, because it directly addresses the newest blocker with
the least Mac-local implementation risk.
