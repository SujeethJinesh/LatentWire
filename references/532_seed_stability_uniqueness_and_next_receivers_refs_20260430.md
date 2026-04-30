# Seed-Stability, Uniqueness, and Next-Receiver References

- date: `2026-04-30`
- purpose: integrate reviewer/uniqueness, systems, and method-synthesis
  subagent outputs while the seed31 n160 verifier gate runs
- current status: supports a scoped source-private communication claim, not a
  broad latent-transfer claim

## What This Helps With

The main blocker is reviewer skepticism that the current result is a
hand-shaped lookup protocol. This memo separates three questions:

1. what prior work already covers latent/cache/text communication;
2. what the current 2-byte packet plus frozen verifier uniquely tests;
3. what next method branch could become a distinct non-table contribution.

## Closest Communication and Compression Work

1. **Cache-to-Cache / C2C**
   - Source: https://arxiv.org/abs/2510.03215 and
     https://fuvty.github.io/C2C_Project_Page/
   - Blocker helped: closest broad "LLMs communicate without text" competitor.
   - Mechanism idea: source KV is projected/fused into the target cache.
   - Next experiment change: keep C2C as high-rate cache-state competitor, not
     as the same threat model.
   - Role: competitor / claim boundary.

2. **KVCOMM / KV cache communication**
   - Sources: https://openreview.net/forum?id=yGOytgjurF and
     https://arxiv.org/abs/2510.03346
   - Blocker helped: reviewers may compare to multi-agent cache sharing.
   - Mechanism idea: selectively reuse or communicate KV cache/context state.
   - Next experiment change: report source-KV exposure flags and byte floors
     whenever comparing to cache methods.
   - Role: competitor / systems baseline.

3. **DroidSpeak**
   - Sources: https://arxiv.org/abs/2411.02820 and
     https://www.microsoft.com/en-us/research/publication/droidspeak-kv-cache-sharing-for-efficient-multi-llm-serving/
   - Blocker helped: shows practical multi-LLM KV/E-cache reuse exists.
   - Mechanism idea: reuse intermediate cache state between related models to
     avoid recomputation.
   - Next experiment change: emphasize that LatentWire sends private evidence,
     not model-visible cache state.
   - Role: systems comparator / framing.

4. **Prompt compression: LLMLingua and LLMLingua-2**
   - Sources: https://arxiv.org/abs/2310.05736 and
     https://arxiv.org/abs/2403.12968
   - Blocker helped: visible-text compression is a strong practical baseline.
   - Mechanism idea: remove low-utility prompt tokens while preserving task
     performance.
   - Next experiment change: keep matched-byte and query-aware text relays in
     tables, with source-text exposure flags.
   - Role: baseline / systems framing.

5. **Tool/RAG handoff: RAG, ReAct, Toolformer**
   - Sources: https://arxiv.org/abs/2005.11401,
     https://arxiv.org/abs/2210.03629, and
     https://arxiv.org/abs/2302.04761
   - Blocker helped: tool observations are the obvious text-relay alternative.
   - Mechanism idea: pass retrieved/tool evidence as text for downstream
     reasoning.
   - Next experiment change: compare source-private packets against visible
     evidence relays under explicit privacy and byte assumptions.
   - Role: baseline / motivation.

6. **KV quantization and sketching: KIVI, KVQuant, QJL, TurboQuant**
   - Sources: https://arxiv.org/abs/2402.02750,
     https://arxiv.org/abs/2401.18079, https://arxiv.org/abs/2406.03482, and
     https://arxiv.org/abs/2504.19874
   - Blocker helped: naive fp16 KV byte comparisons are too weak.
   - Mechanism idea: asymmetric low-bit KV quantization, outlier handling,
     Johnson-Lindenstrauss sign sketches, and protected rotations.
   - Next experiment change: keep compressed KV rows as source-KV-exposed
     comparators and use TurboQuant/QJL ideas only for future geometric packet
     branches.
   - Role: systems baseline / method inspiration.

7. **Source coding with decoder side information: Wyner-Ziv and DISCUS**
   - Sources: https://doi.org/10.1109/TIT.1976.1055508 and
     https://doi.org/10.1109/TIT.2003.809536
   - Blocker helped: gives the right theory language for public target side
     information.
   - Mechanism idea: encode only a syndrome/residual because the decoder has
     correlated side information.
   - Next experiment change: frame the task as source-private residual evidence
     communication, not as generic prompt compression.
   - Role: theory support.

## Next Non-Table Receiver Ideas

1. **Anchor-relative sparse crosscoder receiver**
   - Sources: https://transformer-circuits.pub/2024/crosscoders/index.html and
     https://openreview.net/forum?id=SrC-nwieGJ
   - Mechanism: learn a sparse shared feature basis and send top-k source
     feature innovations instead of diagnostic handles.
   - Next experiment: n256 held-out synonym split with feature-ID permutation,
     source shuffle, public-only sparse classifier, and top-feature knockout.
   - Role: highest-EV next method branch.

2. **Consistency/flow receiver over candidate logits**
   - Sources: https://arxiv.org/abs/2210.02747,
     https://proceedings.mlr.press/v202/song23a, and
     https://arxiv.org/abs/2212.09748
   - Mechanism: learn a small transport map from target-only candidate scores
     to source-conditioned posterior scores under packet conditioning.
   - Next experiment: n64/n160 balanced diag-only with public-only and
     source-destroying views.
   - Role: learned-receiver inspiration / possible contribution.

3. **TurboResidual product receiver**
   - Sources: https://arxiv.org/abs/2504.19874 and
     https://arxiv.org/abs/2406.03482
   - Mechanism: product-codebook centroid plus QJL-style residual sign bits in
     a protected rotation basis.
   - Next experiment: remap 101/103/107 at budgets 2/4/6, requiring paired
     improvement over canonical PQ and scalar WZ, not just clean controls.
   - Role: compression-native method branch.

## Decision

- Promote now: seed-stable frozen verifier as the cleanest receiver evidence if
  seed31 n160 passes.
- Promote with caveat: packet/consumption trace as source-boundary systems
  evidence, not production serving throughput.
- Next branch after seed stability: anchor-relative sparse crosscoder receiver,
  because it directly attacks the lookup/agreed-ontology objection.
- Next systems branch after seed stability: batched verifier consumption trace
  with physical model-call accounting, prompt-token padding waste, RSS, and KV
  byte floors.
