# Protocol Stress And Uniqueness References

- date: `2026-04-29`
- blocker: defend the source-private packet contribution against existing
  model-to-model communication, KV/cache transfer, prompt compression, and
  semantic/source-coding work.

## Direct Competitors And Boundaries

1. **Cache-to-Cache (C2C), OpenReview**
   - source: https://openreview.net/forum?id=LeatkxrBCi
   - blocker helped: reviewers may argue cross-LLM communication is already
     solved by latent/cache transfer.
   - mechanism/design idea: project/fuse source KV/cache information into the
     target model with learned gates.
   - next experiment: keep C2C/KV-style rows as high-rate internal-state
     baselines, but distinguish them from the source-private packet threat
     model.
   - role: baseline and framing.

2. **KVComm, OpenReview**
   - source: https://openreview.net/forum?id=F7rUng23nw
   - blocker helped: selective KV sharing may dominate packet methods if judged
     only by accuracy.
   - mechanism/design idea: communicate selected KV layers/heads under
     importance criteria.
   - next experiment: report bytes/latency/accuracy against a selective-KV
     proxy where possible.
   - role: high-rate baseline.

3. **DroidSpeak**
   - source: https://arxiv.org/abs/2411.02820
   - blocker helped: same-family cache reuse already has clear serving systems
     value.
   - mechanism/design idea: reuse KV/cache state across related or fine-tuned
     LLM variants.
   - next experiment: none immediately; use as scope boundary.
   - role: systems framing baseline.

## Prompt/Text Compression Baselines

4. **LLMLingua**
   - source: https://aclanthology.org/2023.emnlp-main.825/
   - blocker helped: naive matched-byte text is too weak as a text relay
     baseline.
   - mechanism/design idea: coarse-to-fine prompt compression with a budget
     controller.
   - next experiment: add a query-aware compressed-text row to the rate frontier
     or clearly explain why current structured text is a lower-bound baseline.
   - role: baseline.

5. **LongLLMLingua**
   - source: https://aclanthology.org/2024.acl-long.91/
   - blocker helped: source-private tool traces may be compressible by
     query-aware text methods.
   - mechanism/design idea: reorder and compress context around query
     relevance.
   - next experiment: strongest near-term baseline addition is matched-byte
     query-aware trace compression.
   - role: baseline and ablation.

## Source Coding And Semantic Communication

6. **Distributed Indirect Source Coding With Decoder Side Information**
   - source: https://arxiv.org/abs/2405.13483
   - blocker helped: gives the cleanest theory language for source-private
     packet communication under target-side candidate/context side information.
   - mechanism/design idea: source sends only the innovation needed by a decoder
     that already has side information.
   - next experiment: learned Wyner-Ziv/syndrome packet with held-out codebooks.
   - role: theory support and method inspiration.

7. **Deep Joint Source-Channel Coding For Semantic Communications**
   - source: https://arxiv.org/abs/2211.08747
   - blocker helped: moves the paper from a single accuracy row to a
     rate-distortion communication story.
   - mechanism/design idea: task-oriented coding under bandwidth constraints.
   - next experiment: keep rate curves and source-destroying controls as
     first-class paper tables.
   - role: framing and theory support.

## Quantization / Systems Inspiration

8. **TurboQuant**
   - source: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
   - blocker helped: systems reviewers will expect byte, latency, and kernel
     accounting, not only accuracy.
   - mechanism/design idea: extreme compression with protected correction
     components.
   - next experiment: report packet byte/latency frontiers and include
     QJL/TurboQuant-style residual comparators.
   - role: systems inspiration and baseline pressure.

9. **QJL**
   - source: https://arxiv.org/abs/2406.03482
   - blocker helped: random projection plus sign bits is a natural compact
     comparator to learned packets.
   - mechanism/design idea: Johnson-Lindenstrauss projection and low-bit
     correction/sketching.
   - next experiment: keep the QJL residual comparator as a matched-byte
     baseline; do not promote it because current remap rows underperform scalar.
   - role: ablation and systems comparator.

## Connector / Latent Inspiration

10. **BLIP-2 / Q-Former**
    - source: https://arxiv.org/abs/2301.12597
    - blocker helped: learned bottleneck connectors are established, so a
      future latent/adapter contribution needs target-preserving gates and
      strong baselines.
    - mechanism/design idea: query bottleneck connects frozen encoders and
      language models.
    - next experiment: only revive query-bottleneck adapters after a source
      surface has clean source-private signal.
    - role: inspiration and baseline family.

11. **Flamingo**
    - source: https://arxiv.org/abs/2204.14198
    - blocker helped: cross-attention resamplers are prior art for connecting
      frozen systems.
    - mechanism/design idea: gated cross-attention and Perceiver-style
      resampling into a frozen LLM.
    - next experiment: target-preserving gated connector is a future branch, not
      the next local gate.
    - role: inspiration.

12. **Diffusion Transformers With Representation Autoencoders**
    - source: https://openreview.net/forum?id=0u1LigJaab
    - blocker helped: latent bottlenecks need semantically useful
      representations, not only low dimensionality.
    - mechanism/design idea: replace weak latent bottlenecks with
      representation-rich encoders.
    - next experiment: indirect; supports representation-aware packet encoders
      rather than another hand-coded residual.
    - role: future-method inspiration.

## Uniqueness Assessment

The defensible novelty is **not** generic model-to-model communication. C2C,
KVComm, cache reuse, prompt compression, and connector papers already cover
large parts of that space.

The defensible novelty is narrower: **source-private, extreme-rate evidence
packets decoded with target side information, strict source-destroying controls,
exact-ID parity, and systems byte accounting**. That combination remains
distinct from high-rate cache transfer and ordinary prompt compression.

## Experiment Implications

1. Add a learned Wyner-Ziv/syndrome packet gate that transfers across held-out
   codebooks and beats query-aware compressed text at the same byte budget.
2. Extend the systems frontier with query-aware compressed-text controls, not
   only naive JSON/free-text truncation.
3. Keep protocol-stress rows visible: deterministic codebook remap passes,
   learned slot remap passes with weaker margins, canonical RASP is positive but
   still a seven-remap near miss, and learned target-decoder prompt paraphrase
   stress is still missing.
