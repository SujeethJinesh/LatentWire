# Receiver, Systems, And Novelty Scout References

Date: 2026-04-30

Blocker addressed: the current LatentWire paper has strong scoped packet
evidence, but reviewers can still challenge novelty, systems value, and whether
the receiver is too protocol-shaped. This memo consolidates the current
subagent/web scout into paper-facing comparisons and bounded next experiments.

## Sources And Implications

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with
   IO-Awareness**. https://arxiv.org/abs/2205.14135
   - Blocker helped: bytes alone are not enough for a systems paper; reviewers
     expect memory-traffic and hierarchy-aware accounting.
   - Mechanism: analyze movement across memory hierarchy, not only FLOPs.
   - Experiment impact: keep the memory traffic ledger with raw payload bytes,
     64B cache-line rounding, 128B DMA-burst rounding, and batch packing.
   - Role: systems framing.

2. **vLLM / PagedAttention**. https://arxiv.org/abs/2309.06180
   - Blocker helped: KV-cache systems baselines are the strongest serving-side
     neighbor.
   - Mechanism: paging and managing large dynamic KV caches for serving.
   - Experiment impact: compare LatentWire packets against KV/cache movement as
     a different interface regime, not as a direct KV-compression replacement.
   - Role: systems baseline/framing.

3. **DistServe**. https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
   - Blocker helped: the paper needs accepted serving metrics, not only local
     script latency.
   - Mechanism: separate prefill/decode phases and optimize TTFT/TPOT/goodput.
   - Experiment impact: future NVIDIA/endpoint table should report TTFT, TPOT,
     goodput, and placement assumptions.
   - Role: serving-systems framing.

4. **TurboQuant: Taming KV Cache Quantization with Randomized Hadamard
   Transform**. https://arxiv.org/abs/2504.19874
   - Blocker helped: modern KV quantization can make cache baselines much
     stronger than naive fp16 byte floors.
   - Mechanism: rotation plus quantization to preserve attention quality.
   - Experiment impact: keep TurboQuant/QJL rows as byte-floor comparators and
     do not claim native superiority over KV quantization.
   - Role: baseline and ablation inspiration.

5. **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**.
   https://arxiv.org/abs/2402.02750
   - Blocker helped: practical KV compression baseline for long-context
     inference.
   - Mechanism: asymmetric key/value quantization with hardware-friendly 2-bit
     cache storage.
   - Experiment impact: include a practical KV byte-floor row in systems tables.
   - Role: baseline.

6. **Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models**. https://arxiv.org/abs/2510.03215
   - Blocker helped: closest direct competitor to broad "cross-model latent
     communication" claims.
   - Mechanism: project/fuse source KV cache into the target cache.
   - Experiment impact: LatentWire must be positioned as source-private
     rate-capped task packets, not cache fusion.
   - Role: closest competitor/framing.

7. **KVCOMM: Online Cross-context KV-cache Communication for Multi-agent
   Collaboration**. https://arxiv.org/abs/2510.12872
   - Blocker helped: cache communication with agent/system speedups threatens
     any generic efficient-agent-communication claim.
   - Mechanism: align/reuse KV caches across agent contexts through anchors.
   - Experiment impact: add cache-sharing as a related interface, but mark it as
     moving cache state rather than source-private task evidence.
   - Role: competitor/framing.

8. **DroidSpeak: Communication-Aware KV Cache Sharing for LLM Agents**.
   https://arxiv.org/abs/2411.02820
   - Blocker helped: same-family cache reuse already supports efficient
     non-text communication between LLM agents.
   - Mechanism: reuse compatible KV/cache states in multi-agent serving.
   - Experiment impact: separate LatentWire's public candidate side information
     and destructive source controls from same-architecture cache reuse.
   - Role: competitor.

9. **BLIP-2 / Q-Former**. https://arxiv.org/abs/2301.12597
   - Blocker helped: current target decoder is too protocol-shaped.
   - Mechanism: a small query bottleneck bridges frozen encoders and frozen
     language models.
   - Experiment impact: next learned receiver should use query bottleneck or
     resampler structure, but trained against source-destroying controls.
   - Role: architecture inspiration.

10. **Flamingo**. https://arxiv.org/abs/2204.14198
    - Blocker helped: source injection needs target preservation.
    - Mechanism: gated cross-attention injects external information into a
      frozen language model.
    - Experiment impact: train source-preserving gates and measure whether
      controls collapse to target-only while matched packets retain lift.
    - Role: architecture inspiration.

11. **DiT: Scalable Diffusion Models with Transformers**.
    https://arxiv.org/abs/2212.09748
    - Blocker helped: one-shot receivers may be too weak or too brittle.
    - Mechanism: iterative transformer denoising over latent patches.
    - Experiment impact: test a packet-conditioned candidate-score denoiser over
      1/2/4 refinement steps.
    - Role: method inspiration.

12. **Flow Matching for Generative Modeling**. https://arxiv.org/abs/2210.02747
    - Blocker helped: iterative latent refinement needs a principled path rather
      than arbitrary repeated scoring.
    - Mechanism: learn vector fields along probability paths.
    - Experiment impact: train a small vector field that moves target-prior
      candidate logits toward packet-conditioned logits under controls.
    - Role: theory/method inspiration.

13. **Consistency Models**. https://arxiv.org/abs/2303.01469
    - Blocker helped: iterative receiver refinement may be too slow for a
      systems paper.
    - Mechanism: distill multi-step denoising trajectories into a few-step or
      one-step map.
    - Experiment impact: if a packet denoiser works, immediately distill it and
      compare accuracy/control safety/latency.
    - Role: method and latency ablation.

14. **Product Quantization for Nearest Neighbor Search**.
    https://doi.org/10.1109/TPAMI.2010.57
    - Blocker helped: product-codebook packet novelty must be scoped honestly.
    - Mechanism: compact subspace codebooks and asymmetric lookup.
    - Experiment impact: claim PQ as the codec primitive, not as the paper's
      novelty; novelty is source-private task communication under controls.
    - Role: baseline/theory support.

15. **LLMLingua**. https://aclanthology.org/2023.emnlp-main.825/
    - Blocker helped: compressed text relay is a serious baseline.
    - Mechanism: prompt compression for cheaper LLM inference.
    - Experiment impact: keep matched-byte and query-aware text rows prominent;
      add LLMLingua-style text compression only when the source text is exposed.
    - Role: baseline.

## Effect On The Next Experiment

The next experiment should not be another scalar/codebook-only surface unless it
adds a new failure mode. The current highest-value gates are:

1. finish the direct Qwen target-decoder `n=160` all-control run to attack the
   hand-coded receiver objection;
2. if positive, summarize paired uncertainty and update the final evidence
   table;
3. if more method invention is needed, stack a small packet-consistency denoiser
   on the existing product-codebook/semantic-anchor packets, using packet
   corruptions and source-destroying negatives.

## Claim Boundary

LatentWire is not claiming to beat C2C, KVCOMM, TurboQuant, KIVI, or prompt
compression on their native tasks. The defensible claim is a different
interface point: tiny source-private task packets decoded with public target
side information, with explicit destructive controls and hardware-readable
traffic accounting.
