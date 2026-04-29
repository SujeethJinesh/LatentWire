# ICLR Comparison and Next-Method Scout References

- date: `2026-04-29`
- blocker helped: novelty positioning, systems comparisons, and next technical
  contribution after the local endpoint medium rung.
- experiment change: keep the endpoint packet claim scoped as extreme-rate
  source-private evidence transfer; add server-side systems comparisons when
  GPU serving is available; on the Mac, pursue a learned candidate-embedding or
  denoising WZ receiver rather than another static coordinate variant.

## Direct Communication Competitors

1. [C2C: Cache-to-Cache](https://arxiv.org/abs/2510.03215)
   - blocker helped: prevents claiming first general non-text LLM
     communication.
   - mechanism/design idea: higher-bandwidth KV/cache projection and fusion.
   - next experiment: compare as a conceptual GPU/server baseline; LatentWire
     should claim an extreme-rate packet point, not superiority over cache
     fusion.
   - role: direct competitor / framing threat.

2. [Communicating Activations Between Language Model Agents](https://openreview.net/pdf?id=W6RPXUUFic)
   - blocker helped: activation-level agent communication is direct prior art.
   - mechanism/design idea: receiver pauses at an intermediate layer, combines
     sender and receiver activations, then continues decoding.
   - next experiment: emphasize that the current method uses an external
     2-byte packet and frozen endpoint receiver, not activation injection.
   - role: direct latent/activation communication baseline.

3. [KVComm](https://openreview.net/forum?id=F7rUng23nw) and
   [KVCOMM](https://arxiv.org/abs/2510.12872)
   - blocker helped: KV sharing/reuse is a natural systems neighbor.
   - mechanism/design idea: selective KV sharing or cache reuse for efficient
     collaboration/prefill.
   - next experiment: future GPU section should compare packet relay against
     high-bandwidth KV/cache transport where feasible.
   - role: systems competitor.

## Prompt And Trace Baselines

4. [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/),
   [LongLLMLingua](https://arxiv.org/abs/2310.06839), and
   [LLMLingua-2](https://arxiv.org/abs/2403.12968)
   - blocker helped: compressed text can be a strong rate/accuracy baseline.
   - mechanism/design idea: budget-controlled prompt compression and
     query-aware information preservation.
   - next experiment: keep query-aware diagnostic text and matched-byte text in
     the main table; report packet as a far-left Pareto point, not an accuracy
     dominator over all higher-byte relays.
   - role: prompt-compression baseline.

5. [ReAct](https://arxiv.org/abs/2210.03629) and
   [Toolformer](https://arxiv.org/abs/2302.04761)
   - blocker helped: tool traces are established language-mediated evidence
     handoff.
   - mechanism/design idea: expose tool observations or actions as text.
   - next experiment: keep full hidden-log/tool-trace relay as the high-byte
     oracle comparator.
   - role: tool-use / trace-relay baseline.

## Systems And Quantized Transport

6. [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180),
   [DistServe](https://arxiv.org/abs/2401.09670), and
   [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2600/user-guide/docs/perf_benchmark/genai-perf-README.html)
   - blocker helped: Mac-local generation timing is not a server-throughput
     claim.
   - mechanism/design idea: report TTFT, ITL/TPOT, E2E latency, throughput,
     and goodput under fixed serving conditions.
   - next experiment: once GPU serving exists, run packet/text/full-log
     workloads through a common endpoint at several concurrency levels.
   - role: systems measurement standard.

7. [TurboQuant](https://arxiv.org/abs/2504.19874),
   [QJL](https://arxiv.org/abs/2406.03482),
   [KIVI](https://arxiv.org/abs/2402.02750), and
   [KVQuant](https://arxiv.org/abs/2401.18079)
   - blocker helped: modern systems reviewers expect quantized KV/cache or
     random-projection baselines.
   - mechanism/design idea: rotation/projection/low-bit transport can inspire
     protected residual or innovation packet baselines.
   - next experiment: keep QJL/protected residual rows as matched-byte
     comparators; future GPU work can compare with KV quantization when
     backend support exists.
   - role: quantized transport baseline and inspiration.

8. [CacheGen](https://arxiv.org/abs/2310.07240),
   [CacheBlend](https://arxiv.org/abs/2405.16444),
   [H2O](https://arxiv.org/abs/2306.14048), and
   [PyramidKV](https://arxiv.org/abs/2406.02069)
   - blocker helped: cache compression/pruning/reuse can dominate long-context
     systems costs.
   - mechanism/design idea: compress, prune, or reuse KV state rather than
     transmitting a tiny source-private packet.
   - next experiment: frame these as higher-rate or same-model memory systems,
     orthogonal to the current source-private control threat model.
   - role: KV/cache systems baseline.

## Next Learned-Method Inspirations

9. [I-JEPA](https://arxiv.org/abs/2301.08243),
   [V-JEPA](https://arxiv.org/abs/2404.08471), and
   [LLM-JEPA](https://openreview.net/forum?id=GbXKPo9QfH)
   - blocker helped: current endpoint receiver is still hand-designed.
   - mechanism/design idea: predict missing useful representations from
     context without reconstructing text.
   - next experiment: train a small candidate-embedding receiver that predicts
     the source-corrected candidate state from public candidates plus a compact
     packet.
   - role: learned receiver objective inspiration.

10. [Diffusion Transformers](https://arxiv.org/abs/2212.09748),
    [Diffusion of Thoughts](https://arxiv.org/abs/2402.07754), and
    [Block Diffusion](https://arxiv.org/abs/2503.09573)
    - blocker helped: suggests a less static source-private communication
      mechanism.
    - mechanism/design idea: iteratively denoise a target-side candidate
      posterior using a rate-capped source packet.
    - next experiment: implement 1/2/4-step denoising WZ posterior refinement
      with the same zero/shuffle/random/answer-masked controls.
    - role: method inspiration.

11. [Perceiver IO](https://arxiv.org/abs/2107.14795),
    [Flamingo](https://arxiv.org/abs/2204.14198),
    [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597), and
    [LLaVA](https://arxiv.org/abs/2304.08485)
    - blocker helped: connects the packet problem to established frozen-model
      connector designs.
    - mechanism/design idea: use learned queries or a small projection adapter
      to map source/private packet features into a frozen receiver.
    - next experiment: test a 4-8 query Perceiver/Q-Former-style receiver over
      candidate metadata and packet embeddings before larger LLM fine-tuning.
    - role: connector inspiration.

12. [Relative Representations](https://openreview.net/forum?id=SrC-nwieGJ),
    [Model Stitching](https://arxiv.org/abs/2303.11277),
    [SVCCA](https://arxiv.org/abs/1706.05806),
    [CKA](https://proceedings.mlr.press/v97/kornblith19a.html), and
    [Gromov-Wasserstein Alignment](https://aclanthology.org/D18-1214/)
    - blocker helped: static cross-family packets remain asymmetric.
    - mechanism/design idea: select aligned subspaces/layers and encode
      source innovations in anchor-relative or relational coordinates.
    - next experiment: learned Wyner-Ziv anchor packet first; keep OT/GW as a
      diagnostic/falsifier unless it beats random-anchor controls.
    - role: mathematical method inspiration and diagnostics.

## Decision

The current paper should not claim broad latent/KV communication novelty. The
defensible ICLR story is narrower and stronger: source-private, interpretable,
extreme-rate evidence transfer that survives destructive controls and gives a
far-left byte/TTFT point against text/log relays. The next technical
contribution should be a learned candidate-embedding or denoising WZ receiver
that keeps the same control suite and uses the endpoint `n=160` result as the
systems anchor.
