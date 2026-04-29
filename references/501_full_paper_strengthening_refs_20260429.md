# Full-Paper Strengthening References

- date: `2026-04-29`
- blocker: the paper needs sharper novelty positioning, stronger systems
  caveats, and a next method branch that is more than hand-coded sparse atoms.
- role: related-work and experiment-planning memo for the N512 shared sparse
  boundary and next learned synonym-invariant dictionary gate.

## Primary Sources And Paper Use

1. **C2C: Cache-to-Cache Communication**
   - Source: https://arxiv.org/abs/2510.03215
   - Blocker helped: prevents overclaiming “first cross-model communication.”
   - Mechanism/design idea: high-rate internal-state communication by
     projecting/fusing source KV cache into a target cache.
   - Next experiment change: compare as an internals-access, high-rate baseline
     or caveat; do not frame LatentWire as universally better.
   - Role: baseline and novelty boundary.

2. **KVComm / Selective KV Sharing**
   - Source: https://openreview.net/forum?id=F7rUng23nw
   - Blocker helped: systems reviewers may expect KV-sharing baselines.
   - Mechanism/design idea: communicate selected KV cache components instead of
     explicit evidence packets.
   - Next experiment change: keep KV/cache byte accounting in the systems table
     and reserve direct runs for GPU availability.
   - Role: systems baseline.

3. **KVCOMM: Online Cross-Context KV-Cache Communication**
   - Source: https://arxiv.org/abs/2510.12872
   - Blocker helped: separates source-private evidence communication from
     shared-context cache reuse.
   - Mechanism/design idea: reuse/communicate KV state across related contexts.
   - Next experiment change: if implemented later, withhold private diagnostic
     evidence and verify gains vanish.
   - Role: systems baseline and caveat.

4. **Communicating Activations Between Language Model Agents**
   - Source: https://openreview.net/forum?id=W6RPXUUFic
   - Blocker helped: activation-level agent communication is prior art.
   - Mechanism/design idea: transmit internal activations between agents.
   - Next experiment change: use as a broad-latent-transfer threat; LatentWire's
     deployable distinction is API-compatible, tiny, explicit packets.
   - Role: baseline/framing.

5. **LLMLingua**
   - Source: https://aclanthology.org/2023.emnlp-main.825/
   - Blocker helped: compressed text can catch up at higher byte/token budgets.
   - Mechanism/design idea: compress visible prompt/context tokens.
   - Next experiment change: report full rate curves and query-aware text relay
     rather than only naive truncation.
   - Role: text-compression baseline.

6. **TurboQuant**
   - Source: https://arxiv.org/abs/2504.19874
   - Blocker helped: modern vector/KV quantization threatens systems novelty.
   - Mechanism/design idea: low-bit vector/KV transport with rotation and
     residual-aware quantization.
   - Next experiment change: keep 2.5/3.5-bit byte-floor accounting; only run a
     native comparator for a future vector/latent packet branch.
   - Role: systems byte-floor comparator.

7. **QJL**
   - Source: https://arxiv.org/abs/2406.03482
   - Blocker helped: one-bit sign sketches are a strong low-bit vector baseline.
   - Mechanism/design idea: quantized Johnson-Lindenstrauss projections preserve
     inner products.
   - Next experiment change: include same-byte random/sign-sketch controls for
     learned latent packets.
   - Role: compression baseline and theory support.

8. **KIVI and KVQuant**
   - Sources: https://arxiv.org/abs/2402.02750 and
     https://arxiv.org/abs/2401.18079
   - Blocker helped: low-bit KV-cache compression is an obvious systems
     objection.
   - Mechanism/design idea: asymmetric or data-aware low-bit KV-cache storage.
   - Next experiment change: report model-geometry bytes/token, not just
     nominal bit widths.
   - Role: systems baseline.

9. **Flow Matching**
   - Source: https://arxiv.org/abs/2210.02747
   - Blocker helped: suggests a less label-like method than scalar/sparse
     packets.
   - Mechanism/design idea: learn a vector field that transports target priors
     to source-conditioned posteriors.
   - Next experiment change: candidate-set-equivariant flow packet is a
     high-risk successor branch if learned dictionaries fail.
   - Role: method inspiration.

10. **Consistency Models**
    - Source: https://proceedings.mlr.press/v202/song23a.html
    - Blocker helped: endpoint prediction from corrupted/noisy states maps to a
      one-step receiver that should preserve target behavior.
    - Mechanism/design idea: decode stable endpoints rather than reconstruct
      full source traces.
    - Next experiment change: add synonym-consistency and abstention to learned
      packet receivers.
    - Role: method inspiration.

11. **I-JEPA**
    - Source: https://arxiv.org/abs/2301.08243
    - Blocker helped: the next learned packet should predict latent task
      structure rather than reconstruct text labels.
    - Mechanism/design idea: predict abstract target representations from
      context in representation space.
    - Next experiment change: learned dictionary packet should train on
      paraphrase-invariant latent/candidate features, not surface atom tokens.
    - Role: method inspiration.

12. **Sparse Crosscoders / Universal Sparse Autoencoders**
    - Sources: https://transformer-circuits.pub/2024/crosscoders/index.html and
      https://arxiv.org/abs/2502.03714
    - Blocker helped: shared sparse dictionaries are plausible, but current
      atoms are hand-designed.
    - Mechanism/design idea: learn shared and private sparse features across
      models, layers, or representation streams.
    - Next experiment change: implement learned synonym-invariant shared
      dictionary with atom derangement and causal feature knockout.
    - Role: baseline and next-method motivation.

13. **Slepian-Wolf / Wyner-Ziv Source Coding**
    - Sources:
      https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
      and https://doi.org/10.1109/TIT.1976.1055508
    - Blocker helped: decoder side information is classical; the paper must
      claim an empirical LLM/agent instantiation, not new coding theory.
    - Mechanism/design idea: communicate only the residual/syndrome needed by a
      decoder with side information.
    - Next experiment change: report rate-distortion style accuracy-vs-byte
      curves and source-destroying controls.
    - Role: theory framing.

## Decision

The final paper should foreground three claim-ready contributions:

1. a source-private communication benchmark/control protocol;
2. an extreme-rate packet systems frontier with endpoint uncertainty;
3. an interpretable agreed-dictionary sparse packet that now passes a larger
   native `n=512` slice but fails synonym stress.

The next method contribution should be learned synonym-invariant shared
dictionary communication. Diffusion/flow/JEPA ideas are useful only if they
become executable gates with source-destroying controls and byte accounting.
