# Systems Caveat Frontier References

- date: `2026-04-29`
- blocker: the paper needs a reviewer-safe systems contribution without
  overclaiming against native KV/cache compression or production serving work.
- role: source memo for `source_private_systems_caveat_frontier_20260429`.

## Primary Sources And Use

1. C2C cache-to-cache communication (`https://arxiv.org/abs/2510.03215`;
   OpenReview: `https://openreview.net/forum?id=LeatkxrBCi`).
   - blocker helped: closest high-rate cross-model internal-state competitor.
   - mechanism/design idea: project and fuse source KV/cache state into the
     receiver.
   - next experiment change: compare as internals-access/high-rate baseline,
     not a same-byte source-private packet baseline.
   - use: baseline and framing.

2. KVComm / selective or online KV-cache communication
   (`https://openreview.net/forum?id=F7rUng23nw`;
   KVCOMM: `https://arxiv.org/abs/2510.12872`).
   - blocker helped: reviewers may expect multi-agent cache communication
     baselines.
   - mechanism/design idea: transmit selected/compressed KV state between
     contexts or agents.
   - next experiment change: keep TTFT and cache-byte axes separate from
     source-private packet accuracy.
   - use: systems baseline and caveat.

3. TurboQuant (`https://arxiv.org/abs/2504.19874`).
   - blocker helped: modern online vector quantization can make vector/cache
     transport cheaper.
   - mechanism/design idea: rotation/residual-aware low-bit vector transport.
   - next experiment change: use 2.5/3.5-bit byte-floor proxies and reserve
     direct implementation for a promoted latent/vector branch.
   - use: byte-floor systems comparator.

4. QJL (`https://arxiv.org/abs/2406.03482`).
   - blocker helped: one-bit sign sketches are a strong low-bit vector baseline.
   - mechanism/design idea: quantized Johnson-Lindenstrauss projections preserve
     inner products in high dimension.
   - next experiment change: keep QJL-style byte lower bounds and use raw sign
     controls for learned latent packet branches.
   - use: sketch baseline and theory support.

5. KIVI (`https://arxiv.org/abs/2402.02750`) and KVQuant
   (`https://arxiv.org/abs/2401.18079`).
   - blocker helped: low-bit KV-cache compression is the obvious systems
     objection.
   - mechanism/design idea: compare actual model-geometry bytes per token, not
     nominal bit-width alone.
   - next experiment change: report cache-byte lower bounds as caveats, not
     direct kernel comparisons.
   - use: cache compression baseline framing.

6. vLLM / PagedAttention (`https://arxiv.org/abs/2309.06180`) and DistServe
   (`https://arxiv.org/abs/2401.09670`).
   - blocker helped: systems reviewers expect serving metrics such as TTFT,
     TPOT, throughput, memory, and concurrency.
   - mechanism/design idea: separate prefill/decode and do not equate local CPU
     proxy TTFT with production goodput.
   - next experiment change: future NVIDIA run should use an OpenAI/vLLM-
     compatible server and report TTFT/TPOT/throughput.
   - use: metric convention and future serving baseline.

7. Diffusion Transformers (`https://arxiv.org/abs/2212.09748`), Latent
   Diffusion (`https://arxiv.org/abs/2112.10752`), Consistency Models
   (`https://proceedings.mlr.press/v202/song23a.html`), and I-JEPA
   (`https://arxiv.org/abs/2301.08243`).
   - blocker helped: user requested diffusion/JEPA inspiration for a less
     shallow method.
   - mechanism/design idea: a Mac-feasible future method is a consistency-
     distilled discrete/shared-dictionary packet, not a full DiT over LLM
     states.
   - next experiment change: pursue shared sparse crosscoder or VQ-style packet
     with source-destroying controls and feature knockout.
   - use: inspiration and future-method framing.

## Decision

The paper's current systems contribution should be stated as an extreme-rate
source-private communication frontier with local endpoint and derived byte-floor
evidence. It should not claim native superiority over KV compression,
cache-sharing, prompt-compression, or production serving systems until those
native baselines are run directly.
