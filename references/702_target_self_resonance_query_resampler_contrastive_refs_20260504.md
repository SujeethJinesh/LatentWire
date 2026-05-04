# Target Self-Resonance Query-Resampler Contrastive Rescue References

Date: 2026-05-04

This memo records the literature and novelty boundary for the failed
contrastive query-resampler rescue. The result is a negative ablation, not a
positive method contribution.

## Closest Prior Work

- Prefix-Tuning optimizes continuous virtual-token prefixes for frozen language
  models:
  https://aclanthology.org/2021.acl-long.353/
- Prompt Tuning shows that learned soft prompts can become competitive as model
  scale increases:
  https://arxiv.org/abs/2104.08691
- Gist Tokens train language models to compress prompts into smaller cached
  token sets:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html
- AutoCompressors train soft-prompt memories for context compression:
  https://arxiv.org/abs/2305.14788
- ICAE trains compact memory slots for in-context autoencoding:
  https://openreview.net/forum?id=uREj4ZuGJE
- Perceiver uses learned latent queries to attend over high-dimensional inputs:
  https://arxiv.org/abs/2103.03206
- Perceiver IO generalizes latent-query bottlenecks to structured output
  domains:
  https://arxiv.org/abs/2107.14795
- Flamingo uses a Perceiver Resampler to produce compact tokens for a frozen
  language model interface:
  https://arxiv.org/abs/2204.14198
- BLIP-2/Q-Former uses learned queries as a bridge between frozen vision and
  language components:
  https://proceedings.mlr.press/v202/li23q.html
- Relative Representations study anchor-relative latent coordinates for model
  stitching and representation transfer:
  https://openreview.net/forum?id=SrC-nwieGJ
- CKA and SVCCA are relevant diagnostics for deciding whether hidden-state
  agreement reflects meaningful shared representation structure:
  https://arxiv.org/abs/1905.00414
  https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability

## Systems and Competitor Boundary

- C2C proposes direct LLM communication through projected/fused KV-cache
  states:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm communicates through selected KV pairs rather than text or generic
  hidden states:
  https://openreview.net/forum?id=F7rUng23nw
- TurboQuant is a current systems comparator for vector and KV-cache
  quantization, but it is a compression primitive rather than a private
  source-conditioned communication protocol:
  https://openreview.net/forum?id=tO3ASKZlok
- KIVI and KVQuant are relevant KV-cache quantization byte-floor baselines:
  https://arxiv.org/abs/2402.02750
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html
- vLLM/PagedAttention and SGLang remain native serving-system baselines for any
  latency, memory, or throughput claim:
  https://arxiv.org/abs/2309.06180
  https://github.com/sgl-project/sglang

## Novelty Boundary

The local contrastive query-resampler is not novel by architecture. It is a
small learned-query soft-prefix compressor, close to prior prompt-compression
and multimodal connector work. The strict novelty target remains:

```text
source-conditioned compact packet
  > matched target-only slot baseline
  > wrong-source / shuffled-source / target-derived / candidate-deranged controls
at the same byte or slot budget
```

The current result does not satisfy that target. It strengthens the paper only
as a negative ablation that rules out target-only query bottlenecks as the
positive method.

## Promoted Next Branch

Build a source-conditioned residual-slot gate over a frozen target-only
baseline. The result should be interpreted only if it includes zero-source,
wrong-source, source-score shuffle, target-derived-code, candidate derangement,
and paired uncertainty controls.
