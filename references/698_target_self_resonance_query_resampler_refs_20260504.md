# Target Self-Resonance Query-Resampler References

Date: 2026-05-04

This memo records the literature boundary for the failed HellaSwag target
self-resonance query-resampler gate. The local result should be treated as a
negative method gate, not as a novel positive contribution.

## Closest Prior Work

- Prefix-Tuning optimizes continuous prefix vectors for frozen language models:
  https://aclanthology.org/2021.acl-long.353/
- Gist Tokens train special tokens that compress prompt context:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html
- In-context Autoencoder compresses long context into compact memory slots:
  https://openreview.net/forum?id=uREj4ZuGJE
- AutoCompressor trains language models to compress contexts into soft prompts:
  https://arxiv.org/abs/2305.14788
- Perceiver and multimodal query bottlenecks motivate learned query slots over
  variable-length inputs:
  https://arxiv.org/abs/2103.03206
- Flamingo uses a Perceiver Resampler to map visual inputs into compact tokens
  for a frozen/gated language model interface:
  https://arxiv.org/abs/2204.14198
- BLIP-2/Q-Former uses learned queries as a bottleneck between frozen vision and
  language components:
  https://proceedings.mlr.press/v202/li23q.html
- Relative Representations explicitly target latent-space communication and
  stitching through anchor-relative coordinates:
  https://openreview.net/forum?id=SrC-nwieGJ

## Systems and Competitor Boundary

- C2C proposes direct semantic communication between LLMs via projected/fused
  KV-cache states:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm studies efficient LLM communication through selective KV sharing:
  https://openreview.net/forum?id=F7rUng23nw
- TurboQuant is an online vector quantization method relevant to KV/vector
  packet compression and systems baselines:
  https://openreview.net/forum?id=tO3ASKZlok
- vLLM and SGLang remain serving-system comparators for any latency/throughput
  claim:
  https://github.com/vllm-project/vllm
  https://github.com/sgl-project/sglang

## Novelty Boundary

The query-resampler architecture alone is not novel. It resembles existing
query bottlenecks and soft-prompt context compression. A publishable novelty
claim would require the stricter property that a source-conditioned compact
packet improves a frozen receiver over matched target-only, wrong-source,
wrong-row, and candidate-deranged controls at the same byte/slot budget.

The current gate does not establish that property. It weakens the target-only
query-resampler branch and promotes common-basis or anchor-relative source
features as the next branch.

## Implication for the Paper

Do not claim "latent communication" from this result. It can be used as a
negative ablation showing that target-side query bottlenecks alone are
insufficient unless the learned slots separate from target priors and wrong-row
controls.
