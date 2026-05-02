# COLM 2026 Citation Audit References

Date: 2026-05-02

## Purpose

This memo records the paper-facing citation set used in
`colm/2026/latentwire_colm2026.tex` and the claim boundary each source is
allowed to support.

## Datasets and model families

- ARC-Challenge: `https://arxiv.org/abs/1803.05457`
  - Supports the benchmark description and citation for ARC-Challenge.
- OpenBookQA: `https://arxiv.org/abs/1809.02789`
  - Supports the benchmark description and citation for OpenBookQA.
- Qwen2.5 technical report: `https://arxiv.org/abs/2412.15115`
  - Supports the Qwen2.5 same-family model-family citation.
- Phi-3 technical report: `https://arxiv.org/abs/2404.14219`
  - Supports the Phi-3 cross-family/source-family falsification citation.

## Representation and connector prior work

- Relative Representations: `https://openreview.net/forum?id=SrC-nwieGJ`
  - Supports the anchor/relative-coordinate prior-art boundary.
  - Does not support claiming LatentWire is zero-shot model stitching.
- Sparse Autoencoders: `https://arxiv.org/abs/2309.08600`
  - Supports feature dictionary motivation only.
- SAE universality: `https://arxiv.org/abs/2410.06981`
  - Supports common-feature plausibility only.
- SAEBench: `https://arxiv.org/abs/2503.09532`
  - Supports sparse-feature evaluation caution only.
- Prefix-Tuning: `https://arxiv.org/abs/2101.00190`
  - Supports the continuous-prefix baseline family.
- Gist Tokens: `https://arxiv.org/abs/2304.08467`
  - Supports prompt-compression/soft-token baseline framing.
- BLIP-2 / Q-Former: `https://arxiv.org/abs/2301.12597`
  - Supports query-bottleneck connector inspiration; not claimed as a completed LatentWire result.

## Systems and communication competitors

- Cache-to-Cache: `https://arxiv.org/abs/2510.03215`
  - Supports cache/KV communication competitor framing.
  - Does not support claiming LatentWire beats C2C natively.
- KVComm: `https://arxiv.org/abs/2510.03346`
  - Supports selective KV-sharing competitor framing.
  - Does not support claiming LatentWire beats KVComm natively.
- QJL: `https://arxiv.org/abs/2406.03482`
  - Supports a one-token 1-bit KV-state floor in byte/exposure accounting.
- KIVI: `https://arxiv.org/abs/2402.02750`
  - Supports a 2-bit KV-cache quantization comparator floor.
- KVQuant: `https://arxiv.org/abs/2401.18079`
  - Supports sub-4-bit KV-cache quantization comparator framing.
- TurboQuant: `https://arxiv.org/abs/2504.19874`
  - Supports vector/KV quantization comparator framing.
- vLLM/PagedAttention: `https://arxiv.org/abs/2309.06180`
  - Supports native serving substrate context and KV-cache memory motivation.
- SGLang: `https://arxiv.org/abs/2312.07104`
  - Supports structured LLM serving/KV-reuse substrate context.

## Information theory

- Slepian-Wolf, 1973.
  - Supports distributed source coding of correlated sources.
- Wyner-Ziv, 1976.
  - Supports rate-distortion with decoder side information.

## Claim boundary

The citation set supports a conservative COLM story:

- fixed-byte source-private packet evidence transfer;
- public-coordinate packet relation to relative representations;
- systems byte/exposure accounting against KV/cache-state objects;
- negative cross-family and connector evidence.

It does not support claiming:

- universal latent language;
- solved cross-family communication;
- native GPU throughput gains;
- superiority over C2C, KVComm, QJL, KIVI, KVQuant, TurboQuant, vLLM, or SGLang.
