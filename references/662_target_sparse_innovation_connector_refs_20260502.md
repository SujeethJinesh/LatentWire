# Target-Conditioned Sparse Innovation Connector References

Date: 2026-05-02

## Purpose

This memo pins the prior-work boundary for the current ARC hidden/query connector
branch. The intended LatentWire contribution is not generic activation
alignment, prompt compression, or sparse-feature interpretation. The live claim
would be a fixed-byte, source-private packet that carries source-side innovation
the target cannot reconstruct from its own prompt/cache, with destructive
controls and systems byte accounting.

## Primary Sources Checked

- Sparse Autoencoders Find Highly Interpretable Features in Language Models:
  https://arxiv.org/abs/2309.08600
  This motivates sparse feature dictionaries for LM activations. LatentWire
  differs only if the sparse representation becomes a causal communication
  packet evaluated on downstream task transfer.

- Quantifying Feature Space Universality Across LLMs via Sparse Autoencoders:
  https://arxiv.org/abs/2410.06981
  This supports possible feature analogies across models, but it is not a
  packet protocol and does not by itself prove cross-family task transfer.

- BLIP-2 / Q-Former:
  https://arxiv.org/abs/2301.12597
  This is a strong frozen-module query-bottleneck precedent. LatentWire should
  cite it as connector inspiration, while emphasizing language-to-language
  source-private packet transfer rather than multimodal bridging.

- Learning to Compress Prompts with Gist Tokens:
  https://arxiv.org/abs/2304.08467
  Gist tokens are a close prompt-compression baseline. LatentWire must separate
  itself by transmitting answer-key-forbidden source evidence and beating
  target-only and same-byte text controls.

- QJL:
  https://arxiv.org/abs/2406.03482
  QJL is a Johnson-Lindenstrauss plus sign-quantization KV-cache compression
  comparator. It belongs in the systems table as a byte-floor/vector-codec
  baseline, not as evidence for semantic communication.

- TurboQuant:
  https://arxiv.org/abs/2504.19874
  TurboQuant motivates rotation plus residual quantization for vector/KV
  compression. It is a mandatory systems comparator if LatentWire claims byte
  wins over compressed source state.

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
  Use as the primary native serving substrate for TTFT/TPOT/goodput and KV
  memory baselines.

- SGLang:
  https://arxiv.org/abs/2312.07104
  Use as the second native serving/runtime baseline, especially for prefix/cache
  reuse and structured generation workloads.

## Controls Required Before Claiming Novelty

- zero-byte target cache with identical public prompt path;
- same-byte random, shuffled-source, and candidate-roll packets;
- target-only learned packet to expose target-prior leakage;
- common-only, innovation-only, and common-plus-innovation packet ablations;
- source-cache endpoint headroom row, because a weak source can make any
  cross-family connector look bad;
- native byte/accounting rows that include payload bytes, framed bytes, aligned
  transfer bytes, dictionary/codebook residency, and decode overhead.

## Decision Impact

If the ARC Qwen2.5-1.5B hidden/query n32/n64 connector cannot beat both
Qwen-substituted and cached-source packets while controls fail, the immediate
branch should not be widened. The next higher-value move is either a target-loss
soft-prefix/denoising connector on a small frozen slice or a stronger source
endpoint with source score vectors rematerialized under the same
answer-key-forbidden contract.
