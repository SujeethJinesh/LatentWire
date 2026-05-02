# Phi-3 8B Cross-Family Falsification and Uniqueness References

Date: 2026-05-02

## Purpose

This memo records the prior-work boundary after the strict Phi-3 `8B`/b2000
cross-family falsification. The current contribution must be framed as an
extreme-rate source-private evidence-packet protocol with rigorous controls,
not as a solved universal latent language.

## Closest Prior Work and Claim Boundaries

- Cache-to-Cache (C2C): https://arxiv.org/abs/2510.03215
  C2C transfers projected/fused source KV-cache state into a target model.
  Treat it as a high-bandwidth native latent/KV communication baseline.
- KVComm selective KV sharing: https://openreview.net/forum?id=F7rUng23nw
  Treat as a required source-KV exposure comparator for native systems rows.
- KVCOMM online cross-context KV reuse: https://arxiv.org/abs/2510.12872
  Treat as a multi-agent KV reuse baseline for TTFT/goodput comparisons.
- InterLat: https://arxiv.org/abs/2511.09149
  InterLat sends continuous hidden states with learned compression. It is
  motivation-adjacent but not a fixed-byte source-private evidence packet.
- Relative Representations: https://openreview.net/forum?id=SrC-nwieGJ
  This is the strongest public-anchor/common-coordinate precedent. LatentWire
  should not claim anchor coordinates themselves are novel.
- Prefix-Tuning: https://arxiv.org/abs/2101.00190
  Continuous prefixes are a baseline for learned conditioning, not the same as
  per-example source-private evidence packets.
- Gist Tokens: https://arxiv.org/abs/2304.08467
  Gist tokens motivate prompt-compression baselines and same-byte text controls.
- LLMLingua: https://arxiv.org/abs/2310.05736
  LLMLingua is a visible-prompt compression baseline.
- QJL: https://arxiv.org/abs/2406.03482
  QJL is the main 1-bit JL/sign-sketch KV-state byte-floor comparator.
- TurboQuant: https://arxiv.org/abs/2504.19874
  TurboQuant is the strongest current vector/KV quantization byte-rate pressure.
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
  Use vLLM for native serving rows and TTFT/TPOT/goodput reporting.
- SGLang/RadixAttention: https://arxiv.org/abs/2312.07104
  Use SGLang as a second native runtime and KV-reuse serving baseline.
- Sparse autoencoders: https://arxiv.org/abs/2309.08600
  SAEs motivate a future common-feature dictionary branch.
- Universal sparse autoencoders: https://arxiv.org/abs/2502.03714
  USAE-style shared sparse features are a high-value next connector direction
  after the Phi-3 packet-only falsification.
- Diffusion Transformers: https://arxiv.org/abs/2212.09748
  DiT is only inspiration for iterative packet repair/latent denoising.
- Consistency Models: https://arxiv.org/abs/2303.01469
  Consistency objectives motivate future one-step packet repair, but the current
  ARC learned-repair branches are negative.

## Safe Current Claims

- LatentWire has a seed-stable same-family Qwen-source ARC result at `8B`
  payload / `11B` framed, with paired uncertainty and destructive
  anchor/spectral controls.
- The Phi-3 `8B`/b2000 run is a strict negative cross-family result and should
  be reported as a claim boundary.
- The systems contribution currently supports byte/exposure accounting and a
  native ingest gate, not native throughput or HBM savings.

## Unsafe Current Claims

- Do not claim universal cross-family latent communication.
- Do not claim semantic anchor novelty over Relative Representations.
- Do not claim superiority to C2C, KVComm/KVCOMM, InterLat, QJL, TurboQuant,
  vLLM, or SGLang until matched native rows are measured.

