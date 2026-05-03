# HellaSwag Strict Multi-Slice And Systems Boundary References

Date: 2026-05-03

## Purpose

This memo records the citation and novelty boundary after the HellaSwag
hidden-innovation packet passed strict rank/score-channel controls on the
contiguous `0:9216` validation prefix but failed full-validation tail
jackknife stability.

## Benchmark And Communication Boundaries

- HellaSwag:
  https://arxiv.org/abs/1905.07830
  - Adversarially filtered commonsense multiple-choice benchmark. Requires
    frozen slice hashes and full paired per-item reporting to avoid slice
    cherry-picking concerns.
- CIPHER:
  https://openreview.net/forum?id=Yf7PaRar7T
  - Embedding-level debate/communication without sampled natural-language
    tokens. LatentWire must distinguish fixed-byte one-shot private packets
    from iterative embedding-token debate.
- Communicating Activations Between Language Model Agents:
  https://arxiv.org/abs/2501.14082
  - Direct activation exchange between LM agents. LatentWire must not claim
    generic activation communication novelty; its distinction is serialized
    fixed-byte packet transfer without raw activations.
- C2C / Cache-to-Cache:
  https://openreview.net/forum?id=LeatkxrBCi
  - Projects/fuses source KV cache into target KV cache. LatentWire differs
    only if it avoids source KV/cache transfer and pair-specific cache fusion.
- KVComm:
  https://arxiv.org/abs/2510.03346
  - Selective KV sharing; a direct systems competitor whose transmitted bytes
    scale with layers, tokens, heads, and hidden dimensions.
- KVCOMM:
  https://arxiv.org/abs/2510.12872
  - Cross-context KV reuse/offset alignment for multi-agent pipelines. It is a
    cache-reuse system, not a constant-byte semantic packet.
- Relative Representations:
  https://arxiv.org/abs/2209.15430
  - Common anchor-relative coordinates. Useful for future common-basis packets,
    but not a novel contribution unless tied to fixed-byte source-private
    downstream utility.
- Sparse Autoencoders:
  https://arxiv.org/abs/2309.08600
  - Sparse feature decomposition of LM activations. SAE-derived packets should
    be framed as an interpretable basis, not as the novelty.
- Universal SAEs:
  https://arxiv.org/abs/2502.03714
  - Shared sparse features across models. High overlap risk for any future
    cross-family common-feature alignment claim.
- Crosscoders:
  https://transformer-circuits.pub/2024/crosscoders/index.html
  - Shared/exclusive feature decomposition across models. The next packet
    branch should include feature-shuffle and shared-vs-private feature controls.
- Prefix-Tuning:
  https://aclanthology.org/2021.acl-long.353/
  and Gist Tokens:
  https://arxiv.org/abs/2304.08467
  - Soft prompts and prompt compression are prior art. LatentWire must keep
    zero-source/static-prefix/gist-style controls in any soft-token variant.
- Contrastive Activation Addition / activation steering:
  https://arxiv.org/abs/2312.06681
  - Global steering-vector methods. LatentWire must show per-item source
    innovation beyond a global or class-conditional steering vector.

## Systems And Quantization Boundaries

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
- SGLang / RadixAttention:
  https://arxiv.org/abs/2312.07104
- LMCache:
  https://docs.lmcache.ai/
- CacheGen:
  https://arxiv.org/abs/2310.07240
- KIVI:
  https://arxiv.org/abs/2402.02750
- QJL:
  https://arxiv.org/abs/2406.03482
- TurboQuant:
  https://arxiv.org/abs/2504.19874
  and
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- vLLM FP8 KV cache:
  https://vllm.ai/blog/fp8-kvcache
  and
  https://docs.vllm.ai/en/v0.15.0/features/quantization/quantized_kvcache/
- PyTorch MPS:
  https://docs.pytorch.org/docs/stable/notes/mps

The systems claim allowed by current Mac evidence is byte/privacy accounting:
the packet byte cost is fixed and source-private. Throughput, TTFT, ITL, HBM,
FP8/Triton/CUDA, cache-transfer, and serving-economics claims require NVIDIA
measurement.

## Novelty Boundary After This Gate

The defensible novelty is now:

1. a fixed-byte, source-private, per-example hidden-innovation packet;
2. positive HellaSwag decision utility over source label, source rank/index,
   source score, zero-hidden, wrong-row hidden, candidate-roll hidden, and
   score-channel-roll hidden controls;
3. strict paired uncertainty and train-sample jackknife stability on a
   contiguous `9216`-row frozen validation prefix;
4. explicit disclosure that full validation tail stability is unresolved.

Do not claim solved cross-model latent communication, cross-family transfer, or
native systems acceleration yet.

## Next Controls Implied

- Receiver-family/cross-family falsification with the same control ladder.
- Global steering-vector and class-conditional packet controls.
- SAE/crosscoder feature-shuffle controls if using common-feature packets.
- Same-byte text and gist/prefix controls for any soft-token receiver.
- NVIDIA table with quality, bytes, TTFT, ITL, throughput, HBM, peak memory,
  and source-exposure columns against C2C/KVComm/KVCOMM/vLLM/SGLang/LMCache,
  CacheGen, QJL, TurboQuant, KIVI, and vLLM FP8 KV.
