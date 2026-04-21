# Tokenizer, Connector, Diffusion, and Quantization References

Scope: tokenizer/vocab transfer, narrow multimodal connectors, diffusion-style iterative refinement, and quantization-inspired communication rules that can become actual LatentWire ablations.

## Exact source links

- `Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching`  
  https://www.alphaxiv.org/overview/2503.20083v4
- `Model-Aware Tokenizer Transfer`  
  https://arxiv.org/abs/2504.17013
- `Tokenizer-Aware Cross-Lingual Adaptation`  
  https://arxiv.org/abs/2604.07466
- `BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models`  
  https://arxiv.org/abs/2301.12597
- `Perceiver IO: A General Architecture for Structured Inputs & Outputs`  
  https://arxiv.org/abs/2107.14795
- `Perceiver IO` local runnable implementation  
  https://github.com/krasserm/perceiver-io
- `AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration`  
  https://github.com/mit-han-lab/llm-awq
- `EXL2 / ExLlamaV2`  
  https://github.com/turboderp-org/exllamav2
- `KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction`  
  https://arxiv.org/abs/2505.23416
- `KVPress`  
  https://github.com/NVIDIA/kvpress
- `ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification`  
  https://arxiv.org/abs/2405.14256
- `Diffusion-style iterative refinement` reference lane  
  https://arxiv.org/abs/2407.13828

## Why these belong together

Tokenizer/vocab transfer, Q-Former/Perceiver connectors, diffusion-style refinement, and AWQ/EXL2/KV compression are all the same structural question:

- how much information do we pass through the interface,
- what do we protect as salient,
- and how do we refine or reconstruct the missing parts.

That makes them good candidates for a single ablation family rather than independent one-off experiments.

## Three concrete ablations

1. **Tokenizer transfer vs byte-level fallback**
   - Compare a shared or transferred tokenizer against the current tokenizer boundary and a byte-level fallback.
   - Log token overlap, sequence-length inflation, and exact-answer retention under a fixed byte budget.
   - Success criterion: better answer exactness without larger effective communication cost.

2. **Narrow connector family sweep**
   - Compare plain projector, Q-Former-style query bottleneck, and Perceiver-style latent resampler.
   - Keep backbone frozen and vary only connector width, query count, and latent depth.
   - Success criterion: a small latent bank should match or beat the plain projector while staying interpretable.

3. **Protected-communication compression**
   - Compare dense communication against AWQ-style salient-channel protection, EXL2-like mixed-bit packing, and KVzip/ZipCache-style cache compression analogues.
   - Log route entropy, outlier coverage, retained-byte ratio, and exact-answer accuracy together.
   - Success criterion: the protected channel preserves accuracy at a lower byte budget than dense communication.

## Practical note

If we run a refinement variant, keep the iteration count tiny first and measure whether the second pass actually fixes a failure mode or only adds compute. The useful analogue is diffusion-style refinement, not unbounded self-editing.

