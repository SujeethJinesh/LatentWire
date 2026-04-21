# Tokenizer / Interface / Compression References

Scope: tokenizer-vocab transfer, byte- or patch-level interface simplification, multimodal connector transfer, discrete bottlenecks, diffusion-transformer interface ideas, and quantization-inspired communication/compression for cross-model reasoning.

## Sources

- `TokAlign: Efficient Vocabulary Adaptation via Token Alignment`  
  https://arxiv.org/abs/2506.03523
- `Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping`  
  https://arxiv.org/abs/2502.11104
- `Byte Latent Transformer: Patches Scale Better Than Tokens`  
  https://arxiv.org/abs/2412.09871
- `PaLM2-VAdapter: Progressively Aligned Language Model Makes a Strong Vision-language Adapter`  
  https://arxiv.org/abs/2402.10896
- `Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals`  
  https://arxiv.org/abs/2407.05996
- `UniForm: A Unified Diffusion Transformer for Audio-Video Generation`  
  https://arxiv.org/abs/2502.03897
- `KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache`  
  https://arxiv.org/abs/2402.02750
- `ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification`  
  https://arxiv.org/abs/2405.14256
- `AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration`  
  https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf
- Connector anchors worth keeping in view:
  - `BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models`  
    https://arxiv.org/abs/2301.12597
  - `Perceiver IO: A General Architecture for Structured Inputs & Outputs`  
    https://arxiv.org/abs/2107.14795

## Why it matters for us

- Tokenizer and vocab mismatch is a likely root cause when cross-model communication stalls before the latent/interface layer has a chance to help.
- Byte- or patch-level interfaces give us a way to simplify the communication contract without assuming token agreement.
- Q-Former / Perceiver-style narrow connectors are the cleanest analogue for a frozen-backbone + small learned bridge setup.
- Diffusion-transformer ideas matter because they treat interface repair as iterative refinement, not one-shot decoding.
- KIVI / ZipCache / AWQ show how to protect salient information under tight budgets; that maps well to protected latent channels or route banks.

## Concrete ablations / diagnostics

1. **Tokenizer transfer vs byte fallback**
   - Compare shared-tokenizer alignment, tokenizer-transfer, and byte-level fallback.
   - Log token overlap, effective sequence length inflation, and exact-answer retention under a fixed byte budget.

2. **Connector bottleneck sweep**
   - Compare plain projector, Q-Former-style query bottleneck, and Perceiver-style latent resampler.
   - Sweep query count / latent width / latent depth while holding the backbone frozen.
   - Diagnose whether gains come from capacity or from interface regularization.

3. **Protected communication under compression**
   - Compare dense communication, AWQ-style salient-channel protection, and KIVI/ZipCache-style compressed KV analogues.
   - Log route entropy, outlier coverage, retained-byte ratio, and answer accuracy together.

