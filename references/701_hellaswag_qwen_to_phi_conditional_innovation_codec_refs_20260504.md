# HellaSwag Qwen-To-Phi Conditional Innovation Codec References

Date: 2026-05-04

## Purpose

This memo records the literature boundary for the failed conditional innovation
codec gate. The result should be treated as a ruled-out shallow method, not a
new positive ICLR claim.

## Side-Information And Residual Coding

- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources",
  IEEE Transactions on Information Theory 1973.
  https://doi.org/10.1109/TIT.1973.1055037
- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder", IEEE Transactions on Information Theory 1976.
  https://doi.org/10.1109/TIT.1976.1055508
- Shannon, "Coding Theorems for a Discrete Source With a Fidelity Criterion",
  1959. https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf

Boundary: Phi's local score simplex is decoder-side information, and the Qwen
packet is a residual correction. This is an empirical packet gate, not a
formal new side-information coding theorem.

## Prompt / Prefix / Gist Token Boundary

- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
  ACL 2021. https://arxiv.org/abs/2101.00190
- Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning",
  EMNLP 2021. https://arxiv.org/abs/2104.08691
- Liu et al., "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning
  Universally Across Scales and Tasks", ACL 2022. https://arxiv.org/abs/2110.07602
- Mu et al., "Learning to Compress Prompts with Gist Tokens", NeurIPS 2023.
  https://arxiv.org/abs/2304.08467

Boundary: these optimize continuous or learned prompt tokens for a target
model. LatentWire's current gate transmits a discrete fixed-byte packet
derived from source evidence and then tests source-destroying controls.

## Cross-Model / KV Communication Boundary

- C2C, cache-to-cache cross-model communication.
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm/KVCOMM, selective KV communication.
  https://openreview.net/forum?id=F7rUng23nw
- DroidSpeak, cross-LLM communication through optimized messages.
  https://arxiv.org/abs/2411.02820

Boundary: those methods communicate text-like messages, hidden state, or
KV/cache-derived objects. This gate communicates a byte-scale score-residual
packet and forbids source text, KV, hidden vectors, logits, or raw score
vectors at the receiver.

## Quantization And Systems Inspiration

- KIVI: "A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024.
  https://arxiv.org/abs/2402.02750
- KVQuant: "KVQuant: Towards 10 Million Context Length LLM Inference with KV
  Cache Quantization", NeurIPS 2024. https://arxiv.org/abs/2401.18079
- QJL: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead", 2024. https://arxiv.org/abs/2406.03482
- TurboQuant, ternary/online quantization for LLM inference.
  https://arxiv.org/abs/2504.19874

Boundary: these works compress KV or vector state for systems efficiency. The
conditional innovation codec borrows residual/low-bit coding intuition, but it
does not claim native systems superiority or KV-cache compression.

## Representation And Shared-Basis Context

- Sparse crosscoders, Anthropic Transformer Circuits, 2024.
  https://transformer-circuits.pub/2024/crosscoders/index.html
- Cross-architecture sparse crosscoders for model representations.
  https://arxiv.org/abs/2602.11729
- Kornblith et al., "Similarity of Neural Network Representations Revisited",
  ICML 2019. https://arxiv.org/abs/1905.00414

Boundary: crosscoders and representation similarity analysis motivate shared
or aligned bases. The failed gate did not learn an interpretable common basis;
it only tested a shallow Phi-predicted Qwen residual.

## Diffusion / Iterative Refinement Context

- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
  https://arxiv.org/abs/2006.11239
- Peebles and Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
  https://arxiv.org/abs/2212.09748
- Yu et al., "Representation Alignment for Generation: Training Diffusion
  Transformers Is Easier Than You Think", ICLR 2025.
  https://openreview.net/forum?id=DJSZGGZYVi

Boundary: diffusion motivates denoising/refinement language, but this gate is a
one-shot receiver over MCQ scores. Do not frame it as a diffusion-transformer
technical contribution.

## Claim After This Gate

Safe:

- A fixed-byte conditional innovation packet was implemented and tested on the
  frozen Qwen-to-Phi HellaSwag `1024:2048` surface.
- The packet improves over ghost-only and destructive controls only by
  degenerating to fixed hybrid.
- Linear ghost prediction plus discrete residual coding is insufficient.

Unsafe:

- Claiming solved cross-model latent reasoning.
- Claiming novelty over formal side-information or rate-distortion theory.
- Claiming equivalence to prefix/gist token methods.
- Claiming systems wins over C2C, KVComm, QJL, TurboQuant, KIVI, KVQuant,
  vLLM, or SGLang.
