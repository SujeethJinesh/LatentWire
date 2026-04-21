# Lateral Breakthrough References for LatentWire

Primary-source memo for the 2025-2026 literature that is most likely to move LatentWire: direct cross-model communication, tokenizer/vocab bridges, multimodal latent interfaces, iterative latent refinement, symmetry/gauge alignment, and quantization-inspired compression.

## Cross-model communication

- `Cache-to-Cache (C2C)` (2025) is the closest semantic-communication baseline: it projects and fuses source KV with target KV instead of emitting text. Links: [paper](https://arxiv.org/abs/2510.03215), [code](https://github.com/thu-nics/C2C).
- `KVComm: Enabling Efficient LLM Communication through Selective KV Sharing` (2025) is the cleanest shared-KV baseline when communication is sparse rather than fully fused. Link: [paper](https://arxiv.org/abs/2510.03346).
- `KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems` (2025) is the training-free, anchor-based reuse baseline for overlapping contexts. Link: [paper](https://arxiv.org/abs/2510.12872).
- `Q-KVComm` (2025) is useful if we want to study communication plus compression together; it quantizes and calibrates transmitted cache state across heterogeneous models. Link: [paper](https://arxiv.org/abs/2512.17914).

## Tokenizer / vocab bridges

- `Cross-Tokenizer Distillation via Approximate Likelihood Matching` (2025) is the best recent proof that pure distillation can work across mismatched tokenizers and vocabularies. Link: [paper](https://arxiv.org/abs/2503.20083).
- `TokAlign` (2025) replaces vocabulary by learned token-ID alignment, then reuses the rearranged embeddings for fast adaptation. Link: [paper](https://arxiv.org/abs/2506.03523).
- `zip2zip` (2025) is the strongest current adaptive-vocabulary baseline if we want inference-time token compression rather than static remapping. Link: [paper](https://arxiv.org/abs/2506.01084).
- `Model-Aware Tokenizer Transfer` / `MATT` (2025) matters because it distills attention influence, not just embedding similarity, into the new tokenizer warm-start. Link: [paper](https://arxiv.org/abs/2510.21954).
- `Cross-Chain of Thought Distillation via Optimal Transport Alignment for Language Models with Different Tokenizers` (2025) is a useful reasoning-aware tokenizer-bridge control. Link: [paper](https://arxiv.org/abs/2502.16806).
- `Cross-Tokenizer LLM Distillation through a Byte-Level Interface` (2026) is the newest “common interface” result and should be kept in mind as a fallback design. Link: [paper](https://arxiv.org/abs/2604.07466).

## Multimodal latent interfaces

- `LatentLM` (2024/2025) is still the cleanest general-purpose multimodal interface: continuous latents for media, discrete tokens for text, one causal Transformer. Links: [paper](https://arxiv.org/abs/2412.08635), [OpenReview](https://openreview.net/forum?id=YSLFKaVTWL).
- `Latent Speech-Text Transformer` (2025) is the most relevant speech/text bridge because it reduces compute by aggregating speech tokens into latent patches. Link: [paper](https://arxiv.org/abs/2510.06195).
- `Bifrost-1` (2025) is the best current example of a patch-level CLIP latent bridge between pretrained MLLMs and diffusion models. Link: [OpenReview](https://openreview.net/forum?id=z0WhTwZscg).
- `One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers` (2025) is the current “variable-length latent interface” paper to watch. Link: [OpenReview](https://openreview.net/forum?id=fRw11pjvTF).

## Diffusion / iterative latent refinement

- `Transition Matching` (2025) is the strongest general template for discrete-time, continuous-state transitions that unify diffusion and continuous AR generation. Link: [paper](https://arxiv.org/abs/2506.23589).
- `FS-DFM` (2025) is the best few-step language diffusion reference when we care about step-budget control and consistent updates. Link: [paper](https://arxiv.org/abs/2509.20624).
- `Latent Refinement Decoding` (2025) is the most directly transferable idea for LatentWire: keep uncertain positions as mixtures, then refine until confidence stabilizes. Link: [paper](https://arxiv.org/abs/2510.11052).
- `Stop-Think-AutoRegress / STAR-LDM` (2025) is useful if we want to mix latent planning with AR finalization. Link: [OpenReview](https://openreview.net/forum?id=c05qIG1Z2B).
- `VDLM` (2026) is the current modular latent-planning plus text-rendering reference. Link: [paper](https://arxiv.org/abs/2602.15870).

## Symmetry / gauge alignment

- `Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion` (2025) is the key recent source for continuous symmetry-aware parameter matching. Link: [paper](https://arxiv.org/abs/2502.00264).
- `Latent Merging: Dynamic and Reversible Composition of Large Language Models` (2025) is the most relevant state-space composition paper if we want reversible mixing rather than weight surgery. Link: [OpenReview](https://openreview.net/forum?id=ocEoHCrezd).
- `CASK: A Gauge Covariant Transformer for Lattice Gauge Theory` (2025) is not an LLM paper, but it is a clean modern example of making attention respect gauge structure. Link: [paper](https://arxiv.org/abs/2501.16955).

## Quantization-inspired compression

- `AWQ` (2023) remains the baseline “protect salient channels using activation statistics” reference. Link: [paper](https://arxiv.org/abs/2306.00978).
- `EXL2` is the practical mixed-bit deployment baseline; the official ExLlamaV2 docs describe 2-8 bit mixed quantization with per-layer calibration. Links: [repo](https://github.com/turboderp-org/exllamav2), [newer repo](https://github.com/turboderp-org/exllamav3).
- `TurboQuant` (2025) is the strongest theoretical compression reference if we want near-optimal distortion-rate control and unbiased inner-product preservation. Link: [paper](https://arxiv.org/abs/2504.19874).
- `TurboAngle` (2026) is the latest angular-quantization KV-cache result and is useful for per-layer bit allocation without calibration. Link: [paper](https://arxiv.org/abs/2603.27467).
- `InnerQ` (2026) is the current hardware-aware KV-cache quantization paper to watch if runtime dequantization cost matters. Link: [paper](https://arxiv.org/abs/2602.23200).

## Concrete LatentWire ablations

- Compare `text-only relay`, `latent relay`, and `latent relay + quantized payload` at matched source-target budgets.
- For tokenizer bridges, run `shared tokenizer`, `ID-alignment`, `byte-level interface`, and `attention-aware tokenizer transfer` under identical compute.
- For multimodal bridges, swap `token interface`, `latent patch interface`, and `latent patch interface + small renderer` while holding the backbone fixed.
- For refinement, compare `1 / 2 / 4 / 8` steps with the same wall-clock budget, plus a no-op control that spends the same compute without changing state.
- For routing, test `hard commit`, `mixture belief state`, and `entropy-gated skip` on the same examples.
- For symmetry, compare `unaligned`, `permutation-aligned`, and `rotation-aligned` source-target pairing before any learned projector.
- For compression, sweep `8/6/4/2` bits and record whether source quality drops come from quantization noise or from missing semantic transport.
- For cross-model claims, always include `source-alone`, `target-alone`, `text bridge`, and the best cache baseline (`C2C` or `KVComm`).

## Telemetry fields to log

- Communication budget: bytes moved, tokens moved, active layers, active heads, bits per element, and end-to-end latency.
- Alignment quality: source-target KL/JS, hidden-state CKA or Procrustes error, top-1 agreement, and token-fragmentation rate.
- Refinement dynamics: per-step entropy, update norm, cosine drift, commit rate, rollback rate, and early-stop step.
- Routing behavior: head-selection entropy, route overlap, dead-slot rate, and source/target asymmetry by layer.
- Quantization behavior: per-layer reconstruction error, outlier rate, saturation rate, sink-token retention, and memory-bandwidth savings.
- Multimodal bridge behavior: latent length, reconstruction fidelity, modality imbalance, and renderer error when decoding latent outputs back to text.

## What not to overclaim

- Do not claim universal interop. These methods show controlled transfer between specific source-target pairs, not a general solution for arbitrary model families.
- Do not claim “semantic preservation” from compression alone. Low-bit cache packing can preserve throughput while still destroying task-critical structure.
- Do not claim tokenizer bridges solve reasoning transfer in general. They mostly reduce fragmentation and alignment friction.
- Do not claim latent interfaces are modality-agnostic by default. Patch-level CLIP, speech patches, and text latents each need their own tuning.
- Do not claim iterative refinement is always better. Gains depend on budget, stopping rule, and whether uncertainty is actually tracked.
- Do not claim symmetry alignment makes checkpoints equivalent. It helps matching/fusion, but function-preserving equivalence is much narrower.

## Paper-facing framing

LatentWire should read as a bandwidth-controlled latent transport layer with selective refinement, not as a universal translator or universal model merger. The strongest publishable claim is that LatentWire preserves more task-relevant structure per byte than text relay or naive cache compression, and that this comes from explicit alignment, routing, and stopping signals rather than from extra compute alone.
