# Byte-Level Interfaces and Cross-Tokenizer Transfer (2024-2026)

## Sources

- [Byte Latent Transformer: Patches Scale Better Than Tokens (2024)](https://arxiv.org/abs/2412.09871)
  - Byte-level LLMs with dynamic patching; key takeaway is that a fixed vocabulary is not required if the interface can adapt to local entropy.
- [Libra: Building Decoupled Vision System on Large Language Models (2024)](https://arxiv.org/abs/2405.10140)
  - Uses a unified discrete input tokenizer plus routed cross-modal bridging; relevant because it separates interface design from core model capacity.
- [Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries (2025)](https://arxiv.org/abs/2506.01535)
  - Vocabulary transfer via iterative subword removal and embedding estimation; directly relevant to retokenization and vocab remapping.
- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment (2025)](https://arxiv.org/abs/2506.03523)
  - Learns a token-ID alignment map and rearranges embeddings; useful as a lightweight cross-tokenizer initialization baseline.
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface (2026)](https://arxiv.org/abs/2604.07466)
  - Uses a shared byte interface for distillation; strongest direct evidence that byte-level transfer is a viable common ground.
- [CTPD: Cross Tokenizer Preference Distillation (2026)](https://arxiv.org/abs/2601.11865)
  - Maps teacher/student spans onto shared character-level structure; useful for credit assignment when token boundaries differ.
- [DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation (2026)](https://arxiv.org/abs/2602.21669)
  - Uses dual-space weighting plus Soft-DTW; relevant for sequence alignment when one-to-one token correspondence is unreliable.

## Why it matters for LatentWire

- LatentWire’s main failure mode is not just representation mismatch; it is interface mismatch. A shared byte-level or span-level bridge can reduce dependence on tokenizer compatibility before any learned communication policy is applied.
- TokAlign and dictionary-driven transfer suggest that a tokenizer bridge can be treated as a cheap initialization problem, not a full retraining problem.
- BLT and Libra both support the idea that the right interface can be simpler than the internal model: dynamic byte patches for text, or a unified discrete front end for multimodal data.
- CTPD and DWA-KD imply that if we keep tokenizers separate, we need explicit alignment machinery at the boundary; otherwise credit assignment becomes noisy and brittle.
- For cross-model communication, this makes byte-level fallback, span projection, and tokenizer remapping first-class ablations rather than implementation details.

## Concrete ablations/diagnostics

- Byte fallback vs subword bridge:
  - Compare a shared byte interface against the current tokenizer-specific interface on the same latent communication task.
  - Measure whether byte fallback improves transfer under tokenizer mismatch, especially on long-tail or OOV-heavy prompts.
- Token-ID remapping:
  - Add a TokAlign-style one-to-one token map as initialization and compare against random, lexical, and frequency-based mappings.
  - Report reconstruction, compression ratio, and downstream communication accuracy separately.
- Span projection diagnostics:
  - Evaluate whether character-span alignment improves teacher-student agreement when token boundaries differ.
  - Track per-span entropy and error concentration to see whether failures cluster at boundary crossings.
- Patch-size / segmentation sweeps:
  - Sweep byte-patch size or segmentation entropy thresholds to test whether interface granularity is the bottleneck.
  - Log accuracy vs. average patch length to expose compute/quality tradeoffs.
- Cross-tokenizer robustness:
  - Keep the latent method fixed and swap only tokenizers; report the same task under multiple vocabularies to isolate interface sensitivity.
- Failure attribution:
  - For every mistake, annotate whether the error came from interface mismatch, routing mismatch, or core latent decoding.
  - This is necessary so later benchmark pulls can distinguish “bad tokenizer” from “bad communication rule.”
