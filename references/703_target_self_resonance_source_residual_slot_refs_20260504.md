# Target Self-Resonance Source-Residual Slot References

Date: 2026-05-04

This memo records the literature and novelty boundary for the HellaSwag
TinyLlama-to-Qwen source-residual slot gate. The local result is a weak signal
and a failed method gate.

## Soft Prompt And Context Compression Boundary

- Prefix-Tuning learns continuous virtual-token prefixes for frozen language
  models:
  https://arxiv.org/abs/2101.00190
- Prompt Tuning studies learned soft prompts for frozen models:
  https://arxiv.org/abs/2104.08691
- Gist Tokens train models to compress prompts into cached special tokens:
  https://arxiv.org/abs/2304.08467
- AutoCompressors compress long contexts into soft-prompt summary vectors:
  https://arxiv.org/abs/2305.14788
- ICAE trains compact memory slots for in-context compression:
  https://arxiv.org/abs/2307.06945

These works mean that learned soft slots and prompt compression are not by
themselves novel. LatentWire must show source-conditioned communication beyond
target-only prompt compression.

## Query Bottleneck And Representation Boundary

- Perceiver uses latent query bottlenecks over high-dimensional inputs:
  https://arxiv.org/abs/2103.03206
- BLIP-2/Q-Former uses learned query tokens to connect frozen vision and
  language components:
  https://arxiv.org/abs/2301.12597
- SVCCA and CKA are standard representation-comparison tools:
  https://arxiv.org/abs/1706.05806
  https://arxiv.org/abs/1905.00414
- Relative Representations frame model communication through anchor-relative
  coordinates:
  https://arxiv.org/abs/2209.15430

These works motivate common-basis, anchor-relative, and residual-interface
diagnostics, but they also make generic representation alignment an unsafe
novelty claim.

## Communication And Systems Boundary

- C2C communicates by projecting/fusing source KV-cache state into a target:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm communicates through selected KV pairs:
  https://openreview.net/forum?id=F7rUng23nw
- Q-KVComm studies adaptive KV-cache compression for multi-agent
  communication:
  https://arxiv.org/abs/2512.17914
- KIVI, KVQuant, QJL, and TurboQuant are relevant low-bit source-state or
  KV/vector compression comparators:
  https://arxiv.org/abs/2402.02750
  https://arxiv.org/abs/2401.18079
  https://arxiv.org/abs/2406.03482
  https://openreview.net/forum?id=tO3ASKZlok

LatentWire should not claim to beat these systems natively on Mac. The allowed
claim is packet ABI, byte/exposure accounting, and Mac-local transport proxy
until NVIDIA serving rows exist.

## Diffusion, Coding, And Interpretability Inspiration

- DDPM and score-SDE diffusion motivate iterative denoising / residual repair:
  https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html
  https://openreview.net/forum?id=PxTIG12RRHS
- DiT and Diffusion-LM motivate transformer-based latent refinement:
  https://arxiv.org/abs/2212.09748
  https://arxiv.org/abs/2205.14217
- Sparse feature work motivates interpretable common-code diagnostics:
  https://transformer-circuits.pub/2023/monosemantic-features/index.html
  https://transformer-circuits.pub/2024/crosscoders/index.html

These are inspirations for codebook residuals, redundancy, and feature
readouts, not proof that the current residual-slot method is novel.

## Novelty Boundary

The defensible target is:

```text
matched source-conditioned residual packet
  > frozen target slots
  > zero-source residual
  > wrong-source residual
  > target-derived residual
  > random residual
  > direct source-label shortcut
```

with paired uncertainty, adjacent slices, seed repeats, and cross-family
separation. The current gate clears only the weaker part: matched source beats
target/zero/wrong/target-derived/random on one tiny slice, but it does not beat
source-top1 label copy and its paired lower bound is not positive.

## Promoted Next Branch

Move from generic continuous residual slots to a quantized
source-conditioned candidate repair or codebook residual interface. The next
gate should explicitly use source top-1/top-2 evidence to steer target
candidate preferences while preserving all destructive controls.
