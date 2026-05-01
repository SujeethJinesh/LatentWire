# HellaSwag Trained-SAE and Validation[0:6144] Reference Note

## Local Result

This memo covers the May 1, 2026 HellaSwag follow-up in which:

- a trained sparse-autoencoder residual scout failed on validation
  `4096:5120`;
- the dense bagged hidden-innovation method passed the next frozen validation
  slice, `5120:6144`;
- the aggregate dense prefix now passes validation `0:6144`.

The safe paper claim is not that LatentWire invents sparse autoencoders,
crosscoders, or latent-space alignment. The safe claim is that a fixed-byte
source-private packet can carry useful source-hidden innovation under destructive
controls, while multiple common-basis compression attempts have failed so far.

## Closest Common-Basis Work

- Sparse autoencoders for monosemantic features establish the general idea of
  sparse activation dictionaries for language models:
  https://arxiv.org/abs/2309.08600
- SAE feature universality and related work study whether sparse features recur
  across models:
  https://arxiv.org/abs/2410.06981
- Universal Sparse Autoencoders explicitly learn a shared sparse concept space
  across multiple models:
  https://arxiv.org/abs/2502.03714
- Sparse Crosscoders learn shared sparse features across layers/models for
  model diffing and interpretability:
  https://transformer-circuits.pub/2024/crosscoders/index.html
- Relative representations use anchor-relative coordinates for latent-space
  comparison and zero-shot communication:
  https://arxiv.org/abs/2209.15430

Boundary: our trained-SAE scout is not novel as an SAE. It is a falsification
of whether a cheap train-only SAE is sufficient as the common basis for the
existing source-private packet selector.

## Prompt/Prefix Boundary

- Prefix-Tuning learns continuous prefix vectors for generation:
  https://arxiv.org/abs/2101.00190
- Prompt Tuning learns soft prompts for frozen models:
  https://arxiv.org/abs/2104.08691
- P-Tuning v2 extends deep prompt tuning:
  https://arxiv.org/abs/2110.07602
- Adapters insert persistent learned modules:
  https://arxiv.org/abs/1902.00751

Boundary: LatentWire does not insert a learned prefix, prompt, or adapter into
the target model. The communicated object is a per-example discrete packet with
source text, source KV, raw hidden vectors, and raw scores withheld.

## Systems and Quantization Boundary

- Cache-to-Cache transfers source KV-cache information via projection/fusion:
  https://arxiv.org/abs/2510.03215
- KVComm shares selected KV information:
  https://arxiv.org/abs/2510.03346
- QJL uses 1-bit Johnson-Lindenstrauss sketches for KV-cache quantization:
  https://arxiv.org/abs/2406.03482
- TurboQuant combines random rotation, scalar quantization, and QJL residual
  correction for online vector/KV quantization:
  https://arxiv.org/abs/2504.19874
- vLLM/PagedAttention is the relevant serving baseline family:
  https://arxiv.org/abs/2309.06180
- SGLang is another relevant serving baseline family:
  https://arxiv.org/abs/2312.07104

Boundary: current LatentWire systems evidence is byte/exposure accounting plus
Mac-local phase/RSS traces. It is not yet a native throughput, HBM, or goodput
win over C2C, KVComm, QJL, TurboQuant, vLLM, or SGLang.

## Reviewer Controls To Preserve

For any revived SAE/crosscoder method:

1. Beat best label-copy, trained label-bias, and score-only controls by at least
   `0.02` with paired CI95 low above zero.
2. Collapse under wrong-example hidden, candidate-roll hidden, atom-ID shuffle,
   atom-value/sign shuffle, and label-permuted SAE controls.
3. Show top-atom knockout removes a substantial fraction of the lift.
4. Report dictionary/SAE resident bytes, public/preloaded status, and whether
   sparse codes are transmitted.
5. Keep source text, source KV, raw hidden vectors, and raw score vectors out of
   the runtime packet.
