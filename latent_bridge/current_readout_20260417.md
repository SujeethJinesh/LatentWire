# Current Readout: 2026-04-17

This note captures the latest fixed-gate control results on the current best
`Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` checkpoint:
`checkpoints/gsm8k_method_upgrade_20260416/qwen25_to_qwen3_headhalf_lowrank_affine.pt`.

## Setup

- `source_reasoning_mode=brief_analysis`
- `kv_transport=k_only`
- main fused control uses `fusion_rule=cosine_shifted`
- fixed gate confirmation uses `gate=0.15`

## GSM8K held-out (`data/gsm8k_eval_70.jsonl`)

- `target alone`: `0.0143`
- `text-to-text`: `0.1143`
- `k_only_static_g015`: `0.0143`
- `k_only_cosine_shifted_g015`: `0.0571`
- `k_only_cosine_shifted_g015_noquant`: `0.0429`
- `k_only_translated_only`: `0.0000`
- `k_only_attenuation_zero_g015`: `0.0429`
- `k_only_random_translated_g015`: `0.0429`

Read:

- The best current latent path is still fused `K-only + cosine_shifted`.
- It beats `target alone`, static fusion, translated-only, and the zero/random
  target-space controls on this split.
- It still trails `text-to-text`.
- The quantized fused branch is currently better than the no-quant anchor on
  this exact setup.

Gate bracket around the current best setting:

- `gate=0.10`: `0.0571`
- `gate=0.15`: `0.0571`
- `gate=0.20`: `0.0429`

So the useful region looks like low gates, with `0.20` already too strong.

## ARC Challenge held-out (`data/arc_challenge_eval_35.jsonl`)

- `target alone`: `0.4571`
- `text-to-text`: `0.3714`
- `k_only_static_g015`: `0.5143`
- `k_only_cosine_shifted_g015`: `0.4857`
- `k_only_translated_only`: `0.3143`
- `k_only_attenuation_zero_g015`: `0.4857`
- `k_only_random_translated_g015`: `0.4571`

Read:

- ARC still does not support the strongest source-communication story.
- `K-only static` is best on ARC, but zero-byte attenuation ties
  `K-only + cosine_shifted`.
- That keeps ARC in the "cache intervention / control-heavy" bucket rather than
  the clean latent-transfer bucket.

## Paper State

This is not yet enough for an ICLR main-paper claim that cross-model latent KV
communication robustly improves reasoning.

The current defensible story is narrower:

- useful signal is concentrated in translated keys, not values
- target-side values are still necessary
- low-gate fused key transfer can help GSM8K more than matched zero/random
  controls
- ARC remains a limitation and an important confound check

The next method work should focus on strengthening the reasoning-side win
without collapsing back into a zero-byte attenuation story.
