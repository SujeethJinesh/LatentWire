# Current Readout: 2026-04-17

This note supersedes the earlier 2026-04-17 GSM8K readout. The previously
reported `0.0571` / `0.0857` `K-only` gains did **not** survive a clean
environment rerun.

The clean reference environment for the results below is:

- repo-local `venv_arm64`
- `transformers==4.51.0`
- `tokenizers==0.21.1`
- `huggingface_hub==0.36.2`

Checkpoint:

- `checkpoints/gsm8k_method_upgrade_20260416/qwen25_to_qwen3_headhalf_lowrank_affine.pt`

Common evaluation setup:

- source: `Qwen/Qwen2.5-0.5B-Instruct`
- target: `Qwen/Qwen3-0.6B`
- split: `data/gsm8k_eval_70.jsonl`
- `source_reasoning_mode=brief_analysis`
- `kv_transport=k_only`
- `gate=0.10`
- quantized path unless noted

## GSM8K Held-Out

- `target_alone`: `0.042857`
- `text_to_text`: `0.100000`
- `k_only_cosine_dense`: `0.042857`
- `k_only_zero_byte_attenuation`: `0.042857`
- `k_only_random_translated`: `0.042857`
- `k_only_translated_only`: `0.000000`
- `k_only_static_position_ratio_0.5`: `0.000000`
- `k_only_disagreement_position_ratio_0.5`: `0.028571`
- `k_only_kalman_tokenwise`: `0.042857`
- `k_only_cosine_shifted_tokenwise`: `0.042857`

Paired checks on the same 70 examples:

- real `k_only_cosine_dense` vs zero-byte attenuation:
  - delta `0.0000`
  - real-only wins `1`
  - baseline-only wins `1`
  - McNemar `p=1.0000`
- real `k_only_cosine_dense` vs random translated:
  - delta `0.0000`
  - real-only wins `1`
  - baseline-only wins `1`
  - McNemar `p=1.0000`
- real `k_only_cosine_dense` vs translated-only:
  - delta `+0.042857`
  - real-only wins `3`
  - baseline-only wins `0`
  - McNemar `p=0.2482`

## Read

- The clean rerun removes the earlier source-communication win.
- On GSM8K, fused `K-only` currently ties `target_alone`, ties zero-byte
  attenuation, and ties random translated KV.
- Translated-only still collapses to `0.000000`, so target-side fusion matters,
  but there is no clean evidence that the fused gain comes from real source
  information on this checkpoint.
- Naive static sparse-position transport is actively harmful on this branch.
- Target/translation disagreement is a better sparse score than raw translated
  energy, but it still underperforms dense `K-only` and does not recover a
  source-specific win.
- Per-position adaptive fusion (`kalman_tokenwise`,
  `cosine_shifted_tokenwise`) did not recover a source-specific gain.

## Paper State

This is **not** currently an ICLR-ready positive result.

The defensible statement after the clean rerun is:

- translated source KV alone is insufficient
- fused transport can avoid collapse relative to translated-only
- but the current `K-only` branch does not beat matched zero/random controls in
  a clean environment

That means the next method step should be a real source-conditioned change, not
another scalar-gate or static-sparsity sweep. The best next candidates are:

1. query-aware key transport
2. attention-logit-preserving alignment / selection
3. reliability-weighted fusion that depends on source-target disagreement at
   inference time, not a fixed global rule
