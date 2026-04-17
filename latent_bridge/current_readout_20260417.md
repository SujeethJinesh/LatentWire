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

Dense `K-only` branch:

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

Query-aware sparse `K-only` branch (`position_selection_ratio=0.5`):

- `k_only_target_attention_gate_0.10`: `0.057143`
- `k_only_target_attention_gate_0.15`: `0.057143`
- `k_only_target_attention_gate_0.10_ratio_0.25`: `0.028571`
- `k_only_target_attention_gate_0.10_ratio_0.75`: `0.042857`
- `k_only_attention_shuffled_gate_0.10`: `0.042857`
- `k_only_source_attention_gate_0.10`: `0.042857`
- `k_only_random_selector_gate_0.10`: `0.042857`
- `k_only_zero_byte_attenuation_gate_0.10`: `0.042857`
- `k_only_random_translated_gate_0.10`: `0.014286`
- `k_only_translated_only_gate_0.10`: `0.000000`
- transmitted payload at ratio `0.25`: `74,814.225` bytes
- transmitted payload at ratio `0.5`: `149,628.475` bytes
- transmitted payload at ratio `0.75`: `224,442.775` bytes

Paired checks on the same 70 examples:

- target-attention sparse vs target-alone:
  - delta `+0.014286`
  - real-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- target-attention sparse vs zero-byte attenuation:
  - delta `+0.014286`
  - real-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- target-attention sparse vs random selector:
  - delta `+0.014286`
  - real-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- target-attention sparse vs attention-shuffled selector:
  - delta `+0.014286`
  - real-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- target-attention sparse vs source-attention selector:
  - delta `+0.014286`
  - target-only wins `3`
  - source-only wins `2`
  - McNemar `p=1.0000`
- target-attention sparse ratio `0.5` vs ratio `0.25`:
  - delta `+0.028571`
  - ratio-0.5-only wins `2`
  - ratio-0.25-only wins `0`
  - McNemar `p=0.4795`
- target-attention sparse ratio `0.5` vs ratio `0.75`:
  - delta `+0.014286`
  - ratio-0.5-only wins `1`
  - ratio-0.75-only wins `0`
  - McNemar `p=1.0000`

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
- The first branch that now cleanly clears the matched sparse-selector controls
  is target-attention-guided sparse `K-only` transport.
- The effect is narrow but real on this split: at half-budget, target-attention
  sparse `K-only` reaches `0.057143`, while the matched zero-byte, random
  translated, random-selector, attention-shuffled, and source-attention
  controls fall back to `0.042857` or worse.
- The gap is still small: on this 70-example split it is effectively one extra
  solved example over the `0.042857` controls, so this is a best-current
  heuristic, not a statistically secure headline result.
- The rate-distortion curve is non-monotonic: `25%` is too sparse
  (`0.028571`), `50%` is best (`0.057143`), and `75%` / dense `100%` both
  fall back to `0.042857`. The current best branch is a real middle-band sparse
  selector, not “more retained KV is always better.”
- This shifts the best current interpretation from generic cross-model KV
  transfer to target-guided sparse key import.

## Paper State

This is still **not** an ICLR-ready main-paper result, but it is materially
better than the dense negative read.

The defensible statement now is:

- translated source KV alone is insufficient
- fused transport can avoid collapse relative to translated-only
- dense `K-only` still does not beat matched zero/random controls in a clean
  environment
- but target-guided sparse `K-only` does beat the matched selector and
  zero-byte controls on GSM8K held-out, albeit by a small non-significant
  margin on the current 70-example split

That means the best current path is no longer broad KV transport. It is
query-aware sparse key routing. The next most useful checks are:

1. rate-distortion on the target-attention sparse branch
2. attention-logit-preserving alignment / selection inside the kept positions
3. one second reasoning task before any broader model expansion
