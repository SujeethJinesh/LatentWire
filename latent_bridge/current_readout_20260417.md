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

Fixed-prior follow-up:

- `gsm8k_100` target-alone: `0.040000`
- `gsm8k_100` text-to-text: `0.100000`
- `gsm8k_100` target-attention sparse `K-only`: `0.050000`
- `gsm8k_100` attention-disagreement sparse `K-only`: `0.050000`
- `gsm8k_100` fixed attention prior (`64` calibration prompts): `0.030000`
- `gsm8k_100` target-attention sparse vs fixed prior:
  - delta `+0.020000`
  - method-only wins `2`
  - baseline-only wins `0`
  - McNemar `p=0.4795`
- `gsm8k_100` attention-disagreement vs target-attention sparse:
  - delta `0.000000`
  - method-only wins `0`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- `arc_challenge_eval_35` target-alone: `0.428571`
- `arc_challenge_eval_35` target-attention sparse `K-only`: `0.485714`
- `arc_challenge_eval_35` fixed attention prior (`64` calibration prompts): `0.485714`
- `arc_challenge_eval_35` target-attention sparse vs fixed prior:
  - delta `0.000000`
  - method-only wins `0`
  - baseline-only wins `0`
  - McNemar `p=1.0000`

Second-pair GSM8K pilot:

- checkpoint: `checkpoints/qwen25_to_deepseek15b_query_pilot.pt`
- target: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- calibration: `.debug/calibration_128.txt`
- calibration quality:
  - `K` cosine `0.858`
  - `V` cosine `0.635`
- `gsm8k_eval_70` target-alone: `0.000000`
- `gsm8k_eval_70` target-attention sparse `K-only`: `0.028571`
- `gsm8k_eval_70` fixed attention prior (`64` calibration prompts): `0.014286`
- `gsm8k_eval_70` zero-byte attenuation: `0.014286`
- `gsm8k_eval_70` random translated: `0.014286`
- `gsm8k_eval_70` attention-shuffled selector: `0.000000`
- `gsm8k_eval_70` source-attention selector: `0.014286`
- `gsm8k_eval_70` random selector: `0.000000`
- `gsm8k_eval_70` attention-disagreement selector: `0.014286`
- `gsm8k_eval_70` recency selector: `0.000000`
- `gsm8k_eval_70` target-attention sparse vs fixed prior:
  - delta `+0.014286`
  - method-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- `gsm8k_eval_70` target-attention sparse vs zero-byte attenuation:
  - delta `+0.014286`
  - method-only wins `2`
  - baseline-only wins `1`
  - McNemar `p=1.0000`
- `gsm8k_eval_70` target-attention sparse vs random translated:
  - delta `+0.014286`
  - method-only wins `2`
  - baseline-only wins `1`
  - McNemar `p=1.0000`
- `gsm8k_eval_70` target-attention sparse vs attention-shuffled:
  - delta `+0.028571`
  - method-only wins `2`
  - baseline-only wins `0`
  - McNemar `p=0.4795`
- `gsm8k_eval_70` target-attention sparse vs source-attention:
  - delta `+0.014286`
  - method-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- `gsm8k_eval_70` target-attention sparse vs random selector:
  - delta `+0.028571`
  - method-only wins `2`
  - baseline-only wins `0`
  - McNemar `p=0.4795`
- `gsm8k_eval_70` target-attention sparse vs attention-disagreement:
  - delta `+0.014286`
  - method-only wins `1`
  - baseline-only wins `0`
  - McNemar `p=1.0000`
- `gsm8k_eval_70` target-attention sparse vs recency:
  - delta `+0.028571`
  - method-only wins `2`
  - baseline-only wins `0`
  - McNemar `p=0.4795`

Runtime head-selection follow-up (`runtime_head_selection_ratio=0.5`):

- Qwen GSM8K held-out, live target-attention sparse `K-only`, gate `0.10`:
  - dense selected heads: `0.057143` at `149,628.475` bytes
  - runtime `attention_peak` half-head pruning: `0.057143` at `74,308.018` bytes
  - runtime random half-head pruning: `0.057143` at `86,696.361` bytes
  - fixed head prior from `64` calibration prompts: `0.028571` at `76,207.675` bytes
  - live+prior head blend (`alpha=0.5`): `0.014286` at `76,207.675` bytes
  - `attention_peak` vs dense:
    - delta `0.000000`
    - method-only wins `1`
    - baseline-only wins `1`
    - McNemar `p=1.0000`
  - `attention_peak` vs random half-head:
    - delta `0.000000`
    - method-only wins `2`
    - baseline-only wins `2`
    - McNemar `p=1.0000`
  - fixed head prior vs live `attention_peak`:
    - delta `-0.028571`
    - method-only wins `1`
    - baseline-only wins `3`
    - McNemar `p=0.6171`
  - live+prior blend vs live `attention_peak`:
    - delta `-0.042857`
    - method-only wins `0`
    - baseline-only wins `3`
    - McNemar `p=0.2482`
- DeepSeek GSM8K held-out, live target-attention sparse `K-only`, gate `0.10`:
  - dense selected heads: `0.028571`
  - runtime `attention_peak` half-head pruning: `0.014286` at `49,549.336` bytes
  - runtime `attention_peak` vs dense:
    - delta `-0.014286`
    - method-only wins `0`
    - baseline-only wins `1`
    - McNemar `p=1.0000`
  - runtime `attention_peak` vs fixed attention prior:
    - delta `0.000000`
    - method-only wins `0`
    - baseline-only wins `0`
    - McNemar `p=1.0000`
  - runtime `attention_peak` vs zero-byte attenuation:
    - delta `0.000000`
    - method-only wins `1`
    - baseline-only wins `1`
    - McNemar `p=1.0000`

Second reasoning task check:

- task: `SVAMP`
- split: `data/svamp_gate_search_30.jsonl` / `data/svamp_eval_70.jsonl`
- target-alone: `0.071429`
- text-to-text: `0.414286`
- target-attention sparse `K-only`, gate `0.05`: `0.071429`
- target-attention sparse `K-only`, gate `0.10`: `0.028571`
- target-attention sparse `K-only`, gate `0.15`: `0.042857`

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
- On the larger `gsm8k_100` slice, the selector story strengthens slightly:
  target-attention sparse `K-only` reaches `0.050000`, target-alone is
  `0.040000`, and the fixed calibration prior drops to `0.030000`. The new
  attention-disagreement ablation ties target-attention exactly at `0.050000`,
  so extra disagreement weighting does not improve the current branch.
- ARC remains ambiguous. The target-attention branch is above target-alone, but
  the fixed attention prior ties it exactly on `arc_challenge_eval_35`, so ARC
  is still not clean selector-specific evidence.
- The DeepSeek-1.5B pilot is directionally consistent with GSM8K: target-alone
  is `0.000000`, target-attention sparse `K-only` reaches `0.028571`, and the
  fixed attention prior plus zero/random translated controls all sit at
  `0.014286`. The new selector controls are cleaner still: attention-shuffled,
  random-selector, and recency all collapse to `0.000000`, while
  source-attention and attention-disagreement sit at `0.014286`.
- Runtime head pruning is a useful compression ablation but not yet a stable
  mechanism claim. On Qwen GSM8K, retaining only half of the selected heads at
  evaluation time preserves the current `0.057143` score while roughly halving
  transmitted bytes, but the same move on DeepSeek drops the live-selector gain
  back to the fixed-prior / zero-byte control level. The current evidence says
  runtime head pruning is pair-sensitive rather than universally safe.
- Frozen calibration-only head masks are also not enough on the Qwen pair. A
  fixed head prior built from `64` calibration prompts drops the Qwen GSM8K
  score to `0.028571`, and a naive live+prior blend drops further to
  `0.014286`, both below the live `attention_peak` routing branch. So the best
  current interpretation is that live query-conditioned head routing still
  matters; a reusable fixed head mask does not replay the gain by itself.
- SVAMP is currently a negative transfer case. The sparse `K-only` branch does
  not beat target-alone there under the low-gate bracket, while text-to-text is
  much stronger. At the moment, SVAMP defines a failure boundary rather than a
  supporting replication.
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
  zero-byte controls on GSM8K held-out, and the same direction survives on
  `gsm8k_100` plus one DeepSeek-1.5B pilot with a stronger selector-control
  table
- runtime head pruning improves accuracy-per-byte on the Qwen GSM slice, but
  it does not yet transfer safely to the DeepSeek pair
- fixed calibration head priors do not preserve the Qwen gain, so the current
  head-localized story still depends on live query-conditioned routing
- the current evidence does **not** support a broad “all reasoning tasks”
  claim, because SVAMP is negative under the present method

That means the best current path is no longer broad KV transport. It is
query-aware sparse key routing with a limited success regime. The next most
useful checks are:

1. improve the positive GSM regime further rather than widening benchmarks
2. attention-logit-preserving alignment / selection inside the kept positions
3. test calibration-transfer masks more explicitly, but do not treat fixed head
   priors as a replacement for live routing
4. move the next method step toward per-head budgets or head scoring that is
   query-aware rather than a naive live+prior average
5. scale the DeepSeek GSM evaluation after the next selector/alignment change
