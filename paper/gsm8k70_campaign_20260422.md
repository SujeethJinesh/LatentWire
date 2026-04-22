# GSM8K70 Campaign Summary

Date: `2026-04-22`

## Why This Matters

Reviewer feedback was right that GSM8K32 is too small and already
oracle-saturated to rank nearby variants. This larger frozen same-pair
campaign is the first useful check of whether the live
`dynalign_module_replace_residrank16` row survives beyond that tiny slice.

## Setup

- source -> target:
  `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- eval file: `data/gsm8k_eval_70.jsonl`
- frozen slice size: `70`
- candidate:
  `dynalign_module_replace_residrank16`
- seeds completed so far: `0`

Artifacts are local under:

- `results/gsm8k_contract_campaign_slice128_seed0_20260422/`

## Main Read

| Row | Accuracy | Correct |
|---|---:|---:|
| source_alone | 0.0143 | 1 |
| target_alone | 0.0571 | 4 |
| text_to_text | 0.0286 | 2 |
| rotalign_kv | 0.0714 | 5 |
| dynalign_module_replace_residrank16 | 0.1143 | 8 |
| c2c_generate | 0.1286 | 9 |
| oracle(target, candidate) | 0.1429 | 10 |

Paired read versus `target_alone` for the live candidate:

- wins: `6`
- losses: `2`
- ties: `62`
- delta mean: `+0.0571`
- one-seed paired bootstrap CI: `[-0.0143, 0.1429]`

## Interpretation

1. The live same-pair dynalign residual row survives beyond GSM8K32.
   - It improves from `4/70` to `8/70`.
2. The gain is still not paper-ready.
   - It remains one seed.
   - The paired interval still crosses zero.
   - It is still below the external smoke bar `c2c_generate = 9/70`.
3. The mechanism read remains interesting.
   - `source_alone` is only `1/70`.
   - the candidate gets `6` wins over target that source and text both miss
   - this still looks more like a transfer / repair effect than source answer
     copying
4. There is modest remaining headroom on this larger slice.
   - oracle(target, candidate) is `10/70`
   - only `2` examples remain above the candidate

## Reviewer Diagnostics

- source correctness on candidate-only wins: `0 / 6`
- text correctness on candidate-only wins: `0 / 6`
- source correctness on candidate-only losses: `0 / 2`
- text correctness on target-only text losses: `0 / 4`

This supports the current read:

- text relay still poisons the target on some target-correct cases
- the live latent row is not simply forwarding correct source answers

## What This Changes

1. GSM8K32 should remain only a smoke gate and falsification harness.
2. The next required step is multi-seed repetition on this larger frozen slice.
3. The next required falsification is one strict matched cross-family pair.
4. The next method budget should stay narrow:
   - codec-side: anchor-preserving selective precision / codebook tail
   - interface-side: learned query bottleneck / connector
