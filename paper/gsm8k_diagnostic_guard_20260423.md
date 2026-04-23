# GSM8K Diagnostic Guard

- date: `2026-04-23`
- live row: `dynalign_module_replace_residrank16`
- purpose: harden source-correctness and paired-flip diagnostics before any new
  method row is treated as evidence

## Why This Matters

The scalar `W_V.8` stabilization branches now make bad-seed checkpoints finite
but collapse to target parity. Before opening a learned connector branch, the
evaluation contract needs to reject artifact drift and distinguish real source
communication from target-cache regularization.

This pass upgrades `scripts/analyze_gsm8k_contract_diagnostics.py` to report:

- ordered and set `example_id` parity for source, target, text, and candidate
- numeric extraction coverage and empty-prediction counts for each row
- full target/candidate paired flip matrix
- exact two-sided sign p-value on helps vs harms
- source/text/candidate numeric-equality checks on candidate-only wins
- explicit diagnostic gate status

## Regenerated Artifacts

| Slice | Artifact | Gate |
|---|---|---|
| GSM8K32 seed 0 | `results/gsm8k_contract_residual_rank16_dynalign_20260421/dynalign_module_replace_residrank16_diagnostics_20260423.{json,md}` | `positive_noncopy_but_oracle_saturated` |
| GSM8K70 seed 0 | `results/gsm8k70_seed_repeat_full_20260422/seed0/dynalign_module_replace_residrank16_diagnostics_20260423.{json,md}` | `positive_noncopy_with_headroom` |
| GSM8K70 seed 3 | `results/gsm8k70_seed_repeat_full_20260422/seed3/dynalign_module_replace_residrank16_diagnostics_20260423.{json,md}` | `invalid_artifact` |

## Main Read

| Slice | Candidate | Target | Source | Text | Wins | Losses | Oracle | Source-copy wins | Latent non-copy wins | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GSM8K32 seed 0 | `4/32` | `2/32` | `1/32` | `1/32` | `2` | `0` | `4/32` | `0/2` | `2/2` | pass |
| GSM8K70 seed 0 | `8/70` | `4/70` | `1/70` | `2/70` | `6` | `2` | `10/70` | `0/6` | `6/6` | pass |
| GSM8K70 seed 3 | `2/70` | `4/70` | `1/70` | `2/70` | `1` | `3` | `5/70` | `0/1` | `1/1` | fail: candidate numeric coverage `69/70` |

## Interpretation

The source-copying hypothesis remains weak on the finite positive rows: every
candidate-only win on GSM8K32 seed 0 and GSM8K70 seed 0 has both source and
text wrong, and candidate numeric answers do not equal source or text numeric
answers on those wins.

The paper story is still not promotable. GSM8K70 seed 0 has real non-copy
headroom, but seed 3 is negative and now fails the stricter numeric-coverage
guard, while seeds 1 and 2 are still non-finite in the older campaign. The live
row stays a mechanism clue, not a stable method.

## Decision

Use this hardened diagnostic script as a required guard for future live rows.
Reject any claimed improvement that fails ID parity, numeric coverage, or the
source-control/copy-risk checks.

Next exact gate:

1. run shuffled-source and zero-source controls on GSM8K70 seed 0 and the next
   finite seed
2. if controls collapse to target level, pursue the 16-query learned resampler
   connector as the main method pivot
3. if controls preserve the win pattern, demote the live row to target-cache
   regularization and stop treating it as communication evidence

## Repro Commands

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_contract_diagnostics.py --candidate-prediction-output results/gsm8k_contract_residual_rank16_dynalign_20260421/dynalign_module_replace_residrank16.jsonl --candidate-label dynalign_module_replace_residrank16 --baseline-results-dir results/gsm8k_smoke_contract_20260421 --results-dir results/gsm8k_contract_residual_rank16_dynalign_20260421 --output-tag 20260423
./venv_arm64/bin/python scripts/analyze_gsm8k_contract_diagnostics.py --candidate-prediction-output results/gsm8k70_seed_repeat_full_20260422/seed0/dynalign_module_replace_residrank16.jsonl --candidate-label dynalign_module_replace_residrank16 --baseline-results-dir results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke --results-dir results/gsm8k70_seed_repeat_full_20260422/seed0 --output-tag 20260423
./venv_arm64/bin/python scripts/analyze_gsm8k_contract_diagnostics.py --candidate-prediction-output results/gsm8k70_seed_repeat_full_20260422/seed3/dynalign_module_replace_residrank16.jsonl --candidate-label dynalign_module_replace_residrank16 --baseline-results-dir results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke --results-dir results/gsm8k70_seed_repeat_full_20260422/seed3 --output-tag 20260423
```
