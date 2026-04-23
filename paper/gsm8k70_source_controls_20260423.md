# GSM8K70 Source-Control Readout

- date: `2026-04-23`
- live row: `dynalign_module_replace_residrank16`, seed `0`
- implementation commit: `4408e61a`
- readout: `results/gsm8k70_source_controls_20260423/seed0/source_control_readout_20260423.md`
- live artifact: `results/gsm8k70_seed_repeat_full_20260422/seed0/dynalign_module_replace_residrank16.jsonl`
- zero-source artifact: `results/gsm8k70_source_controls_20260423/seed0/zero_source.jsonl`
- shuffled-source artifact: `results/gsm8k70_source_controls_20260423/seed0/shuffled_source_salt0.jsonl`

## What Changed

`latent_bridge/evaluate.py` now supports `--source-prompt-control
shuffle_examples`, a deterministic cross-example source derangement controlled
by `--random-salt`. The target prompt, target example id, decoding contract,
checkpoint, and scoring path stay fixed. Prediction records now log
`source_prompt_control`, `source_control_source_index`, and
`source_control_source_example_id`.

`scripts/analyze_gsm8k_source_controls.py` was added to compare a live row
against source controls with exact id parity, paired flips vs target/live,
live-win retention, control-only wins, control telemetry, and validity flags.

## Results

| Row | Correct | Pair vs target | Pair vs live | Live-win retention | Numeric coverage | Source deranged |
|---|---:|---:|---:|---:|---:|---:|
| target_alone | `4/70` | `0/0/70` | n/a | n/a | `70/70` | n/a |
| live matched-source | `8/70` | `6/2/62` | n/a | `6` wins | `70/70` | no |
| zero_source | `0/70` | `0/4/66` | `0/8/62` | `0/6` | `0/70` | no |
| shuffled_source_salt0 | `0/70` | `0/4/66` | `0/8/62` | `0/6` | `1/70` | yes |

## Interpretation

This weakens the target-cache explanation for the live seed-0 row: both
zero-source and cross-example shuffled-source controls erase every live
candidate-only win and produce no control-only wins. The shuffled control has
ordered id parity and telemetry confirms every source prompt was deranged.

It does not yet clear the strict reviewer gate. Both controls collapse through
poor numeric coverage and many empty/non-numeric generations, so the current
evidence says the live row depends on matched source state, but the control
surface is too destructive to claim a clean, validity-preserving source
dependence result.

## Status

- alive: matched-source same-family transfer signal on seed `0`; live wins are
  not retained by zero or shuffled sources.
- weakened: the live branch as a reviewer-ready positive method, because the
  controls are validity-weak and finite seed repeats remain unstable/negative.
- saturated: closed-form `W_V.8` ridge/protected-ridge stabilization; it fixes
  nonfinites but collapses to target parity.
- blocked: paper claim that the method robustly communicates useful source
  information rather than poisoning or overfitting the target prefix.

## Next Gate

Run a validity-preserving mismatch control before widening benchmarks:

1. Keep matched target prompt and exact GSM8K70 seed-0 slice.
2. Use shuffled source examples, but add a target-preserving safety path:
   runtime verifier/fallback, calibrated fusion shrinkage, or learned
   bottleneck/resampler that cannot catastrophically poison the target prefix.
3. Require full id parity, no empty predictions, high numeric coverage, and
   collapse of live-win retention under mismatch.

If that clears, repeat on the next finite valid seed and then run one strict
cross-family falsification pair. If it fails, demote the live row to a brittle
source-state perturbation and pivot to a safer connector method.
