# GSM8K32 Conditioned Bad-Seed Controls

Date: `2026-04-23`

## Why This Note Exists

The runtime-only `V`-only anchor-tail wrapper was a clean negative: it did not
move the repeated layer-8 `V`-side checkpoint collapse upstream.

The next bounded question was simpler and lower risk:

1. do source whitening + target whitening remove the repeated bad-seed
   non-finite failure on the live `dynalign_module_replace_residrank16` lane?
2. if so, do they preserve the live ceiling or only the floor?

## Code / Harness Change

`scripts/run_gsm8k_contract_residual_sweep.py` now exposes `--whitening` and
`--target-whitening`, passes them through to `scripts/calibrate.py`, and makes
conditioning part of the checkpoint namespace:

- `_srcwhite`
- `_tgtwhite`

This prevents stale checkpoint reuse from being misread as a conditioning gain.

## Runs

All runs were on the exact frozen GSM8K32 contract with:

- base: `dynalign_module_replace`
- residual rank: `16`
- transport: `k_only`
- bits: `4`

Commands:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed0_srcwhite_tgtwhite_20260423 \
  --seed 0 \
  --whitening \
  --target-whitening

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_srcwhite_tgtwhite_20260423 \
  --seed 1 \
  --whitening \
  --target-whitening

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed2_srcwhite_tgtwhite_20260423 \
  --seed 2 \
  --whitening \
  --target-whitening
```

## Result

| Seed | Status | Accuracy | Correct | Win vs target | Loss vs target | Numeric coverage | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | `ok` | `0.0625` | `2/32` | `0` | `0` | `32/32` | `0` | `-` |
| 1 | `ok` | `0.0625` | `2/32` | `1` | `1` | `32/32` | `0` | `-` |
| 2 | `ok` | `0.0938` | `3/32` | `1` | `0` | `32/32` | `0` | `-` |

Compared with the unconditioned read:

- seed `0` dropped from the live `0.1250` row to a target tie
- seed `1` moved from `checkpoint_nonfinite` to a finite target tie
- seed `2` moved from `checkpoint_nonfinite` to a finite positive row above
  target

## Interpretation

This is the first real robustness clue since the seed-collapse note:

1. source + target whitening removes the catastrophic replicated bad-seed
   checkpoint failure on GSM8K32
2. the gain is not free:
   - it regularizes seed `0` down to the target floor
3. so conditioning currently looks like a stabilizer, not yet the final method

The right read is:

- **better than wrapper-only anchor-tail**
- **good enough to keep alive**
- **not yet promotable as the paper method**

## Updated Branch Read

Global conditioning is now the best bounded robustness clue:

- it fixes the worst failure mode on the small frozen slice
- it preserves a positive row on one previously bad seed
- but it currently trades away the best-seed ceiling

So the next method change should be more selective than global whitening:

1. layer-local or value-only conditioning, especially around the layer-8 `V`
   family
2. then the same seed controls again
3. only after that, reopen GSM70

## Next Exact Gate

Patch conditioning to be selective rather than global:

1. target the layer-8 `V` fit, or
2. target `V` only during alignment/conditioning,

and rerun the same GSM8K32 seed `0/1/2` control. Promote only if the bad seeds
stay finite **and** seed `0` recovers the old `0.0938-0.1250` floor.
