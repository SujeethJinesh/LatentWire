# GSM8K32 Layer-8 K/V Conditioning Control

Date: `2026-04-23`

## Why This Note Exists

Two narrower value-side conditioning screens were finite but not promotable:

- layer-8 `V`-only conditioning: seed `1` tied target, seed `0` fell below
  target
- all-layer `V`-only conditioning: seed `1` tied target, seed `0` again fell
  below target

This was the last simple whitening screen: condition both `K` and `V`, but only
at target layer `8`.

## Provenance

- code commit: `7906d2cde0c1802f61420a90d0b92dfdc67f2038`
- calibration file: `.debug/calibration_64.txt`
- calibration SHA256:
  `462d9313b8f0a8bd310860484ddc3d1fa128e5a3af8256ed409add554c33984a`
- materialized eval SHA256:
  `04d3006a6b37aa691347f290d442279bca23bbe119cf9a9b86002263fded20e1`

Each seed used a fresh result directory and the checkpoint namespace
`_srcwhite_tgtwhite_layers8`.

## Runs

Both runs used the exact frozen GSM8K32 contract with:

- base: `dynalign_module_replace`
- residual rank: `16`
- transport: `k_only`
- bits: `4`
- conditioning: `--whitening --target-whitening --conditioning-target-layer 8`

Commands:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_srcwhite_tgtwhite_l8_20260423 \
  --seed 1 \
  --whitening \
  --target-whitening \
  --conditioning-target-layer 8

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed0_srcwhite_tgtwhite_l8_20260423 \
  --seed 0 \
  --whitening \
  --target-whitening \
  --conditioning-target-layer 8
```

## Result

| Seed | Status | Accuracy | Correct | Win vs target | Loss vs target | Numeric coverage | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ok` | `0.0938` | `3/32` | `1` | `0` | `32/32` | `0` | `-` |
| 0 | `ok` | `0.0312` | `1/32` | `0` | `1` | `32/32` | `0` | `-` |

Seed `2` was not run because the seed-0 guard failed the promotion gate.

## Interpretation

Layer-8 `K/V` source+target conditioning is the strongest selective whitening
screen on the bad seed:

- it turns seed `1` from `checkpoint_nonfinite` into a finite positive row
- it preserves full numeric coverage
- it beats target on the frozen GSM8K32 contract

But it is still not a viable paper-method branch:

- seed `0` falls to `0.0312`, below target and below every acceptable floor
- the branch therefore fails the required seed-0 guard

This ends the simple whitening sweep. Conditioning reliably touches the
numerical failure surface, but every tested whitening variant erases the live
seed-0 signal.

## Updated Branch Read

- **Promoted:** layer-local preconditioning is the right area to manipulate.
- **Weakened:** simple source+target whitening as a paper-method fix.
- **Killed:** layer-8 `K/V` source+target conditioning as a promotable branch.

## Next Exact Gate

Stop simple whitening sweeps. The next robustness method should patch the
layer-8 `V` calibration fit more selectively:

1. implement a layer-8 `V` outlier-escrow / protected-channel calibration
   branch
2. run GSM8K32 seed `1`
3. if finite and positive, run seed `0`
4. promote only if seed `0 >= 0.0938`, then run seed `2` and only then reopen
   GSM70
