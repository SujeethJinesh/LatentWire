# GSM8K32 V-Only All-Layer Conditioning Control

Date: `2026-04-23`

## Why This Note Exists

Exact layer-8 `V`-only source+target conditioning was finite but too narrow:
seed `1` only tied target and seed `0` fell below target.

The next broader test kept the evidence-backed `V` stream restriction but
removed the layer allowlist:

1. condition the `V` stream on every target layer
2. leave the `K` anchor path unconditioned
3. keep the live `dynalign_module_replace_residrank16` row otherwise unchanged

## Provenance

- code commit: `0cc6ba434dfa80ae2c64c39ee76a561afc7e2828`
- calibration file: `.debug/calibration_64.txt`
- calibration SHA256:
  `462d9313b8f0a8bd310860484ddc3d1fa128e5a3af8256ed409add554c33984a`

Each seed used a fresh result directory and a distinct checkpoint namespace:
`_srcwhitev_tgtwhitev`.

## Runs

Both runs used the exact frozen GSM8K32 contract with:

- base: `dynalign_module_replace`
- residual rank: `16`
- transport: `k_only`
- bits: `4`
- conditioning: `--whitening --whitening-streams v --target-whitening
  --target-whitening-streams v`

Commands:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_srcwhitev_tgtwhitev_20260423 \
  --seed 1 \
  --whitening \
  --target-whitening \
  --whitening-streams v \
  --target-whitening-streams v

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed0_srcwhitev_tgtwhitev_20260423 \
  --seed 0 \
  --whitening \
  --target-whitening \
  --whitening-streams v \
  --target-whitening-streams v
```

## Result

| Seed | Status | Accuracy | Correct | Win vs target | Loss vs target | Numeric coverage | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ok` | `0.0625` | `2/32` | `1` | `1` | `32/32` | `0` | `-` |
| 0 | `ok` | `0.0312` | `1/32` | `0` | `1` | `32/32` | `0` | `-` |

Seed `2` was not run because the seed-0 guard already failed the promotion
gate.

## Interpretation

All-layer `V`-only source+target conditioning is also a real finiteness
stabilizer, but it is not a viable paper-method branch:

- seed `1` stays finite but only ties target
- seed `0` again falls below target and below the global-conditioning guard

This weakens the hypothesis that a simple `V`-stream whitening restriction is
enough to preserve the live signal while removing checkpoint collapse.

## Updated Branch Read

- **Promoted:** conditioning/preconditioning remains the right failure surface;
  both selective `V` variants are finite.
- **Weakened:** `V`-only source+target whitening, even across all layers.
- **Killed:** exact all-layer `V`-only conditioning as a promotable paper
  branch.

## Next Exact Gate

Run one broader coupling screen before leaving whitening entirely:

1. layer-8 source+target conditioning across both `K/V`
2. if that also preserves finiteness but kills seed `0`, stop whitening sweeps
   and move to layer-8 `V` outlier-escrow smoothing / protected-channel
   calibration.

Do not reopen GSM70 until a GSM8K32 conditioning branch keeps bad seeds finite
and restores seed `0 >= 0.0938`.
