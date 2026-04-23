# GSM8K32 Layer-8 V-Only Conditioning Control

Date: `2026-04-23`

## Why This Note Exists

Global source+target whitening removed the repeated bad-seed checkpoint
collapse on the live `dynalign_module_replace_residrank16` lane, but it also
flattened the seed-0 ceiling from `0.1250` to a target tie.

The narrowest follow-up was:

1. condition only the value stream, where the repeated bad keys localize
2. condition only target layer `8`, where the first non-finite key appears
3. keep the live `dynalign_module_replace_residrank16` row otherwise unchanged

## Runs

Both runs used the exact frozen GSM8K32 contract with:

- base: `dynalign_module_replace`
- residual rank: `16`
- transport: `k_only`
- bits: `4`
- conditioning: `--whitening --whitening-streams v --target-whitening
  --target-whitening-streams v --conditioning-target-layer 8`

Commands:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed1_srcwhitev_tgtwhitev_l8_20260423 \
  --seed 1 \
  --whitening \
  --target-whitening \
  --whitening-streams v \
  --target-whitening-streams v \
  --conditioning-target-layer 8

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_resid16_seed0_srcwhitev_tgtwhitev_l8_20260423 \
  --seed 0 \
  --whitening \
  --target-whitening \
  --whitening-streams v \
  --target-whitening-streams v \
  --conditioning-target-layer 8
```

## Result

| Seed | Status | Accuracy | Correct | Win vs target | Loss vs target | Numeric coverage | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `ok` | `0.0625` | `2/32` | `0` | `0` | `32/32` | `0` | `-` |
| 0 | `ok` | `0.0312` | `1/32` | `0` | `1` | `32/32` | `0` | `-` |

Seed `2` was not run because the seed-0 guard already failed the promotion
gate.

## Interpretation

Layer-8 `V`-only source+target conditioning is a real finiteness stabilizer:
seed `1` no longer reproduces the `W_V.8` non-finite checkpoint collapse.

It is not a viable paper-method branch as-is:

- seed `1` only ties target
- seed `0` falls below target and below both the unconditioned live row and the
  global-conditioning guard

So the exact intersection branch is weaker than global conditioning. The next
conditioning screen should broaden one axis rather than make the branch more
selective:

1. `V`-only source+target conditioning across all layers, or
2. layer-8 source+target conditioning across both `K/V`,
3. then only keep a branch alive if seed `0` recovers at least `0.0938` while
   bad seeds stay finite.

## Updated Branch Read

- **Promoted:** conditioning touches the real numerical failure surface.
- **Weakened:** the layer-8-only diagnosis is too narrow as a standalone
  method fix.
- **Killed:** exact `layer8 + V-only + source+target whitening` as a promotable
  paper branch.

## Next Exact Gate

Run the broader GSM8K32 seed-1 screen before any GSM70 rerun:

1. `V`-only source+target conditioning across all layers
2. layer-8 source+target conditioning across both `K/V`

Promote only a branch that keeps seed `1` finite and then clears the seed-0
guard at `>= 0.0938`.
