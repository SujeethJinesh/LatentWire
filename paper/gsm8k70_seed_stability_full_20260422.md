# GSM8K70 Seed Stability Full Read

Date: `2026-04-22`

## Why This Note Exists

The first repeat-seed collapse was already enough to block paper promotion.
This full read closes the immediate question: was seed `1` a one-off failure,
or is the live same-pair lane genuinely unstable across seeds?

## Current Seeds

On the same frozen `70`-example same-pair slice:

| Seed | Status | Accuracy | Correct | Numeric coverage | Win vs target | Loss vs target | Non-finite | First bad key |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | `ok` | 0.1143 | 8 | 70 | 6 | 2 | 0 | `-` |
| 1 | `checkpoint_nonfinite` | 0.0000 | 0 | 0 | 0 | 0 | 2381056 | `W_V.8` |
| 2 | `checkpoint_nonfinite` | 0.0000 | 0 | 0 | 0 | 0 | 2381056 | `W_V.8` |
| 3 | `ok` | 0.0286 | 2 | 69 | 1 | 3 | 0 | `-` |

Campaign aggregate over seeds `0, 1, 2, 3`:

- accuracy mean: `0.0357`
- accuracy min / max: `0.0000 / 0.1143`
- paired delta mean vs `target_alone` on numerically valid seeds: `+0.0143`
- paired CI mean on numerically valid seeds: `[-0.0500, 0.0857]`
- positive seeds: `1 / 4`

Artifacts are local under:

- `results/gsm8k70_seed_repeat_full_20260422/gsm8k_contract_campaign.md`
- `results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k_contract_residual_sweep_20260421.json`
- `results/gsm8k70_seed_repeat_full_20260422/seed1/gsm8k_contract_residual_sweep_20260421.json`
- `results/gsm8k70_seed_repeat_full_20260422/seed2/gsm8k_contract_residual_sweep_20260421.json`
- `results/gsm8k70_seed_repeat_full_20260422/seed3/gsm8k_contract_residual_sweep_20260421.json`

Checkpoint-health sidecars now include:

- `checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_seed1.pt.health.json`
- `checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_seed2.pt.health.json`

## What Changed

The paper story is now materially weaker than the one-seed GSM70 read:

1. The live row is not just seed-fragile.
   - Only `1 / 4` attempted seeds is both finite and above target.
2. The failure mode is not random.
   - Seeds `1` and `2` both collapse into the same layer-8 `V` family:
     `W_V.8`, `quant_proj_V.8`, `quant_aux_proj_V.8`, and the matching
     residual-slot tensors.
3. Finite does not imply good.
   - Seed `3` is numerically valid but lands at `2/70`, below
     `target_alone = 4/70`.

## Interpretation

This is no longer a story about ordinary benchmark variance.

- There is one real positive seed.
- There are now two replicated numerical collapses in the same tensor family.
- There is one additional finite seed that fails to beat target.

So the current live lane should be treated as:

- a promising but unstable mechanism clue
- not a stable positive method
- and not ready for cross-family promotion yet

The diagnostics on the two finite seeds still support the earlier
mechanism-level read:

- candidate-only wins across seeds `0` and `3`: `7`
- source correct on those wins: `0`
- text correct on those wins: `0`

So the remaining problem is still robustness, not a new sign of source-answer
copying.

## New Stability Read

The exact blocker is now sharper:

1. eliminate or explain the repeated layer-8 `V`-side non-finite failure
2. explain why a finite seed can still degrade below target
3. only after that, spend more budget on matched cross-family evaluation

## Immediate Next Gate

The next bounded method branch should be the codec-side follow-up already
ranked highest in the ledger:

1. `V`-only anchor-preserving selective precision on top of
   `dynalign_module_replace_residrank16`
2. keep a small exact anchor set in the `V` family
3. quantize or code only the residual `V` tail
4. rerun the exact GSM70 seed audit on seeds `0, 1, 2, 3`

Promote that branch only if it:

- removes the repeated `W_V.8` / `quant_proj_V.8` / `quant_aux_proj_V.8`
  collapse on the bad seeds
- keeps numeric extraction coverage intact
- and preserves at least the old `0.0938` real-benchmark ceiling while
  matching or improving the live seed-0 row
