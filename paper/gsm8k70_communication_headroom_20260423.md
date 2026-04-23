# GSM8K70 Communication Headroom Diagnostic

Date: 2026-04-23

## Status

The selector-gap runtime fallback branch remains killed as a publishable method.
The new diagnostic makes the reason sharper: the current wins do not align with
measured source correctness, and the selector score itself is not
source-specific under zero/shuffled controls.

Artifacts:

- gated-control readout:
  `results/gsm8k70_communication_headroom_20260423/headroom.md`
- raw-control readout:
  `results/gsm8k70_communication_headroom_20260423/headroom_raw_controls.md`
- analyzer:
  `scripts/analyze_gsm8k_communication_headroom.py`

## Commands

Gated-control readout:

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_communication_headroom.py \
  --baseline-predictions results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl \
  --source seed0=results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k32_source_alone.jsonl \
  --candidate raw_live=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16.jsonl \
  --candidate gated_live=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736.jsonl \
  --control zero_source=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/zero_source_accept_selector_gap_min_ge_0p02923736.jsonl \
  --control shuffled_source_salt0=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/shuffled_source_salt0_accept_selector_gap_min_ge_0p02923736.jsonl \
  --score-field selector_gap_min \
  --output-json results/gsm8k70_communication_headroom_20260423/headroom.json \
  --output-md results/gsm8k70_communication_headroom_20260423/headroom.md
```

Raw-control readout:

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_communication_headroom.py \
  --baseline-predictions results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl \
  --source seed0=results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k32_source_alone.jsonl \
  --candidate raw_live=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16.jsonl \
  --control zero_source_raw=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/zero_source_raw.jsonl \
  --control shuffled_source_salt0_raw=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/shuffled_source_salt0_raw.jsonl \
  --score-field selector_gap_min \
  --output-json results/gsm8k70_communication_headroom_20260423/headroom_raw_controls.json \
  --output-md results/gsm8k70_communication_headroom_20260423/headroom_raw_controls.md
```

## Readout

- target-alone: `4/70`
- text-to-text: `2/70`
- source-alone: `1/70`, paired vs target `1/4/65`
- target-or-source oracle: `5/70`
- raw live: `8/70`, paired vs target `6/2/62`
- gated live: `7/70`, paired vs target `3/0/67`
- raw controls retain `3/6` raw live wins under both zero-source and
  shuffled-source controls
- gated controls retain `2/3` gated live wins under both zero-source and
  shuffled-source controls
- source-alone explains `0/6` raw live wins and `0/3` gated live wins
- `selector_gap_min` score contrast keeps `0/6` raw live wins at margin `0`
  against raw controls; all six raw live wins have exactly equal matched,
  zero-source, and shuffled-source scores

Raw live target-relative wins:

| Example ID | Norm | Source-correct | Raw-control retained |
|---|---:|---:|---:|
| `31715a2b361f0b6d` | 60 | no | yes |
| `5731a4ad3129a17c` | 75 | no | no |
| `645a38303f97c7b7` | 7 | no | no |
| `c594490a62aaf8d6` | 70 | no | yes |
| `d93e09b5fea44c89` | 100 | no | no |
| `e100c479d9fc22f8` | 187 | no | yes |

## Decision

Do not spend more compute on selector-gap thresholding. It cannot become the
paper method because the score is source-invariant on the current decisive
surface.

The next exact gate is one of:

1. Implement a control-contrastive learned innovation connector with
   matched-source positives and zero/shuffled-source penalties, then rerun the
   same GSM70 seed-0 source-control gate.
2. If we believe GSM8K70 source-alone is too weak to expose communication,
   first choose a stronger-source frozen slice where source-alone or source
   latent features have measurable target-complementary headroom, then run the
   same control-contrastive connector gate there.

The current evidence favors option 1 only if the connector is explicitly trained
against source controls. A selector/gate layered on the existing score is not
credible.

## Checksums

- `headroom.json`:
  `6289d8ae3df836d8d2ea47db8ee77b775fbcd3ca68a2728846252fc6bbf167a1`
- `headroom.md`:
  `e79885952eb418ac682668dad71e48bddbf039c92243956f04e03d682c59e447`
- `headroom_raw_controls.json`:
  `8f325231448420f5dbc881546fdc1f85d4cfafbab6689226a8aee393a012a0a8`
- `headroom_raw_controls.md`:
  `31a59cba8cac0a139e5b6d39fd3bea790820200902adf103e77d2fe3eba57f1f`
