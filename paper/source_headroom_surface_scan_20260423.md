# Source-Headroom Surface Scan

Date: 2026-04-23

## Status

The current GSM70 same-pair source-alone lane is not a good decision surface
for a learned communication connector: source-alone adds only one source-only
win over target. Before training a new connector, the project needs a frozen
surface where the source has measurable target-complementary information.

This memo adds a reusable scanner:

- script: `scripts/analyze_source_headroom_surfaces.py`
- readout:
  `results/source_headroom_surfaces_20260423/headroom_surfaces.md`
- GSM70 C2C/live overlap readout:
  `results/source_headroom_surfaces_20260423/gsm70_c2c_live_headroom.md`

## Command

```bash
./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface gsm70_source_alone=target_path=results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl,source_path=results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k32_source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=gsm70_source_alone \
  --surface gsm32_source_alone=target_path=results/gsm8k_smoke_contract_20260421/gsm8k32_latentwire.jsonl,source_path=results/gsm8k_contract_residual_rank16_dynalign_20260421/gsm8k32_source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=gsm32_source_alone \
  --surface svamp70_text_relay=path=results/svamp_replication_20260417/predictions/svamp70_attention_g010_pos05.jsonl,target_method=target_alone,source_method=text_to_text,eval_file=data/svamp_eval_70.jsonl,note=svamp_text_relay \
  --surface svamp70_process_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_strict_selector_telemetry.jsonl,target_method=target_alone,source_method=process_repair_selected_route,note=svamp_process_repair \
  --surface svamp70_c2c=target_path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_strict_selector_telemetry.jsonl,source_path=results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl,target_method=target_alone,source_method=c2c_generate,note=svamp_c2c \
  --surface gsm70_c2c=target_path=results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl,source_path=results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl,target_method=target_alone,source_method=c2c_generate,note=gsm_c2c \
  --surface gsm100_text_relay=path=results/gsm8k_fixed_prior_cycle_20260417/predictions/gsm8k100_attention_g010_pos05.jsonl,target_method=target_alone,source_method=text_to_text,eval_file=data/gsm8k_100.jsonl,note=gsm100_text \
  --surface gsm70_old_text_relay=path=results/gsm8k_k_only_fixed_20260417/predictions/baseline_brief.jsonl,target_method=target_alone,source_method=text_to_text,eval_file=data/gsm8k_eval_70.jsonl,note=gsm70_old_text \
  --min-source-only 5 \
  --output-json results/source_headroom_surfaces_20260423/headroom_surfaces.json \
  --output-md results/source_headroom_surfaces_20260423/headroom_surfaces.md
```

## Readout

| Surface | Target | Source-like row | Source-only wins | Oracle |
|---|---:|---:|---:|---:|
| GSM70 source-alone | 4/70 | 1/70 | 1 | 5/70 |
| GSM32 source-alone | 2/32 | 1/32 | 1 | 3/32 |
| SVAMP70 text relay | 5/70 | 29/70 | 26 | 31/70 |
| SVAMP70 C2C | 21/70 | 31/70 | 18 | 39/70 |
| SVAMP70 process repair | 21/70 | 38/70 | 17 | 38/70 |
| GSM70 C2C | 4/70 | 9/70 | 7 | 11/70 |
| GSM100 text relay | 4/100 | 10/100 | 8 | 12/100 |
| older GSM70 text relay | 1/70 | 8/70 | 8 | 9/70 |

The strongest available headroom surface is SVAMP70 text relay. The strongest
strict example-ID surface is SVAMP70 C2C/process repair. GSM70 C2C has modest
headroom (`7` source-only wins), while true GSM source-alone remains weak.

## GSM70 C2C Overlap With Raw Live

The raw dynalign live row overlaps C2C on `2/6` target-relative raw live wins,
but both overlapping IDs are also retained by raw zero-source and shuffled-source
controls. This means the current dynalign wins do not yet look like
source-specific C2C-style innovations.

Readout:

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_communication_headroom.py \
  --baseline-predictions results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl \
  --source c2c_gsm70=results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl \
  --source-method c2c_generate \
  --candidate raw_live=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16.jsonl \
  --control zero_source_raw=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/zero_source_raw.jsonl \
  --control shuffled_source_salt0_raw=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/shuffled_source_salt0_raw.jsonl \
  --score-field selector_gap_min \
  --output-json results/source_headroom_surfaces_20260423/gsm70_c2c_live_headroom.json \
  --output-md results/source_headroom_surfaces_20260423/gsm70_c2c_live_headroom.md
```

## Attempted Fresh SVAMP Materialization

I attempted to generate fresh exact-ID SVAMP70 `source_alone`, `target_alone`,
and `text_to_text` rows using `latent_bridge/evaluate.py`. The run stayed
healthy but produced no partial artifact after more than 18 minutes, so I killed
it to keep this turn bounded. There is no tracked output from that attempt.

Next time, run this as a resumable split job with one method per output file and
possibly a smaller first limit, then merge the exact-ID rows.

## Decision

Do not train the learned connector on the current GSM70 source-alone surface as
the primary gate. It has too little measured source-complementary headroom.

Next exact gate:

1. Materialize fresh exact-ID SVAMP70 source/target/text/C2C rows in separate
   resumable jobs.
2. If `source_alone` or a latent-accessible source row has at least `5` to `10`
   source-only wins on the frozen slice, train the control-contrastive connector
   there first.
3. If only text/C2C has headroom, use SVAMP70 as a teacher/upper-bound surface
   and train the connector to target source-specific C2C/text innovations, with
   zero/shuffled/wrong-source controls required before any paper claim.

This aligns with the interface-redesign direction in
`references/440_interface_redesign_connector_refs.md` and the Q-former/Perceiver
bottleneck direction in `references/447_qformer_connector_followup_refs.md`.

## Checksums

- `headroom_surfaces.json`:
  `09587d094cdfa54f5307db03fac552c6cc2bc319fa791e3b218efd05735f0638`
- `headroom_surfaces.md`:
  `e423946014bddc8f8fc4fffad9420afdb9ace912a80bf07258e3adbb76bf49bf`
- `gsm70_c2c_live_headroom.json`:
  `319eb22ed97b087ea9d412783b89d9e73d50ae77ec6688f688fa8f3e706d77e3`
- `gsm70_c2c_live_headroom.md`:
  `49239758037a18e154a820ee9f55aa9843d52540cba4cebeeb8c34ad612eb35f`
