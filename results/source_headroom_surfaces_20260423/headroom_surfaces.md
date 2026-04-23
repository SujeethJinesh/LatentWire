# Source-Headroom Surface Scan

- date: `2026-04-23`
- min source-only threshold: `5`

| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |
|---|---|---:|---:|---:|---:|---|---|
| gsm70_source_alone | `weak_source_complementary_surface` | 4/70 | 1/70 | 1 | 5/70 | example_id/example_id | gsm70_source_alone |
| gsm32_source_alone | `weak_source_complementary_surface` | 2/32 | 1/32 | 1 | 3/32 | example_id/example_id | gsm32_source_alone |
| svamp70_text_relay | `strong_source_complementary_surface` | 5/70 | 29/70 | 26 | 31/70 | eval_file/eval_file | svamp_text_relay |
| svamp70_process_repair | `strong_source_complementary_surface` | 21/70 | 38/70 | 17 | 38/70 | example_id/example_id | svamp_process_repair |
| svamp70_c2c | `strong_source_complementary_surface` | 21/70 | 31/70 | 18 | 39/70 | example_id/example_id | svamp_c2c |
| gsm70_c2c | `strong_source_complementary_surface` | 4/70 | 9/70 | 7 | 11/70 | example_id/example_id | gsm_c2c |
| gsm100_text_relay | `strong_source_complementary_surface` | 4/100 | 10/100 | 8 | 12/100 | eval_file/eval_file | gsm100_text |
| gsm70_old_text_relay | `strong_source_complementary_surface` | 1/70 | 8/70 | 8 | 9/70 | eval_file/eval_file | gsm70_old_text |

## Ranked Decision

- `svamp70_text_relay`: status=`strong_source_complementary_surface`, source_only=`26`, oracle=`31/70`, strict_ids=`True`
- `svamp70_c2c`: status=`strong_source_complementary_surface`, source_only=`18`, oracle=`39/70`, strict_ids=`True`
- `svamp70_process_repair`: status=`strong_source_complementary_surface`, source_only=`17`, oracle=`38/70`, strict_ids=`True`
- `gsm70_old_text_relay`: status=`strong_source_complementary_surface`, source_only=`8`, oracle=`9/70`, strict_ids=`True`
- `gsm100_text_relay`: status=`strong_source_complementary_surface`, source_only=`8`, oracle=`12/100`, strict_ids=`True`

## Artifact Paths

- `gsm70_source_alone` target `results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl` (target_alone); source `results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k32_source_alone.jsonl` (source_alone)
- `gsm32_source_alone` target `results/gsm8k_smoke_contract_20260421/gsm8k32_latentwire.jsonl` (target_alone); source `results/gsm8k_contract_residual_rank16_dynalign_20260421/gsm8k32_source_alone.jsonl` (source_alone)
- `svamp70_text_relay` target `results/svamp_replication_20260417/predictions/svamp70_attention_g010_pos05.jsonl` (target_alone); source `results/svamp_replication_20260417/predictions/svamp70_attention_g010_pos05.jsonl` (text_to_text)
- `svamp70_process_repair` target `results/process_repair_holdout_20260421/qwen_svamp70_process_repair_strict_selector_telemetry.jsonl` (target_alone); source `results/process_repair_holdout_20260421/qwen_svamp70_process_repair_strict_selector_telemetry.jsonl` (process_repair_selected_route)
- `svamp70_c2c` target `results/process_repair_holdout_20260421/qwen_svamp70_process_repair_strict_selector_telemetry.jsonl` (target_alone); source `results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl` (c2c_generate)
- `gsm70_c2c` target `results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl` (target_alone); source `results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl` (c2c_generate)
- `gsm100_text_relay` target `results/gsm8k_fixed_prior_cycle_20260417/predictions/gsm8k100_attention_g010_pos05.jsonl` (target_alone); source `results/gsm8k_fixed_prior_cycle_20260417/predictions/gsm8k100_attention_g010_pos05.jsonl` (text_to_text)
- `gsm70_old_text_relay` target `results/gsm8k_k_only_fixed_20260417/predictions/baseline_brief.jsonl` (target_alone); source `results/gsm8k_k_only_fixed_20260417/predictions/baseline_brief.jsonl` (text_to_text)
