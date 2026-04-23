# GSM8K Communication Headroom Diagnostic

- date: `2026-04-23`
- gate: `control_retention_blocks_positive_claim`
- target: `4 / 70`
- text_to_text: `2 / 70`

## Candidate Summary

| Candidate | Correct | Pair vs target | Wins vs target | Max control retained wins | Max source-correct wins | Score-contrast wins | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| raw_live | 8/70 | 6/2/62 | 6 | 3 | 2 | 0/6 | `candidate_wins_not_source_specific_under_controls` |

## Source Rows

| Source | Correct | Pair vs target | Oracle target/source |
|---|---:|---:|---:|
| c2c_gsm70 | 9/70 | 7/2/61 | 11 |

## Control Retention

### raw_live
| Control | Correct | Wins vs target | Retained candidate wins |
|---|---:|---:|---:|
| zero_source_raw | 5 | 3 | 3/6 |
| shuffled_source_salt0_raw | 6 | 3 | 3/6 |

## Candidate-Win Provenance

### raw_live
| Example ID | Candidate norm | Source-correct labels | Control-correct labels | Candidate score minus max control |
|---|---:|---|---|---:|
| 31715a2b361f0b6d | 60 | c2c_gsm70 | shuffled_source_salt0_raw, zero_source_raw | 0 |
| 5731a4ad3129a17c | 75 | none | none | 0 |
| 645a38303f97c7b7 | 7 | none | none | 0 |
| c594490a62aaf8d6 | 70 | c2c_gsm70 | shuffled_source_salt0_raw, zero_source_raw | 0 |
| d93e09b5fea44c89 | 100 | none | none | 0 |
| e100c479d9fc22f8 | 187 | none | shuffled_source_salt0_raw, zero_source_raw | 0 |

## Gate Notes

- At least one candidate has target-relative wins retained by zero/shuffle controls.
- raw_live score contrast `selector_gap_min` kept 0 / 6 target-relative wins at margin 0; exact-equal score wins: 6.

## Artifact Paths

- baseline_predictions: `results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl`
- candidates.raw_live: `.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16.jsonl`
- sources.c2c_gsm70: `results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl`
- controls.zero_source_raw: `.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/zero_source_raw.jsonl`
- controls.shuffled_source_salt0_raw: `.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/shuffled_source_salt0_raw.jsonl`
