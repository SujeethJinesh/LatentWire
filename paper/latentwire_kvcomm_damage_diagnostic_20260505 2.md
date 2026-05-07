# KVComm Damage Diagnostic

This diagnostic asks whether local KVComm failures come from source content or from the cache-injection mechanism itself. High matched-vs-zero agreement means the transferred source values are not the main driver; the cache prefix/position path is.

## ARC-Challenge

- prediction file: `results/dense_baseline_mcqa_smoke_20260505/kvcomm_arc_n16_controls_constrained.jsonl`
- paired examples: `16`
- matched vs zero-source prediction agreement: `1.000`
- matched damages target-only: `0.625`
- matched repairs target-only: `0.000`

| method | n | accuracy | mean communicated bytes | answer distribution |
|---|---:|---:|---:|---|
| target_only | 16 | 0.688 | 0.000 | {"a": 4, "b": 5, "c": 5, "d": 2} |
| kvcomm_matched | 16 | 0.062 | 2757120.000 | {"a": 6, "b": 3, "c": 2, "d": 5} |
| kvcomm_zero_source | 16 | 0.062 | 2757120.000 | {"a": 6, "b": 3, "c": 2, "d": 5} |
| kvcomm_shuffled_source | 16 | 0.250 | 2628096.000 | {"a": 8, "b": 1, "c": 2, "d": 5} |

| paired diagnostic | value |
|---|---:|
| matched_zero_prediction_agreement | 1.000 |
| matched_target_prediction_agreement | 0.188 |
| zero_target_prediction_agreement | 0.188 |
| matched_damages_target | 0.625 |
| matched_repairs_target | 0.000 |
| zero_damages_target | 0.625 |
| zero_repairs_target | 0.000 |
| shuffled_target_prediction_agreement | 0.250 |
| shuffled_damages_target | 0.562 |
| shuffled_repairs_target | 0.125 |

## OpenBookQA

- prediction file: `results/dense_baseline_mcqa_smoke_20260505/kvcomm_openbookqa_n16_controls_constrained.jsonl`
- paired examples: `16`
- matched vs zero-source prediction agreement: `1.000`
- matched damages target-only: `0.188`
- matched repairs target-only: `0.125`

| method | n | accuracy | mean communicated bytes | answer distribution |
|---|---:|---:|---:|---|
| target_only | 16 | 0.250 | 0.000 | {"a": 11, "b": 1, "d": 4} |
| kvcomm_matched | 16 | 0.188 | 1913856.000 | {"a": 8, "b": 2, "c": 4, "d": 2} |
| kvcomm_zero_source | 16 | 0.188 | 1913856.000 | {"a": 8, "b": 2, "c": 4, "d": 2} |
| kvcomm_shuffled_source | 16 | 0.375 | 1959936.000 | {"a": 4, "b": 1, "c": 8, "d": 3} |

| paired diagnostic | value |
|---|---:|
| matched_zero_prediction_agreement | 1.000 |
| matched_target_prediction_agreement | 0.375 |
| zero_target_prediction_agreement | 0.375 |
| matched_damages_target | 0.188 |
| matched_repairs_target | 0.125 |
| zero_damages_target | 0.188 |
| zero_repairs_target | 0.125 |
| shuffled_target_prediction_agreement | 0.375 |
| shuffled_damages_target | 0.188 |
| shuffled_repairs_target | 0.312 |

