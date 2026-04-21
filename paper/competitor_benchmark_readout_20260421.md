# Competitor Benchmark Readout

Date: 2026-04-21

This readout separates direct cross-model communication competitors from
same-model cache-compression controls. It is not a final efficiency claim until
prompt, token, byte, latency, and repair budgets are all matched in one harness.

## Current Direct Comparators

| Split | Method | Accuracy | N | Source artifact | Interpretation |
|---|---|---:|---:|---|---|
| `gsm8k_gate_search_30` | `C2C` native smoke | 0.0667 | 30 | `results/competitor_bootstrap_20260421/c2c_qwen_gsm30_native_20260421.jsonl` | Exact Qwen-pair direct peer ties target-alone and trails strict rerank / repair methods on the same slice. |
| `gsm8k_eval_70` | `C2C` replay | 0.1286 | 70 | `results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl` | Strongest no-extra-repair direct GSM70 external bar. |
| `gsm8k_eval_70` | `KVComm` ported replay | 0.0000 | 70 | `results/kvcomm_gsm70_20260419/qwen_gsm70_kvcomm_ported.jsonl` | Ported heterogeneous replay is not competitive yet. |
| `gsm8k_100` | `C2C` replay | 0.1100 | 100 | `results/c2c_gsm100_20260418/qwen_gsm100_c2c.jsonl` | Larger GSM anchor for the C2C baseline. |
| `svamp_eval_70` | `C2C` replay | 0.4429 | 70 | `results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl` | Strongest no-extra-repair direct SVAMP70 external bar. |

## Current Same-Model Compression Controls

| Split | Method | Accuracy | N | Source artifact | Interpretation |
|---|---|---:|---:|---|---|
| `gsm8k_5` | `KVPress` none | 0.2000 | 5 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm5_none_20260421.jsonl` | Same-model baseline control, not a cross-model communication peer. |
| `gsm8k_5` | `KVPress` expected attention `0.5` | 0.2000 | 5 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm5_expected_attention_c050_20260421.jsonl` | Compression does not improve the tiny GSM5 smoke. |
| `gsm8k_gate_search_30` | `KVPress` none | 0.0667 | 30 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm30_none_20260421.jsonl` | Same-model GSM30 control ties target-alone. |
| `gsm8k_gate_search_30` | `KVPress` expected attention `0.5` | 0.0667 | 30 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm30_expected_attention_c050_20260421.jsonl` | Expected-attention compression is neutral on this slice. |

## LatentWire Comparison State

| Split | LatentWire row | Accuracy | Comparator row | Comparator accuracy | Read |
|---|---|---:|---|---:|---|
| `gsm8k_gate_search_30` | strict selector + process repair | 0.2333 | `C2C` native smoke | 0.0667 | Positive on the small exact-pair smoke, but adds target-side repair compute. |
| `gsm8k_eval_70` | selected route, no repair | 0.1286 | `C2C` replay | 0.1286 | Candidate selection alone ties the direct external bar. |
| `gsm8k_eval_70` | target self-repair | 0.1714 | `C2C` replay | 0.1286 | Same-prompt target-side repair already beats C2C, so this is a fairness control, not cross-model evidence. |
| `gsm8k_eval_70` | strict selector + process repair | 0.2000 | target self-repair | 0.1714 | Route-specific increment is positive but modest: `+0.0286`. |
| `svamp_eval_70` | selected route, no repair | 0.3571 | `C2C` replay | 0.4429 | Candidate selection alone trails C2C on SVAMP. |
| `svamp_eval_70` | target self-repair | 0.5000 | `C2C` replay | 0.4429 | Same-prompt target repair beats C2C and must stay in the main table. |
| `svamp_eval_70` | strict selector + process repair | 0.5429 | target self-repair | 0.5000 | Route-specific increment is again positive but modest: `+0.0429`. |

## Next Fair Comparison Requirements

1. Put `target_alone`, text-to-text, `C2C`, selected-route no-repair,
   target self-repair, and selected-route repair in one exact-split summary
   table.
2. Log prompt bytes, generated tokens, repair prompt bytes, target-side repair
   calls, latency, and any transported latent/cache bytes for every row.
3. Report direct cross-model competitors separately from same-model compression
   controls; do not treat `KVPress`/`Quest`/`H2O`/`SnapKV` as semantic
   communication baselines.
4. Treat target self-repair as the strongest immediate control; a final
   positive-method claim needs route-specific gains over that control at
   matched repair budget.
