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

## Pending Direct Comparator Bootstrap

| Method | Local state | Native support | Next artifact | Interpretation |
|---|---|---|---|---|
| `LatentMAS` | clean local clone at `references/repos/LatentMAS`, commit `b9b2095`; repo remains ignored; LatentWire wrapper now exists at `scripts/run_latentmas_competitor_eval.py` with lazy imports and a non-vLLM `SamplingParams` shim | native GSM8K via `baseline`, `text_mas`, `latent_mas`; SVAMP supported through wrapper row conversion | matched GSM/SVAMP limit smokes emitted as JSONL plus `.jsonl.meta.json` | Closest latent-communication competitor; baseline/text-MAS plumbing works on cached Qwen2.5, while latent-MAS direct mode remains runtime-blocked locally. |

## LatentMAS Harness Probes

These rows are not fair competitor results. They use cached
`Qwen/Qwen2.5-0.5B-Instruct` on `N=1` only to prove the wrapper can load vendor
methods, emit JSONL/meta telemetry, and survive local device constraints.

| Split | Method | Accuracy | N | Latency sec | Source artifact | Interpretation |
|---|---|---:|---:|---:|---|---|
| `gsm8k_eval_70` limit-1 | `LatentMAS` baseline probe | 0.0000 | 1 | 10.7217 | `results/latentmas_competitor_20260421/qwen25_05b_gsm1_baseline_probe.jsonl` | Wrapper, row conversion, baseline import, parsing, and meta emission work on cached Qwen2.5. |
| `gsm8k_eval_70` limit-1 | `LatentMAS` text-MAS probe | 0.0000 | 1 | 15.3653 | `results/latentmas_competitor_20260421/qwen25_05b_gsm1_text_mas_probe.jsonl` | Multi-agent text prompt path works; this remains a plumbing row only. |
| `gsm8k_eval_70` limit-1 | `LatentMAS` latent-MAS CPU blocker probe | blocked | 1 | - | `results/latentmas_competitor_20260421/qwen25_05b_gsm1_latent_mas_cpu_probe.jsonl.meta.json` | Wrapper exits cleanly and emits a blocker artifact; CPU and MPS fallback both reach HF cache-position failure inside vendor latent generation. |

## Current Same-Model Compression Controls

| Split | Method | Accuracy | N | Source artifact | Interpretation |
|---|---|---:|---:|---|---|
| `gsm8k_5` | `KVPress` none | 0.2000 | 5 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm5_none_20260421.jsonl` | Same-model baseline control, not a cross-model communication peer. |
| `gsm8k_5` | `KVPress` expected attention `0.5` | 0.2000 | 5 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm5_expected_attention_c050_20260421.jsonl` | Compression does not improve the tiny GSM5 smoke. |
| `gsm8k_eval_70` limit-5 | `KVPress` none | 0.2000 | 5 | `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit5.jsonl` | Bounded control row for the exact next-runnable matrix. |
| `gsm8k_eval_70` limit-5 | `KVPress` expected attention `0.5` | 0.2000 | 5 | `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit5.jsonl` | Neutral against no-press on the tiny GSM limit-5 smoke. |
| `gsm8k_eval_70` limit-10 | `KVPress` none | 0.1000 | 10 | `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit10.jsonl` | Widened bounded control row; CPU fallback. |
| `gsm8k_eval_70` limit-10 | `KVPress` expected attention `0.5` | 0.1000 | 10 | `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit10.jsonl` | Neutral against no-press on GSM, but faster in this smoke. |
| `gsm8k_eval_70` limit-20 | `KVPress` none | 0.1000 | 20 | `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit20.jsonl` | Widened bounded control row; CPU fallback. |
| `gsm8k_eval_70` limit-20 | `KVPress` expected attention `0.5` | 0.0500 | 20 | `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit20.jsonl` | Accuracy regresses versus no-press while latency improves. |
| `gsm8k_gate_search_30` | `KVPress` none | 0.0667 | 30 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm30_none_20260421.jsonl` | Same-model GSM30 control ties target-alone. |
| `gsm8k_gate_search_30` | `KVPress` expected attention `0.5` | 0.0667 | 30 | `results/competitor_bootstrap_20260421/kvpress_qwen3_gsm30_expected_attention_c050_20260421.jsonl` | Expected-attention compression is neutral on this slice. |
| `svamp_eval_70` limit-5 | `KVPress` none | 0.4000 | 5 | `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit5.jsonl` | Bounded SVAMP control row. |
| `svamp_eval_70` limit-5 | `KVPress` expected attention `0.5` | 0.6000 | 5 | `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit5.jsonl` | Positive tiny smoke against no-press, but underpowered and not paper-ready. |
| `svamp_eval_70` limit-10 | `KVPress` none | 0.2000 | 10 | `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit10.jsonl` | Widened bounded SVAMP control row; CPU fallback. |
| `svamp_eval_70` limit-10 | `KVPress` expected attention `0.5` | 0.5000 | 10 | `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit10.jsonl` | Positive smoke against no-press with lower latency, still underpowered. |
| `svamp_eval_70` limit-20 | `KVPress` none | 0.1500 | 20 | `results/competitor_next_runnable_20260421/kvpress_svamp70_none_limit20.jsonl` | Widened bounded SVAMP control row; CPU fallback. |
| `svamp_eval_70` limit-20 | `KVPress` expected attention `0.5` | 0.3000 | 20 | `results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050_limit20.jsonl` | Accuracy improves versus no-press but latency is slightly worse in this run. |

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
5. Repeat limit-20 or run a bounded limit-50 KVPress batch before using
   same-model compression controls as stability evidence; limit-5/10/20 remain
   harness-health rows only.
6. Fix `LatentMAS` latent-mode runtime, then run matched limit smokes as a
   direct latent-communication competitor with accuracy, output-token, latency,
   and traceability reporting.
7. Do not stage `references/repos/LatentMAS`; keep vendor code ignored and
   implement wrappers on the LatentWire side.
8. Keep `paper/matched_competitor_matrix_20260421.md` updated; missing fair
   rows should remain visible until real artifacts exist.

## LatentMAS Wrapper Command Matrix

The wrapper is intentionally outside `references/repos/LatentMAS` and emits
paper-grade telemetry rather than relying on vendor stdout logs.

```bash
./venv_arm64/bin/python scripts/run_latentmas_competitor_eval.py \
  --method baseline \
  --model-name Qwen/Qwen3-4B \
  --task gsm8k \
  --eval-file data/gsm8k_eval_70.jsonl \
  --limit 10 \
  --prediction-output results/latentmas_competitor_20260421/gsm10_baseline.jsonl
```

Repeat the same command with `--method text_mas` and `--method latent_mas
--latent-steps 10`, then mirror on `data/svamp_eval_70.jsonl` with
`--task svamp --latentmas-task gsm8k`. The required comparison table should
join each `.meta.json` with target-alone, target self-repair, selected-route
no-repair, selected-route repair, and C2C on the same example IDs.
