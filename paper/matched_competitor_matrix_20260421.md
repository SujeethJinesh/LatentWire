# Matched Competitor Matrix

Date: 2026-04-21

This lightweight matrix reads existing JSONL and `.meta.json` artifacts only. Missing heavyweight rows stay explicit so the paper cannot accidentally compare against an incomplete competitor set.

| Row | Family | Status | Accuracy | N | Latency sec | Token/byte proxy | Source | Artifact | Note |
|---|---|---|---:|---:|---:|---:|---|---|---|
| Target alone | LatentWire control | present | 0.0571 | 70 | - | 63.5143 | meta:method_summary+jsonl | `results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl` | - |
| Target self-repair | LatentWire control | present | 0.1714 | 70 | - | 63.5143 | meta:method_summary+jsonl | `results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl` | - |
| Selected route, no repair | LatentWire | present | 0.1286 | 70 | - | 1367987.2 | meta:method_summary | `results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl` | - |
| Selected route + repair | LatentWire | present | 0.2000 | 70 | - | 1367987.2 | meta:method_summary | `results/process_repair_holdout_20260421/qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl` | - |
| C2C | Direct competitor | present | 0.1286 | 70 | 678.7744 | 63.6000 | jsonl | `results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl` | - |
| KVComm | Direct competitor | present | 0.0000 | 70 | 384.6541 | 64.0000 | jsonl | `results/kvcomm_gsm70_20260419/qwen_gsm70_kvcomm_ported.jsonl` | - |
| KVPress no press | Compression control | present | 0.1000 | 20 | 10.2290 | 40.9000 | meta:run | `results/competitor_next_runnable_20260421/kvpress_gsm70_none_limit20.jsonl` | - |
| KVPress expected-attn 0.5 | Compression control | present | 0.0500 | 20 | 8.9745 | 40.7000 | meta:run | `results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050_limit20.jsonl` | - |
| LatentMAS baseline harness probe | Latent competitor probe | present | 0.0000 | 1 | 10.7217 | 167.0000 | meta:run | `results/latentmas_competitor_20260421/qwen25_05b_gsm1_baseline_probe.jsonl` | - |
| LatentMAS text-MAS harness probe | Latent competitor probe | present | 0.0000 | 1 | 15.3653 | 1102.0 | meta:run | `results/latentmas_competitor_20260421/qwen25_05b_gsm1_text_mas_probe.jsonl` | - |
| LatentMAS baseline | Latent competitor | missing | - | - | - | - | - | `results/latentmas_competitor_20260421/gsm10_baseline.jsonl` | artifact missing |
| LatentMAS text-MAS | Latent competitor | missing | - | - | - | - | - | `results/latentmas_competitor_20260421/gsm10_text_mas.jsonl` | artifact missing |
| LatentMAS latent-MAS | Latent competitor | missing | - | - | - | - | - | `results/latentmas_competitor_20260421/gsm10_latent_mas.jsonl` | artifact missing |

## Reading Notes

- `Token/byte proxy` is intentionally a proxy because historical artifacts log different counters: generated tokens, trace token counts, or transported byte/bit averages.
- `KVPress` rows are same-model compression controls, not semantic cross-model communication baselines.
- `LatentMAS` harness probe rows use the cached Qwen2.5-0.5B model on `N=1`; they are plumbing checks, not fair competitor rows.
- Full `LatentMAS` rows are expected to remain `missing` until bounded matched wrapper runs are executed; this is preferable to silently omitting them.
- Promote a positive-method claim only when selected-route repair beats target self-repair on matched IDs with comparable repair, token, byte, and latency budgets.
