# Competitor Smoke Summary

Date: 2026-04-21

## Direct Competitor

| Method | Model pair | Eval slice | Accuracy | Avg latency sec | Notes |
|---|---|---|---:|---:|---|
| C2C | `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` | first 5 from `gsm8k_gate_search_30.jsonl` | 0.0000 | 7.3765 | Native published artifact resolved and ran on MPS. |
| C2C | `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` | `gsm8k_gate_search_30.jsonl` | 0.0667 | 8.1573 | Full GSM30 smoke on the same slice as the stochastic reranker. |

Artifact:

- `c2c_qwen_gsm5_native_20260421.jsonl`
- `c2c_qwen_gsm5_native_20260421.jsonl.meta.json`
- `c2c_qwen_gsm30_native_20260421.jsonl`
- `c2c_qwen_gsm30_native_20260421.jsonl.meta.json`

Interpretation:

C2C is runnable on the exact Qwen pair. The first 5-example GSM smoke is below
LatentWire’s current GSM5 method smoke, while the full GSM30 smoke ties
target-alone at `0.0667` and trails the strict stochastic selector at `0.1667`.
This is still a smoke result, not a final paper table row; C2C remains stronger
on the older GSM70/SVAMP held-out tables and needs parser-matched scaling.

## Same-Model Compression Control

| Method | Model | Eval slice | Accuracy | Avg latency sec | Tokens/sec |
|---|---|---|---:|---:|---:|
| KVPress none | `Qwen/Qwen3-0.6B` | `gsm8k_5.jsonl` | 0.2000 | 5.6321 | 6.7826 |
| KVPress expected_attention, compression `0.5` | `Qwen/Qwen3-0.6B` | `gsm8k_5.jsonl` | 0.2000 | 6.2749 | 6.1834 |

Artifacts:

- `kvpress_qwen3_gsm5_none_20260421.jsonl`
- `kvpress_qwen3_gsm5_none_20260421.jsonl.meta.json`
- `kvpress_qwen3_gsm5_expected_attention_c050_20260421.jsonl`
- `kvpress_qwen3_gsm5_expected_attention_c050_20260421.jsonl.meta.json`

Interpretation:

KVPress is runnable as a same-model compression control. On this small GSM5
slice, expected-attention compression preserves accuracy but is slower than no
press under the local MPS wrapper. The next meaningful comparison is GSM30
with exact matched decoding and sidecar-normalized latency.

## GSM30 Same-Model Compression Control

| Method | Model | Eval slice | Accuracy | Avg latency sec | Tokens/sec | Avg generated tokens |
|---|---|---|---:|---:|---:|---:|
| KVPress none | `Qwen/Qwen3-0.6B` | `gsm8k_gate_search_30.jsonl` | 0.0667 | 5.9344 | 6.8022 | 40.3667 |
| KVPress expected_attention, compression `0.5` | `Qwen/Qwen3-0.6B` | `gsm8k_gate_search_30.jsonl` | 0.0667 | 6.2869 | 6.1768 | 38.8333 |

Artifacts:

- `kvpress_qwen3_gsm30_none_20260421.jsonl`
- `kvpress_qwen3_gsm30_none_20260421.jsonl.meta.json`
- `kvpress_qwen3_gsm30_expected_attention_c050_20260421.jsonl`
- `kvpress_qwen3_gsm30_expected_attention_c050_20260421.jsonl.meta.json`

Interpretation:

On GSM30, KVPress expected-attention compression is neutral relative to no
press and matches the target-alone `0.0667` baseline. It does not explain the
stochastic-route oracle gap, so it should remain a same-model compression
control rather than the central positive-method lane.
