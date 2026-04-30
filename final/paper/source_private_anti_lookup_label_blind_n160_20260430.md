# Source-Private Anti-Lookup Label-Blind n160 Scale-Up

- date: `2026-04-30`
- status: passed medium anti-lookup stress
- result root: `results/source_private_anti_lookup_label_blind_20260430/`
- scale rung: medium anti-lookup stress

## Readiness

This strengthens the scoped ICLR story but does not make the paper a broad
latent-transfer claim. The current story is source-private residual evidence
communication with decoder side information: the source sends a tiny packet,
and the target can use it only when the public candidate-side table or receiver
contract supplies the needed side information.

The exact blocker addressed here is the coded-label/lookup objection. A
reviewer can ask whether the positive `n=160` endpoint result is just hidden
labels or repair keys leaking through the prompt. This gate hides the public
repair-key table and original candidate labels, then checks whether all opaque
source messages collapse to target-only behavior.

## Commands

```bash
./venv_arm64/bin/python scripts/run_source_private_mac_endpoint_proxy_frontier.py \
  --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl \
  --output-dir results/source_private_anti_lookup_label_blind_20260430/core_seed29_qwen3_n160_label_blind \
  --model Qwen/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
  --limit 160 \
  --max-new-tokens 8 \
  --no-enable-thinking \
  --prompt-style label_strict \
  --candidate-view label_blind \
  --conditions target_only matched_packet matched_byte_text_2 random_same_byte_packet deranged_candidate_diag_table query_aware_diag_span structured_json_diag structured_free_text_diag full_hidden_log

./venv_arm64/bin/python scripts/run_source_private_mac_endpoint_proxy_frontier.py \
  --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/benchmark.jsonl \
  --output-dir results/source_private_anti_lookup_label_blind_20260430/holdout_seed30_qwen3_n160_label_blind \
  --model Qwen/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
  --limit 160 \
  --max-new-tokens 8 \
  --no-enable-thinking \
  --prompt-style label_strict \
  --candidate-view label_blind \
  --conditions target_only matched_packet matched_byte_text_2 random_same_byte_packet deranged_candidate_diag_table query_aware_diag_span structured_json_diag structured_free_text_diag full_hidden_log

./venv_arm64/bin/python scripts/build_source_private_anti_lookup_label_blind_summary.py \
  --label-blind-summary results/source_private_anti_lookup_label_blind_20260430/core_seed29_qwen3_n160_label_blind/summary.json \
  --positive-summary results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json \
  --label-blind-summary results/source_private_anti_lookup_label_blind_20260430/holdout_seed30_qwen3_n160_label_blind/summary.json \
  --positive-summary results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json \
  --output-dir results/source_private_anti_lookup_label_blind_20260430
```

The endpoint commands exit nonzero because the endpoint-positive pass gate is
false under label-blind candidate views. That is expected for this collapse
control; the summary artifact is the gate that should pass.

## Result

- pass gate: `true`
- rows: `2`
- surfaces: core `n=160`, holdout `n=160`
- exact-ID parity: `true`
- matched packet valid rate: `1.000`
- max opaque-payload accuracy minus target: `0.000`
- max opaque paired-bootstrap CI95 high versus target: `0.000`
- max opaque strict paired-bootstrap CI95 high versus target: `0.000`
- positive diagnostic-table comparator lift: minimum `+0.425`

| Surface | Target | Matched packet | Max opaque payload | Opaque-target | Positive diagnostic-table lift |
|---|---:|---:|---:|---:|---:|
| core `n=160` label-blind | 0.250 | 0.250 | 0.250 | 0.000 | +0.425 |
| holdout `n=160` label-blind | 0.250 | 0.244 | 0.250 | 0.000 | +0.438 |

Core generated the same fallback distribution for every condition: `Option A`
on `80/160`, `Option C` on `40/160`, and `Option D` on `40/160`. Holdout was
the same distribution up to one invalid/missed fallback in the matched-packet
and deranged-table rows. Full hidden logs, structured JSON, structured
free-text, query-aware spans, matched 2-byte packets, and random same-byte
packets all failed to improve when the repair-key table was hidden.

## Interpretation

This is a strong anti-lookup result. The positive endpoint rows are not
explained by the target model simply recognizing opaque repair keys, hidden
candidate labels, or verbose private logs. When the public key-to-candidate
side information is removed, even the full hidden log collapses to the fallback
target behavior.

The result also narrows the paper claim. The method should be framed as
source-private communication with decoder side information, close to a
Wyner-Ziv/Slepian-Wolf style syndrome resolved by public target context. It is
not protocol-free semantic transfer.

## Next Gate

For the final ICLR version, the next anti-lookup choice is either a deterministic
`n=500` label-blind stress on the large frozen surface or a learned/shared
dictionary receiver that reduces the public-table shape of the method. Given
the one-month deadline, the higher-value research gate is the learned receiver;
the higher-value reviewer-defense gate is the large deterministic
label-blind stress.
