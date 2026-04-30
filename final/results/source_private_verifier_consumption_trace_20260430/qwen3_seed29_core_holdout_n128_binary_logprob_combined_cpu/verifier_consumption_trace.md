# Source-Private Verifier Consumption Trace

- pass gate: `True`
- result dirs: `2`
- min matched accuracy: `1.000`
- max target-only accuracy: `0.250`
- min matched minus best control: `0.750`
- max matched p50 latency ms: `1665.54`
- max matched binary forward passes/example: `4.00`
- matched source-boundary payload bytes: `2.00`
- matched packet record bytes: `5.00`

| result | condition | role | acc | payload B | record B | line B | DMA B | fwd/ex | p50 ms | p95 ms | exposure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | deranged_candidate_diag_table | source_destroying_control | 0.000 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1665.33 | 1777.40 | source-private, destroyed |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | matched_packet | source_packet | 1.000 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1665.54 | 1782.47 | source-private |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | random_noncandidate_same_byte | source_destroying_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1659.98 | 1797.43 | source-private, destroyed |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | random_same_byte | source_destroying_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1664.00 | 1780.19 | source-private, destroyed |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | shuffled_packet | source_destroying_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1670.46 | 1789.33 | source-private, destroyed |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | structured_free_text_2byte | matched_byte_text_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 0.00 | 0.00 | 0.00 | source-private, text, destroyed |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | structured_json_2byte | matched_byte_text_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 0.00 | 0.01 | 0.01 | source-private, text, destroyed |
| qwen3_seed29_core_n128_binary_logprob_combined_cpu | target_only | no_source | 0.250 | 0.00 | 0.00 | 0 | 0 | 0.00 | 0.00 | 0.00 |  |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | deranged_candidate_diag_table | source_destroying_control | 0.000 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1631.26 | 1743.95 | source-private, destroyed |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | matched_packet | source_packet | 1.000 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1630.75 | 1806.65 | source-private |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | random_noncandidate_same_byte | source_destroying_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1629.18 | 1800.11 | source-private, destroyed |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | random_same_byte | source_destroying_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1631.89 | 1749.64 | source-private, destroyed |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | shuffled_packet | source_destroying_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 4.00 | 1640.81 | 1743.00 | source-private, destroyed |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | structured_free_text_2byte | matched_byte_text_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 0.00 | 0.00 | 0.00 | source-private, text, destroyed |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | structured_json_2byte | matched_byte_text_control | 0.250 | 2.00 | 5.00 | 64 | 128 | 0.00 | 0.01 | 0.01 | source-private, text, destroyed |
| qwen3_seed29_holdout_n128_binary_logprob_combined_cpu | target_only | no_source | 0.250 | 0.00 | 0.00 | 0 | 0 | 0.00 | 0.00 | 0.00 |  |

## Interpretation

This trace measures the cost of consuming the source-private packet with the frozen target-side binary verifier. It separates boundary payload bytes from target-side compute: the source sends a 2-byte payload inside a small packet record, while the current verifier spends one target forward pass per candidate.

## Non-Claims

- This is Mac CPU receiver telemetry, not production GPU/vLLM throughput.
- Cache-line and DMA values are deterministic accounting proxies.
- The current binary verifier has a target-side compute cost that must be reduced or amortized for a systems headline.
