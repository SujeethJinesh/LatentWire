# Source-Private Mac Endpoint-Proxy Frontier

- date: `2026-04-29`
- artifact: `results/source_private_mac_endpoint_proxy_frontier_20260429/`
- script: `scripts/run_source_private_mac_endpoint_proxy_frontier.py`
- tests: `tests/test_run_source_private_mac_endpoint_proxy_frontier.py`,
  `tests/test_build_source_private_cpu_systems_frontier.py`
- scale rung: strict-small endpoint-proxy smoke

## Purpose

The deterministic byte frontier showed that a `2` byte packet can carry the
private diagnostic evidence more efficiently than structured text or full
hidden-log relay. This gate asks whether the same packet remains useful under a
frozen target-model receiver prompt while logging endpoint-style prompt bytes,
prompt tokens, generated tokens, p50/p95 TTFT, and p50/p95 E2E CPU decode time.

This is not a vLLM/server throughput benchmark. It is a local Mac CPU endpoint
proxy that keeps the receiver, candidate pool, payload, and decode settings
fixed across conditions.

## Conditions

- `target_only`: public candidates plus target prior, no source payload.
- `matched_packet`: two-character private diagnostic packet.
- `matched_byte_text_2`: structured text relay truncated to the same two-byte
  budget.
- `query_aware_diag_span`: shortest query-aware diagnostic-span text.
- `structured_json_diag`: JSON diagnostic relay.
- `structured_free_text_diag`: free-text diagnostic relay.
- `full_hidden_log`: complete private source log relay.

## Results

| Surface | Condition | Accuracy | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---|---:|---:|---:|---:|---:|
| core n8 | target-only | 0.250 | 0.0 | 218.4 | 494.6 | 2056.2 |
| core n8 | matched packet | 0.750 | 2.0 | 216.4 | 488.1 | 1412.6 |
| core n8 | matched-byte text | 0.250 | 2.0 | 215.4 | 560.6 | 2120.7 |
| core n8 | query-aware text | 0.750 | 14.0 | 219.8 | 510.0 | 2075.6 |
| core n8 | structured JSON | 1.000 | 21.0 | 221.4 | 569.7 | 1400.5 |
| core n8 | structured free text | 1.000 | 17.0 | 219.4 | 544.9 | 860.8 |
| core n8 | full hidden log | 1.000 | 366.5 | 303.9 | 669.2 | 1373.7 |
| holdout n8 | target-only | 0.250 | 0.0 | 218.5 | 532.8 | 2109.6 |
| holdout n8 | matched packet | 0.750 | 2.0 | 216.5 | 481.0 | 1434.9 |
| holdout n8 | matched-byte text | 0.250 | 2.0 | 215.5 | 493.9 | 2038.8 |
| holdout n8 | query-aware text | 0.625 | 14.0 | 220.0 | 558.7 | 2038.4 |
| holdout n8 | structured JSON | 0.875 | 21.0 | 221.5 | 532.5 | 861.4 |
| holdout n8 | structured free text | 0.875 | 17.0 | 219.5 | 498.3 | 933.2 |
| holdout n8 | full hidden log | 1.000 | 373.5 | 306.9 | 760.5 | 1440.8 |
| core n16 | target-only | 0.250 | 0.0 | 218.4 | 494.6 | 2095.7 |
| core n16 | matched packet | 0.688 | 2.0 | 216.4 | 525.7 | 2057.4 |
| core n16 | matched-byte text | 0.250 | 2.0 | 215.4 | 544.8 | 2139.0 |
| core n16 | query-aware text | 0.812 | 14.0 | 219.8 | 454.9 | 2075.8 |
| core n16 | structured JSON | 1.000 | 21.0 | 221.4 | 502.8 | 1544.9 |
| core n16 | structured free text | 1.000 | 17.0 | 219.4 | 566.7 | 825.4 |
| core n16 | full hidden log | 1.000 | 366.5 | 303.9 | 691.1 | 1421.1 |
| holdout n16 | target-only | 0.250 | 0.0 | 218.4 | 502.9 | 1978.8 |
| holdout n16 | matched packet | 0.688 | 2.0 | 216.4 | 454.1 | 1989.8 |
| holdout n16 | matched-byte text | 0.250 | 2.0 | 215.4 | 446.7 | 2018.2 |
| holdout n16 | query-aware text | 0.750 | 14.0 | 219.9 | 474.1 | 1993.1 |
| holdout n16 | structured JSON | 0.938 | 21.0 | 221.4 | 532.2 | 855.9 |
| holdout n16 | structured free text | 0.938 | 17.0 | 219.4 | 467.6 | 865.0 |
| holdout n16 | full hidden log | 1.000 | 373.5 | 306.8 | 644.8 | 1366.7 |

Both frozen surfaces pass the endpoint-proxy gate at `n=8` and `n=16`:

- core: packet `0.750` versus target-only and matched-byte text `0.250`;
  query-aware text is `7.0x` larger and full hidden-log relay is `183.2x`
  larger. Full-log p50 TTFT is `+181.1 ms` versus the packet.
- holdout: packet `0.750` versus target-only and matched-byte text `0.250`;
  query-aware text is `7.0x` larger and full hidden-log relay is `186.7x`
  larger. Full-log p50 TTFT is `+279.5 ms` versus the packet.
- core `n=16`: packet `0.688` versus target-only and matched-byte text
  `0.250`; full-log p50 TTFT is `+165.4 ms` versus the packet.
- holdout `n=16`: packet `0.688` versus target-only and matched-byte text
  `0.250`; full-log p50 TTFT is `+190.7 ms` versus the packet.

## Interpretation

This strengthens the systems contribution, but the claim should be precise.
The 2-byte packet is not always more accurate than verbose relays; structured
JSON/free-text and full hidden logs often recover oracle accuracy because they
carry the same private diagnostic in a more verbose format. The win is the
far-left byte/TTFT frontier: the packet gives the frozen receiver a large gain
over target-only and matched-byte text at two bytes, while full-log relay adds
substantial prompt tokens and CPU TTFT. E2E latency is not a clean win in this
CPU proxy because verbose relays often cause the model to emit fewer completion
tokens; TTFT and byte/prompt-token deltas are the safer systems readout.

The first attempted run exposed a useful harness issue: Qwen sometimes emitted
the diagnostic code (`G0`) rather than a candidate label. The parser now maps a
unique emitted diagnostic code back to the candidate whose public
`handles_repair_diag` field matches it. This is auditable and preserves the
target-side public-side-information interpretation.

## Reviewer Caveats

- This is an `n=16 + n=16` endpoint-proxy smoke plus earlier `n=8 + n=8`
  diagnostic rows, not a large benchmark.
- Timing is local CPU generate timing, not real vLLM/OpenAI-compatible serving
  TTFT or throughput.
- The receiver prompt explicitly describes the packet interface, so the next
  robustness gate should test prompt paraphrases and an `n=64`/`n=160` slice.
- Verbose relays remain strong accuracy oracles; our claim is rate efficiency
  and source-control cleanliness, not dominance over all higher-byte relays.

## Next Gate

Run the same endpoint-proxy gate at `n=64` or `n=160` with prompt paraphrase
stress and, when NVIDIA GPUs are available, a server-side vLLM/GenAI-Perf style
TTFT/throughput benchmark against structured text, query-aware text, full-log
relay, and KV/cache transport baselines.
