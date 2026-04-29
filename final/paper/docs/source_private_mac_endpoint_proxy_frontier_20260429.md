# Source-Private Mac Endpoint-Proxy Frontier

- date: `2026-04-29`
- artifact: `results/source_private_mac_endpoint_proxy_frontier_20260429/`
- script: `scripts/run_source_private_mac_endpoint_proxy_frontier.py`
- tests: `tests/test_run_source_private_mac_endpoint_proxy_frontier.py`,
  `tests/test_build_source_private_cpu_systems_frontier.py`
- scale rung: strict-small endpoint-proxy prompt-stress smoke

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
- `random_same_byte_packet`: random valid-looking two-character packet with
  the same payload budget.
- `deranged_candidate_diag_table`: matched packet with the public candidate
  diagnostic table deranged.
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
| core n16 terse | target-only | 0.250 | 0.0 | 169.4 | 533.0 | 2966.2 |
| core n16 terse | matched packet | 0.250 | 2.0 | 169.4 | 643.7 | 2898.3 |
| core n16 terse | matched-byte text | 0.250 | 2.0 | 168.4 | 643.5 | 2848.0 |
| core n16 audit | target-only | 0.250 | 0.0 | 193.4 | 536.9 | 2058.9 |
| core n16 audit | matched packet | 0.750 | 2.0 | 193.4 | 455.9 | 775.3 |
| core n16 audit | matched-byte text | 0.250 | 2.0 | 192.4 | 432.2 | 2017.4 |
| holdout n16 audit | target-only | 0.312 | 0.0 | 193.3 | 442.4 | 2037.1 |
| holdout n16 audit | matched packet | 0.875 | 2.0 | 193.3 | 438.2 | 761.6 |
| holdout n16 audit | matched-byte text | 0.312 | 2.0 | 192.3 | 411.7 | 2017.1 |
| core n32 audit | target-only | 0.250 | 0.0 | 193.4 | 475.7 | 2014.0 |
| core n32 audit | matched packet | 0.719 | 2.0 | 193.4 | 484.0 | 815.5 |
| core n32 audit | matched-byte text | 0.281 | 2.0 | 192.4 | 483.1 | 1992.7 |
| holdout n32 audit | target-only | 0.312 | 0.0 | 193.3 | 455.7 | 1985.8 |
| holdout n32 audit | matched packet | 0.844 | 2.0 | 193.3 | 452.1 | 821.8 |
| holdout n32 audit | matched-byte text | 0.312 | 2.0 | 192.3 | 434.8 | 1954.5 |
| core n16 audit strict controls | target-only | 0.250 | 0.0 | 193.4 | 753.8 | 2917.7 |
| core n16 audit strict controls | matched packet | 0.750 | 2.0 | 193.4 | 727.7 | 1067.1 |
| core n16 audit strict controls | strict-label matched packet | 0.062 | 2.0 | 193.4 | 727.7 | 1067.1 |
| core n16 audit strict controls | random same-byte packet | 0.000 | 2.0 | 193.4 | 649.1 | 1761.5 |
| core n16 audit strict controls | deranged public table | 0.000 | 2.0 | 193.4 | 822.5 | 1278.9 |
| holdout n16 audit strict controls | target-only | 0.312 | 0.0 | 193.3 | 478.5 | 2025.1 |
| holdout n16 audit strict controls | matched packet | 0.875 | 2.0 | 193.3 | 461.2 | 858.3 |
| holdout n16 audit strict controls | strict-label matched packet | 0.250 | 2.0 | 193.3 | 461.2 | 858.3 |
| holdout n16 audit strict controls | random same-byte packet | 0.125 | 2.0 | 193.3 | 458.5 | 2118.4 |
| holdout n16 audit strict controls | deranged public table | 0.000 | 2.0 | 193.3 | 450.0 | 1338.4 |
| core n32 audit strict controls | target-only | 0.250 | 0.0 | 193.3 | 468.5 | 2051.1 |
| core n32 audit strict controls | matched packet | 0.719 | 2.0 | 193.3 | 460.8 | 789.1 |
| core n32 audit strict controls | strict-label matched packet | 0.156 | 2.0 | 193.3 | 460.8 | 789.1 |
| core n32 audit strict controls | matched-byte text | 0.281 | 2.0 | 192.3 | 451.3 | 1950.0 |
| core n32 audit strict controls | random same-byte packet | 0.031 | 2.0 | 193.3 | 493.2 | 1928.1 |
| core n32 audit strict controls | deranged public table | 0.000 | 2.0 | 193.3 | 447.5 | 856.1 |
| holdout n32 audit strict controls | target-only | 0.312 | 0.0 | 193.3 | 469.9 | 2037.7 |
| holdout n32 audit strict controls | matched packet | 0.844 | 2.0 | 193.3 | 461.2 | 736.0 |
| holdout n32 audit strict controls | strict-label matched packet | 0.219 | 2.0 | 193.3 | 461.2 | 736.0 |
| holdout n32 audit strict controls | matched-byte text | 0.312 | 2.0 | 192.3 | 447.8 | 2000.6 |
| holdout n32 audit strict controls | random same-byte packet | 0.094 | 2.0 | 193.3 | 464.6 | 1441.5 |
| holdout n32 audit strict controls | deranged public table | 0.000 | 2.0 | 193.3 | 463.8 | 857.4 |
| core n64 audit payload-gated | target-only | 0.250 | 0.0 | 193.3 | 493.8 | 2210.9 |
| core n64 audit payload-gated | matched packet | 0.750 | 2.0 | 193.3 | 466.2 | 905.6 |
| core n64 audit payload-gated | matched-byte text | 0.203 | 2.0 | 192.3 | 466.8 | 2174.1 |
| core n64 audit payload-gated | random same-byte packet | 0.000 | 2.0 | 193.3 | 471.9 | 2057.2 |
| core n64 audit payload-gated | deranged public table | 0.000 | 2.0 | 193.3 | 501.0 | 905.2 |
| core n16 label-strict | target-only | 0.250 | 0.0 | 218.4 | 481.2 | 2166.2 |
| core n16 label-strict | matched packet | 0.688 | 2.0 | 218.4 | 508.8 | 2130.9 |
| core n16 label-strict | matched-byte text | 0.250 | 2.0 | 217.4 | 479.9 | 2194.9 |
| core n16 label-strict | random same-byte packet | 0.000 | 2.0 | 218.4 | 480.6 | 2152.7 |
| core n16 label-strict | deranged public table | 0.188 | 2.0 | 218.4 | 478.4 | 2179.0 |
| holdout n16 label-strict | target-only | 0.250 | 0.0 | 218.4 | 497.8 | 2231.2 |
| holdout n16 label-strict | matched packet | 0.625 | 2.0 | 218.4 | 509.4 | 2089.4 |
| holdout n16 label-strict | matched-byte text | 0.250 | 2.0 | 217.4 | 603.9 | 2118.6 |
| holdout n16 label-strict | random same-byte packet | 0.000 | 2.0 | 218.4 | 477.3 | 2187.8 |
| holdout n16 label-strict | deranged public table | 0.250 | 2.0 | 218.4 | 503.5 | 2095.9 |
| core n32 label-strict | target-only | 0.250 | 0.0 | 218.3 | 519.9 | 2083.7 |
| core n32 label-strict | matched packet | 0.688 | 2.0 | 218.3 | 495.9 | 2101.8 |
| core n32 label-strict | matched-byte text | 0.250 | 2.0 | 217.3 | 497.3 | 2090.8 |
| core n32 label-strict | random same-byte packet | 0.000 | 2.0 | 218.3 | 514.6 | 2079.7 |
| core n32 label-strict | deranged public table | 0.219 | 2.0 | 218.3 | 466.4 | 2141.0 |
| holdout n32 label-strict | target-only | 0.250 | 0.0 | 218.3 | 484.0 | 2156.4 |
| holdout n32 label-strict | matched packet | 0.656 | 2.0 | 218.3 | 505.5 | 2131.4 |
| holdout n32 label-strict | matched-byte text | 0.250 | 2.0 | 217.3 | 484.0 | 2158.0 |
| holdout n32 label-strict | random same-byte packet | 0.000 | 2.0 | 218.3 | 477.5 | 2124.1 |
| holdout n32 label-strict | deranged public table | 0.250 | 2.0 | 218.3 | 492.2 | 2145.0 |
| core n64 label-strict | target-only | 0.250 | 0.0 | 218.3 | 533.6 | 2319.7 |
| core n64 label-strict | matched packet | 0.703 | 2.0 | 218.3 | 569.9 | 2275.8 |
| core n64 label-strict | matched-byte text | 0.250 | 2.0 | 217.3 | 537.4 | 2332.8 |
| core n64 label-strict | random same-byte packet | 0.000 | 2.0 | 218.3 | 563.5 | 2323.5 |
| core n64 label-strict | deranged public table | 0.234 | 2.0 | 218.3 | 565.5 | 2268.9 |
| holdout n64 label-strict | target-only | 0.250 | 0.0 | 218.3 | 501.4 | 2079.2 |
| holdout n64 label-strict | matched packet | 0.672 | 2.0 | 218.3 | 508.0 | 2071.2 |
| holdout n64 label-strict | matched-byte text | 0.250 | 2.0 | 217.3 | 481.1 | 2097.2 |
| holdout n64 label-strict | random same-byte packet | 0.000 | 2.0 | 218.3 | 485.2 | 2085.0 |
| holdout n64 label-strict | deranged public table | 0.250 | 2.0 | 218.3 | 509.0 | 2088.0 |

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
- audit paraphrase `n=32`: core packet `0.719` versus target `0.250` and
  matched-byte text `0.281`; holdout packet `0.844` versus target and
  matched-byte text `0.312`. Full-log p50 TTFT remains `+157.4 ms` to
  `+163.4 ms` versus the packet.
- terse paraphrase `n=16`: core packet collapses to target-only (`0.250`),
  showing that the receiver needs a sufficiently explicit public side-
  information contract.
- strict-control audit with payload-gated parsing: accuracy remains high, but
  the old audit rows no longer pass the valid-output gate because Qwen sometimes
  emits unrelated diagnostic codes. Core `n=64` is an important near miss:
  packet `0.750`, target `0.250`, best source-destroying control `0.203`, but
  packet valid rate is only `0.781`.
- label-strict receiver prompt: both surfaces pass with exact-label outputs at
  `n=16`, `n=32`, and `n=64`. At `n=64`, core packet is `0.703` versus best
  source-destroying control `0.250`, and holdout packet is `0.672` versus best
  source-destroying control `0.250`; valid prediction rate is `1.000` on both.
  Full-log p50 TTFT is `+217.2 ms` core and `+192.7 ms` holdout versus the
  packet.
- paired uncertainty for the `n=64` label-strict rows passes with `5000`
  bootstrap samples. The minimum packet-vs-target lower CI is `+0.297`, the
  minimum packet-vs-best-source-destroying-control lower CI is `+0.297`, and
  the minimum strict-label packet-vs-target lower CI is `+0.281`. Query-aware
  diagnostic text is accuracy-comparable (`packet-query` CI ranges from
  `[-0.109, 0.078]`) but costs `14` bytes rather than the `2` byte packet, so
  it remains a rate/quality comparator, not a failed control.
- core `n=160` label-strict scale-up passes after the same all-condition CPU
  endpoint gate. The packet reaches `0.675` accuracy and `0.662` strict-label
  accuracy versus target-only and matched-byte text at `0.250`, random
  same-byte at `0.000`, deranged public table at `0.244`, and best
  source-destroying control at `0.250`. Paired bootstrap lower CIs are
  `+0.350` versus target and best control, and `+0.338` for strict-label
  packet versus target. Query-aware text is `0.694` at `14` bytes; the packet
  is slightly lower accuracy but `7.0x` smaller. Full-log relay is `183.2x`
  larger and adds `+164.3 ms` p50 TTFT versus the packet.

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
the diagnostic code (`G0`) rather than a candidate label, and could hallucinate
codes that were not in the source payload. The parser is now payload-gated: a
diagnostic code is mapped only if that code was actually transmitted. This
demotes the audit prompt from pass to near-miss under the valid-output gate.
The new `label_strict` prompt fixes the stronger reviewer risk by requiring a
full candidate label copied exactly; at `n=64`, packet strict-label accuracy is
`0.672` core and `0.656` holdout with valid rate `1.000`.

## Reviewer Caveats

- This is an `n=64 + n=64` label-strict endpoint-proxy gate plus an `n=64`
  payload-gated audit near-miss, not a large benchmark.
- Timing is local CPU generate timing, not real vLLM/OpenAI-compatible serving
  TTFT or throughput.
- The receiver prompt explicitly describes the packet interface. The next
  robustness gate should scale canonical/audit prompts and the strict controls
  to `n=64` or `n=160`.
- Verbose relays remain strong accuracy oracles; our claim is rate efficiency
  and source-control cleanliness, not dominance over all higher-byte relays.

## Next Gate

Run the frozen holdout `n=160` `label_strict` endpoint gate and then the paired
core+holdout `n=160` uncertainty summary. When NVIDIA GPUs are available, run a
server-side vLLM/GenAI-Perf style TTFT/throughput benchmark against structured
text, query-aware text, full-log relay, and KV/cache transport baselines.
