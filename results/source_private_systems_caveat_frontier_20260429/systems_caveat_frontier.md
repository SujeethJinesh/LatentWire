# Source-Private Systems Caveat Frontier

- pass gate: `True`
- paper claim scope: Mac-local endpoint proxy plus derived KV/cache byte floors; production serving remains future work.

## Headline

- passing_endpoint_rows: `2`
- endpoint_rows: `2`
- packet_payload_bytes: `2.0`
- min_packet_minus_target_accuracy: `0.42500000000000004`
- min_packet_minus_best_control_accuracy: `0.42500000000000004`
- min_packet_vs_query_payload_compression: `7.0`
- min_packet_vs_full_log_payload_compression: `183.25`
- min_full_log_ttft_delta_vs_packet_ms: `164.27308350102976`
- min_packet_vs_target_ci95_low: `0.35`
- min_qjl_1bit_cache_bytes_vs_packet: `10752.0`
- terse_prompt_pass_gate: `False`

## Endpoint Rows

| Surface | Pass | Packet acc | Target acc | Best control | Packet bytes | Query bytes | Full-log bytes | Full-log TTFT delta ms | Caveat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| core n160 label_strict | `True` | 0.675 | 0.250 | 0.250 | 2.0 | 14.0 | 366.5 | 164.3 | Packet beats target and source-destroying controls at 2 bytes; query-aware/structured text can match or beat accuracy at higher byte rates; TTFT is local CPU proxy telemetry. |
| holdout n160 label_strict | `True` | 0.688 | 0.250 | 0.250 | 2.0 | 14.0 | 373.5 | 183.5 | Packet beats target and source-destroying controls at 2 bytes; query-aware/structured text can match or beat accuracy at higher byte rates; TTFT is local CPU proxy telemetry. |
| core n16 terse | `False` | 0.250 | 0.250 | 0.250 | 2.0 | 14.0 | 366.5 | 283.4 | Under-specified prompt contract: the packet collapses to target accuracy, so the receiver contract is a required part of the method. |

## Related Systems Positioning

| Method | Source | Role | Comparison axis |
|---|---|---|---|
| LatentWire 2-byte source-private packet | this work | headline systems row | source-private residual evidence, 2-byte payload, strict controls |
| C2C cache-to-cache communication | https://arxiv.org/abs/2510.03215 | closest high-rate cross-model internal-state baseline | source/target KV-cache projection rather than public-side-info packet |
| KVComm selective KV sharing | https://openreview.net/forum?id=F7rUng23nw | KV communication baseline/framing | selected KV pairs/layers rather than extreme-rate private packet |
| TurboQuant | https://arxiv.org/abs/2504.19874 | low-bit vector/KV byte-floor comparator | same-model vector quantization, not source-destroying communication control |
| QJL | https://arxiv.org/abs/2406.03482 | 1-bit sign-sketch byte-floor comparator | inner-product preserving high-dimensional sketch |
| vLLM / PagedAttention and DistServe | https://arxiv.org/abs/2309.06180; https://arxiv.org/abs/2401.09670 | serving metric convention | future production TTFT/TPOT/throughput baseline |
| Diffusion/JEPA latent prediction | https://arxiv.org/abs/2212.09748; https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf | future learned-interface inspiration | latent prediction objective, not current systems baseline |

## Interpretation

The systems contribution is an extreme-rate communication frontier: a 2-byte source-private packet passes strict endpoint controls on Mac CPU n160 rows, while visible text/KV-style alternatives occupy higher byte-rate regimes. This artifact deliberately records that query-aware structured text can match accuracy at 14 bytes and that the measured TTFT/E2E rows are local CPU proxy telemetry, not production throughput.

## Non-Claims

- No claim of beating TurboQuant, QJL, KIVI, KVQuant, C2C, or KVComm on their native same-model/KV tasks.
- No production GPU serving throughput claim until vLLM/OpenAI-compatible server runs are available.
- No broad cross-family latent-transfer claim; the current cross-family appendix is negative-boundary evidence.
- No prompt-contract-free receiver claim; the terse prompt failure shows the public receiver contract matters.
