# Source-Private Systems Rate And Assumption Frontier

- pass gate: `True`
- claim scope: Source-private extreme-rate task communication with explicit public side information; KV/cache rows are byte-floor assumption contrasts, not native accuracy baselines.

## Headline

- endpoint_packet_rows: `2`
- endpoint_packet_rows_passing: `2`
- semantic_medium_pass_rows: `18`
- semantic_medium_total_rows: `18`
- packet_payload_bytes_min: `2.0`
- semantic_packet_budget_bytes: `[4, 8]`
- min_endpoint_packet_delta_vs_target: `0.42500000000000004`
- min_semantic_packet_delta_vs_target: `0.5`
- same_byte_text_accuracy_max: `0.25`
- query_aware_text_oracle_bytes_min: `14.0`
- query_aware_text_bytes_vs_packet: `7.0`
- full_log_bytes_vs_packet_min: `183.225`
- min_full_log_ttft_delta_vs_packet_ms: `164.27308350102976`
- min_kv_byte_floor_vs_packet: `10752.0`
- contract_failure_packet_accuracy: `0.25`
- external_reference_rows: `6`

## Frontier Rows

| Group | Method | Surface | Bytes | Text exposed | KV exposed | Accuracy | Target | Best control | Byte ratio | Native claim |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| endpoint_packet | LatentWire endpoint packet | core n160 label_strict | 2.0 | `False` | `False` | 0.675 | 0.250 | 0.250 | 1.0 | Mac-local endpoint proxy source-private task evidence |
| endpoint_text_relay | query-aware diagnostic text | core n160 label_strict | 14.0 | `True` | `False` |  | 0.250 |  | 7.0 | higher-rate text relay |
| endpoint_text_relay | full hidden-log relay | core n160 label_strict | 366.5 | `True` | `False` |  | 0.250 |  | 183.2 | visible text oracle relay |
| endpoint_packet | LatentWire endpoint packet | holdout n160 label_strict | 2.0 | `False` | `False` | 0.688 | 0.250 | 0.250 | 1.0 | Mac-local endpoint proxy source-private task evidence |
| endpoint_text_relay | query-aware diagnostic text | holdout n160 label_strict | 14.0 | `True` | `False` |  | 0.250 |  | 7.0 | higher-rate text relay |
| endpoint_text_relay | full hidden-log relay | holdout n160 label_strict | 373.5 | `True` | `False` |  | 0.250 |  | 186.8 | visible text oracle relay |
| contract_failure | LatentWire packet with under-specified receiver | core n16 terse | 2.0 | `False` | `False` | 0.250 | 0.250 | 0.250 | 1.0 | negative prompt-contract ablation |
| semantic_anchor_medium | semantic-anchor source-private packet | heldout paraphrase n512 x 3 seeds | 4.0 | `False` | `False` |  | 0.250 | 0.250 | 1.0 | medium seed-stable held-out paraphrase source-private communication |
| semantic_anchor_medium | semantic-anchor source-private packet | heldout paraphrase n512 x 3 seeds | 8.0 | `False` | `False` |  | 0.250 | 0.254 | 1.0 | medium seed-stable held-out paraphrase source-private communication |
| rate_frontier_text | same-byte structured text | core+holdout rate frontier | 2.0 | `True` | `False` | 0.250 | 0.250 |  | 1.0 | same-byte visible text control |
| rate_frontier_text | query-aware structured text oracle | core+holdout rate frontier | 14.0 | `True` | `False` | 1.000 | 0.250 |  | 7.0 | higher-rate visible text oracle |
| rate_frontier_text | JSON/free-text oracle relay | core+holdout rate frontier | 21.0 | `True` | `False` | 1.000 | 0.250 |  | 10.5 | higher-rate visible text oracle |
| kv_byte_floor | QJL-style 1-bit source KV byte floor | endpoint source context byte accounting | 21504.0 | `False` | `True` |  |  |  | 10752.0 | KV/cache compression byte-floor neighbor |
| kv_byte_floor | KIVI/KVQuant-style 2-bit source KV byte floor | endpoint source context byte accounting | 43008.0 | `False` | `True` |  |  |  | 21504.0 | KV/cache compression byte-floor neighbor |
| external_reference | C2C cache-to-cache communication | external primary-source reference |  | `False` | `True` |  |  |  |  | cross-model internal-state communication |
| external_reference | KVComm / KVCOMM selective KV communication | external primary-source reference |  | `False` | `True` |  |  |  |  | cross-context or multi-agent KV-cache communication |
| external_reference | TurboQuant | external primary-source reference |  | `False` | `True` |  |  |  |  | same-model online vector/KV quantization |
| external_reference | QJL | external primary-source reference |  | `False` | `True` |  |  |  |  | same-model KV/sign-sketch compression |
| external_reference | LLMLingua / LLMLingua-2 | external primary-source reference |  | `False` | `False` |  |  |  |  | prompt compression |
| external_reference | Gist tokens | external primary-source reference |  | `False` | `False` |  |  |  |  | learned prompt/context compression |

## Related Work Positioning

| Method | Source | Role | Positioning |
|---|---|---|---|
| C2C / cache-to-cache communication | https://arxiv.org/abs/2510.03215 | closest high-rate internal-state communication baseline | different access model: dense source/target KV or cache state, not public endpoint packet |
| KVComm / KV sharing | https://openreview.net/forum?id=F7rUng23nw | KV communication systems neighbor | selects KV tensors; useful as assumption contrast, not same source-private packet task |
| TurboQuant | https://arxiv.org/abs/2504.19874 | low-bit online vector/KV quantization neighbor | byte-floor comparator only unless run on native KV task |
| QJL | https://arxiv.org/abs/2406.03482 | sign-sketch / low-bit KV quantization neighbor | compares estimated KV payload bytes, not endpoint accuracy |
| KIVI / KVQuant | https://arxiv.org/abs/2402.02750; https://arxiv.org/abs/2401.18079 | 2-bit / ultra-long-context KV compression neighbor | same-model cache compression, not source-private communication |
| LLMLingua / Gist tokens | https://aclanthology.org/2023.emnlp-main.825/; https://openreview.net/forum?id=2DtxPCL3T5 | prompt compression baseline family | text/context compression; include rate ladder rather than overclaiming packet superiority |

## Non-Claims

- No claim of beating TurboQuant, QJL, KIVI, KVQuant, C2C, or KVComm on native KV/cache tasks.
- No production GPU serving throughput claim from Mac-local CPU proxy rows.
- No prompt-contract-free claim; the terse receiver row is a recorded failure.
- No broad latent-transfer claim; the semantic-anchor receiver uses public candidate side information.

## Interpretation

The systems win is an assumption-aware rate frontier: LatentWire packets occupy a 2-8 byte source-private regime where same-byte text fails, higher-byte text relays catch up by exposing private text, and KV/cache methods require internal-state transport and much larger byte floors.
