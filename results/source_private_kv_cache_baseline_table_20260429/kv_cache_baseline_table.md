# Source-Private KV/Cache Baseline Table

This is derived accounting over existing endpoint summaries. It is a systems
baseline table, not a KV quantization kernel benchmark.

## Model Geometry

- model_type: `qwen3`
- num_hidden_layers: `28`
- num_attention_heads: `16`
- num_key_value_heads: `8`
- head_dim: `128`
- hidden_size: `1024`

## Bytes Per Prompt Token

| Scheme | Bytes/token |
|---|---:|
| `fp16_bf16` | 114688.0 |
| `int8` | 57344.0 |
| `int4` | 28672.0 |
| `turboquant_3p5bit_proxy` | 25088.0 |
| `turboquant_2p5bit_proxy` | 17920.0 |
| `kivi_2bit_proxy` | 14336.0 |
| `qjl_1bit_sign_proxy` | 7168.0 |

## Headline

- packet payload bytes: `[2.0]`
- minimum non-packet QJL-style 1-bit cache bytes / packet bytes: `10752.0x`
- minimum non-packet KIVI-style 2-bit cache bytes / packet bytes: `21504.0x`

## Rows

| Surface | Condition | Acc | Payload bytes | Prompt delta vs packet | QJL 1-bit bytes | KIVI 2-bit bytes | p50 TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| core n160 label_strict | `matched_packet` | 0.675 | 2.0 | 0.00 | 0.0 | 0.0 | 471.6 |
| core n160 label_strict | `matched_byte_text_2` | 0.250 | 2.0 | 0.00 | 0.0 | 0.0 | 473.0 |
| core n160 label_strict | `query_aware_diag_span` | 0.694 | 14.0 | 3.31 | 23699.2 | 47398.4 | 452.0 |
| core n160 label_strict | `structured_free_text_diag` | 0.713 | 17.0 | 3.00 | 21504.0 | 43008.0 | 474.2 |
| core n160 label_strict | `structured_json_diag` | 0.575 | 21.0 | 5.00 | 35840.0 | 71680.0 | 478.6 |
| core n160 label_strict | `full_hidden_log` | 0.463 | 366.5 | 87.43 | 626707.2 | 1253414.4 | 635.9 |
| holdout n160 label_strict | `matched_packet` | 0.688 | 2.0 | 0.00 | 0.0 | 0.0 | 547.2 |
| holdout n160 label_strict | `matched_byte_text_2` | 0.250 | 2.0 | 0.00 | 0.0 | 0.0 | 545.2 |
| holdout n160 label_strict | `query_aware_diag_span` | 0.688 | 14.0 | 3.31 | 23699.2 | 47398.4 | 558.9 |
| holdout n160 label_strict | `structured_free_text_diag` | 0.719 | 17.0 | 3.00 | 21504.0 | 43008.0 | 557.7 |
| holdout n160 label_strict | `structured_json_diag` | 0.594 | 21.0 | 5.00 | 35840.0 | 71680.0 | 552.5 |
| holdout n160 label_strict | `full_hidden_log` | 0.531 | 373.5 | 90.18 | 646419.2 | 1292838.4 | 730.8 |

## Comparison Axes

| Method family | Source-private? | Decoder side info? | Source-destroying controls? | Systems axis | Paper use |
|---|---:|---:|---:|---|---|
| LatentWire 2-byte source-private packet | `True` | `True` | `True` | extreme-rate private evidence communication | headline method row |
| TurboQuant-style KV/vector quantization | `False` | `False` | `False` | same-model vector/KV compression | byte-floor baseline and caveat |
| QJL-style 1-bit sign sketch | `False` | `False` | `False` | inner-product-preserving KV sketch | same-byte sketch ablation if sparse receiver becomes live |
| KIVI/KVQuant-style low-bit KV cache | `False` | `False` | `False` | same-model long-context KV memory reduction | cache payload accounting baseline |
| SnapKV/CacheGen-style pruning or cache streaming | `False` | `False` | `False` | selected/cache-streamed model-visible context | higher-byte systems comparator |
| vLLM/PagedAttention/DistServe-style serving systems | `False` | `False` | `False` | throughput, TTFT, TPOT, memory scheduling | metric convention and future GPU-serving comparator |

## Interpretation

Derived byte accounting only: this is not a kernel implementation of TurboQuant, QJL, KIVI, or KVQuant. It estimates the minimum KV-cache payload needed to relay the extra private payload tokens in the existing endpoint summaries under several bits-per-element assumptions.

Reviewer caveat: cache quantization methods such as TurboQuant, QJL, KIVI,
KVQuant, SnapKV, and CacheGen attack model-visible KV/context movement.
LatentWire's claim should remain source-private residual communication,
not generic KV-cache compression superiority.
