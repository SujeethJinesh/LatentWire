# Source-Private Byte-Amplification Ablation

- pass gate: `True`
- benchmark rows: `4`
- interface rows: `40`
- packet framed-byte range: `4-15B`
- max single-request cache-line amplification: `16.0x`
- minimum KV floor vs 64B padded packet: `12.0x`
- score-vector floor vs minimum packet: `2.0x`

## Table

| Dataset | Split | Interface | Object | Accuracy | Bytes | Cacheline | Batch64 | Private | Exposure | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| ARC-Challenge | test | latentwire_framed_packet | framed source-private task packet | 0.344 | 15B | 64B | 15B | `true` | none | mac_local_cached_packet |
| ARC-Challenge | test | latentwire_cacheline_padded_packet | same packet padded to one 64B cache line | 0.344 | 64B | 64B | 64B | `true` | none | cacheline_amplification_counterfactual |
| ARC-Challenge | test | source_score_vector_fp16_floor | 4-choice fp16 source score vector | 0.344 | 8B | 64B | 8B | `false` | score | score_vector_floor_not_source_private |
| ARC-Challenge | test | qjl_1bit_kv_floor | one-token K+V state at 1 bit/element | 0.344 | 768B | 768B | 768B | `false` | KV | optimistic_qjl_byte_floor_not_native |
| ARC-Challenge | test | turboquant_3p5bit_kv_floor | one-token K+V state at 3.5 bits/element | 0.344 | 2688B | 2688B | 2688B | `false` | KV | turboquant_byte_floor_not_native |
| ARC-Challenge | test | c2c_fp16_kv_floor | one-token source K+V cache at fp16 | 0.344 | 12288B | 12288B | 12288B | `false` | KV | c2c_floor_not_native |
| OpenBookQA | test | latentwire_framed_packet | framed source-private task packet | 0.378 | 6B | 64B | 6B | `true` | none | mac_local_cached_packet |
| OpenBookQA | test | latentwire_cacheline_padded_packet | same packet padded to one 64B cache line | 0.378 | 64B | 64B | 64B | `true` | none | cacheline_amplification_counterfactual |
| OpenBookQA | test | source_score_vector_fp16_floor | 4-choice fp16 source score vector | 0.378 | 8B | 64B | 8B | `false` | score | score_vector_floor_not_source_private |
| OpenBookQA | test | qjl_1bit_kv_floor | one-token K+V state at 1 bit/element | 0.378 | 768B | 768B | 768B | `false` | KV | optimistic_qjl_byte_floor_not_native |
| OpenBookQA | test | turboquant_3p5bit_kv_floor | one-token K+V state at 3.5 bits/element | 0.378 | 2688B | 2688B | 2688B | `false` | KV | turboquant_byte_floor_not_native |
| OpenBookQA | test | c2c_fp16_kv_floor | one-token source K+V cache at fp16 | 0.378 | 12288B | 12288B | 12288B | `false` | KV | c2c_floor_not_native |
| HellaSwag | validation_first1024 | latentwire_framed_packet | framed source-private task packet | 0.461 | 5B | 64B | 5B | `true` | none | mac_local_cached_packet |
| HellaSwag | validation_first1024 | latentwire_cacheline_padded_packet | same packet padded to one 64B cache line | 0.461 | 64B | 64B | 64B | `true` | none | cacheline_amplification_counterfactual |
| HellaSwag | validation_first1024 | source_score_vector_fp16_floor | 4-choice fp16 source score vector | 0.461 | 8B | 64B | 8B | `false` | score | score_vector_floor_not_source_private |
| HellaSwag | validation_first1024 | qjl_1bit_kv_floor | one-token K+V state at 1 bit/element | 0.461 | 768B | 768B | 768B | `false` | KV | optimistic_qjl_byte_floor_not_native |
| HellaSwag | validation_first1024 | turboquant_3p5bit_kv_floor | one-token K+V state at 3.5 bits/element | 0.461 | 2688B | 2688B | 2688B | `false` | KV | turboquant_byte_floor_not_native |
| HellaSwag | validation_first1024 | c2c_fp16_kv_floor | one-token source K+V cache at fp16 | 0.461 | 12288B | 12288B | 12288B | `false` | KV | c2c_floor_not_native |
| HellaSwag | validation_full_compaction | latentwire_framed_packet | framed source-private task packet | 0.619 | 4B | 64B | 4B | `true` | none | mac_local_cached_packet |
| HellaSwag | validation_full_compaction | latentwire_cacheline_padded_packet | same packet padded to one 64B cache line | 0.619 | 64B | 64B | 64B | `true` | none | cacheline_amplification_counterfactual |
| HellaSwag | validation_full_compaction | source_score_vector_fp16_floor | 4-choice fp16 source score vector | 0.619 | 8B | 64B | 8B | `false` | score | score_vector_floor_not_source_private |
| HellaSwag | validation_full_compaction | qjl_1bit_kv_floor | one-token K+V state at 1 bit/element | 0.619 | 768B | 768B | 768B | `false` | KV | optimistic_qjl_byte_floor_not_native |
| HellaSwag | validation_full_compaction | turboquant_3p5bit_kv_floor | one-token K+V state at 3.5 bits/element | 0.619 | 2688B | 2688B | 2688B | `false` | KV | turboquant_byte_floor_not_native |
| HellaSwag | validation_full_compaction | c2c_fp16_kv_floor | one-token source K+V cache at fp16 | 0.619 | 12288B | 12288B | 12288B | `false` | KV | c2c_floor_not_native |

## Interpretation

This ablation holds cached packet predictions fixed and varies the communicated object. It shows how much byte movement is added by single-request cache-line padding, and how far even optimistic one-token KV/source-state floors remain from the packet regime. The fp16 score-vector row is intentionally included as a reviewer stress test: it is byte-small, but it exposes raw source scores and is not source-private.

## Non-Claims

- This is not a native NVIDIA serving benchmark.
- QJL, TurboQuant, KIVI, KVQuant, KVComm, and C2C rows are byte floors or counterfactual same-prediction accounting rows, not defeated native baselines.
- Exact prediction equivalence for source-state rows is a held-fixed accounting assumption, not a measured quality result.
- Small source-score vectors are not source-private and should not be merged with LatentWire packet rows.
