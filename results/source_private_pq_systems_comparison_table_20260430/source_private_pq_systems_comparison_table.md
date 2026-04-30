# Source-Private PQ Systems Comparison Table

- pass gate: `True`
- PQ geometry rows: `4`
- PQ mitigation rows: `3`
- PQ min delta vs best source-destroying control: `0.212`
- canonical PQ max cached decode p50: `0.0212 ms`
- protected Hadamard min unique payloads: `386`
- frozen verifier min accuracy: `1.000`
- frozen verifier max Mac CPU p50: `1674.1 ms`
- same-byte text max accuracy: `0.250`
- query-aware text raw-byte ratio: `7.0x`
- full-log raw-byte ratio min: `183.25x`
- KV raw-byte ratio min: `10752.0x`

## Compact Rows

| Group | Method | Bytes | Accuracy | Target | Best control | Pass | Exposes text | Exposes KV | Paper use |
|---|---|---:|---:|---:|---:|---|---|---|---|
| pq_geometry_method | canonical 4-byte product-codebook packet | 4 | 0.482-0.52 | 0.25 | 0.268 | pass_source_controls_lookup_risk | false | false | baseline compression-native PQ packet |
| pq_geometry_method | protected Hadamard product-codebook packet | 4 | 0.498-0.514 | 0.25 | 0.268 | pass_mitigation | false | false | hardware-friendly geometry mitigation |
| pq_geometry_method | utility-OPQ product-codebook packet | 4 | 0.48-0.514 | 0.25 | 0.268 | pass_mitigation | false | false | public-mean-sensitive geometry mitigation |
| pq_geometry_method | utility-protected Hadamard product-codebook packet | 4 | 0.504-0.516 | 0.25 | 0.268 | pass_mitigation | false | false | strongest lookup-risk mitigation row |
| source_coding_baseline | scalar Wyner-Ziv residual packet | 4 | 0.424-0.504 | 0.25 | 0.268 | source_coding_comparator | false | false | direct learned residual-code comparator |
| model_mediated_receiver | frozen Qwen3 binary-verifier packet | 2 | 1-1 | 0.25 | 0.25 | pass_model_mediated_controls | false | false | model-mediated consumption evidence |
| endpoint_text_relay | query-aware diagnostic text | 14 | - | 0.25 |  | text_rate_comparator | true | false | prompt/text compression frontier |
| endpoint_text_relay | full hidden-log relay | 366.5 | - | 0.25 |  | text_rate_comparator | true | false | upper text relay / TTFT contrast |
| endpoint_text_relay | query-aware diagnostic text | 14 | - | 0.25 |  | text_rate_comparator | true | false | prompt/text compression frontier |
| endpoint_text_relay | full hidden-log relay | 373.5 | - | 0.25 |  | text_rate_comparator | true | false | upper text relay / TTFT contrast |
| rate_frontier_text | same-byte structured text | 2 | 0.25-0.25 | 0.25 |  | text_rate_comparator | true | false | text baseline |
| rate_frontier_text | query-aware structured text oracle | 14 | 1-1 | 0.25 |  | text_rate_comparator | true | false | honest text catches-up row |
| rate_frontier_text | JSON/free-text oracle relay | 21 | 1-1 | 0.25 |  | text_rate_comparator | true | false | structured text rate ladder |
| kv_byte_floor | QJL-style 1-bit source KV byte floor | 21504 | - |  |  | accounting_contrast | false | true | assumption contrast, not accuracy baseline |
| kv_byte_floor | KIVI/KVQuant-style 2-bit source KV byte floor | 43008 | - |  |  | accounting_contrast | false | true | assumption contrast, not accuracy baseline |
| external_reference | C2C cache-to-cache communication |  | - |  |  | reference_only | false | true | reference_only |
| external_reference | KVComm / KVCOMM selective KV communication |  | - |  |  | reference_only | false | true | reference_only |
| external_reference | TurboQuant |  | - |  |  | reference_only | false | true | accounting_only |
| external_reference | QJL |  | - |  |  | reference_only | false | true | accounting_only |
| external_reference | LLMLingua / LLMLingua-2 |  | - |  |  | reference_only | false | false | reference_only |
| external_reference | Gist tokens |  | - |  |  | reference_only | false | false | reference_only |

Claim boundary: This table supports a source-private boundary-traffic and residual-code systems claim. It does not claim production GPU serving speedup, native KV-cache compression superiority, or protocol-free latent reasoning.

## Non-Claims

- No native GPU/vLLM TTFT, TPOT, or goodput claim is made.
- KV/cache rows are byte-floor or related-work comparators, not implemented kernel baselines.
- PQ/OPQ/Hadamard rows are source-private residual-code methods on a controlled task, not broad latent reasoning.
